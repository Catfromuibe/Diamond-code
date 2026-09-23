"""Callback for hierarchical diffusion-guided patch masking."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn import functional as F

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class DiffusionMaskCallback(BasicTSCallback):
    """
    Training helper for DiffusionMaskPatchTST:
      - Combines prediction + diffusion recon (+ controller) losses
      - Every K epochs: freeze encoder/task heads, run paired counterfactual recovery
      - Sync auditor stats into HierarchicalMaskController
    """

    def __init__(
        self,
        counterfactual_interval: int = 2,
        counterfactual_probes: int = 4,
        recon_loss_weight: float = 0.5,
        mask_update_interval: Optional[int] = None,
        confidence_probe_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.counterfactual_interval = (
            mask_update_interval if mask_update_interval is not None else counterfactual_interval
        )
        self.counterfactual_probes = counterfactual_probes
        self.recon_loss_weight = recon_loss_weight
        self.confidence_probe_prob = confidence_probe_prob
        self._last_batch: Optional[dict] = None

    def _unwrap_model(self, runner: "BasicTSRunner"):
        from basicts.models.DiffusionMaskPatchTST import DiffusionMaskPatchTST
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        if not isinstance(model, DiffusionMaskPatchTST):
            raise TypeError("DiffusionMaskCallback requires DiffusionMaskPatchTST model.")
        return model

    def on_train_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        model = self._unwrap_model(runner)
        device = next(model.parameters()).device
        model.auditor.to(device)
        runner.logger.info(
            "DiffusionMask hierarchical controller enabled: "
            f"warmup_epochs={model.warmup_epochs}, "
            f"counterfactual_interval={self.counterfactual_interval}, "
            f"probes={self.counterfactual_probes}."
        )

    def on_compute_loss(self, runner: "BasicTSRunner", **kwargs) -> None:
        forward_return = kwargs["forward_return"]
        self._last_batch = kwargs.get("data")
        pred_loss = runner._metric_forward(runner.loss, forward_return)
        total_loss = pred_loss
        if "aux_loss" in forward_return:
            total_loss = total_loss + forward_return["aux_loss"]
        forward_return["loss"] = total_loss
        forward_return["__mtl_loss__"] = True

    @torch.no_grad()
    def _counterfactual_audit(self, runner: "BasicTSRunner", model) -> None:
        data = self._last_batch
        if data is None:
            return
        inputs = data["inputs"]
        targets = data.get("targets")
        if targets is None:
            return
        if model._last_binary_mask is None:
            return

        # Freeze shared encoder + two task heads during paired recovery
        frozen = [
            model.encoder, model.forecasting_head, model.recon_head,
            model.value_embedding,
        ]
        was_training = {m: m.training for m in frozen}
        for m in frozen:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        batch = min(inputs.size(0), model._last_binary_mask.size(0))
        mask = model._last_binary_mask[:batch]
        inputs_b = inputs[:batch]
        targets_b = targets[:batch]

        base_pred = model.predict_with_mask(inputs_b, mask)
        base_loss = F.l1_loss(base_pred, targets_b).item()
        model.auditor.update_baseline(base_loss)

        # Probe a few currently-masked patches
        masked_idx = (mask.sum(dim=0) > 0).nonzero(as_tuple=False).view(-1).tolist()
        if not masked_idx:
            masked_idx = list(range(model.num_patches))
        probes = random.sample(masked_idx, k=min(self.counterfactual_probes, len(masked_idx)))

        for patch_idx in probes:
            restored = mask.clone()
            restored[:, patch_idx] = 0.0
            restored_pred = model.predict_with_mask(inputs_b, restored)
            restored_loss = F.l1_loss(restored_pred, targets_b).item()
            model.auditor.observe(patch_idx, base_loss, restored_loss)
            delta = base_loss - restored_loss
            tag = "protect" if delta > 0 else "safe-mask"
            runner.logger.info(
                f"[DiffusionMask][CF] patch={patch_idx} "
                f"masked={base_loss:.4f} restored={restored_loss:.4f} "
                f"delta={delta:.4f} -> {tag}"
            )

        model.mask_controller.update_audit_stats(model.auditor.protect_score)

        for m in frozen:
            for p in m.parameters():
                p.requires_grad_(True)
            if was_training[m]:
                m.train()

    def on_step_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        # Optional light online probe (usually disabled; CF audit is the main signal)
        if self.confidence_probe_prob <= 0 or random.random() > self.confidence_probe_prob:
            return
        data = self._last_batch
        if data is None:
            return
        model = self._unwrap_model(runner)
        targets = data.get("targets")
        if targets is None or model._last_binary_mask is None:
            return
        masked_idx = (model._last_binary_mask[0] > 0).nonzero(as_tuple=False).view(-1)
        if len(masked_idx) == 0:
            return
        patch_idx = int(masked_idx[random.randrange(len(masked_idx))].item())
        model.probe_prediction_loss(data["inputs"], targets, patch_idx)

    def on_epoch_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        model = self._unwrap_model(runner)
        epoch = runner.epoch + 1
        warm_up = epoch <= model.warmup_epochs
        budget = (
            model._last_budgets.mean().item()
            if model._last_budgets is not None else model.config.warmup_mask_ratio
        )
        ratio = (
            model._last_binary_mask.mean().item()
            if model._last_binary_mask is not None else 0.0
        )
        runner.logger.info(
            f"[DiffusionMask] Epoch {epoch}: "
            f"{'warm-up' if warm_up else 'controller'} | "
            f"budget={budget:.4f}, realized_mask={ratio:.4f}, "
            f"protect_mean={model.auditor.protect_score.mean().item():.4f}."
        )

        if warm_up:
            return
        if epoch % self.counterfactual_interval != 0:
            return

        self._counterfactual_audit(runner, model)
        model.update_mask_scalars()
        safe_budget = model.auditor.scalar_target(model.config.p_min, model.config.p_max)
        runner.logger.info(
            f"[DiffusionMask] Epoch {epoch}: counterfactual audit done. "
            f"safe_budget_ema={safe_budget:.4f}."
        )
