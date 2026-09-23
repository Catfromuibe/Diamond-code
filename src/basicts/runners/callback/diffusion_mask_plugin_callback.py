"""DiffusionMask Plugin — DropoutTS-style plugin on an unchanged backbone.

Paper Methodology (plugin, not a standalone model):
  1) Shared backbone encoder f_θ. Heads: forecast + diffusion noise prediction.
  2) Recoverability evidence e_rec from diffusion difficulty (EMA).
  3) Hierarchical controller π_φ: budget + patch-safety → Top-k mask.
  4) Same mask for forecasting view (learned mask token) and diffusion view.
  5) Counterfactual audit every K_a epochs → L_rank + safe budget ρ*.
Inference: plugin off; full inputs → backbone (m = 0).
"""

from __future__ import annotations

import inspect
import random
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from basicts.modules.diffusion_mask import (
    CounterfactualAuditor,
    DiffusionPatchReconHead,
    HierarchicalMaskController,
    fold_patches,
    unfold_patches,
)
from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class DiffusionMaskPluginCallback(BasicTSCallback):
    """Plugin: PatchTST backbone + hierarchical DiffusionMask (paper-aligned)."""

    def __init__(
        self,
        p_min: float = 0.05,
        p_max: float = 0.2,
        warmup_mask_ratio: float = 0.05,
        warmup_epochs: int = 15,
        recon_loss_weight: float = 0.15,
        controller_loss_weight: float = 0.05,
        lambda_rho: float = 1.0,
        loss_tol: float = 0.02,
        counterfactual_interval: int = 5,
        counterfactual_probes: int = 4,
        num_diffusion_steps: int = 4,
        patch_len: int = 16,
        patch_stride: Optional[int] = None,
        # Paper: same mask defines forecasting view and diffusion view
        mask_prediction: bool = True,
        candidate_ratios: Optional[Sequence[float]] = None,
        # legacy aliases
        mask_update_interval: Optional[int] = None,
        confidence_probe_prob: float = 0.0,
        **_ignored,
    ) -> None:
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max
        self.warmup_mask_ratio = warmup_mask_ratio
        self.warmup_epochs = warmup_epochs
        self.recon_loss_weight = recon_loss_weight
        self.controller_loss_weight = controller_loss_weight
        self.lambda_rho = lambda_rho
        self.loss_tol = loss_tol
        self.counterfactual_interval = (
            mask_update_interval if mask_update_interval is not None else counterfactual_interval
        )
        self.counterfactual_probes = counterfactual_probes
        self.num_diffusion_steps = num_diffusion_steps
        self.patch_len = patch_len
        self.patch_stride = patch_stride if patch_stride is not None else patch_len
        self.mask_prediction = mask_prediction
        self.confidence_probe_prob = confidence_probe_prob
        self.candidate_ratios = (
            list(candidate_ratios) if candidate_ratios is not None else None
        )

        self.mask_controller: Optional[HierarchicalMaskController] = None
        self.recon_head: Optional[DiffusionPatchReconHead] = None
        self.auditor: Optional[CounterfactualAuditor] = None
        self.patch_proj: Optional[nn.Linear] = None
        self.mask_token: Optional[nn.Parameter] = None

        self._model = None
        self._original_forward = None
        self._original_backbone_forward = None
        self._model_wrapped = False

        self._last_batch: Optional[dict] = None
        self._last_patches: Optional[torch.Tensor] = None
        self._last_binary_mask: Optional[torch.Tensor] = None
        self._last_budgets: Optional[torch.Tensor] = None
        self._last_direction_scores: Optional[torch.Tensor] = None
        self._last_hidden: Optional[torch.Tensor] = None
        self._last_recon_per_patch: Optional[torch.Tensor] = None
        self._probe_mask: Optional[torch.Tensor] = None

        self.input_len = 0
        self.num_features = 1
        self.num_patches = 0
        self.hidden_size = 256
        self.padding = None

    def _default_candidate_ratios(self) -> List[float]:
        """Candidate set R containing 0 and values in [p_min, p_max]."""
        grid = [0.0]
        for r in (self.p_min, 0.5 * (self.p_min + self.p_max), self.p_max):
            if r > 0 and r not in grid:
                grid.append(float(r))
        # denser mid grid
        for r in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45):
            if self.p_min <= r <= self.p_max and r not in grid:
                grid.append(float(r))
        return sorted(grid)

    def _unwrap_model(self, runner: "BasicTSRunner"):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        # PatchTST seasonal/trend split is the only decomp path this plugin cannot wrap.
        if getattr(model, "decomp", False) and hasattr(model, "seasonal_backbone"):
            raise NotImplementedError("decomp=True is not supported by the plugin yet.")
        return model

    def _infer_patch_geometry(self, model, runner: "BasicTSRunner") -> None:
        self.input_len = runner.cfg.input_len
        cfg = runner.cfg.model_config
        self.num_features = int(getattr(cfg, "num_features", 1) or 1)
        for src in (model, getattr(model, "backbone", None)):
            if src is not None and hasattr(src, "num_features"):
                self.num_features = int(src.num_features)
                break
        hidden = getattr(cfg, "hidden_size", None)
        if hidden:
            self.hidden_size = int(hidden)

        backbone = getattr(model, "backbone", None)
        embed = getattr(backbone, "patch_embedding", None) if backbone is not None else None
        if embed is not None:
            self.patch_len = embed.patch_len
            self.patch_stride = getattr(embed, "stride", self.patch_len)
            padding_layer = getattr(embed, "padding_layer", None)
            if padding_layer is None:
                padding_layer = getattr(embed, "padding_patch_layer", None)
            self.padding = (0, self.patch_stride) if padding_layer is not None else None
            self.num_patches = getattr(backbone, "num_patches", None) or int(
                (self.input_len - self.patch_len) / self.patch_stride + 1
            )
            self.num_patches = int(self.num_patches)
            if hasattr(embed, "value_embedding") and hasattr(embed.value_embedding, "out_features"):
                self.hidden_size = embed.value_embedding.out_features
        else:
            if getattr(model, "patch_len", None):
                self.patch_len = int(model.patch_len)
                self.patch_stride = int(getattr(model, "patch_stride", self.patch_len) or self.patch_len)
            self.padding = None
            if self.input_len < self.patch_len:
                self.patch_len = max(1, self.input_len)
                self.patch_stride = self.patch_len
            self.num_patches = int(
                (self.input_len - self.patch_len) / self.patch_stride + 1
            )
            self.num_patches = max(1, self.num_patches)

    def _init_modules(self, model, device: torch.device) -> None:
        ratios = self.candidate_ratios or self._default_candidate_ratios()
        self.mask_controller = HierarchicalMaskController(
            self.hidden_size,
            self.num_patches,
            p_min=self.p_min,
            p_max=self.p_max,
            warmup_mask_ratio=self.warmup_mask_ratio,
            lambda_rho=self.lambda_rho,
        ).to(device)
        self.recon_head = DiffusionPatchReconHead(
            self.hidden_size,
            self.patch_len,
            num_features=self.num_features,
            num_diffusion_steps=self.num_diffusion_steps,
        ).to(device)
        self.auditor = CounterfactualAuditor(
            self.num_patches,
            loss_tol=self.loss_tol,
            candidate_ratios=ratios,
        ).to(device)
        self.patch_proj = nn.Linear(self.patch_len, self.hidden_size).to(device)
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, self.patch_len, device=device)
        )
        nn.init.normal_(self.mask_token, std=0.02)

    def _plugin_parameters(self):
        params = list(self.mask_controller.parameters()) + list(self.recon_head.parameters())
        params += list(self.patch_proj.parameters())
        params.append(self.mask_token)
        return params

    def _add_plugin_params_to_optimizer(self, runner: "BasicTSRunner") -> None:
        optimizer_param_ids = {
            id(p) for group in runner.optimizer.param_groups for p in group["params"]
        }
        params_to_add = [p for p in self._plugin_parameters() if id(p) not in optimizer_param_ids]
        if params_to_add:
            runner.optimizer.add_param_group({
                "params": params_to_add,
                "lr": runner.optimizer.param_groups[0]["lr"],
            })
            runner.logger.info(
                f"[DiffusionMaskPlugin] Added {len(params_to_add)} plugin params to optimizer."
            )

    def _apply_binary_mask(self, patches: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
        """Replace masked patches with learnable token. patches [B,C,P,L], mask [B,P]."""
        mask = binary_mask.view(patches.size(0), 1, patches.size(2), 1)
        token = self.mask_token.expand_as(patches)
        return patches * (1.0 - mask) + token * mask

    def _mask_sequence(self, inputs: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
        patches = unfold_patches(inputs, self.patch_len, self.patch_stride, self.padding)
        masked_patches = self._apply_binary_mask(patches, binary_mask)
        return fold_patches(
            masked_patches, self.input_len, self.patch_len, self.patch_stride, self.padding,
        )

    def _patch_repr(self, patches: torch.Tensor) -> torch.Tensor:
        """[B,C,P,L] -> [B,P,H] for controller."""
        x = patches.mean(dim=1)
        return self.patch_proj(x)

    def _wrap_backbone_capture(self, model) -> None:
        # Only PatchTST-style backbones expose patch tokens as [B, C, P, H].
        # Other models keep the plugin input-side; recon uses _patch_repr.
        if type(model).__name__ != "PatchTSTForForecasting":
            self._original_backbone_forward = None
            return
        backbone = getattr(model, "backbone", None)
        if backbone is None or getattr(backbone, "patch_embedding", None) is None:
            self._original_backbone_forward = None
            return
        self._original_backbone_forward = backbone.forward

        def wrapped_backbone_forward(*args, **kwargs):
            result = self._original_backbone_forward(*args, **kwargs)
            hidden_states = result[0] if isinstance(result, tuple) else result
            if model.training and torch.is_tensor(hidden_states) and hidden_states.dim() == 4:
                # Channel-averaged patch states ≈ visible-context c_V
                self._last_hidden = hidden_states.mean(dim=1)
            return result

        backbone.forward = wrapped_backbone_forward

    def _wrap_model(self, runner: "BasicTSRunner") -> None:
        if self._model_wrapped:
            return
        model = self._unwrap_model(runner)
        self._model = model
        self._wrap_backbone_capture(model)
        self._original_forward = model.forward

        def wrapped_forward(inputs: torch.Tensor, *args, **kwargs):
            # Inference / eval: full-context, no masking, no diffusion overhead
            if not model.training:
                self._probe_mask = None
                return self._original_forward(inputs, *args, **kwargs)

            epoch = getattr(runner, "epoch", 0)
            warm_up = epoch < self.warmup_epochs

            patches = unfold_patches(
                inputs, self.patch_len, self.patch_stride, self.padding,
            )
            if self._probe_mask is not None:
                binary_mask = self._probe_mask
                if binary_mask.size(0) != inputs.size(0):
                    binary_mask = binary_mask[:1].expand(inputs.size(0), -1)
                budgets = binary_mask.mean(dim=-1)
                dir_scores = torch.zeros_like(binary_mask)
            else:
                patch_repr = self._patch_repr(patches).detach()
                binary_mask, budgets, dir_scores = self.mask_controller(
                    patch_repr, warm_up=warm_up,
                )

            masked_inputs = self._mask_sequence(inputs, binary_mask)

            # Paper: same mask defines forecasting view (unless explicitly disabled)
            pred_inputs = masked_inputs if self.mask_prediction else inputs
            result = self._original_forward(pred_inputs, *args, **kwargs)
            prediction = result["prediction"] if isinstance(result, dict) else result

            recon_loss = torch.zeros((), device=inputs.device)
            recon_per_patch = torch.zeros(self.num_patches, device=inputs.device)
            aux_loss = torch.zeros((), device=inputs.device)
            if self.recon_loss_weight > 0:
                if not self.mask_prediction:
                    # Need a masked encode for c_V when forecast path is clean
                    _ = self._original_forward(masked_inputs, *args, **kwargs)
                context = self._last_hidden if self._last_hidden is not None else self._patch_repr(patches)
                if context.dim() != 3 or context.size(1) != self.num_patches:
                    context = self._patch_repr(patches)
                recon_loss, recon_per_patch = self.recon_head(
                    context, patches, binary_mask=binary_mask,
                )
                self.mask_controller.update_recon_stats(recon_per_patch.detach())
                aux_loss = recon_loss * self.recon_loss_weight

            controller_loss = torch.zeros((), device=inputs.device)
            if (not warm_up) and dir_scores.requires_grad:
                pairs = self.auditor.ranking_pairs()
                controller_loss = self.mask_controller.controller_loss(
                    budgets,
                    dir_scores,
                    scalar_target=self.auditor.scalar_target(self.p_min, self.p_max),
                    direction_target=self.auditor.direction_target().to(inputs.device),
                    ranking_pairs=pairs,
                )
                aux_loss = aux_loss + self.controller_loss_weight * controller_loss

            self._last_patches = patches.detach()
            self._last_binary_mask = binary_mask.detach()
            self._last_budgets = budgets.detach()
            self._last_direction_scores = dir_scores.detach()
            self._last_recon_per_patch = recon_per_patch.detach()

            return {
                "prediction": prediction,
                "aux_loss": aux_loss,
                "recon_loss": recon_loss.detach(),
                "controller_loss": controller_loss.detach(),
                "mask_budget": budgets.detach().mean(),
                "mask_ratio": binary_mask.detach().mean(),
            }

        model.forward = wrapped_forward
        self._model_wrapped = True
        runner.logger.info(
            f"[DiffusionMaskPlugin] Wrapped {type(model).__name__} (paper): "
            f"mask_prediction={self.mask_prediction}, "
            "diffusion noise-pred + hierarchical controller + CF ρ*/rank."
        )

    def on_train_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        model = self._unwrap_model(runner)
        device = next(model.parameters()).device
        self._infer_patch_geometry(model, runner)
        self._init_modules(model, device)
        self._wrap_model(runner)
        self._add_plugin_params_to_optimizer(runner)
        runner.logger.info(
            f"[DiffusionMaskPlugin] warmup_epochs={self.warmup_epochs}, "
            f"p_min={self.p_min}, p_max={self.p_max}, "
            f"recon_w={self.recon_loss_weight}, λ_ρ={self.lambda_rho}, δ={self.loss_tol}, "
            f"mask_prediction={self.mask_prediction}, "
            f"counterfactual_interval={self.counterfactual_interval}, "
            f"patches={self.num_patches}, patch_len={self.patch_len}."
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

    def _forward_extra_kwargs(self, inputs: torch.Tensor) -> dict:
        extra = {}
        if self._last_batch is None or self._original_forward is None:
            return extra
        try:
            params = inspect.signature(self._original_forward).parameters
        except (TypeError, ValueError):
            params = {}
        for k in ("targets", "inputs_timestamps", "targets_timestamps"):
            if k not in params or k not in self._last_batch:
                continue
            v = self._last_batch[k]
            if torch.is_tensor(v) and v.size(0) != inputs.size(0):
                v = v[: inputs.size(0)]
            extra[k] = v
        return extra

    @torch.no_grad()
    def _predict_with_mask(self, model, inputs: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
        was_training = model.training
        model.eval()
        masked_inputs = self._mask_sequence(inputs, binary_mask)
        extra = self._forward_extra_kwargs(inputs)
        out = self._original_forward(masked_inputs, **extra)
        if was_training:
            model.train()
        if isinstance(out, dict):
            out = out["prediction"]
        return out

    @torch.no_grad()
    def _mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return F.mse_loss(pred, target).item()

    @torch.no_grad()
    def _search_safe_budget(
        self,
        model,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        safety_scores: torch.Tensor,
    ) -> float:
        """ρ* = max{ρ ∈ R | ℓ(m_ρ) - ℓ(0) ≤ δ}."""
        full_mask = torch.zeros(
            inputs.size(0), self.num_patches, device=inputs.device, dtype=safety_scores.dtype,
        )
        full_pred = self._predict_with_mask(model, inputs, full_mask)
        full_loss = self._mse_loss(full_pred, targets)

        scores = safety_scores
        if scores.dim() == 1:
            scores = scores.unsqueeze(0).expand(inputs.size(0), -1)
        elif scores.size(0) != inputs.size(0):
            scores = scores[:1].expand(inputs.size(0), -1)

        ratios = self.auditor.candidate_ratios or self._default_candidate_ratios()
        best = 0.0
        for rho in sorted(ratios):
            m = self.mask_controller.mask_from_scores(scores, rho)
            pred = self._predict_with_mask(model, inputs, m)
            loss = self._mse_loss(pred, targets)
            if loss - full_loss <= self.loss_tol:
                best = float(rho)
            else:
                break  # denser sorted grid: stop once unsafe
        return best

    @torch.no_grad()
    def _counterfactual_audit(self, runner: "BasicTSRunner", model) -> None:
        """Freeze θ/θ_d (no_grad): patch Δ_i probes + ρ* target construction."""
        data = self._last_batch
        if data is None or self._last_binary_mask is None:
            return
        inputs = data["inputs"]
        targets = data.get("targets")
        if targets is None:
            return

        batch = min(inputs.size(0), self._last_binary_mask.size(0))
        mask = self._last_binary_mask[:batch]
        inputs_b, targets_b = inputs[:batch], targets[:batch]

        base_pred = self._predict_with_mask(model, inputs_b, mask)
        base_loss = self._mse_loss(base_pred, targets_b)
        self.auditor.update_baseline(base_loss)

        masked_idx = (mask.sum(dim=0) > 0).nonzero(as_tuple=False).view(-1).tolist()
        if not masked_idx:
            masked_idx = list(range(self.num_patches))
        probes = random.sample(masked_idx, k=min(self.counterfactual_probes, len(masked_idx)))

        for patch_idx in probes:
            restored = mask.clone()
            restored[:, patch_idx] = 0.0
            restored_pred = self._predict_with_mask(model, inputs_b, restored)
            restored_loss = self._mse_loss(restored_pred, targets_b)
            self.auditor.observe(patch_idx, base_loss, restored_loss)
            delta = base_loss - restored_loss
            tag = "protect" if delta > 0 else "safe-mask"
            runner.logger.info(
                f"[DiffusionMaskPlugin][CF] patch={patch_idx} "
                f"masked={base_loss:.4f} restored={restored_loss:.4f} "
                f"delta={delta:.4f} -> {tag}"
            )

        self.mask_controller.update_audit_stats(self.auditor.delta_ema)

        scores = self._last_direction_scores
        if scores is None:
            scores = torch.zeros(batch, self.num_patches, device=inputs.device)
        else:
            scores = scores[:batch]
        rho_star = self._search_safe_budget(model, inputs_b, targets_b, scores)
        rho_ema = self.auditor.update_safe_budget(rho_star)
        runner.logger.info(
            f"[DiffusionMaskPlugin][CF] ρ*={rho_star:.4f} safe_budget_ema={rho_ema:.4f} "
            f"δ={self.loss_tol}."
        )

    def on_epoch_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        if self.mask_controller is None:
            return
        epoch = runner.epoch + 1
        warm_up = epoch <= self.warmup_epochs
        budget = (
            self._last_budgets.mean().item()
            if self._last_budgets is not None else self.warmup_mask_ratio
        )
        ratio = (
            self._last_binary_mask.mean().item()
            if self._last_binary_mask is not None else 0.0
        )
        runner.logger.info(
            f"[DiffusionMaskPlugin] Epoch {epoch}: "
            f"{'warm-up' if warm_up else 'controller'} | "
            f"budget={budget:.4f}, realized_mask={ratio:.4f}."
        )
        if warm_up:
            return
        if epoch % self.counterfactual_interval != 0:
            return
        model = self._unwrap_model(runner)
        self._counterfactual_audit(runner, model)

    def on_train_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        if not self._model_wrapped or self._model is None:
            return
        if self._original_backbone_forward is not None:
            self._model.backbone.forward = self._original_backbone_forward
        if self._original_forward is not None:
            self._model.forward = self._original_forward
        self._model_wrapped = False
        runner.logger.info(
            f"[DiffusionMaskPlugin] Restored original {type(self._model).__name__} forward."
        )
