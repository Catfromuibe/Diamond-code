from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from basicts.modules.diffusion_mask import (
    CounterfactualAuditor,
    DiffusionPatchReconHead,
    HierarchicalMaskController,
    unfold_patches,
)
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import Encoder, EncoderLayer, MultiHeadAttention
from basicts.models.PatchTST.arch.patchtst_layers import PatchTSTBatchNorm, PatchTSTHead

from ..config.diffusion_mask_patchtst_config import DiffusionMaskPatchTSTConfig


class DiffusionMaskPatchTST(nn.Module):
    """
    Shared-encoder dual-head model with hierarchical patch masking:
      - Warm-up: fixed low-ratio random mask
      - Controller: Scalar Head (budget) + Direction Head (which) -> Top-k mask
      - Prediction view: masked patches -> mask token -> forecast head
      - Diffusion view: reconstruct masked patches conditioned on visible context
      - Inference: no masking / no diffusion; full history -> prediction
    """

    def __init__(self, config: DiffusionMaskPatchTSTConfig):
        super().__init__()
        self.config = config
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.patch_len = config.patch_len
        self.stride = config.patch_stride
        self.padding = (0, config.patch_stride) if config.padding else None
        self.warmup_epochs = config.warmup_epochs

        self.num_patches = int((config.input_len - config.patch_len) / config.patch_stride + 1)
        if config.padding:
            self.num_patches += 1

        # Shallow patch embedding: Linear(patch_len -> H) + position
        self.value_embedding = nn.Linear(config.patch_len, config.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, config.hidden_size) * 0.02,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)
        self.embed_dropout = nn.Dropout(config.fc_dropout)

        norm_type = nn.LayerNorm if config.norm_type == "layer_norm" else PatchTSTBatchNorm
        self.encoder = Encoder(
            nn.ModuleList([
                EncoderLayer(
                    MultiHeadAttention(config.hidden_size, config.n_heads, config.attn_dropout),
                    MLPLayer(
                        config.hidden_size, config.intermediate_size,
                        hidden_act=config.hidden_act, dropout=config.fc_dropout,
                    ),
                    layer_norm=(norm_type, config.hidden_size),
                    norm_position="post",
                )
                for _ in range(config.num_layers)
            ])
        )
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecasting_head = PatchTSTHead(
            self.num_patches * config.hidden_size,
            config.output_len,
            config.individual_head,
            config.num_features,
            config.head_dropout,
        )
        self.recon_head = DiffusionPatchReconHead(
            config.hidden_size,
            config.patch_len,
            num_features=config.num_features,
            num_diffusion_steps=config.num_diffusion_steps,
        )
        self.mask_controller = HierarchicalMaskController(
            config.hidden_size,
            self.num_patches,
            p_min=config.p_min,
            p_max=config.p_max,
            warmup_mask_ratio=config.warmup_mask_ratio,
        )
        self.auditor = CounterfactualAuditor(self.num_patches)
        # Legacy alias for callback compatibility
        self.confidence_tracker = self.auditor

        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                config.num_features, affine=config.affine, subtract_last=config.subtract_last,
            )

        self._last_recon_per_patch: Optional[torch.Tensor] = None
        self._last_binary_mask: Optional[torch.Tensor] = None
        self._last_budgets: Optional[torch.Tensor] = None
        self._last_direction_scores: Optional[torch.Tensor] = None
        self._probe_mask: Optional[torch.Tensor] = None

    def _patchify(self, inputs: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, C, P, L]."""
        return unfold_patches(inputs, self.patch_len, self.stride, self.padding)

    def _embed_patches(
        self,
        patches: torch.Tensor,
        binary_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Channel-independent patch embedding with optional mask tokens.
        patches: [B, C, P, L]
        binary_mask: [B, P] 1=masked
        Returns: [B*C, P, H]
        """
        batch, num_features, num_patches, patch_len = patches.shape
        x = patches.reshape(batch * num_features, num_patches, patch_len)
        tokens = self.value_embedding(x) + self.position_embedding
        if binary_mask is not None:
            mask = binary_mask.unsqueeze(1).expand(-1, num_features, -1).reshape(
                batch * num_features, num_patches, 1,
            )
            tokens = tokens * (1.0 - mask) + self.mask_token * mask
        return self.embed_dropout(tokens)

    def _encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B*C, P, H] -> hidden [B, C, P, H]."""
        hidden, _ = self.encoder(tokens)
        return hidden.reshape(-1, self.num_features, hidden.shape[-2], hidden.shape[-1])

    def _predict_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(hidden_states)
        return self.forecasting_head(flat).transpose(1, 2)

    def _build_mask(
        self,
        patches: torch.Tensor,
        epoch: int,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return binary_mask, budgets, direction_scores, is_warmup."""
        if self._probe_mask is not None:
            mask = self._probe_mask
            budget = mask.mean(dim=-1)
            return mask, budget, torch.zeros_like(mask), False

        if not train:
            batch = patches.size(0)
            zeros = torch.zeros(batch, self.num_patches, device=patches.device)
            return zeros, zeros.mean(dim=-1), zeros, False

        warm_up = epoch < self.warmup_epochs
        # Controller features: detach so controller does not fight encoder via this path
        tokens = self._embed_patches(patches, binary_mask=None)
        patch_repr = tokens.reshape(
            patches.size(0), self.num_features, self.num_patches, -1,
        ).mean(dim=1).detach()

        mask, budgets, scores = self.mask_controller(patch_repr, warm_up=warm_up)
        return mask, budgets, scores, warm_up

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        epoch: int = 0,
        train: bool = True,
        probe_patch_idx: Optional[int] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        patches = self._patchify(inputs)
        binary_mask, budgets, dir_scores, warm_up = self._build_mask(patches, epoch, train)

        # Prediction view: replace masked patches with mask token
        tokens = self._embed_patches(patches, binary_mask if train else None)
        hidden_states = self._encode_tokens(tokens)
        prediction = self._predict_from_hidden(hidden_states)
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")

        if not train:
            return prediction

        # Diffusion reconstruction view (shared encoder states)
        # Average channel dimension for recon head conditioning
        enc_for_recon = hidden_states.mean(dim=1)  # [B, P, H]
        recon_loss, recon_per_patch = self.recon_head(enc_for_recon, patches, binary_mask)
        self._last_recon_per_patch = recon_per_patch.detach()
        self._last_binary_mask = binary_mask.detach()
        self._last_budgets = budgets.detach()
        self._last_direction_scores = dir_scores.detach()

        # Update recon EMA (clipped) for controller inputs
        self.mask_controller.update_recon_stats(recon_per_patch.detach())

        aux_loss = recon_loss * self.config.recon_loss_weight
        controller_loss = torch.zeros((), device=inputs.device)
        if not warm_up and dir_scores.requires_grad:
            controller_loss = self.mask_controller.controller_loss(
                budgets,
                dir_scores,
                scalar_target=self.auditor.scalar_target(self.config.p_min, self.config.p_max),
                direction_target=self.auditor.direction_target().to(inputs.device),
            )
            aux_loss = aux_loss + self.config.controller_loss_weight * controller_loss

        return {
            "prediction": prediction,
            "aux_loss": aux_loss,
            "recon_loss": recon_loss.detach(),
            "controller_loss": controller_loss.detach() if torch.is_tensor(controller_loss) else controller_loss,
            "mask_budget": budgets.detach().mean(),
            "mask_ratio": binary_mask.detach().mean(),
            "warm_up": torch.tensor(float(warm_up), device=inputs.device),
        }

    @torch.no_grad()
    def update_mask_scalars(self) -> None:
        """Legacy hook: sync auditor stats into controller."""
        self.mask_controller.update_audit_stats(self.auditor.protect_score)

    @torch.no_grad()
    def block_harmful_patch(self, patch_idx: int) -> None:
        self.auditor.protect_score[patch_idx] = self.auditor.protect_score[patch_idx] + 1.0
        self.mask_controller.update_audit_stats(self.auditor.protect_score)

    def set_probe_mask(self, mask: Optional[torch.Tensor]) -> None:
        self._probe_mask = mask

    @torch.no_grad()
    def predict_with_mask(
        self,
        inputs: torch.Tensor,
        binary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward prediction under a fixed binary mask (for counterfactual)."""
        was_training = self.training
        self.eval()
        if self.use_revin:
            norm_inputs = self.revin(inputs, "norm")
        else:
            norm_inputs = inputs
        patches = self._patchify(norm_inputs)
        tokens = self._embed_patches(patches, binary_mask)
        hidden = self._encode_tokens(tokens)
        prediction = self._predict_from_hidden(hidden)
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        if was_training:
            self.train()
        return prediction

    def probe_prediction_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, patch_idx: int,
    ) -> float:
        """Restore one masked patch and measure prediction MAE."""
        with torch.no_grad():
            if self._last_binary_mask is None:
                return F.l1_loss(
                    self.predict_with_mask(
                        inputs,
                        torch.zeros(inputs.size(0), self.num_patches, device=inputs.device),
                    ),
                    targets,
                ).item()
            mask = self._last_binary_mask[: inputs.size(0)].clone()
            if mask.size(0) != inputs.size(0):
                mask = mask[:1].expand(inputs.size(0), -1).clone()
            # Base: current mask
            base_pred = self.predict_with_mask(inputs, mask)
            base_loss = F.l1_loss(base_pred, targets).item()
            # Restored: unmask target patch
            restored = mask.clone()
            restored[:, patch_idx] = 0.0
            restored_pred = self.predict_with_mask(inputs, restored)
            restored_loss = F.l1_loss(restored_pred, targets).item()
            self.auditor.observe(patch_idx, base_loss, restored_loss)
            return restored_loss
