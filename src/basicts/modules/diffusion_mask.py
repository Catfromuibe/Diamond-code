"""Diffusion-guided hierarchical patch masking (paper Methodology).

Training (plugin on an unchanged backbone):
  1) Warm-up: fixed low-ratio random mask; L_joint = L_pred + λ_diff L_diff
  2) Diffusion recoverability evidence e_rec (EMA of denoising difficulty)
  3) Hierarchical controller π_φ: budget head + patch-safety head → Top-k mask
  4) Same mask defines forecasting view (mask token) and diffusion view
  5) Counterfactual audit every K_a epochs → ranking + safe budget ρ*
Inference: controller & diffusion off; full history → backbone (m = 0)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from basicts.modules.mlps import MLPLayer


def unfold_patches(
    inputs: torch.Tensor,
    patch_len: int,
    stride: int,
    padding: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Extract patches: [B, T, C] -> [B, C, P, patch_len]."""
    x = inputs.transpose(1, 2)
    if padding is not None:
        x = F.pad(x, padding, mode="replicate")
    return x.unfold(dimension=-1, size=patch_len, step=stride)


def fold_patches(
    patches: torch.Tensor,
    seq_len: int,
    patch_len: int,
    stride: int,
    padding: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Overlap-add fold patches back to [B, T, C]."""
    batch_size, num_features, num_patches, _ = patches.shape
    padded_len = seq_len + (padding[0] + padding[1] if padding else 0)
    out = torch.zeros(
        batch_size, num_features, padded_len,
        device=patches.device, dtype=patches.dtype,
    )
    counts = torch.zeros_like(out)
    for p in range(num_patches):
        start = p * stride
        end = start + patch_len
        out[:, :, start:end] += patches[:, :, p, :]
        counts[:, :, start:end] += 1.0
    out = out / counts.clamp_min(1.0)
    if padding is not None:
        out = out[:, :, padding[0]: padding[0] + seq_len]
    return out.transpose(1, 2)


class CounterfactualAuditor:
    """Task-aligned supervision from counterfactual masking effects Δ_i."""

    def __init__(
        self,
        num_patches: int,
        ema_decay: float = 0.9,
        loss_tol: float = 0.02,
        candidate_ratios: Optional[Sequence[float]] = None,
    ) -> None:
        self.num_patches = num_patches
        self.ema_decay = ema_decay
        self.loss_tol = loss_tol  # δ in Eq. (safe budget)
        # Higher Δ̄ => more harmful to mask (protect)
        self.delta_ema = torch.zeros(num_patches)
        self.safe_budget_ema = 0.0
        self.baseline_loss_ema = 0.0
        self.candidate_ratios = list(candidate_ratios) if candidate_ratios is not None else []

    def to(self, device: torch.device) -> "CounterfactualAuditor":
        self.delta_ema = self.delta_ema.to(device)
        return self

    def update_baseline(self, pred_loss: float) -> None:
        if self.baseline_loss_ema == 0.0:
            self.baseline_loss_ema = pred_loss
        else:
            self.baseline_loss_ema = (
                self.ema_decay * self.baseline_loss_ema + (1.0 - self.ema_decay) * pred_loss
            )

    def observe(self, patch_idx: int, masked_loss: float, restored_loss: float) -> None:
        """Δ_i = ℓ(m) - ℓ(m^(-i)); positive => masking patch i is harmful."""
        delta = masked_loss - restored_loss
        prev = self.delta_ema[patch_idx]
        self.delta_ema[patch_idx] = self.ema_decay * prev + (1.0 - self.ema_decay) * delta

    def ranking_pairs(self) -> List[Tuple[int, int]]:
        """Q = {(i,j) | Δ̄_i < Δ̄_j}: i should receive a higher maskability score."""
        pairs: List[Tuple[int, int]] = []
        d = self.delta_ema
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                if i == j:
                    continue
                if d[i] < d[j]:
                    pairs.append((i, j))
        return pairs

    def update_safe_budget(self, rho_star: float) -> float:
        self.safe_budget_ema = (
            self.ema_decay * self.safe_budget_ema + (1.0 - self.ema_decay) * rho_star
        )
        return float(self.safe_budget_ema)

    # ---- backward-compatible aliases used by older call sites ----
    @property
    def protect_score(self) -> torch.Tensor:
        return self.delta_ema

    def direction_target(self) -> torch.Tensor:
        """Higher = safer to mask (inverse of harmfulness)."""
        score = -self.delta_ema
        score = score - score.mean()
        return torch.sigmoid(score)

    def scalar_target(self, p_min: float, p_max: float) -> float:
        budget = float(self.safe_budget_ema)
        if budget <= 0.0:
            protect = torch.sigmoid(self.delta_ema).mean().item()
            budget = p_min + (p_max - p_min) * max(0.0, 1.0 - protect)
        return max(p_min, min(p_max, budget))


class HierarchicalMaskController(nn.Module):
    """Budget head + patch-safety head → Top-k binary mask (Eq. budget/safety/topk)."""

    def __init__(
        self,
        hidden_size: int,
        num_patches: int,
        p_min: float = 0.05,
        p_max: float = 0.5,
        warmup_mask_ratio: float = 0.1,
        evidence_momentum: float = 0.9,
        lambda_rho: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.p_min = p_min
        self.p_max = p_max
        self.warmup_mask_ratio = warmup_mask_ratio
        self.evidence_momentum = evidence_momentum
        self.lambda_rho = lambda_rho

        # g_φb([z_g || e_rec]), e_rec ∈ R^P
        self.budget_head = nn.Sequential(
            nn.Linear(hidden_size + num_patches, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        # g_φs([z_i || e_rec,i])
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.pos_embed = nn.Embedding(num_patches, hidden_size)
        # Recoverability evidence: lower => more recoverable
        self.register_buffer("e_rec", torch.zeros(num_patches), persistent=True)
        # legacy buffers (kept for checkpoint / older code paths)
        self.register_buffer("recon_ema", torch.zeros(num_patches), persistent=True)
        self.register_buffer("audit_ema", torch.zeros(num_patches), persistent=True)
        self.register_buffer("global_recon_ema", torch.zeros(1), persistent=True)
        self.register_buffer("global_audit_ema", torch.zeros(1), persistent=True)

    @torch.no_grad()
    def update_recon_stats(self, difficulty_per_patch: torch.Tensor, ema: Optional[float] = None) -> None:
        """Eq. recoverability evidence: e ← γ e + (1-γ) sg(d)."""
        gamma = self.evidence_momentum if ema is None else ema
        d = difficulty_per_patch.detach()
        self.e_rec.mul_(gamma).add_(d, alpha=1.0 - gamma)
        self.recon_ema.copy_(self.e_rec)
        self.global_recon_ema.fill_(self.e_rec.mean())

    @torch.no_grad()
    def update_audit_stats(self, delta_ema: torch.Tensor, ema: float = 0.9) -> None:
        self.audit_ema.mul_(ema).add_(delta_ema.detach(), alpha=1.0 - ema)
        self.global_audit_ema.fill_(self.audit_ema.mean())

    def _evidence_feats(self) -> torch.Tensor:
        """Stabilize e_rec for controller inputs without changing ranking."""
        e = self.e_rec
        return torch.log1p(e.clamp_min(0.0))

    def warm_up_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fixed low-ratio random binary mask. 1 = masked. [B, P]."""
        k = max(1, int(round(self.warmup_mask_ratio * self.num_patches)))
        scores = torch.rand(batch_size, self.num_patches, device=device)
        topk = torch.topk(scores, k=k, dim=-1).indices
        mask = torch.zeros(batch_size, self.num_patches, device=device)
        mask.scatter_(1, topk, 1.0)
        return mask

    def mask_from_scores(
        self,
        scores: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Build Top-k mask from safety scores at a fixed ratio ρ."""
        batch, num_patches = scores.shape
        k = int(torch.floor(torch.tensor(num_patches * float(ratio))).item())
        k = max(0, min(num_patches, k))
        mask = torch.zeros_like(scores)
        if k == 0:
            return mask
        idx = torch.topk(scores, k=k, dim=-1).indices
        mask.scatter_(1, idx, 1.0)
        return mask

    def forward(
        self,
        patch_repr: torch.Tensor,
        warm_up: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_repr: [B, P, H] position-aware pre-mask patch representations
            warm_up: if True, use fixed random mask
        Returns:
            binary_mask [B, P] (1=masked), budgets [B], safety_scores [B, P] (pre-sigmoid logits)
        """
        batch_size, num_patches, _ = patch_repr.shape
        if warm_up:
            mask = self.warm_up_mask(batch_size, patch_repr.device)
            budget = torch.full(
                (batch_size,), self.warmup_mask_ratio,
                device=patch_repr.device, dtype=patch_repr.dtype,
            )
            return mask, budget, torch.zeros_like(mask)

        pos = self.pos_embed(
            torch.arange(num_patches, device=patch_repr.device)
        ).unsqueeze(0).expand(batch_size, -1, -1)
        z = patch_repr + pos  # position-aware z_i
        z_g = z.mean(dim=1)  # pooled sample representation

        e = self._evidence_feats()
        e_b = e.view(1, -1).expand(batch_size, -1)
        budgets = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(
            self.budget_head(torch.cat([z_g, e_b], dim=-1)).squeeze(-1)
        )

        e_i = e.view(1, num_patches, 1).expand(batch_size, -1, -1)
        safety_logits = self.safety_head(torch.cat([z, e_i], dim=-1)).squeeze(-1)

        mask = torch.zeros_like(safety_logits)
        for i in range(batch_size):
            k = int(torch.floor(budgets[i] * num_patches).item())
            k = max(0, min(num_patches - 1, k))  # leave ≥1 visible when possible
            if k <= 0:
                continue
            idx = torch.topk(safety_logits[i], k=k).indices
            mask[i, idx] = 1.0
        return mask, budgets, safety_logits

    def ranking_loss(
        self,
        safety_logits: torch.Tensor,
        pairs: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        """L_rank over audited pairs (i,j) with Δ̄_i < Δ̄_j."""
        if not pairs:
            return safety_logits.new_zeros(())
        s = torch.sigmoid(safety_logits).mean(dim=0)  # [P]
        losses = []
        for i, j in pairs:
            losses.append(F.softplus(-(s[i] - s[j])))
        return torch.stack(losses).mean()

    def controller_loss(
        self,
        budgets: torch.Tensor,
        direction_scores: torch.Tensor,
        scalar_target: float,
        direction_target: Optional[torch.Tensor] = None,
        ranking_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """L_ctrl = L_rank + λ_ρ (ρ̂ - sg(ρ*))²."""
        rho_star = budgets.new_tensor(float(scalar_target))
        budget_loss = F.mse_loss(budgets, rho_star.expand_as(budgets).detach())

        if ranking_pairs:
            rank_loss = self.ranking_loss(direction_scores, ranking_pairs)
        elif direction_target is not None:
            # Soft fallback when pairwise set is empty
            dir_tgt = direction_target.view(1, -1).expand_as(direction_scores)
            rank_loss = F.binary_cross_entropy_with_logits(direction_scores, dir_tgt)
        else:
            rank_loss = budgets.new_zeros(())
        return rank_loss + self.lambda_rho * budget_loss


class DiffusionPatchReconHead(nn.Module):
    """Noise-prediction diffusion head conditioned on visible-context encoder states."""

    def __init__(
        self,
        hidden_size: int,
        patch_len: int,
        num_features: int = 1,
        num_diffusion_steps: int = 4,
    ) -> None:
        super().__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.patch_len = patch_len
        self.num_features = num_features
        self.time_embed = nn.Embedding(num_diffusion_steps, hidden_size)
        self.noisy_proj = nn.Linear(patch_len * num_features, hidden_size)
        self.decoder = MLPLayer(hidden_size, hidden_size * 2, hidden_act="gelu", dropout=0.1)
        self.out_proj = nn.Linear(hidden_size, patch_len * num_features)
        # Linear cumulative ᾱ_t schedule in (0,1]
        steps = torch.arange(1, num_diffusion_steps + 1, dtype=torch.float32)
        alpha_bar = steps / float(num_diffusion_steps)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_patches: torch.Tensor,
        binary_mask: Optional[torch.Tensor] = None,
        mask_rates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict injected noise on masked patches (Eq. forward_diffusion / patch_difficulty).

        Args:
            hidden_states: [B, P, H] visible-context representation c_V
            target_patches: [B, C, P, L]
            binary_mask: [B, P] 1=masked
        Returns:
            L_diff scalar, per-patch difficulty d ∈ R^P
        """
        if binary_mask is None:
            if mask_rates is None:
                raise ValueError("Need binary_mask or mask_rates.")
            batch = target_patches.size(0)
            if hidden_states.dim() == 3 and hidden_states.size(0) == batch * target_patches.size(1):
                hidden_states = hidden_states.reshape(
                    batch, target_patches.size(1), hidden_states.size(1), -1,
                ).mean(dim=1)
            binary_mask = mask_rates.view(1, -1).expand(batch, -1)

        batch, num_patches, _ = hidden_states.shape
        targets = target_patches.permute(0, 2, 1, 3).reshape(
            batch, num_patches, -1,
        )  # [B, P, C*L]

        # Per-sample diffusion step t ∈ T
        step = torch.randint(
            0, self.num_diffusion_steps, (batch,), device=hidden_states.device,
        )
        alpha_bar = self.alpha_bar[step].view(batch, 1, 1)
        eps = torch.randn_like(targets)
        mask = binary_mask.unsqueeze(-1).to(targets.dtype)

        # Only corrupt masked patches; visible patches stay clean (no recon target)
        x_t = (
            torch.sqrt(alpha_bar) * targets
            + torch.sqrt(1.0 - alpha_bar) * eps
        )
        x_t = targets * (1.0 - mask) + x_t * mask

        t_emb = self.time_embed(step).unsqueeze(1)
        noisy_h = self.noisy_proj(x_t) * mask
        h = hidden_states + t_emb + noisy_h
        pred_eps = self.out_proj(self.decoder(h))

        feat_dim = targets.size(-1)
        per_elem = (pred_eps - eps) ** 2
        # d_i averaged over batch where patch i is masked
        masked_sum = (per_elem * mask).sum(dim=-1)  # [B, P]
        denom = mask.squeeze(-1).sum(dim=0).clamp_min(1.0)
        per_patch = masked_sum.sum(dim=0) / (denom * feat_dim)
        n_masked = mask.sum().clamp_min(1.0)
        loss = (per_elem * mask).sum() / (n_masked * feat_dim)
        return loss, per_patch


# Backward-compatible aliases
class PatchMaskController(HierarchicalMaskController):
    """Deprecated alias kept for imports."""

    def mask_rates(self) -> torch.Tensor:
        return torch.full(
            (self.num_patches,),
            (self.p_min + self.p_max) * 0.5,
            device=self.e_rec.device,
        )

    def apply_mask(self, patches: torch.Tensor, rates=None) -> torch.Tensor:
        batch = patches.size(0)
        mask = self.warm_up_mask(batch, patches.device)
        keep = (1.0 - mask).view(batch, 1, self.num_patches, 1)
        return patches * keep

    @torch.no_grad()
    def update_from_recon_loss(self, recon_loss_per_patch: torch.Tensor, step_size: float = 0.1) -> None:
        self.update_recon_stats(recon_loss_per_patch)

    @torch.no_grad()
    def block_patch(self, patch_idx: int) -> None:
        self.audit_ema[patch_idx] = self.audit_ema[patch_idx] + 1.0


class ConfidenceTracker(CounterfactualAuditor):
    """Deprecated alias for CounterfactualAuditor."""

    def __init__(self, num_patches: int, ema_decay: float = 0.9, trend_threshold: float = 0.0):
        super().__init__(num_patches, ema_decay)
        self.trend_threshold = trend_threshold
        self.blocked = torch.zeros(num_patches, dtype=torch.bool)
        self.delta_ema_legacy = torch.zeros(num_patches)

    def to(self, device: torch.device) -> "ConfidenceTracker":
        super().to(device)
        self.blocked = self.blocked.to(device)
        self.delta_ema_legacy = self.delta_ema_legacy.to(device)
        return self

    def observe(self, patch_idx: int, pred_loss: float, restored_loss: Optional[float] = None) -> None:
        if restored_loss is None:
            restored_loss = self.baseline_loss_ema
        super().observe(patch_idx, pred_loss, restored_loss)
        if self.delta_ema[patch_idx] > self.trend_threshold:
            self.blocked[patch_idx] = True

    def is_blocked(self, patch_idx: int) -> bool:
        return bool(self.blocked[patch_idx].item())


class DiffusionMaskContext:
    """Optional runtime context for external mask overrides."""

    _mask: Optional[torch.Tensor] = None

    @classmethod
    def set_mask(cls, mask: Optional[torch.Tensor]) -> None:
        cls._mask = mask

    @classmethod
    def get_mask(cls) -> Optional[torch.Tensor]:
        return cls._mask

    @classmethod
    def clear(cls) -> None:
        cls._mask = None
