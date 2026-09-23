from dataclasses import dataclass, field

from basicts.models.PatchTST.config.patchtst_config import PatchTSTConfig


@dataclass
class DiffusionMaskPatchTSTConfig(PatchTSTConfig):
    """Config for hierarchical diffusion-guided patch masking."""

    # Prefer non-overlapping patches (method default)
    patch_len: int = field(default=16, metadata={"help": "Patch length."})
    patch_stride: int = field(default=16, metadata={"help": "Stride; set equal to patch_len for non-overlap."})
    padding: bool = field(default=False, metadata={"help": "Pad before patching."})

    p_min: float = field(default=0.05, metadata={"help": "Minimum sample mask budget."})
    p_max: float = field(default=0.5, metadata={"help": "Maximum sample mask budget."})
    warmup_mask_ratio: float = field(default=0.1, metadata={"help": "Fixed random mask ratio during warm-up."})
    warmup_epochs: int = field(default=5, metadata={"help": "Warm-up epochs before enabling controller."})

    recon_loss_weight: float = field(default=0.5, metadata={"help": "Weight for diffusion reconstruction loss."})
    controller_loss_weight: float = field(default=0.1, metadata={"help": "Weight for mask-controller supervision."})
    num_diffusion_steps: int = field(default=4, metadata={"help": "Diffusion timesteps."})

    counterfactual_interval: int = field(default=2, metadata={"help": "Run counterfactual recovery every K epochs."})
    counterfactual_probes: int = field(default=4, metadata={"help": "Number of masked patches to probe per audit."})
    confidence_probe_prob: float = field(default=0.1, metadata={"help": "Unused legacy; kept for compat."})
    confidence_trend_threshold: float = field(default=0.0, metadata={"help": "Unused legacy; kept for compat."})
    mask_update_interval: int = field(default=2, metadata={"help": "Alias of counterfactual_interval."})
