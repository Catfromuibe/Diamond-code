from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class DynamicTMoEConfig(BasicTSModelConfig):
    """
    Config for Dynamic TMoE (ICML 2026).

    Official code: https://github.com/andone-07/Dynamic-TMoE
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting."})
    num_features: int = field(default=None, metadata={"help": "Number of features / channels."})
    hidden_size: int = field(default=512, metadata={"help": "Model hidden size (official d_model)."})
    patch_len: int = field(default=16, metadata={"help": "Patch length."})
    patch_stride: int = field(default=16, metadata={"help": "Stride for patching."})
    dropout: float = field(default=0.2, metadata={"help": "Dropout rate."})
    cycle_length: int = field(default=1, metadata={"help": "Cyclic period used by the relation layer."})
    channel_independence: bool = field(default=True, metadata={"help": "Use per-channel forecasting heads."})
    use_relation_layer: bool = field(default=True, metadata={"help": "Use cyclic relation layer in experts."})
    use_revin: bool = field(default=True, metadata={"help": "Use RevIN (always on in official code)."})
    revin_affine: bool = field(default=True, metadata={"help": "Affine parameters in RevIN."})
    num_temporal_moe_layers: int = field(default=2, metadata={"help": "Number of Temporal MoE layers."})
    num_experts: int = field(default=4, metadata={"help": "Number of base heterogeneous experts."})
    num_drift_experts: int = field(default=3, metadata={"help": "Number of reserved drift experts."})
    num_rnn_layers: int = field(default=2, metadata={"help": "Router GRU depth (official hyper-parameter)."})
    top_k: int = field(default=2, metadata={"help": "Sparse routing top-k."})
    enable_drift_detection: bool = field(default=True, metadata={"help": "Enable MMD drift detection and expert evolution."})
    drift_window_size: int = field(default=576, metadata={"help": "MMD window size in samples."})
    drift_history_size: int = field(default=20, metadata={"help": "MMD history buffer size."})
    drift_k_sigma: float = field(default=3.0, metadata={"help": "k-sigma threshold for MMD drift."})
    finetune_epochs: int = field(default=10, metadata={"help": "Epochs for localized expert pretraining."})
    finetune_lr: float = field(default=1e-4, metadata={"help": "Learning rate for localized expert pretraining."})
    finetune_patience: int = field(default=5, metadata={"help": "Early-stop patience for expert pretraining."})
