from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

from ..config.dynamic_tmoe_config import DynamicTMoEConfig
from .moe_layer import TemporalDynamicMoELayer
from .patch_embed import PatchEmbedding
from .revin import RevIN


class DynamicTMoE(nn.Module):
    """
    Paper: Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework
           for Non-Stationary Time Series Forecasting
    Official Code: https://github.com/andone-07/Dynamic-TMoE
    Venue: ICML 2026
    Task: Long-term Time Series Forecasting

    Adapted from the official `models.Dynamic_TMoE.model.Model` with import
    paths rewritten for BasicTS. Architecture and drift logic are unchanged.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        padding = self.stride
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            self.patch_len,
            self.stride,
            padding,
            configs.dropout
        )

        self.num_patches = int((configs.seq_len - self.patch_len) / self.stride + 2)

        self.num_temporal_moe_layers = getattr(configs, "num_temporal_moe_layers", 2)
        num_experts = getattr(configs, "num_experts", 4)
        num_drift_experts = getattr(configs, "num_drift_experts", 3)
        num_rnn_layers = getattr(configs, "num_rnn_layers", 2)
        top_k = getattr(configs, "top_k", 2)

        self.drift_detection_enabled = getattr(configs, "enable_drift_detection", True)
        self.drift_window_size = getattr(configs, "drift_window_size", 576)
        self.drift_history_size = getattr(configs, "drift_history_size", 20)
        self.drift_k_sigma = getattr(configs, "drift_k_sigma", 2.0)

        self.train_epochs = getattr(configs, "train_epochs", 10)
        self.learning_rate = getattr(configs, "learning_rate", 0.0001)
        self.finetune_patience = getattr(configs, "finetune_patience", 5)

        self.cycle_length = configs.cycle_length

        self.channel_independence = getattr(configs, "channel_independence", 1)
        self.use_relation_layer = getattr(configs, "use_relation_layer", 1)

        self.temporal_moe_layers = nn.ModuleList([
            TemporalDynamicMoELayer(
                d_model=configs.d_model,
                num_patches=self.num_patches,
                n_vars=configs.enc_in,
                cycle_length=self.cycle_length,
                num_experts=num_experts,
                num_drift_experts=num_drift_experts,
                dropout=configs.dropout,
                num_rnn_layers=num_rnn_layers,
                drift_window_size=self.drift_window_size,
                use_relation_layer=self.use_relation_layer,
                enable_drift_detection=bool(self.drift_detection_enabled),
                top_k=top_k
            ) for _ in range(self.num_temporal_moe_layers)
        ])

        revin_affine = getattr(configs, "revin_affine", True)
        self.revin_layer = RevIN(configs.enc_in, eps=1e-5, affine=revin_affine)

        self.register_buffer("global_patch_counter", torch.tensor(0, dtype=torch.long))

        self.head_input_dim = configs.d_model * int((configs.seq_len - self.patch_len) / self.stride + 2)
        if self.channel_independence:
            self.projection_layers = nn.ModuleList([
                nn.Linear(self.head_input_dim, configs.pred_len)
                for _ in range(configs.enc_in)
            ])
        else:
            self.projection_layer = nn.Linear(self.head_input_dim, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        Args:
            x_enc: [B, S, N]
            x_mark_enc: [B, S, mark_dim] or None
        Returns:
            predictions: [B, pred_len, N]
        """
        B, S, N = x_enc.shape
        x_norm, means, stdev = self.revin_layer.normalize(x_enc)

        x_patches = x_norm.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_patches)

        if x_mark_enc is not None:
            first_timestamp = x_mark_enc[:, 0, :]
            patch_cycle_indices = (first_timestamp[:, 0] % self.cycle_length).long()
        else:
            patch_cycle_indices = torch.zeros(B, dtype=torch.long, device=x_enc.device)

        temporal_hidden_states = [None] * self.num_temporal_moe_layers
        final_temporal_hidden_states = []

        batch_start_patch_idx = self.global_patch_counter.item()
        BN = enc_out.shape[0]
        P = enc_out.shape[1]

        for layer_idx, moe_layer in enumerate(self.temporal_moe_layers):
            enc_out, new_hidden = moe_layer(
                enc_out,
                hidden_state=temporal_hidden_states[layer_idx],
                batch_start_patch_idx=batch_start_patch_idx,
                patch_cycle_indices=patch_cycle_indices
            )
            final_temporal_hidden_states.append(new_hidden)

        if self.training:
            self.global_patch_counter += BN * P

        self._last_temporal_hidden_states = final_temporal_hidden_states

        P = enc_out.shape[1]
        D = enc_out.shape[2]
        enc_out = enc_out.view(B, N, P, D)
        enc_out_flat = enc_out.reshape(B, N, P * D)

        if self.channel_independence:
            predictions = torch.stack([
                self.projection_layers[i](enc_out_flat[:, i, :])
                for i in range(N)
            ], dim=2)
        else:
            predictions = self.projection_layer(enc_out_flat)
            predictions = predictions.permute(0, 2, 1)

        predictions = self.revin_layer.denormalize(predictions, means, stdev)
        return predictions

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise NotImplementedError(f"Task {self.task_name} not supported")

    def enable_drift_detection(self, window_size=576, history_size=50, k_sigma=3.0):
        for moe_layer in self.temporal_moe_layers:
            moe_layer.enable_drift_detection(
                window_size=window_size,
                history_size=history_size,
                k_sigma=k_sigma
            )

    def check_and_handle_drift(self, train_mode=True):
        for moe_layer in self.temporal_moe_layers:
            moe_layer.check_and_handle_drift(
                finetune_epochs=self.train_epochs,
                finetune_lr=self.learning_rate,
                finetune_patience=self.finetune_patience,
                train_mode=train_mode
            )


class DynamicTMoEForForecasting(nn.Module):
    """
    BasicTS forecasting wrapper around official Dynamic TMoE.

    `forward(inputs, inputs_timestamps=None)` matches other BasicTS models and
    returns [batch, output_len, num_features]. During training, MMD drift checks
    and expert evolution are triggered the same way as the official trainer.
    """

    def __init__(self, config: DynamicTMoEConfig):
        super().__init__()
        official = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=config.input_len,
            pred_len=config.output_len,
            enc_in=config.num_features,
            d_model=config.hidden_size,
            patch_len=config.patch_len,
            stride=config.patch_stride,
            dropout=config.dropout,
            cycle_length=max(int(config.cycle_length), 1),
            channel_independence=int(bool(config.channel_independence)),
            use_relation_layer=int(bool(config.use_relation_layer)),
            revin_affine=bool(config.revin_affine),
            num_temporal_moe_layers=config.num_temporal_moe_layers,
            num_experts=config.num_experts,
            num_drift_experts=config.num_drift_experts,
            num_rnn_layers=config.num_rnn_layers,
            top_k=config.top_k,
            enable_drift_detection=bool(config.enable_drift_detection),
            drift_window_size=config.drift_window_size,
            drift_history_size=config.drift_history_size,
            drift_k_sigma=config.drift_k_sigma,
            train_epochs=config.finetune_epochs,
            learning_rate=config.finetune_lr,
            finetune_patience=config.finetune_patience,
        )
        self.output_len = config.output_len
        self.backbone = DynamicTMoE(official)
        if official.enable_drift_detection:
            self.backbone.enable_drift_detection(
                window_size=official.drift_window_size,
                history_size=official.drift_history_size,
                k_sigma=official.drift_k_sigma,
            )

    def _maybe_handle_drift(self):
        if not self.training or not self.backbone.drift_detection_enabled:
            return
        first_layer = self.backbone.temporal_moe_layers[0]
        if first_layer.drift_detector is None:
            return
        if not first_layer.drift_detector.should_check_drift():
            return
        with torch.no_grad():
            self.backbone.check_and_handle_drift(train_mode=True)

    def forward(
            self,
            inputs: torch.Tensor,
            inputs_timestamps: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, input_len, num_features]
            inputs_timestamps: optional [batch_size, input_len, num_timestamps]
        Returns:
            prediction: [batch_size, output_len, num_features]
        """
        self._maybe_handle_drift()
        return self.backbone.forecast(inputs, inputs_timestamps)
