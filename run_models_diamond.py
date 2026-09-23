#!/usr/bin/env python3
"""Run DiffusionMask PLUGIN on all comparison forecasting models.

Backbones: iTransformer, TimeMixer, Crossformer, DynamicTMoE, TimesNet
(PatchTST+DM already has its own queue in run_real_diffusionmask.py.)
Illness is skipped: input_len=96 makes val/test empty.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.Crossformer import Crossformer, CrossformerConfig
from basicts.models.DynamicTMoE import DynamicTMoEConfig, DynamicTMoEForForecasting
from basicts.models.Informer import Informer, InformerConfig
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.models.TimeMixer import TimeMixerConfig, TimeMixerForForecasting
from basicts.models.TimesNet import TimesNetConfig, TimesNetForForecasting
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting
from basicts.runners.callback import DiffusionMaskPluginCallback, EarlyStopping
from basicts.utils import seed_everything

GPU = os.environ.get("GPU", "0")
TORCH_SEED = int(os.environ.get("TORCH_SEED", "42"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "20"))
DEFAULT_MODELS = "iTransformer,TimeMixer,Crossformer,DynamicTMoE,TimesNet,PatchTST"
MODELS = [m.strip() for m in os.environ.get("MODELS", DEFAULT_MODELS).split(",") if m.strip()]
DEFAULT_DATASETS = "Weather,ETTh1,ETTh2,ETTm1,ETTm2,Electricity"
DATASETS = [d.strip() for d in os.environ.get("DATASETS", DEFAULT_DATASETS).split(",") if d.strip()]
DEFAULT_HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "96,192,336,720").split(",")]
ILLNESS_HORIZONS = [int(x) for x in os.environ.get("ILLNESS_HORIZONS", "24,36,48,60").split(",")]
INPUT_LEN = int(os.environ.get("INPUT_LEN", "96"))

CYCLE_LEN = {
    "Weather": 144,
    "Electricity": 24,
    "ETTh1": 24,
    "ETTh2": 24,
    "ETTm1": 96,
    "ETTm2": 96,
    "Illness": 52,
    "Solar": 144,
    "Traffic": 24,
}

DM_HP = dict(
    p_min=0.05, p_max=0.35, warmup_epochs=5, warmup_mask_ratio=0.08,
    recon_loss_weight=0.1, controller_loss_weight=0.03, counterfactual_interval=4,
    counterfactual_probes=4, loss_tol=0.02, lambda_rho=2.0, mask_prediction=True,
)

OUT = Path(script_dir) / "logs" / "real_models_dm"
OUT.mkdir(parents=True, exist_ok=True)
RESULT = OUT / os.environ.get("RESULT_FILE", "results.json")
LOG = OUT / os.environ.get("LOG_FILE", "run.log")
SKIP_RESUME = os.environ.get("SKIP_RESUME", "0") == "1"

REGISTRY = {
    "iTransformer": (iTransformerForForecasting, iTransformerConfig),
    "TimeMixer": (TimeMixerForForecasting, TimeMixerConfig),
    "Crossformer": (Crossformer, CrossformerConfig),
    "DynamicTMoE": (DynamicTMoEForForecasting, DynamicTMoEConfig),
    "TimesNet": (TimesNetForForecasting, TimesNetConfig),
    "Informer": (Informer, InformerConfig),
    "PatchTST": (PatchTSTForForecasting, PatchTSTConfig),
}


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def num_features(dataset: str) -> int:
    meta = Path(script_dir) / "datasets" / dataset / "meta.json"
    if not meta.exists():
        raise FileNotFoundError(f"Missing prepared dataset: {meta}")
    return int(json.loads(meta.read_text())["num_vars"])


def batch_size_for(model_name: str, nfeat: int) -> int:
    # Slightly smaller than raw: plugin adds recon/controller tensors.
    heavy = {"TimesNet", "Crossformer", "DynamicTMoE", "Informer"}
    if nfeat >= 300:
        return 4 if model_name in heavy else 8
    if nfeat >= 100:
        return 8
    if model_name == "TimesNet":
        return 16
    if model_name == "Informer":
        return 16
    return 32


def make_model_config(model_name: str, input_len: int, horizon: int, nfeat: int, dataset: str):
    if model_name == "iTransformer":
        return iTransformerConfig(
            input_len=input_len, output_len=horizon, num_features=nfeat,
            hidden_size=128, n_heads=8, intermediate_size=256, num_layers=2, dropout=0.1,
        )
    if model_name == "TimeMixer":
        return TimeMixerConfig(
            input_len=input_len, output_len=horizon, num_features=nfeat,
            hidden_size=64, num_layers=2, down_sampling_window=2, down_sampling_layers=3,
            intermediate_size=128, dropout=0.1,
        )
    if model_name == "Crossformer":
        return CrossformerConfig(
            input_len=input_len, output_len=horizon, num_features=nfeat,
            hidden_size=256, n_heads=8, num_layers=2, intermediate_size=512,
            patch_len=16, dropout=0.05,
        )
    if model_name == "DynamicTMoE":
        return DynamicTMoEConfig(
            input_len=input_len, output_len=horizon, num_features=nfeat,
            hidden_size=128, patch_len=16, patch_stride=16,
            cycle_length=CYCLE_LEN.get(dataset, 1), dropout=0.2,
        )
    if model_name == "TimesNet":
        return TimesNetConfig(
            input_len=input_len, output_len=horizon, num_features=nfeat,
            hidden_size=32, intermediate_size=32, num_layers=2, top_k=5, dropout=0.1,
        )
    if model_name == "Informer":
        return InformerConfig(
            input_len=input_len, output_len=horizon, num_features=nfeat,
            hidden_size=128, n_heads=8, intermediate_size=512,
            num_encoder_layers=2, num_decoder_layers=1, dropout=0.05,
            use_timestamps=False,
        )
    if model_name == "PatchTST":
        return PatchTSTConfig(input_len=input_len, output_len=horizon, num_features=nfeat)
    raise KeyError(model_name)


def ckpt_root(model_cls) -> Path:
    return Path(script_dir) / "checkpoints" / f"{model_cls.__name__}_DM"


def read_metrics(model_cls, dataset: str, horizon: int, input_len: int) -> dict:
    base = ckpt_root(model_cls)
    cands = sorted(
        base.glob(f"{dataset}_{NUM_EPOCHS}_{input_len}_{horizon}/*/test_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No metrics for {model_cls.__name__}_DM {dataset} H={horizon}")
    return json.loads(cands[0].read_text())["overall"]


def run_one(model_name: str, dataset: str, horizon: int, nfeat: int, input_len: int) -> dict:
    seed_everything(TORCH_SEED)
    model_cls, _ = REGISTRY[model_name]
    model_config = make_model_config(model_name, input_len, horizon, nfeat, dataset)
    bs = int(os.environ["BATCH_SIZE"]) if os.environ.get("BATCH_SIZE") else batch_size_for(model_name, nfeat)
    log(
        f"  seed_everything({TORCH_SEED}) | CUBLAS={os.environ.get('CUBLAS_WORKSPACE_CONFIG')} "
        f"| bs={bs}"
    )
    cfg = BasicTSForecastingConfig(
        model=model_cls,
        model_config=model_config,
        dataset_name=dataset,
        input_len=input_len,
        output_len=horizon,
        use_timestamps=False,
        use_clean_targets=False,
        gpus=GPU,
        num_epochs=NUM_EPOCHS,
        batch_size=bs,
        callbacks=[
            DiffusionMaskPluginCallback(**DM_HP),
            EarlyStopping(patience=15, start_after_epoch=0),
        ],
        seed=TORCH_SEED,
        deterministic=True,
        tf32=False,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        cudnn_determinstic=True,
        train_data_num_workers=0,
        val_data_num_workers=0,
        test_data_num_workers=0,
        ckpt_save_strategy=list(range(1, NUM_EPOCHS + 1)),
        ckpt_save_dir=str(
            ckpt_root(model_cls) / f"{dataset}_{NUM_EPOCHS}_{input_len}_{horizon}"
        ),
    )
    run_dir = Path(cfg.ckpt_save_dir) / cfg.md5
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    log(f"[Start] DiffusionMask-plugin | {model_name} | {dataset}(C={nfeat}) {input_len}->{horizon}")
    BasicTSLauncher.launch_training(cfg)
    return read_metrics(model_cls, dataset, horizon, input_len)


def main() -> None:
    seed_everything(TORCH_SEED)
    table = {} if SKIP_RESUME or not RESULT.exists() else json.loads(RESULT.read_text())

    available = []
    for d in DATASETS:
        meta = Path(script_dir) / "datasets" / d / "meta.json"
        if not meta.exists():
            log(f"[skip missing data] {d}")
            continue
        if d == "Illness" and int(os.environ.get("INPUT_LEN", "36")) >= 96:
            log(f"[skip] {d}: input_len>=96 leaves empty val/test")
            continue
        available.append(d)

    unknown = [m for m in MODELS if m not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; known={list(REGISTRY)}")

    log(
        f"Real-world DiffusionMask PLUGIN | models={MODELS} | datasets={available} "
        f"| ep={NUM_EPOCHS} | seed={TORCH_SEED}"
    )

    for model_name in MODELS:
        row_m = table.get(model_name, {})
        for dataset in available:
            nfeat = num_features(dataset)
            input_len = 36 if dataset == "Illness" else INPUT_LEN
            hs = ILLNESS_HORIZONS if dataset == "Illness" else DEFAULT_HORIZONS
            row = row_m.get(dataset, {"per_h": {}})
            for h in hs:
                hk = str(h)
                if hk in row.get("per_h", {}) and "MSE" in row["per_h"][hk]:
                    log(f"  [skip] {model_name} {dataset} H={h} MSE={row['per_h'][hk]['MSE']:.4f}")
                    continue
                try:
                    met = run_one(model_name, dataset, h, nfeat, input_len)
                except Exception as exc:
                    log(f"[FAIL] {model_name} {dataset} H={h}: {exc}")
                    import traceback
                    traceback.print_exc()
                    continue
                row.setdefault("per_h", {})[hk] = {"MSE": met["MSE"], "MAE": met["MAE"]}
                log(f"[Done] {model_name} {dataset} H={h}: MSE={met['MSE']:.4f} MAE={met['MAE']:.4f}")
                mses = [row["per_h"][str(x)]["MSE"] for x in hs if str(x) in row["per_h"]]
                maes = [row["per_h"][str(x)]["MAE"] for x in hs if str(x) in row["per_h"]]
                if len(mses) == len(hs):
                    row["avg_mse"] = sum(mses) / len(mses)
                    row["avg_mae"] = sum(maes) / len(maes)
                    log(f"  {model_name} {dataset} AVG MSE={row['avg_mse']:.4f} MAE={row['avg_mae']:.4f}")
                row_m[dataset] = row
                table[model_name] = row_m
                RESULT.write_text(json.dumps(table, indent=2))

    log("===== SUMMARY =====")
    for model_name, row_m in table.items():
        for ds, row in row_m.items():
            if isinstance(row, dict) and "avg_mse" in row:
                log(f"  {model_name} {ds}: AVG MSE={row['avg_mse']:.4f} MAE={row['avg_mae']:.4f}")


if __name__ == "__main__":
    main()
