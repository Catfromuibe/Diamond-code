#!/usr/bin/env python3
"""Run remaining forecasting models on real datasets (no Informer).

Models: iTransformer, TimeMixer, Crossformer, DynamicTMoE, TimesNet
Datasets: ETTh1/h2/m1/m2, Weather, Electricity, Illness (skip missing)
Horizons: 96/192/336/720, except Illness uses 24/36/48/60.
Uses seed_everything + CUBLAS_WORKSPACE_CONFIG for reproducibility.
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
from basicts.models.TimeMixer import TimeMixerConfig, TimeMixerForForecasting
from basicts.models.TimesNet import TimesNetConfig, TimesNetForForecasting
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting
from basicts.runners.callback import EarlyStopping
from basicts.utils import seed_everything

GPU = os.environ.get("GPU", "0")
TORCH_SEED = int(os.environ.get("TORCH_SEED", "42"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "20"))
DEFAULT_MODELS = "iTransformer,TimeMixer,Crossformer,DynamicTMoE,TimesNet"
MODELS = [m.strip() for m in os.environ.get("MODELS", DEFAULT_MODELS).split(",") if m.strip()]
DEFAULT_DATASETS = "Weather,ETTh1,ETTh2,ETTm1,ETTm2,Electricity,Illness"
DATASETS = [d.strip() for d in os.environ.get("DATASETS", DEFAULT_DATASETS).split(",") if d.strip()]
DEFAULT_HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "96,192,336,720").split(",")]
ILLNESS_HORIZONS = [int(x) for x in os.environ.get("ILLNESS_HORIZONS", "24,36,48,60").split(",")]

CYCLE_LEN = {
    "Weather": 144,
    "Electricity": 24,
    "ETTh1": 24,
    "ETTh2": 24,
    "ETTm1": 96,
    "ETTm2": 96,
    "Illness": 52,
}

OUT = Path(script_dir) / "logs" / "real_models"
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


def horizons_for(dataset: str) -> list[int]:
    return ILLNESS_HORIZONS if dataset == "Illness" else DEFAULT_HORIZONS


def batch_size_for(model_name: str, nfeat: int) -> int:
    if nfeat >= 300:
        return 8 if model_name in {"TimesNet", "Crossformer", "DynamicTMoE"} else 16
    if nfeat >= 100:
        return 16
    return 32 if model_name == "TimesNet" else 64


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
    raise KeyError(model_name)


def read_metrics(model_cls, dataset: str, horizon: int, input_len: int) -> dict:
    base = Path(script_dir) / "checkpoints" / model_cls.__name__
    cands = sorted(
        base.glob(f"{dataset}_{NUM_EPOCHS}_{input_len}_{horizon}/*/test_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No metrics for {model_cls.__name__} {dataset} H={horizon}")
    return json.loads(cands[0].read_text())["overall"]


def run_one(model_name: str, dataset: str, horizon: int, nfeat: int, input_len: int) -> dict:
    seed_everything(TORCH_SEED)
    model_cls, _ = REGISTRY[model_name]
    model_config = make_model_config(model_name, input_len, horizon, nfeat, dataset)
    bs = batch_size_for(model_name, nfeat)
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
        callbacks=[EarlyStopping(patience=15, start_after_epoch=0)],
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
    )
    run_dir = Path(cfg.ckpt_save_dir) / cfg.md5
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    log(f"[Start] {model_name} | {dataset}(C={nfeat}) {input_len}->{horizon}")
    BasicTSLauncher.launch_training(cfg)
    return read_metrics(model_cls, dataset, horizon, input_len)


def main() -> None:
    seed_everything(TORCH_SEED)
    table = {} if SKIP_RESUME or not RESULT.exists() else json.loads(RESULT.read_text())

    available = []
    for d in DATASETS:
        meta = Path(script_dir) / "datasets" / d / "meta.json"
        if meta.exists():
            available.append(d)
        else:
            log(f"[skip missing data] {d}")

    unknown = [m for m in MODELS if m not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; known={list(REGISTRY)}")

    log(
        f"Real-world other models | models={MODELS} | datasets={available} "
        f"| ep={NUM_EPOCHS} | seed={TORCH_SEED}"
    )

    for model_name in MODELS:
        row_m = table.get(model_name, {})
        for dataset in available:
            nfeat = num_features(dataset)
            input_len = 96
            hs = horizons_for(dataset)
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
