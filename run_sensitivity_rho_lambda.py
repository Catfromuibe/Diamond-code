#!/usr/bin/env python3
"""One-factor sensitivity of ρ_max and λ_diff.

Writes checkpoints/*_DMsens so production *_DM dirs are never touched.
Default (ρ_max=0.35, λ_diff=0.1) is reused from production JSONs by the plotter.
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
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.models.TimeMixer import TimeMixerConfig, TimeMixerForForecasting
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting
from basicts.runners.callback import DiffusionMaskPluginCallback, EarlyStopping
from basicts.utils import seed_everything

GPU = os.environ.get("GPU", "0")
TORCH_SEED = int(os.environ.get("TORCH_SEED", "42"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "20"))
MODEL = os.environ.get("MODEL", "iTransformer")
DATASET = os.environ.get("DATASET", "Illness")
HORIZONS = [int(x) for x in os.environ.get("HORIZONS", "24").split(",") if x.strip()]
P_MAX = float(os.environ.get("P_MAX", "0.35"))
LAMBDA_DIFF = float(os.environ.get("LAMBDA_DIFF", "0.1"))

DM_HP = dict(
    p_min=0.05, p_max=P_MAX, warmup_epochs=5, warmup_mask_ratio=0.08,
    recon_loss_weight=LAMBDA_DIFF, controller_loss_weight=0.03, counterfactual_interval=4,
    counterfactual_probes=4, loss_tol=0.02, lambda_rho=2.0, mask_prediction=True,
)

REGISTRY = {
    "iTransformer": (iTransformerForForecasting, iTransformerConfig),
    "TimeMixer": (TimeMixerForForecasting, TimeMixerConfig),
    "PatchTST": (PatchTSTForForecasting, PatchTSTConfig),
}

OUT = Path(script_dir) / "logs" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
TAG = os.environ.get(
    "TAG",
    f"{MODEL}_{DATASET}_pmax{P_MAX:g}_ldiff{LAMBDA_DIFF:g}_s{TORCH_SEED}",
)
RESULT = OUT / f"results_{TAG}.json"
LOG = OUT / f"run_{TAG}.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def num_features(dataset: str) -> int:
    meta = Path(script_dir) / "datasets" / dataset / "meta.json"
    return int(json.loads(meta.read_text())["num_vars"])


def batch_size_for(model_name: str, nfeat: int) -> int:
    if nfeat >= 100:
        return 8
    return 32


def make_model_config(model_name: str, input_len: int, horizon: int, nfeat: int):
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
    if model_name == "PatchTST":
        return PatchTSTConfig(input_len=input_len, output_len=horizon, num_features=nfeat)
    raise KeyError(model_name)


def hp_tag() -> str:
    return f"pmax{P_MAX:g}_ldiff{LAMBDA_DIFF:g}"


def run_one(model_name: str, dataset: str, horizon: int, nfeat: int, input_len: int) -> dict:
    seed_everything(TORCH_SEED)
    model_cls, _ = REGISTRY[model_name]
    bs = int(os.environ["BATCH_SIZE"]) if os.environ.get("BATCH_SIZE") else batch_size_for(model_name, nfeat)
    ckpt_dir = (
        Path(script_dir) / "checkpoints" / f"{model_cls.__name__}_DMsens"
        / f"{dataset}_{NUM_EPOCHS}_{input_len}_{horizon}_{hp_tag()}_s{TORCH_SEED}"
    )
    cfg = BasicTSForecastingConfig(
        model=model_cls,
        model_config=make_model_config(model_name, input_len, horizon, nfeat),
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
        ckpt_save_dir=str(ckpt_dir),
    )
    run_dir = Path(cfg.ckpt_save_dir) / cfg.md5
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    log(
        f"[Start] {model_name} {dataset} H={horizon} {hp_tag()} "
        f"ep={NUM_EPOCHS} bs={bs} ckpt={ckpt_dir}"
    )
    BasicTSLauncher.launch_training(cfg)
    mets = json.loads((run_dir / "test_metrics.json").read_text())["overall"]
    return {"MSE": mets["MSE"], "MAE": mets["MAE"]}


def main() -> None:
    if MODEL not in REGISTRY:
        raise SystemExit(f"unknown MODEL={MODEL}")
    nfeat = num_features(DATASET)
    input_len = 36 if DATASET == "Illness" else 96
    payload = {
        "model": MODEL,
        "dataset": DATASET,
        "seed": TORCH_SEED,
        "epochs": NUM_EPOCHS,
        "p_max": P_MAX,
        "lambda_diff": LAMBDA_DIFF,
        "per_h": {},
    }
    for h in HORIZONS:
        mets = run_one(MODEL, DATASET, h, nfeat, input_len)
        payload["per_h"][str(h)] = mets
        log(f"[Done] {MODEL} {DATASET} H={h} {hp_tag()} MSE={mets['MSE']:.4f} MAE={mets['MAE']:.4f}")
        RESULT.write_text(json.dumps(payload, indent=2))
    mses = [payload["per_h"][str(h)]["MSE"] for h in HORIZONS]
    maes = [payload["per_h"][str(h)]["MAE"] for h in HORIZONS]
    payload["avg_mse"] = sum(mses) / len(mses)
    payload["avg_mae"] = sum(maes) / len(maes)
    RESULT.write_text(json.dumps(payload, indent=2))
    log(
        f"[AVG] {MODEL} {DATASET} {hp_tag()} "
        f"MSE={payload['avg_mse']:.4f} MAE={payload['avg_mae']:.4f}"
    )


if __name__ == "__main__":
    main()
