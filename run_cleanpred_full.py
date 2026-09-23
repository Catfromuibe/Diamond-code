"""Sequential full sweep: PatchTST + DiffusionMask clean-pred plugin."""
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
from basicts.runners.callback import DiffusionMaskPluginCallback, EarlyStopping
from basicts.utils import seed_everything

TORCH_SEED = int(os.environ.get("TORCH_SEED", "42"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "100"))
GPU = os.environ.get("GPU", "0")

DATASETS = [
    ("SyntheticTS_noise0.1", 1),
    ("SyntheticTS_noise0.3", 1),
    ("SyntheticTS_noise0.5", 1),
    ("SyntheticTS_noise0.7", 1),
    ("SyntheticTS_noise0.9", 1),
]
HORIZONS = [96, 192, 336, 720]


def run_one(dataset_name: str, num_features: int, horizon: int) -> None:
    seed_everything(TORCH_SEED)
    model_cfg = PatchTSTConfig(
        input_len=96, output_len=horizon, num_features=num_features,
    )
    callbacks = [
        DiffusionMaskPluginCallback(
            p_min=0.05,
            p_max=0.4,
            warmup_mask_ratio=0.1,
            recon_loss_weight=0.3,
            controller_loss_weight=0.05,
            warmup_epochs=3,
            counterfactual_interval=3,
            counterfactual_probes=4,
            hard_time_mask=False,
            mask_fill="token",
            soft_bernoulli=False,
            reset_best_after_warmup=False,
            mask_prediction=False,
        ),
        EarlyStopping(patience=15, start_after_epoch=0),
    ]
    cfg = BasicTSForecastingConfig(
        model=PatchTSTForForecasting,
        model_config=model_cfg,
        dataset_name=dataset_name,
        input_len=96,
        output_len=horizon,
        use_timestamps=False,
        use_clean_targets=True,
        gpus=GPU,
        num_epochs=NUM_EPOCHS,
        batch_size=64,
        callbacks=callbacks,
        seed=TORCH_SEED,
        deterministic=True,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        cudnn_determinstic=True,
        train_data_num_workers=0,
        val_data_num_workers=0,
        test_data_num_workers=0,
        # Keep every epoch ckpt briefly; avoid rename-to-.bak races
        ckpt_save_strategy=list(range(1, NUM_EPOCHS + 1)),
    )
    run_dir = Path(cfg.ckpt_save_dir) / cfg.md5
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    print(
        f"[Start] {dataset_name} 96->{horizon} | seed={TORCH_SEED} epochs={NUM_EPOCHS}",
        flush=True,
    )
    BasicTSLauncher.launch_training(cfg)
    print(f"[Done] {dataset_name} 96->{horizon}", flush=True)


if __name__ == "__main__":
    seed_everything(TORCH_SEED)
    tasks = [(d, n, h) for d, n in DATASETS for h in HORIZONS]
    print(
        f"PatchTST + DiffusionMask clean-pred | sequential | "
        f"{len(tasks)} tasks | epochs={NUM_EPOCHS} | seed={TORCH_SEED}",
        flush=True,
    )
    for i, (dataset, nfeat, horizon) in enumerate(tasks, 1):
        print(f"\n===== Task {i}/{len(tasks)} =====", flush=True)
        try:
            run_one(dataset, nfeat, horizon)
        except Exception as e:
            print(f"[Error] {dataset} H={horizon}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    print("All experiments finished.", flush=True)
