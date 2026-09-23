import os
import sys
import time
from itertools import product
from multiprocessing import Process, Queue

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import DiffusionMaskPluginCallback, EarlyStopping
from basicts.utils import seed_everything
from basicts import BasicTSLauncher

AVAILABLE_GPUS = [0]
TORCH_SEED = int(os.environ.get("TORCH_SEED", "42"))
MODELS = ["PatchTST"]  # backbone; DiffusionMask is the plugin (like DropoutTS)

DATASETS = [
    ("SyntheticTS_noise0.1", 1),
    ("SyntheticTS_noise0.3", 1),
    ("SyntheticTS_noise0.5", 1),
    ("SyntheticTS_noise0.7", 1),
    ("SyntheticTS_noise0.9", 1),
]

DATASET_CONFIGS = {
    "default": {"input_lens": [96], "output_lens": [96, 192, 336, 720]}
}

HPARAMS = {
    "p_min": [0.05],
    "p_max": [0.2],
    "recon_loss_weight": [0.15],
    "warmup_epochs": [15],
    "counterfactual_interval": [5],
}
USE_CLEAN_TARGETS = True
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "100"))


def get_model_config(model_name, input_len, output_len, num_features, **kwargs):
    if model_name != "PatchTST":
        raise ValueError(f"Unknown backbone: {model_name}")
    cfg = PatchTSTConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=num_features,
    )
    return PatchTSTForForecasting, cfg


def run_experiment(model_name, dataset_name, num_features, input_len, output_len, gpu_id, **kwargs):
    seed_everything(TORCH_SEED)
    model_class, model_config = get_model_config(
        model_name, input_len, output_len, num_features, **kwargs
    )
    callbacks = [
        DiffusionMaskPluginCallback(
            p_min=kwargs.get("p_min", 0.05),
            p_max=kwargs.get("p_max", 0.2),
            warmup_mask_ratio=kwargs.get("warmup_mask_ratio", 0.05),
            recon_loss_weight=kwargs.get("recon_loss_weight", 0.15),
            controller_loss_weight=kwargs.get("controller_loss_weight", 0.05),
            warmup_epochs=kwargs.get("warmup_epochs", 15),
            counterfactual_interval=kwargs.get("counterfactual_interval", 5),
            counterfactual_probes=4,
        ),
        EarlyStopping(patience=15),
    ]
    cfg = BasicTSForecastingConfig(
        model=model_class, model_config=model_config,
        dataset_name=dataset_name, input_len=input_len, output_len=output_len,
        use_timestamps=False, use_clean_targets=USE_CLEAN_TARGETS,
        gpus=gpu_id, num_epochs=NUM_EPOCHS, batch_size=64, callbacks=callbacks, seed=TORCH_SEED,
        deterministic=True,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        cudnn_determinstic=True,
        train_data_num_workers=0, val_data_num_workers=0, test_data_num_workers=0,
    )
    # Do not resume leftover checkpoints from earlier ablation runs.
    import shutil
    from pathlib import Path
    run_dir = Path(cfg.ckpt_save_dir) / cfg.md5
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    BasicTSLauncher.launch_training(cfg)


def worker_task(gpu_queue, model_name, dataset_name, num_features, input_len, output_len):
    gpu_id = None
    task_id = f"{model_name}+DiffusionMaskPlugin/{dataset_name} ({input_len}->{output_len})"
    try:
        gpu_id = gpu_queue.get()
        seed_everything(TORCH_SEED)
        print(f"[Start] {task_id} on GPU {gpu_id} | seed={TORCH_SEED} | epochs={NUM_EPOCHS}")

        param_combinations = list(product(
            HPARAMS["p_min"], HPARAMS["p_max"],
            HPARAMS["recon_loss_weight"], HPARAMS["warmup_epochs"],
            HPARAMS["counterfactual_interval"],
        ))
        for idx, (p_min, p_max, recon_w, warm, cf_int) in enumerate(param_combinations):
            if p_max <= p_min:
                continue
            print(f"    [Exp {idx+1}/{len(param_combinations)}] recon_w={recon_w} warm={warm}")
            run_experiment(
                model_name, dataset_name, num_features, input_len, output_len, str(gpu_id),
                p_min=p_min, p_max=p_max, recon_loss_weight=recon_w,
                warmup_epochs=warm, counterfactual_interval=cf_int,
            )
    except Exception as e:
        print(f"[Error] {task_id} failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if gpu_id is not None:
            gpu_queue.put(gpu_id)
            print(f"[Done] {task_id} released GPU {gpu_id}")


if __name__ == "__main__":
    seed_everything(TORCH_SEED)
    gpu_queue = Queue()
    for gpu_id in AVAILABLE_GPUS:
        gpu_queue.put(gpu_id)

    processes = []
    print(
        f"PatchTST + DiffusionMask plugin | "
        f"masked-input forecast + diffusion recon + CF protect | "
        f"GPUs: {AVAILABLE_GPUS} | epochs: {NUM_EPOCHS} | seed: {TORCH_SEED}"
    )
    for model_name in MODELS:
        for dataset_name, num_features in DATASETS:
            config = DATASET_CONFIGS["default"]
            for input_len in config["input_lens"]:
                for output_len in config["output_lens"]:
                    p = Process(
                        target=worker_task,
                        args=(gpu_queue, model_name, dataset_name, num_features, input_len, output_len),
                    )
                    p.start()
                    processes.append(p)
                    time.sleep(0.1)

    print(f"Scheduled {len(processes)} tasks.")
    for p in processes:
        p.join()
    print("All experiments finished.")
