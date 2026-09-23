# DiffusionMask

Diffusion-Informed Adaptive Patch Masking with Counterfactual Supervision for Time Series Forecasting.

This package is the **DiffusionMask plugin** on unchanged BasicTS backbones (PatchTST, iTransformer, TimeMixer, TimesNet, Crossformer, DynamicTMoE). It is **not** DropoutTS.

## Method

Training-time plugin (mask is **off** at inference):

1. Warm-up: fixed low-ratio random mask; joint loss \(L_{\text{pred}} + \lambda_{\text{diff}} L_{\text{diff}}\).
2. Diffusion recoverability evidence \(e_{\text{rec}}\) (EMA of denoising difficulty).
3. Hierarchical controller \(\pi_\varphi\): budget head + patch-safety head \(\rightarrow\) Top-\(k\) mask.
4. The same mask defines the forecasting view (mask token) and the diffusion view.
5. Counterfactual audit every \(K_a\) epochs \(\rightarrow\) ranking + safe budget \(\rho^*\).

Default plugin HPs used in the main table (`seed=42`, 20 epochs, MAE train, MSE/MAE test):

```
p_min=0.05  p_max=0.35  warmup=5 / 0.08
lambda_diff=0.1  lambda_ctrl=0.03  lambda_rho=2.0
T=4  K_a=4  probes=4  delta=0.02
```

Lookback \(L=96\) (ILI \(L=36\)); horizons \(H\in\{96,192,336,720\}\) (ILI \(\{24,36,48,60\}\)).

## Layout

```
src/basicts/
  modules/diffusion_mask.py                         # controller, diffusion head, CF auditor
  runners/callback/diffusion_mask_plugin_callback.py
  models/{PatchTST,iTransformer,TimeMixer,TimesNet,Crossformer,DynamicTMoE}/
  models/DiffusionMaskPatchTST/                     # optional dual-head variant
scripts/data_preparation/                           # CSV → training npz
run_real_models_diffusionmask.py                    # main multi-backbone protocol
```

## Setup

```bash
pip install -e .
# data zip already has datasets/; otherwise generate from raw CSV:
python scripts/data_preparation/ETTh1/generate_training_data.py
```

Torch is not pinned here; use the same CUDA build as training (`conda activate redout` on the original machine).

## Run

```bash
# one backbone × one dataset, plugin on
GPU=0 MODELS=TimeMixer DATASETS=ETTh1 HORIZONS=96 \
  python -u run_real_models_diffusionmask.py

# full paper protocol (6 backbones, all datasets / horizons)
python -u run_real_models_diffusionmask.py
```

Raw backbone: omit `DiffusionMaskPluginCallback` (same runner, no plugin).
Inference always uses the full history (`m=0`).
