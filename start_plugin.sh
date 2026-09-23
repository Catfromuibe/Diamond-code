#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export NUM_EPOCHS="${NUM_EPOCHS:-20}"
export TORCH_SEED="${TORCH_SEED:-42}"
mkdir -p logs
exec python -u run_real_models_diffusionmask.py
