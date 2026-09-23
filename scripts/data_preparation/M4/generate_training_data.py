#!/usr/bin/env python3
"""Organize the M4 short-term forecasting dataset (100,000 univariate series).

M4 is not a single (T, C) grid like ETT/PeMS: each frequency has its own
horizon (6–48). We keep the official Time-Series-Library files and write
meta + per-frequency summaries so the six subsets can be loaded later.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

RAW = os.path.join(base_dir, "datasets", "raw_data", "M4")
OUT = os.path.join(base_dir, "datasets", "M4")

# TimeMixer / N-BEATS M4Meta
M4_META = {
    "Yearly": {"horizon": 6, "frequency": 1, "seasonality": 1},
    "Quarterly": {"horizon": 8, "frequency": 4, "seasonality": 4},
    "Monthly": {"horizon": 18, "frequency": 12, "seasonality": 12},
    "Weekly": {"horizon": 13, "frequency": 1, "seasonality": 1},
    "Daily": {"horizon": 14, "frequency": 1, "seasonality": 1},
    "Hourly": {"horizon": 48, "frequency": 24, "seasonality": 24},
}


def main() -> None:
    print("---------- Organizing M4 data ----------")
    os.makedirs(OUT, exist_ok=True)
    for fname in ["M4-info.csv", "training.npz", "test.npz"]:
        src = os.path.join(RAW, fname)
        dst = os.path.join(OUT, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        print(f"  copied {fname} ({os.path.getsize(dst)} bytes)")

    info = pd.read_csv(os.path.join(OUT, "M4-info.csv"))
    train = np.load(os.path.join(OUT, "training.npz"), allow_pickle=True)
    test = np.load(os.path.join(OUT, "test.npz"), allow_pickle=True)

    subsets = {}
    for sp, counts in info["SP"].value_counts().sort_index().items():
        mask = info["SP"].values == sp
        idx = np.where(mask)[0]
        tr_lens = [len(train[i]) for i in idx[: min(64, len(idx))]]
        te_lens = [len(test[i]) for i in idx[: min(64, len(idx))]]
        meta_sp = M4_META.get(sp, {})
        subsets[sp] = {
            "n_series": int(counts),
            "horizon": int(info.loc[mask, "Horizon"].iloc[0]),
            "frequency": int(info.loc[mask, "Frequency"].iloc[0]),
            "seasonality": meta_sp.get("seasonality"),
            "train_len_sample": tr_lens[:5],
            "test_len_sample": te_lens[:5],
        }
        print(f"  {sp:10} n={counts:6d} H={subsets[sp]['horizon']} freq={subsets[sp]['frequency']}")

    description = {
        "name": "M4",
        "domain": "mixed (M4 competition)",
        "n_series": int(len(info)),
        "subsets": subsets,
        "files": ["M4-info.csv", "training.npz", "test.npz"],
        "regular_settings": {
            "metrics": ["SMAPE", "MASE", "OWA"],
            "note": "Univariate; train/test are per-series arrays, not a shared (T, C) grid.",
        },
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(description, f, indent=4)
    print(f"  meta -> {OUT}/meta.json")
    print()


if __name__ == "__main__":
    main()
