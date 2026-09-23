#!/usr/bin/env python3
"""Prepare TimeMixer short-term PeMS datasets (PEMS03/04/07/08).

Uses the official 6:2:2 split and the traffic-flow channel only, matching
TimesNet / TimeMixer Dataset_PEMS (input 96, predict 12).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# 5-minute sampling → 288 steps / day
STEPS_PER_DAY = 288
FREQUENCY = 5
START_DATES = {
    "PEMS03": "2018-09-01 00:00:00",
    "PEMS04": "2018-01-01 00:00:00",
    "PEMS07": "2017-05-01 00:00:00",
    "PEMS08": "2016-07-01 00:00:00",
}


def add_temporal_features(n: int, start: str) -> np.ndarray:
    index = np.arange(n)
    dates = np.datetime64(start) + index.astype("timedelta64[m]") * FREQUENCY
    # numpy datetime64[m] → pandas-like fields without importing pandas
    # time of day from step index (more stable than calendar for 5-min grids)
    tod = (index % STEPS_PER_DAY) / STEPS_PER_DAY
    # day-of-week etc. from the synthetic calendar
    # 1970-01-01 was Thursday; compute via datetime
    import datetime as dt

    origin = dt.datetime.fromisoformat(start)
    dow, dom, doy = [], [], []
    delta = dt.timedelta(minutes=FREQUENCY)
    cur = origin
    for _ in range(n):
        dow.append(cur.weekday() / 7)
        dom.append((cur.day - 1) / 31)
        doy.append((cur.timetuple().tm_yday - 1) / 366)
        cur += delta
    return np.stack([tod, np.array(dow), np.array(dom), np.array(doy)], axis=-1).astype(np.float32)


def prepare_one(name: str) -> None:
    npz_path = os.path.join(base_dir, "datasets", "raw_data", "PEMS", f"{name}.npz")
    out_dir = os.path.join(base_dir, "datasets", name)
    os.makedirs(out_dir, exist_ok=True)

    raw = np.load(npz_path, allow_pickle=True)["data"]
    # (T, N, C) → flow channel, same as TimeMixer/TimesNet
    if raw.ndim == 3:
        data = raw[:, :, 0]
    else:
        data = raw
    print(f"{name} raw {raw.shape} -> flow {data.shape}")

    n = data.shape[0]
    train_len = int(n * 0.6)
    val_len = int(n * 0.2)
    timestamps = add_temporal_features(n, START_DATES[name])

    splits = {
        "train": (0, train_len),
        "val": (train_len, train_len + val_len),
        "test": (train_len + val_len, n),
    }
    for split, (a, b) in splits.items():
        np.save(os.path.join(out_dir, f"{split}_data.npy"), data[a:b].astype(np.float32))
        np.save(os.path.join(out_dir, f"{split}_timestamps.npy"), timestamps[a:b])
        print(f"  {split}: {b - a} x {data.shape[1]}")

    adj_src = os.path.join(base_dir, "datasets", "raw_data", "PEMS", f"{name}.csv")
    adj_dst = os.path.join(out_dir, f"{name}.csv")
    if os.path.exists(adj_src) and not os.path.exists(adj_dst):
        with open(adj_src, "rb") as fsrc, open(adj_dst, "wb") as fdst:
            fdst.write(fsrc.read())

    description = {
        "name": name,
        "domain": "transportation",
        "frequency (minutes)": FREQUENCY,
        "shape": list(data.shape),
        "timestamps_shape": list(timestamps.shape),
        "timestamps_description": ["time of day", "day of week", "day of month", "day of year"],
        "num_time_steps": int(data.shape[0]),
        "num_vars": int(data.shape[1]),
        "has_graph": os.path.exists(adj_dst),
        "regular_settings": {
            "train_val_test_ratio": [0.6, 0.2, 0.2],
            "norm_each_channel": True,
            "rescale": False,
            "metrics": ["MAE", "MAPE", "RMSE"],
            "null_val": 0.0,
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(description, f, indent=4)
    print(f"  meta -> {out_dir}/meta.json")


def main() -> None:
    names = sys.argv[1:] or ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
    for name in names:
        print(f"---------- Generating {name} data ----------")
        prepare_one(name)
        print()


if __name__ == "__main__":
    main()
