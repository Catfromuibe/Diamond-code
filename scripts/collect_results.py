#!/usr/bin/env python3
"""Collect test MSE/MAE from PatchTST + DiffusionMask plugin checkpoints."""
import json
import glob
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "checkpoints/PatchTSTForForecasting")
PATTERN = re.compile(r"SyntheticTS_noise(?P<noise>0\.\d+)_(?P<epochs>\d+)_96_(?P<horizon>\d+)$")


def main():
    results = defaultdict(dict)
    paths = glob.glob(os.path.join(BASE, "SyntheticTS_noise*", "*", "test_metrics.json"))
    for path in paths:
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m = PATTERN.match(run_dir)
        if not m:
            continue
        noise = m.group("noise")
        horizon = int(m.group("horizon"))
        epochs = int(m.group("epochs"))
        with open(path) as f:
            metrics = json.load(f)["overall"]
        prev = results[noise].get(horizon)
        if prev is None or epochs >= prev["epochs"]:
            results[noise][horizon] = {**metrics, "epochs": epochs}

    noises = sorted(results.keys(), key=float)
    horizons = sorted({h for d in results.values() for h in d})

    print("\n=== PatchTST + DiffusionMask Plugin ===")
    print("\n=== Per Horizon (MSE / MAE) ===")
    print(f"{'Noise':<8}" + "".join(f"{'H='+str(h)+' MSE':>12}{'H='+str(h)+' MAE':>12}" for h in horizons))
    for noise in noises:
        row = f"{noise:<8}"
        for h in horizons:
            m = results[noise].get(h, {})
            row += f"{m.get('MSE', float('nan')):>12.4f}{m.get('MAE', float('nan')):>12.4f}"
        print(row)

    print("\n=== Averaged over horizons (paper style) ===")
    print(f"{'Noise':<8}{'MSE':>12}{'MAE':>12}")
    for noise in noises:
        ms = [results[noise][h]["MSE"] for h in horizons if h in results[noise]]
        mas = [results[noise][h]["MAE"] for h in horizons if h in results[noise]]
        if ms:
            print(f"{noise:<8}{sum(ms)/len(ms):>12.4f}{sum(mas)/len(mas):>12.4f}")


if __name__ == "__main__":
    main()
