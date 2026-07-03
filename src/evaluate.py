"""Evaluate a trained checkpoint on the held-out eval towns (or a saved
eval dataset), against a slope-blind random baseline.

Writes results.json and comparison figures.
"""

import argparse
import json
import os

import numpy as np
import torch

from data import GRID, slope_from_elevation
from metrics import evaluate, random_baseline
from model import UNet, Diffusion
from sample import load_model, expand_once, binarize


def eval_dataset(diff, npz_path, out_dir, n_max=64, seed=0):
    d = np.load(npz_path, allow_pickle=True)
    conds, targets, meta = d["cond"], d["target"], d["meta"]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(conds))
    # eval sets are built with jitter=0 but 8x augmentation; take every 8th
    idx = idx[::8][:n_max]
    rows, rows_base = [], []
    os.makedirs(out_dir, exist_ok=True)
    for n, i in enumerate(idx):
        cond, target = conds[i], targets[i]
        # reconstruct pieces from the sample
        ez, sl_norm, core_built, core_roads = cond
        slope = sl_norm * 30.0
        real_ring_roads = (target[0] > 0).astype(np.uint8)
        real_ring_built = (target[1] > 0).astype(np.uint8)
        real_roads = real_ring_roads | core_roads.astype(np.uint8)
        real_built = real_ring_built | core_built.astype(np.uint8)
        # model
        elev_proxy = ez * 3.0 * 100.0  # only relative values matter downstream
        gen_roads, gen_built, _ = expand_once(
            diff, elev_proxy, core_built, core_roads, seed=seed + n)
        m = evaluate(slope, core_built, core_roads,
                     gen_roads.astype(bool), gen_built,
                     real_roads, real_built)
        m["town"] = str(meta[i][0]); m["region"] = str(meta[i][2])
        rows.append(m)
        # baseline
        b_roads, b_built = random_baseline(rng, slope, core_built,
                                           core_roads, real_built)
        mb = evaluate(slope, core_built, core_roads,
                      b_roads.astype(bool), b_built, real_roads, real_built)
        mb["town"] = str(meta[i][0]); mb["region"] = str(meta[i][2])
        rows_base.append(mb)
        print(f"[eval] {n+1}/{len(idx)} {m['town']}: "
              f"slope_ks={m['slope_ks']:.3f} conn={m['road_connectivity']:.3f}")

    def agg(rs, key):
        v = [r[key] for r in rs if not np.isnan(r[key])]
        return dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v))

    keys = ["slope_ks", "road_connectivity", "built_contiguity", "volume_ratio"]
    summary = {
        "model": {k: agg(rows, k) for k in keys},
        "random_baseline": {k: agg(rows_base, k) for k in keys},
        "per_town_model": rows,
        "per_town_baseline": rows_base,
    }
    for region in ("europe", "asia"):
        rr = [r for r in rows if r["region"] == region]
        if rr:
            summary[f"model_{region}"] = {k: agg(rr, k) for k in keys}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary["model"][k] for k in keys}, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/default/ckpt.pt")
    ap.add_argument("--data", default="data/eval.npz")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff = load_model(args.ckpt, device)
    eval_dataset(diff, args.data, args.out, n_max=args.n)
