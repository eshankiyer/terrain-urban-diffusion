"""Sampling: real-town expansion mode and sandbox (game) mode.

Real mode:    condition on a real town's terrain + current footprint.
Sandbox mode: condition on any heightmap (fractal here, or a game-exported
              heightmap PNG) with a seed settlement; loop the model to grow
              a city iteratively -- each generation becomes the next core.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from data import (GRID, binary_dilate, fractal_heightmap,
                  slope_from_elevation, fetch_elevation, fetch_osm,
                  rasterize_osm, make_sample)
from model import UNet, Diffusion


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet(base=ck["args"].get("base", 64)).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return Diffusion(model, device)


def binarize(x, thresh=0.0):
    """Model output in [-1,1] -> binary mask."""
    return (x > thresh).astype(np.uint8)


def cond_from_arrays(elev, core_built, core_roads):
    ez = (elev - elev.mean()) / (elev.std() + 1e-6)
    ez = np.clip(ez, -3, 3) / 3.0
    sl = np.clip(slope_from_elevation(elev) / 30.0, 0, 1)
    return np.stack([ez, sl, core_built.astype(np.float32),
                     core_roads.astype(np.float32)]).astype(np.float32)


@torch.no_grad()
def expand_once(diff, elev, core_built, core_roads, steps=50, seed=None):
    device = diff.device
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    cond = torch.from_numpy(
        cond_from_arrays(elev, core_built, core_roads))[None].to(device)
    out = diff.sample_ddim(cond, steps=steps, generator=g)[0].cpu().numpy()
    new_roads = binarize(out[0]) | core_roads.astype(np.uint8)
    new_built = binarize(out[1]) | core_built.astype(np.uint8)
    return new_roads, new_built, out


def sandbox_grow(diff, n_rounds=4, seed=0, relief=300.0, heightmap=None):
    """Game mode: grow a city from a seed on arbitrary terrain."""
    rng = np.random.default_rng(seed)
    elev = heightmap if heightmap is not None else fractal_heightmap(rng, relief=relief)
    slope = slope_from_elevation(elev)
    flat = slope < np.percentile(slope, 25)
    cy, cx = np.unravel_index(np.argmax(flat * rng.random(elev.shape)), elev.shape)
    built = np.zeros((GRID, GRID), dtype=np.uint8)
    built[max(cy-3,0):cy+3, max(cx-3,0):cx+3] = 1
    roads = np.zeros_like(built)
    roads[cy, max(cx-8, 0):cx+8] = 1
    stages = [(roads.copy(), built.copy())]
    for r in range(n_rounds):
        roads, built, _ = expand_once(diff, elev, built, roads,
                                      seed=seed * 100 + r)
        stages.append((roads.copy(), built.copy()))
        print(f"[sandbox] round {r+1}: built px={int(built.sum())} "
              f"road px={int(roads.sum())}")
    return elev, stages


def real_expand(diff, lat, lon, steps=50, seed=0):
    """Real mode: fetch a real town, erode to core, generate expansion."""
    elev = fetch_elevation(lat, lon)
    roads, built = rasterize_osm(fetch_osm(lat, lon), lat, lon)
    s = make_sample(elev, roads, built)
    if s is None:
        raise ValueError("window has too little settlement")
    cond, target = s
    core_built, core_roads = cond[2], cond[3]
    new_roads, new_built, raw = expand_once(
        diff, elev, core_built, core_roads, steps=steps, seed=seed)
    return dict(elev=elev, core_built=core_built, core_roads=core_roads,
                real_roads=roads, real_built=built,
                gen_roads=new_roads, gen_built=new_built, raw=raw,
                target=target)


def save_png(arr01, path):
    Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/default/ckpt.pt")
    ap.add_argument("--mode", choices=["sandbox", "real"], default="sandbox")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff = load_model(args.ckpt, device)
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "sandbox":
        elev, stages = sandbox_grow(diff, args.rounds, seed=args.seed)
        e = (elev - elev.min()) / (np.ptp(elev) + 1e-9)
        save_png(e, f"{args.out}/terrain.png")
        for i, (rd, bl) in enumerate(stages):
            rgb = np.stack([e * 0.6 + rd * 0.4, e * 0.6 + bl * 0.4, e * 0.6], -1)
            save_png(rgb, f"{args.out}/stage_{i}.png")
    else:
        res = real_expand(diff, args.lat, args.lon, seed=args.seed)
        for k in ("gen_roads", "gen_built", "real_roads", "real_built"):
            save_png(res[k].astype(float), f"{args.out}/{k}.png")
    print(f"[sample] wrote outputs to {args.out}/")
