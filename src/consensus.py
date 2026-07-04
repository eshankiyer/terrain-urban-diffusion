"""Multi-sample consensus and uncertainty maps.

One DDIM draw is one arbitrary future. For planning we sample n futures under
identical conditioning, decode the density channel, and report the per-pixel
share of futures in which a pixel gets built ("built in 87% of futures"),
plus a contested band (0.3-0.7 agreement) marking frontiers the model cannot
decide on. This is the cheapest conversion of an image generator into a
planning tool.

Honest caveats: agreement is a statistic of the model's learned distribution,
not a calibrated probability -- the erosion-proxy training targets and finite
n both bias it. The roads channel [0] is currently unreliable and is ignored.
Sampling is decoupled from torch: any callable (cond_batch [b,4,G,G], seed)
-> [b,out_ch,G,G] works, so everything here tests numpy-only; the torch
wrapper lives in make_ddim_sampler and imports torch lazily.
"""

import os

import numpy as np

from data import GRID, M_PER_PX


def _decode(x):
    """Model channels live in [-1,1]; decode to [0,1] density."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def make_ddim_sampler(diff, steps=24):
    """Wrap model.Diffusion into a plain (cond_np, seed) -> np sampler."""
    import torch

    def sampler(cond_np, seed):
        cond = torch.from_numpy(np.ascontiguousarray(cond_np, dtype=np.float32))
        cond = cond.to(diff.device)
        gen = torch.Generator(device=diff.device).manual_seed(int(seed))
        x = diff.sample_ddim(cond, steps=steps, generator=gen)
        return x.cpu().numpy().astype(np.float32)

    return sampler


def sample_futures(sampler, cond_np, n=32, batch=8, seed=0):
    """Draw n futures for one conditioning [4,G,G]; returns [n,out_ch,G,G].

    Batched, one distinct deterministic seed per batch."""
    cond = np.asarray(cond_np, dtype=np.float32)
    if cond.ndim == 3:
        cond = cond[None]
    outs = []
    for k, i in enumerate(range(0, n, batch)):
        b = min(batch, n - i)
        outs.append(np.asarray(sampler(np.repeat(cond, b, axis=0),
                                       int(seed) * 1000003 + k),
                               dtype=np.float32))
    return np.concatenate(outs, axis=0)[:n]


def consensus_maps(futures, d0, thr=0.25):
    """Per-pixel agreement stats across futures [n,out_ch,G,G]."""
    dens = _decode(futures[:, 1])
    free = np.asarray(d0) < 0.03           # not already built
    p_growth = (dens > thr).mean(axis=0).astype(np.float32)
    p_growth[~free] = 0.0
    contested = (p_growth >= 0.3) & (p_growth <= 0.7)
    if futures.shape[1] > 2:
        p_amen = (_decode(futures[:, 2]) > 0.35).mean(axis=0).astype(np.float32)
    else:
        p_amen = np.zeros_like(p_growth)
    return {"p_growth": p_growth,
            "contested": contested,
            "mean_dens": dens.mean(axis=0).astype(np.float32),
            "std_dens": dens.std(axis=0).astype(np.float32),
            "p_amen": p_amen}


def summarize(maps):
    """Scalar digest for logs/tables."""
    p = maps["p_growth"]
    grow = p > 0.5
    return {"growth_area_px": int(grow.sum()),
            "contested_area_px": int(maps["contested"].sum()),
            "mean_agreement": float(p[grow].mean()) if grow.any() else 0.0}


def render_consensus(elev, d0, roads0, maps, title="", path=None):
    """Agreement map in the render.py visual voice: hillshade base, existing
    town grey, p_growth as a sequential overlay, contested frontier outlined."""
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from render import hillshade, DENSITY_THR

    fig, ax = plt.subplots(figsize=(7.6, 7))
    ax.imshow(hillshade(elev), cmap="gray", vmin=0, vmax=1, alpha=0.9)

    def overlay(mask, color, alpha):
        rgba = np.zeros((GRID, GRID, 4))
        rgba[mask.astype(bool)] = list(to_rgb(color)) + [alpha]
        ax.imshow(rgba)

    overlay(np.asarray(d0) > DENSITY_THR, "#8a8a8a", 0.75)
    overlay(np.asarray(roads0).astype(bool), "#222222", 0.95)

    p = np.ma.masked_less_equal(maps["p_growth"], 0.05)
    im = ax.imshow(p, cmap="YlOrRd", vmin=0, vmax=1, alpha=0.85,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("share of futures built", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if maps["contested"].any():
        ax.contour(maps["contested"].astype(float), levels=[0.5],
                   colors="#0b5fa5", linewidths=1.3)

    handles = [Patch(facecolor="#8a8a8a", label="existing town"),
               Patch(facecolor="#f4a259", label="growth agreement"),
               Line2D([], [], color="#0b5fa5",
                      label="contested frontier (30-70%)"),
               Line2D([], [], color="#222222", label="roads")]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)

    bar_px = 500.0 / M_PER_PX
    y0, x0 = GRID - 8, GRID - 10 - bar_px
    ax.plot([x0, x0 + bar_px], [y0, y0], color="black", lw=3)
    ax.text(x0 + bar_px / 2, y0 - 3, "500 m", ha="center", fontsize=7)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, GRID - 1)
    ax.set_ylim(GRID - 1, 0)
    ax.axis("off")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
    return ax


if __name__ == "__main__":
    from scipy.ndimage import uniform_filter

    G = GRID
    yy, xx = np.mgrid[:G, :G].astype(np.float64)
    rng = np.random.default_rng(7)
    elev = 200 + 60 * uniform_filter(rng.normal(size=(G, G)), 21,
                                     mode="nearest")
    d0 = np.clip(1.2 - np.hypot(yy - 64, xx - 64) / 22.0, 0, 1)
    roads0 = np.zeros((G, G), np.float32)
    roads0[64, :] = 1.0
    roads0[:, 64] = 1.0
    cond = np.stack([(elev - elev.mean()) / (elev.std() + 1e-6),
                     np.zeros((G, G)), d0, roads0]).astype(np.float32)
    attractor = np.clip(1.0 - np.hypot(yy - 64, xx - 64) / 48.0, 0, 1)

    def fake_sampler(cond_b, seed):
        assert cond_b.ndim == 4 and cond_b.shape[1] == 4
        r = np.random.default_rng(seed)
        out = np.empty((cond_b.shape[0], 3, G, G), np.float32)
        for i in range(cond_b.shape[0]):
            blob = uniform_filter(r.normal(size=(G, G)), 11, mode="wrap") * 3.5
            dens01 = np.clip(attractor - 0.15 + blob, 0, 1)
            amen01 = np.clip(uniform_filter(r.normal(size=(G, G)), 9,
                                            mode="wrap") * 4.0 + 0.2, 0, 1)
            out[i, 0] = -1.0
            out[i, 1] = dens01 * 2 - 1
            out[i, 2] = amen01 * 2 - 1
        return out

    fut = sample_futures(fake_sampler, cond, n=12, batch=5, seed=3)
    assert fut.shape == (12, 3, G, G) and fut.dtype == np.float32
    assert np.array_equal(fut, sample_futures(fake_sampler, cond,
                                              n=12, batch=5, seed=3))
    assert not np.array_equal(fut, sample_futures(fake_sampler, cond,
                                                  n=12, batch=5, seed=4))
    print("sampling shapes + determinism ok")

    maps = consensus_maps(fut, d0)
    p = maps["p_growth"]
    assert p.shape == (G, G) and 0.0 <= p.min() and p.max() <= 1.0
    assert np.all(p[d0 >= 0.03] == 0.0)
    assert np.all(p[maps["contested"]] > 0.0)
    assert maps["contested"].any() and (p > 0.5).any()
    for k in ("mean_dens", "std_dens", "p_amen"):
        assert maps[k].shape == (G, G) and np.isfinite(maps[k]).all()
    print("consensus maps ok")

    s = summarize(maps)
    assert all(np.isfinite(v) for v in s.values()) and s["growth_area_px"] > 0
    print(f"summarize {s} ok")

    out = "/tmp/consensus_selftest.png"
    render_consensus(elev, d0, roads0, maps,
                     title="consensus self-test (12 fake futures)", path=out)
    assert os.path.getsize(out) > 10 * 1024
    print(f"render {out} ({os.path.getsize(out)//1024} KB) ok")
