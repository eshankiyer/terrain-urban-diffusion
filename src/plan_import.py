"""Bring-your-own-plan: turn an existing plan drawing into model input.

A planner (or player) supplies a plan as a simple image following a
drawing convention, and the model generates growth that continues it:

  black / dark lines .... roads
  red / orange / brown .. built area
  green ................. ignored (kept free)
  white ................. empty land

The image is resampled to the model grid (128 px = 1.92 km at 15 m/px).
Terrain can come from a real location (fetch_elevation), a supplied
heightmap, or procedural relief. Works with any v2/v3 checkpoint since
the conditioning layout is unchanged.

Dependencies: numpy, PIL.
"""

import numpy as np
from PIL import Image

from data import (GRID, binary_dilate, fractal_heightmap,
                  slope_from_elevation)

ROAD_LUMINANCE = 70        # below this = drawn road line
BUILT_SATURATION = 60      # colourful & warm = built area


def load_plan_image(path_or_img):
    """(roads, built) uint8 masks from a plan drawing."""
    img = (path_or_img if isinstance(path_or_img, Image.Image)
           else Image.open(path_or_img))
    img = img.convert("RGB").resize((GRID, GRID), Image.BILINEAR)
    a = np.asarray(img).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = (r * 299 + g * 587 + b * 114) // 1000
    sat = a.max(-1) - a.min(-1)
    roads = (lum < ROAD_LUMINANCE).astype(np.uint8)
    warm = (r > g) & (r >= b)          # red/orange/brown family
    built = ((sat > BUILT_SATURATION) & warm & ~roads.astype(bool))
    return roads, built.astype(np.uint8)


def cond_from_plan(roads, built, elev=None, relief=300.0, seed=0,
                   density=0.5):
    """4-channel conditioning tensor from plan masks. `elev` may be a real
    DEM window; otherwise procedural terrain is generated."""
    if elev is None:
        elev = fractal_heightmap(np.random.default_rng(seed),
                                 relief=relief)
    ez = (elev - elev.mean()) / (elev.std() + 1e-6)
    ez = np.clip(ez, -3, 3) / 3.0
    sl = np.clip(slope_from_elevation(elev) / 30.0, 0, 1)
    dens = built.astype(np.float32) * density
    foot = binary_dilate((dens > 0.03).astype(np.uint8), 2)
    cond = np.stack([ez, sl, dens,
                     (roads * foot).astype(np.float32)]).astype(np.float32)
    return cond, elev


def extend_plan(diff, cond, n=8, steps=50, device="cuda"):
    """Sample n continuations of the plan. Returns model outputs with
    channel semantics matching the checkpoint (2ch: roads, density;
    3ch v3: +proposed amenity density), plus the candidate list that
    sustainability.rank_samples expects."""
    import torch
    c = torch.from_numpy(cond)[None].repeat(n, 1, 1, 1).to(device)
    with torch.no_grad():
        out = diff.sample_ddim(c, steps=steps).cpu().numpy()
    roads0 = (cond[3] > 0.5).astype(np.uint8)
    cands = []
    for i in range(n):
        new_roads = (out[i, 0] > 0.0).astype(np.uint8)
        dens_new = np.clip((out[i, 1] + 1) / 2, 0, 1)
        cands.append((np.maximum(roads0, new_roads), dens_new))
    return out, cands


def demo_sketch():
    """A synthetic hand-drawn-style plan (main street, crossroad, a few
    blocks) so notebooks can demo plan continuation without an upload."""
    img = Image.new("RGB", (GRID, GRID), (255, 255, 255))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    dr.line([(10, 70), (118, 62)], fill=(20, 20, 20), width=2)   # main road
    dr.line([(64, 20), (60, 108)], fill=(20, 20, 20), width=2)   # crossroad
    for cx, cy in [(50, 60), (72, 58), (58, 76), (70, 74)]:
        dr.rectangle([cx - 6, cy - 5, cx + 6, cy + 5],
                     fill=(200, 90, 60))                          # blocks
    dr.rectangle([(84, 40), (104, 56)], fill=(120, 190, 110))     # park
    return img
