"""Evaluation metrics for generated expansions.

1. Slope compliance: distribution of terrain slope under generated built-up
   pixels vs. under real built-up pixels (KS distance; lower = better).
2. Road connectivity: fraction of generated road pixels in the largest
   connected component that touches the core road network.
3. Built-up contiguity: fraction of generated built pixels adjacent
   (8-neighbourhood) to other built or core pixels.
4. Expansion volume ratio: generated ring area / real ring area.
"""

import numpy as np


def ks_distance(a, b, bins=50, lo=0.0, hi=45.0):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    ca = np.cumsum(ha) / max(ha.sum(), 1)
    cb = np.cumsum(hb) / max(hb.sum(), 1)
    return float(np.abs(ca - cb).max())


def slope_compliance(slope, gen_built, real_built, core_built):
    gen_ring = gen_built.astype(bool) & ~core_built.astype(bool)
    real_ring = real_built.astype(bool) & ~core_built.astype(bool)
    return ks_distance(slope[gen_ring].ravel(), slope[real_ring].ravel())


def _components(mask):
    """Label 8-connected components without scipy."""
    mask = mask.astype(bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    cur = 0
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < h and 0 <= nx < w and
                                    mask[ny, nx] and labels[ny, nx] == 0):
                                labels[ny, nx] = cur
                                stack.append((ny, nx))
    return labels, cur


def road_connectivity(gen_roads, core_roads):
    total = gen_roads.astype(bool) | core_roads.astype(bool)
    labels, n = _components(total)
    if n == 0:
        return 0.0
    core_labels = set(np.unique(labels[core_roads.astype(bool)])) - {0}
    if not core_labels:
        return 0.0
    connected = np.isin(labels, list(core_labels)) & gen_roads.astype(bool)
    denom = max(int(gen_roads.astype(bool).sum()), 1)
    return float(connected.sum()) / denom


def built_contiguity(gen_built, core_built):
    ring = gen_built.astype(bool) & ~core_built.astype(bool)
    if ring.sum() == 0:
        return float("nan")
    context = gen_built.astype(bool) | core_built.astype(bool)
    pad = np.pad(context, 1)
    neigh = np.zeros_like(ring, dtype=np.int32)
    h, w = ring.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neigh += pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return float((neigh[ring] >= 2).mean())


def volume_ratio(gen_built, real_built, core_built):
    g = (gen_built.astype(bool) & ~core_built.astype(bool)).sum()
    r = (real_built.astype(bool) & ~core_built.astype(bool)).sum()
    return float(g) / max(float(r), 1.0)


def evaluate(slope, core_built, core_roads, gen_roads, gen_built,
             real_roads, real_built):
    return {
        "slope_ks": slope_compliance(slope, gen_built, real_built, core_built),
        "road_connectivity": road_connectivity(
            gen_roads & ~core_roads.astype(bool), core_roads),
        "built_contiguity": built_contiguity(gen_built, core_built),
        "volume_ratio": volume_ratio(gen_built, real_built, core_built),
    }


def random_baseline(rng, slope, core_built, core_roads, real_built):
    """Slope-blind random baseline: scatters the same ring area uniformly."""
    ring_area = int((real_built.astype(bool) & ~core_built.astype(bool)).sum())
    flat_idx = rng.choice(slope.size, size=min(ring_area, slope.size),
                          replace=False)
    gen_built = core_built.astype(np.uint8).copy()
    gb = gen_built.ravel()
    gb[flat_idx] = 1
    gen_roads = core_roads.astype(np.uint8)
    return gen_roads, gen_built.reshape(slope.shape)
