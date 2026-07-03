"""Sustainability / 15-minute-town scoring for generated plans.

Everything here is raster-based and training-free: metrics are computed
directly on (roads, density) outputs plus the DEM, and `rank_samples`
turns the trained model into an optimizer by sampling N futures and
keeping the most sustainable ones (best-of-N selection). The same
scorecard also grades real plans for critique mode.

Metrics
  fifteen_min_coverage  share of built-up area within a 15-minute walk
                        (network distance over the road raster) of a
                        service centre. Centres are proxied by density
                        peaks until an amenity channel exists.
  infill_share          share of new growth inside the existing footprint
                        (densification) rather than greenfield sprawl.
  land_efficiency       added density per newly consumed pixel: rewards
                        housing more people on less land.
  circuity              network / straight-line distance between random
                        road-pixel pairs (lower = walkable, connected).
  earthwork_index       mean terrain slope under NEW roads (grading cost
                        and erosion-risk proxy for mountain towns).

Dependencies: numpy only.
"""

import heapq
import math

import numpy as np

from data import GRID, M_PER_PX, binary_dilate, slope_from_elevation

WALK_SPEED_M_MIN = 80.0     # ~4.8 km/h
WALK_LIMIT_MIN = 15.0
ACCESS_BUFFER_PX = 3        # built pixels may be this far from a road
DENSITY_THR = 0.05

_STEPS = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
          (0, -1, 1.0), (0, 1, 1.0),
          (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))]


# ----------------------------------------------------------------------------
# Network walk times
# ----------------------------------------------------------------------------

def walk_time_map(roads, origins, m_per_px=M_PER_PX):
    """Minutes of walking from the nearest origin, along road pixels
    (Dijkstra on the raster). Non-road pixels are unreachable (inf)."""
    h, w = roads.shape
    minutes = np.full((h, w), np.inf, dtype=np.float64)
    pq = []
    for y, x in origins:
        y, x = int(y), int(x)
        if 0 <= y < h and 0 <= x < w and roads[y, x]:
            minutes[y, x] = 0.0
            heapq.heappush(pq, (0.0, y, x))
    while pq:
        t, y, x = heapq.heappop(pq)
        if t > minutes[y, x]:
            continue
        for dy, dx, cost in _STEPS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and roads[ny, nx]:
                nt = t + cost * m_per_px / WALK_SPEED_M_MIN
                if nt < minutes[ny, nx]:
                    minutes[ny, nx] = nt
                    heapq.heappush(pq, (nt, ny, nx))
    return minutes


def centers_from_density(density, n_centers=3, min_sep_px=20):
    """Service-centre proxies: strongest well-separated density peaks."""
    d = density.copy()
    centers = []
    for _ in range(n_centers):
        idx = int(np.argmax(d))
        y, x = divmod(idx, d.shape[1])
        if d[y, x] <= DENSITY_THR:
            break
        centers.append((y, x))
        y0, y1 = max(0, y - min_sep_px), min(d.shape[0], y + min_sep_px)
        x0, x1 = max(0, x - min_sep_px), min(d.shape[1], x + min_sep_px)
        d[y0:y1, x0:x1] = 0.0
    return centers


def _snap_to_roads(centers, roads, max_r=8):
    snapped = []
    ys, xs = np.nonzero(roads)
    if len(ys) == 0:
        return []
    for y, x in centers:
        d2 = (ys - y) ** 2 + (xs - x) ** 2
        i = int(np.argmin(d2))
        if d2[i] <= max_r ** 2:
            snapped.append((int(ys[i]), int(xs[i])))
    return snapped


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def fifteen_min_coverage(density, roads, n_centers=3):
    """Fraction of built-up area within a 15-minute network walk of a
    service centre. Also returns the walk-time map for plotting."""
    built = density > DENSITY_THR
    if built.sum() == 0:
        return 0.0, np.full_like(density, np.inf)
    centers = _snap_to_roads(centers_from_density(density, n_centers), roads)
    if not centers:
        return 0.0, np.full_like(density, np.inf)
    minutes = walk_time_map(roads.astype(bool), centers)
    # a built pixel is served if any road pixel within the access buffer
    # is within the walk limit
    reach = (minutes <= WALK_LIMIT_MIN).astype(np.uint8)
    served = binary_dilate(reach, ACCESS_BUFFER_PX).astype(bool)
    return float((built & served).sum() / built.sum()), minutes


def infill_share(d0, d1, growth_thr=0.05):
    growth = np.clip(d1 - d0, 0, None)
    total = growth[growth > growth_thr].sum()
    if total <= 0:
        return 1.0
    foot0 = binary_dilate((d0 > DENSITY_THR).astype(np.uint8), 2).astype(bool)
    inside = growth[(growth > growth_thr) & foot0].sum()
    return float(inside / total)


def land_efficiency(d0, d1):
    """Added density per newly consumed pixel, in [0, 1]."""
    newly = (d1 > DENSITY_THR) & (d0 <= DENSITY_THR)
    if newly.sum() == 0:
        return 1.0
    return float(np.clip(d1 - d0, 0, None)[newly].mean())


def circuity(roads, n_pairs=60, seed=0):
    """Mean network/euclidean distance ratio between random road pixels.
    1.0 is a perfectly direct network; disconnected pairs are penalised."""
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(roads)
    if len(ys) < 10:
        return 4.0
    ratios = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(ys), 2)
        eu = math.hypot(float(ys[i]) - float(ys[j]), float(xs[i]) - float(xs[j]))
        if eu < 15:
            continue
        minutes = walk_time_map(roads.astype(bool), [(ys[i], xs[i])])
        net_px = minutes[ys[j], xs[j]] * WALK_SPEED_M_MIN / M_PER_PX
        ratios.append(min(net_px / eu, 4.0) if np.isfinite(net_px) else 4.0)
        if len(ratios) >= 20:
            break
    return float(np.mean(ratios)) if ratios else 4.0


def earthwork_index(new_roads, elev):
    """Mean slope (degrees) under new roads; high values mean expensive,
    erosion-prone grading."""
    if new_roads.sum() == 0:
        return 0.0
    slope = slope_from_elevation(elev)
    return float(slope[new_roads.astype(bool)].mean())


# ----------------------------------------------------------------------------
# Scorecard and best-of-N selection
# ----------------------------------------------------------------------------

WEIGHTS = {"coverage": 0.40, "infill": 0.20, "efficiency": 0.15,
           "circuity": 0.15, "earthwork": 0.10}


def scorecard(roads_all, density_new, d0, elev, n_centers=3):
    """0-100 sustainability score plus subscores. `roads_all` is the full
    road raster (existing + generated), `density_new` the generated
    density field, `d0` the pre-existing density."""
    cov, _ = fifteen_min_coverage(np.maximum(density_new, d0), roads_all,
                                  n_centers)
    inf_s = infill_share(d0, np.maximum(density_new, d0))
    eff = land_efficiency(d0, np.maximum(density_new, d0))
    circ = circuity(roads_all)
    foot0 = binary_dilate((d0 > DENSITY_THR).astype(np.uint8), 2)
    new_roads = (roads_all.astype(bool) & ~foot0.astype(bool))
    ew = earthwork_index(new_roads, elev)
    sub = {
        "coverage": cov,                                  # up is good
        "infill": inf_s,                                  # up is good
        "efficiency": min(eff / 0.3, 1.0),                # saturating
        "circuity": max(0.0, 1.0 - (circ - 1.0) / 1.5),   # 1.0 -> best
        "earthwork": max(0.0, 1.0 - ew / 20.0),           # flat -> best
    }
    total = 100.0 * sum(WEIGHTS[k] * v for k, v in sub.items())
    return total, sub


def rank_samples(candidates, d0, elev, n_centers=3):
    """Rank sampled futures by sustainability. `candidates` is a list of
    (roads_all, density_new) arrays. Returns indices best-first plus the
    scores, so callers can show the top-k most 15-minute-compatible
    futures instead of an arbitrary one."""
    scored = []
    for i, (roads_all, dens) in enumerate(candidates):
        total, sub = scorecard(roads_all, dens, d0, elev, n_centers)
        scored.append((total, i, sub))
    scored.sort(reverse=True)
    order = [i for _, i, _ in scored]
    return order, {i: (t, s) for t, i, s in scored}
