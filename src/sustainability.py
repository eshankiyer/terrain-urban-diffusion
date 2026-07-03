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
    # a network too compact to contain a >=15 px pair is local, not circuitous
    return float(np.mean(ratios)) if ratios else 1.0


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


def rank_samples(candidates, d0, elev, n_centers=3, env=None, roads0=None):
    """Rank sampled futures by sustainability. `candidates` is a list of
    (roads_all, density_new) arrays. Returns indices best-first plus the
    scores. If `env` (from environment.fetch_environment) is given, the
    full 11-metric scorecard_v2 is used; otherwise the legacy scorecard.

    Delivery adjustment: all avoidance metrics score 1.0 for a plan that
    builds nothing, so the do-nothing sample would otherwise dominate
    best-of-N. Ranking therefore multiplies the score by a factor of
    0.5 + 0.5 * min(growth / median_growth, 1), which penalises samples
    that deliver far less housing than their batch peers while leaving
    the per-plan scorecard itself untouched."""
    scored = []
    growths = []
    for i, (roads_all, dens) in enumerate(candidates):
        if env is not None:
            total, sub = scorecard_v2(roads_all, dens, d0, elev, env, roads0)
            growths.append(sub.get("_growth_px", 0))
        else:
            total, sub = scorecard(roads_all, dens, d0, elev, n_centers)
        scored.append((total, i, sub))
    positive = [g for g in growths if g > 0]
    if env is not None and positive:
        med = float(np.median(positive))
        if med > 0:
            scored = [(t * (0.5 + 0.5 * min(s.get("_growth_px", 0) / med,
                                            1.0)), i, s)
                      for t, i, s in scored]
    scored.sort(key=lambda t: (-t[0], t[1]))
    order = [i for _, i, _ in scored]
    return order, {i: (t, s) for t, i, s in scored}


# ----------------------------------------------------------------------------
# v2 scorecard: real amenities, greenspace, hazards, congestion, equity.
# Metrics M1-M11 and weights follow the multi-perspective design spec.
# ----------------------------------------------------------------------------

WEIGHTS_V2 = {"amenity": 0.18, "infill": 0.10, "efficiency": 0.08,
              "circuity": 0.06, "earthwork": 0.06, "green_preserve": 0.10,
              "green_access": 0.08, "flood": 0.10, "landslide": 0.08,
              "congestion": 0.08, "equity": 0.08}

GREEN_ACCESS_MIN = 300.0 / WALK_SPEED_M_MIN   # 300 m walk ~ 3.75 min


def _snap_points_to_roads(points, roads, max_r=8):
    return _snap_to_roads(points, roads, max_r)


def amenity_access(density, roads, amenities, walk_limit=WALK_LIMIT_MIN):
    """Per-pixel served fraction over amenity categories PRESENT in the
    window (categories absent from the whole window are excluded so small
    towns are compared on what exists; documented deviation from a fixed
    six-category denominator). A category that exists but cannot be reached
    over the road network contributes ZERO coverage — it stays in the
    denominator. Returns (served_fraction map, n_categories)."""
    present = {c: p for c, p in amenities.items() if p}
    if not present:
        return np.zeros_like(density, dtype=np.float64), 0
    served = np.zeros_like(density, dtype=np.float64)
    for cat, pts in present.items():
        snapped = _snap_points_to_roads(pts, roads)
        if not snapped:
            continue  # unreachable category: zero reach, still counted below
        minutes = walk_time_map(roads.astype(bool), snapped)
        reach = binary_dilate((minutes <= walk_limit).astype(np.uint8),
                              ACCESS_BUFFER_PX)
        served += reach.astype(np.float64)
    return served / len(present), len(present)


def new_development_masks(dens_future, d0, roads_all, roads0=None):
    """(new_roads, new_dev) boolean masks. Pre-existing roads are NOT new
    development: pass `roads0` (the baseline road raster) so mountain
    switchbacks that predate the plan are excluded. Without `roads0` the
    legacy footprint heuristic is used (documented, less accurate)."""
    foot0 = binary_dilate((d0 > DENSITY_THR).astype(np.uint8), 2).astype(bool)
    if roads0 is not None:
        old_roads = binary_dilate(roads0.astype(np.uint8), 1).astype(bool)
        new_roads = roads_all.astype(bool) & ~old_roads
    else:
        new_roads = roads_all.astype(bool) & ~foot0
    new_dev = ((dens_future > DENSITY_THR) & ~foot0) | new_roads
    return new_roads, new_dev


def green_metrics(dens_future, d0, env, new_roads=None):
    """(M6 preservation, M7 access). Preservation is measured against the
    BASELINE functional green: new green elsewhere cannot buy back cleared
    green, and NEW roads clear green just like buildings do. Access is a
    300 m NETWORK walk (not straight-line) to a functional green patch."""
    from environment import functional_green
    g0 = env["green0_functional"].astype(bool)
    cleared = dens_future > DENSITY_THR
    if new_roads is not None:
        cleared = cleared | new_roads
    if g0.sum() == 0:
        preserve = 1.0
    else:
        preserve = float((g0 & ~cleared).sum() / g0.sum())
    remaining = functional_green((g0 & ~cleared).astype(np.uint8))
    built = np.maximum(dens_future, d0) > DENSITY_THR
    if built.sum() == 0 or remaining.sum() == 0:
        return preserve, 0.0
    # network walk: seed Dijkstra at road pixels adjacent to remaining green
    roads = env.get("_roads_all")
    if roads is None:
        from scipy.ndimage import distance_transform_edt
        dist_px = distance_transform_edt(~remaining.astype(bool))
        served = dist_px <= (300.0 / M_PER_PX)
    else:
        near_green = binary_dilate(remaining, 1).astype(bool) & roads.astype(bool)
        seeds = list(zip(*np.nonzero(near_green)))
        if not seeds:
            return preserve, 0.0
        minutes = walk_time_map(roads.astype(bool), seeds)
        served = binary_dilate((minutes <= GREEN_ACCESS_MIN).astype(np.uint8),
                               ACCESS_BUFFER_PX).astype(bool)
    access = float((built & served).sum() / built.sum())
    return preserve, access


def hazard_metrics(dens_future, d0, roads_all, env, roads0=None):
    """(M8 flood avoidance, M9 landslide avoidance) over NEW development
    only. Severe (>35 deg) landslide cells count double via the nested
    masks: severe cells are already in slide_prone, so prone+severe = 2x."""
    _, new_dev = new_development_masks(dens_future, d0, roads_all, roads0)
    n = new_dev.sum()
    if n == 0:
        return 1.0, 1.0
    flood = float(1.0 - (new_dev & env["flood"].astype(bool)).sum() / n)
    prone = (new_dev & env["slide_prone"].astype(bool)).sum()
    severe = (new_dev & env["slide_severe"].astype(bool)).sum()
    landslide = float(1.0 - min((prone + severe) / n, 1.0))
    return flood, landslide


def _gini(values):
    v = np.sort(np.asarray(values, dtype=np.float64))
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def _cluster_endpoints(edges, radius=2):
    """Skeleton junctions are clusters of adjacent pixels, so raw polyline
    endpoints rarely coincide. Union endpoints within a Chebyshev radius so
    edges meeting at the same junction share a graph node."""
    pts = []
    for e in edges:
        pts.extend([e[0], e[-1]])
    pts = list(dict.fromkeys(pts))
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if (abs(pts[i][0] - pts[j][0]) <= radius
                    and abs(pts[i][1] - pts[j][1]) <= radius):
                parent[find(i)] = find(j)
    return {p: find(i) for i, p in enumerate(pts)}


def congestion_bottleneck(roads, seed=0):
    """M10: 1 - Gini of edge betweenness on the road graph. Dendritic
    single-spine networks concentrate load (high Gini, low score); redundant
    grids spread it. Structural proxy, not a traffic forecast."""
    import networkx as nx
    from bikelanes import skeleton_to_edges
    edges = skeleton_to_edges(roads)
    if len(edges) < 4:
        return 0.0
    node_of = _cluster_endpoints(edges)
    g = nx.Graph()
    for pts in edges:
        a, b = node_of[pts[0]], node_of[pts[-1]]
        if a != b:
            g.add_edge(a, b, weight=float(len(pts)))
    if g.number_of_edges() < 3:
        return 0.0
    k = min(64, g.number_of_nodes())
    bc = nx.edge_betweenness_centrality(g, k=k, weight="weight", seed=seed)
    return float(max(0.0, 1.0 - max(0.0, _gini(list(bc.values())))))


def access_equity(served_fraction, density):
    """M11: 1 - population-weighted coefficient of variation of the served
    fraction over residential cells. Spatial access equity ONLY; income,
    tenure, affordability and displacement are human inputs, not scored."""
    res = density > DENSITY_THR
    w = density[res]
    a = served_fraction[res]
    if len(a) == 0 or w.sum() == 0:
        return 0.0
    mean = float(np.average(a, weights=w))
    if mean <= 1e-9:
        return 0.0
    var = float(np.average((a - mean) ** 2, weights=w))
    return float(max(0.0, 1.0 - math.sqrt(var) / mean))


def scorecard_v2(roads_all, density_new, d0, elev, env, roads0=None):
    """0-100 sustainability score with the 11-metric spec. `env` comes from
    environment.fetch_environment (real amenities, green, water, hazards).
    Pass `roads0` (baseline roads) so pre-existing roads are not scored as
    new development in M5/M8/M9 and green metrics."""
    dens_future = np.maximum(density_new, d0)
    new_roads, new_dev = new_development_masks(dens_future, d0, roads_all,
                                               roads0)

    served, n_cats = amenity_access(dens_future, roads_all, env["amenities"])
    res = dens_future > DENSITY_THR
    m1 = float(served[res].mean()) if res.sum() and n_cats else 0.0
    m2 = infill_share(d0, dens_future)
    m3 = min(land_efficiency(d0, dens_future) / 0.3, 1.0)
    m4 = max(0.0, 1.0 - (circuity(roads_all) - 1.0) / 1.5)
    m5 = max(0.0, 1.0 - earthwork_index(new_roads.astype(np.uint8),
                                        elev) / 20.0)
    env = dict(env, _roads_all=roads_all)
    m6, m7 = green_metrics(dens_future, d0, env, new_roads=new_roads)
    m8, m9 = hazard_metrics(dens_future, d0, roads_all, env, roads0)
    m10 = congestion_bottleneck(roads_all)
    m11 = access_equity(served, dens_future)

    sub = {"amenity": m1, "infill": m2, "efficiency": m3, "circuity": m4,
           "earthwork": m5, "green_preserve": m6, "green_access": m7,
           "flood": m8, "landslide": m9, "congestion": m10, "equity": m11}
    total = 100.0 * sum(WEIGHTS_V2[k] * sub[k] for k in WEIGHTS_V2)
    sub["_growth_px"] = int(new_dev.sum())
    return total, sub
