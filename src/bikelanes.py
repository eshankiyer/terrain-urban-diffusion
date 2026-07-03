"""Post-hoc bike-infrastructure assignment on road graphs.

Bike-lane tagging in OSM is dense in the flat cycling countries and nearly
absent in the mountain towns the diffusion model is trained on, so bike
lanes are NOT a diffusion output channel. Instead this module:

  1. trains a per-edge classifier on well-tagged European towns
     (features are physical, so the model transfers: grade along the edge,
     road class, edge length, distance to the town centre);
  2. extracts a road graph from any generated road raster (skeleton ->
     junction-to-junction polylines);
  3. assigns a bike-infrastructure probability to every edge.

Dependencies: numpy, scikit-image, scikit-learn, requests (training only).
"""

import math
import time

import numpy as np

from data import (GRID, M_PER_PX, window_bbox, lonlat_to_px,
                  fetch_elevation, OVERPASS_URLS)

# Flat, well-tagged towns for training the edge classifier.
BIKE_TOWNS = [
    ("Houten", 52.028, 5.168), ("Veenendaal", 52.026, 5.554),
    ("Assen", 52.996, 6.562), ("Meppel", 52.696, 6.194),
    ("Winterswijk", 51.973, 6.720), ("Middelburg", 51.498, 3.610),
    ("Frederikssund", 55.839, 12.068), ("Silkeborg", 56.176, 9.549),
    ("Nordhorn", 52.431, 7.070), ("Bocholt", 51.838, 6.615),
    ("Emmen", 52.785, 6.897), ("Barneveld", 52.140, 5.585),
]

MAJOR = ("motorway", "trunk", "primary", "secondary")
CYCLE_QUERY = """
[out:json][timeout:90];
way({s},{w},{n},{e})["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|cycleway)"];
out geom tags;
"""


# ----------------------------------------------------------------------------
# Edge features (shared between OSM training and raster inference)
# ----------------------------------------------------------------------------

def _grade_along(px_pts, elev):
    """Mean absolute grade (rise/run) along a pixel polyline."""
    if len(px_pts) < 2:
        return 0.0
    grades = []
    for (x0, y0), (x1, y1) in zip(px_pts[:-1], px_pts[1:]):
        run = math.hypot(x1 - x0, y1 - y0) * M_PER_PX
        if run < 1.0:
            continue
        z0 = elev[min(int(round(y0)), GRID - 1), min(int(round(x0)), GRID - 1)]
        z1 = elev[min(int(round(y1)), GRID - 1), min(int(round(x1)), GRID - 1)]
        grades.append(abs(z1 - z0) / run)
    return float(np.mean(grades)) if grades else 0.0


def edge_features(px_pts, elev, is_major):
    """Feature vector for one edge: [mean |grade|, major-road flag,
    log length (m), normalised distance of midpoint to window centre]."""
    length = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1)
                 in zip(px_pts[:-1], px_pts[1:])) * M_PER_PX
    mid = px_pts[len(px_pts) // 2]
    half = GRID / 2.0
    d_center = math.hypot(mid[0] - half, mid[1] - half) / half
    return [_grade_along(px_pts, elev), float(is_major),
            math.log1p(length), d_center]


# ----------------------------------------------------------------------------
# Training on OSM cycleway tags
# ----------------------------------------------------------------------------

def _has_bike_infra(tags):
    if tags.get("highway") == "cycleway":
        return True
    for k, v in tags.items():
        if k.startswith("cycleway") and v not in ("no", "none", "separate"):
            return True
    return False


def fetch_cycle_ways(lat, lon, session=None, sleep=1.0):
    import requests
    session = session or requests.Session()
    s, w, n, e = window_bbox(lat, lon)
    q = CYCLE_QUERY.format(s=s, w=w, n=n, e=e)
    last = None
    for url in OVERPASS_URLS:
        try:
            r = session.post(url, data={"data": q}, timeout=120)
            r.raise_for_status()
            time.sleep(sleep)
            return r.json()
        except Exception as err:  # noqa: BLE001
            last = err
    raise RuntimeError(f"Overpass failed for {lat},{lon}: {last}")


def town_edge_data(lat, lon, session=None):
    """(X, y) for one town: one row per OSM way inside the window."""
    elev = fetch_elevation(lat, lon, session)
    osm = fetch_cycle_ways(lat, lon, session)
    bbox = window_bbox(lat, lon)
    X, y = [], []
    for el in osm.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        pts = [lonlat_to_px(g["lon"], g["lat"], bbox) for g in el["geometry"]]
        pts = [(x, yy) for x, yy in pts if 0 <= x < GRID and 0 <= yy < GRID]
        if len(pts) < 2:
            continue
        X.append(edge_features(pts, elev, tags.get("highway") in MAJOR))
        y.append(int(_has_bike_infra(tags)))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def train_classifier(towns=BIKE_TOWNS, verbose=True):
    """Leave-one-town-out evaluation, then fit on everything.
    Returns (fitted classifier, mean AUC)."""
    import requests
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    session = requests.Session()
    per_town = []
    for name, lat, lon in towns:
        try:
            X, y = town_edge_data(lat, lon, session)
        except Exception as err:  # noqa: BLE001
            if verbose:
                print(f"  skip {name}: {err}")
            continue
        if len(y) >= 30 and 0 < y.sum() < len(y):
            per_town.append((name, X, y))
            if verbose:
                print(f"  {name}: {len(y)} ways, {y.mean():.0%} with bike infra")
    if len(per_town) < 3:
        raise RuntimeError("not enough towns with usable cycleway tags")
    aucs = []
    for i, (name, X_te, y_te) in enumerate(per_town):
        X_tr = np.vstack([X for j, (_, X, _) in enumerate(per_town) if j != i])
        y_tr = np.concatenate([y for j, (_, _, y) in enumerate(per_town) if j != i])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        if verbose:
            print(f"  leave-out {name}: AUC {aucs[-1]:.3f}")
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(np.vstack([X for _, X, _ in per_town]),
            np.concatenate([y for _, _, y in per_town]))
    return clf, float(np.mean(aucs))


# ----------------------------------------------------------------------------
# Road raster -> graph
# ----------------------------------------------------------------------------

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_to_edges(road_mask):
    """Junction-to-junction pixel polylines from a binary road raster."""
    from skimage.morphology import skeletonize
    skel = skeletonize(road_mask.astype(bool))
    h, w = skel.shape

    def nbrs(y, x):
        return [(y + dy, x + dx) for dy, dx in _N8
                if 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx]]

    deg = np.zeros_like(skel, dtype=np.int8)
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        deg[y, x] = len(nbrs(y, x))
    nodes = {(y, x) for y, x in zip(ys, xs) if deg[y, x] != 2}

    edges, visited = [], set()
    for ny, nx in nodes:
        for sy, sx in nbrs(ny, nx):
            if ((ny, nx), (sy, sx)) in visited:
                continue
            path = [(ny, nx), (sy, sx)]
            visited.add(((ny, nx), (sy, sx)))
            prev, cur = (ny, nx), (sy, sx)
            while cur not in nodes:
                nxt = [p for p in nbrs(*cur) if p != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            visited.add((path[-1], path[-2]))
            if len(path) >= 3:
                edges.append([(x, y) for y, x in path])  # -> (x, y) order
    return edges


def assign_bike_lanes(road_mask, elev, clf, width_map=None):
    """Probability of warranting bike infrastructure for each edge of a
    (generated) road raster. Returns (edges, probs, painted raster)."""
    edges = skeleton_to_edges(road_mask)
    if not edges:
        return [], np.zeros(0), np.zeros_like(road_mask, dtype=np.float32)
    feats = []
    for pts in edges:
        if width_map is not None:
            mid_x, mid_y = pts[len(pts) // 2]
            is_major = width_map[int(mid_y), int(mid_x)] > 1.5
        else:
            is_major = False
        feats.append(edge_features(pts, elev, is_major))
    probs = clf.predict_proba(np.array(feats))[:, 1]
    painted = np.zeros_like(road_mask, dtype=np.float32)
    for pts, p in zip(edges, probs):
        for x, y in pts:
            painted[int(y), int(x)] = max(painted[int(y), int(x)], p)
    return edges, probs, painted


if __name__ == "__main__":
    clf, auc = train_classifier()
    print(f"mean leave-one-town-out AUC: {auc:.3f}")
