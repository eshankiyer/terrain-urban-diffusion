"""Post-hoc typed-zone assignment for generated plans.

OSM landuse labels are patchy outside Europe and dominated by residential,
so zone types are NOT a diffusion output yet (see ROADMAP, v3 typed
zoning). Instead a per-pixel classifier is trained only where labels
exist, on physical features the generator already produces, and then
paints residential / commercial / industrial / institutional classes onto
new development. Leave-one-town-out macro-F1 keeps the transfer claim
testable, mirroring bikelanes.py.

Dependencies: numpy, scipy, scikit-learn.
"""

import os

import numpy as np

from data import GRID, slope_from_elevation
from data_v3 import ZONE_NAMES, _box_blur

MAX_PX_PER_CLASS = 400   # per town, to cap residential dominance
N_FEATURES = 6


def feature_stack(elev, roads, dens, amen):
    """(GRID, GRID, 6) float features shared by training and inference:
    distance to window centre, slope, road proximity, local density,
    amenity density, local road density."""
    from scipy.ndimage import distance_transform_edt
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    half = GRID / 2.0
    d_center = np.hypot(yy - half, xx - half) / half
    slope = np.clip(slope_from_elevation(elev) / 30.0, 0, 1)
    if roads.sum() > 0:
        d_road = np.clip(distance_transform_edt(~roads.astype(bool)) / 20.0,
                         0, 1)
    else:
        d_road = np.ones((GRID, GRID))
    local_dens = _box_blur(dens.astype(np.float64), 4)
    road_dens = _box_blur(roads.astype(np.float64), 6)
    return np.stack([d_center, slope, d_road, local_dens,
                     amen.astype(np.float64), road_dens],
                    axis=-1).astype(np.float32)


def town_zone_data(town_npz, rng):
    """(X, y) sampled from one town cache written by build_dataset_v3."""
    d = np.load(town_npz, allow_pickle=True)
    zones = d["zones"]
    feats = feature_stack(d["elev"], d["roads"], d["dens2020"], d["amenity"])
    X, y = [], []
    for cls in ZONE_NAMES:
        ys, xs = np.nonzero(zones == cls)
        if len(ys) == 0:
            continue
        take = rng.permutation(len(ys))[:MAX_PX_PER_CLASS]
        X.append(feats[ys[take], xs[take]])
        y.append(np.full(len(take), cls, dtype=np.int64))
    if not X:
        return (np.zeros((0, N_FEATURES), np.float32),
                np.zeros(0, np.int64))
    return np.concatenate(X), np.concatenate(y)


def train_zone_classifier(town_cache_dir="data/town_cache_v3", verbose=True,
                          seed=0):
    """Leave-one-town-out evaluation, then fit on everything.
    Returns (classifier, mean macro-F1)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    per_town = []
    for fn in sorted(os.listdir(town_cache_dir)):
        if not fn.endswith(".npz"):
            continue
        X, y = town_zone_data(os.path.join(town_cache_dir, fn), rng)
        if len(y) >= 60 and len(np.unique(y)) >= 2:
            per_town.append((fn, X, y))
    if len(per_town) < 3:
        raise RuntimeError("not enough towns with usable zone labels")
    scores = []
    for i, (fn, X_te, y_te) in enumerate(per_town):
        X_tr = np.concatenate([X for j, (_, X, _) in enumerate(per_town)
                               if j != i])
        y_tr = np.concatenate([y for j, (_, _, y) in enumerate(per_town)
                               if j != i])
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X_tr, y_tr)
        scores.append(f1_score(y_te, clf.predict(X_te), average="macro"))
        if verbose:
            print(f"  leave-out {fn}: macro-F1 {scores[-1]:.3f}")
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(np.concatenate([X for _, X, _ in per_town]),
            np.concatenate([y for _, _, y in per_town]))
    return clf, float(np.mean(scores))


def assign_zones(dens_new, roads_all, elev, amen, clf, d0=None,
                 thr=0.05):
    """Class raster over NEW development pixels (0 elsewhere)."""
    from data import binary_dilate
    new_dev = dens_new > thr
    if d0 is not None:
        foot0 = binary_dilate((d0 > thr).astype(np.uint8), 2).astype(bool)
        new_dev = new_dev & ~foot0
    out = np.zeros((GRID, GRID), dtype=np.uint8)
    ys, xs = np.nonzero(new_dev)
    if len(ys) == 0:
        return out
    feats = feature_stack(elev, roads_all, dens_new, amen)
    out[ys, xs] = clf.predict(feats[ys, xs]).astype(np.uint8)
    return out
