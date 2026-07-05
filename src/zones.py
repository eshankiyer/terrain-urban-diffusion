"""Post-hoc typed-zone assignment for generated plans.

OSM landuse labels are patchy outside Europe and dominated by residential,
so zone types are NOT a diffusion output yet (see ROADMAP, v3 typed
zoning). Instead a per-pixel classifier is trained only where labels
exist, on physical features the generator already produces, and then
paints residential / commercial / industrial / institutional / mixed /
farmland classes onto new development. Leave-one-town-out macro-F1 keeps
the transfer claim testable, mirroring bikelanes.py.

Class 5 ("mixed") is a labelling-side addition, not a new OSM tag: OSM has
no first-class mixed-use landuse value at this raster scale (18 m/px),
so a residential parcel next door to a corner shop is simply tagged
"residential" and a live/work block over a retail strip is tagged
"retail". The honest proxy available from the data we already have is
proximity co-occurrence: pixels that are nominally residential but sit
close to commercial/retail activity (and vice versa) behave like mixed-use
frontage in practice, so they are relabelled class 5 before training.
This only touches the boundary band between the two uses; a residential
block with no commercial pixels within range is untouched. relabel_mixed_use
only inspects classes 1 (residential) and 2 (commercial); class 6
(farmland, data_v3.ZONE_NAMES) is a real OSM landuse tag, not a proximity
proxy, so it is never relabelled and never triggers a relabel of its
neighbours.

Per-class abstention (v5): assign_zones and zone_potential accept
class_margins, a dict mapping class id -> minimum classifier margin for
that class specifically. This exists because institutional (class 4)
turned out to have the loosest decision region: on sparse US suburban
windows it absorbed most of the map (Middleburg Heights: 697/1099 growth
cells). The feature-side fix lives in data_v3.amenity_density; the
margin knob is the belt to that suspender, letting a caller demand more
evidence before painting institutional without also gutting classes that
were behaving. Untouched classes keep the uniform min_margin.

Dependencies: numpy, scipy, scikit-learn.
"""

import os

import numpy as np

from data import GRID, slope_from_elevation
from data_v3 import ZONE_NAMES as _ZONE_NAMES_V3, _box_blur

# Class 5 is assigned post-hoc by relabel_mixed_use (proximity co-occurrence
# of residential and commercial/retail), not present in the raw OSM raster.
# Class 6 ("farmland") is inherited as-is from data_v3.ZONE_NAMES: it is a
# real landuse tag, so it needs no post-hoc labelling step here.
ZONE_NAMES = dict(_ZONE_NAMES_V3)
ZONE_NAMES[5] = "mixed"

MAX_PX_PER_CLASS = 400  # per town, to cap residential dominance
N_FEATURES = 6
MIXED_USE_CLASS = 5
MIXED_USE_RADIUS = 3  # Chebyshev (square) radius for co-occurrence check


def _chebyshev_dilate(mask, r):
    """True where `mask` has a True pixel within Chebyshev radius r
    (i.e. any pixel in the (2r+1)x(2r+1) square neighbourhood). Distinct
    from data.binary_dilate, which dilates over a Euclidean disk."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, r, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= padded[r + dy:r + dy + mask.shape[0],
                          r + dx:r + dx + mask.shape[1]]
    return out


def relabel_mixed_use(zones, radius=MIXED_USE_RADIUS):
    """Relabel residential (1) pixels within Chebyshev `radius` of a
    commercial/retail (2) pixel, and vice versa, as class 5 ("mixed").
    Other classes (industrial=3, institutional=4) are left untouched.
    Operates on a copy; returns a new array."""
    out = zones.copy()
    res_mask = zones == 1
    com_mask = zones == 2
    if not res_mask.any() or not com_mask.any():
        return out
    near_com = _chebyshev_dilate(com_mask, radius)
    near_res = _chebyshev_dilate(res_mask, radius)
    out[res_mask & near_com] = MIXED_USE_CLASS
    out[com_mask & near_res] = MIXED_USE_CLASS
    return out


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
    """(X, y) sampled from one town cache written by build_dataset_v3.
    Residential/commercial pixels within MIXED_USE_RADIUS of each other
    are relabelled class 5 ("mixed") before sampling -- see module
    docstring for why proximity co-occurrence is used as the mixed-use
    proxy instead of an OSM tag."""
    d = np.load(town_npz, allow_pickle=True)
    zones = relabel_mixed_use(d["zones"])
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


def predict_with_margin(clf, X):
    """(labels, margin) where margin = top raw score minus runner-up score.
    Works for either HistGradientBoostingClassifier scoring interface:
    decision_function for binary problems returns a 1-D array (no
    runner-up, margin is just |score|), while the multiclass case and
    predict_proba both return a (n_samples, n_classes) score matrix that
    we can rank per-row. Falls back to predict_proba if decision_function
    is unavailable."""
    if len(X) == 0:
        return (np.zeros(0, dtype=clf.classes_.dtype),
                np.zeros(0, dtype=np.float64))
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    else:
        scores = clf.predict_proba(X)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        # binary decision_function: single margin-like score per sample
        labels = clf.classes_[(scores > 0).astype(int)]
        margin = np.abs(scores)
        return labels, margin
    order = np.argsort(scores, axis=1)
    top_idx = order[:, -1]
    runner_idx = order[:, -2] if scores.shape[1] > 1 else order[:, -1]
    rows = np.arange(scores.shape[0])
    top = scores[rows, top_idx]
    runner = scores[rows, runner_idx]
    labels = clf.classes_[top_idx]
    margin = top - runner
    return labels, margin


def _margin_keep(labels, margin, min_margin=0.0, class_margins=None):
    """Bool mask: which predictions clear their margin requirement.
    Every prediction must clear min_margin; a class listed in
    class_margins must additionally clear its own (higher) bar. A
    class_margins entry BELOW min_margin has no effect -- this knob only
    tightens, never loosens."""
    thr = np.full(len(labels), float(min_margin))
    if class_margins:
        for cls, m in class_margins.items():
            sel = labels == cls
            thr[sel] = np.maximum(thr[sel], float(m))
    return margin >= thr


def assign_zones(dens_new, roads_all, elev, amen, clf, d0=None,
                 thr=0.05, min_margin=0.0, min_context=0.0,
                 class_margins=None):
    """Class raster over NEW development pixels (0 elsewhere).

    min_margin / min_context implement abstention: a pixel whose
    classifier margin (predict_with_margin) is below min_margin, OR whose
    local density context (box-blurred dens_new, radius 4, i.e. feature
    f3 already computed in feature_stack) is below min_context, is left
    class 0 ("untyped") instead of being forced into one of the trained
    classes. class_margins raises the margin bar for specific classes
    only (institutional over-prediction was the motivating case; see
    module docstring). Defaults keep the old always-assign behaviour.

    new_dev comes from data.built_mask (thresh + despeckle), the same
    helper render.py uses, so a 1-2 px density blip can't get a zone
    colour painted on the map that the plan renderer would never show."""
    from data import built_mask
    new_dev = built_mask(dens_new, thr=thr, d0=d0)
    out = np.zeros((GRID, GRID), dtype=np.uint8)
    ys, xs = np.nonzero(new_dev)
    if len(ys) == 0:
        return out
    feats = feature_stack(elev, roads_all, dens_new, amen)
    X = feats[ys, xs]
    labels, margin = predict_with_margin(clf, X)
    keep = _margin_keep(labels, margin, min_margin, class_margins)
    if min_context > 0.0:
        context = X[:, 3]  # local_dens from feature_stack
        keep &= context >= min_context
    assigned = np.zeros(len(ys), dtype=np.uint8)
    assigned[keep] = labels[keep].astype(np.uint8)
    out[ys, xs] = assigned
    return out


def zone_potential(d0, roads_now, elev, amen, clf, ring_px=8,
                   exclude=None, min_margin=0.0, class_margins=None):
    """Advisory zoning over land that COULD develop next, before any model
    sample commits growth there. Candidates are the undeveloped ring within
    ring_px (Chebyshev) of the current footprint, minus roads and any
    exclude mask (water, steep slopes, protected land). Each candidate is
    typed with the same features and classifier used for generated growth;
    predictions below their margin bar (uniform min_margin, per-class
    class_margins) abstain to class 0. This answers a different question
    than assign_zones: not "what did the model build and what is it", but
    "if this parcel develops, what use fits its context". Pairing it with
    the generated-density map gives a planner both where expansion is
    likely and what to put there."""
    foot = d0 > 0.05
    ring = _chebyshev_dilate(foot, ring_px) & ~_chebyshev_dilate(foot, 1)
    cand = ring & ~roads_now.astype(bool)
    if exclude is not None:
        cand &= ~np.asarray(exclude, dtype=bool)
    out = np.zeros((GRID, GRID), dtype=np.uint8)
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        return out
    feats = feature_stack(elev, roads_now, d0, amen)
    labels, margin = predict_with_margin(clf, feats[ys, xs])
    keep = _margin_keep(labels, margin, min_margin, class_margins)
    lab = np.zeros(len(ys), dtype=np.uint8)
    lab[keep] = labels[keep].astype(np.uint8)
    out[ys, xs] = lab
    return out


def _self_test():
    """numpy-only sanity checks: relabel_mixed_use behaviour, and the
    per-class margin mask."""
    zones = np.zeros((40, 40), dtype=np.uint8)
    zones[5:15, 5:15] = 1    # residential block A
    zones[5:15, 16:20] = 2   # commercial strip, touching A
    zones[30:36, 30:36] = 1  # isolated residential block B
    zones[5:15, 21:26] = 6   # farmland, touching the same strip

    out = relabel_mixed_use(zones, radius=MIXED_USE_RADIUS)

    # Farmland touching the commercial strip must stay class 6: proximity
    # co-occurrence only applies to the 1<->2 pair, never to class 6.
    assert np.all(out[5:15, 21:26] == 6), \
        "farmland adjacent to commercial must not be relabelled mixed"

    # Block A: pixels near the residential/commercial boundary become 5.
    assert out[9, 14] == MIXED_USE_CLASS, \
        "residential pixel adjacent to commercial should become mixed"
    assert out[9, 17] == MIXED_USE_CLASS, \
        "commercial pixel adjacent to residential should become mixed"
    # Far corner of block A (row 5, col 5) is > radius=3 Chebyshev cells
    # from the commercial strip (nearest commercial col is 16), so it
    # should remain plain residential.
    assert out[5, 5] == 1, \
        "residential pixel far from commercial should stay residential"

    # Isolated block B has no class-2 pixels anywhere nearby -> untouched.
    assert np.all(out[30:36, 30:36] == 1), \
        "isolated residential block should stay class 1 (no mixed pixels)"
    assert not np.any(out == MIXED_USE_CLASS) or \
        np.any(out[5:15, 12:20] == MIXED_USE_CLASS), \
        "mixed-use band should appear only near the residential/commercial edge"

    n_mixed = int(np.sum(out == MIXED_USE_CLASS))
    n_res = int(np.sum(zones == 1))
    assert 0 < n_mixed < n_res, \
        "mixed relabelling should touch some but not all residential pixels"

    print(f"relabel_mixed_use self-test OK: {n_mixed} px relabelled mixed, "
          f"isolated block untouched, far corner stays residential.")

    # --- per-class margin mask ---
    labels = np.array([1, 4, 4, 2, 4], dtype=np.int64)
    margin = np.array([0.3, 0.3, 1.2, 0.1, 0.6])
    # uniform bar only: everything above 0.2 stays
    keep = _margin_keep(labels, margin, min_margin=0.2)
    assert keep.tolist() == [True, True, True, False, True]
    # raise the institutional bar to 1.0: the two weak class-4 hits
    # abstain, the strong one survives, other classes are untouched
    keep = _margin_keep(labels, margin, min_margin=0.2,
                        class_margins={4: 1.0})
    assert keep.tolist() == [True, False, True, False, False]
    # a class_margins entry below min_margin must not loosen the bar
    keep = _margin_keep(labels, margin, min_margin=0.5,
                        class_margins={4: 0.1})
    assert keep.tolist() == [False, False, True, False, True]
    print("per-class margin mask self-test OK: institutional bar raised "
          "without touching other classes, and it never loosens.")


if __name__ == "__main__":
    _self_test()
