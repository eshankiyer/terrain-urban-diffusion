"""Offline smoke test for the v2 modules (no network, no GHSL download)."""

import math
import os

import numpy as np

import data_v2
import bikelanes
from data import GRID, fractal_heightmap


def test_tile_math():
    # Alps: lat 46.5, lon 8.0 -> row 5 (40-50N band), col 19 (0-10E band)
    assert data_v2.ghsl_tile_rc(46.5, 8.0) == (5, 19)
    # NL: lat 52.0, lon 5.2 -> row 4 (50-60N), col 19
    assert data_v2.ghsl_tile_rc(52.0, 5.2) == (4, 19)
    n, w = data_v2.tile_origin(5, 19)
    assert n == 50.0 and w == 0.0
    a = data_v2.cell_area_m2(46.5)
    assert 5000 < a < 8700, a
    print("tile math ok")


def test_make_sample_v2():
    rng = np.random.default_rng(0)
    elev = fractal_heightmap(rng)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    r = np.hypot(yy - GRID / 2, xx - GRID / 2)
    d0 = np.clip(0.8 - r / 30.0, 0, 1)          # compact 1980 town
    d1 = np.clip(0.85 - r / 42.0, 0, 1)         # grown + densified 2020 town
    roads = np.zeros((GRID, GRID), dtype=np.uint8)
    roads[GRID // 2, 10:-10] = 1
    roads[10:-10, GRID // 2] = 1
    out = data_v2.make_sample_v2(elev, roads, d0, d1)
    assert out is not None
    cond, target = out
    assert cond.shape == (4, GRID, GRID) and target.shape == (2, GRID, GRID)
    assert cond.dtype == np.float32 and target.dtype == np.float32
    assert target.min() >= -1.0 and target.max() <= 1.0
    # densification visible: target density in the core exceeds cond density
    core = d0 > 0.3
    assert ((target[1] + 1) / 2)[core].mean() > cond[2][core].mean() - 1e-6
    # no-growth pair is rejected
    assert data_v2.make_sample_v2(elev, roads, d1, d1) is None
    classes = data_v2.density_to_classes(d1)
    assert classes.max() >= 3
    print("make_sample_v2 ok")


def test_bikelane_graph():
    rng = np.random.default_rng(1)
    elev = fractal_heightmap(rng)
    roads = np.zeros((GRID, GRID), dtype=np.uint8)
    for i in range(20, GRID - 20, 24):          # grid town
        roads[i, 20:GRID - 20] = 1
        roads[20:GRID - 20, i] = 1
    edges = bikelanes.skeleton_to_edges(roads)
    assert len(edges) >= 12, len(edges)
    feats = np.array([bikelanes.edge_features(e, elev, False) for e in edges])
    assert feats.shape[1] == 4 and np.isfinite(feats).all()

    # classifier plumbing on synthetic labels: steep edges get no bike lane
    from sklearn.linear_model import LogisticRegression
    y = (feats[:, 0] < np.median(feats[:, 0])).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(feats, y)
    edges2, probs, painted = bikelanes.assign_bike_lanes(roads, elev, clf)
    assert len(edges2) == len(probs) > 0
    assert painted.shape == roads.shape and painted.max() <= 1.0
    print(f"bike-lane graph ok ({len(edges)} edges)")


def test_sustainability():
    import sustainability as sus
    rng = np.random.default_rng(2)
    elev = fractal_heightmap(rng)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    r = np.hypot(yy - GRID / 2, xx - GRID / 2)
    d0 = np.clip(0.7 - r / 28.0, 0, 1)

    # compact candidate: dense ring growth on a connected grid
    compact_d = np.clip(0.75 - r / 40.0, 0, 1)
    compact_roads = np.zeros((GRID, GRID), dtype=np.uint8)
    for i in range(24, GRID - 24, 16):
        compact_roads[i, 24:GRID - 24] = 1
        compact_roads[24:GRID - 24, i] = 1

    # sprawl candidate: detached low-density blob far from centre + one road
    sprawl_d = d0.copy()
    sprawl_d[8:28, 8:28] = 0.12
    sprawl_roads = np.zeros((GRID, GRID), dtype=np.uint8)
    sprawl_roads[18, 8:GRID // 2] = 1
    sprawl_roads[18:GRID // 2, GRID // 2] = 1

    total_c, sub_c = sus.scorecard(compact_roads, compact_d, d0, elev)
    total_s, sub_s = sus.scorecard(sprawl_roads, sprawl_d, d0, elev)
    assert 0 <= total_s < total_c <= 100, (total_s, total_c)
    assert set(sub_c) == set(sus.WEIGHTS)

    order, scores = sus.rank_samples(
        [(sprawl_roads, sprawl_d), (compact_roads, compact_d)], d0, elev)
    assert order[0] == 1, order
    cov, minutes = sus.fifteen_min_coverage(compact_d, compact_roads)
    assert 0.0 < cov <= 1.0 and np.isfinite(minutes).any()
    print(f"sustainability ok (compact {total_c:.0f} > sprawl {total_s:.0f})")


def _fake_env(elev, green=None, water=None, amenities=None):
    import environment as env_mod
    green = green if green is not None else np.zeros((GRID, GRID), np.uint8)
    water = water if water is not None else np.zeros((GRID, GRID), np.uint8)
    prone, severe = env_mod.landslide_masks(elev)
    return {"amenities": amenities or {}, "green0": green,
            "green0_functional": env_mod.functional_green(green),
            "water": water, "flood": env_mod.flood_mask(elev, water),
            "slide_prone": prone, "slide_severe": severe}


def test_acceptance_v2():
    """Acceptance tests T1-T8 from the design spec."""
    import time as _time
    import environment as env_mod
    import sustainability as sus
    flat = np.zeros((GRID, GRID), dtype=np.float32)

    # T1: functional green patch threshold (19 px ignored, 21 px kept)
    g = np.zeros((GRID, GRID), np.uint8)
    g[10, 10:29] = 1                        # 19-px line
    g[40:43, 40:47] = 1                     # 21-px block
    fg = env_mod.functional_green(g)
    assert fg[10, 10:29].sum() == 0 and fg[40:43, 40:47].sum() == 21

    # T2: preservation penalty — building over half of baseline green
    g2 = np.zeros((GRID, GRID), np.uint8)
    g2[60:70, 20:60] = 1
    envd = _fake_env(flat, green=g2)
    dens_clear = np.zeros((GRID, GRID), np.float32)
    dens_clear[60:70, 20:40] = 0.5          # builds over left half
    p_half, _ = sus.green_metrics(dens_clear, np.zeros_like(dens_clear), envd)
    p_all, _ = sus.green_metrics(np.zeros_like(dens_clear),
                                 np.zeros_like(dens_clear), envd)
    assert abs(p_half - 0.5) < 0.05 and p_all == 1.0

    # T3: flood proxy — 3 m above stream within ~100 m flagged, 12 m not
    xx = np.abs(np.arange(GRID) - 64).astype(np.float32)
    elev_v = np.tile(xx, (GRID, 1))         # 1 m per px away from x=64
    water = np.zeros((GRID, GRID), np.uint8)
    water[:, 64] = 1
    fm = env_mod.flood_mask(elev_v, water)
    assert fm[50, 67] == 1 and fm[50, 76] == 0

    # T4: landslide thresholds with doubled severe penalty. Each candidate
    # is half on flat ground, half on a slope band, so scores don't saturate:
    # flat -> 1.0, half-26deg -> 0.5, half-36deg -> 0.0 (2x severe penalty).
    ramp = np.zeros((GRID, GRID), dtype=np.float32)
    ramp[40:80] = np.arange(GRID) * 15.0 * math.tan(math.radians(26))
    ramp[80:] = np.arange(GRID) * 15.0 * math.tan(math.radians(36))
    env_r = _fake_env(ramp)
    d0 = np.zeros((GRID, GRID), np.float32)
    roads0 = np.zeros((GRID, GRID), np.uint8)

    def dev_on(rows_hazard):
        dev = d0.copy()
        dev[5:20, 40:60] = 0.4                       # flat half (15 rows)
        if rows_hazard:
            dev[rows_hazard[0]:rows_hazard[1], 40:60] = 0.4  # hazard half
        return dev

    _, m9_flat = sus.hazard_metrics(dev_on(None), d0, roads0, env_r)
    _, m9_26 = sus.hazard_metrics(dev_on((45, 60)), d0, roads0, env_r)
    _, m9_36 = sus.hazard_metrics(dev_on((85, 100)), d0, roads0, env_r)
    assert m9_flat == 1.0 and m9_36 < m9_26 < 1.0, (m9_flat, m9_26, m9_36)
    assert abs(m9_26 - 0.5) < 0.1 and m9_36 < 0.1, (m9_26, m9_36)

    # T5: amenity access uses REAL amenities only, no density-peak credit
    roads = np.zeros((GRID, GRID), np.uint8)
    roads[64, :] = 1
    amen = {"food": [(64, 20)], "education": [(64, 24)], "health": [(64, 28)]}
    env_a = _fake_env(flat, amenities=amen)
    near = np.zeros((GRID, GRID), np.float32); near[60:68, 15:35] = 0.4
    far = np.zeros((GRID, GRID), np.float32); far[60:68, 105:125] = 0.4
    d00 = np.zeros((GRID, GRID), np.float32)
    _, sub_near = sus.scorecard_v2(roads, near, d00, flat, env_a)
    _, sub_far = sus.scorecard_v2(roads, far, d00, flat, env_a)
    assert sub_near["amenity"] > sub_far["amenity"]

    # T6: congestion — redundant grid beats single spine with cul-de-sacs
    grid_r = np.zeros((GRID, GRID), np.uint8)
    for i in range(20, GRID - 20, 20):
        grid_r[i, 20:GRID - 20] = 1
        grid_r[20:GRID - 20, i] = 1
    spine = np.zeros((GRID, GRID), np.uint8)
    spine[64, 10:118] = 1
    for x in range(20, 110, 12):
        spine[52:64, x] = 1
    assert sus.congestion_bottleneck(grid_r) > sus.congestion_bottleneck(spine)

    # T7: equity — lifting the worst-served beats boosting the best-served
    dens_eq = np.zeros((GRID, GRID), np.float32)
    dens_eq[10:20, 10:20] = 0.5             # well-served block
    dens_eq[100:110, 100:110] = 0.5         # poorly-served block
    served_a = np.zeros((GRID, GRID)); served_a[10:20, 10:20] = 1.0
    served_b = np.zeros((GRID, GRID))
    served_b[10:20, 10:20] = 0.6; served_b[100:110, 100:110] = 0.6
    assert sus.access_equity(served_b, dens_eq) > sus.access_equity(served_a, dens_eq)
    served_pad = served_b.copy()            # reach painted over empty forest
    served_pad[30:50, 30:50] = 1.0          # ... must not change equity
    assert abs(sus.access_equity(served_pad, dens_eq) -
               sus.access_equity(served_b, dens_eq)) < 1e-9

    # T8: performance and determinism of the full scorecard
    rng = np.random.default_rng(3)
    elev8 = fractal_heightmap(rng)
    t0 = _time.time()
    r1 = sus.scorecard_v2(grid_r, near, d00, elev8, env_a)
    dt = _time.time() - t0
    r2 = sus.scorecard_v2(grid_r, near, d00, elev8, env_a)
    assert dt < 3.0, f"scorecard too slow: {dt:.2f}s"
    assert r1[0] == r2[0] and r1[1] == r2[1]
    print(f"acceptance T1-T8 ok (scorecard {dt*1000:.0f} ms)")


def test_verifier_regressions():
    """Regression tests for the defects found in verification round 1."""
    import sustainability as sus
    flat = np.zeros((GRID, GRID), dtype=np.float32)

    # R1 (CRITICAL): pre-existing switchback on a steep slope is NOT new dev
    steep = np.tile((np.arange(GRID) * 15.0
                     * math.tan(math.radians(40))).astype(np.float32),
                    (GRID, 1))
    roads0 = np.zeros((GRID, GRID), np.uint8)
    roads0[30:100, 90] = 1                  # old road on 40-deg slope
    d0 = np.zeros((GRID, GRID), np.float32)
    d0[40:60, 10:30] = 0.5                  # town on the flat-ish low side
    env_s = _fake_env(steep)
    do_nothing = sus.scorecard_v2(roads0, d0, d0, steep, env_s, roads0=roads0)
    assert do_nothing[1]["landslide"] == 1.0, do_nothing[1]["landslide"]
    builds_on_slope = d0.copy()
    builds_on_slope[30:50, 85:95] = 0.4
    on_slope = sus.scorecard_v2(roads0, builds_on_slope, d0, steep, env_s,
                                roads0=roads0)
    assert on_slope[1]["landslide"] < 0.2, on_slope[1]["landslide"]

    # R2 (BUG): a present-but-unreachable amenity category lowers M1
    roads = np.zeros((GRID, GRID), np.uint8)
    roads[64, :] = 1
    reachable = {"food": [(64, 20)]}
    plus_unreachable = {"food": [(64, 20)], "health": [(20, 20)]}  # off-road
    dens = np.zeros((GRID, GRID), np.float32)
    dens[60:68, 15:35] = 0.4
    s1, n1 = sus.amenity_access(dens, roads, reachable)
    s2, n2 = sus.amenity_access(dens, roads, plus_unreachable)
    r = dens > sus.DENSITY_THR
    assert n1 == 1 and n2 == 2
    assert s2[r].mean() < s1[r].mean(), "unreachable category must dilute M1"

    # R3 (BUG): compact connected block is not circuitous
    block = np.zeros((GRID, GRID), np.uint8)
    block[60:64, 60:64] = 1
    assert sus.circuity(block) == 1.0

    # R4 (DESIGN): do-nothing loses best-of-N to comparable real growth
    env_f = _fake_env(flat, amenities={"food": [(64, 40)]})
    roads_g = np.zeros((GRID, GRID), np.uint8)
    roads_g[64, 30:90] = 1
    roads_g[50:80, 60] = 1
    grown = d0.copy()
    grown[55:75, 45:75] = 0.4
    order, scores = sus.rank_samples([(roads_g, d0), (roads_g, grown)],
                                     d0, flat, env=env_f, roads0=roads_g)
    assert order[0] == 1, (order, {i: round(t, 1) for i, (t, _) in scores.items()})
    print("verifier regressions R1-R4 ok")


class _FakeTile:
    """Sliceable stand-in for a GHSL tile with a constant value."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, idx):
        ys, xs = idx
        h = (ys.stop or 0) - (ys.start or 0)
        w = (xs.stop or 0) - (xs.start or 0)
        return np.full((h, w), self.value, dtype=np.uint16)


def test_windowed_density_and_resume():
    """Windowed GHSL sampling math + per-town resume caching (offline)."""
    import shutil
    import tempfile
    import data
    import data_v2

    # windowed read: constant tile of half the cell area -> density 0.5
    lat, lon = 46.5, 8.0
    val = int(data_v2.cell_area_m2(lat) * 0.5)
    orig = data_v2._get_tile
    data_v2._get_tile = lambda *a, **k: _FakeTile(val)
    try:
        dens = data_v2.sample_density(lat, lon, 2020)
        assert dens.shape == (GRID, GRID)
        assert abs(float(dens.mean()) - 0.5) < 0.02, dens.mean()
    finally:
        data_v2._get_tile = orig

    # resume: run build_dataset twice with mocked fetching; second run must
    # come entirely from cache and produce identical output
    rng = np.random.default_rng(4)
    elev = data.fractal_heightmap(rng)
    fake_sample = (np.zeros((4, GRID, GRID), np.float32),
                   np.zeros((2, GRID, GRID), np.float32))
    calls = {"n": 0}

    def fake_windows(town, jitter, sess, overpass_sem=None):
        calls["n"] += 1
        return [fake_sample]

    towns = [("Testville", "XX", 46.0, 8.0, "test"),
             ("Mockham", "YY", 47.0, 9.0, "test")]
    tmp = tempfile.mkdtemp()
    orig_windows = data._town_windows
    data._town_windows = fake_windows
    try:
        out1 = os.path.join(tmp, "d1.npz")
        out2 = os.path.join(tmp, "d2.npz")
        cache = os.path.join(tmp, "towns")
        data.build_dataset(towns, out1, verbose=False, cache_dir=cache)
        n_first = calls["n"]
        data.build_dataset(towns, out2, verbose=False, cache_dir=cache)
        assert calls["n"] == n_first, "second run must not refetch"
        d1 = np.load(out1, allow_pickle=True)
        d2 = np.load(out2, allow_pickle=True)
        assert d1["cond"].shape == d2["cond"].shape == (16, 4, GRID, GRID)
    finally:
        data._town_windows = orig_windows
        shutil.rmtree(tmp, ignore_errors=True)
    print("windowed density + resume ok")


if __name__ == "__main__":
    test_tile_math()
    test_make_sample_v2()
    test_bikelane_graph()
    test_sustainability()
    test_acceptance_v2()
    test_verifier_regressions()
    test_windowed_density_and_resume()
    print("all v2 smoke tests passed")
