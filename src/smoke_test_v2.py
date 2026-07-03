"""Offline smoke test for the v2 modules (no network, no GHSL download)."""

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


if __name__ == "__main__":
    test_tile_math()
    test_make_sample_v2()
    test_bikelane_graph()
    test_sustainability()
    print("all v2 smoke tests passed")
