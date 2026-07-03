"""Offline smoke test for the v3 modules (no network)."""

import os
import shutil
import tempfile

import numpy as np

import data_v3
import zones
from data import GRID, window_bbox, fractal_heightmap


def _fake_way(lat, lon, tags, dy, dx, h, w):
    """Synthetic OSM way: a small rectangle offset from the window centre
    by (dy, dx) pixels, h x w pixels in size."""
    s, west, n, east = window_bbox(lat, lon)
    dlat, dlon = (n - s) / GRID, (east - west) / GRID
    lat0 = n - (GRID / 2 + dy) * dlat
    lon0 = west + (GRID / 2 + dx) * dlon
    ring = [(lat0, lon0), (lat0 - h * dlat, lon0), (lat0 - h * dlat,
            lon0 + w * dlon), (lat0, lon0 + w * dlon), (lat0, lon0)]
    return {"type": "way", "tags": tags,
            "geometry": [{"lat": la, "lon": lo} for la, lo in ring]}


def test_zone_raster_and_amenity():
    lat, lon = 46.5, 8.0
    osm = {"elements": [
        _fake_way(lat, lon, {"landuse": "residential"}, -30, -30, 25, 25),
        _fake_way(lat, lon, {"landuse": "industrial"}, 10, 10, 20, 20),
        _fake_way(lat, lon, {"amenity": "school"}, -25, -25, 8, 8),
        {"type": "node", "lat": lat, "lon": lon, "tags": {"shop": "bakery"}},
    ]}
    z = data_v3.zone_raster(osm, lat, lon)
    assert set(np.unique(z)) >= {0, 1, 3, 4}, np.unique(z)
    assert (z == 4).sum() > 0 and (z == 1).sum() > (z == 4).sum()

    amen = data_v3.amenity_density(osm, lat, lon)
    assert amen.shape == (GRID, GRID) and 0 <= amen.min() and amen.max() <= 1
    # density concentrates near the two amenities (shop at centre, school
    # offset), and is zero far away from both
    assert amen[GRID // 2, GRID // 2] > 0.3
    assert amen[GRID - 5, GRID - 5] == 0.0
    print("zone raster + amenity density ok")


def test_make_sample_v3_and_trainer_inference():
    rng = np.random.default_rng(0)
    elev = fractal_heightmap(rng)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    r = np.hypot(yy - GRID / 2, xx - GRID / 2)
    d0 = np.clip(0.8 - r / 30.0, 0, 1)
    d1 = np.clip(0.85 - r / 42.0, 0, 1)
    roads = np.zeros((GRID, GRID), np.uint8)
    roads[GRID // 2, 10:-10] = 1
    amen = np.clip(1.0 - r / 20.0, 0, 1).astype(np.float32)
    s = data_v3.make_sample_v3(elev, roads, d0, d1, amen)
    assert s is not None
    cond, target = s
    assert cond.shape == (4, GRID, GRID) and target.shape == (3, GRID, GRID)
    assert target.min() >= -1.0 and target.max() <= 1.0

    # trainer channel inference: a 3-channel dataset yields UNet out_ch=3
    try:
        import torch
    except ImportError:
        print("make_sample_v3 ok (torch missing; trainer check skipped)")
        return
    from train import ExpansionDataset
    from model import UNet
    tmp = tempfile.mkdtemp()
    try:
        np.savez(os.path.join(tmp, "d.npz"), cond=np.stack([cond] * 2),
                 target=np.stack([target] * 2))
        ds = ExpansionDataset(os.path.join(tmp, "d.npz"))
        cch, och = ds.cond.shape[1], ds.target.shape[1]
        assert (cch, och) == (4, 3)
        net = UNet(cond_ch=cch, out_ch=och)
        with torch.no_grad():
            out = net(torch.randn(1, och, GRID, GRID),
                      ds.cond[:1], torch.tensor([10]))
        assert out.shape == (1, och, GRID, GRID)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("make_sample_v3 + trainer inference ok")


def test_zone_classifier_loop():
    rng = np.random.default_rng(1)
    tmp = tempfile.mkdtemp()
    try:
        # three synthetic towns whose zones follow a learnable rule:
        # industrial far from centre, residential near it
        for i in range(3):
            elev = fractal_heightmap(np.random.default_rng(i))
            roads = np.zeros((GRID, GRID), np.uint8)
            roads[GRID // 2, :] = 1
            yy, xx = np.mgrid[0:GRID, 0:GRID]
            r = np.hypot(yy - GRID / 2, xx - GRID / 2)
            dens = np.clip(0.7 - r / 40.0, 0, 1).astype(np.float32)
            amen = np.clip(0.9 - r / 25.0, 0, 1).astype(np.float32)
            zon = np.zeros((GRID, GRID), np.uint8)
            zon[r < 25] = 1
            zon[(r > 45) & (r < 58)] = 3
            np.savez(os.path.join(tmp, f"town{i}_XX.npz"), zones=zon,
                     amenity=amen, elev=elev.astype(np.float32),
                     roads=roads, dens2020=dens)
        clf, f1 = zones.train_zone_classifier(tmp, verbose=False)
        assert f1 > 0.8, f1

        dens_new = np.clip(0.6 - np.hypot(*np.mgrid[0:GRID, 0:GRID]
                                          - GRID / 2) / 50.0, 0, 1)
        painted = zones.assign_zones(dens_new.astype(np.float32),
                                     np.zeros((GRID, GRID), np.uint8) | 0,
                                     fractal_heightmap(rng),
                                     np.zeros((GRID, GRID), np.float32), clf)
        assert painted.shape == (GRID, GRID)
        assert set(np.unique(painted)) <= {0, 1, 2, 3, 4}
        assert (painted > 0).sum() > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"zone classifier loop ok (synthetic macro-F1 {f1:.2f})")


def test_plan_import():
    import plan_import
    img = plan_import.demo_sketch()
    roads, built = plan_import.load_plan_image(img)
    assert roads.sum() > 50, roads.sum()          # the two drawn roads
    assert built.sum() > 100, built.sum()         # the blocks
    # the green park must be neither road nor built
    assert roads[40:56, 84:104].sum() == 0
    assert built[45:52, 88:100].sum() == 0
    cond, elev = plan_import.cond_from_plan(roads, built, seed=1)
    assert cond.shape == (4, GRID, GRID) and cond.dtype == np.float32
    assert np.isfinite(cond).all() and elev.shape == (GRID, GRID)
    assert cond[3].sum() > 0                      # roads near built = core
    print("plan import ok")


if __name__ == "__main__":
    test_zone_raster_and_amenity()
    test_make_sample_v3_and_trainer_inference()
    test_zone_classifier_loop()
    test_plan_import()
    print("all v3 smoke tests passed")
