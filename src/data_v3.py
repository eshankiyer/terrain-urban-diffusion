"""v3/v4 data: typed zones, a generated amenity-density channel, and (v4)
a water conditioning channel.

Extends the v2 GHSL temporal pairs in three ways:

1. The diffusion target gains a third channel: log-scaled amenity density
   (shops, schools, clinics, cafes...) rasterized from OSM points. The
   model learns where local centres sit relative to fabric, so at sampling
   time it PROPOSES amenity locations for new growth instead of being
   scored only against existing ones. Known proxy: amenities are observed
   today, growth targets are the historical epoch pair (same limitation
   as the road channel, documented in the paper).

2. Each town also gets a zone-class raster from OSM landuse polygons
   (1 residential, 2 commercial/retail, 3 industrial, 4 institutional;
   0 unlabeled) plus the aux rasters zones.py needs to train the post-hoc
   zone classifier. Labels are patchy outside Europe, which is exactly
   why zoning is a masked post-hoc stage and not a diffusion output yet.

3. (v4) A filled OSM water mask becomes conditioning channel 5 (channels
   0-3 unchanged, so anything trained on the v3 cond layout -- e.g. the
   router's conditioning stats in moe.py -- keeps meaning). Without this,
   a flat lake reads to the model exactly like a flat buildable valley
   floor, since nothing else in cond distinguishes them. Training targets
   are also forced to no-growth on water: real lakes don't get infilled,
   so the diffusion target shouldn't teach the model otherwise.

v5 change, and it matters for anything trained on this file's output:
amenity density is now on an ABSOLUTE scale (see amenity_density and
AMENITY_REF) instead of each window being stretched to its own maximum.
The old per-window normalization forced SOMETHING in every window to
read 1.0, so one isolated suburban school became a full-strength
"amenity core" with a blurred halo -- the direct cause of the
institutional over-prediction bug found at Middleburg Heights, OH.
Caches and classifiers built before this change are incompatible; use a
fresh town_cache directory.

cond is 4 channels for v3 callers (zones.py's post-hoc classifier only
looks at physical features, not water) and 5 for v4 datasets built here;
train.py infers cond_ch from whatever build_dataset_v3 wrote.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

from data import (GRID, window_bbox, lonlat_to_px, fetch_elevation,
                  fetch_osm, rasterize_osm, augment)
from data_v2 import EPOCHS, sample_density, make_sample_v2
from environment import _fetch, _element_point

ZONE_NAMES = {1: "residential", 2: "commercial", 3: "industrial",
              4: "institutional"}
_LANDUSE_TO_ZONE = {"residential": 1, "retail": 2, "commercial": 2,
                    "industrial": 3}
_INSTITUTIONAL = ("school", "hospital", "university", "college", "townhall",
                  "community_centre", "library", "clinic")
_AMENITY_POINTS = ("restaurant", "cafe", "pharmacy", "bank", "marketplace",
                   "school", "kindergarten", "clinic", "doctors", "library",
                   "community_centre", "post_office")

V3_FILTERS = (
    'landuse~"^(residential|retail|commercial|industrial)$"',
    f'amenity~"^({"|".join(_INSTITUTIONAL)})$"',
    f'amenity~"^({"|".join(_AMENITY_POINTS)})$"',
    'shop~"."',
    'natural~"^water$"',
    'waterway~"^riverbank$"',
    'landuse~"^(reservoir|basin)$"',
)

AMENITY_BLUR_PX = 4  # ~60 m smoothing radius for the density channel

# Absolute calibration for the amenity channel. Empirically (through
# this file's _box_blur, whose cumsum window is effectively 8 px wide,
# and log1p), a tight cluster of ~25 POIs peaks at 0.18 -- a genuinely
# dense local centre. That maps to 1.0. A single isolated POI (school
# on a parking lot) lands near 0.07 instead of being stretched to 1.0
# by a per-window max.
AMENITY_REF = 0.18


def fetch_env_v3(lat, lon, session=None):
    """One Overpass query per town: landuse polygons + amenity/shop POIs
    + water polygons (lakes, reservoirs, river banks)."""
    return _fetch(lat, lon, V3_FILTERS, session)


def zone_raster(osm_json, lat, lon):
    """GRID x GRID uint8 class raster from OSM polygons. Later classes are
    drawn last so small institutional sites win over broad landuse."""
    bbox = window_bbox(lat, lon)
    im = Image.new("L", (GRID, GRID), 0)
    dr = ImageDraw.Draw(im)
    elements = osm_json.get("elements", [])

    def draw(el, cls):
        pts = [lonlat_to_px(g["lon"], g["lat"], bbox)
               for g in el.get("geometry", [])]
        if len(pts) >= 3:
            dr.polygon(pts, fill=cls)

    for el in elements:  # landuse first
        if el.get("type") == "way" and "landuse" in el.get("tags", {}):
            cls = _LANDUSE_TO_ZONE.get(el["tags"]["landuse"])
            if cls:
                draw(el, cls)
    for el in elements:  # institutional on top
        tags = el.get("tags", {})
        if (el.get("type") == "way" and tags.get("amenity") in _INSTITUTIONAL
                and "geometry" in el):
            draw(el, 4)
    return np.asarray(im, dtype=np.uint8)


def water_raster(osm_json, lat, lon):
    """GRID x GRID float32 mask in {0, 1}: filled lakes/reservoirs/river
    banks from OSM polygons. This is v4 conditioning channel 5 -- see the
    module docstring for why it exists (a flat lake otherwise looks like a
    flat buildable valley floor to the model)."""
    bbox = window_bbox(lat, lon)
    im = Image.new("L", (GRID, GRID), 0)
    dr = ImageDraw.Draw(im)
    for el in osm_json.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        is_water = (tags.get("natural") == "water"
                    or tags.get("waterway") == "riverbank"
                    or tags.get("landuse") in ("reservoir", "basin"))
        if not is_water:
            continue
        pts = [lonlat_to_px(g["lon"], g["lat"], bbox) for g in el["geometry"]]
        if len(pts) >= 3:
            dr.polygon(pts, fill=1)
    return np.asarray(im, dtype=np.float32)


def _box_blur(a, r):
    """Separable box blur with cumulative sums (no scipy needed here)."""
    if r <= 0:
        return a.astype(np.float64)
    out = a.astype(np.float64)
    for axis in (0, 1):
        pad = np.take(out, [0] * r, axis=axis), out, \
            np.take(out, [-1] * r, axis=axis)
        p = np.concatenate(pad, axis=axis)
        c = np.cumsum(p, axis=axis)
        lo = np.take(c, range(0, out.shape[axis]), axis=axis)
        hi = np.take(c, range(2 * r, 2 * r + out.shape[axis]), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out


def amenity_density(osm_json, lat, lon, blur_px=AMENITY_BLUR_PX,
                    ref=AMENITY_REF):
    """Log-scaled, blurred amenity point density in [0, 1], on an
    ABSOLUTE scale: dens / ref, clipped to 1.

    It used to be dens / dens.max() per window. That normalization
    forced something in every window to hit 1.0, so a single isolated
    school in a car-dependent suburb became the window's "amenity core"
    with a wide blurred halo around it -- a shape of signal the zone
    classifier never saw in its dense-town training data, which it
    resolved by painting half the window institutional (confirmed at
    Middleburg Heights, OH: 697/1099 growth cells). On an absolute
    scale a lone POI reads ~0.05 and only a real cluster saturates, so
    sparse and dense windows are finally on the same axis."""
    bbox = window_bbox(lat, lon)
    counts = np.zeros((GRID, GRID), dtype=np.float64)
    for el in osm_json.get("elements", []):
        tags = el.get("tags", {})
        if not ("shop" in tags or tags.get("amenity") in _AMENITY_POINTS
                or tags.get("amenity") in _INSTITUTIONAL):
            continue
        p = _element_point(el, bbox)
        if p is not None:
            counts[p] += 1.0
    dens = np.log1p(_box_blur(counts, blur_px))
    return np.clip(dens / ref, 0.0, 1.0).astype(np.float32)


def _amenity_selftest():
    """Offline check for the Middleburg Heights failure mode. A lone POI
    must read LOW, a dense cluster HIGH, and the two must not be
    normalized into looking alike (which is what per-window max
    normalization did)."""
    lone = np.zeros((GRID, GRID))
    lone[64, 64] = 1.0
    cluster = np.zeros((GRID, GRID))
    for i in range(5):
        for j in range(5):
            cluster[60 + i, 60 + j] = 1.0

    def scale(counts):
        dens = np.log1p(_box_blur(counts, AMENITY_BLUR_PX))
        return np.clip(dens / AMENITY_REF, 0.0, 1.0)

    lo, hi = scale(lone), scale(cluster)
    assert lo.max() < 0.10, f"lone POI should read low, got {lo.max():.3f}"
    assert hi.max() > 0.90, f"cluster should saturate, got {hi.max():.3f}"
    # the old behaviour, shown for contrast: per-window max stretches the
    # lone POI to exactly 1.0, indistinguishable from a real centre
    dens = np.log1p(_box_blur(lone, AMENITY_BLUR_PX))
    old = dens / dens.max()
    assert old.max() == 1.0
    print(f"amenity scale self-test ok: lone POI {lo.max():.3f}, "
          f"cluster {hi.max():.3f} (old per-window norm gave the lone "
          f"POI {old.max():.1f})")


def make_sample_v3(elev, roads, d0, d1, amen, water=None):
    """v2 sample plus the amenity target channel and, when `water` is
    given (v4), a 5th water conditioning channel. Channels 0-3 of cond are
    unchanged so the router's conditioning stats in moe.py keep meaning.
    Training targets are forced to no-growth on water: a real lake never
    gets infilled, so the diffusion target shouldn't teach otherwise."""
    s = make_sample_v2(elev, roads, d0, d1)
    if s is None:
        return None
    cond, target = s
    if water is not None:
        cond = np.concatenate([cond, water.astype(np.float32)[None]])
        wet = water.astype(bool)
        if wet.any():
            target = target.copy()
            target[0][wet] = -1.0  # no new roads on water
            target[1][wet] = (d0.astype(np.float32) * 2.0 - 1.0)[wet]  # no growth on water
    amen_scaled = (amen.astype(np.float32) * 2.0 - 1.0)[None]
    return cond, np.concatenate([target, amen_scaled]).astype(np.float32)


def build_dataset_v3(towns, out_path, cache_dir="ghsl_cache", verbose=True,
                     workers=3, town_cache_dir="data/town_cache_v3",
                     with_water=False):
    """Diffusion dataset with 3-channel targets, plus per-town aux caches
    (zone raster, amenity map, elev, roads, 2020 density) that zones.py
    trains from. Pass with_water=True (v4) to append the water channel to
    cond and force no-growth targets on water; town_cache_dir must be a
    fresh directory when doing so -- v3 caches have no water channel and
    must not be reused. The same freshness rule applies across the v5
    amenity rescale: caches store the amenity map, so anything cached
    under the old per-window normalization is poison for a v5 build.
    Resumable and parallel like the v2 builder."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests

    os.makedirs(town_cache_dir, exist_ok=True)
    tile_cache = {}
    overpass_sem = threading.Semaphore(1)
    cond_ch = 5 if with_water else 4

    def worker(town):
        name, cc, lat, lon, _region = town
        safe = f"{name}_{cc}".replace(" ", "_").replace("/", "_")
        cache = os.path.join(town_cache_dir, safe + ".npz")
        if os.path.exists(cache):
            d = np.load(cache, allow_pickle=True)
            return list(zip(d["cond"], d["target"])), True
        session = requests.Session()
        elev = fetch_elevation(lat, lon, session)
        with overpass_sem:
            osm = fetch_osm(lat, lon, session, sleep=1.2)
        with overpass_sem:
            env = fetch_env_v3(lat, lon, session)
        roads, _ = rasterize_osm(osm, lat, lon)
        zones = zone_raster(env, lat, lon)
        amen = amenity_density(env, lat, lon)
        water = water_raster(env, lat, lon) if with_water else None
        dens = {ep: sample_density(lat, lon, ep, cache_dir, session,
                                   tile_cache) for ep in EPOCHS}
        raw = []
        for e0, e1 in zip(EPOCHS[:-1], EPOCHS[1:]):
            s = make_sample_v3(elev, roads, dens[e0], dens[e1], amen, water)
            if s is not None:
                raw.append(s)
        save_kwargs = dict(
            cond=np.stack([c for c, _ in raw]) if raw
            else np.zeros((0, cond_ch, GRID, GRID), np.float32),
            target=np.stack([t for _, t in raw]) if raw
            else np.zeros((0, 3, GRID, GRID), np.float32),
            zones=zones, amenity=amen, elev=elev.astype(np.float32),
            roads=roads, dens2020=dens[2020].astype(np.float32))
        if with_water:
            save_kwargs["water"] = water
        np.savez_compressed(cache, **save_kwargs)
        return raw, False

    conds, targets = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, t): t for t in towns}
        for fut in as_completed(futures):
            name = futures[fut][0]
            done += 1
            try:
                raw, cached = fut.result()
            except Exception as err:  # noqa: BLE001 - skip failures
                if verbose:
                    print(f"  {done}/{len(towns)} skip {name}: {err}")
                continue
            for cond, target in raw:
                for k in range(4):
                    for flip in (False, True):
                        c_a, t_a = augment(cond, target, k, flip)
                        conds.append(c_a)
                        targets.append(t_a)
            if verbose:
                tag = "cached" if cached else "fetched"
                print(f"  {done}/{len(towns)} {name} ({tag}): "
                      f"{len(raw)} pairs")
    if not conds:
        raise RuntimeError("no v3 samples built")
    np.savez_compressed(out_path, cond=np.stack(conds),
                        target=np.stack(targets))
    if verbose:
        print(f"saved {len(conds)} samples -> {out_path}")
    return len(conds)


if __name__ == "__main__":
    from towns import TOWNS
    build_dataset_v3(TOWNS, "data/dataset_v3.npz")
