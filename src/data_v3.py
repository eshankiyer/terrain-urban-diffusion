"""v3 data: typed zones + a generated amenity-density channel.

Extends the v2 GHSL temporal pairs in two ways:

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

cond stays 4 channels, so the v2-trained model is unaffected; train.py
infers out_ch=3 from this dataset automatically.
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
)

AMENITY_BLUR_PX = 4   # ~60 m smoothing radius for the density channel


def fetch_env_v3(lat, lon, session=None):
    """One Overpass query per town: landuse polygons + amenity/shop POIs."""
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

    for el in elements:                       # landuse first
        if el.get("type") == "way" and "landuse" in el.get("tags", {}):
            cls = _LANDUSE_TO_ZONE.get(el["tags"]["landuse"])
            if cls:
                draw(el, cls)
    for el in elements:                       # institutional on top
        tags = el.get("tags", {})
        if (el.get("type") == "way" and tags.get("amenity") in _INSTITUTIONAL
                and "geometry" in el):
            draw(el, 4)
    return np.asarray(im, dtype=np.uint8)


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


def amenity_density(osm_json, lat, lon, blur_px=AMENITY_BLUR_PX):
    """Log-scaled, blurred amenity point density in [0, 1]."""
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
    m = dens.max()
    return (dens / m if m > 0 else dens).astype(np.float32)


def make_sample_v3(elev, roads, d0, d1, amen):
    """v2 sample plus the amenity channel: target [roads_new, d1, amenity]."""
    s = make_sample_v2(elev, roads, d0, d1)
    if s is None:
        return None
    cond, target = s
    amen_scaled = (amen.astype(np.float32) * 2.0 - 1.0)[None]
    return cond, np.concatenate([target, amen_scaled]).astype(np.float32)


def build_dataset_v3(towns, out_path, cache_dir="ghsl_cache", verbose=True,
                     workers=3, town_cache_dir="data/town_cache_v3"):
    """Diffusion dataset with 3-channel targets, plus per-town aux caches
    (zone raster, amenity map, elev, roads, 2020 density) that zones.py
    trains from. Resumable and parallel like the v2 builder."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests

    os.makedirs(town_cache_dir, exist_ok=True)
    tile_cache = {}
    overpass_sem = threading.Semaphore(1)

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
        dens = {ep: sample_density(lat, lon, ep, cache_dir, session,
                                   tile_cache) for ep in EPOCHS}
        raw = []
        for e0, e1 in zip(EPOCHS[:-1], EPOCHS[1:]):
            s = make_sample_v3(elev, roads, dens[e0], dens[e1], amen)
            if s is not None:
                raw.append(s)
        np.savez_compressed(
            cache,
            cond=np.stack([c for c, _ in raw]) if raw
            else np.zeros((0, 4, GRID, GRID), np.float32),
            target=np.stack([t for _, t in raw]) if raw
            else np.zeros((0, 3, GRID, GRID), np.float32),
            zones=zones, amenity=amen, elev=elev.astype(np.float32),
            roads=roads, dens2020=dens[2020].astype(np.float32))
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
