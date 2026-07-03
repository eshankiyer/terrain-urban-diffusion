"""Data pipeline: terrain + OpenStreetMap -> training rasters.

For each town we build a GRID x GRID raster window (default 128 px at 15 m/px,
i.e. ~1.9 km) centred on the town, with channels:

  conditioning (4): elevation (z-scored per window), slope (normalised),
                    core built-up mask, core road raster
  target (2):       ring roads, ring built-up

"Core" and "ring" come from a concentric-erosion proxy for historical growth:
the observed built-up mask is eroded by ERODE_PX pixels to approximate an
earlier stage of the settlement; the difference (the ring) is what the model
learns to generate. This proxy is discussed in the paper's limitations.

Only dependencies: numpy, Pillow, requests (no GDAL/rasterio needed).
Elevation comes from the AWS Terrain Tiles open dataset (terrarium encoding,
no API key). Roads/buildings come from the Overpass API.
"""

import io
import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw

GRID = 128           # raster size in pixels
M_PER_PX = 15.0      # metres per pixel  -> window ~1.92 km
ERODE_PX = 10        # erosion radius for the growth proxy (~150 m)
TILE_Z = 13          # terrain tile zoom (~19 m/px at equator, resampled)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def window_bbox(lat, lon):
    """Return (south, west, north, east) for the GRID window centred on lat/lon."""
    half_m = GRID * M_PER_PX / 2.0
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def lonlat_to_px(lon, lat, bbox):
    """Project lon/lat into pixel coordinates of the window (equirectangular)."""
    s, w, n, e = bbox
    x = (lon - w) / (e - w) * (GRID - 1)
    y = (n - lat) / (n - s) * (GRID - 1)
    return x, y


# ----------------------------------------------------------------------------
# Terrain (AWS Terrain Tiles, terrarium encoding)
# ----------------------------------------------------------------------------

def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _decode_terrarium(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def fetch_elevation(lat, lon, session=None):
    """Return a GRID x GRID float32 elevation array for the window."""
    import requests
    session = session or requests.Session()
    bbox = window_bbox(lat, lon)
    s, w, n, e = bbox
    z = TILE_Z
    x0, y1 = _tile_xy(n, w, z)   # NW tile
    x1, y0 = _tile_xy(s, e, z)   # SE tile
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    tiles = {}
    for tx in xs:
        for ty in ys:
            r = session.get(TERRAIN_URL.format(z=z, x=tx, y=ty), timeout=30)
            r.raise_for_status()
            tiles[(tx, ty)] = _decode_terrarium(Image.open(io.BytesIO(r.content)))
    # mosaic
    mos = np.zeros((len(list(ys)) * 256, len(list(xs)) * 256))
    for i, tx in enumerate(xs):
        for j, ty in enumerate(ys):
            mos[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = tiles[(tx, ty)]
    # sample mosaic at window pixel centres (bilinear via PIL)
    n_tiles_x, n_tiles_y = len(list(xs)), len(list(ys))
    tx_org, ty_org = min(xs), min(ys)

    def merc_px(lon_, lat_):
        nn = 2 ** z
        fx = (lon_ + 180.0) / 360.0 * nn
        fy = (1.0 - math.asinh(math.tan(math.radians(lat_))) / math.pi) / 2.0 * nn
        return (fx - tx_org) * 256.0, (fy - ty_org) * 256.0

    px0, py0 = merc_px(w, n)
    px1, py1 = merc_px(e, s)
    im = Image.fromarray(mos.astype(np.float32), mode="F")
    win = im.transform((GRID, GRID), Image.QUAD,
                       (px0, py0, px0, py1, px1, py1, px1, py0),
                       resample=Image.BILINEAR)
    return np.asarray(win, dtype=np.float32)


def slope_from_elevation(elev, m_per_px=M_PER_PX):
    gy, gx = np.gradient(elev.astype(np.float64), m_per_px)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


# ----------------------------------------------------------------------------
# OpenStreetMap (Overpass)
# ----------------------------------------------------------------------------

QUERY = """
[out:json][timeout:90];
(
  way({s},{w},{n},{e})["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|service|pedestrian)"];
  way({s},{w},{n},{e})["building"];
  way({s},{w},{n},{e})["landuse"~"^(residential|retail|commercial|industrial)$"];
);
out geom;
"""


def fetch_osm(lat, lon, session=None, sleep=1.0, rounds=2):
    """Query Overpass with mirror rotation and a second retry round, so a
    transient timeout doesn't silently drop a town from the dataset."""
    import requests
    session = session or requests.Session()
    s, w, n, e = window_bbox(lat, lon)
    q = QUERY.format(s=s, w=w, n=n, e=e)
    last_err = None
    for attempt in range(rounds):
        for url in OVERPASS_URLS:
            try:
                r = session.post(url, data={"data": q}, timeout=120)
                r.raise_for_status()
                time.sleep(sleep)  # be polite to the public API
                return r.json()
            except Exception as err:  # noqa: BLE001 - retry on any failure
                last_err = err
        if attempt + 1 < rounds:
            time.sleep(8.0)  # back off before the retry round
    raise RuntimeError(f"Overpass failed for {lat},{lon}: {last_err}")


def rasterize_osm(osm_json, lat, lon):
    """Return (roads, built) uint8 arrays of shape GRID x GRID in {0,1}."""
    bbox = window_bbox(lat, lon)
    roads_im = Image.new("L", (GRID, GRID), 0)
    built_im = Image.new("L", (GRID, GRID), 0)
    rd, bd = ImageDraw.Draw(roads_im), ImageDraw.Draw(built_im)
    for el in osm_json.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        pts = [lonlat_to_px(g["lon"], g["lat"], bbox) for g in el["geometry"]]
        tags = el.get("tags", {})
        if "highway" in tags:
            major = tags["highway"] in ("motorway", "trunk", "primary", "secondary")
            rd.line(pts, fill=255, width=2 if major else 1)
        elif "building" in tags and len(pts) >= 3:
            bd.polygon(pts, fill=255)
        elif "landuse" in tags and len(pts) >= 3:
            bd.polygon(pts, fill=120)  # landuse counts at lower weight
    roads = (np.asarray(roads_im) > 0).astype(np.uint8)
    built_raw = np.asarray(built_im).astype(np.float32) / 255.0
    # dilate buildings slightly so the built-up mask is contiguous
    built = binary_dilate((built_raw > 0.4).astype(np.uint8), 1)
    return roads, built


# ----------------------------------------------------------------------------
# Morphology (pure numpy, no scipy needed)
# ----------------------------------------------------------------------------

def _disk_offsets(r):
    off = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                off.append((dy, dx))
    return off


def _morph(mask, r, op):
    if r <= 0:
        return mask.copy()
    pad = np.pad(mask, r, mode="constant", constant_values=0 if op == "d" else 1)
    out = np.zeros_like(mask) if op == "d" else np.ones_like(mask)
    h, w = mask.shape
    for dy, dx in _disk_offsets(r):
        sl = pad[r + dy:r + dy + h, r + dx:r + dx + w]
        out = np.maximum(out, sl) if op == "d" else np.minimum(out, sl)
    return out


def binary_dilate(mask, r):
    return _morph(mask, r, "d")


def binary_erode(mask, r):
    return _morph(mask, r, "e")


# ----------------------------------------------------------------------------
# Sample assembly
# ----------------------------------------------------------------------------

def make_sample(elev, roads, built, erode_px=ERODE_PX):
    """Assemble one training sample. Returns (cond[4,H,W], target[2,H,W]) float32,
    or None if the window has too little settlement to be useful."""
    if built.sum() < 200 or roads.sum() < 150:
        return None
    core_built = binary_erode(built, erode_px)
    if core_built.sum() < 30:  # town too small to erode; use half radius
        core_built = binary_erode(built, erode_px // 2)
        if core_built.sum() < 30:
            return None
    core_zone = binary_dilate(core_built, 2)
    core_roads = roads * core_zone
    ring_built = (built & ~core_built.astype(bool)).astype(np.float32)
    ring_roads = (roads & ~core_roads.astype(bool)).astype(np.float32)

    ez = (elev - elev.mean()) / (elev.std() + 1e-6)
    ez = np.clip(ez, -3, 3) / 3.0
    sl = np.clip(slope_from_elevation(elev) / 30.0, 0, 1)

    cond = np.stack([ez, sl, core_built.astype(np.float32),
                     core_roads.astype(np.float32)]).astype(np.float32)
    target = np.stack([ring_roads, ring_built]).astype(np.float32) * 2.0 - 1.0
    return cond, target


def augment(cond, target, k, flip):
    c = np.rot90(cond, k, axes=(1, 2))
    t = np.rot90(target, k, axes=(1, 2))
    if flip:
        c, t = c[:, :, ::-1], t[:, :, ::-1]
    return np.ascontiguousarray(c), np.ascontiguousarray(t)


def _town_windows(town, jitter, sess, overpass_sem=None):
    """Fetch + rasterize the jittered windows for one town. The jitter rng
    is seeded per town so results don't depend on completion order."""
    name, cc, lat, lon, region = town
    rng = np.random.default_rng(abs(hash((name, cc))) % (2 ** 32))
    centres = [(lat, lon)] + [
        (lat + rng.uniform(-2e-3, 2e-3), lon + rng.uniform(-2e-3, 2e-3))
        for _ in range(jitter)]
    samples = []
    for la, lo in centres:
        elev = fetch_elevation(la, lo, sess)
        if overpass_sem is None:
            osm = fetch_osm(la, lo, sess)
        else:
            with overpass_sem:
                osm = fetch_osm(la, lo, sess, sleep=0.3)
        roads, built = rasterize_osm(osm, la, lo)
        s = make_sample(elev, roads, built)
        if s is not None:
            samples.append(s)
    return samples


def build_dataset(towns, out_path, jitter=3, verbose=True, workers=4,
                  cache_dir="data/town_cache"):
    """Download + rasterize every town, with augmentation. Saves an .npz.

    Resumable: each town's raw windows are cached in `cache_dir` as soon as
    they are fetched, so an interrupted or re-run build skips finished towns
    in milliseconds instead of refetching everything.
    Parallel: towns are fetched by a small thread pool (the work is network
    bound), with Overpass capped at 2 concurrent requests to stay polite."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    overpass_sem = threading.Semaphore(2)
    n_cond = 4  # conditioning channels per sample

    def worker(town):
        name, cc, _, _, _ = town
        safe = f"{name}_{cc}".replace(" ", "_").replace("/", "_")
        cache = os.path.join(cache_dir, safe + ".npz")
        if os.path.exists(cache):
            d = np.load(cache, allow_pickle=True)
            return list(zip(d["cond"], d["target"])), True
        sess = requests.Session()
        sess.headers["User-Agent"] = ("terrain-urban-diffusion research "
                                      "(contact: repo issues)")
        samples = _town_windows(town, jitter, sess, overpass_sem)
        cond = (np.stack([c for c, _ in samples]) if samples
                else np.zeros((0, n_cond, GRID, GRID), np.float32))
        target = (np.stack([t for _, t in samples]) if samples
                  else np.zeros((0, 2, GRID, GRID), np.float32))
        np.savez_compressed(cache, cond=cond, target=target)
        return samples, False

    conds, targets, meta = [], [], []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, t): t for t in towns}
        for fut in as_completed(futures):
            name, cc, _, _, region = futures[fut]
            done += 1
            try:
                samples, cached = fut.result()
            except Exception as err:  # noqa: BLE001 - skip towns that fail
                print(f"[data] SKIP {name}: {err}")
                continue
            for c0, t0 in samples:
                for k in range(4):
                    for flip in (False, True):
                        c, t = augment(c0, t0, k, flip)
                        conds.append(c); targets.append(t)
                        meta.append((name, cc, region))
            if verbose:
                tag = "cached" if cached else "fetched"
                print(f"[data] {done}/{len(towns)} {name}, {cc} ({tag}): "
                      f"total samples {len(conds)}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path,
                        cond=np.stack(conds), target=np.stack(targets),
                        meta=np.array(meta, dtype=object))
    print(f"[data] wrote {len(conds)} samples -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# Synthetic fixtures (offline smoke tests + sandbox mode heightmaps)
# ----------------------------------------------------------------------------

def fractal_heightmap(rng, size=GRID, octaves=5, relief=300.0):
    """Simple diamond-square-ish fractal terrain via upsampled noise."""
    acc = np.zeros((size, size))
    for o in range(octaves):
        n = 2 ** (o + 2)
        layer = rng.standard_normal((n, n))
        im = Image.fromarray(layer.astype(np.float32), mode="F")
        acc += np.asarray(im.resize((size, size), Image.BICUBIC)) / (2 ** o)
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-9)
    return (acc * relief).astype(np.float32)


def synthetic_town(rng, elev):
    """Procedural 'town' that prefers flat land: for smoke tests only."""
    slope = slope_from_elevation(elev)
    flat = slope < np.percentile(slope, 35)
    cy, cx = np.unravel_index(np.argmax(
        binary_dilate(flat.astype(np.uint8), 3) *
        (rng.random(elev.shape) * 0.2 + flat)), elev.shape)
    built = np.zeros_like(elev, dtype=np.uint8)
    roads = np.zeros_like(elev, dtype=np.uint8)
    im_r = Image.new("L", elev.shape[::-1], 0)
    dr = ImageDraw.Draw(im_r)
    for ang in rng.uniform(0, 2 * np.pi, 5):
        x2 = cx + np.cos(ang) * 60; y2 = cy + np.sin(ang) * 60
        dr.line([(cx, cy), (x2, y2)], fill=255, width=1)
    roads = (np.asarray(im_r) > 0).astype(np.uint8)
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    dist = np.hypot(yy - cy, xx - cx)
    p = np.exp(-dist / 25.0) * flat * (rng.random(elev.shape) > 0.3)
    built = (p > 0.1).astype(np.uint8)
    built = binary_dilate(built, 1)
    return roads, built


def build_synthetic_dataset(n_towns, out_path, seed=0):
    rng = np.random.default_rng(seed)
    conds, targets, meta = [], [], []
    while len(conds) < n_towns * 8:
        elev = fractal_heightmap(rng)
        roads, built = synthetic_town(rng, elev)
        s = make_sample(elev, roads, built, erode_px=6)
        if s is None:
            continue
        for k in range(4):
            for flip in (False, True):
                c, t = augment(*s, k, flip)
                conds.append(c); targets.append(t)
                meta.append(("synthetic", "XX", "synthetic"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, cond=np.stack(conds),
                        target=np.stack(targets),
                        meta=np.array(meta, dtype=object))
    print(f"[data] wrote {len(conds)} synthetic samples -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dataset.npz")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="if >0, build a synthetic dataset with this many towns")
    ap.add_argument("--eval", action="store_true", help="build eval-town split")
    args = ap.parse_args()
    if args.synthetic:
        build_synthetic_dataset(args.synthetic, args.out)
    else:
        from towns import TOWNS, EVAL_TOWNS
        build_dataset(EVAL_TOWNS if args.eval else TOWNS, args.out,
                      jitter=0 if args.eval else 3)
