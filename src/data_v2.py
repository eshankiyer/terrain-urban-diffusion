"""v2 data pipeline: real temporal densification pairs from GHSL.

Replaces the concentric-erosion growth proxy in data.py with observed
built-up change between epochs of GHS-BUILT-S (R2023A, 3 arc-second,
WGS84 tile grid). For each town and each consecutive epoch pair
(t0, t1) with enough observed growth we emit one sample:

  conditioning (4): elevation (z-scored per window), slope (normalised),
                    built density at t0 in [0, 1],
                    roads inside the t0 footprint (proxy: present-day OSM
                    roads restricted to where density was already non-zero
                    at t0 -- OSM has no history for these towns)
  target (2):       roads outside the t0 footprint,
                    built density at t1

Channel counts match data.py exactly, so UNet(cond_ch=4, out_ch=2),
train.py and sample.py run unchanged. Density is continuous in [0, 1]
(built-up fraction of each cell); ordinal classes for evaluation are
derived by binning (density_to_classes).

Extra dependency vs. v1: tifffile (pure python + numpy).
GHSL is free with attribution: Pesaresi et al., GHS-BUILT-S R2023A,
JRC, https://ghsl.jrc.ec.europa.eu/
"""

import io
import math
import os
import zipfile

import numpy as np
from PIL import Image

from data import (GRID, M_PER_PX, window_bbox, fetch_elevation,
                  slope_from_elevation, fetch_osm, rasterize_osm,
                  binary_dilate, augment)

EPOCHS = [1980, 1990, 2000, 2010, 2020]
GHSL_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{epoch}_GLOBE_R2023A_4326_3ss/"
            "V1-0/tiles/GHS_BUILT_S_E{epoch}_GLOBE_R2023A_4326_3ss_V1_0_R{r}_C{c}.zip")
TILE_DEG = 10.0        # each GHSL 4326 tile spans 10 x 10 degrees
TILE_PX = 12000        # at 3 arc-seconds
NODATA = 65535
MIN_GROWTH_PX = 30     # minimum pixels of meaningful growth per epoch pair
                       # (lowered from 60: slow-growing towns still teach
                       # infill, and 36 pairs was too few to train on)
FOOTPRINT_THR = 0.03   # density above this counts as "already built"


# ----------------------------------------------------------------------------
# GHSL tile access
# ----------------------------------------------------------------------------

def ghsl_tile_rc(lat, lon):
    """Row/column of the GHSL 4326 tile containing (lat, lon). R1 starts at 90N,
    C1 at 180W."""
    r = int((90.0 - lat) // TILE_DEG) + 1
    c = int((lon + 180.0) // TILE_DEG) + 1
    return r, c


def tile_origin(r, c):
    """(north, west) corner of tile in degrees."""
    return 90.0 - (r - 1) * TILE_DEG, -180.0 + (c - 1) * TILE_DEG


def _tile_path(cache_dir, epoch, r, c):
    return os.path.join(cache_dir, f"ghsl_E{epoch}_R{r}_C{c}.tif")


_TILE_LOCKS = {}
_TILE_LOCKS_GUARD = None  # created lazily to keep import light


def _tile_lock(key):
    import threading
    global _TILE_LOCKS_GUARD
    if _TILE_LOCKS_GUARD is None:
        _TILE_LOCKS_GUARD = threading.Lock()
    with _TILE_LOCKS_GUARD:
        if key not in _TILE_LOCKS:
            _TILE_LOCKS[key] = threading.Lock()
        return _TILE_LOCKS[key]


def _ensure_tile_file(epoch, r, c, cache_dir, session=None):
    """Download a GHSL tile once (thread-safe). Returns the local path."""
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    path = _tile_path(cache_dir, epoch, r, c)
    with _tile_lock((epoch, r, c)):
        if not os.path.exists(path):
            session = session or requests.Session()
            url = GHSL_URL.format(epoch=epoch, r=r, c=c)
            resp = session.get(url, timeout=300)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
                if not tif_names:
                    raise RuntimeError(f"no .tif inside {url}")
                with zf.open(tif_names[0]) as fh, open(path + ".part", "wb") as out:
                    out.write(fh.read())
            os.replace(path + ".part", path)
    return path


def _open_tile(path):
    """Open a tile for WINDOWED reads (zarr view over the tiled TIFF), so a
    ~20 px town window doesn't materialise a 288 MB array. Falls back to a
    full read if zarr is unavailable."""
    import tifffile
    try:
        import zarr
        return zarr.open(tifffile.imread(path, aszarr=True), mode="r")
    except Exception:  # noqa: BLE001 - zarr missing or non-tiled tif
        return tifffile.imread(path)


def load_tile(epoch, r, c, cache_dir="ghsl_cache", session=None):
    """Download (once) and open a GHSL tile. Returns a sliceable 2D array
    (zarr view or ndarray)."""
    return _open_tile(_ensure_tile_file(epoch, r, c, cache_dir, session))


def cell_area_m2(lat):
    """Approximate area of one 3-arcsecond cell at this latitude."""
    step_m = TILE_DEG / TILE_PX * 111_320.0  # ~92.77 m
    return step_m * step_m * max(math.cos(math.radians(lat)), 0.05)


def _get_tile(epoch, r, c, cache_dir, session, _tile_cache):
    key = (epoch, r, c)
    if _tile_cache is not None and key in _tile_cache:
        return _tile_cache[key]
    arr = load_tile(epoch, r, c, cache_dir, session)
    # Only cache lazy zarr views. When _open_tile falls back to a full read,
    # the array is ~288 MB; caching dozens of those OOM-kills a Colab VM.
    if _tile_cache is not None and not isinstance(arr, np.ndarray):
        _tile_cache[key] = arr
    return arr


def sample_density(lat, lon, epoch, cache_dir="ghsl_cache", session=None,
                   _tile_cache=None):
    """GRID x GRID built-up *fraction* in [0, 1] for the window around
    (lat, lon) at the given epoch. Reads only the ~20 px patch the window
    needs (windowed zarr slice), mosaicking across tile borders if hit."""
    s, w, n, e = window_bbox(lat, lon)
    deg_px = TILE_DEG / TILE_PX
    # global fractional pixel coords: x east from 180W, y south from 90N
    gx0, gx1 = (w + 180.0) / deg_px, (e + 180.0) / deg_px
    gy0, gy1 = (90.0 - n) / deg_px, (90.0 - s) / deg_px
    ix0, iy0 = int(gx0) - 2, int(gy0) - 2
    ix1, iy1 = int(gx1) + 3, int(gy1) + 3
    patch = np.zeros((iy1 - iy0, ix1 - ix0), dtype=np.float32)
    r0, r1 = iy0 // TILE_PX + 1, (iy1 - 1) // TILE_PX + 1
    c0, c1 = ix0 // TILE_PX + 1, (ix1 - 1) // TILE_PX + 1
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            tx0, ty0 = (c - 1) * TILE_PX, (r - 1) * TILE_PX
            sx0, sx1 = max(ix0, tx0), min(ix1, tx0 + TILE_PX)
            sy0, sy1 = max(iy0, ty0), min(iy1, ty0 + TILE_PX)
            if sx0 >= sx1 or sy0 >= sy1:
                continue
            arr = _get_tile(epoch, r, c, cache_dir, session, _tile_cache)
            sub = np.asarray(arr[sy0 - ty0:sy1 - ty0,
                                 sx0 - tx0:sx1 - tx0]).astype(np.float32)
            sub[sub == NODATA] = 0.0
            patch[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = sub
    im = Image.fromarray(patch, mode="F")
    win = im.transform((GRID, GRID), Image.QUAD,
                       (gx0 - ix0, gy0 - iy0, gx0 - ix0, gy1 - iy0,
                        gx1 - ix0, gy1 - iy0, gx1 - ix0, gy0 - iy0),
                       resample=Image.BILINEAR)
    dens = np.asarray(win, dtype=np.float32) / cell_area_m2(lat)
    return np.clip(dens, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Sample assembly
# ----------------------------------------------------------------------------

def density_to_classes(d, bins=(0.02, 0.10, 0.25, 0.50)):
    """Ordinal density classes 0..len(bins) for evaluation/figures."""
    return np.digitize(d, bins).astype(np.uint8)


def make_sample_v2(elev, roads, d0, d1):
    """One training sample from a real epoch pair. Returns (cond, target)
    float32 arrays or None if there is too little observed growth."""
    growth = (d1 - d0) > 0.05
    if growth.sum() < MIN_GROWTH_PX or d0.max() < FOOTPRINT_THR:
        return None
    foot0 = binary_dilate((d0 > FOOTPRINT_THR).astype(np.uint8), 2)
    roads_core = (roads * foot0).astype(np.float32)
    roads_new = (roads * (1 - foot0)).astype(np.float32)

    ez = (elev - elev.mean()) / (elev.std() + 1e-6)
    ez = np.clip(ez, -3, 3) / 3.0
    sl = np.clip(slope_from_elevation(elev) / 30.0, 0, 1)

    cond = np.stack([ez, sl, d0.astype(np.float32), roads_core]).astype(np.float32)
    target = np.stack([roads_new, d1.astype(np.float32)]).astype(np.float32) * 2.0 - 1.0
    return cond, target


def build_dataset_v2(towns, out_path, cache_dir="ghsl_cache", verbose=True,
                     workers=3, town_cache_dir="data/town_cache_v2"):
    """Build the v2 dataset: for each town, one sample per consecutive epoch
    pair with observed growth, plus rotation/flip augmentation.

    Towns are (name, country, lat, lon, region) tuples as in towns.py.
    Resumable via per-town caches in `town_cache_dir`; parallel across a
    small thread pool (Overpass capped at 2 concurrent, GHSL tile downloads
    deduplicated by per-tile locks)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests

    os.makedirs(town_cache_dir, exist_ok=True)
    tile_cache = {}
    overpass_sem = threading.Semaphore(1)  # strictly serialise Overpass

    def worker(town):
        name, cc, lat, lon, _region = town
        safe = f"{name}_{cc}".replace(" ", "_").replace("/", "_")
        cache = os.path.join(town_cache_dir, safe + ".npz")
        if os.path.exists(cache):
            d = np.load(cache, allow_pickle=True)
            return list(zip(d["cond"], d["target"])), list(d["pair"]), True
        session = requests.Session()
        elev = fetch_elevation(lat, lon, session)
        with overpass_sem:
            osm = fetch_osm(lat, lon, session, sleep=1.2)
        roads, _ = rasterize_osm(osm, lat, lon)
        dens = {ep: sample_density(lat, lon, ep, cache_dir, session,
                                   tile_cache) for ep in EPOCHS}
        raw, pairs = [], []
        for e0, e1 in zip(EPOCHS[:-1], EPOCHS[1:]):
            sample = make_sample_v2(elev, roads, dens[e0], dens[e1])
            if sample is not None:
                raw.append(sample)
                pairs.append(f"{name}_{e0}_{e1}")
        np.savez_compressed(
            cache,
            cond=np.stack([c for c, _ in raw]) if raw
            else np.zeros((0, 4, GRID, GRID), np.float32),
            target=np.stack([t for _, t in raw]) if raw
            else np.zeros((0, 2, GRID, GRID), np.float32),
            pair=np.array(pairs))
        return raw, pairs, False

    conds, targets, names = [], [], []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, t): t for t in towns}
        for fut in as_completed(futures):
            town_name = futures[fut][0]
            done += 1
            try:
                raw, pairs, cached = fut.result()
            except Exception as err:  # noqa: BLE001 - skip towns that fail
                if verbose:
                    print(f"  {done}/{len(towns)} skip {town_name}: {err}")
                continue
            for (cond, target), pair in zip(raw, pairs):
                for k in range(4):
                    for flip in (False, True):
                        c_a, t_a = augment(cond, target, k, flip)
                        conds.append(c_a)
                        targets.append(t_a)
                        names.append(pair)
            if verbose:
                tag = "cached" if cached else "fetched"
                print(f"  {done}/{len(towns)} {town_name} ({tag}): "
                      f"{len(pairs)} epoch pairs")
    if not conds:
        raise RuntimeError("no v2 samples built")
    np.savez_compressed(out_path, cond=np.stack(conds),
                        target=np.stack(targets), names=np.array(names))
    if verbose:
        print(f"saved {len(conds)} samples -> {out_path}")
    return len(conds)


if __name__ == "__main__":
    from towns import TOWNS
    build_dataset_v2(TOWNS, "data/dataset_v2.npz")
