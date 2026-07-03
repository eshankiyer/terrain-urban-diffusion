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
MIN_GROWTH_PX = 60     # minimum number of pixels with meaningful growth
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


def load_tile(epoch, r, c, cache_dir="ghsl_cache", session=None):
    """Download (once) and open a GHSL tile. Returns a 2D uint16 array."""
    import requests
    import tifffile
    os.makedirs(cache_dir, exist_ok=True)
    path = _tile_path(cache_dir, epoch, r, c)
    if not os.path.exists(path):
        session = session or requests.Session()
        url = GHSL_URL.format(epoch=epoch, r=r, c=c)
        resp = session.get(url, timeout=300)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
            if not tif_names:
                raise RuntimeError(f"no .tif inside {url}")
            with zf.open(tif_names[0]) as fh, open(path, "wb") as out:
                out.write(fh.read())
    return tifffile.imread(path)


def cell_area_m2(lat):
    """Approximate area of one 3-arcsecond cell at this latitude."""
    step_m = TILE_DEG / TILE_PX * 111_320.0  # ~92.77 m
    return step_m * step_m * max(math.cos(math.radians(lat)), 0.05)


def sample_density(lat, lon, epoch, cache_dir="ghsl_cache", session=None,
                   _tile_cache=None):
    """GRID x GRID built-up *fraction* in [0, 1] for the window around
    (lat, lon) at the given epoch. Mosaics up to 4 tiles if the window
    crosses tile borders."""
    s, w, n, e = window_bbox(lat, lon)
    corners = [(n, w), (n, e), (s, w), (s, e)]
    rcs = sorted({ghsl_tile_rc(la, lo) for la, lo in corners})
    tiles = {}
    for r, c in rcs:
        key = (epoch, r, c)
        if _tile_cache is not None and key in _tile_cache:
            tiles[(r, c)] = _tile_cache[key]
        else:
            arr = load_tile(epoch, r, c, cache_dir, session)
            tiles[(r, c)] = arr
            if _tile_cache is not None:
                _tile_cache[key] = arr

    deg_px = TILE_DEG / TILE_PX

    def read_patch():
        # bounding box in global fractional pixel coords of the row/col grid
        rows = sorted({rc[0] for rc in rcs})
        cols = sorted({rc[1] for rc in rcs})
        north0, west0 = tile_origin(rows[0], cols[0])
        h = TILE_PX * len(rows)
        wpx = TILE_PX * len(cols)
        mosaic = np.zeros((h, wpx), dtype=np.float32)
        for (r, c), arr in tiles.items():
            i = (r - rows[0]) * TILE_PX
            j = (c - cols[0]) * TILE_PX
            a = arr.astype(np.float32)
            a[a == NODATA] = 0.0
            mosaic[i:i + TILE_PX, j:j + TILE_PX] = a
        y0 = (north0 - n) / deg_px
        y1 = (north0 - s) / deg_px
        x0 = (w - west0) / deg_px
        x1 = (e - west0) / deg_px
        im = Image.fromarray(mosaic, mode="F")
        win = im.transform((GRID, GRID), Image.QUAD,
                           (x0, y0, x0, y1, x1, y1, x1, y0),
                           resample=Image.BILINEAR)
        return np.asarray(win, dtype=np.float32)

    dens = read_patch() / cell_area_m2(lat)
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


def build_dataset_v2(towns, out_path, cache_dir="ghsl_cache", verbose=True):
    """Build the v2 dataset: for each town, one sample per consecutive epoch
    pair with observed growth, plus rotation/flip augmentation."""
    import requests
    session = requests.Session()
    tile_cache = {}
    conds, targets, names = [], [], []
    for name, lat, lon in towns:
        try:
            elev = fetch_elevation(lat, lon, session)
            roads, _ = rasterize_osm(fetch_osm(lat, lon, session), lat, lon)
            dens = {ep: sample_density(lat, lon, ep, cache_dir, session,
                                       tile_cache) for ep in EPOCHS}
        except Exception as err:  # noqa: BLE001 - skip towns that fail
            if verbose:
                print(f"  skip {name}: {err}")
            continue
        n_pairs = 0
        for e0, e1 in zip(EPOCHS[:-1], EPOCHS[1:]):
            sample = make_sample_v2(elev, roads, dens[e0], dens[e1])
            if sample is None:
                continue
            cond, target = sample
            for k in range(4):
                for flip in (False, True):
                    c_a, t_a = augment(cond, target, k, flip)
                    conds.append(c_a)
                    targets.append(t_a)
                    names.append(f"{name}_{e0}_{e1}")
            n_pairs += 1
        if verbose:
            print(f"  {name}: {n_pairs} epoch pairs")
    if not conds:
        raise RuntimeError("no v2 samples built")
    np.savez_compressed(out_path, cond=np.stack(conds),
                        target=np.stack(targets), names=np.array(names))
    if verbose:
        print(f"saved {len(conds)} samples -> {out_path}")
    return len(conds)


if __name__ == "__main__":
    from towns import TRAIN_TOWNS
    build_dataset_v2(TRAIN_TOWNS, "dataset_v2.npz")
