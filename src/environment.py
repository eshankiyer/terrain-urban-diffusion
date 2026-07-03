"""Environmental context layers: amenities, greenspace, water, hazard proxies.

Fetched once per town from OSM (Overpass) and derived from the DEM; consumed
by sustainability.scorecard_v2. Design follows the multi-perspective spec:
amenities are REAL fixed destinations (the density-peak service-centre
heuristic is retired), greenspace only counts in functional patches, and the
flood/landslide masks are explicitly terrain PROXIES (HAND + slope), not
hydraulic or geotechnical models.

Dependencies: numpy, PIL, scipy (distance transform), requests (fetch only).
"""

import time

import numpy as np
from PIL import Image, ImageDraw

from data import (GRID, M_PER_PX, window_bbox, lonlat_to_px,
                  slope_from_elevation, OVERPASS_URLS)

# Amenity categories for 15-minute access (spec section 2.1)
AMENITY_CATEGORIES = {
    "food": ('shop~"^(supermarket|convenience|bakery|greengrocer)$"',
             'amenity=marketplace'),
    "education": ('amenity~"^(school|kindergarten|college)$"',),
    "health": ('amenity~"^(pharmacy|clinic|doctors|hospital)$"',),
    "civic": ('amenity~"^(library|community_centre|townhall|post_office)$"',),
    "recreation": ('leisure~"^(sports_centre|pitch|fitness_centre|playground)$"',),
    "transit": ('highway=bus_stop', 'railway~"^(station|tram_stop)$"',
                'public_transport=stop_position'),
    "social": ('amenity~"^(restaurant|cafe|bank|fuel)$"',),
}

GREEN_FILTERS = ('leisure~"^(park|garden|nature_reserve|recreation_ground|dog_park)$"',
                 'landuse~"^(forest|meadow|grass|recreation_ground)$"',
                 'natural~"^(wood|grassland|scrub|heath)$"')
WATER_FILTERS = ('natural~"^(water|wetland)$"',
                 'waterway~"^(river|stream|canal|drain|ditch)$"',
                 'landuse~"^(reservoir|basin)$"')

# Hazard proxy thresholds (spec 2.3; px = 15 m)
HAND_LIMIT_M = 5.0        # flood-prone if height above nearest water < 5 m
WATER_DIST_PX = 7         # ... and within ~105 m of mapped water
SLOPE_PRONE_DEG = 25.0    # landslide-prone
SLOPE_SEVERE_DEG = 35.0   # severe (double penalty)
GREEN_MIN_PATCH_PX = 20   # ~0.45 ha: minimum functional green patch


def _overpass_query(lat, lon, filters):
    s, w, n, e = window_bbox(lat, lon)
    parts = []
    for f in filters:
        if "=" in f and "~" not in f:
            k, v = f.split("=", 1)
            sel = f'["{k}"="{v}"]'
        else:
            k, v = f.split("~", 1)
            sel = f'["{k}"~{v}]'
        for elem in ("node", "way"):
            parts.append(f'{elem}({s},{w},{n},{e}){sel};')
    return "[out:json][timeout:90];(" + "".join(parts) + ");out geom;"


def _fetch(lat, lon, filters, session=None, sleep=1.0, rounds=4):
    import requests
    session = session or requests.Session()
    q = _overpass_query(lat, lon, filters)
    last = None
    for attempt in range(rounds):
        for url in OVERPASS_URLS:
            try:
                r = session.post(url, data={"data": q}, timeout=120)
                if r.status_code == 429:
                    last = f"429 rate-limited by {url}"
                    continue
                r.raise_for_status()
                time.sleep(sleep)
                return r.json()
            except Exception as err:  # noqa: BLE001
                last = err
        if attempt + 1 < rounds:
            time.sleep(15.0 * (attempt + 1))
    raise RuntimeError(f"Overpass failed for {lat},{lon}: {last}")


def _element_point(el, bbox):
    """Representative pixel of a node or way element, or None."""
    if el.get("type") == "node" and "lat" in el:
        x, y = lonlat_to_px(el["lon"], el["lat"], bbox)
    elif "center" in el:
        x, y = lonlat_to_px(el["center"]["lon"], el["center"]["lat"], bbox)
    elif "geometry" in el and el["geometry"]:
        xs, ys = zip(*[lonlat_to_px(g["lon"], g["lat"], bbox)
                       for g in el["geometry"]])
        x, y = float(np.mean(xs)), float(np.mean(ys))
    else:
        return None
    if 0 <= x < GRID and 0 <= y < GRID:
        return int(y), int(x)
    return None


def fetch_amenities(lat, lon, session=None):
    """{category: [(y, x), ...]} pixel locations of real amenities."""
    bbox = window_bbox(lat, lon)
    out = {}
    for cat, filters in AMENITY_CATEGORIES.items():
        pts = []
        for el in _fetch(lat, lon, filters, session).get("elements", []):
            p = _element_point(el, bbox)
            if p is not None:
                pts.append(p)
        out[cat] = pts
    return out


def _draw_geometry(dr, geometry, bbox, line_width):
    pts = [lonlat_to_px(g["lon"], g["lat"], bbox) for g in geometry]
    if len(pts) >= 3 and pts[0] == pts[-1]:
        dr.polygon(pts, fill=255)
    elif len(pts) >= 2:
        dr.line(pts, fill=255, width=line_width)


def _rasterize(osm_json, lat, lon, line_width=1):
    bbox = window_bbox(lat, lon)
    im = Image.new("L", (GRID, GRID), 0)
    dr = ImageDraw.Draw(im)
    for el in osm_json.get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            _draw_geometry(dr, el["geometry"], bbox, line_width)
        elif el.get("type") == "relation":
            # multipolygon lakes/parks: draw outer member ways
            for mem in el.get("members", []):
                if mem.get("role") != "inner" and "geometry" in mem:
                    _draw_geometry(dr, mem["geometry"], bbox, line_width)
    return (np.asarray(im) > 0).astype(np.uint8)


def fetch_green_mask(lat, lon, session=None):
    return _rasterize(_fetch(lat, lon, GREEN_FILTERS, session), lat, lon)


def fetch_water_mask(lat, lon, session=None):
    return _rasterize(_fetch(lat, lon, WATER_FILTERS, session), lat, lon)


# ----------------------------------------------------------------------------
# Derived layers (no network needed)
# ----------------------------------------------------------------------------

def functional_green(green_mask, min_patch_px=GREEN_MIN_PATCH_PX):
    """Keep only connected green patches of at least min_patch_px pixels."""
    from skimage.measure import label
    lab = label(green_mask.astype(bool), connectivity=2)
    out = np.zeros_like(green_mask, dtype=np.uint8)
    for i in range(1, lab.max() + 1):
        comp = lab == i
        if comp.sum() >= min_patch_px:
            out[comp] = 1
    return out


def flood_mask(elev, water_mask):
    """Terrain flood PROXY: HAND < 5 m and within ~105 m of mapped water.
    Not a hydraulic model. Returns all-zero if no water is mapped."""
    from scipy.ndimage import distance_transform_edt
    if water_mask.sum() == 0:
        return np.zeros_like(water_mask, dtype=np.uint8)
    dist_px, idx = distance_transform_edt(~water_mask.astype(bool),
                                          return_indices=True)
    z_water = elev[idx[0], idx[1]]
    hand = np.clip(elev - z_water, 0, None)
    return ((hand < HAND_LIMIT_M) & (dist_px <= WATER_DIST_PX)).astype(np.uint8)


def landslide_masks(elev):
    """(prone, severe) slope-threshold PROXY masks in degrees."""
    slope = slope_from_elevation(elev)
    return ((slope > SLOPE_PRONE_DEG).astype(np.uint8),
            (slope > SLOPE_SEVERE_DEG).astype(np.uint8))


def fetch_environment(lat, lon, elev, session=None):
    """Everything scorecard_v2 needs for one town window."""
    amen = fetch_amenities(lat, lon, session)
    green = fetch_green_mask(lat, lon, session)
    water = fetch_water_mask(lat, lon, session)
    prone, severe = landslide_masks(elev)
    return {
        "amenities": amen,
        "green0": green,
        "green0_functional": functional_green(green),
        "water": water,
        "flood": flood_mask(elev, water),
        "slide_prone": prone,
        "slide_severe": severe,
    }
