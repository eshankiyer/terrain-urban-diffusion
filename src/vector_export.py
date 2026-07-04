"""GeoJSON export of generated plans for GIS interop (QGIS/ArcGIS).

Rasters are only useful to us; a working planner needs vectors in WGS84.
This module converts generated road / zone / density rasters into a single
GeoJSON FeatureCollection using the same window math as data.py: a GRID x
GRID window at M_PER_PX metres per pixel centred on (lat, lon), pixel
(row 0, col 0) at the NW corner, equirectangular degree conversion
(dLat = HALF_M/111320, dLon = HALF_M/(111320*cos(lat))).

Design:
  * roads: skeletonize -> junction-to-junction polylines (reuses
    bikelanes.skeleton_to_edges) plus a second pass recovering isolated
    cycles that extractor misses (it only walks from degree!=2 nodes);
    Douglas-Peucker simplify in pixel units.
  * zones/density: connected components -> unary_union of pixel squares.
    Slower than find_contours but yields valid polygons with holes
    handled for free; boundaries are pixel staircases (documented, not a
    bug -- simplifying them can create invalid or overlapping rings).

Honest caveats: equirectangular math is fine at ~1 km half-size but drifts
near the poles; coordinates are pixel corners/centres of a 15 m raster, so
positional accuracy is ~15 m at best -- these are sketches, not surveys.
Pixel-coord convention: polylines are lists of (row, col); shapely
polygons use x=col, y=row.

Dependencies: numpy, scikit-image, shapely.
"""

import json
import math

import numpy as np

from bikelanes import skeleton_to_edges
from data import GRID, M_PER_PX

ZONE_NAMES = {1: "residential", 2: "commercial", 3: "industrial",
              4: "institutional"}  # mirrors data_v3.ZONE_NAMES
_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def px_to_lonlat(lat, lon):
    """(row, col) -> (lon_deg, lat_deg) mapper for a window at (lat, lon)."""
    half_m = GRID * M_PER_PX / 2.0
    dlat = half_m / 111320.0
    dlon = half_m / (111320.0 * math.cos(math.radians(lat)))
    lat0, lon0 = lat + dlat, lon - dlon  # NW corner = pixel (0, 0)
    return lambda row, col: (lon0 + col * (2.0 * dlon / GRID),
                             lat0 - row * (2.0 * dlat / GRID))


def roads_to_lines(roads_raster, min_len_px=4, simplify_tol_px=1.2):
    """Binary road raster -> list of polylines [(row, col), ...]."""
    from shapely.geometry import LineString
    from skimage.morphology import skeletonize
    mask = np.asarray(roads_raster) > 0
    if not mask.any():
        return []
    paths = [[(y, x) for x, y in pts] for pts in skeleton_to_edges(mask)]
    # recover isolated cycles (every pixel degree 2 -> no start node above)
    skel = skeletonize(mask)
    rem = {(y, x) for y, x in zip(*np.nonzero(skel))}
    rem -= {p for pts in paths for p in pts}
    h, w = skel.shape
    while rem:
        start = rem.pop()
        path, prev, cur = [start], None, start
        while True:
            nxt = [(cur[0] + dy, cur[1] + dx) for dy, dx in _N8
                   if (cur[0] + dy, cur[1] + dx) in rem]
            nxt = [p for p in nxt if p != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            rem.discard(cur)
            path.append(cur)
        if len(path) >= 3:
            path.append(start)
            paths.append(path)
    out = []
    for pts in paths:
        ls = LineString([(c, r) for r, c in pts]).simplify(simplify_tol_px)
        if ls.length >= min_len_px:
            out.append([(y, x) for x, y in ls.coords])
    return out


def _mask_to_polys(mask, min_area_px):
    from shapely.geometry import box
    from shapely.ops import unary_union
    from skimage.measure import label
    lab = label(mask, connectivity=1)
    polys = []
    for i in range(1, lab.max() + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_area_px:
            continue
        u = unary_union([box(x, y, x + 1.0, y + 1.0)
                         for y, x in zip(ys, xs)])
        for g in getattr(u, "geoms", [u]):
            if g.area >= min_area_px:
                polys.append(g)
    return polys


def zones_to_polygons(zones_raster, min_area_px=6):
    """Class raster -> [(class_id, shapely Polygon in px coords), ...]."""
    z = np.asarray(zones_raster)
    out = []
    for cls in sorted(ZONE_NAMES):
        for g in _mask_to_polys(z == cls, min_area_px):
            out.append((cls, g))
    return out


def density_to_polygons(dens01, thr=0.25, min_area_px=6):
    """Density [0,1] raster -> [("growth", Polygon), ...] above threshold."""
    return [("growth", g)
            for g in _mask_to_polys(np.asarray(dens01) >= thr, min_area_px)]


def _ring_lonlat(coords, to_ll):
    return [list(to_ll(y, x)) for x, y in coords]


def export_geojson(path, lat, lon, roads=None, zones=None, dens=None,
                   amen_points=None, dens_thr=0.25):
    """Write one WGS84 FeatureCollection from raster layers; returns dict."""
    to_ll = px_to_lonlat(lat, lon)
    feats = []
    if roads is not None:
        for pts in roads_to_lines(roads):
            feats.append({"type": "Feature", "properties": {"kind": "road"},
                          "geometry": {"type": "LineString", "coordinates":
                                       [list(to_ll(r, c)) for r, c in pts]}})
    if zones is not None:
        for cls, poly in zones_to_polygons(zones):
            rings = [_ring_lonlat(poly.exterior.coords, to_ll)] + \
                    [_ring_lonlat(i.coords, to_ll) for i in poly.interiors]
            feats.append({"type": "Feature",
                          "properties": {"kind": "zone",
                                         "zone": ZONE_NAMES[cls]},
                          "geometry": {"type": "Polygon",
                                       "coordinates": rings}})
    if dens is not None:
        for _, poly in density_to_polygons(dens, thr=dens_thr):
            rings = [_ring_lonlat(poly.exterior.coords, to_ll)] + \
                    [_ring_lonlat(i.coords, to_ll) for i in poly.interiors]
            feats.append({"type": "Feature", "properties": {"kind": "growth"},
                          "geometry": {"type": "Polygon",
                                       "coordinates": rings}})
    for i, (r, c) in enumerate(amen_points or []):
        feats.append({"type": "Feature",
                      "properties": {"kind": "proposed_centre", "rank": i},
                      "geometry": {"type": "Point",
                                   "coordinates": list(to_ll(r, c))}})
    fc = {"type": "FeatureCollection",
          "properties": {"center": [lon, lat], "grid": GRID,
                         "m_per_px": M_PER_PX},
          "features": feats}
    with open(path, "w") as f:
        json.dump(fc, f)
    return fc


if __name__ == "__main__":
    import os
    import tempfile

    LAT, LON = 46.0, 10.0
    R = np.zeros((GRID, GRID), np.uint8)
    drawn = 0.0
    prev = None
    for c in range(20, 101):  # horizontal arm with two diagonal kinks
        r = 60 + (c >= 55) + (c >= 80)
        R[r, c] = 1
        if prev is not None:
            drawn += math.hypot(r - prev[0], c - prev[1])
        prev = (r, c)
    for r in range(20, 101):  # vertical arm through the junction
        R[r, 64] = 1
    drawn += 80.0

    lines = roads_to_lines(R)
    assert len(lines) >= 2, f"expected >=2 polylines, got {len(lines)}"
    ends = [p for ln in lines for p in (ln[0], ln[-1])]
    for tip in [(60, 20), (62, 100), (20, 64), (100, 64)]:
        d = min(math.hypot(e[0] - tip[0], e[1] - tip[1]) for e in ends)
        assert d <= 4, f"no endpoint near arm tip {tip} (min dist {d:.1f})"
    total = sum(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(ln[:-1], ln[1:])) for ln in lines)
    assert abs(total - drawn) / drawn <= 0.20, (total, drawn)
    print(f"roads: {len(lines)} lines, length {total:.1f} vs {drawn:.1f} ok")

    Z = np.zeros((GRID, GRID), np.uint8)
    Z[30:42, 30:42] = 2
    zp = zones_to_polygons(Z)
    assert len(zp) == 1 and zp[0][0] == 2
    assert abs(zp[0][1].area - 144) / 144 <= 0.25, zp[0][1].area
    assert zp[0][1].is_valid
    print(f"zones: 1 polygon, area {zp[0][1].area:.0f} ok")

    yy, xx = np.mgrid[0:GRID, 0:GRID]
    D = np.exp(-((yy - 90) ** 2 + (xx - 30) ** 2) / (2 * 8.0 ** 2))
    gp = density_to_polygons(D)
    assert len(gp) == 1 and gp[0][0] == "growth" and gp[0][1].is_valid
    print("density: 1 growth polygon ok")

    tmp = os.path.join(tempfile.gettempdir(), "vector_export_test.geojson")
    fc = export_geojson(tmp, LAT, LON, roads=R, zones=Z, dens=D,
                        amen_points=[(64, 64), (30, 90)])
    fc2 = json.loads(open(tmp).read())
    assert fc2 == json.loads(json.dumps(fc))
    half_m = GRID * M_PER_PX / 2.0
    dlat, dlon = half_m / 111320.0, half_m / (111320.0 *
                                              math.cos(math.radians(LAT)))
    kinds = {"road": "LineString", "zone": "Polygon", "growth": "Polygon",
             "proposed_centre": "Point"}

    def all_coords(c):
        if isinstance(c[0], (int, float)):
            yield c
        else:
            for s in c:
                yield from all_coords(s)

    for ft in fc2["features"]:
        g = ft["geometry"]
        assert g["type"] == kinds[ft["properties"]["kind"]], ft
        for x, y in all_coords(g["coordinates"]):
            assert LON - dlon - 1e-9 <= x <= LON + dlon + 1e-9, (x, y)
            assert LAT - dlat - 1e-9 <= y <= LAT + dlat + 1e-9, (x, y)
    print(f"export: {len(fc2['features'])} features in window ok")

    ln0 = next(f for f in fc2["features"]
               if f["properties"]["kind"] == "road")
    x0, y0 = ln0["geometry"]["coordinates"][0]
    col = (x0 - (LON - dlon)) / (2 * dlon / GRID)   # tiny inverse
    row = ((LAT + dlat) - y0) / (2 * dlat / GRID)
    lines2 = roads_to_lines(R)
    d = min(math.hypot(row - r, col - c) for ln in lines2
            for r, c in (ln[0], ln[-1]))
    assert d < 1e-6, d
    print("inverse pixel round-trip ok")
