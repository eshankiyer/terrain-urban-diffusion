"""Human-readable plan rendering: hillshaded terrain, roads, zone colours,
green/water, and proposed local centres, drawn like a planning map with a
legend and scale bar instead of raw model rasters.

Dependencies: numpy, matplotlib.
"""

import math

import numpy as np

from data import GRID, M_PER_PX, slope_from_elevation

ZONE_COLORS = {1: "#f2d16b",   # residential (yellow)
               2: "#e0655f",   # commercial/retail (red)
               3: "#a58bc4",   # industrial (purple)
               4: "#6fa8dc"}   # institutional (blue)
ZONE_LABELS = {1: "residential", 2: "commercial", 3: "industrial",
               4: "institutional"}
DENSITY_THR = 0.05


def hillshade(elev, azimuth=315.0, altitude=45.0):
    """Classic Lambertian hillshade in [0, 1]."""
    gy, gx = np.gradient(elev.astype(np.float64), M_PER_PX)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(azimuth), math.radians(altitude)
    shade = (np.sin(alt) * np.cos(slope)
             + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip((shade + 1.0) / 2.0, 0, 1)


def amenity_peaks(amen, n=5, min_sep_px=15, thr=0.25):
    """Strongest well-separated peaks of a proposed amenity channel."""
    a = amen.copy()
    peaks = []
    for _ in range(n):
        idx = int(np.argmax(a))
        y, x = divmod(idx, a.shape[1])
        if a[y, x] < thr:
            break
        peaks.append((y, x))
        y0, y1 = max(0, y - min_sep_px), min(a.shape[0], y + min_sep_px)
        x0, x1 = max(0, x - min_sep_px), min(a.shape[1], x + min_sep_px)
        a[y0:y1, x0:x1] = 0.0
    return peaks


def render_plan(elev, roads, dens, zones=None, green=None, water=None,
                amen_proposed=None, existing_dens=None, title="",
                path=None, ax=None):
    """Draw one plan. `zones` colours new development by use class;
    `existing_dens` (e.g. the 2020 footprint) is drawn in grey so new
    growth reads at a glance. Returns the matplotlib axis."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(hillshade(elev), cmap="gray", vmin=0, vmax=1, alpha=0.9)

    def overlay(mask, color, alpha):
        rgba = np.zeros((GRID, GRID, 4))
        rgba[mask.astype(bool)] = list(color) + [alpha]
        ax.imshow(rgba)

    from matplotlib.colors import to_rgb
    if green is not None and green.sum():
        overlay(green, to_rgb("#79b473"), 0.55)
    if water is not None and water.sum():
        overlay(water, to_rgb("#5b9bd5"), 0.75)

    if existing_dens is not None:
        overlay(existing_dens > DENSITY_THR, to_rgb("#8a8a8a"), 0.75)

    built = dens > DENSITY_THR
    if existing_dens is not None:
        built = built & ~(existing_dens > DENSITY_THR)
    if zones is not None:
        for cls, hexcol in ZONE_COLORS.items():
            overlay(built & (zones == cls), to_rgb(hexcol), 0.9)
        overlay(built & (zones == 0), to_rgb(ZONE_COLORS[1]), 0.6)
    else:
        overlay(built, to_rgb(ZONE_COLORS[1]), 0.8)

    overlay(roads.astype(bool), to_rgb("#222222"), 0.95)

    handles = []
    if amen_proposed is not None:
        for y, x in amenity_peaks(amen_proposed):
            ax.scatter([x], [y], marker="*", s=180, color="#d62828",
                       edgecolors="white", linewidths=0.8, zorder=5)
        handles.append(Line2D([], [], marker="*", linestyle="none",
                              markersize=12, color="#d62828",
                              label="proposed local centre"))

    if existing_dens is not None:
        handles.append(Patch(facecolor="#8a8a8a", label="existing town"))
    if zones is not None:
        handles += [Patch(facecolor=ZONE_COLORS[c], label=ZONE_LABELS[c])
                    for c in ZONE_COLORS]
    else:
        handles.append(Patch(facecolor=ZONE_COLORS[1], label="new growth"))
    handles.append(Line2D([], [], color="#222222", label="roads"))
    if green is not None and green.sum():
        handles.append(Patch(facecolor="#79b473", label="green space"))
    if water is not None and water.sum():
        handles.append(Patch(facecolor="#5b9bd5", label="water"))
    ax.legend(handles=handles, loc="lower left", fontsize=7,
              framealpha=0.9)

    # 500 m scale bar
    bar_px = 500.0 / M_PER_PX
    y0, x0 = GRID - 8, GRID - 10 - bar_px
    ax.plot([x0, x0 + bar_px], [y0, y0], color="black", lw=3)
    ax.text(x0 + bar_px / 2, y0 - 3, "500 m", ha="center", fontsize=7)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, GRID - 1)
    ax.set_ylim(GRID - 1, 0)
    ax.axis("off")
    if own_fig and path:
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(path, dpi=170, bbox_inches="tight")
        plt.close()
    return ax
