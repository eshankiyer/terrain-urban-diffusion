"""Raster-scale compliance checks for generated plans.

Development review asks whether a proposal violates the rules that apply
to it. At 15 m per pixel there are no setbacks or parking counts to
check, so this module covers the subset of review that raster geometry
can support: hazard placement, land use adjacency, intensity, and
leapfrog development. Each check returns a finding with a code, a
severity, a pixel count, and a sentence a staff report can quote. The
check thresholds are visible module constants, not learned quantities,
because a compliance rule that cannot be pointed at is not a rule.

Severity levels: "violation" means the plan conflicts with a hard
constraint the pipeline itself enforces, so any nonzero count indicates
a bug upstream or a hand-edited plan; "caution" means a pattern most
codes regulate and a human should look at; "note" is informational.

Dependencies: numpy; zones for the shared dilation helper.
"""

import numpy as np

from zones import _chebyshev_dilate

SLOPE_LIMIT_DEG = 25.0        # matches the pipeline's landslide proxy
INDUSTRIAL_BUFFER_PX = 2      # ~30 m separation industrial to residential
LEAPFROG_RING_PX = 12         # ~180 m past the footprint counts as leapfrog
HIGH_DENSITY = 0.6            # top residential display tier
HIGH_DENSITY_SHARE = 0.5      # above this share of growth, flag intensity
D1_MIN_GROWTH_PX = 20         # share is meaningless below this many px
DENSITY_THR = 0.05


def _slope_deg(elev):
    gy, gx = np.gradient(np.asarray(elev, float), 15.0)
    return np.degrees(np.arctan(np.hypot(gy, gx)))


def check_plan(dens_new, roads_all, zones_r, d0, elev,
               water=None, protected=None, limits=None):
    """Run every check against one candidate plan.

    dens_new: generated density raster. roads_all: road mask including
    existing roads; reserved for future connectivity checks and
    currently unchecked, the parameter stays for API stability.
    zones_r: typed zone raster from assign_zones, zero where untyped.
    d0: existing density. elev: elevation in metres. water, protected:
    optional masks. limits: optional dict overriding the module
    thresholds for a specific jurisdiction (keys slope_limit_deg,
    industrial_buffer_px, leapfrog_ring_px, high_density,
    high_density_share; missing keys keep the module defaults). See
    jurisdiction.thresholds_for, which builds this dict from a rule
    pack. Raises ValueError if dens_new or elev contain NaN or infinite
    values. Returns a list of finding dicts ordered violations first.
    """
    dens_new = np.asarray(dens_new, np.float32)
    d0 = np.asarray(d0, np.float32)
    elev = np.asarray(elev, float)
    if not np.isfinite(dens_new).all():
        raise ValueError("dens_new contains non-finite values")
    if not np.isfinite(elev).all():
        raise ValueError("elev contains non-finite values")
    lim = limits or {}
    slope_limit = float(lim.get("slope_limit_deg", SLOPE_LIMIT_DEG))
    ind_buffer = int(lim.get("industrial_buffer_px",
                             INDUSTRIAL_BUFFER_PX))
    leap_ring = int(lim.get("leapfrog_ring_px", LEAPFROG_RING_PX))
    hi_dens = float(lim.get("high_density", HIGH_DENSITY))
    hi_share = float(lim.get("high_density_share", HIGH_DENSITY_SHARE))
    growth = (dens_new > DENSITY_THR) & ~(d0 > DENSITY_THR)
    findings = []

    if water is not None:
        n = int((growth & (np.asarray(water) > 0.5)).sum())
        if n:
            findings.append({"code": "W1", "severity": "violation",
                             "count": n,
                             "message": f"{n} growth pixels fall on mapped "
                             "water. The water lock should make this "
                             "impossible; treat as a pipeline defect."})

    steep = _slope_deg(elev) > slope_limit
    n = int((growth & steep).sum())
    if n:
        findings.append({"code": "S1", "severity": "violation", "count": n,
                         "message": f"{n} growth pixels sit on slopes over "
                         f"{slope_limit:.0f} degrees, past the applicable "
                         "slope threshold."})

    if protected is not None:
        n = int((growth & (np.asarray(protected) > 0.5)).sum())
        if n:
            findings.append({"code": "P1", "severity": "violation",
                             "count": n,
                             "message": f"{n} growth pixels intrude on "
                             "protected area."})

    z = np.asarray(zones_r)
    res = (z == 1) | (z == 5)
    ind = z == 3
    if ind.any() and res.any():
        near_ind = _chebyshev_dilate(ind, ind_buffer)
        n = int((res & near_ind).sum())
        if n:
            findings.append({"code": "A1", "severity": "caution",
                             "count": n,
                             "message": f"{n} residential or mixed use "
                             "pixels lie within "
                             f"{ind_buffer * 15} m of industrial "
                             "typing. Most codes require separation or "
                             "screening here."})

    foot = d0 > DENSITY_THR
    if foot.any():
        ring = _chebyshev_dilate(foot, leap_ring)
        n = int((growth & ~ring).sum())
        if n:
            findings.append({"code": "L1", "severity": "caution",
                             "count": n,
                             "message": f"{n} growth pixels sit more than "
                             f"{leap_ring * 15} m from the existing "
                             "footprint, a leapfrog pattern that raises "
                             "service costs."})
    else:
        findings.append({"code": "L0", "severity": "note", "count": 0,
                         "message": "No existing footprint above the "
                         "density threshold, so the leapfrog check was "
                         "not applicable."})

    n_growth = int(growth.sum())
    if n_growth >= D1_MIN_GROWTH_PX:
        hi = int((growth & (dens_new > hi_dens)).sum())
        share = hi / n_growth
        if share > hi_share:
            findings.append({"code": "D1", "severity": "note",
                             "count": hi,
                             "message": f"{hi} of {n_growth} growth "
                             f"pixels ({share:.0%}) land in the highest "
                             "density tier. Worth checking against the "
                             "intended intensity for the area."})

    untyped = int((growth & (z == 0)).sum())
    if n_growth and untyped / n_growth > 0.5:
        findings.append({"code": "U1", "severity": "note",
                         "count": untyped,
                         "message": f"{untyped} of {n_growth} growth "
                         "pixels are untyped after abstention. The use "
                         "recommendation for this plan is weak."})

    order = {"violation": 0, "caution": 1, "note": 2}
    findings.sort(key=lambda f: (order[f["severity"]], f["code"]))
    return findings


def to_markdown(findings):
    if not findings:
        return ("## Compliance findings\n\nNo findings. All checks "
                "passed at raster scale. Dimensional standards, parking, "
                "and legal review remain outside this tool's reach.")
    lines = ["## Compliance findings", ""]
    for f in findings:
        lines.append(f"- [{f['severity'].upper()}] {f['code']}: "
                     f"{f['message']}")
    lines.append("")
    lines.append("Raster checks cover placement and adjacency only. "
                 "Setbacks, height, parking, and anything requiring "
                 "parcel geometry are not checked here.")
    return "\n".join(lines)


if __name__ == "__main__":
    g = 64
    yy, xx = np.mgrid[0:g, 0:g]
    d0 = np.zeros((g, g), np.float32)
    d0[28:36, 28:36] = 0.5                       # existing core
    elev = (xx * 12.0).astype(np.float32)        # steep toward the east
    water = np.zeros((g, g)); water[:, :4] = 1   # river on the west edge

    dens = d0.copy()
    dens[36:40, 28:36] = 0.7                     # infill next to core
    dens[2:6, 50:54] = 0.5                       # leapfrog patch, far away
    dens[30:32, 1:3] = 0.5                       # on the river: violation
    zones_r = np.zeros((g, g), np.uint8)
    zones_r[36:40, 28:32] = 1                    # residential growth
    zones_r[36:40, 32:36] = 3                    # industrial right beside it
    roads = np.zeros((g, g), np.uint8)

    fs = check_plan(dens, roads, zones_r, d0, elev, water=water)
    codes = [f["code"] for f in fs]
    assert "W1" in codes, codes
    assert "S1" in codes, codes                  # eastern leapfrog is steep
    assert "A1" in codes, codes
    assert "L1" in codes, codes
    sev = [f["severity"] for f in fs]
    assert sev == sorted(sev, key={"violation": 0, "caution": 1,
                                   "note": 2}.get)

    # a clean infill-only plan produces no violations
    clean = d0.copy()
    clean[36:38, 28:34] = 0.4
    zc = np.zeros((g, g), np.uint8)
    zc[36:38, 28:34] = 1
    fs2 = check_plan(clean, roads, zc, d0, np.zeros((g, g), np.float32),
                     water=water)
    assert not any(f["severity"] == "violation" for f in fs2), fs2

    # non-finite inputs are rejected up front
    nan_dens = dens.copy(); nan_dens[0, 0] = np.nan
    try:
        check_plan(nan_dens, roads, zones_r, d0, elev, water=water)
        raise AssertionError("accepted NaN density")
    except ValueError:
        pass
    inf_elev = elev.copy(); inf_elev[0, 0] = np.inf
    try:
        check_plan(dens, roads, zones_r, d0, inf_elev, water=water)
        raise AssertionError("accepted infinite elevation")
    except ValueError:
        pass

    # empty existing footprint yields an L0 note, never a crash
    fs3 = check_plan(clean, roads, zc, np.zeros((g, g), np.float32),
                     np.zeros((g, g), np.float32))
    assert any(f["code"] == "L0" for f in fs3), fs3

    # D1 reports absolute counts and is gated on 20+ growth pixels
    d1 = [f for f in fs if f["code"] == "D1"]
    assert d1 and " of " in d1[0]["message"], d1
    tiny = d0.copy(); tiny[36:38, 28:32] = 0.9       # 8 growth px, all hi
    fs4 = check_plan(tiny, roads, zones_r, d0, elev)
    assert not any(f["code"] == "D1" for f in fs4), fs4

    md = to_markdown(fs)
    assert "W1" in md and chr(0x2014) not in md
    assert "No findings" in to_markdown([])
    print("compliance self-tests passed")
