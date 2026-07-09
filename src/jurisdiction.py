"""Jurisdiction rule packs: local regulation as data, not code.

The compliance checks ship with one set of thresholds, and no single set
of thresholds is the law anywhere. This module makes the applicable
rules an input. A rule pack is a small dict (loadable from JSON) naming
the jurisdiction, citing the legal instruments it draws from, and
setting the numeric knobs that compliance.check_plan accepts. Cities
differ by data, not by forked code.

Three packs ship as seeds. "default" is the pipeline's own thresholds
and claims no legal force. "us_nfip_floodplain" encodes the one nearly
universal rule of United States floodplain regulation: communities in
the National Flood Insurance Program must regulate development in
Special Flood Hazard Areas (44 CFR Part 60, sec. 60.3), so growth on or
near mapped water is treated as a regulatory matter, not merely a
hazard note. "ch_nonbuilding_zones" reflects the Swiss planning act
(RPG, SR 700, Art. 22 and 24): construction belongs inside designated
building zones and needs exceptional permission outside them, which the
pack approximates by tightening the leapfrog ring.

The limits of this design are stated where they matter. A raster water
mask is a proxy for a FIRM flood map; a distance ring is a proxy for a
zone-plan boundary; a pack marked "illustrative" has not been reviewed
by anyone with authority over the jurisdiction it names. Every pack
must carry a verify_locally statement, and every findings list produced
through a pack ends with a provenance note naming the pack and its
citations, so a staff report can never silently launder a proxy into a
legal determination.

Dependencies: json from the standard library; compliance for the
check wrapper.
"""

import json

PX_M = 15.0  # raster resolution, metres per pixel

_REQUIRED_KEYS = ("id", "name", "jurisdiction", "level", "citations",
                  "verify_locally", "rules")
_LEVELS = ("pipeline-default", "illustrative")
_RULE_KEYS = ("slope_limit_deg", "industrial_buffer_m", "leapfrog_max_m",
              "high_density", "high_density_share", "water_regulated")

PACKS = {
    "default": {
        "id": "default",
        "name": "Pipeline defaults",
        "jurisdiction": "none",
        "level": "pipeline-default",
        "citations": [],
        "verify_locally": "These are the pipeline's own analysis "
                          "thresholds. They carry no legal force "
                          "anywhere.",
        "rules": {},
    },
    "us_nfip_floodplain": {
        "id": "us_nfip_floodplain",
        "name": "US NFIP floodplain management (generic)",
        "jurisdiction": "United States, NFIP participating communities",
        "level": "illustrative",
        "citations": [
            {"instrument": "44 CFR Part 60 (Floodplain Management "
                           "Criteria)", "provision": "sec. 60.3"},
        ],
        "verify_locally": "SFHA boundaries come from the community's "
                          "effective FIRM, and the local flood damage "
                          "prevention ordinance controls. The raster "
                          "water mask is a proxy and understates the "
                          "regulated area.",
        "rules": {"water_regulated": True},
    },
    "ch_nonbuilding_zones": {
        "id": "ch_nonbuilding_zones",
        "name": "Swiss building-zone separation (generic)",
        "jurisdiction": "Switzerland",
        "level": "illustrative",
        "citations": [
            {"instrument": "Raumplanungsgesetz (RPG, SR 700)",
             "provision": "Art. 22, Art. 24"},
        ],
        "verify_locally": "The controlling boundary is the communal "
                          "Nutzungsplan (zone plan), which this pack "
                          "approximates with a distance ring around the "
                          "existing footprint. Cantonal law adds "
                          "further requirements.",
        "rules": {"leapfrog_max_m": 90.0},
    },
}


def validate_pack(pack):
    """Raise ValueError unless the pack satisfies the schema."""
    if not isinstance(pack, dict):
        raise ValueError("pack must be a dict")
    missing = [k for k in _REQUIRED_KEYS if k not in pack]
    if missing:
        raise ValueError(f"pack missing keys: {missing}")
    if pack["level"] not in _LEVELS:
        raise ValueError(f"pack level must be one of {_LEVELS}")
    if pack["level"] != "pipeline-default":
        if not pack["citations"]:
            raise ValueError("a jurisdictional pack must cite at least "
                             "one legal instrument")
        if not str(pack["verify_locally"]).strip():
            raise ValueError("a jurisdictional pack must state how to "
                             "verify it locally")
    unknown = [k for k in pack["rules"] if k not in _RULE_KEYS]
    if unknown:
        raise ValueError(f"unknown rule keys: {unknown}; known: "
                         f"{_RULE_KEYS}")
    r = pack["rules"]
    for key, lo, hi in (("slope_limit_deg", 1.0, 60.0),
                        ("industrial_buffer_m", 0.0, 600.0),
                        ("leapfrog_max_m", 15.0, 1500.0),
                        ("high_density", 0.1, 1.0),
                        ("high_density_share", 0.05, 1.0)):
        if key in r and not (lo <= float(r[key]) <= hi):
            raise ValueError(f"{key} out of range [{lo}, {hi}]: "
                             f"{r[key]!r}")
    return pack


def load_pack(source):
    """Accept a pack id, a dict, or a path to a JSON file."""
    if isinstance(source, str) and source in PACKS:
        return validate_pack(PACKS[source])
    if isinstance(source, dict):
        return validate_pack(source)
    with open(source, encoding="utf-8") as fh:
        return validate_pack(json.load(fh))


def thresholds_for(pack):
    """Translate a pack's rules into compliance.check_plan limits.

    Distances in metres convert to pixels at 15 m each, rounded to the
    nearest whole pixel with a floor of one. Keys the pack does not set
    are omitted, so check_plan keeps its defaults for them.
    """
    r = validate_pack(pack)["rules"]
    lim = {}
    if "slope_limit_deg" in r:
        lim["slope_limit_deg"] = float(r["slope_limit_deg"])
    if "industrial_buffer_m" in r:
        lim["industrial_buffer_px"] = max(
            1, round(float(r["industrial_buffer_m"]) / PX_M))
    if "leapfrog_max_m" in r:
        lim["leapfrog_ring_px"] = max(
            1, round(float(r["leapfrog_max_m"]) / PX_M))
    if "high_density" in r:
        lim["high_density"] = float(r["high_density"])
    if "high_density_share" in r:
        lim["high_density_share"] = float(r["high_density_share"])
    return lim


def check_with_pack(pack, dens_new, roads_all, zones_r, d0, elev,
                    water=None, protected=None):
    """compliance.check_plan under a rule pack, with provenance.

    Escalates water findings to regulatory wording when the pack sets
    water_regulated, appends a note when a regulated layer is missing
    from the inputs, and always appends a provenance note naming the
    pack, its citations, and its verify_locally statement.
    """
    from compliance import check_plan
    pack = validate_pack(pack)
    findings = check_plan(dens_new, roads_all, zones_r, d0, elev,
                          water=water, protected=protected,
                          limits=thresholds_for(pack))
    if pack["rules"].get("water_regulated"):
        for f in findings:
            if f["code"] == "W1":
                f["message"] += (" Under this jurisdiction's pack, "
                                 "development in the mapped flood area "
                                 "is a permit matter, not only a hazard "
                                 "note.")
        if water is None:
            findings.append({"code": "J1", "severity": "note", "count": 0,
                             "message": "This pack regulates flood "
                             "areas but no water mask was supplied, so "
                             "the regulated layer was not checked."})
    cites = "; ".join(f"{c['instrument']}, {c['provision']}"
                      for c in pack["citations"]) or "none"
    findings.append({"code": "J0", "severity": "note", "count": 0,
                     "message": f"Checked under rule pack "
                     f"'{pack['id']}' ({pack['name']}; level: "
                     f"{pack['level']}; citations: {cites}). "
                     + pack["verify_locally"]})
    return findings


if __name__ == "__main__":
    import numpy as np

    for pid in PACKS:
        validate_pack(PACKS[pid])
    assert thresholds_for(PACKS["default"]) == {}
    ch = thresholds_for(PACKS["ch_nonbuilding_zones"])
    assert ch == {"leapfrog_ring_px": 6}, ch

    # schema rejections
    bad = dict(PACKS["ch_nonbuilding_zones"], citations=[])
    try:
        validate_pack(bad)
        raise AssertionError("accepted a jurisdictional pack with no "
                             "citations")
    except ValueError:
        pass
    bad2 = dict(PACKS["default"], rules={"parking_minimum": 2})
    try:
        validate_pack(bad2)
        raise AssertionError("accepted an unknown rule key")
    except ValueError:
        pass
    bad3 = dict(PACKS["ch_nonbuilding_zones"])
    bad3["rules"] = {"leapfrog_max_m": 5000.0}
    try:
        validate_pack(bad3)
        raise AssertionError("accepted an out-of-range distance")
    except ValueError:
        pass

    # a custom pack round-trips through JSON
    import tempfile, os
    custom = {"id": "test_town", "name": "Test town hillside rules",
              "jurisdiction": "Testville", "level": "illustrative",
              "citations": [{"instrument": "Testville Municipal Code",
                             "provision": "ch. 17"}],
              "verify_locally": "Confirm with the Testville planning "
                                "department.",
              "rules": {"slope_limit_deg": 15.0,
                        "industrial_buffer_m": 60.0}}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(custom, fh)
    loaded = load_pack(path)
    os.unlink(path)
    lim = thresholds_for(loaded)
    assert lim == {"slope_limit_deg": 15.0, "industrial_buffer_px": 4}

    # the stricter pack finds violations the default does not
    g = 64
    yy, xx = np.mgrid[0:g, 0:g]
    d0 = np.zeros((g, g), np.float32)
    d0[28:36, 28:36] = 0.5
    elev = (xx * 5.0).astype(np.float32)   # about 18 degrees: over the
                                           # pack's 15, under the base 25
    dens = d0.copy()
    dens[36:40, 28:36] = 0.5               # growth on the moderate slope
    zones_r = np.zeros((g, g), np.uint8)
    from compliance import check_plan
    base = check_plan(dens, None, zones_r, d0, elev)
    assert not any(f["code"] == "S1" for f in base)
    strict = check_with_pack(loaded, dens, None, zones_r, d0, elev)
    assert any(f["code"] == "S1" for f in strict)
    assert any(f["code"] == "J0" and "Testville" in f["message"]
               for f in strict)

    # NFIP pack: missing water mask is reported, present one escalates
    nfip = PACKS["us_nfip_floodplain"]
    fs = check_with_pack(nfip, dens, None, zones_r, d0, elev)
    assert any(f["code"] == "J1" for f in fs)
    water = np.zeros((g, g)); water[36:40, 28:30] = 1
    fs2 = check_with_pack(nfip, dens, None, zones_r, d0, elev,
                          water=water)
    w1 = [f for f in fs2 if f["code"] == "W1"]
    assert w1 and "permit matter" in w1[0]["message"]
    assert not any(f["code"] == "J1" for f in fs2)
    provenance = [f for f in fs2 if f["code"] == "J0"]
    assert provenance and "44 CFR" in provenance[0]["message"]
    print("jurisdiction self-tests passed")
