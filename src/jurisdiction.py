"""Jurisdiction rule packs inferred from planning documents.

The compliance checks ship with one set of thresholds, and no single
set of thresholds is the law anywhere. This module makes the applicable
rules an input, and it obtains them by reading the documents a
jurisdiction actually publishes: a zoning ordinance, a comprehensive
plan, a hillside or floodplain chapter, or an excerpt of a model code
such as the International Building Code. No code text is bundled here;
model codes are copyrighted and local amendments control anyway, so the
caller supplies the text and the module extracts from exactly what it
was given.

Extraction is two-layered. A deterministic sentence-level extractor
recognises the common numeric patterns of planning English (slopes
exceeding N percent, separations of N feet between industrial and
residential uses, contiguity requirements within N feet of existing
development, floodplain permit language) and works offline. A language
model, supplied through llm_adapter's pluggable backend, reads the same
text chunk by chunk and proposes additional candidates in a fixed JSON
shape. Both layers produce candidates, never conclusions: every
candidate carries the quote it came from, candidates merge under a
strictest-wins rule, merged values must pass the same range validation
as hand-written packs, and the resulting pack is marked "extracted",
carries its evidence, and states in its own text that each value must
be confirmed against the document before reliance. An extracted number
is a reading aid; the document remains the authority.

A pack, however produced, is a small dict (JSON-serialisable) naming
its source, citing provisions, and setting the numeric thresholds that
compliance.check_plan accepts. Hand-written packs are still supported
through the same schema with level "manual".

Dependencies: json and re from the standard library; llm_adapter for
the optional model layer; compliance for the check wrapper.
"""

import json
import math
import re

from llm_adapter import EchoBackend, as_backend, generate_json

PX_M = 15.0  # raster resolution, metres per pixel
FT_M = 0.3048

_REQUIRED_KEYS = ("id", "name", "jurisdiction", "level", "citations",
                  "verify_locally", "rules")
_LEVELS = ("pipeline-default", "extracted", "manual")
_RULE_KEYS = ("slope_limit_deg", "industrial_buffer_m", "leapfrog_max_m",
              "high_density", "high_density_share", "water_regulated")
_RANGES = {"slope_limit_deg": (1.0, 60.0),
           "industrial_buffer_m": (0.0, 600.0),
           "leapfrog_max_m": (15.0, 1500.0),
           "high_density": (0.1, 1.0),
           "high_density_share": (0.05, 1.0)}

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
            raise ValueError("a jurisdictional pack must cite its source")
        if not str(pack["verify_locally"]).strip():
            raise ValueError("a jurisdictional pack must state how to "
                             "verify it locally")
    unknown = [k for k in pack["rules"] if k not in _RULE_KEYS]
    if unknown:
        raise ValueError(f"unknown rule keys: {unknown}; known: "
                         f"{_RULE_KEYS}")
    for key, (lo, hi) in _RANGES.items():
        if key in pack["rules"]:
            v = float(pack["rules"][key])
            if not lo <= v <= hi:
                raise ValueError(f"{key} out of range [{lo}, {hi}]: {v!r}")
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
    """Translate a pack's rules into compliance.check_plan limits."""
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


# ---------------------------------------------------------------------------
# Rule extraction from document text
# ---------------------------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_SLOPE_RE = re.compile(
    r"slopes?\s+(?:of\s+)?(?:exceeding|greater\s+than|steeper\s+than|"
    r"over|above|in\s+excess\s+of)\s+" + _NUM +
    r"\s*(percent|%|degrees?)", re.IGNORECASE)
_DIST_RE = re.compile(
    _NUM + r"\s*(feet|foot|ft|metres?|meters?|m)\b", re.IGNORECASE)
_FLOOD_RE = re.compile(
    r"special\s+flood\s+hazard|floodplain\s+(?:development\s+)?permit|"
    r"flood\s+damage\s+prevention|regulatory\s+floodway", re.IGNORECASE)
_CONTIG_RE = re.compile(
    r"contiguous|adjacent\s+to\s+existing|urban\s+growth\s+boundary|"
    r"within\s+.{0,30}\s+of\s+existing\s+(?:development|urban)",
    re.IGNORECASE)


def _sentences(text):
    for s in re.split(r"(?<=[.;])\s+|\n{2,}", str(text)):
        s = " ".join(s.split())
        if s:
            yield s


def _to_metres(value, unit):
    return value * FT_M if unit.lower().startswith("f") else value


def _to_degrees(value, unit):
    if unit.startswith("deg"):
        return value
    return math.degrees(math.atan(value / 100.0))


def extract_rules_regex(text, source="document"):
    """Deterministic candidate extraction from planning English.

    Returns a list of candidate dicts {key, value, quote, via}. Each
    candidate quotes the sentence it came from. The extractor is
    intentionally narrow: it reads only patterns whose meaning is
    unambiguous at sentence scope, and silence is its answer to
    everything else.
    """
    cands = []
    for sent in _sentences(text):
        low = sent.lower()
        m = _SLOPE_RE.search(sent)
        if m:
            deg = _to_degrees(float(m.group(1)), m.group(2).lower())
            cands.append({"key": "slope_limit_deg", "value": deg,
                          "quote": sent, "via": "regex"})
        if "industrial" in low and ("residential" in low
                                    or "dwelling" in low):
            dm = _DIST_RE.search(sent)
            if dm and any(w in low for w in ("buffer", "setback",
                                             "separation", "separated",
                                             "screening", "distance")):
                cands.append({"key": "industrial_buffer_m",
                              "value": _to_metres(float(dm.group(1)),
                                                  dm.group(2)),
                              "quote": sent, "via": "regex"})
        if _CONTIG_RE.search(sent):
            dm = _DIST_RE.search(sent)
            if dm:
                cands.append({"key": "leapfrog_max_m",
                              "value": _to_metres(float(dm.group(1)),
                                                  dm.group(2)),
                              "quote": sent, "via": "regex"})
        if _FLOOD_RE.search(sent):
            cands.append({"key": "water_regulated", "value": True,
                          "quote": sent, "via": "regex"})
    return cands


_EXTRACT_SYSTEM = """You read planning and building code text and
extract numeric development rules. Respond ONLY with JSON of the form
{"rules": [{"key": K, "value": V, "unit": U, "quote": Q}]}.
Allowed keys and expected units:
  slope_limit_deg: maximum buildable slope. unit "percent" or "degrees".
  industrial_buffer_m: required separation between industrial and
    residential uses. unit "feet" or "meters".
  leapfrog_max_m: maximum distance new development may sit from
    existing development or a growth boundary. unit "feet" or "meters".
  high_density_share: maximum share of new development at the highest
    intensity tier, as a fraction 0 to 1. unit "fraction".
  water_regulated: true if the text requires permits or prohibits
    development in flood hazard areas. unit "bool".
Q must be a verbatim quote from the text containing the number. Extract
only rules the text actually states. If the text states none, return
{"rules": []}. Never guess values."""


def _llm_candidates(text, backend, chunk_chars=4000):
    backend = as_backend(backend)
    cands = []
    text = str(text)
    for start in range(0, len(text), chunk_chars):
        chunk = text[start:start + chunk_chars]
        payload = generate_json(backend, chunk, system=_EXTRACT_SYSTEM,
                                default={"rules": []})
        for item in payload.get("rules", []):
            try:
                key = item["key"]
                if key not in _RULE_KEYS:
                    continue
                quote = str(item.get("quote", ""))[:400]
                if key == "water_regulated":
                    cands.append({"key": key, "value": bool(item["value"]),
                                  "quote": quote, "via": "llm"})
                    continue
                value = float(item["value"])
                unit = str(item.get("unit", "")).lower()
                if key == "slope_limit_deg":
                    value = _to_degrees(value, unit or "degrees")
                elif key in ("industrial_buffer_m", "leapfrog_max_m"):
                    value = _to_metres(value, unit or "m")
                cands.append({"key": key, "value": value, "quote": quote,
                              "via": "llm"})
            except (KeyError, TypeError, ValueError):
                continue
    return cands


def merge_candidates(cands):
    """(rules, evidence, warnings): strictest value wins per key.

    Strictest means the lowest slope limit, the largest separation
    buffer, the tightest contiguity distance, and the lowest intensity
    share; water_regulated is true if any candidate says so. Candidates
    outside the validation ranges are dropped with a warning rather
    than clamped, since a clamped number would misquote the document.
    """
    strictest = {"slope_limit_deg": min, "industrial_buffer_m": max,
                 "leapfrog_max_m": min, "high_density": min,
                 "high_density_share": min}
    rules, evidence, warnings = {}, [], []
    for c in cands:
        key = c["key"]
        if key == "water_regulated":
            if c["value"]:
                rules[key] = True
                evidence.append(c)
            continue
        lo, hi = _RANGES[key]
        v = float(c["value"])
        if not lo <= v <= hi:
            warnings.append(f"dropped {key}={v:.3g} from "
                            f"{c['via']} (outside [{lo}, {hi}]): "
                            f"{c['quote'][:80]!r}")
            continue
        if key not in rules or strictest[key](rules[key], v) == v:
            rules[key] = v
        evidence.append(c)
    return rules, evidence, warnings


def extract_pack(text, doc_name, jurisdiction="unspecified",
                 backend=None, use_llm=True, pack_id=None):
    """Infer a rule pack from a planning document's text.

    Runs the deterministic extractor always and the language-model
    extractor when a backend is available (default_backend falls back
    to an offline echo that contributes nothing, so this function works
    with no model installed). Returns (pack, warnings). The pack level
    is "extracted", its citations quote the document, its evidence list
    preserves every accepted candidate, and its verify_locally text
    says what an extracted pack is: a machine reading that must be
    confirmed against the document before any reliance.
    """
    cands = extract_rules_regex(text, source=doc_name)
    if use_llm:
        cands += _llm_candidates(text, backend)
    rules, evidence, warnings = merge_candidates(cands)
    citations = []
    seen = set()
    for e in evidence:
        q = e["quote"][:200]
        if q and q not in seen:
            seen.add(q)
            citations.append({"instrument": doc_name, "provision": q})
        if len(citations) >= 8:
            break
    if not rules:
        warnings.append("no extractable rules found; pack carries "
                        "pipeline defaults")
    pack = {
        "id": pack_id or re.sub(r"[^a-z0-9]+", "_", doc_name.lower())[:40],
        "name": f"Extracted from {doc_name}",
        "jurisdiction": jurisdiction,
        "level": "extracted",
        "citations": citations or [{"instrument": doc_name,
                                    "provision": "no provisions "
                                    "extracted"}],
        "verify_locally": "This pack was extracted automatically from "
                          f"the supplied text of {doc_name}. Confirm "
                          "every value against the document and the "
                          "adopting jurisdiction before reliance; "
                          "quoted provisions are evidence of a reading, "
                          "not an interpretation.",
        "rules": rules,
        "evidence": [{k: e[k] for k in ("key", "value", "quote", "via")}
                     for e in evidence],
    }
    core = {k: v for k, v in pack.items() if k != "evidence"}
    validate_pack(core)
    return pack, warnings


def check_with_pack(pack, dens_new, roads_all, zones_r, d0, elev,
                    water=None, protected=None):
    """compliance.check_plan under a rule pack, with provenance.

    Escalates water findings to regulatory wording when the pack sets
    water_regulated, reports a regulated-but-unsupplied water layer,
    and always appends a provenance note naming the pack, its level,
    its citations, and its verification statement.
    """
    from compliance import check_plan
    core = {k: v for k, v in pack.items() if k != "evidence"}
    validate_pack(core)
    findings = check_plan(dens_new, roads_all, zones_r, d0, elev,
                          water=water, protected=protected,
                          limits=thresholds_for(core))
    if pack["rules"].get("water_regulated"):
        for f in findings:
            if f["code"] == "W1":
                f["message"] += (" The applicable pack treats "
                                 "development in the flood area as a "
                                 "permit matter, not only a hazard "
                                 "note.")
        if water is None:
            findings.append({"code": "J1", "severity": "note", "count": 0,
                             "message": "The pack regulates flood areas "
                             "but no water mask was supplied, so the "
                             "regulated layer was not checked."})
    cites = "; ".join(f"{c['instrument']}: {c['provision'][:60]}"
                      for c in pack["citations"][:3]) or "none"
    findings.append({"code": "J0", "severity": "note", "count": 0,
                     "message": f"Checked under rule pack "
                     f"'{pack['id']}' (level: {pack['level']}; "
                     f"evidence: {cites}). " + pack["verify_locally"]})
    return findings


if __name__ == "__main__":
    import numpy as np

    validate_pack(PACKS["default"])
    assert thresholds_for(PACKS["default"]) == {}

    ordinance = """
    Chapter 17: Hillside Development. No structure shall be erected on
    slopes exceeding 15 percent without an engineered grading plan.
    Chapter 8: Industrial Performance Standards. A landscaped buffer of
    at least 100 feet shall be maintained between industrial uses and
    any residential district. Chapter 4: Growth Management. New
    subdivisions shall be contiguous with existing development, and no
    lot shall be located more than 300 feet from existing urban
    services. Chapter 12: Flood Damage Prevention. A floodplain
    development permit is required prior to any construction within a
    special flood hazard area.
    """

    # offline extraction: regex layer alone recovers all four rules
    pack, warns = extract_pack(ordinance, "Testville Municipal Code",
                               jurisdiction="Testville",
                               backend=EchoBackend())
    r = pack["rules"]
    assert abs(r["slope_limit_deg"] - math.degrees(
        math.atan(0.15))) < 0.01, r
    assert abs(r["industrial_buffer_m"] - 100 * FT_M) < 0.01, r
    assert abs(r["leapfrog_max_m"] - 300 * FT_M) < 0.01, r
    assert r["water_regulated"] is True
    assert pack["level"] == "extracted"
    assert any("100 feet" in c["provision"] for c in pack["citations"])
    assert "Confirm" in pack["verify_locally"]

    # a model layer merges in, and strictest wins per key
    def fake_model(prompt, system):
        return json.dumps({"rules": [
            {"key": "slope_limit_deg", "value": 25, "unit": "percent",
             "quote": "slopes exceeding 25 percent are prohibited"},
            {"key": "industrial_buffer_m", "value": 60, "unit": "meters",
             "quote": "a 60 metre separation is required"},
            {"key": "leapfrog_max_m", "value": 9000, "unit": "meters",
             "quote": "way out of range"},
        ]})
    pack2, warns2 = extract_pack(ordinance, "Testville Municipal Code",
                                 backend=fake_model)
    r2 = pack2["rules"]
    # slope: regex 15% (8.53 deg) is stricter than the model's 25%
    assert abs(r2["slope_limit_deg"] - math.degrees(
        math.atan(0.15))) < 0.01
    # buffer: the model's 60 m beats the regex 30.48 m (strictest = max)
    assert abs(r2["industrial_buffer_m"] - 60) < 0.01
    # out-of-range candidate dropped with a warning, not clamped
    assert any("9e+03" in w or "9000" in w for w in warns2), warns2
    assert abs(r2["leapfrog_max_m"] - 300 * FT_M) < 0.01

    # a rule-free document yields defaults plus a warning
    pack3, warns3 = extract_pack("The council meets on Tuesdays.",
                                 "Meeting Minutes",
                                 backend=EchoBackend())
    assert pack3["rules"] == {} and any("no extractable" in w
                                        for w in warns3)

    # extracted packs drive the checks and carry provenance
    g = 64
    yy, xx = np.mgrid[0:g, 0:g]
    d0 = np.zeros((g, g), np.float32)
    d0[28:36, 28:36] = 0.5
    elev = (xx * 3.0).astype(np.float32)   # about 11 degrees: over the
                                           # extracted 8.5, under 25
    dens = d0.copy()
    dens[36:40, 28:36] = 0.5
    zones_r = np.zeros((g, g), np.uint8)
    from compliance import check_plan
    base = check_plan(dens, None, zones_r, d0, elev)
    assert not any(f["code"] == "S1" for f in base)
    strict = check_with_pack(pack, dens, None, zones_r, d0, elev)
    assert any(f["code"] == "S1" for f in strict)
    assert any(f["code"] == "J1" for f in strict)   # water regulated,
                                                    # no mask given
    j0 = [f for f in strict if f["code"] == "J0"]
    assert j0 and "Testville Municipal Code" in j0[0]["message"]

    # hand-written packs still validate through the same schema
    manual = {"id": "m1", "name": "Manual pack", "jurisdiction": "X",
              "level": "manual",
              "citations": [{"instrument": "X Code", "provision": "s.1"}],
              "verify_locally": "Confirm with X planning department.",
              "rules": {"slope_limit_deg": 20.0}}
    assert thresholds_for(load_pack(manual)) == {"slope_limit_deg": 20.0}
    bad = dict(manual, citations=[])
    try:
        validate_pack(bad)
        raise AssertionError("accepted an uncited manual pack")
    except ValueError:
        pass
    print("jurisdiction self-tests passed")
