"""v4 skeleton: natural-language planning agent over existing tools.

A planner types "we need more commercial near the centre and protect the
floodplain" and it becomes a small list of VERIFIED operations (Regenerate,
SetZoneBias, AddAmenityTarget, Protect, Rerank) routed onto tools that
already exist in this repo: RePaint inpainting (inpaint.py), the zone
classifier (zones.py), the amenity channel, and the sustainability
scorecard. The parser -- an LLM via Ollama, or the rule parser standing in
for it -- NEVER invents geometry: regions come from a fixed mask grammar
(region_mask), zones/metrics are validated against closed vocabularies,
and every candidate set still flows through the same scorecard, so a
request that would pave the floodplain gets a visibly low score.

Honest caveats:
- The rule parser is keyword matching, one op per clause with fixed
  precedence; anything it cannot map becomes a warning, never a guessed op.
- Zone bias reimplements the assign_zones pixel loop so it can scale
  clf.predict_proba per class inside the region before argmax; if the
  classifier lacks predict_proba (or zclf is None) the bias is logged and
  skipped -- no relabel is invented.
- rank_samples does not accept weight overrides, so Rerank recomputes
  totals from its per-metric subscores; the delivery adjustment it applied
  is preserved as the ratio adjusted_total / raw_weighted_total.
- Rerank metrics absent from the active scorecard contribute zero and are
  warned about, not silently honoured.

Dependencies: numpy (+ the repo modules; torch/sklearn only on the
model/classifier paths, never at import time).
"""

import dataclasses
import json
import re
import urllib.request
from dataclasses import dataclass

import numpy as np

from data import GRID, binary_dilate, slope_from_elevation
from sustainability import (WEIGHTS, WEIGHTS_V2, centers_from_density,
                            rank_samples)

ZONE_IDS = {"residential": 1, "commercial": 2, "industrial": 3,
            "institutional": 4}
METRICS = sorted(set(WEIGHTS) | set(WEIGHTS_V2))
REGION_TOKENS = ("north", "south", "east", "west", "centre", "center",
                 "edge", "fringe", "floodplain", "steep", "riverside",
                 "everywhere")
DENSITY_THR = 0.05


# ----------------------------------------------------------------------------
# Op schema
# ----------------------------------------------------------------------------

class _Op:
    def to_dict(self):
        d = {"op": type(self).__name__}
        d.update(dataclasses.asdict(self))
        return d


@dataclass
class Regenerate(_Op):
    region: str


@dataclass
class SetZoneBias(_Op):
    zone: str
    weight: float
    region: str


@dataclass
class AddAmenityTarget(_Op):
    category: str
    region: str


@dataclass
class Protect(_Op):
    region: str


@dataclass
class Rerank(_Op):
    weights: dict


_OP_TYPES = {c.__name__: c for c in
             (Regenerate, SetZoneBias, AddAmenityTarget, Protect, Rerank)}


def _check_region(region):
    toks = [t for t in re.split(r"[\s+]+", str(region).strip().lower()) if t]
    if not toks or any(t not in REGION_TOKENS for t in toks):
        raise ValueError(f"unknown region {region!r}; known: {REGION_TOKENS}")


def validate_op(op):
    """Raise ValueError unless op is inside the closed vocabularies."""
    if isinstance(op, (Regenerate, Protect)):
        _check_region(op.region)
    elif isinstance(op, SetZoneBias):
        if op.zone not in ZONE_IDS:
            raise ValueError(f"unknown zone {op.zone!r}; known: "
                             f"{sorted(ZONE_IDS)}")
        if not isinstance(op.weight, (int, float)) or not -0.99 <= op.weight <= 4:
            raise ValueError(f"zone weight out of range: {op.weight!r}")
        _check_region(op.region)
    elif isinstance(op, AddAmenityTarget):
        if not str(op.category).strip():
            raise ValueError("empty amenity category")
        _check_region(op.region)
    elif isinstance(op, Rerank):
        if not isinstance(op.weights, dict) or not op.weights:
            raise ValueError("Rerank.weights must be a non-empty dict")
        for k, v in op.weights.items():
            if k not in METRICS:
                raise ValueError(f"unknown metric {k!r}; known: {METRICS}")
            if not isinstance(v, (int, float)) or not 0 <= v <= 2:
                raise ValueError(f"metric weight out of range: {k}={v!r}")
    else:
        raise ValueError(f"not an op: {op!r}")


def op_from_dict(d):
    """Strict inverse of to_dict: unknown op names or fields are rejected."""
    if not isinstance(d, dict) or "op" not in d:
        raise ValueError(f"op dict needs an 'op' key: {d!r}")
    cls = _OP_TYPES.get(d["op"])
    if cls is None:
        raise ValueError(f"unknown op {d['op']!r}; known: {sorted(_OP_TYPES)}")
    body = {k: v for k, v in d.items() if k != "op"}
    fields = {f.name for f in dataclasses.fields(cls)}
    if set(body) != fields:
        raise ValueError(f"{d['op']} fields must be {sorted(fields)}, "
                         f"got {sorted(body)}")
    return cls(**body)


# ----------------------------------------------------------------------------
# Region grammar -> masks (the only place words become geometry)
# ----------------------------------------------------------------------------

def region_mask(region, ctx):
    """float32 [G,G] mask for a region phrase. A phrase may combine several
    known tokens ("east edge" = intersection). ctx needs d0 always, elev for
    'steep', env['flood'|'water'] for 'floodplain'/'riverside'."""
    d0 = np.asarray(ctx["d0"], np.float32)
    g = d0.shape[0]
    toks = [t for t in re.split(r"[\s+]+", str(region).strip().lower()) if t]
    if not toks:
        raise ValueError(f"empty region; known: {REGION_TOKENS}")
    yy, xx = np.mgrid[0:g, 0:g]
    out = np.ones((g, g), dtype=bool)
    for t in toks:
        if t == "north":
            m = yy < g / 2
        elif t == "south":
            m = yy >= g / 2
        elif t == "west":
            m = xx < g / 2
        elif t == "east":
            m = xx >= g / 2
        elif t in ("centre", "center"):
            w = np.clip(d0, 0, None)
            if w.sum() > 0:
                cy = float((yy * w).sum() / w.sum())
                cx = float((xx * w).sum() / w.sum())
            else:
                cy = cx = g / 2.0
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= (g / 6.0) ** 2
        elif t in ("edge", "fringe"):
            foot = (d0 > DENSITY_THR).astype(np.uint8)
            m = binary_dilate(foot, 4).astype(bool) & ~foot.astype(bool)
        elif t == "floodplain":
            m = np.asarray(ctx["env"]["flood"]).astype(bool)
        elif t == "steep":
            m = slope_from_elevation(np.asarray(ctx["elev"],
                                                np.float32)) > 25.0
        elif t == "riverside":
            water = np.asarray(ctx["env"]["water"]).astype(np.uint8)
            m = binary_dilate(water, 3).astype(bool)
        elif t == "everywhere":
            m = np.ones((g, g), dtype=bool)
        else:
            raise ValueError(f"unknown region {t!r}; known: {REGION_TOKENS}")
        out &= m
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Rule-based intent parser (the stand-in LLM)
# ----------------------------------------------------------------------------

_REGION_WORDS = {
    "north": "north", "northern": "north", "south": "south",
    "southern": "south", "east": "east", "eastern": "east", "west": "west",
    "western": "west", "centre": "centre", "center": "centre",
    "central": "centre", "middle": "centre", "downtown": "centre",
    "edge": "edge", "fringe": "edge", "outskirts": "edge",
    "periphery": "edge", "floodplain": "floodplain",
    "floodplains": "floodplain", "steep": "steep", "hillside": "steep",
    "riverside": "riverside", "river": "riverside",
    "waterfront": "riverside", "everywhere": "everywhere",
    "citywide": "everywhere"}
_ZONE_WORDS = {"residential": "residential", "housing": "residential",
               "homes": "residential", "commercial": "commercial",
               "retail": "commercial", "industrial": "industrial",
               "industry": "industrial", "factories": "industrial",
               "institutional": "institutional"}
_METRIC_WORDS = {"flood": "flood", "flooding": "flood",
                 "landslide": "landslide", "landslides": "landslide",
                 "green": "green_preserve", "greenspace": "green_preserve",
                 "access": "amenity", "accessibility": "amenity",
                 "walkability": "coverage", "coverage": "coverage",
                 "equity": "equity", "fairness": "equity",
                 "congestion": "congestion", "traffic": "congestion",
                 "infill": "infill", "sprawl": "infill",
                 "earthwork": "earthwork", "grading": "earthwork",
                 "circuity": "circuity", "efficiency": "efficiency"}
_AMEN_WORDS = ("school", "schools", "clinic", "bakery", "shop", "shops",
               "amenity", "centre", "center", "market", "playground")
_PROTECT = {"protect", "keep", "preserve", "save"}
_REGEN = {"regenerate", "redo", "rework", "rebuild"}
_RERANK = {"prioritize", "prioritise", "weight", "weigh", "emphasize"}
_POS = {"more", "add", "increase", "boost", "grow", "expand"}
_NEG = {"less", "reduce", "fewer", "decrease", "remove", "shrink"}
_AMEN_TRIG = {"add", "need", "needs", "want", "wants", "put", "build",
              "place"}


def parse_intent(text):
    """(ops, warnings). One op per clause (split on ,;. and 'and'), fixed
    precedence protect > rerank > regenerate > zone bias > amenity target.
    Unparseable clauses become warnings -- never hallucinated ops."""
    ops, warns = [], []
    for clause in re.split(r"[,;.]|\band\b", text.lower()):
        words = re.findall(r"[a-z]+", clause)
        if not words:
            continue
        wset = set(words)
        regions = []
        for w in words:
            r = _REGION_WORDS.get(w)
            if r and r not in regions:
                regions.append(r)
        region = " ".join(regions) or "everywhere"
        zone = next((_ZONE_WORDS[w] for w in words if w in _ZONE_WORDS), None)
        metric = next((_METRIC_WORDS[w] for w in words
                       if w in _METRIC_WORDS), None)
        cat = next((w for w in words if w in _AMEN_WORDS), None)
        if wset & _PROTECT:
            ops.append(Protect(region))
        elif wset & _RERANK and metric:
            base = WEIGHTS_V2.get(metric, WEIGHTS.get(metric, 0.1))
            ops.append(Rerank({metric: round(2.0 * base, 4)}))
        elif wset & _REGEN:
            ops.append(Regenerate(region))
        elif zone and (wset & _POS or wset & _NEG):
            ops.append(SetZoneBias(zone, -0.5 if wset & _NEG else 0.5,
                                   region))
        elif cat and wset & _AMEN_TRIG:
            if _REGION_WORDS.get(cat) in regions and len(regions) > 1:
                regions.remove(_REGION_WORDS[cat])
                region = " ".join(regions)
            ops.append(AddAmenityTarget(cat, region))
        else:
            warns.append(f"could not parse clause: {clause.strip()!r}")
    return ops, warns


# ----------------------------------------------------------------------------
# LLM intent parser (Ollama), strict-validated, rule fallback on ANY failure
# ----------------------------------------------------------------------------

_LLM_SYSTEM = """Translate a town planner's request into JSON operations.
Respond ONLY with JSON of the form {"ops": [...]}. Allowed ops:
  {"op":"Regenerate","region":R}
  {"op":"SetZoneBias","zone":Z,"weight":W,"region":R}   (W in [-0.9,4], +0.5=more, -0.5=less)
  {"op":"AddAmenityTarget","category":C,"region":R}     (C short free text, e.g. "school")
  {"op":"Protect","region":R}
  {"op":"Rerank","weights":{metric: weight}}
Z must be one of: residential commercial industrial institutional.
R is 1+ space-separated tokens from: north south east west centre edge
floodplain steep riverside everywhere.
Metrics: """ + " ".join(METRICS) + """
Never invent coordinates or geometry. Omit clauses you cannot map.
Examples:
"more commercial near the centre and protect the floodplain" ->
{"ops":[{"op":"SetZoneBias","zone":"commercial","weight":0.5,"region":"centre"},{"op":"Protect","region":"floodplain"}]}
"add a school in the north" ->
{"ops":[{"op":"AddAmenityTarget","category":"school","region":"north"}]}
"prioritize flood safety and redo the eastern edge" ->
{"ops":[{"op":"Rerank","weights":{"flood":0.2}},{"op":"Regenerate","region":"east edge"}]}"""


def parse_intent_llm(text, url="http://localhost:11434", model="llama3.2",
                     timeout=30):
    """(ops, warnings) via a local Ollama chat model at temperature 0.
    Every returned op is schema-validated; on ANY failure (connection, bad
    JSON, invalid op) falls back to parse_intent(text) with a warning."""
    try:
        body = json.dumps({
            "model": model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": _LLM_SYSTEM},
                         {"role": "user", "content": text}]}).encode()
        req = urllib.request.Request(
            url.rstrip("/") + "/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.loads(r.read())["message"]["content"]
        payload = json.loads(reply[reply.index("{"):reply.rindex("}") + 1])
        raw = payload["ops"] if isinstance(payload, dict) else payload
        ops = [op_from_dict(d) for d in raw]
        for op in ops:
            validate_op(op)
        return ops, []
    except Exception as err:  # noqa: BLE001 -- any failure means fallback
        ops, warns = parse_intent(text)
        warns.append(f"llm parse unavailable ({type(err).__name__}); "
                     "rule parser used")
        return ops, warns


# ----------------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------------

def rerank_scores(scores, weights, base_weights):
    """Re-sort {i: (total, sub)} under new metric weights. The delivery
    adjustment rank_samples baked into total is preserved as the ratio
    total / raw base-weighted score. Returns (order, new_scores)."""
    out = {}
    for i, (tot, sub) in scores.items():
        raw = 100.0 * sum(base_weights[k] * sub.get(k, 0.0)
                          for k in base_weights)
        deliver = tot / raw if raw > 1e-9 else 1.0
        new = 100.0 * sum(w * sub.get(k, 0.0) for k, w in weights.items())
        out[i] = (deliver * new, sub)
    order = sorted(out, key=lambda i: (-out[i][0], i))
    return order, out


def biased_zone_assign(dens_new, roads_all, elev, amen, clf, bias_ops, ctx,
                       d0=None, thr=DENSITY_THR):
    """assign_zones with per-class probability bias: predict_proba on the
    same feature stack, scale the biased zone's column by (1+weight) inside
    the region, then argmax. Mirrors zones.assign_zones pixel selection."""
    from zones import feature_stack
    new_dev = np.asarray(dens_new) > thr
    if d0 is not None:
        foot0 = binary_dilate((np.asarray(d0) > thr).astype(np.uint8),
                              2).astype(bool)
        new_dev = new_dev & ~foot0
    out = np.zeros(np.asarray(dens_new).shape, dtype=np.uint8)
    ys, xs = np.nonzero(new_dev)
    if len(ys) == 0:
        return out
    feats = feature_stack(elev, roads_all, dens_new, amen)
    proba = clf.predict_proba(feats[ys, xs])
    classes = np.asarray(clf.classes_)
    for op in bias_ops:
        sel = region_mask(op.region, ctx)[ys, xs] > 0.5
        col = np.nonzero(classes == ZONE_IDS[op.zone])[0]
        if len(col):
            proba[sel, col[0]] *= (1.0 + op.weight)
    out[ys, xs] = classes[np.argmax(proba, axis=1)].astype(np.uint8)
    return out


def execute(ops, ctx):
    """Route validated ops onto the existing tools. ctx keys: diff (or
    None), cond [4,G,G], roads0, d0, elev, env (dict or None), zclf (or
    None), amen_now, optionally candidates [(roads_all, dens), ...].
    Returns {"ops", "log", "keep_mask", "candidates_ranked", "warnings"}
    (+ "amenity_field"/"proposed_centres"/"zones" when produced). Every
    branch degrades gracefully and logs what it could not do."""
    log, warns = [], []
    res = {"ops": [o.to_dict() for o in ops], "log": log, "keep_mask": None,
           "candidates_ranked": None, "warnings": warns}
    good = []
    for op in ops:
        try:
            validate_op(op)
            good.append(op)
        except ValueError as err:
            warns.append(f"dropped invalid op {op.to_dict()}: {err}")
    ops = good
    d0 = np.asarray(ctx["d0"], np.float32)
    g = d0.shape[0]

    protected = np.zeros((g, g), dtype=bool)
    regen = np.zeros((g, g), dtype=bool)
    for op in ops:
        if isinstance(op, Protect):
            protected |= region_mask(op.region, ctx) > 0.5
            log.append(f"protect: {op.region}")
        elif isinstance(op, Regenerate):
            regen |= region_mask(op.region, ctx) > 0.5
            log.append(f"regenerate requested: {op.region}")
    if regen.any():
        res["keep_mask"] = ((~regen) | protected).astype(np.float32)
    elif protected.any():
        res["keep_mask"] = protected.astype(np.float32)

    candidates = None
    if regen.any():
        if ctx.get("diff") is None:
            log.append("no diffusion model in ctx; regeneration skipped")
        else:
            try:
                candidates = _inpaint_candidates(ctx, res["keep_mask"])
                log.append(f"inpainted {len(candidates)} candidates")
            except ImportError:
                log.append("inpaint unavailable")
            except Exception as err:  # noqa: BLE001
                warns.append(f"inpaint failed: {type(err).__name__}: {err}")
    if candidates is None:
        candidates = ctx.get("candidates")
        log.append("using ctx-provided candidates" if candidates
                   else "no candidates available; nothing to score")

    # amenity targets: smooth bump inside the region, then peak picking
    amen = np.array(ctx.get("amen_now") if ctx.get("amen_now") is not None
                    else np.zeros_like(d0), dtype=np.float32)
    amen_ops = [op for op in ops if isinstance(op, AddAmenityTarget)]
    if amen_ops:
        yy, xx = np.mgrid[0:g, 0:g]
        sig = g / 16.0
        for op in amen_ops:
            m = region_mask(op.region, ctx)
            if m.sum() == 0:
                warns.append(f"amenity region {op.region!r} is empty")
                continue
            cy = float((np.mgrid[0:g, 0:g][0] * m).sum() / m.sum())
            cx = float((np.mgrid[0:g, 0:g][1] * m).sum() / m.sum())
            bump = 0.4 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                / (2 * sig ** 2))
            amen = np.clip(amen + bump * (m > 0.5), 0.0, 1.0)
            log.append(f"amenity target {op.category!r} in {op.region} "
                       f"at ({cy:.0f},{cx:.0f})")
        res["amenity_field"] = amen
        res["proposed_centres"] = centers_from_density(amen, n_centers=3)

    env = ctx.get("env")
    full_env = env if isinstance(env, dict) and "amenities" in env else None
    if env is not None and full_env is None:
        log.append("env lacks amenity/green layers; legacy scorecard used")

    if candidates:
        order, scores = rank_samples(candidates, d0, ctx["elev"],
                                     env=full_env, roads0=ctx.get("roads0"))
        log.append(f"scored {len(candidates)} candidates; best index "
                   f"{order[0]}")
        rr = [op for op in ops if isinstance(op, Rerank)]
        if rr:
            base = dict(WEIGHTS_V2 if full_env is not None else WEIGHTS)
            new_w = dict(base)
            for op in rr:
                new_w.update(op.weights)
            missing = [k for k in new_w
                       if k not in next(iter(scores.values()))[1]]
            if missing:
                warns.append(f"rerank metrics not in active scorecard "
                             f"(count as 0): {missing}")
            order, scores = rerank_scores(scores, new_w, base)
            log.append(f"reranked with weights {new_w}; best index "
                       f"{order[0]}")
        res["candidates_ranked"] = {"order": order, "scores": scores}

    zb = [op for op in ops if isinstance(op, SetZoneBias)]
    if zb:
        if not candidates:
            log.append("zone bias skipped: no candidates")
        elif ctx.get("zclf") is None:
            log.append("zone bias skipped: zclf=None "
                       "(bias recorded in ops only)")
        elif not hasattr(ctx["zclf"], "predict_proba"):
            log.append("zone bias skipped: classifier has no predict_proba")
        else:
            best = res["candidates_ranked"]["order"][0]
            roads_b, dens_b = candidates[best]
            try:
                res["zones"] = biased_zone_assign(
                    dens_b, roads_b, ctx["elev"], amen, ctx["zclf"], zb,
                    ctx, d0=d0)
                log.append(f"zone bias applied on candidate {best}: "
                           + ", ".join(f"{o.zone}{o.weight:+.2f}@{o.region}"
                                       for o in zb))
            except Exception as err:  # noqa: BLE001
                warns.append(f"zone bias failed: {type(err).__name__}: "
                             f"{err}")
    return res


def _inpaint_candidates(ctx, keep_mask, n=8, steps=40):
    """RePaint n candidates; raises ImportError if inpaint.py is absent."""
    from inpaint import encode_known, sample_inpaint
    import torch
    diff = ctx["diff"]
    xk = encode_known(np.asarray(ctx["roads0"], np.float32),
                      np.asarray(ctx["d0"], np.float32),
                      ctx.get("amen_now"))
    cond = torch.as_tensor(np.asarray(ctx["cond"], np.float32),
                           device=diff.device)[None].repeat(n, 1, 1, 1)
    out = sample_inpaint(diff, cond, keep_mask, xk, steps=steps)
    out = ((out.cpu().numpy() + 1.0) / 2.0).astype(np.float32)
    r0 = (np.asarray(ctx["roads0"]) > 0.5)
    return [(np.maximum(out[i, 0] > 0.5, r0).astype(np.uint8), out[i, 1])
            for i in range(n)]


# ----------------------------------------------------------------------------
# Self-test: numpy-only, no torch, no network
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    G = GRID

    ops, w = parse_intent("we need more commercial near the centre "
                          "and protect the floodplain")
    assert [type(o) for o in ops] == [SetZoneBias, Protect] and not w
    assert ops[0].zone == "commercial" and ops[0].weight == 0.5
    assert ops[0].region == "centre" and ops[1].region == "floodplain"
    assert ops[1].to_dict() == {"op": "Protect", "region": "floodplain"}

    ops, w = parse_intent("add a school in the north")
    assert ops == [AddAmenityTarget("school", "north")] and not w
    ops, w = parse_intent("reduce industrial everywhere")
    assert ops == [SetZoneBias("industrial", -0.5, "everywhere")] and not w
    ops, w = parse_intent("prioritize flood safety")
    assert ops == [Rerank({"flood": 0.2})] and not w
    ops, w = parse_intent("regenerate the eastern edge")
    assert ops == [Regenerate("east edge")] and not w
    ops, w = parse_intent("preserve the riverside, and increase "
                          "residential in the west")
    assert ops == [Protect("riverside"),
                   SetZoneBias("residential", 0.5, "west")] and not w
    ops, w = parse_intent("we want a clinic near the centre")
    assert ops == [AddAmenityTarget("clinic", "centre")] and not w
    ops, w = parse_intent("purple monkey dishwasher")
    assert ops == [] and len(w) == 1
    for op in (op_from_dict({"op": "Regenerate", "region": "north"}),):
        validate_op(op)
    for bad in ({"op": "PaveEverything", "region": "north"},
                {"op": "SetZoneBias", "zone": "casino", "weight": 0.5,
                 "region": "north"},
                {"op": "Regenerate", "region": "north", "extra": 1}):
        try:
            validate_op(op_from_dict(bad))
            raise AssertionError(f"accepted bad op {bad}")
        except ValueError:
            pass
    print("parse_intent ok")

    yy, xx = np.mgrid[0:G, 0:G]
    d0 = (0.8 * np.exp(-((yy - 80) ** 2 + (xx - 64) ** 2)
                       / (2 * 12.0 ** 2))).astype(np.float32)
    elev = (yy * 2.0).astype(np.float32)
    env = {"flood": (yy > 100).astype(np.uint8),
           "water": (yy > 110).astype(np.uint8)}
    ctx = {"d0": d0, "elev": elev, "env": env}

    north = region_mask("north", ctx)
    assert north[:G // 2].min() == 1 and north[G // 2:].max() == 0
    cen = region_mask("centre", ctx)
    assert cen[80, 64] == 1 and cen[0, 0] == 0
    assert np.array_equal(region_mask("floodplain", ctx),
                          env["flood"].astype(np.float32))
    ee = region_mask("east edge", ctx)
    foot = d0 > DENSITY_THR
    assert ee.sum() > 0 and ee[:, :G // 2].max() == 0
    assert not (ee.astype(bool) & foot).any()
    try:
        region_mask("narnia", ctx)
        raise AssertionError("narnia accepted")
    except ValueError as err:
        assert "north" in str(err)
    print("region_mask ok")

    roads0 = np.zeros((G, G), np.uint8)
    roads0[80, :] = 1
    roads0[:, 64] = 1
    grow = (0.5 * np.exp(-((yy - 70) ** 2 + (xx - 70) ** 2)
                         / (2 * 8.0 ** 2))).astype(np.float32)
    ra = roads0.copy(); ra[70, 40:100] = 1
    far = (0.5 * np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2)
                        / (2 * 8.0 ** 2))).astype(np.float32)
    rb = roads0.copy(); rb[10, :] = 1; rb[:, 10] = 1
    cands = [(ra, np.clip(d0 + grow, 0, 1)),
             (rb, np.clip(d0 + far, 0, 1)),
             (roads0, d0.copy())]
    amen_now = np.zeros((G, G), np.float32)
    amen_now[100, 64] = 0.3
    ctx = {"diff": None, "cond": np.zeros((4, G, G), np.float32),
           "roads0": roads0, "d0": d0, "elev": elev, "env": env,
           "zclf": None, "amen_now": amen_now, "candidates": cands}
    ops = [Protect("floodplain"), Regenerate("north"),
           AddAmenityTarget("school", "north"),
           SetZoneBias("commercial", 0.5, "centre"),
           Rerank({"coverage": 0.8, "earthwork": 0.05})]
    res = execute(ops, ctx)
    km = res["keep_mask"]
    assert km is not None and km[:G // 2].max() == 0 and km[G // 2:].min() == 1
    ay, ax = divmod(int(np.argmax(res["amenity_field"])), G)
    assert ay < G // 2, "amenity argmax did not move into the north"
    assert res["proposed_centres"]
    rk = res["candidates_ranked"]
    assert sorted(rk["order"]) == [0, 1, 2]
    assert all(np.isfinite(rk["scores"][i][0]) for i in rk["order"])
    assert any("zone bias skipped" in ln for ln in res["log"])
    assert any("reranked" in ln for ln in res["log"])
    assert any("no diffusion model" in ln for ln in res["log"])
    assert len(res["ops"]) == 5 and res["log"]
    fake = {0: (50.0, {"coverage": 0.1, "infill": 0.9}),
            1: (40.0, {"coverage": 0.9, "infill": 0.1})}
    order2, sc2 = rerank_scores(fake, {"coverage": 0.9, "infill": 0.1},
                                {"coverage": 0.5, "infill": 0.5})
    assert order2 == [1, 0] and sc2[1][0] > sc2[0][0]
    print("execute ok")

    ops_llm, w_llm = parse_intent_llm("add a school in the north",
                                      url="http://127.0.0.1:1", timeout=2)
    assert ops_llm == [AddAmenityTarget("school", "north")]
    assert any("rule parser used" in x for x in w_llm)
    print("parse_intent_llm fallback ok")
