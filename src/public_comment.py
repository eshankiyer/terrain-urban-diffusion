"""Public comment intake and synthesis.

Planning departments receive written comments on proposals and someone
has to read all of them, count who supports what, and tell the hearing
body which requests came up often enough to matter. This module does the
counting half of that job. It tags each comment with a stance and a set
of topics, aggregates across the batch, and converts frequently repeated
requests into validated operations from planner_agent's closed schema,
so public input can steer the same tools a planner steers by hand.

The stance and topic tagging is lexicon based, deliberately. A lexicon
is inspectable, fails loudly on vocabulary it does not know, and costs
nothing at inference time. It also inherits the limits of every lexicon:
sarcasm and comments written in languages other than English will be
tagged neutral or wrong. Simple negation is handled: clauses containing
an explicit negation marker (not, n't, never, "no longer") are dropped
before op extraction, with a warning, and concern words preceded by
"no" or "without" do not count toward the concern stance. Negation that
crosses a clause boundary is still missed. The aggregate counts are
therefore a reading aid for a human, not a vote tally, and the module
says so in its own output.

Ops extraction reuses planner_agent.parse_intent, which never invents
geometry and drops what it cannot map. A request only becomes an op when
at least min_mentions distinct comments produce the same op, which keeps
one loud comment from steering the plan. Comments that are identical
after text normalization count once toward op mentions, so a
copy-pasted campaign cannot inflate the tally.

Dependencies: numpy transitively via planner_agent; nothing else.
"""

import json
import re
from collections import Counter

from planner_agent import parse_intent, validate_op

STANCES = ("support", "oppose", "concern", "neutral")

_SUPPORT = {"support", "supports", "favor", "favour", "approve", "approves",
            "welcome", "welcomes", "yes", "excited", "glad", "great",
            "benefit", "benefits", "improve", "improves", "needed"}
_OPPOSE = {"oppose", "opposes", "against", "reject", "rejects",
           "stop", "halt", "terrible", "awful", "ruin", "ruins", "destroy",
           "destroys", "overdevelopment", "unacceptable"}
_CONCERN = {"concern", "concerns", "concerned", "worried", "worry",
            "worries", "afraid", "fear", "fears", "risk", "risks",
            "problem", "problems", "unsure", "question", "questions"}

_TOPIC_WORDS = {
    "traffic": "traffic", "congestion": "traffic", "parking": "traffic",
    "flood": "flooding", "flooding": "flooding", "drainage": "flooding",
    "stormwater": "flooding",
    "school": "schools", "schools": "schools",
    "housing": "housing", "homes": "housing", "apartments": "housing",
    "affordable": "housing", "density": "housing",
    "park": "green space", "parks": "green space", "green": "green space",
    "trees": "green space", "playground": "green space",
    "walk": "walkability", "walkable": "walkability",
    "walkability": "walkability", "sidewalk": "walkability",
    "sidewalks": "walkability", "bike": "walkability",
    "noise": "nuisance", "lights": "nuisance", "smell": "nuisance",
    "industrial": "industrial use", "factory": "industrial use",
    "factories": "industrial use",
    "commercial": "commercial use", "retail": "commercial use",
    "shops": "commercial use", "business": "commercial use",
    "character": "neighbourhood character", "historic":
    "neighbourhood character", "views": "neighbourhood character",
}


def _words(text):
    out, cur = [], []
    for ch in str(text).lower():
        if ch.isalpha():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|no\s+more|no\s+new)\b|n't",
    re.IGNORECASE)


def _drop_negated_clauses(text):
    """Remove clauses that contain an explicit negation marker.

    parse_intent cannot see negation, so "do not protect the hillside"
    would otherwise invert into a Protect op, and "no more factories"
    would invert into a bias toward industrial because "more" is a
    positive trigger. Clauses (split on . , ; ! and ?) containing not,
    n't, never, "no longer", "no more", or "no new" are dropped before
    op extraction. Returns (kept_text, n_dropped).
    """
    raw = str(text)
    if not _NEGATION_RE.search(raw):
        return raw, 0
    kept, dropped = [], 0
    for clause in re.split(r"[.;!?,]", raw):
        if not clause.strip():
            continue
        if _NEGATION_RE.search(clause):
            dropped += 1
        else:
            kept.append(clause.strip())
    return ". ".join(kept), dropped


def analyze_comment(text):
    """Tag one comment. Returns {"stance", "topics", "ops", "warnings"}.

    Stance is the label whose lexicon matches most words, with ties
    resolved concern > oppose > support, since misfiling an objection as
    support is the costlier error in a hearing packet. Concern words
    preceded by "no" or "without" are suppressed, and "no concern(s)" or
    "no problem(s)" counts as support. Topics are the matched topic
    labels in order of first appearance. Ops come from parse_intent,
    applied after negated clauses are dropped, and are already
    validated.
    """
    words = _words(text)
    sup_hits, opp_hits, con_hits = set(), set(), set()
    for i, w in enumerate(words):
        prev = words[i - 1] if i else ""
        if w in _CONCERN:
            if prev in ("no", "without"):
                if prev == "no" and w in ("concern", "concerns",
                                          "problem", "problems"):
                    sup_hits.add(prev + " " + w)
                continue
            con_hits.add(w)
        if w in _OPPOSE:
            opp_hits.add(w)
        if w in _SUPPORT:
            sup_hits.add(w)
    n_sup, n_opp, n_con = len(sup_hits), len(opp_hits), len(con_hits)
    best = max(n_con, n_opp, n_sup)
    if best == 0:
        stance = "neutral"
    elif n_con == best:
        stance = "concern"
    elif n_opp == best:
        stance = "oppose"
    else:
        stance = "support"
    topics, seen = [], set()
    for w in words:
        t = _TOPIC_WORDS.get(w)
        if t and t not in seen:
            seen.add(t)
            topics.append(t)
    kept, n_neg = _drop_negated_clauses(text)
    if kept.strip():
        ops, warns = parse_intent(kept)
    else:
        ops, warns = [], []
    warns = list(warns)
    if n_neg:
        warns.append(f"{n_neg} negated clause(s) skipped during op "
                     "extraction")
    return {"stance": stance, "topics": topics, "ops": ops,
            "warnings": warns}


def synthesize(comments, min_mentions=2):
    """Aggregate a batch of comments for a hearing packet.

    Returns a dict with the batch size, stance counts, topic counts, the
    min_mentions threshold used, the ops requested by at least
    min_mentions distinct comments (validated, deduplicated by their
    JSON form), and per-comment tags for audit. Comments identical after
    text normalization count once toward op mentions; stance and topic
    counts still cover every submission.
    """
    tagged = [analyze_comment(c) for c in comments]
    stance_counts = Counter(t["stance"] for t in tagged)
    topic_counts = Counter(tp for t in tagged for tp in t["topics"])
    op_counts = Counter()
    op_by_key = {}
    seen_norm = set()
    for c, t in zip(comments, tagged):
        norm = " ".join(_words(c))
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        seen_this_comment = set()
        for op in t["ops"]:
            key = json.dumps(op.to_dict(), sort_keys=True)
            if key in seen_this_comment:
                continue
            seen_this_comment.add(key)
            op_counts[key] += 1
            op_by_key[key] = op
    frequent = []
    for key, n in op_counts.most_common():
        if n < min_mentions:
            continue
        op = op_by_key[key]
        validate_op(op)
        frequent.append({"op": op, "mentions": n})
    return {"n_comments": len(comments),
            "stances": dict(stance_counts),
            "topics": dict(topic_counts),
            "min_mentions": min_mentions,
            "frequent_ops": frequent,
            "tagged": tagged}


def ops_from_comments(comments, min_mentions=2):
    """The bridge into the executor: ops requested repeatedly, validated."""
    return [f["op"] for f in synthesize(comments, min_mentions)
            ["frequent_ops"]]


def to_markdown(summary):
    """Hearing-packet section. States the method's limits in its own text."""
    lines = [f"## Public comment summary ({summary['n_comments']} "
             "comments received)", ""]
    st = summary["stances"]
    parts = [f"{st.get(s, 0)} {s}" for s in STANCES if st.get(s, 0)]
    if parts:
        lines.append("Stance tagging (lexicon based, indicative only): "
                     + ", ".join(parts) + ".")
    if summary["topics"]:
        top = sorted(summary["topics"].items(), key=lambda kv: -kv[1])
        lines.append("Topics raised most often: "
                     + ", ".join(f"{k} ({v})" for k, v in top[:6]) + ".")
    if summary["frequent_ops"]:
        mm = summary.get("min_mentions", 2)
        lines.append("")
        lines.append(f"Requests raised by {mm} or more commenters, "
                     "mapped to plan operations:")
        for f in summary["frequent_ops"]:
            d = f["op"].to_dict()
            body = ", ".join(f"{k}={v}" for k, v in d.items() if k != "op")
            lines.append(f"- {d['op']} ({body}); {f['mentions']} mentions")
    lines.append("")
    lines.append("Automated tagging misses sarcasm, cross-clause negation, "
                 "and non-English comments. Staff read of the full record "
                 "is still required.")
    return "\n".join(lines)


if __name__ == "__main__":
    batch = [
        "I support the new housing but I'm worried about traffic on Elm.",
        "Please protect the floodplain, we flood every spring.",
        "Protect the floodplain! My basement has flooded twice.",
        "We need a school in the north part of town.",
        "Please add a school in the north.",
        "Against this. It will ruin the neighbourhood character.",
        "No more factories near homes. Reduce industrial everywhere.",
        "Sidewalks please, the kids walk to the park.",
        "purple monkey dishwasher",
        "Great plan overall, glad to see the greenspace kept.",
    ]
    s = synthesize(batch, min_mentions=2)
    assert s["n_comments"] == 10
    assert s["min_mentions"] == 2
    assert s["stances"].get("concern", 0) >= 1
    assert s["stances"].get("oppose", 0) >= 1
    assert s["stances"].get("support", 0) >= 1
    assert s["topics"]["flooding"] >= 1 and s["topics"]["schools"] >= 2
    freq = {f["op"].to_dict()["op"] for f in s["frequent_ops"]}
    assert "Protect" in freq, freq
    assert "AddAmenityTarget" in freq, freq
    ops = ops_from_comments(batch)
    assert any(o.to_dict()["op"] == "Protect" and
               o.to_dict()["region"] == "floodplain" for o in ops)
    md = to_markdown(s)
    assert "Public comment summary (10" in md
    assert "2 or more commenters" in md
    assert chr(0x2014) not in md
    one = analyze_comment("I am concerned about flooding and traffic.")
    assert one["stance"] == "concern"
    assert one["topics"] == ["flooding", "traffic"]
    neutral = analyze_comment("The meeting is on Tuesday.")
    assert neutral["stance"] == "neutral" and not neutral["topics"]

    # "no" alone is no longer treated as opposition
    calm = analyze_comment("No more factories near homes.")
    assert calm["stance"] != "oppose", calm["stance"]

    # negated request must not invert into an op, and must warn
    neg = analyze_comment("Please do not protect the floodplain.")
    assert not any(o.to_dict().get("op") == "Protect" for o in neg["ops"])
    assert any("negated" in w for w in neg["warnings"]), neg["warnings"]

    # "no concerns" reads as support or neutral, never concern
    nc = analyze_comment("I have no concerns about this plan.")
    assert nc["stance"] in ("support", "neutral"), nc["stance"]

    # duplicated comments count once toward op mentions
    dup = ["Protect the floodplain.", "Protect the floodplain.",
           "protect  THE floodplain"]
    sd = synthesize(dup, min_mentions=2)
    assert not sd["frequent_ops"], sd["frequent_ops"]

    # empty batch renders without a dangling stance sentence
    empty = synthesize([], min_mentions=3)
    assert empty["min_mentions"] == 3
    md_empty = to_markdown(empty)
    assert "Public comment summary (0" in md_empty
    assert "Stance tagging" not in md_empty
    print("public_comment self-tests passed")
