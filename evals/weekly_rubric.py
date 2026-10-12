"""The checklist a weekly review is scored against. Pure functions: no model, no database.

Four kinds of item, each a yes/no:

  hallucination  invented numbers; a recommendation that contradicts a target status; claims about a metric that has
                 NO data this week
  instruction    the format rules in the prompt: exactly two paragraphs, no markdown, no label/heading, no greeting or
                 sign-off, about 4 lines each, no advice in the observations, 1-3 numbered recommendations
  coverage       the observations restate each metric's average; the recommendations address every metric that is
                 off target (so a review cannot quietly skip the problem)
  specificity    every recommendation names a concrete action (an amount, a time or a habit), and none is filler
                 like "stay consistent"

What counts as "off target" is DERIVED from each case's data, so a new case needs no extra bookkeeping.

This measures what a checklist can measure. It cannot tell whether advice is wise; that needs a person reading it
(evals/blind_compare.py) or real-world feedback (the thumbs buttons).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import weekly_analysis as wa

KINDS = ("hallucination", "instruction", "coverage", "specificity")


@dataclass
class Item:
    kind: str
    id: str
    ok: bool
    detail: str = ""


# ── what the data says ───────────────────────────────────────────────────────

CALORIE_TOLERANCE = 0.08       # more than 8% away from the calorie target counts as off target


def metric_state(data: dict) -> dict:
    """{'steps': 'met'|'below'|'nodata', 'protein': ..., 'calories': 'on'|'over'|'under'|'nodata'}"""
    g = data["goals"]
    out = {}
    for key, goal_key in (("steps", "step_goal"), ("protein", "protein_target")):
        m, goal = data[key], g.get(goal_key, 0)
        out[key] = "nodata" if (m.get("status") != "ok" or not goal) else ("met" if m["avg"] >= goal else "below")
    c, target = data["calories"], g.get("calories_target", 0)
    if c.get("status") != "ok" or not target:
        out["calories"] = "nodata"
    else:
        rel = (c["avg"] - target) / target
        out["calories"] = "over" if rel > CALORIE_TOLERANCE else "under" if rel < -CALORIE_TOLERANCE else "on"
    return out


# ── text helpers ─────────────────────────────────────────────────────────────

def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


_GREETING = re.compile(r"^\s*(hi|hey|hello|dear|good (morning|evening)|great (job|week))\b", re.I)
_SIGNOFF = re.compile(r"(let me know|good luck|keep (it up|crushing)|you('ve| have) got this|cheers|best,)", re.I)
_LABEL = re.compile(r"^\s*(\*\*)?(observations?|recommendations?|analysis|summary|stage \d)\s*[:\-]", re.I | re.M)
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|`|^\s*[-*]\s+\S)", re.M)
_ADVICE = re.compile(r"\b(recommend|should|consider|try to|aim to|i suggest|suggest that|you could|make sure|focus on)\b", re.I)
_GENERIC = re.compile(r"(stay consistent|keep it up|keep up the (good|great) work|do your best|keep pushing|"
                      r"stay (the course|motivated|focused)|maintain consistency|continue (to )?(do|doing) what)", re.I)
_ANCHOR = re.compile(r"(\d|minutes?|hours?|daily|each day|every day|per day|a day|weekly|each week|twice|once|"
                     r"morning|evening|afternoon|lunch|dinner|breakfast|snack|before|after|bedtime|"
                     r"gram|serving|portion|scoop|shake|walk|session|workout|"
                     r"meat|dairy|source|fish|egg|chicken|yogurt|vegetable|stairs|jog|run|cycle|swim|gym|lift|sleep|"
                     r"water|track|log|weigh|measure|plan|prep|schedule|reminder|break)", re.I)
_KEYWORDS = {
    "steps": re.compile(r"\b(steps?|walk\w*|move|moving|activity|active|stairs|cardio)\b", re.I),
    "protein": re.compile(r"\b(protein|meals?|shake|eggs?|chicken|fish|yogurt|whey|lean)\b", re.I),
    "calories": re.compile(r"\b(calor\w*|kcal|intake|portions?|eat\w*|deficit|surplus|food|meals?)\b", re.I),
}
_NO_DATA_CLAIM = re.compile(r"\bprotein\b[^.]{0,70}\b(met|exceed\w*|below|short|lack\w*|low|high|under|over|steady|consistent\w*|"
                            r"adequate|enough|hit|missed|average\w*)\b", re.I)


def numbered_items(rec_paragraph: str) -> list[str]:
    """Split 'text 1. foo 2. bar' into ['foo', 'bar'] (accepts '1.' and '1)')."""
    parts = re.split(r"(?:(?<=\s)|^)(\d)[.)]\s+", rec_paragraph)
    # parts = [pre, '1', text1, '2', text2, ...]
    items = []
    for i in range(1, len(parts) - 1, 2):
        items.append((int(parts[i]), parts[i + 1].strip()))
    return items


# ── the checklist ────────────────────────────────────────────────────────────

def score(text: str, data: dict, mode: str = "numbers") -> list[Item]:
    """Every item for one answer. `text` is the final joined review.

    mode "numbers" is the original checklist. mode "labels" is for a review written from weekly_labels.py's words:
    the model was never shown a figure, so ANY digit in the observations is invented, the averages cannot be
    restated, and the contradiction check also covers calories (eating more when over, less when under)."""
    # labels2 / labels3 = weekly_labels_v2.py / _v3.py: the label checks plus the workout one; labels3 also checks that
    # recommendation one takes on the first priority.
    labels = mode.startswith("labels")             # every label mode, so a new version can't be scored as 'numbers'
    # mode "matched6" (evals/matched_numeric_weekly.py, ablation A9): labels v6's checklist, except that the model was
    # shown the figures, so digits are allowed in the observations and a figure must be one of the figures shown.
    matched = mode == "matched6"
    state = metric_state(data)
    summary = wa.format_data_summary(data)
    if matched:
        from evals import matched_numeric_weekly
    shown = (matched_numeric_weekly.shown_block(data) if matched
             else wa._label4_facts_block(data) if mode in ("labels4", "labels5", "labels6")
             else wa._label3_facts_block(data) if mode == "labels3"
             else wa._label2_facts_block(data) if mode == "labels2"
             else wa._label_facts_block(data) if labels else wa._facts_block(data, summary))
    paras = paragraphs(text)
    obs = paras[0] if paras else ""
    recs = paras[1] if len(paras) > 1 else ""
    out: list[Item] = []

    # hallucination
    # Observations only describe the week, so every number in them must be real. In the recommendations a
    # PROPOSAL ("a 20 minute walk") is advice, not a fact. And only the recommendations can contradict a target.
    bad = wa._grounding_check(obs, shown) + wa._grounding_check(recs, shown, allow_proposals=True)
    if len(paras) != 2:                                    # unparseable: check the whole text, strictly
        bad = wa._grounding_check(text, shown)
    out.append(Item("hallucination", "no_invented_numbers", not bad, f"not in the data: {sorted(set(bad))}" if bad else ""))
    check = wa._label_directional_check if labels or matched else wa._directional_check
    contradictions = check(recs if len(paras) == 2 else text, data)
    out.append(Item("hallucination", "respects_target_status", not contradictions, "; ".join(contradictions)))
    if labels:
        digits = re.findall(r"\d[\d,.]*", obs if len(paras) == 2 else text)
        out.append(Item("hallucination", "observations_use_words_not_digits", not digits, f"digits: {digits}" if digits else ""))
    if state["protein"] == "nodata":
        claim = _NO_DATA_CLAIM.search(text)
        out.append(Item("hallucination", "no_claims_without_data", not claim,
                        f"talks about protein as if measured: {claim.group(0)!r}" if claim else ""))

    # instruction following
    out.append(Item("instruction", "two_paragraphs", len(paras) == 2, f"{len(paras)} paragraphs"))
    out.append(Item("instruction", "no_markdown", not _MARKDOWN.search(text)))
    out.append(Item("instruction", "no_label_or_heading", not _LABEL.search(text)))
    out.append(Item("instruction", "no_greeting", not _GREETING.search(text)))
    out.append(Item("instruction", "no_sign_off", not _SIGNOFF.search(paras[-1] if paras else text)))
    lens = [len(p) for p in paras[:2]]
    out.append(Item("instruction", "about_four_lines_each", bool(lens) and all(150 <= n <= 700 for n in lens), f"lengths {lens}"))
    out.append(Item("instruction", "observations_hold_no_advice", not (len(paras) == 2 and _ADVICE.search(obs)),
                    (_ADVICE.search(obs).group(0) if len(paras) == 2 and _ADVICE.search(obs) else "")))
    items = numbered_items(recs)
    numbered_ok = bool(items) and items[0][0] == 1 and 1 <= len(items) <= 3 and [n for n, _ in items] == list(range(1, len(items) + 1))
    out.append(Item("instruction", "one_to_three_numbered_recommendations", numbered_ok, f"found {[n for n, _ in items]}"))

    # coverage
    missing_numbers = []
    for key, goal_key in (("steps", None), ("calories", None), ("protein", None)):
        if state[key] == "nodata":
            continue
        if labels or matched:                         # no figures to restate: the metric just has to be discussed
            if not _KEYWORDS[key].search(obs):
                missing_numbers.append(key)
            continue
        avg = data[key]["avg"]
        found = wa._extract_numbers(obs)
        if not any(abs(f - avg) <= 0.015 * avg + 0.5 for f in found):
            missing_numbers.append(key)
    out.append(Item("coverage", "observations_cover_each_metric" if labels or matched else "observations_restate_each_average",
                    not missing_numbers,
                    (f"not discussed: {missing_numbers}" if labels or matched else f"average missing for: {missing_numbers}")
                    if missing_numbers else ""))
    unaddressed = []
    for key in ("steps", "protein"):
        if state[key] == "below" and not _KEYWORDS[key].search(recs):
            unaddressed.append(key)
    if state["calories"] in ("over", "under") and not _KEYWORDS["calories"].search(recs):
        unaddressed.append("calories")
    out.append(Item("coverage", "recommendations_address_every_off_target_metric", not unaddressed,
                    f"not addressed: {unaddressed}" if unaddressed else ""))

    if mode in ("labels3", "labels4", "labels5", "labels6", "matched6"):   # v4-v6 use v3's verdicts and priorities
        import weekly_labels_v3
        needs = weekly_labels_v3.attention(weekly_labels_v3.classify(data))
        first = wa._first_numbered_item(recs)
        ok_first = not needs or bool(first and weekly_labels_v3.KEYWORDS[needs[0]].search(first))
        out.append(Item("coverage", "first_recommendation_takes_the_first_priority", ok_first,
                        "" if ok_first else f"first priority is {needs[0]}; recommendation one: {first[:60]!r}"))
    if mode in ("labels2", "labels3", "labels4", "labels5", "labels6", "matched6") and not data.get("workouts"):   # no workouts
        import weekly_labels_v2
        kw = weekly_labels_v2.KEYWORDS["workouts"]
        missing = [name for name, part in (("observations", obs), ("recommendations", recs)) if not kw.search(part)]
        out.append(Item("coverage", "no_workouts_is_noticed_and_addressed", not missing,
                        f"not mentioned in: {missing}" if missing else ""))

    # specificity
    vague =[t[:50] for _n, t in items if not _ANCHOR.search(t)]
    out.append(Item("specificity", "every_recommendation_is_concrete", bool(items) and not vague,
                    f"no concrete action in: {vague}" if vague else ("" if items else "no numbered recommendations")))
    filler = _GENERIC.search(text)
    out.append(Item("specificity", "no_filler_phrases", not filler, filler.group(0) if filler else ""))
    return out


def failed(items: list[Item]) -> list[Item]:
    return [i for i in items if not i.ok]


def summarize(items: list[Item]) -> dict:
    """{kind: (passed, total)} plus 'all': True when every item passed."""
    by = {k: [0, 0] for k in KINDS}
    for i in items:
        by[i.kind][1] += 1
        by[i.kind][0] += i.ok
    return {**{k: tuple(v) for k, v in by.items()}, "all": all(i.ok for i in items)}
