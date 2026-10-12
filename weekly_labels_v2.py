"""Label approach, second version. weekly_labels.py (v1) is untouched and still selectable; this one adds what the
first round of evaluation (2026-10-04) showed the labels were missing:

  - "no workouts" is a thing that NEEDS ATTENTION (v1 left it as one line among many, so the model ignored it);
  - an explicit "keep as they are, do not ask for more" list (v1 let the model tell the user to eat more protein
    when protein was already met);
  - the SIZE of each gap in words ("about a fifth short") and the DIRECTION to move ("eat less"), so advice for a
    far-off metric is not thin and does not point the wrong way;
  - an "advice goal" line that says what good advice looks like this week, including "protect the routine" when
    nothing is wrong (v1 invented generic habits for a good week);
  - several wordings per verdict, picked by rule, so the model stops copying one stock phrase.

Same contract as v1: pure functions, no model, no database, and the text it returns has no digits.
"""
from __future__ import annotations

import re

import weekly_labels as v1

KEYWORDS = {**v1.KEYWORDS,
            "workouts": re.compile(r"\b(workouts?|training|sessions?|strength|gym|lifting|weights?|exercise|resistance)\b", re.I)}

_TITLES = {"steps": "Steps", "calories": "Calories", "protein": "Protein"}
_ORDER = ("steps", "protein", "calories")

# verdict -> alternative wordings. Which one a metric gets is fixed by the metric (so the same week always reads the
# same), but different metrics in one week get different ones.
_VARIANTS = {
    "far below the goal": ["far short of the goal", "way under the goal", "nowhere near the goal"],
    "well below the goal": ["well short of the goal", "clearly under the goal", "a good way under the goal"],
    "a little below the goal": ["just short of the goal", "slightly under the goal", "a touch under the goal"],
    "right at the goal, just above it": ["on the goal", "meeting the goal", "just past the goal"],
    "well above the goal": ["comfortably past the goal", "well beyond the goal", "clear of the goal"],
    "right on the target": ["bang on the target", "matching the target", "spot on the target"],
    "slightly over the target": ["a touch over the target", "just above the target", "marginally over the target"],
    "slightly under the target": ["a touch under the target", "just below the target", "marginally under the target"],
    "clearly over the target": ["well over the target", "clearly above the target", "a good way over the target"],
    "far over the target": ["way over the target", "far above the target", "miles over the target"],
    "clearly under the target": ["well under the target", "clearly below the target", "a good way under the target"],
    "far under the target": ["way under the target", "far below the target", "miles under the target"],
}

# how big a gap is, in words (fraction of the goal/target)
_GAP_WORDS = [(0.05, "only a few percent"), (0.12, "about a tenth"), (0.2, "about a fifth"),
              (0.3, "about a quarter"), (0.4, "about a third"), (0.6, "about half"), (9.9, "more than half")]


def _gap_words(fraction: float) -> str:
    f = abs(fraction)
    return next(words for limit, words in _GAP_WORDS if f < limit)


def _wording(metric: str, level: str) -> str:
    options = _VARIANTS.get(level)
    return options[_ORDER.index(metric) % len(options)] if options else level


def _gap_fraction(data: dict, metric: str) -> float:
    goal = data["goals"].get({"steps": "step_goal", "protein": "protein_target", "calories": "calories_target"}[metric], 0)
    return (data[metric]["avg"] - goal) / goal if goal else 0.0


def classify(data: dict) -> dict:
    """v1's verdicts plus the size of each gap, how to move, the workout verdict and the ordering of the problems."""
    lab = v1.classify(data)
    for key in _ORDER:
        m = lab[key]
        if m["status"] == "nodata":
            continue
        frac = _gap_fraction(data, key)
        m["wording"] = _wording(key, m["level"])
        m["gap"] = _gap_words(frac) if m["status"] in ("below", "over", "under") else ""
        m["size"] = abs(frac)
        m["direction"] = {"below": "do more" if key == "steps" else "eat more",
                          "over": "eat less", "under": "eat more"}.get(m["status"], "keep as it is")
    lab["no_workouts"] = not (data.get("workouts") or [])
    return lab


def attention(lab: dict) -> list[str]:
    """What needs attention, biggest gap first, then the workout problem."""
    items = sorted(v1.off_target(lab), key=lambda k: -lab[k]["size"])
    return items + (["workouts"] if lab["no_workouts"] else [])


def label_summary(data: dict) -> str:
    """The text block the model reads. It has no digits at all."""
    lab = classify(data)
    lines = [f"Phase: {lab['phase']}", ""]
    for key in ("steps", "calories", "protein"):
        m = lab[key]
        if m["status"] == "nodata":
            lines.append(f"{_TITLES[key]}: no data this week.")
            continue
        gap = f" ({m['gap']} off)" if m["gap"] else ""
        lines.append(f"{_TITLES[key]}: {m['wording']}{gap}; {m['trend']}; {m['consistency']}.")
    lines.append("")

    needs = attention(lab)
    detail = lambda k: "workouts (none this week)" if k == "workouts" else f"{k} ({lab[k]['wording']}, {lab[k]['gap']} off)"
    lines.append("Needs attention, biggest first: " + (", ".join(detail(k) for k in needs) if needs else "nothing") + ".")
    fine = v1.on_track(lab)
    lines.append("Keep as they are, do not ask for more: " + (", ".join(fine) if fine else "nothing") + ".")
    missing = v1.no_data(lab)
    if missing:
        lines.append("No data this week (say nothing about these): " + ", ".join(missing) + ".")
    lines.append("")

    moves = [f"{k}: {lab[k]['direction']}" for k in ("steps", "calories", "protein") if lab[k]["status"] != "nodata"]
    lines.append("Direction for any advice: " + "; ".join(moves) + ".")
    if needs:
        lines.append(f"Advice goal: close the biggest gap first ({needs[0]}), then the others, and leave everything "
                     "else exactly as it is.")
    else:
        lines.append("Advice goal: nothing is wrong, so protect the routine that is working. Do not invent problems "
                     "and do not suggest new habits for their own sake.")
    lines.append("")
    lines.append("Workouts this week: none, which needs attention." if lab["no_workouts"] else f"Workouts this week: {lab['workouts']}.")
    lines.append(f"Days calories landed almost exactly on the target (this says nothing about over or under): {lab['goal_days']}.")
    lines.append(f"Most active time of day: {lab['peak']}." if lab["peak"] else "Most active time of day: no data yet.")
    return "\n".join(lines)
