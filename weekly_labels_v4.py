"""Label approach, fourth version: the same meaning as v3, but each QUESTION gets its own VIEW of it.

v1-v3 (weekly_labels.py, _v2.py, _v3.py) are untouched and still selectable. v3 gave the model what each gap means,
and the advice it wrote was the best so far (10/10 blind wins on the holdout weeks, semantic items 41/42), but its
OBSERVATIONS degraded. Reading them showed why:

  - the observations question was shown the priorities, the kinds of fix and the keep-as-is list, so the description
    drifted into advice ("the focus should remain on protein", "keep calories as they are");
  - each meaning sat in its own "What it means:" sentence, and the model sometimes attached it to the wrong metric
    ("protein was under, which means less movement ...");
  - the headline started in lower case and was copied verbatim.

v4 therefore splits the description into two views:
  observations_view(data)     what happened and what it means: one sentence per metric with its meaning INLINE,
                              the workouts, anything worth a mention, the only true links. No priorities, no fixes,
                              no keep list, nothing that asks for action.
  recommendations_view(data)  what to work on, in order, each with its size, its meaning and the kind of fix; what to
                              leave unchanged (as a constraint, not an item to recommend); what to guard against.

The verdicts themselves (classify, attention, links, watch) are v3's: v4 changes only how they are presented, so any
difference between v3 and v4 is due to presentation. Same contract: pure functions, no digits in the output.
"""
from __future__ import annotations

import weekly_labels as v1
import weekly_labels_v3 as v3

KEYWORDS = v3.KEYWORDS
classify = v3.classify
attention = v3.attention
links = v3.links
watch = v3.watch

_TITLES = {"steps": "Steps", "calories": "Calories", "protein": "Protein"}
_ORDINALS = ["First", "Then", "Last"]


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def main_point(lab: dict) -> str:
    return _sentence(v3._headline(lab, attention(lab))) + "."


def _metric_line(key: str, lab: dict) -> str:
    """One sentence per metric, its meaning in the same sentence so it cannot drift to another metric."""
    m = lab[key]
    if m["status"] == "nodata":
        return f"{_TITLES[key]}: no data this week."
    gap = f", a {m['severity']} gap ({m['gap']} off)" if m.get("severity") else ""
    parts = f"{m['wording']}{gap}, {m['trend_meaning']}"
    if m.get("consistency_meaning"):
        parts += f", and {m['consistency_meaning']}"
    if m.get("meaning"):
        parts += f"; for the goal: {m['meaning']}"
    return f"{_TITLES[key]}: {parts}."


def observations_view(data: dict) -> str:
    """What happened and what it means. Nothing in it asks for action."""
    lab = classify(data)
    lines = [f"Phase: {lab['phase']}", f"Main point: {main_point(lab)}", ""]
    lines += [_metric_line(k, lab) for k in ("steps", "calories", "protein")]
    w = lab["workout_meaning"]
    lines.append(f"Workouts: {lab['workouts']}" + (f"; {w}" if w else "") + ".")
    if lab["calories"]["status"] == "on" and lab["goal_days"] != "not enough data yet":
        lines.append(f"Calories landed close to the target {lab['goal_days']}.")
    early = watch(lab)
    if early:
        lines.append("Worth a mention (fine on average, but): " + "; ".join(f"{k}, {why}" for k, why in early) + ".")
    missing = v1.no_data(lab)
    if missing:
        lines.append("No data this week (say nothing about it): " + ", ".join(missing) + ".")
    found = links(lab)
    lines.append("")
    if found:
        lines.append("Links you may mention (the only true ones): " + "; ".join(found) + ".")
    else:
        lines.append("Links: none this week. Do not suggest any connection between the metrics, the time of day or "
                     "the workouts.")
    return "\n".join(lines)


def recommendations_view(data: dict) -> str:
    """What to work on, in order, and what to leave alone."""
    lab = classify(data)
    needs = attention(lab)
    lines = [f"Phase: {lab['phase']}", f"Main point: {main_point(lab)}", ""]
    if needs:
        lines.append("What to work on, most important first:")
        for i, k in enumerate(needs[:3]):
            if k == "workouts":
                lines.append(f"  {_ORDINALS[min(i, 2)]}: workouts, none this week. Why it matters: "
                             f"{lab['workout_meaning']}. Kind of fix: {v3._fix(k, lab)}.")
            else:
                m = lab[k]
                lines.append(f"  {_ORDINALS[min(i, 2)]}: {k}, {m['wording']} ({m['gap']} off). Why it matters: "
                             f"{m['meaning']}. Kind of fix: {v3._fix(k, lab)}.")
    else:
        lines.append("What to work on: nothing is off target. Protect the routine that is working; do not invent "
                     "problems or new habits.")
    fine = v1.on_track(lab)
    if fine:
        lines.append("Leave unchanged (make no recommendation about these): " + ", ".join(fine) + ".")
    early = watch(lab)
    if early:
        lines.append("Worth guarding against: " + "; ".join(f"{k}, {why}" for k, why in early) + ".")
    missing = v1.no_data(lab)
    if missing:
        lines.append("No data this week (say nothing about it): " + ", ".join(missing) + ".")
    return "\n".join(lines)


def label_summary(data: dict) -> str:
    """Both views, for measurement and for the number check (the model never sees them together)."""
    return observations_view(data) + "\n\n" + recommendations_view(data)
