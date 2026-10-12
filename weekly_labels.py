"""Turn the week's numbers into plain-language labels, deterministically, before the language model sees anything.

A small model is good at reasoning over words and unreliable at reasoning over numbers (it called 2650 kcal "slightly
below" a 2000 kcal target). So the numeric work, comparing an average with its target, reading a slope, judging how
variable a week was, is done HERE, by code that is always right, and the model is only handed the verdicts as text.
Even the averages become words ("a little below the goal"), so the review contains no digits to misquote.

Pure functions: no model, no database, no app imports.

    classify(data)       -> {"steps": {...}, "calories": {...}, "protein": {...}, "workouts": ..., ...}
    label_summary(data)  -> the text block shown to the model (contains no digits)
"""
from __future__ import annotations

import re

CALORIE_BAND = 0.08            # within 8% of the calorie target counts as on target (same band as the eval rubric)

_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven"]

# metric -> words that show a sentence is about it (used to check that every off-target metric gets addressed)
KEYWORDS = {
    "steps": re.compile(r"\b(steps?|walk\w*|move|moving|activity|active|stairs|cardio)\b", re.I),
    "protein": re.compile(r"\b(protein|meals?|shake|eggs?|chicken|fish|yogurt|whey|lean)\b", re.I),
    "calories": re.compile(r"\b(calor\w*|kcal|intake|portions?|eat\w*|deficit|surplus|food|meals?)\b", re.I),
}
METRIC_NAMES = {"steps": "steps", "calories": "calories", "protein": "protein"}


def _level_vs_goal(avg: float, goal: float) -> tuple[str, str]:
    """(status, words) for a metric where reaching the goal is the aim (steps, protein)."""
    r = avg / goal
    if r < 0.60:
        return "below", "far below the goal"
    if r < 0.85:
        return "below", "well below the goal"
    if r < 1.0:
        return "below", "a little below the goal"
    if r < 1.10:
        return "met", "right at the goal, just above it"
    return "met", "well above the goal"


def _level_calories(avg: float, target: float) -> tuple[str, str]:
    """(status, words) for calories, where both too high and too low are off target."""
    d = (avg - target) / target
    if abs(d) <= 0.03:
        return "on", "right on the target"
    if abs(d) <= CALORIE_BAND:
        return "on", ("slightly over the target" if d > 0 else "slightly under the target")
    if d > 0:
        return "over", ("far over the target" if d > 0.25 else "clearly over the target")
    return "under", ("far under the target" if d < -0.25 else "clearly under the target")


def _trend(avg: float, slope: float) -> str:
    rel = slope / avg if avg else 0.0
    if abs(rel) < 0.003:                      # under ~2% over a week counts as holding steady
        return "holding steady through the week"
    word = "rising" if rel > 0 else "falling"
    return f"{word} quickly through the week" if abs(rel) >= 0.02 else f"{word} gradually through the week"


def _consistency(cv: float) -> str:
    if cv < 10:
        return "very consistent from day to day"
    if cv < 20:
        return "fairly consistent from day to day"
    if cv < 30:
        return "quite variable from day to day"
    return "very erratic from day to day"


def _hit_days(hit, total: int) -> str:
    if hit is None:
        return "not enough data yet"
    if hit == 0:
        return "never"
    if hit >= total:
        return "every day"
    share = hit / total
    if share < 0.35:
        return "on only a few days"
    if share < 0.65:
        return "on about half the days"
    return "on most days"


def _workouts(n: int, total: int) -> str:
    if n == 0:
        return "no workouts"
    if n <= 2:
        return f"a light week, {_WORDS[n]} session{'s' if n > 1 else ''}"
    if n <= 4:
        return f"a steady week, {_WORDS[n]} sessions"
    return f"a heavy week, {_WORDS[min(n, 7)]} sessions"


def _time_of_day(hour: int) -> str:
    if hour < 12:
        return "the morning"
    if hour < 17:
        return "the afternoon"
    if hour < 20:
        return "the early evening"
    return "the late evening"


def classify(data: dict) -> dict:
    """Every verdict about the week, as words. `status` is what the checks use; the rest is what the model reads."""
    goals = data["goals"]
    total = data.get("total_days") or 7
    out: dict = {"phase": goals.get("phase", "unknown")}
    for key, goal_key in (("steps", "step_goal"), ("protein", "protein_target"), ("calories", "calories_target")):
        m, goal = data[key], goals.get(goal_key, 0)
        if m.get("status") != "ok" or not goal:
            out[key] = {"status": "nodata"}
            continue
        status, level = (_level_calories if key == "calories" else _level_vs_goal)(m["avg"], goal)
        out[key] = {"status": status, "level": level, "trend": _trend(m["avg"], m["slope"]),
                    "consistency": _consistency(m["coefficient_of_variation_pct"])}
    out["workouts"] = _workouts(len(data.get("workouts") or []), total)
    out["goal_days"] = _hit_days(data.get("goal_hit_days"), total)
    hourly = data.get("hourly") or {}
    out["peak"] = _time_of_day(hourly["peak_hour"]) if hourly.get("status") == "ok" else None
    return out


def off_target(labels: dict) -> list[str]:
    """Metrics that need attention, in a fixed order."""
    return [k for k in ("steps", "protein", "calories") if labels[k]["status"] in ("below", "over", "under")]


def on_track(labels: dict) -> list[str]:
    return [k for k in ("steps", "protein", "calories") if labels[k]["status"] in ("met", "on")]


def no_data(labels: dict) -> list[str]:
    return [k for k in ("steps", "protein", "calories") if labels[k]["status"] == "nodata"]


def label_summary(data: dict) -> str:
    """The text block the model reads. It has no digits at all."""
    lab = classify(data)
    lines = [f"Phase: {lab['phase']}", ""]
    titles = {"steps": "Steps", "calories": "Calories", "protein": "Protein"}
    for key in ("steps", "calories", "protein"):
        m = lab[key]
        if m["status"] == "nodata":
            lines.append(f"{titles[key]}: no data this week.")
        else:
            lines.append(f"{titles[key]}: {m['level']}; {m['trend']}; {m['consistency']}.")
    lines.append("")
    needs, fine, missing = off_target(lab), on_track(lab), no_data(lab)
    detail = lambda k: f"{k} ({lab[k]['level']})"
    lines.append("Needs attention: " + (", ".join(detail(k) for k in needs) if needs else "nothing") + ".")
    lines.append("On track: " + (", ".join(fine) if fine else "nothing") + ".")
    if missing:
        lines.append("No data this week (say nothing about these): " + ", ".join(missing) + ".")
    lines.append("")
    lines.append(f"Workouts this week: {lab['workouts']}.")
    lines.append(f"Days calories landed almost exactly on the target (this says nothing about over or under): {lab['goal_days']}.")
    lines.append(f"Most active time of day: {lab['peak']}." if lab["peak"] else "Most active time of day: no data yet.")
    return "\n".join(lines)
