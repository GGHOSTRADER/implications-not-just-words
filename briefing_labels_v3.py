"""Labels v3 of the morning briefing (BRIEFING_MODE=labels3 / labels3_noguard, 2026-10-05).

The yesterday part is v1's (briefing_labels.py, unchanged: v2's version lost on held-out cases). The today part gets
what the real checkpoint message showed was missing (label_context_v3.py): steps are walking steps, no exercises are
named, training is only the plan's session, the weather rule is rain / great heat / strong wind -> indoors, sunny ->
a walk or a treadmill as equals, unknown -> leans indoors. A suggested duration ("a twenty-minute walk") is allowed.
"""
from __future__ import annotations

import re

import briefing_labels as bl
import checkpoint_labels as cl
import checkpoint_labels_v3 as cl3
import label_context_v3 as ctx3


def today_view_v3(inp: dict) -> str:
    t = inp.get("today") or {}
    carry = t.get("carry") or {}
    lines = [f"Today: {inp.get('day', '')}, a {'weekend day' if inp.get('weekend') else 'weekday'}."]
    w_line, w_subject = bl._workout_words(t.get("workout", ""), 2)
    lines.append(f"Workout (from the user's plan): {w_line}.")
    cal_w, cal_dir, cal_size = bl._target_words(t.get("cal_base"), t.get("cal_today"), "calorie target")
    reason = {("calories", -1): "to make up for eating over yesterday", ("calories_boost", 1): "because you ate under yesterday"}
    cal_line = f"Calories today: {cal_w}"
    if (carry.get("strategy"), cal_dir) in reason:
        cal_line += f", {reason[(carry['strategy'], cal_dir)]}"
    if carry.get("pending"):
        cal_line += (f"; yesterday ended {'over' if carry.get('direction') == 'over' else 'under'} the target and the "
                     "user will be asked how to spread it, so this may still change")
    lines.append(cal_line + ".")
    st_w, st_dir, _size = bl._target_words(t.get("steps_base"), t.get("steps_today"), "step goal")
    st_reason = {("steps", 1): "to offset yesterday's extra calories", ("steps_reduce", -1): "because you ate under yesterday"}
    lines.append(f"Steps today: {st_w}" + (f", {st_reason[(carry.get('strategy'), st_dir)]}"
                                          if (carry.get("strategy"), st_dir) in st_reason else "") + ".")
    weather = t.get("weather")
    if weather:
        lines.append(f"Weather now: {cl.weather_words(weather)}.")
    lines.append(ctx3.steps_line(weather))
    likes = t.get("food_likes")
    if likes:
        lines.append(f"Foods the user likes: {likes}.")
    lines.append("")
    budget = (f"a {cal_size.replace('a little', 'slightly')} {'smaller' if cal_dir < 0 else 'bigger'}" if cal_dir
              else "a normal")
    breakfast = "breakfast idea" if cal_dir > 0 else "light breakfast idea"
    lines.append("Plan: two or three sentences, with no greeting and no name because this continues the briefing: "
                 f"first a {breakfast}{' from the foods the user likes' if likes else ''} that suits {budget} "
                 "calorie budget; then one way to add steps by walking, from the ways listed; then a short note on "
                 f"{w_subject}, as planned. No numbers except the temperature and a suggested duration, no lists, no "
                 "sign-off.")
    return "\n".join(lines)


def today_prompt_v3(inp: dict, identity: str) -> str:
    return (f"{identity}\n\n" + bl._INTRO.format(temp=", except the temperature") + today_view_v3(inp) + "\n\n"
            + ctx3.NO_EXERCISES)


def problems_v3(part: str, text: str, inp: dict) -> list[str]:
    """Yesterday: v1's checks. Today: v2's checks with weather-aware outdoor, plus exercises and invented workouts."""
    if part == "yesterday":
        return bl.problems("yesterday", text, inp)
    text = (text or "").replace("’", "'").replace("‘", "'")
    out = [p for p in bl.problems_v2("today", text, inp) if "going outside" not in p]
    for m in cl3._EXERCISE.finditer(text):
        if not cl3._negated(text, m.start()):
            out.append(f"it names an exercise ({m.group(0)}); steps are walking steps, so suggest only walking, the "
                       "stairs, pacing or a treadmill, and never exercises")
            break
    for m in cl3._EXTRA_WORKOUT.finditer(text):
        if not cl3._negated(text, m.start()):
            out.append("it suggests a workout; training comes only from the user's plan, so do not add or invent one")
            break
    weather = (inp.get("today") or {}).get("weather")
    verdict, reason = ctx3.weather_verdict(weather)
    if verdict != "fine" and cl.suggests_outside(text, allow_conditional=verdict == "unknown"):
        out.append(f"it sends the user outside ({reason}); suggest steps indoors: a treadmill, the stairs, pacing or a "
                   "covered walk")
    return out
