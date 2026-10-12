"""Labels v3 of the checkpoint comment (CHECKPOINT_MODE=labels3 / labels3_noguard, 2026-10-05).

v2 (checkpoint_labels.py, unchanged) plus what a real message showed it lacked (label_context_v3.py): steps are walking
steps only and no exercises are ever named; training is never invented; rain, great heat and wind send the steps
indoors, a sunny day offers a walk or a treadmill as equals, unknown weather leans indoors; the message knows how far
into the day it is and what still fits; and the food list leaves out supplements (Fish Oil) and, when calories are far
over plan, becomes "lighter portions" instead of a list of foods.
"""
from __future__ import annotations

import re

import checkpoint_labels as cl
import label_context_v3 as ctx3

_KEEP_SIZE = cl._KEEP_SIZE


def pick_foods(lab: dict, ctx: dict, n: int = 3) -> tuple[str, list[str]]:
    """v1's picker without supplements; no food list at all when calories are far over plan."""
    clean = dict(ctx)
    clean["inventory"] = [(nm, k, p) for nm, k, p in (ctx.get("inventory") or [])
                          if not ctx3.is_supplement(nm, float(k or 0), float(p or 0))]
    why, foods = cl.pick_foods(lab, clean, n)
    if why == "light options" and lab["metrics"]["calories"]["day_dev"] > 0.3:
        return "", []                                  # far over: the advice is lighter portions, not another food
    return why, foods


def view_v3(inp: dict) -> str:
    lab = cl.classify(inp)
    last = lab["is_last"]
    start, end, end_min = inp["block_start_hour"], inp["end_hour"], inp.get("end_minute", 0)
    pr = lab["priorities"]
    ctx = inp.get("context") or {}
    lines = [f"Phase: {lab['phase']}"]
    if last:
        lines.append(f"This is the last check of the day: the block from {cl.hour_words(start)} to "
                     f"{cl.hour_words(end, end_min)} has ended, and so has the day's plan.")
    else:
        lines.append(f"The block that just ended ran from {cl.hour_words(start)} to {cl.hour_words(end, end_min)}. It is "
                     "over; anything forward-looking is for the rest of the day.")
    lines.append(ctx3.time_line(end, end_min, cl.hour_words(end, end_min), last))
    lines.append("")
    if lab["no_food"]:
        lines.append("Food: nothing logged yet today. That usually means not eaten yet or not logged yet, which is "
                     "not falling behind: ask which it is and nudge to log or to eat. Do not call calories or protein "
                     "behind.")
    for key in cl._ORDER:
        if lab["no_food"] and key in ("calories", "protein"):
            continue
        m = lab["metrics"][key]
        parts = [f"in this block {m['block']}", m["day"]]
        if m["rest"]:
            parts.append(m["rest"])
        line = f"{cl._TITLES[key]}: " + "; ".join(parts) + "."
        if m["meaning"]:
            line += f" For the goal: {m['meaning']}."
        lines.append(line)
    w = ctx.get("workout")
    if w:
        done = ("logged" if w.get("logged") else "not logged") if last else ("done" if w.get("logged") else "not done yet")
        lines.append("Workout today (from the user's plan): "
                     + ("cardio day" if w["kind"] == "cardio" else "rest day" if w["kind"] == "rest"
                        else f"{w['type']}, {done}") + ".")
    if not last:
        weather = ctx.get("weather")
        if weather:
            lines.append(f"Weather now: {cl.weather_words(weather)}.")
        if "steps" in pr:
            lines.append(ctx3.steps_line(weather))
        why, foods = pick_foods(lab, ctx)
        if foods:
            lines.append(f"Foods from the user's list that fit ({why}): " + ", ".join(foods) + ".")
        if ctx.get("food_dislikes"):
            lines.append(f"Never suggest: {ctx['food_dislikes']}.")
        if ctx.get("morning_feedback"):
            lines.append(f"How the user said they felt this morning: {ctx['morning_feedback']}")
    lines.append("")
    if last:
        lines.append("Plan: two or three sentences that sum up how the day went on calories, protein and steps, most "
                     f"important first. {_KEEP_SIZE} Nothing about the weather, food, walks or plans for tomorrow; the "
                     "morning briefing covers that. The day is over: say whether each target was met, over or short of "
                     "plan, and never say on track, still time, catch up, keep going or anything about later.")
    elif pr:
        names = {"food": "logging or eating", "calories": "calories", "protein": "protein", "steps": "steps"}
        lines.append("What matters for the rest of the day, most important first: "
                     + ", then ".join(names[p] for p in pr) + ".")
        _why, foods = pick_foods(lab, ctx)
        first = pr[0]
        over_no_list = lab["metrics"]["calories"]["day_dev"] > 0.05 and not foods and not lab["no_food"]
        action = ("one concrete action for the rest of the day for the first item"
                  + (", using the foods listed" if foods else "")
                  + (": lighter portions at the next meal and no extra snack" if first == "calories" and over_no_list else "")
                  + ("; and because calories are over plan, also say lighter portions at the next meal and no extra snack"
                     if first != "calories" and over_no_list else ""))
        lines.append("Plan: one short paragraph of three or four sentences. The first says how the block went, "
                     f"starting with what matters most. {_KEEP_SIZE} The next gives {action}"
                     + ("; if steps are on the list, add one way to add steps by walking, from the ways listed"
                        if "steps" in pr else "") + ". End with one short line of encouragement.")
    else:
        lines.append("Everything is on plan: say so in two or three sentences, with one small idea to keep it going "
                     "and one short line of encouragement.")
    return "\n".join(lines)


def prompt_v3(inp: dict, identity: str) -> str:
    last = bool(inp.get("is_last"))
    return (
        f"{identity}\n\n"
        f"This is the user's {inp.get('ordinal', '')} check of the day{' (the last one)' if last else ''}. Below is what "
        "happened, worked out for you from their data. There are no numbers on purpose"
        + (".\n\n" if last else ", except the temperature.\n\n")
        + view_v3(inp) + "\n\n"
        "Write plain text: no greeting, no sign-off, no headings, no lists. Use the descriptions as they are and do not "
        "contradict them, in your own words. Write in words, never with digits"
        + ("." if last else ", except the temperature. Never suggest a food listed under 'Never suggest'.")
        + " " + ctx3.NO_EXERCISES
    )


# ── answer checks (labels3 with checks) ──────────────────────────────────────

_EXERCISE = re.compile(r"\b(squats?|lunges?|push-?ups?|pull-?ups?|sit-?ups?|planks?|burpees?|jumping jacks?|crunch(?:es)?|"
                       r"calisthenics|body-?weight|hiit|circuit|yoga|pilates|dumbbells?|kettlebells?|deadlifts?|"
                       r"resistance bands?|home workout|workout at home|sets? of|reps?)\b", re.I)
_EXTRA_WORKOUT = re.compile(r"\b(?:extra|another|additional|new|different|quick|short)\s+(?:workout|training session|"
                            r"gym session|strength session)\b|\b(?:add|do|try|start|squeeze in|fit in)\b[^.!?]{0,20}\b"
                            r"(?:a|an|another|your)\s+(?:\w+\s+)?(?:workout|training session|gym session)\b", re.I)
_MEAL_WORDS = {"breakfast": 11, "lunch": 15, "dinner": 22, "snack": 22}     # first fire hour at which each is over
_MOVE_WORD = re.compile(r"\b(walk\w*|treadmill|stairs|pacing|steps?|stroll\w*)\b", re.I)
_TOMORROW = re.compile(r"\b(tomorrow|tonight's|next morning)\b", re.I)
# at the day's last check nothing is left to steer: "on track", "still time", "keep going" talk as if the day went on
_LAST_FRAMING = re.compile(r"\b(on track|off track|back on track|still (?:have )?time|plenty of (?:time|day)|keep (?:it )?up|"
                           r"keep going|stay on track|catch up|rest of the day|later today|next block|next meal|"
                           r"tomorrow|from here)\b", re.I)


def _negated(text: str, pos: int) -> bool:
    return bool(re.search(r"\b(?:no|not|never|without|avoid|skip|don'?t|instead of|rather than|nothing)\b[^.!?]{0,25}$",
                          text[max(0, pos - 40):pos], re.I))


def problems_v3(text: str, inp: dict) -> list[str]:
    """v2's checks, plus: exercises named, workouts invented, suggestions that do not fit the time of day, outdoor-only
    ideas when the steps belong indoors."""
    text = (text or "").replace("’", "'").replace("‘", "'")
    out = list(cl.problems_v2(text, inp))
    out = [p for p in out if "going outside" not in p]            # v3 has its own, weather-aware version
    last = bool(inp.get("is_last"))
    hour = inp["end_hour"] + (1 if inp.get("end_minute", 0) >= 30 else 0)
    ctx = inp.get("context") or {}
    verdict, reason = ctx3.weather_verdict(ctx.get("weather"))
    for m in _EXERCISE.finditer(text):
        if not _negated(text, m.start()):
            out.append(f"it names an exercise ({m.group(0)}); steps are walking steps, so suggest only walking, the "
                       "stairs, pacing or a treadmill, and never exercises")
            break
    w = ctx.get("workout") or {}
    for m in _EXTRA_WORKOUT.finditer(text):
        if not _negated(text, m.start()):
            out.append("it suggests a workout; training comes only from the user's plan, so do not add or invent one")
            break
    if last:
        for m in _LAST_FRAMING.finditer(text):
            if not _negated(text, m.start()):
                out.append(f"it talks as if the day went on ({m.group(0)!r}); the day is over, so say only whether each "
                           "target was met, over or short of plan")
                break
    if not last:
        for word, over_at in _MEAL_WORDS.items():
            for m in re.finditer(rf"\b{word}s?\b", text, re.I):
                if hour >= over_at and not _negated(text, m.start()) and not _TOMORROW.search(text[max(0, m.start() - 40):m.end() + 40]) \
                        and re.search(r"\b(try|have|eat|grab|choose|opt|add|go for|plan|start|enjoy|make|prepare)\b",
                                      text[max(0, m.start() - 50):m.end() + 20], re.I):
                    out.append(f"it suggests {word}, but at this hour that is over; suggest only what still fits the "
                               "time of day")
                    break
        if hour >= 22:
            for s in re.split(r"(?<=[.!?])\s+", text):
                if _MOVE_WORD.search(s) and re.search(r"\b(try|take|go|get|add|do|fit|squeeze|aim|consider)\b", s, re.I):
                    out.append("it suggests walking, but it is too late for movement ideas")
                    break
        if verdict != "fine" and cl.suggests_outside(text, allow_conditional=verdict == "unknown"):
            out.append(f"it sends the user outside ({reason}); suggest steps indoors: a treadmill, the stairs, pacing "
                       "or a covered walk")
    return out
