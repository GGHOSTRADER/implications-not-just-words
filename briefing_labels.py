"""Morning briefing as words: code turns yesterday's results and today's targets into labels; the model writes the
wording. Same principle as the weekly review (weekly_labels*.py) and checkpoint comments (checkpoint_labels.py).

The original briefing prompts (background_checks._briefing_yesterday_prompt / _briefing_today_prompt) already give
yesterday as classifications, but today as figures: "Calories today: adjusted (1,900 kcal)", "Carryover: - Carry
Over: 300 kcal over yesterday", weather in °C, mm and km/h, and a "1. 2. 3." list of what to write. Here:
  yesterday  each metric's result, what it meant for this phase, the link to today's carry-over ("that is why
             today's target is a little lower"), what went well and the one thing to do better today;
  today      targets as words against the usual ones and why, the weather with the temperature as a number (agreed
             2026-10-04) and the outdoor verdict, today's workout, and the plan as one sentence instead of a list
             (a listed plan was copied into the answers in the weekly-review study).

Pure functions, no database, no model.
"""
from __future__ import annotations

import re

import checkpoint_labels as cpl

FAT_LOSS, MUSCLE_GAIN, MAINTENANCE = cpl.FAT_LOSS, cpl.MUSCLE_GAIN, cpl.MAINTENANCE
_WEIGHT = {
    FAT_LOSS:    {"calories_over": 1.5, "calories_under": 0.3, "protein_short": 1.3, "steps_short": 1.0},
    MUSCLE_GAIN: {"calories_over": 0.5, "calories_under": 1.5, "protein_short": 1.3, "steps_short": 0.6},
    MAINTENANCE: {"calories_over": 1.2, "calories_under": 0.8, "protein_short": 1.0, "steps_short": 1.0},
}


def _phase(phase: str) -> str:
    return phase if phase in _WEIGHT else FAT_LOSS


# ── yesterday ────────────────────────────────────────────────────────────────

def _goal_words(value, goal) -> tuple[str, float]:
    """Where a 'reach the goal' metric (steps, protein) ended. Returns (words, shortfall fraction or 0)."""
    if value is None or not goal:
        return "no data", 0.0
    r = value / goal
    if r < 0.6:
        return f"far below the goal ({cpl.gap_words(1 - r)} short)", 1 - r
    if r < 0.85:
        return f"well below the goal ({cpl.gap_words(1 - r)} short)", 1 - r
    if r < 0.95:
        return f"a little short of the goal ({cpl.gap_words(1 - r)} short)", 1 - r
    if r < 1:
        return "just short of the goal", 1 - r
    return ("met, a little above the goal" if r < 1.1 else "well above the goal"), 0.0


def _calorie_words(value, target) -> tuple[str, float]:
    """Where calories ended against the target. Returns (words, signed deviation)."""
    if value is None or not target:
        return "no data", 0.0
    d = (value - target) / target
    if abs(d) <= 0.03:
        return "right on the target", d
    side = "over" if d > 0 else "under"
    size = "a little" if abs(d) <= 0.08 else "clearly" if abs(d) <= 0.25 else "far"
    return f"{size} {side} the target ({cpl.gap_words(d)} {side})", d


def _meaning(phase: str, key: str, dev: float) -> str:
    if key == "calories":
        if abs(dev) <= 0.03:
            return ""
        if dev > 0:
            return {FAT_LOSS: "in a cut this ate into the deficit", MUSCLE_GAIN: "a bigger surplus than planned",
                    MAINTENANCE: "this nudges weight up a little"}[phase]
        if phase == FAT_LOSS:
            return "a deeper deficit than planned, which is hard to keep up" if dev < -0.08 else "fine in a cut"
        return {MUSCLE_GAIN: "the surplus fell short", MAINTENANCE: "this nudges weight down a little"}[phase]
    if key == "protein":
        return {FAT_LOSS: "protein protects muscle in a cut", MUSCLE_GAIN: "protein drives recovery and growth",
                MAINTENANCE: "protein supports recovery"}[phase] if dev > 0 else ""
    return {FAT_LOSS: "less movement means a smaller deficit",
            MUSCLE_GAIN: "movement matters less than food in a gain phase",
            MAINTENANCE: "less movement makes it harder to stay level"}[phase] if dev > 0 else ""


def _carry_link(carry: dict | None) -> dict:
    """{metric: sentence} linking yesterday's result to today's adjusted target, when there is one."""
    if not carry:
        return {}
    if carry.get("pending"):
        return {"calories": "you will be asked how to spread that difference, so today's target may still change"}
    sentence = {"calories": "that is why today's calorie target is a little lower",
                "calories_boost": "that is why today's calorie target is a little higher",
                "steps": "that is why today's step goal is a little higher",
                "steps_reduce": "that is why today's step goal is a little lower"}.get(carry.get("strategy"), "")
    return {"calories": sentence} if sentence else {}


def yesterday_view(inp: dict) -> str:
    """v1."""
    return _yesterday(inp, version=1)


def yesterday_view_v2(inp: dict) -> str:
    """v2: only misses of 5% or more go on the to-do list, and the plan asks to name every item on it (v1's plan said
    'the first thing to do better', and the answers dropped a second big miss)."""
    return _yesterday(inp, version=2)


def _results(inp: dict) -> tuple:
    y = inp.get("yesterday") or {}
    steps_w, steps_short = _goal_words(y.get("steps"), y.get("step_goal"))
    cal_w, cal_dev = _calorie_words(y.get("calories"), y.get("calories_target"))
    prot_w, prot_short = _goal_words(y.get("protein"), y.get("protein_target"))
    return _phase(inp.get("phase", "")), y, steps_w, steps_short, cal_w, cal_dev, prot_w, prot_short


def _to_do(inp: dict, version: int) -> list[str]:
    """What to do better today, most important first (phase-weighted). v2 lists steps or protein only when 5%+ short."""
    phase, y, _sw, steps_short, _cw, cal_dev, _pw, prot_short = _results(inp)
    w = _WEIGHT[phase]
    floor = 0.05 if version >= 2 else 0.0
    scored = [(steps_short * w["steps_short"] if steps_short >= floor else 0, "steps"),
              (prot_short * w["protein_short"] if prot_short >= floor else 0, "protein"),
              (abs(cal_dev) * w["calories_over" if cal_dev > 0 else "calories_under"] if abs(cal_dev) > 0.03 else 0,
               "calories")]
    better = [k for s, k in sorted(scored, key=lambda t: -t[0]) if s > 0]
    if y.get("workout_day") and not y.get("workout_logged"):
        better.append("logging the workout")
    if not y.get("weight_logged"):
        better.append("the weigh-in")
    return better


def _yesterday(inp: dict, version: int) -> str:
    phase, y, steps_w, steps_short, cal_w, cal_dev, prot_w, prot_short = _results(inp)
    lines = [f"Phase: {phase}", "", "Yesterday:"]
    link = _carry_link((inp.get("today") or {}).get("carry")).get("calories", "")

    def line(title, words, meaning, extra=""):
        out = f"{title}: {words}."
        tail = "; ".join(x for x in (meaning, extra) if x)
        return out + (f" For the goal: {tail}." if tail else "")

    lines.append(line("Steps", steps_w, _meaning(phase, "steps", steps_short)))
    lines.append(line("Calories", cal_w, _meaning(phase, "calories", cal_dev), link if abs(cal_dev) > 0.03 else ""))
    lines.append(line("Protein", prot_w, _meaning(phase, "protein", prot_short)))
    if y.get("workout_day"):
        lines.append(f"Workout: {y.get('workout_type')}, " + ("done." if y.get("workout_logged") else
                                                               "planned but not logged."))
    else:
        lines.append("Workout: rest day.")
    lines.append("Weigh-in: recorded." if y.get("weight_logged") else
                 "Weigh-in: missed; a daily weigh-in keeps the progress estimate reliable.")
    lines.append("")

    well = [n for n, ok in (("steps", steps_short == 0 and steps_w != "no data"),
                            ("calories", abs(cal_dev) <= 0.03 and cal_w != "no data"),
                            ("protein", prot_short == 0 and prot_w != "no data"),
                            ("the workout", y.get("workout_day") and y.get("workout_logged")),
                            ("the weigh-in", y.get("weight_logged"))) if ok]
    better = _to_do(inp, version)
    lines.append("Went well: " + (", ".join(well) if well else "nothing stood out") + ".")
    lines.append("To do better today, most important first: " + (", then ".join(better) if better else "nothing") + ".")
    lines.append("")
    if version >= 2:
        todo = ("what to do better today, naming every item on that list in order and saying how for the first one."
                if better else "that the routine is working.")
    else:
        todo = "the first thing to do better today and how." if better else "that the routine is working."
    lines.append("Plan: start with a short, warm greeting that uses the user's name once (the name is in the first "
                 "line above). Then two or three sentences: first what went well, then " + todo
                 + " No numbers, no lists, no sign-off.")
    return "\n".join(lines)


# ── today ────────────────────────────────────────────────────────────────────

def _size(d: float) -> str:
    return "a little" if abs(d) < 0.08 else "noticeably" if abs(d) < 0.2 else "much"


def _target_words(base, today, what: str) -> tuple[str, int, str]:
    """('a little lower than usual (...)', -1, 'a little') against the usual target."""
    if not base or today is None or abs(today - base) / base < 0.01:
        return f"your usual {what}", 0, ""
    d = (today - base) / base
    return (f"{_size(d)} {'higher' if d > 0 else 'lower'} than usual ({cpl.gap_words(d)} {'up' if d > 0 else 'down'})",
            (1 if d > 0 else -1), _size(d))


def _workout_words(w: str, version: int = 1) -> tuple[str, str]:
    """(line, note subject) for today's workout string as the briefing has it."""
    w = (w or "").strip()
    low = w.lower()
    if not w or low.startswith("none"):
        # v1's "a short note on the day without a plan" came back as a vague "a quiet day", with no word on training
        return (("no workout planned today", "the fact that no workout is planned today") if version >= 2
                else ("no workout plan set", "the day without a plan"))
    if low.startswith("rest"):
        return "rest day", "the rest day"
    if low.startswith("cardio"):
        return "cardio day", "today's cardio"
    return f"{w} planned", f"today's {w} session"


def today_view(inp: dict) -> str:
    """v1."""
    return _today(inp, version=1)


def today_view_v2(inp: dict) -> str:
    """v2: a day with no workout plan says so plainly, and a suggested duration ('a twenty-minute walk') is allowed
    (v1's 'no numbers except the temperature' sent half the briefings back for one)."""
    return _today(inp, version=2)


def _today(inp: dict, version: int) -> str:
    t = inp.get("today") or {}
    carry = t.get("carry") or {}
    lines = [f"Today: {inp.get('day', '')}, a {'weekend day' if inp.get('weekend') else 'weekday'}."]
    w_line, w_subject = _workout_words(t.get("workout", ""), version)
    lines.append(f"Workout: {w_line}.")
    cal_w, cal_dir, cal_size = _target_words(t.get("cal_base"), t.get("cal_today"), "calorie target")
    # a reason only when it matches the direction (eating over yesterday lowers today's target, under raises it)
    reason = {("calories", -1): "to make up for eating over yesterday", ("calories_boost", 1): "because you ate under yesterday"}
    cal_line = f"Calories today: {cal_w}"
    if (carry.get("strategy"), cal_dir) in reason:
        cal_line += f", {reason[(carry['strategy'], cal_dir)]}"
    if carry.get("pending"):
        cal_line += (f"; yesterday ended {'over' if carry.get('direction') == 'over' else 'under'} the target and the "
                     "user will be asked how to spread it, so this may still change")
    lines.append(cal_line + ".")
    st_w, st_dir, _st_size = _target_words(t.get("steps_base"), t.get("steps_today"), "step goal")
    st_reason = {("steps", 1): "to offset yesterday's extra calories", ("steps_reduce", -1): "because you ate under yesterday"}
    lines.append(f"Steps today: {st_w}" + (f", {st_reason[(carry.get('strategy'), st_dir)]}"
                                          if (carry.get("strategy"), st_dir) in st_reason else "") + ".")
    weather = t.get("weather")
    if weather:
        lines.append(f"Weather now: {cpl.weather_words(weather)}.")
    lines.append({"suitable": "Outdoor walking: fine.",
                  "not suitable": "Outdoor walking: not suitable; any movement idea must be indoors.",
                  "unknown": "Outdoor walking: unknown; keep any movement idea indoors or weather-neutral."}[
        cpl.outdoor(weather)])
    likes = t.get("food_likes")
    if likes:
        lines.append(f"Foods the user likes: {likes}.")
    lines.append("")
    budget = (f"a {cal_size.replace('a little', 'slightly')} {'smaller' if cal_dir < 0 else 'bigger'}" if cal_dir
              else "a normal")
    # "light" only when it agrees with the budget: a light breakfast for a bigger budget is a mixed message
    breakfast = "breakfast idea" if cal_dir > 0 else "light breakfast idea"
    numbers = ("No numbers except the temperature and a suggested duration (such as a twenty-minute walk)"
               if version >= 2 else "No numbers except the temperature")
    lines.append("Plan: two or three sentences, with no greeting and no name because this continues the briefing: "
                 f"first a {breakfast}{' from the foods the user likes' if likes else ''} that suits {budget} "
                 f"calorie budget; then one movement idea that follows the outdoor rule; then a short note on "
                 f"{w_subject}. {numbers}, no lists, no sign-off.")
    return "\n".join(lines)


# ── prompts and guards ───────────────────────────────────────────────────────

_INTRO = ("This is part of the user's morning briefing. Below is a description worked out for you from their data; "
          "there are no numbers on purpose{temp}. Use it as it is and do not contradict it, in your own words.\n\n")


def yesterday_prompt(inp: dict, identity: str) -> str:
    return f"{identity}\n\n" + _INTRO.format(temp="") + yesterday_view(inp)


def today_prompt(inp: dict, identity: str) -> str:
    return f"{identity}\n\n" + _INTRO.format(temp=", except the temperature") + today_view(inp)


_GREETING = re.compile(r"^\W*(hi|hey|hello|good morning|morning)\b", re.I)
_NUM = re.compile(r"\d+(?:[.,:]\d+)?")


def problems(part: str, text: str, inp: dict) -> list[str]:
    """What is wrong with one part's answer ('yesterday' or 'today'). Empty = fine."""
    shown = set(_NUM.findall(today_view(inp) if part == "today" else yesterday_view(inp)))
    out = []
    stray = [n for n in _NUM.findall(text) if n not in shown]
    if stray:
        out.append("it uses numbers (" + ", ".join(sorted(set(stray))) + "); write in words"
                   + (", except the temperature" if part == "today" else ""))
    if cpl._LIST.search(text):
        out.append("it uses a list; write sentences")
    if part == "today":
        if _GREETING.search(text):
            out.append("it greets the user; this continues the briefing, so start with the suggestions")
        verdict = cpl.outdoor((inp.get("today") or {}).get("weather"))
        if verdict != "suitable" and cpl.suggests_outside(text, allow_conditional=verdict == "unknown"):
            out.append("it suggests going outside although outdoor walking is not suitable; keep movement indoors")
    return out


# ── version 2 (BRIEFING_MODE=labels2 / labels2_noguard, 2026-10-04) ──────────────────────────────────────────────
# v1 (views, prompts, problems) is unchanged, so its measured runs stay valid.

def yesterday_prompt_v2(inp: dict, identity: str) -> str:
    return f"{identity}\n\n" + _INTRO.format(temp="") + yesterday_view_v2(inp)


def today_prompt_v2(inp: dict, identity: str) -> str:
    return f"{identity}\n\n" + _INTRO.format(temp=", except the temperature") + today_view_v2(inp)


_DURATION = re.compile(r"\b\d+(?:\s*(?:[-–]|to)\s*\d+)?\s*-?\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?)\b", re.I)
_MENTIONS = {"steps": re.compile(r"\b(steps?|walk\w*|movement|moving|move|active)\b", re.I),
             "calories": re.compile(r"\b(calor\w*|food|eat\w*|ate|intake|portions?|meals?|surplus|deficit)\b", re.I),
             "protein": re.compile(r"\bprotein\b", re.I)}


def problems_v2(part: str, text: str, inp: dict) -> list[str]:
    """v2's checks: v1's, with a suggested duration allowed, plus every to-do item named (yesterday) and a plain note
    when no workout is planned (today)."""
    text = (text or "").replace("’", "'").replace("‘", "'")
    shown = set(_NUM.findall(today_view_v2(inp) if part == "today" else yesterday_view_v2(inp)))
    out = []
    stray = [n for n in _NUM.findall(_DURATION.sub(" ", text)) if n not in shown]
    if stray:
        out.append("it uses numbers (" + ", ".join(sorted(set(stray))) + "); write in words"
                   + (", except the temperature and a suggested duration" if part == "today" else ""))
    if cpl._LIST.search(text):
        out.append("it uses a list; write sentences")
    if part == "today":
        if _GREETING.search(text):
            out.append("it greets the user; this continues the briefing, so start with the suggestions")
        verdict = cpl.outdoor((inp.get("today") or {}).get("weather"))
        if verdict != "suitable" and cpl.suggests_outside(text, allow_conditional=verdict == "unknown"):
            out.append("it suggests going outside although outdoor walking is not suitable; keep movement indoors")
        workout = ((inp.get("today") or {}).get("workout") or "").strip().lower()
        if (not workout or workout.startswith("none")) and not re.search(r"\b(workout|training|session|plan\w*|rest)\b",
                                                                          text, re.I):
            out.append("it does not say that no workout is planned today")
    else:
        missing = [k for k in _to_do(inp, 2) if k in _MENTIONS and not _MENTIONS[k].search(text)]
        if missing:
            out.append("it leaves out " + " and ".join(missing) + ", which the list of what to do better includes; "
                       "name every item on it")
    return out
