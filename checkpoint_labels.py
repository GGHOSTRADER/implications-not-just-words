"""Checkpoint comment as words: code turns the checkpoint's numbers into labels, the model only writes the wording.

Same principle as the weekly review's labels (weekly_labels*.py, evals/REPORT.md): a small local model reads words well
and numbers badly. The original checkpoint prompt (background_checks._checkpoint_prompt) gives it "live 1850 kcal vs
expected 1400 kcal (status 3, over); target next block 320 kcal (raw carry -130 ...)" plus rules for decoding the
status codes, then asks it to pick foods that "fit the budget" from a list of kcal/protein pairs. Here code does all
of that: what happened in the block that just closed, where the day stands against plan, what that means in this
phase, the carry-over as the rest of the day's budget compared with a normal one, the priority order, foods from the
user's list that fit, and the plan for the message.

Pure functions, no database, no model. The only figures shown are the temperature (agreed 2026-10-04) and anything the
user typed in their morning feedback.
"""
from __future__ import annotations

import re

FAT_LOSS, MUSCLE_GAIN, MAINTENANCE = "Fat Loss", "Muscle Gain", "Maintenance"
_TITLES = {"calories": "Calories", "protein": "Protein", "steps": "Steps"}
_ORDER = ("calories", "protein", "steps")

# day-so-far deviation that counts as a problem worth acting on (|cumulative / plan - 1|)
_ON_PLAN = 0.05
# how much each kind of problem matters, by phase (priority = size x weight), as in weekly_labels_v3
_WEIGHT = {
    FAT_LOSS:    {"calories_over": 1.5, "calories_under": 0.0, "protein_behind": 1.3, "steps_behind": 1.0},
    MUSCLE_GAIN: {"calories_over": 0.5, "calories_under": 1.5, "protein_behind": 1.3, "steps_behind": 0.6},
    MAINTENANCE: {"calories_over": 1.2, "calories_under": 0.8, "protein_behind": 1.0, "steps_behind": 1.0},
}

_GAP_WORDS = [(0.05, "only a few percent"), (0.09, "a bit under a tenth"), (0.13, "about a tenth"),
              (0.18, "about a sixth"), (0.23, "about a fifth"), (0.29, "about a quarter"), (0.4, "about a third"),
              (0.6, "about half"), (0.9, "most"), (9e9, "far more than the whole share")]


def gap_words(fraction: float) -> str:
    f = abs(fraction)
    return next(w for limit, w in _GAP_WORDS if f < limit)


def hour_words(hour: int, minute: int = 0) -> str:
    """15 -> 'three in the afternoon'; the last block (22:59) ends at eleven at night."""
    if minute >= 30:
        hour += 1
    hour %= 24
    if hour == 12:
        return "noon"
    if hour == 0:
        return "midnight"
    names = ["twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]
    part = ("in the morning" if hour < 12 else "in the afternoon" if hour < 17 else "in the evening" if hour < 21
            else "at night")
    return f"{names[hour % 12]} {part}"


def _phase(phase: str) -> str:
    return phase if phase in _WEIGHT else FAT_LOSS          # goals_loader defaults to Fat Loss


# ── per metric ───────────────────────────────────────────────────────────────

def _block_words(key: str, m: dict) -> str:
    """What happened in the block that just closed, against this block's share."""
    live, exp = m["block_live"], m["block_expected"]
    if exp <= 0:
        return "nothing was planned for this block" + (", but some came in anyway" if live > 0 else "")
    dev = live / exp - 1
    if abs(dev) < 0.05:
        return "right on this block's share"
    if dev >= 1:
        return "more than double this block's share"
    if dev >= 0.6:
        return "nearly double this block's share"
    if dev <= -0.95:
        return "almost none of this block's share"
    if dev <= -0.6:
        return "less than half of this block's share"
    return f"{gap_words(dev)} {'over' if dev > 0 else 'short of'} this block's share"


def _day_dev(m: dict) -> float:
    plan = m["daily_target"] * m["cum_frac"]
    return (m["cum_live"] / plan - 1) if plan > 0 else 0.0


def _day_words(key: str, dev: float) -> str:
    if abs(dev) < _ON_PLAN:
        return "the day so far is on plan"
    side = ("over" if dev > 0 else "under") if key == "calories" else ("ahead of" if dev > 0 else "behind")
    size = "a little" if abs(dev) < 0.12 else "well" if abs(dev) < 0.3 else "far"
    return f"the day so far is {size} {side} plan" if key == "calories" else f"the day so far is {size} {side} plan"


def _rest_words(key: str, m: dict) -> str:
    """The carry-over: what is left for the rest of the day compared with a normal rest of the day."""
    normal = m["daily_target"] * (1 - m["cum_frac"])
    left = m["daily_target"] - m["cum_live"]
    if normal <= 0:
        return ""
    if left <= 0:
        return {"calories": "the day's calorie budget is already used up",
                "protein": "the day's protein goal is already reached",
                "steps": "the day's step goal is already reached"}[key]
    r = left / normal
    what = {"calories": "food", "protein": "protein", "steps": "walking"}[key]
    if 0.97 <= r <= 1.03:
        return f"the rest of the day needs a normal amount of {what}"
    size = ("a little" if abs(r - 1) < 0.15 else "a fair bit" if abs(r - 1) < 0.5 else "much")
    if key == "calories":
        return f"the rest of the day has {size} {'more' if r > 1 else 'less'} food to work with than usual"
    if r < 1:                              # protein and steps are floors: being ahead just makes the goal easier
        return f"the rest of the day needs {size} less {what} than usual to reach the goal"
    return f"the rest of the day needs {size} more {what} than usual"


def _meaning_past(key: str, phase: str, dev: float) -> str:
    """For the day's last check: what the day's result meant, looking back (no 'next meal')."""
    if abs(dev) < _ON_PLAN:
        return ""
    if key == "calories":
        if dev > 0:
            return {FAT_LOSS: "ending over plan ate into the day's deficit",
                    MUSCLE_GAIN: "ending over plan in a gain phase adds a little extra fat",
                    MAINTENANCE: "ending over plan nudges weight up a little"}[phase]
        return {FAT_LOSS: "ending under plan is fine in a cut",
                MUSCLE_GAIN: "ending under plan left the surplus short today",
                MAINTENANCE: "ending under plan nudges weight down a little"}[phase]
    if key == "protein":
        return "protein fell short today, which matters for keeping muscle" if dev < 0 else "protein was well covered"
    return "movement fell short today" if dev < 0 else "movement was well covered"


def _meaning(key: str, phase: str, dev: float) -> str:
    """What the day-so-far position means for this phase's goal."""
    if abs(dev) < _ON_PLAN:
        return ""
    if key == "calories":
        if dev > 0:
            return {FAT_LOSS: "in a cut this eats into today's deficit",
                    MUSCLE_GAIN: "in a gain phase being ahead is fine as long as the day does not end far over",
                    MAINTENANCE: "the rest of the day needs to be lighter to stay level"}[phase]
        return {FAT_LOSS: "in a cut being under is fine; there is no need to catch up",
                MUSCLE_GAIN: "in a gain phase the surplus needs these calories, so the rest of the day needs more food",
                MAINTENANCE: "there is room for a normal meal later"}[phase]
    if key == "protein":
        if dev < 0:
            return {FAT_LOSS: "protein protects muscle in a cut, so the next meal should lead with protein",
                    MUSCLE_GAIN: "protein drives recovery and growth, so the next meal should lead with protein",
                    MAINTENANCE: "the next meal should lead with protein"}[phase]
        return "protein is well covered"
    if dev < 0:
        return {FAT_LOSS: "less movement means a smaller deficit",
                MUSCLE_GAIN: "movement matters less than food in a gain phase, but it still helps",
                MAINTENANCE: "less movement makes it harder to stay level"}[phase]
    return "movement is well covered"


# ── the whole checkpoint ─────────────────────────────────────────────────────

def classify(inp: dict) -> dict:
    """All verdicts for one checkpoint. `inp` holds phase, is_last, hours, per-metric figures and the context data."""
    phase = _phase(inp.get("phase", ""))
    ctx = inp.get("context") or {}
    no_food = not ctx.get("food_entries") and round(inp["metrics"]["calories"]["cum_live"]) == 0 \
        and round(inp["metrics"]["protein"]["cum_live"]) == 0
    out = {"phase": phase, "no_food": no_food, "is_last": bool(inp.get("is_last")), "metrics": {}}
    for key in _ORDER:
        m = inp["metrics"][key]
        dev = _day_dev(m)
        out["metrics"][key] = {
            "block": _block_words(key, m), "day_dev": dev, "day": _day_words(key, dev),
            "rest": "" if out["is_last"] else _rest_words(key, m),
            "meaning": _meaning_past(key, phase, dev) if out["is_last"] else _meaning(key, phase, dev),
        }
    out["priorities"] = priorities(out)
    w = ctx.get("weather")
    out["outdoor"] = outdoor(w)
    out["weather"] = weather_words(w)
    out["foods"] = pick_foods(out, ctx)
    return out


def priorities(lab: dict) -> list[str]:
    """What to act on in the rest of the day, most important first."""
    if lab["no_food"]:
        steps = lab["metrics"]["steps"]["day_dev"]
        return ["food"] + (["steps"] if steps <= -_ON_PLAN else [])
    weights = _WEIGHT[lab["phase"]]
    scored = []
    for key in _ORDER:
        dev = lab["metrics"][key]["day_dev"]
        if abs(dev) < _ON_PLAN:
            continue
        kind = (f"calories_{'over' if dev > 0 else 'under'}" if key == "calories"
                else f"{key}_behind" if dev < 0 else None)
        if kind and weights.get(kind, 0) > 0:
            scored.append((abs(dev) * weights[kind], key))
    return [k for _s, k in sorted(scored, key=lambda t: -t[0])]


def outdoor(weather: dict | None) -> str:
    """The same rule the original prompts use (WMO code 51+ or 0.5 mm of rain = wet), as a verdict."""
    if not weather:
        return "unknown"
    code, precip = weather.get("weather_code"), weather.get("precipitation_mm") or 0
    return "not suitable" if (code is not None and code >= 51) or precip >= 0.5 else "suitable"


def weather_words(weather: dict | None) -> str:
    if not weather:
        return ""
    words = [weather.get("description") or "weather unknown"]
    if weather.get("temp_c") is not None:
        words.append(f"{round(weather['temp_c'])}°C")                 # the one figure kept (agreed 2026-10-04)
    wind = weather.get("wind_kph") or 0
    if wind >= 30:
        words.append("very windy")
    elif wind >= 18:
        words.append("windy")
    return ", ".join(words)


def _liked(name: str, likes: str) -> bool:
    return any(w and w in name.lower() for w in re.split(r"[,;/]\s*|\s+and\s+", (likes or "").lower()))


def pick_foods(lab: dict, ctx: dict, n: int = 3) -> tuple[str, list[str]]:
    """(why, food names) from the user's food list that fit what the rest of the day needs; ('', []) when no food
    advice is needed. Dislikes are never picked; likes win ties."""
    inv = [(name, float(kcal or 0), float(prot or 0)) for name, kcal, prot in (ctx.get("inventory") or [])]
    dislikes = ctx.get("food_dislikes") or ""
    inv = [f for f in inv if not (dislikes and _liked(f[0], dislikes))]
    if lab["is_last"] or not inv:
        return "", []
    pr = lab["priorities"]
    cal_dev = lab["metrics"]["calories"]["day_dev"]
    protein_behind = "protein" in pr or lab["no_food"]
    # Each kind of advice only lists foods that actually qualify; a short list beats a wrong one.
    if protein_behind and lab["phase"] == MUSCLE_GAIN and cal_dev < 0:
        why, key = "protein-rich and filling, for protein and calories together", lambda f: -(f[2] + f[1] / 20)
        ok = lambda f: f[2] >= 8
    elif protein_behind:
        why, key = "lean protein for few calories", lambda f: -(f[2] / f[1] if f[1] > 0 else 0)
        ok = lambda f: f[2] >= 8 and f[1] > 0 and f[2] / f[1] >= 0.05        # at least about 5 g per 100 kcal
    elif "calories" in pr and cal_dev > 0:
        why, key = "light options", lambda f: f[1]
        ok = lambda f: 0 < f[1] <= 200
    elif "calories" in pr and cal_dev < 0:
        why, key = "energy-dense options", lambda f: -f[1]
        ok = lambda f: f[1] >= 250
    else:
        return "", []
    likes = ctx.get("food_likes") or ""
    ranked = sorted((f for f in inv if ok(f)), key=lambda f: (0 if _liked(f[0], likes) else 1, key(f)))
    return (why, [f[0] for f in ranked[:n]]) if ranked else ("", [])


def prompt(inp: dict, identity: str) -> str:
    """The whole prompt for the labels mode (one call, like the original)."""
    last = " (the last one)" if inp.get("is_last") else ""
    return (
        f"{identity}\n\n"
        f"This is the user's {inp.get('ordinal', '')} check of the day{last}. Below is what happened, worked out for "
        "you from their data. There are no numbers on purpose, except the temperature.\n\n"
        + view(inp) + "\n\n"
        "Write plain text: no greeting, no sign-off, no headings, no lists. Use the descriptions as they are and do not "
        "contradict them, in your own words. Write in words, never with digits, except the temperature. Never "
        "suggest a food listed under 'Never suggest', and never suggest going outside unless outdoor walking is fine."
    )


# ── the guards (labels-with-guards mode): checked on the answer, one redraft ─
# These mirror instructions in the prompt. The evaluation's checklist (evals/message_rubric.py) is written
# separately, and a labels-without-guards setup measures how much of any gain comes from the guards alone.

_OUTSIDE = re.compile(r"\b(outside|outdoors?|park|fresh air|around the block|neighbou?rhood)\b", re.I)
# A mention is not a suggestion: the first version of this check fired on "without stepping outside" and "outdoor
# walking isn't suitable, so stay in", and its redrafts fixed answers that were already right (2026-10-04 run).
_NOT_OUT_BEFORE = re.compile(r"\b(without|instead of|rather than|avoid\w*|skip\w*|no|not|n't|never|unsuitable|"
                             r"too (?:wet|rainy|windy))\b[^.!?]{0,30}$", re.I)
_NOT_OUT_AFTER = re.compile(r"^[^.!?]{0,25}?\b(?:isn'?t|is not|not|un)\s*(?:really\s+)?(?:suitable|ideal|an option|possible|"
                            r"safe|advisable)", re.I)
_ADVICE_CUE = re.compile(r"\b(try|take|go|head|get|step|consider|aim|grab|fit|squeeze|enjoy|walk|stroll|jog|can|could|"
                         r"would|should|why not|how about|perfect for|great for|good for|a (?:short|brisk|quick|light))\b",
                         re.I)
_CONDITIONAL = re.compile(r"\bif\b[^.!?]{0,40}\b(?:weather|conditions?|dry|clear|nice|good|rain (?:stops|eases|clears)|"
                          r"it clears|it stops)\b|\bwhen (?:the rain|it) (?:stops|clears|eases)\b", re.I)


def suggests_outside(text: str, allow_conditional: bool = False) -> bool:
    """An outdoor movement idea, not a mention. With `allow_conditional` (weather unknown), 'if the weather is good,
    a walk outside' is allowed, as the original prompt allows 'a walk if conditions allow'."""
    text = (text or "").replace("’", "'").replace("‘", "'")          # the model writes "isn’t"
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not _ADVICE_CUE.search(sentence) or (allow_conditional and _CONDITIONAL.search(sentence)):
            continue
        for m in _OUTSIDE.finditer(sentence):
            if not (_NOT_OUT_BEFORE.search(sentence[:m.start()]) or _NOT_OUT_AFTER.search(sentence[m.end():])):
                return True
    return False
_SCOLD = re.compile(r"\b(falling behind|fell behind|way off|behind on (?:your )?(?:calories|protein)|not doing (?:it|well)|"
                    r"failing|slacking|off track)\b", re.I)
_STILL_GOING = re.compile(r"\b(still early|early stages|still in (?:this|the) block|rest of this block|"
                          r"this block is (?:still|just) (?:starting|beginning))\b", re.I)
_EAT_MORE = re.compile(r"\b(eat more|add (?:a |an |another )?(?:snack|meal|calories)|extra (?:snack|calories|portion)|"
                       r"bigger (?:meal|portion)|calorie-dense|energy-dense)\b", re.I)
_CATCH_UP = re.compile(r"\b(catch up on (?:your )?calories|eat more to (?:hit|reach|meet) (?:your )?calorie)", re.I)
_LIST = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s")
_NUM = re.compile(r"\d+(?:[.,:]\d+)?")
_SLIGHT = re.compile(r"\b(slightly|a (?:little|bit|touch)|marginally|just|barely)\s+(?:\w+\s+)?"
                     r"(over|under|short|behind|above|below|ahead)\b", re.I)


def problems(text: str, inp: dict) -> list[str]:
    """What is wrong with an answer, in words the redraft prompt can use. Empty = fine."""
    text = (text or "").replace("’", "'").replace("‘", "'")          # the model writes "isn’t"
    lab = classify(inp)
    shown = set(_NUM.findall(view(inp)))
    out = []
    stray = [n for n in _NUM.findall(text) if n not in shown]
    if stray:
        out.append("it uses numbers (" + ", ".join(sorted(set(stray))) + "); write in words, except the temperature")
    if lab["outdoor"] != "suitable" and suggests_outside(text, allow_conditional=lab["outdoor"] == "unknown"):
        out.append("it suggests going outside although outdoor walking is not suitable; keep movement indoors")
    # "this is not falling behind" is the reassurance the plan asks for, not scolding
    if lab["no_food"] and any(not re.search(r"\b(?:not|n't|no|never)\b[^.]{0,20}$", text[max(0, m.start() - 40):m.start()],
                                            re.I) for m in _SCOLD.finditer(text)):
        out.append("it treats nothing-logged-yet as falling behind; ask whether they have eaten or forgot to log")
    if _STILL_GOING.search(text):
        out.append("it talks about the block as if it were still going; the block has ended")
    cal = lab["metrics"]["calories"]["day_dev"]
    if not lab["no_food"] and cal >= _ON_PLAN and _EAT_MORE.search(text):
        out.append("it asks for more food although calories are over plan")
    if lab["phase"] == FAT_LOSS and cal <= -_ON_PLAN and _CATCH_UP.search(text):
        out.append("it asks to catch up on calories, which a cut does not need")
    if _LIST.search(text):
        out.append("it uses a list; write sentences")
    # a far-off metric softened to "slightly over" contradicts the description (seen in the first smoke run)
    for key, name in (("calories", r"calor\w*"), ("protein", r"protein"), ("steps", r"steps?")):
        if (lab["no_food"] and key != "steps") or abs(lab["metrics"][key]["day_dev"]) < 0.3:
            continue
        if any(re.search(rf"\b{name}\b", s, re.I) and _SLIGHT.search(s) for s in re.split(r"(?<=[.!?])\s+", text)):
            out.append(f"it calls the {key} gap slight, but the description says the day is far off plan")
    dislikes = [w for w in re.split(r"[,;/]\s*|\s+and\s+", ((inp.get("context") or {}).get("food_dislikes") or "").lower())
                if len(w) > 2]
    if any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in dislikes):
        out.append("it suggests a food the user does not eat")
    return out


def view(inp: dict) -> str:
    """The text the model reads (v1). No figures except the temperature and whatever the user typed."""
    return _view(inp, version=1)


def view_v2(inp: dict) -> str:
    """v2: the day's last check holds only how the day went; the plan keeps the size words as given."""
    return _view(inp, version=2)


_KEEP_SIZE = "Keep the size words as given (far over stays far over)."


def _view(inp: dict, version: int) -> str:
    lab = classify(inp)
    last_v2 = version >= 2 and lab["is_last"]
    start, end = inp["block_start_hour"], inp["end_hour"]
    end_min = inp.get("end_minute", 0)
    lines = [f"Phase: {lab['phase']}"]
    if lab["is_last"]:
        lines.append(f"This is the last check of the day: the block from {hour_words(start)} to "
                     f"{hour_words(end, end_min)} has ended, and so has the day's plan.")
    else:
        lines.append(f"The block that just ended ran from {hour_words(start)} to {hour_words(end, end_min)}. It is "
                     "over; anything forward-looking is for the rest of the day.")
    lines.append("")
    if lab["no_food"]:
        lines.append("Food: nothing logged yet today. That usually means not eaten yet or not logged yet, which is "
                     "not falling behind: ask which it is and nudge to log or to eat. Do not call calories or protein "
                     "behind.")
    for key in _ORDER:
        if lab["no_food"] and key in ("calories", "protein"):
            continue
        m = lab["metrics"][key]
        parts = [f"in this block {m['block']}", m["day"]]
        if m["rest"]:
            parts.append(m["rest"])
        line = f"{_TITLES[key]}: " + "; ".join(parts) + "."
        if m["meaning"]:
            line += f" For the goal: {m['meaning']}."
        lines.append(line)
    ctx = inp.get("context") or {}
    w = ctx.get("workout")
    if w:
        done = ("logged" if w.get("logged") else "not logged") if last_v2 else (
            "done" if w.get("logged") else "not done yet")
        lines.append("Workout today: " + ("cardio day" if w["kind"] == "cardio" else "rest day" if w["kind"] == "rest"
                                          else f"{w['type']}, {done}") + ".")
    why, foods = lab["foods"]
    if not last_v2:                         # v1's answers repeated these lines at 23:59; the day is over by then
        if lab["weather"]:
            lines.append(f"Weather now: {lab['weather']}.")
        lines.append({"suitable": "Outdoor walking: fine.",
                      "not suitable": "Outdoor walking: not suitable; any movement idea must be indoors.",
                      "unknown": "Outdoor walking: unknown; keep any movement idea indoors or weather-neutral."}[
            lab["outdoor"]])
        if foods:
            lines.append(f"Foods from the user's list that fit ({why}): " + ", ".join(foods) + ".")
        if ctx.get("food_dislikes"):
            lines.append(f"Never suggest: {ctx['food_dislikes']}.")
        if ctx.get("morning_feedback"):
            lines.append(f"How the user said they felt this morning: {ctx['morning_feedback']}")
    lines.append("")
    pr = lab["priorities"]
    if last_v2:
        lines.append("Plan: two or three sentences that sum up how the day went on calories, protein and steps, most "
                     f"important first. {_KEEP_SIZE} Nothing about the weather or movement ideas, and no plan for "
                     "tomorrow; the morning briefing covers that.")
    elif lab["is_last"]:
        lines.append("Plan: two or three sentences that sum up how the day went on calories, protein and steps, most "
                     "important first. No plan for tomorrow; the morning briefing covers that.")
    elif pr:
        names = {"food": "logging or eating", "calories": "calories", "protein": "protein", "steps": "steps"}
        order = ", then ".join(names[p] for p in pr)
        lines.append(f"What matters for the rest of the day, most important first: {order}.")
        lines.append("Plan: one short paragraph of three or four sentences. The first says how the block went, "
                     "starting with what matters most" + (f". {_KEEP_SIZE} The next" if version >= 2 else ". The next")
                     + " gives one concrete action for the rest of the day "
                     "for the first item" + (", using the foods listed" if foods else "")
                     + ("; if steps are on the list, add one movement idea that follows the outdoor rule" if "steps" in pr
                        else "") + ". End with one short line of encouragement.")
    else:
        lines.append("Everything is on plan: say so in two or three sentences, with one small idea to keep it going "
                     "and one short line of encouragement.")
    return "\n".join(lines)


# ── version 2 prompt and checks (CHECKPOINT_MODE=labels2 / labels2_noguard, 2026-10-04) ───────────────────────────
# From v1's failures on the tuning cases (evals/REPORT.md section 6): at the day's last check the answers repeated
# the weather and outdoor lines and ran too long; once a far-off gap became "slightly over"; and v1's checks read
# "the extra calories came from the oatmeal" as a request for more food. v1 (view, prompt, problems) is unchanged.

_EAT_MORE_REQUEST = re.compile(
    r"\b(eat more|add (?:a |an |another )?(?:snack|meal|calories)"
    r"|(?:have|grab|add|eat|get|go for)\s+(?:an?\s+|some\s+)?(?:extra|bigger|larger|second)\s+"
    r"(?:snack|calories|portion|serving|meal|plate)"
    r"|(?:go for|choose|pick|add|eat)\s+(?:some\s+)?(?:calorie|energy)-dense)\b", re.I)
_WEATHER_TALK = re.compile(r"\b(weather|outdoors?|outside|temperature|degrees|°C|sunny|rain\w*|clear sky)", re.I)


def prompt_v2(inp: dict, identity: str) -> str:
    last = bool(inp.get("is_last"))
    return (
        f"{identity}\n\n"
        f"This is the user's {inp.get('ordinal', '')} check of the day{' (the last one)' if last else ''}. Below is what "
        "happened, worked out for you from their data. There are no numbers on purpose"
        + (".\n\n" if last else ", except the temperature.\n\n")
        + view_v2(inp) + "\n\n"
        "Write plain text: no greeting, no sign-off, no headings, no lists. Use the descriptions as they are and do not "
        "contradict them, in your own words. Write in words, never with digits"
        + ("." if last else ", except the temperature. Never suggest a food listed under 'Never suggest', and never "
                            "suggest going outside unless outdoor walking is fine.")
    )


def problems_v2(text: str, inp: dict) -> list[str]:
    """v2's checks: v1's, with "eat more" read as a request only, plus the last check's length and weather talk."""
    text = (text or "").replace("’", "'").replace("‘", "'")
    lab = classify(inp)
    last = lab["is_last"]
    shown = set(_NUM.findall(view_v2(inp)))
    out = []
    stray = [n for n in _NUM.findall(text) if n not in shown]
    if stray:
        out.append("it uses numbers (" + ", ".join(sorted(set(stray))) + "); write in words"
                   + ("" if last else ", except the temperature"))
    if not last and lab["outdoor"] != "suitable" and suggests_outside(text, allow_conditional=lab["outdoor"] == "unknown"):
        out.append("it suggests going outside although outdoor walking is not suitable; keep movement indoors")
    if lab["no_food"] and any(not re.search(r"\b(?:not|n't|no|never)\b[^.]{0,20}$", text[max(0, m.start() - 40):m.start()],
                                            re.I) for m in _SCOLD.finditer(text)):
        out.append("it treats nothing-logged-yet as falling behind; ask whether they have eaten or forgot to log")
    if _STILL_GOING.search(text):
        out.append("it talks about the block as if it were still going; the block has ended")
    cal = lab["metrics"]["calories"]["day_dev"]
    if not lab["no_food"] and cal >= _ON_PLAN and any(
            not re.search(r"\b(?:not|n't|no|never|avoid|skip|without)\b[^.]{0,25}$", text[:m.start()], re.I)
            for m in _EAT_MORE_REQUEST.finditer(text)):
        out.append("it asks for more food although calories are over plan")
    if lab["phase"] == FAT_LOSS and cal <= -_ON_PLAN and _CATCH_UP.search(text):
        out.append("it asks to catch up on calories, which a cut does not need")
    if _LIST.search(text):
        out.append("it uses a list; write sentences")
    for key, name in (("calories", r"calor\w*"), ("protein", r"protein"), ("steps", r"steps?")):
        if (lab["no_food"] and key != "steps") or abs(lab["metrics"][key]["day_dev"]) < 0.3:
            continue
        if any(re.search(rf"\b{name}\b", s, re.I) and _SLIGHT.search(s) for s in re.split(r"(?<=[.!?])\s+", text)):
            out.append(f"it calls the {key} gap slight, but the description says the day is far off plan")
    dislikes = [w for w in re.split(r"[,;/]\s*|\s+and\s+", ((inp.get("context") or {}).get("food_dislikes") or "").lower())
                if len(w) > 2]
    if any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in dislikes):
        out.append("it suggests a food the user does not eat")
    if last:
        n = len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])
        if n > 4:
            out.append(f"it has {n} sentences; the day's last check is two or three")
        if _WEATHER_TALK.search(text):
            out.append("it talks about the weather or going outside; the day is over, so say only how it went")
    return out
