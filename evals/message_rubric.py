"""Frozen checklist for the checkpoint comment and the morning briefing (both designs scored by the same items).

Like rubric_v2 (the weekly review), it imports nothing from the system under test (background_checks,
checkpoint_labels, briefing_labels): every rule is computed from a case's raw figures and the answer text, so changing
a prompt, a label or a guard cannot change how answers are measured. Bump MESSAGE_RUBRIC_VERSION whenever a rule
changes; the report re-scores every saved answer with the current version.

Items are only added where they apply (a rain rule only on wet days), so compare per-item pass rates.
"""
from __future__ import annotations

import re

from evals.rubric_v2 import Item

# v1.1: every 10%+ miss, not only the biggest one. v1.2 (after reading every flag of the first run): the outdoor rule
# flags suggestions, not mentions ("without stepping outside" was flagged); "see you" in "great to see you hit your
# steps" is not a sign-off; "20-30 minutes" is a proposal; "make sure to hit your protein goal today" is advice, not
# praise; on a day with no workout plan, a note about the plan counts as the workout note.
# v1.3: curly apostrophes normalised before scoring ("outdoor walking isn’t suitable" was read as a suggestion, since
# every negation pattern expected "isn't"); "overplanning" counts as a note on a day without a plan.
# v1.4 (reading the holdout flags): "eat more" needs a request ("the extra calories may have come from the oatmeal"
# and "the oatmeal you logged, which is calorie-dense" describe the past); an outdoor suggestion needs a movement word
# ("enjoy the light rain outside" is not one).
# v1.5 (2026-10-05), after a real 12:00 message: "light squats or lunges at home to add to your steps" in the rain.
# New items: steps are walking (no exercises named), no invented workouts, suggestions fit the time of day, no
# supplement offered as food. The weather rule gains great heat (32 C+) and strong wind (40 km/h+): a walk is then
# fine only next to an indoor alternative (treadmill, stairs, mall), as a sunny day is no automatic "go outside".
# v1.6 (2026-10-05): the day's last check is over, so it may not talk as if the day went on ("on track", "still time",
# "keep going"); the user's review of version 3.
# v1.7 (2026-10-05): by_workout_weighin_right read "Don't forget to get your weigh-in done" as praise for a missed
# weigh-in (advice, not praise: 3 false flags) and did not flag "You nailed the weigh-in" (a miss). Praise is now read
# per sentence, and a sentence that looks forward (today, don't forget, make sure, get ... done) is advice.
MESSAGE_RUBRIC_VERSION = "messages-v1.7 (2026-10-05)"
KINDS = ("hallucination", "instruction", "semantic")
NAME = "G"                                   # the user's name in the evaluation's identity header

ITEM_DOCS = {
    # checkpoint
    "cp_no_invented_numbers": "every figure matches the case's data within 2% (targets, totals, this block, what is "
                              "left, the food list, weather, clock hours; a percentage within 5 points of a real gap); "
                              "proposals such as 'a 15-minute walk', '30g of protein' or 'aim for 2,000 more steps' "
                              "are allowed",
    "cp_daily_budget_claim_true": "never says the day's calorie budget is exceeded or used up while the day's total is "
                                  "still under the daily target",
    "cp_calorie_position_correct": "a sentence about calories alone never says over when the day so far is under, or "
                                   "under when it is over",
    "cp_size_right": "a gap of 30%+ (in the block and the day alike) is never called slight, and one under 10% is "
                     "never called far or way off",
    "cp_right_direction": "no 'eat more' while calories are over plan, no 'catch up on calories' in a cut, no "
                          "'eat less' while under plan outside a cut, no 'more protein' once the day's protein is met",
    "cp_no_greeting_or_sign_off": "no greeting, no valediction ('let me know', 'cheers', 'have a great day')",
    "cp_no_headings_or_lists": "plain sentences, no headings, no lists",
    "cp_length": "two to nine sentences (two to five for the day's last check)",
    "cp_block_is_over": "never talks about the block as if it were still going ('still early', 'early stages')",
    "cp_last_check_stays_a_summary": "the day's last check (23:59) gives no advice for a next meal, dinner or tonight",
    "cp_last_check_is_over": "the day's last check does not talk as if the day went on: no 'on track', 'still time', "
                             "'keep going', 'catch up', 'rest of the day' or 'tomorrow'",
    "cp_outdoor_rule": "no outdoor movement suggestion when outdoor walking is not suitable or unknown (a mention such "
                       "as 'without stepping outside' is not one; a conditional one, 'if the weather is good', is "
                       "allowed only when the weather is unknown, as the original prompt allows)",
    "cp_no_food_handled_kindly": "with nothing logged: no scolding, and it asks about or nudges logging or eating",
    "cp_no_disliked_food": "never suggests a food the user dislikes (a disliked 'fish' covers named fish)",
    "cp_food_fits_the_day": "with calories over plan no energy-dense food is suggested; with protein behind a "
                            "concrete protein food is named",
    "cp_every_problem_addressed": "every metric off plan in a way that matters for the phase is mentioned (calories "
                                  "5%+ over, or under outside a cut; protein or steps 5%+ behind)",
    "cp_steps_are_walking": "steps are walking: no exercise is named (squats, lunges, push-ups, circuits, yoga, sets, "
                            "reps, bodyweight work), however it is framed",
    "cp_no_invented_workout": "no extra, additional or different workout is suggested; after a session is done, or on a "
                              "rest day, no workout at all (training comes only from the user's plan)",
    "cp_time_fits": "suggestions fit the hour: no breakfast from 11:00, no lunch from 15:00, no dinner or snack from "
                    "22:00, no walking idea from 22:00 (a check before the last one)",
    "cp_no_supplement_as_food": "a supplement in the user's food list (fish oil, creatine, vitamins) is not offered as "
                                "food",
    "cp_no_internal_codes": "none of the system's internal terms reach the user ('status 3', 'raw carry', "
                            "'scenario', 'expected' as a field)",
    # briefing, yesterday part
    "by_greeting_and_name_once": "starts with a greeting and uses the user's name exactly once",
    "by_length": "two to five sentences including the greeting",
    "by_no_repeated_numbers": "no figures (proposals such as '30 minutes' allowed)",
    "by_no_lists_or_sign_off": "plain sentences, no list, no valediction",
    "by_no_wrong_praise": "never praises a metric that was off, or calls a met one short",
    "by_workout_weighin_right": "never praises a workout or weigh-in that was not logged, or calls a logged one missed",
    "by_big_misses_mentioned": "every metric that missed by 10% or more yesterday (steps or protein short, calories "
                               "over or under) is mentioned",
    # briefing, today part
    "bt_no_greeting_or_name": "continues the briefing: no greeting, no name",
    "bt_length": "two to four sentences",
    "bt_no_repeated_numbers": "no figures except the temperature (proposals such as '30 minutes' allowed)",
    "bt_workout_day_right": "a training day is never called a rest or recovery day",
    "bt_no_lists_or_sign_off": "plain sentences, no list, no valediction",
    "bt_covers_food_movement_workout": "a food idea, a movement idea and a note on today's workout or rest",
    "bt_outdoor_rule": "as cp_outdoor_rule, for today's movement idea",
    "bt_budget_direction_right": "when today's calorie target was lowered or raised, the answer never says the "
                                 "opposite",
    "bt_food_fits_budget": "a lowered target gets no hearty/big/energy-dense breakfast; a raised one is not told to "
                           "eat less",
    "bt_rest_day_respected": "on a rest day, no push to train",
    "bt_steps_are_walking": "the step idea is walking: no exercise is named (squats, circuits, yoga, sets, reps)",
    "bt_no_invented_workout": "no extra, additional or different workout is suggested; on a rest day, none at all",
}

# ── shared text helpers ──────────────────────────────────────────────────────

_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
_TIMES = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*[-–]\s*\d{1,2}\s*(?:am|pm)\b", re.I)
# proposals: amounts the answer suggests rather than claims about the day
_PROPOSAL = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:[-–]|to)\s*\d+(?:\.\d+)?)?\s*-?\s*(?:minutes?|mins?|hours?|hrs?|reps?|sets?|times|seconds?|"
    r"secs?|sessions?|laps?|flights?)\b"
    r"|\b(?:at least|extra|additional|another|up to|aim for|add(?:ing)?|grab|have|eat|take|go for|walk|get in|"
    r"squeeze in|fit in|try)\s+(?:an?\s+)?(?:extra\s+|another\s+)?\d[\d,]*(?:\.\d+)?\s*(?:g|grams?|steps?|kcal|cal|"
    r"calories)?\b"
    r"|\b\d[\d,]*(?:\.\d+)?\s*(?:more|extra|additional)\s+(?:steps|kcal|calories|grams|g)\b"
    r"|\ba\s+\d+(?:\.\d+)?\s*-?\s*(?:g|grams?|calories?|kcal|cal|steps?)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:g|grams?)\s+(?:of\s+)?[a-z]"
    r"|\b\d+(?:\.\d+)?\s*(?:eggs?|egg whites|slices?|cups?|scoops?|servings?|pieces?|handfuls?|tbsp|tablespoons?|oz|"
    r"bowls?|glasses?)\b", re.I)
_ADVICE_VERB = re.compile(r"\b(keep|aim|choose|pick|go for|opt|try|have|eat|grab|stick|make|plan|target|swap|limit)\b",
                          re.I)
_BOUND = re.compile(r"\b(?:under|below|less than|no more than|at most|max(?:imum)?|around|about|roughly|within)\s+"
                    r"\d[\d,]*\s*(?:kcal|calories|cal)\b", re.I)
_NEG = re.compile(r"\b(?:without|avoid\w*|not|never|no|don'?t|do not|instead of|rather than|skip\w*|swap\w*|limit\w*|"
                  r"cut\w* back on|hold off on|go easy on|ease off|less|fewer)\b[^.]{0,30}$", re.I)
# a greeting opens the text, possibly after the name ("G, good morning!")
_GREETING = re.compile(rf"^\W*(?:{NAME}\W+)?(hi|hey|hello|good (?:morning|afternoon|evening)|morning[,!])", re.I)
_SIGNOFF = re.compile(r"(let me know|good luck|cheers|best,|talk soon|see you (?:soon|tomorrow|later|tonight)|"
                      r"have an? (?:great|good|nice|wonderful|fantastic|amazing|productive) (?:day|one|morning|evening))",
                      re.I)
_LIST = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s")
_HEADING = re.compile(r"(?m)^\s*(?:#|\*\*[^*\n]{1,40}\*\*\s*:?\s*$)")
_OUTSIDE = re.compile(r"\b(outside|outdoors?|park|fresh air|around the block|neighbou?rhood|in the sun)\b", re.I)
_OUT_DENIED_BEFORE = re.compile(r"\b(without|instead of|rather than|avoid\w*|skip\w*|no|not|n't|never|unsuitable|"
                                r"too (?:wet|rainy|windy)|limit\w*|made|impossible)\b[^.!?]{0,30}$", re.I)
_OUT_DENIED_AFTER = re.compile(r"^[^.!?]{0,25}?\b(?:isn'?t|is not|not|un)\s*(?:really\s+)?(?:suitable|ideal|an option|"
                               r"possible|safe|advisable)|^[^.!?]{0,25}?\bimpossible\b", re.I)
_OUT_ADVICE = re.compile(r"\b(try|take|go|head|get|step|consider|aim|grab|fit|squeeze|enjoy|walk|stroll|jog|can|could|"
                         r"would|should|why not|how about|perfect for|great for|good for|a (?:short|brisk|quick|light))\b",
                         re.I)
_OUT_MOVE = re.compile(r"\b(walk\w*|stroll\w*|jog\w*|run\w*|hik\w*|bik\w*|cycl\w*|mov\w*|steps?|activ\w*|exercis\w*|"
                       r"head out|go out|get out|step out)\b", re.I)
_OUT_IF = re.compile(r"\bif\b[^.!?]{0,40}\b(?:weather|conditions?|dry|clear|nice|good|rain (?:stops|eases|clears)|"
                     r"it clears|it stops)\b|\bwhen (?:the rain|it) (?:stops|clears|eases)\b", re.I)


_ALT_INDOOR = re.compile(r"\b(treadmill|indoors?|inside|stairs|mall|covered|air-?condition\w*|pacing)\b", re.I)


def suggests_outdoors(text: str, conditional_ok: bool, alt_ok: bool = False) -> str:
    """The first outdoor SUGGESTION ('take a walk outside'), or ''. A mention is not one ('without stepping outside',
    'outdoor walking isn't suitable, so stay in', 'the rain limited outdoor activity'). With `alt_ok` (great heat,
    strong wind) a sentence that also offers an indoor alternative ('a walk or the treadmill') is allowed."""
    for s in re.split(r"(?<=[.!?])\s+", text or ""):
        if not _OUT_ADVICE.search(s) or not _OUT_MOVE.search(s) or (conditional_ok and _OUT_IF.search(s)):
            continue
        if alt_ok and _ALT_INDOOR.search(s):
            continue
        for m in _OUTSIDE.finditer(s):
            if not (_OUT_DENIED_BEFORE.search(s[:m.start()]) or _OUT_DENIED_AFTER.search(s[m.end():])):
                return s.strip()
    return ""
_METRIC = {"calories": re.compile(r"\b(calor\w*|kcal|food|eat\w*|ate|meals?|portions?|intake|surplus|deficit)\b", re.I),
           "protein": re.compile(r"\bprotein\b", re.I),
           "steps": re.compile(r"\b(steps?|walk\w*|movement|move\w*|active)\b", re.I)}
_PROTEIN_FOOD = re.compile(r"\b(shakes?|whey|eggs?|egg whites|chicken|fish|tuna|salmon|yogh?urt|cottage cheese|tofu|"
                           r"beef|turkey|beans|lentils|cheese|milk|quark|tempeh|edamame|shrimp|prawns|jerky|"
                           r"protein (?:shake|bar|powder|smoothie|pudding))\b", re.I)
_FISH = r"(?:fish|salmon|tuna|cod|tilapia|sardines?|mackerel|trout|halibut|haddock|sea bass|anchov\w*)"


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def clauses(text: str) -> list[str]:
    parts = re.split(r"[.;!?,:]|\b(?:but|while|though|although|however|and|whereas)\b", text or "", flags=re.I)
    return [p.strip() for p in parts if p and p.strip()]


def _asks(pattern: re.Pattern, text: str) -> bool:
    """A match that is not negated within its sentence ('avoid the trail mix' does not ask for trail mix)."""
    for m in pattern.finditer(text):
        start = max(text.rfind(".", 0, m.start()), text.rfind("!", 0, m.start()), text.rfind("?", 0, m.start())) + 1
        if not _NEG.search(text[start:m.start()]):
            return True
    return False


def _numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUMBER.findall(text or "")]


def _rainy(weather: dict | None) -> bool:
    return bool(weather) and ((weather.get("weather_code") or 0) >= 51 or (weather.get("precipitation_mm") or 0) >= 0.5)


def _soft_bad(weather: dict | None) -> bool:
    """Great heat or strong wind without rain: outside is possible, but not the default."""
    return bool(weather) and not _rainy(weather) and ((weather.get("temp_c") or 0) >= 32 or (weather.get("wind_kph") or 0) >= 40)


def _outdoor_ok(weather: dict | None) -> bool:
    return bool(weather) and not _rainy(weather) and not _soft_bad(weather)


# steps are walking: exercises are not steps (the real message said "light squats or lunges at home to add to your steps")
_EXERCISE = re.compile(r"\b(squats?|lunges?|push-?ups?|pull-?ups?|sit-?ups?|planks?|burpees?|jumping jacks?|crunch(?:es)?|"
                       r"calisthenics|body-?weight|hiit|circuit|yoga|pilates|dumbbells?|kettlebells?|deadlifts?|"
                       r"resistance bands?|home workout|workout at home|sets? of|reps?)\b", re.I)
_EXTRA_WORKOUT = re.compile(r"\b(?:extra|another|additional|second|different|new)\s+(?:\w+\s+)?(?:workout|training session|"
                            r"gym session|strength session)\b|\badd(?:ing)?\b[^.!?]{0,15}\b(?:workout|training session|"
                            r"gym session)\b", re.I)
_DO_WORKOUT = re.compile(r"\b(?:do|try|squeeze in|fit in|get in|start|begin)\b[^.!?]{0,20}\b(?:workout|training session|"
                         r"gym session|exercise session)\b", re.I)
_SUPPLEMENT_NAME = re.compile(r"\b(oil|vitamins?|creatine|supplements?|capsules?|tablets?|omega|probiotic|electrolytes?|bcaa)\b",
                              re.I)
_MEAL_OVER_AT = {"breakfast": 11, "lunch": 15, "dinner": 22, "snack": 22}
_EATING = re.compile(r"\b(try|have|eat|grab|choose|opt|add|go for|plan|start|enjoy|make|prepare)\b", re.I)
_TOMORROW = re.compile(r"\b(tomorrow|tonight's|next morning)\b", re.I)
_LAST_OVER = re.compile(r"\b(on track|off track|back on track|still (?:have )?time|plenty of (?:time|day)|keep (?:it )?up|"
                        r"keep going|stay on track|catch up|rest of the day|later today|next block|next meal|tomorrow|"
                        r"from here)\b", re.I)


def _dislike_patterns(dislikes: str) -> list[tuple[str, str]]:
    out = []
    for d in re.split(r"[,;/]\s*|\s+and\s+", (dislikes or "").lower()):
        d = d.strip()
        if len(d) > 2:
            out.append((d, _FISH if d == "fish" else re.escape(d)))
    return out


# ── checkpoint ───────────────────────────────────────────────────────────────

def checkpoint_truth(case: dict) -> dict:
    """Ground truth from the case's raw figures (not from any label module)."""
    cp, cum, goals, ctx = case["cp"], case["cum"], case["goals"], case["context_data"]
    frac = cp["frac"]
    daily = {"calories": goals["calories_target"], "protein": goals["protein_target"], "steps": goals["step_goal"]}
    dev = {k: (cum[k] / (daily[k] * frac) - 1) if daily[k] * frac else 0.0 for k in daily}
    phase = case["phase"]
    no_food = round(cum["calories"]) == 0 and round(cum["protein"]) == 0
    problems = []
    if not no_food:
        if dev["calories"] >= 0.05 or (dev["calories"] <= -0.05 and phase != "Fat Loss"):
            problems.append("calories")
        if dev["protein"] <= -0.05:
            problems.append("protein")
    if dev["steps"] <= -0.05:
        problems.append("steps")
    figures, pcts = set(), {5.0, 20.0}
    for k in daily:
        m = cp[k]
        figures |= {float(daily[k]), float(cum[k]), float(daily[k] * frac), float(round(daily[k] * frac)),
                    abs(float(daily[k] - cum[k])), float(daily[k] * (1 - frac)), float(m["live"]), float(m["expected"]),
                    abs(float(m["nextb_raw"])), abs(float(m["nextb"])), abs(float(m["live"] - m["expected"]))}
        pcts.add(abs(dev[k]) * 100)
        if m["expected"]:
            pcts.add(abs(m["live"] / m["expected"] - 1) * 100)
    for _name, kcal, prot in ctx.get("inventory") or []:
        figures |= {float(kcal), float(prot)}
    for e in ctx.get("food_entries") or []:
        figures |= {float(e["calories"]), float(round(e["calories"])), float(e["protein"]), float(round(e["protein"]))}
    w = ctx.get("weather")
    if w:
        figures |= {float(w.get("temp_c") or 0), float(round(w.get("temp_c") or 0)), float(w.get("wind_kph") or 0),
                    float(w.get("precipitation_mm") or 0)}
    figures |= {12.0, 15.0, 19.0, 23.0, 11.0, 59.0, 0.0, 30.0, float(cp["hour"]), float(cp["block_start_hour"])}
    block_dev = {k: (cp[k]["live"] / cp[k]["expected"] - 1) if cp[k]["expected"] else None for k in daily}
    return {"phase": phase, "dev": dev, "block_dev": block_dev, "outdoor_ok": _outdoor_ok(w), "is_last": cp["is_last"],
            "no_food": no_food, "weather_unknown": not w, "weather_soft": _soft_bad(w),
            "hour": cp["hour"] + (1 if str(cp.get("label", "")).endswith(("59",)) else 0),
            "workout": ctx.get("workout") or {}, "inventory": [n for n, _k, _p in (ctx.get("inventory") or [])],
            "problems": problems, "dislikes": _dislike_patterns(ctx.get("food_dislikes") or ""), "figures": figures,
            "pcts": pcts, "cal_under_daily": cum["calories"] < daily["calories"] * 0.98,
            "protein_met": cum["protein"] >= daily["protein"]}


def invented_numbers(text: str, figures: set, pcts: set) -> list[float]:
    """Figures in the answer that match nothing in the data (proposals, clock times and small counts excluded)."""
    body = _TIMES.sub(" ", text or "")
    bad = [float(p) for p in _PERCENT.findall(body) if float(p) > 7 and not any(abs(float(p) - q) <= 5 for q in pcts)]
    body = _PERCENT.sub(" ", body)
    kept = []
    for c in re.split(r"(?<=[.;!?])\s+", body):
        c = _PROPOSAL.sub(" ", c)
        if _ADVICE_VERB.search(c):
            c = _BOUND.sub(" ", c)
        kept.append(c)
    for n in _numbers(" ".join(kept)):
        n = abs(n)
        if n > 7 and not any(abs(n - f) <= 0.5 or (f and abs(n - f) / f <= 0.02) for f in figures):
            bad.append(n)
    return sorted(set(bad))


_DAILY_BUDGET_GONE = re.compile(
    r"\b(?:over|above|exceed\w*|blown|past|beyond|used up|maxed out)\b[^.]{0,25}\b(?:daily|day's|whole day's|full day's|"
    r"entire day's|total|for the day)\b[^.]{0,15}\b(?:budget|target|goal|limit|allowance)\b"
    r"|\b(?:daily|day's)\s+(?:calorie\s+)?(?:budget|target|goal|limit|allowance)\b[^.]{0,25}\b(?:is |was |has been )?"
    r"(?:exceeded|blown|used up|gone|maxed out|already (?:hit|reached|met))\b", re.I)
_CAL_NAME = re.compile(r"\b(calor\w*|kcal)\b", re.I)
_OTHER_METRIC = re.compile(r"\b(protein|steps?|walk\w*|movement)\b", re.I)
_NOT_POSITION = re.compile(
    r"\bunder control\b|\bover the (?:next|rest|course|afternoon|evening|morning|day|coming)\b|\bover time\b"
    r"|\b(?:under|below|less than|fewer than|over|above|more than)\s+\d[\d,]*"
    r"|\b(?:a|the|one|quick|brief)\s+short\b|\bshort (?:walk|stroll|break|session|burst|bout|while|time)\b"
    r"|\bbehind you\b|\bahead (?:of (?:the|your) (?:next|evening|workout|dinner|meal))|\b(?:day|evening|week) ahead\b",
    re.I)
_SAYS_OVER = re.compile(r"\b(over|above|exceed\w*|too much|ahead of (?:plan|pace|target|schedule)|more than planned|"
                        r"overshot|overate|surplus of)\b", re.I)
_SAYS_UNDER = re.compile(r"\b(under|below|short of|fell short|falls? short|came up short|a (?:bit|little) short|"
                         r"behind|less than planned|fewer than planned|not enough|undershot|room left|"
                         r"room to spare|still have room)\b", re.I)
_EAT_MORE = re.compile(r"\b(eat more|add (?:a |an |another )?(?:snack|meal|calories)|load up"
                       r"|(?:have|add|grab|eat|include|get|take|go for|aim for|fit in|squeeze in)\s+(?:an?\s+|some\s+)?"
                       r"(?:extra|bigger|larger|second)\s+(?:snack|calories|portion|food|serving|meal|plate)"
                       r"|(?:add\w*|eat\w*|have|prioriti[sz]\w*|includ\w*|choos\w*|go for|opt for|consider|more)\b[^.]{0,30}"
                       r"\b(?:calorie-dense|energy-dense))\b", re.I)
_CATCH_UP = re.compile(r"\b(catch up on (?:your )?calories|make up (?:the|those|your) calories|"
                       r"eat more to (?:hit|reach|meet))", re.I)
_EAT_LESS = re.compile(r"\b(eat less|cut back|lighter (?:meals?|dinner|lunch|options?|choices?|portions?)|"
                       r"smaller (?:meals?|portions?|plates?)|skip (?:a |the )?(?:meal|snack|dinner|lunch))\b", re.I)
_MORE_PROTEIN = re.compile(r"\b(?:more|extra|another|additional|add\w*|top up|boost\w*)\b[^.]{0,25}\bprotein\b", re.I)
_DENSE = re.compile(r"\b(peanut butter|trail mix|nuts|granola|rice bowl|pasta|pizza|burger|fries|chocolate|cake|"
                    r"pastry|ice cream|(?<!cottage )cheese(?!cake)|calorie-dense|energy-dense)\b", re.I)
_SCOLD = re.compile(r"\b(falling behind|fell behind|way off|behind on (?:your )?(?:calories|protein)|not doing (?:it|well)|"
                    r"failing|slacking|off track|missed your (?:calories|protein))\b", re.I)
_SCOLD_NEG = re.compile(r"\b(?:not|n't|no|never|isn'?t|aren'?t|doesn'?t|don'?t)\b[^.]{0,20}$", re.I)
_LOG_OR_EAT = re.compile(r"\b(log\w*|eat\w*|ate|meal|breakfast|lunch|snack|forg[eo]t)\b", re.I)
_STILL_GOING = re.compile(r"\b(still early|early stages|still in (?:this|the) block|rest of this block|"
                          r"this block is (?:still|just) (?:starting|beginning))\b", re.I)
_FORWARD = re.compile(r"\b(next meal|for dinner|dinner|this evening|tonight|before bed|rest of the (?:day|evening)|"
                      r"late[- ]night snack|evening snack)\b", re.I)
_FORWARD_ADVICE = re.compile(r"\b(try|aim|consider|grab|have|eat|add|take|go for|make sure|focus|choose|opt|keep|"
                             r"plan|should|could|would help|swap)\b", re.I)
_SLIGHT = re.compile(r"\b(slightly|a (?:little|bit|touch)|marginally|just|barely|a tad)\s+(?:\w+\s+)?"
                     r"(over|under|short|behind|above|below|ahead)\b", re.I)
_LARGE = re.compile(r"\b(way|far|massively|significantly|seriously|hugely|well|substantially|considerably)\s+"
                    r"(over|under|short|behind|above|below|ahead|off)\b", re.I)
_INTERNAL = re.compile(r"\b(status\s*[123]|raw carry|scenario\s*\d|nextb|block_expected|expected \d)", re.I)
_METRIC_NAME = {"calories": re.compile(r"\b(calor\w*|kcal)\b", re.I), "protein": re.compile(r"\bprotein\b", re.I),
                "steps": re.compile(r"\b(steps?|walking)\b", re.I)}


def _plain(text: str) -> str:
    return (text or "").replace("’", "'").replace("‘", "'")


def score_checkpoint(text: str, case: dict) -> list[Item]:
    t = checkpoint_truth(case)
    text = _plain(text)
    out: list[Item] = []
    add = lambda kind, iid, ok, detail="": out.append(Item(kind, iid, bool(ok), "" if ok else detail))
    sents = sentences(text)
    cal = t["dev"]["calories"]

    bad = invented_numbers(text, t["figures"], t["pcts"])
    add("hallucination", "cp_no_invented_numbers", not bad, f"not in the data: {bad}")
    if not t["no_food"]:
        m = _DAILY_BUDGET_GONE.search(text)
        add("hallucination", "cp_daily_budget_claim_true", not (m and t["cal_under_daily"]),
            f"says {m.group(0)!r} but the day's total is under the daily target" if m else "")
    if not t["no_food"] and abs(cal) >= 0.05:
        misstated = []
        for s in sents:
            if not _CAL_NAME.search(s) or _OTHER_METRIC.search(s):
                continue                        # only sentences about calories alone
            s2 = _NOT_POSITION.sub(" ", s)
            over, under = _SAYS_OVER.search(s2), _SAYS_UNDER.search(s2)
            if (cal > 0 and under and not over) or (cal < 0 and over and not under):
                misstated.append(s)
        add("hallucination", "cp_calorie_position_correct", not misstated,
            f"the day is {'over' if cal > 0 else 'under'}, but: {misstated[0][:100]!r}" if misstated else "")
    sized, eligible = [], False
    for key, pat in _METRIC_NAME.items():
        if t["no_food"] and key != "steps":
            continue
        day, block = abs(t["dev"][key]), t["block_dev"][key]
        both_large = day >= 0.3 and block is not None and abs(block) >= 0.3
        both_small = day < 0.1 and (block is None or abs(block) < 0.1)
        if not (both_large or both_small):
            continue
        eligible = True
        for c in clauses(text):
            if not pat.search(c) or sum(1 for p in _METRIC_NAME.values() if p.search(c)) != 1:
                continue
            if both_large and _SLIGHT.search(c):
                sized.append(f"{key} {day:+.0%} called {_SLIGHT.search(c).group(0)!r}")
            if both_small and _LARGE.search(c):
                sized.append(f"{key} {day:+.0%} called {_LARGE.search(c).group(0)!r}")
    if eligible:
        add("hallucination", "cp_size_right", not sized, "; ".join(sized))
    wrong = []
    if not t["no_food"] and cal >= 0.05 and _asks(_EAT_MORE, text):
        wrong.append("asks for more food while over plan")
    if t["phase"] == "Fat Loss" and not t["no_food"] and cal <= -0.05 and _asks(_CATCH_UP, text):
        wrong.append("asks to catch up on calories in a cut")
    if t["phase"] != "Fat Loss" and not t["no_food"] and cal <= -0.05 and _asks(_EAT_LESS, text):
        wrong.append("asks for less food while under plan outside a cut")
    if t["protein_met"] and _asks(_MORE_PROTEIN, text):
        wrong.append("asks for more protein although the day's protein is met")
    add("hallucination", "cp_right_direction", not wrong, "; ".join(wrong))

    add("instruction", "cp_no_greeting_or_sign_off", not _GREETING.search(text) and not _SIGNOFF.search(text),
        (_GREETING.search(text) or _SIGNOFF.search(text) or [""])[0] if (_GREETING.search(text) or _SIGNOFF.search(text))
        else "")
    add("instruction", "cp_no_headings_or_lists", not _LIST.search(text) and not _HEADING.search(text))
    lo, hi = (2, 5) if t["is_last"] else (2, 9)
    add("instruction", "cp_length", lo <= len(sents) <= hi, f"{len(sents)} sentences")
    m = _STILL_GOING.search(text)
    add("instruction", "cp_block_is_over", not m, m.group(0) if m else "")

    if t["is_last"]:
        fwd = [s for s in sents if _FORWARD.search(s) and _FORWARD_ADVICE.search(s)]
        add("semantic", "cp_last_check_stays_a_summary", not fwd, fwd[0][:100] if fwd else "")
    if not t["outdoor_ok"]:
        o = suggests_outdoors(text, conditional_ok=t["weather_unknown"], alt_ok=t["weather_soft"])
        add("semantic", "cp_outdoor_rule", not o, o[:100])
    if t["no_food"]:
        scold = [m.group(0) for m in _SCOLD.finditer(text) if not _SCOLD_NEG.search(text[max(0, m.start() - 40):m.start()])]
        add("semantic", "cp_no_food_handled_kindly", not scold and _LOG_OR_EAT.search(text),
            f"scolds: {scold[0]}" if scold else "does not ask about logging or eating")
    if t["dislikes"]:
        hit = [d for d, pat in t["dislikes"] if _asks(re.compile(rf"\b{pat}", re.I), text)]
        add("semantic", "cp_no_disliked_food", not hit, f"suggests {hit}")
    if not t["is_last"] and not t["no_food"] and (cal >= 0.05 or "protein" in t["problems"]):
        issues = []
        if cal >= 0.05 and _asks(_DENSE, text):
            issues.append("suggests energy-dense food while over plan")
        if "protein" in t["problems"] and not _PROTEIN_FOOD.search(text):
            issues.append("names no protein food while protein is behind")
        add("semantic", "cp_food_fits_the_day", not issues, "; ".join(issues))
    if t["problems"]:
        missing = [p for p in t["problems"] if not _METRIC[p].search(text)]
        add("semantic", "cp_every_problem_addressed", not missing, f"not mentioned: {missing}")
    if t["is_last"]:
        fr = [m.group(0) for m in _LAST_OVER.finditer(text) if not _NEG.search(text[max(0, m.start() - 40):m.start()])]
        add("semantic", "cp_last_check_is_over", not fr, f"talks as if the day went on: {fr[0]!r}" if fr else "")
    ex = [m.group(0) for m in _EXERCISE.finditer(text) if not _NEG.search(text[max(0, m.start() - 40):m.start()])]
    add("semantic", "cp_steps_are_walking", not ex, f"names an exercise: {ex[0]!r}" if ex else "")
    w_ = t["workout"]
    done_or_rest = w_.get("kind") == "rest" or (w_.get("kind") == "session" and w_.get("logged"))
    inv = [m.group(0) for m in _EXTRA_WORKOUT.finditer(text) if not _NEG.search(text[max(0, m.start() - 40):m.start()])]
    if done_or_rest:
        inv += [m.group(0) for m in _DO_WORKOUT.finditer(text) if not _NEG.search(text[max(0, m.start() - 40):m.start()])]
    add("semantic", "cp_no_invented_workout", not inv, f"suggests a workout: {inv[0]!r}" if inv else "")
    if not t["is_last"]:
        late = []
        for word, over_at in _MEAL_OVER_AT.items():
            for m in re.finditer(rf"\b{word}s?\b", text, re.I):
                around = text[max(0, m.start() - 50):m.end() + 20]
                if (t["hour"] >= over_at and _EATING.search(around) and not _TOMORROW.search(around)
                        and not _NEG.search(text[max(0, m.start() - 40):m.start()])):
                    late.append(f"{word} at hour {t['hour']}")
        if t["hour"] >= 22:
            for sent in sentences(text):
                if re.search(r"\b(walk\w*|treadmill|stairs|pacing|stroll\w*)\b", sent, re.I) and re.search(
                        r"\b(try|take|go|get|add|do|fit|squeeze|aim|consider)\b", sent, re.I):
                    late.append("a walking idea after 22:00")
                    break
        add("semantic", "cp_time_fits", not late, "; ".join(late))
    supplements = [n for n in t["inventory"] if _SUPPLEMENT_NAME.search(n)]
    if supplements:
        hit = [n for n in supplements if re.search(rf"\b{re.escape(n.lower())}\b", text.lower())]
        add("semantic", "cp_no_supplement_as_food", not hit, f"offers {hit[0]!r} as food" if hit else "")
    code = _INTERNAL.search(text)
    add("semantic", "cp_no_internal_codes", not code, code.group(0) if code else "")
    return out


# ── briefing ─────────────────────────────────────────────────────────────────

def briefing_truth(case: dict) -> dict:
    tr = case["truth"]
    w = tr["weather"]
    steps_r = tr["steps"] / tr["step_goal"] if tr["step_goal"] else 1
    cal_d = (tr["calories"] - tr["calories_target"]) / tr["calories_target"] if tr["calories_target"] else 0
    prot_r = tr["protein"] / tr["protein_target"] if tr["protein_target"] else 1
    gaps = {"steps": max(0.0, 1 - steps_r), "protein": max(0.0, 1 - prot_r),
            "calories": abs(cal_d) if abs(cal_d) > 0.03 else 0.0}
    top = max(gaps, key=gaps.get) if max(gaps.values()) > 0.03 else None
    # one "biggest" gap was too narrow: steps 14% short and calories 10% over are both worth a line (v1.1)
    big = [k for k in ("steps", "calories", "protein") if gaps[k] >= 0.10]
    adj = tr["adjustment"]
    strategy = adj[0] if adj else None
    y = case["original"]["yesterday"]
    return {"outdoor_ok": _outdoor_ok(w), "weather_soft": _soft_bad(w), "steps_met": steps_r >= 1, "steps_below": steps_r < 0.97,
            "cal_off": abs(cal_d) > 0.03, "protein_met": prot_r >= 1, "protein_short": prot_r < 0.97, "top": top,
            "big_misses": big,
            "lower_today": strategy == "calories", "higher_today": strategy == "calories_boost",
            "rest_day": (tr["workout_today"] or "").lower().startswith("rest"),
            "training_day": not re.match(r"(rest|none|cardio|unknown)", (tr["workout_today"] or "").lower())
                            and bool(tr["workout_today"]),
            "no_plan": (tr["workout_today"] or "").lower().startswith("none"),
            "temp": w.get("temp_c") if w else None,
            "workout_day": y["yesterday_was_workout_day"], "workout_logged": y["yesterday_workout_logged"],
            "weight_logged": y["yesterday_weight_logged"]}


# praise is a claim of achievement; "to stay on track" or "a great way to" is not (smoke run, 2026-10-04)
_PRAISE = (r"(?:hit|nailed|smashed|crushed|reached|met|spot on|right on|on target|perfect|excellent|great job|well done|"
           r"nice work|good job|strong|solid)")
_FORWARD_LOOKING = re.compile(r"\b(today|tomorrow|let'?s|make sure|don'?t forget|remember|aim|try|focus|need|should|"
                              r"will|going to|follow|keep|to (?:hit|reach|meet|get))\b", re.I)
_SHORT = r"(?:short|missed|below|under|behind|fell|lagg\w*|not enough|didn'?t (?:hit|reach|meet)|low)"
_WORKOUT_PRAISE = re.compile(r"\b(great|nice|solid|good|strong|well done|awesome|excellent)\b[^.]{0,25}\b(workout|session|"
                             r"training|lift\w*)\b|\b(completed|finished|crushed|smashed|nailed|got)\b[^.]{0,15}"
                             r"\b(workout|session|training)\b", re.I)
_WORKOUT_MISSED = re.compile(r"\b(missed|skipped)\b[^.]{0,20}\b(workout|session|training)\b|\b(workout|session)\b[^.]{0,20}"
                             r"\b(wasn'?t|was not|not) (logged|done|completed)\b", re.I)
_WEIGH_PRAISE = re.compile(r"\b(logged|recorded|did|got|kept up)\b[^.]{0,20}\b(weigh-?in|weight)\b|\bweigh-?in\b[^.]{0,15}"
                           r"\b(done|logged|recorded)\b", re.I)
_WEIGH_NAILED = re.compile(r"\b(nail\w*|crush\w*|smash\w*|great|solid|good job|well done|perfect|excellent|on point)\b"
                           r"[^.]{0,25}\b(weigh-?in|weight log)\b", re.I)
_WEIGH_MISSED = re.compile(r"\b(missed|skipped|forgot)\b[^.]{0,20}\b(weigh-?in|weight)\b|\b(weigh-?in|weight)\b[^.]{0,20}"
                           r"\b(wasn'?t|was not|not) (logged|recorded)\b", re.I)
_TODAY_FOOD = re.compile(r"\b(breakfast|eat\w*|food|meal|snack|yogh?urt|eggs?|oat\w*|fruit|toast|shake|chicken|protein|tofu|"
                         r"porridge|smoothie|cottage|salad|tuna|salmon|beef|rice|pasta|banana|apple|berries|omelet\w*|"
                         r"peanut|bowl)\b", re.I)
_TODAY_MOVE = re.compile(r"\b(walk\w*|steps?|move\w*|movement|stairs|stretch\w*|treadmill|cycl\w*|jog\w*|mobility|"
                         r"active|stroll|bike)\b", re.I)
_TODAY_WORKOUT = re.compile(r"\b(workout|session|training|train|lift\w*|gym|rest(?! of)|recover\w*|cardio|legs?|upper|lower|"
                            r"push|pull|exercise|strength)\b", re.I)
_REST_CLAIM = re.compile(r"\b(rest day|day off|recovery day|off day|rest today|rest and recover today)\b", re.I)
_TRAIN_PUSH = re.compile(r"\b(hit the gym|do (?:a|your) (?:workout|session)|train(?:ing)? (?:hard )?today|"
                         r"your (?:workout|session) today|get a workout in|lift(?:ing)? today|squeeze in a workout)\b", re.I)
_HEARTY = re.compile(r"\b(hearty|big|large|heavy|indulgent|generous|calorie-dense|energy-dense|double)\b[^.]{0,30}"
                     r"\b(breakfast|meal|plate|portion|bowl|stack)\b", re.I)
_TARGET_WORD = r"(?:calorie target|calorie budget|calorie goal|calorie allowance|budget|allowance|target|room)"
_SAYS_HIGHER = re.compile(rf"\b(higher|bigger|larger|raised|increased|extra|more)\b[^.]{{0,20}}\b{_TARGET_WORD}\b", re.I)
_SAYS_LOWER = re.compile(rf"\b(lower|smaller|reduced|tighter|trimmed|less|leaner)\b[^.]{{0,20}}\b{_TARGET_WORD}\b", re.I)


def _wrong_praise(y: str, t: dict) -> list[str]:
    out = []
    for c in clauses(y):
        named = [k for k, pat in (("calories", r"\bcalor\w*"), ("steps", r"\bsteps?\b"), ("protein", r"\bprotein\b"))
                 if re.search(pat, c, re.I)]
        if len(named) != 1:
            continue
        k = named[0]
        if _FORWARD_LOOKING.search(c):
            continue                            # "make sure to hit your protein goal today" is advice, not a verdict
        praise = re.search(rf"\b{_PRAISE}\b", c, re.I) and not re.search(r"\b(not|n't|never|almost|nearly)\b", c, re.I)
        short = re.search(rf"\b{_SHORT}\b", c, re.I)
        if k == "calories" and t["cal_off"] and praise and not short and not re.search(r"\b(over|under|above|below)\b", c, re.I):
            out.append(f"praises calories that were off: {c[:80]!r}")
        if k == "steps" and t["steps_below"] and praise and not short:
            out.append(f"praises steps that were below the goal: {c[:80]!r}")
        if k == "protein" and t["protein_short"] and praise and not short:
            out.append(f"praises protein that was short: {c[:80]!r}")
        if k == "protein" and t["protein_met"] and short and not praise:
            out.append(f"calls protein short although it was met: {c[:80]!r}")
        if k == "steps" and t["steps_met"] and short and not praise:
            out.append(f"calls steps short although the goal was met: {c[:80]!r}")
    return out


def score_briefing(texts: dict, case: dict) -> list[Item]:
    """`texts` = {'yesterday': ..., 'today': ...}."""
    t = briefing_truth(case)
    out: list[Item] = []
    add = lambda kind, iid, ok, detail="": out.append(Item(kind, iid, bool(ok), "" if ok else detail))

    y = _plain(texts.get("yesterday"))
    names = len(re.findall(rf"\b{NAME}\b", y))
    add("instruction", "by_greeting_and_name_once", _GREETING.search(y) and names == 1,
        f"greeting={bool(_GREETING.search(y))}, name used {names}x")
    ys = sentences(y)
    add("instruction", "by_length", 2 <= len(ys) <= 5, f"{len(ys)} sentences")
    nums = _numbers(_PROPOSAL.sub(" ", _TIMES.sub(" ", y)))
    add("instruction", "by_no_repeated_numbers", not nums, f"figures: {nums}")
    add("instruction", "by_no_lists_or_sign_off", not _LIST.search(y) and not _SIGNOFF.search(y))
    praise = _wrong_praise(y, t)
    add("hallucination", "by_no_wrong_praise", not praise, "; ".join(praise))
    wrong = []
    if t["workout_day"] and not t["workout_logged"] and _WORKOUT_PRAISE.search(y) and not _WORKOUT_MISSED.search(y):
        wrong.append("praises a workout that was not logged")
    if t["workout_day"] and t["workout_logged"] and _WORKOUT_MISSED.search(y):
        wrong.append("calls a logged workout missed")
    if not t["weight_logged"]:
        for sent in sentences(y):
            if (_WEIGH_PRAISE.search(sent) or _WEIGH_NAILED.search(sent)) and not _WEIGH_MISSED.search(sent) \
                    and not _FORWARD_LOOKING.search(sent) and not _NEG.search(sent[:sent.lower().find("weigh")]):
                wrong.append("praises a weigh-in that was missed")
                break
    if t["weight_logged"] and _WEIGH_MISSED.search(y):
        wrong.append("calls a recorded weigh-in missed")
    add("hallucination", "by_workout_weighin_right", not wrong, "; ".join(wrong))
    if t["big_misses"]:
        missing = [k for k in t["big_misses"] if not _METRIC[k].search(y)]
        add("semantic", "by_big_misses_mentioned", not missing, f"not mentioned: {missing}")

    d = _plain(texts.get("today"))
    add("instruction", "bt_no_greeting_or_name", not _GREETING.search(d) and not re.search(rf"\b{NAME}\b", d))
    ds = sentences(d)
    add("instruction", "bt_length", 2 <= len(ds) <= 4, f"{len(ds)} sentences")
    allowed = {float(round(t["temp"])), float(t["temp"])} if t["temp"] is not None else set()
    stray = [n for n in _numbers(_PROPOSAL.sub(" ", _TIMES.sub(" ", d))) if n not in allowed]
    add("instruction", "bt_no_repeated_numbers", not stray, f"figures: {stray}")
    if t["training_day"]:
        r = [m.group(0) for m in _REST_CLAIM.finditer(d) if not re.search(r"\b(not|n't|no|never)\b[^.]{0,15}$",
                                                                         d[max(0, m.start() - 30):m.start()], re.I)]
        add("hallucination", "bt_workout_day_right", not r, f"calls a training day {r[0]!r}" if r else "")
    add("instruction", "bt_no_lists_or_sign_off", not _LIST.search(d) and not _SIGNOFF.search(d))
    workout_note = _TODAY_WORKOUT.search(d) or (t["no_plan"] and re.search(r"plan", d, re.I))
    missing = [n for n, pat in (("food", _TODAY_FOOD), ("movement", _TODAY_MOVE)) if not pat.search(d)]
    missing += [] if workout_note else ["workout note"]
    add("semantic", "bt_covers_food_movement_workout", not missing, f"missing: {missing}")
    if not t["outdoor_ok"]:
        o = suggests_outdoors(d, conditional_ok=t["temp"] is None, alt_ok=t["weather_soft"])
        add("semantic", "bt_outdoor_rule", not o, o[:100])
    if t["lower_today"] or t["higher_today"]:
        cal_clauses = [c for c in clauses(d) if not re.search(r"\bsteps?\b", c, re.I)]
        if t["lower_today"]:
            bad = [c for c in cal_clauses if _SAYS_HIGHER.search(c) and not _SAYS_LOWER.search(c)]
        else:
            bad = [c for c in cal_clauses if _SAYS_LOWER.search(c) and not _SAYS_HIGHER.search(c)]
        add("hallucination", "bt_budget_direction_right", not bad, f"says the opposite: {bad[0][:80]!r}" if bad else "")
        if t["lower_today"]:
            h = _HEARTY.search(d)
            add("semantic", "bt_food_fits_budget", not (h and _asks(_HEARTY, d)), h.group(0) if h else "")
        else:
            add("semantic", "bt_food_fits_budget", not _asks(_EAT_LESS, d), "asks to eat less on a raised target")
    if t["rest_day"]:
        p = _TRAIN_PUSH.search(d)
        add("semantic", "bt_rest_day_respected", not p, p.group(0) if p else "")
    ex = [m.group(0) for m in _EXERCISE.finditer(d) if not _NEG.search(d[max(0, m.start() - 40):m.start()])]
    add("semantic", "bt_steps_are_walking", not ex, f"names an exercise: {ex[0]!r}" if ex else "")
    inv = [m.group(0) for m in _EXTRA_WORKOUT.finditer(d) if not _NEG.search(d[max(0, m.start() - 40):m.start()])]
    if t["rest_day"]:
        inv += [m.group(0) for m in _DO_WORKOUT.finditer(d) if not _NEG.search(d[max(0, m.start() - 40):m.start()])]
    add("semantic", "bt_no_invented_workout", not inv, f"suggests a workout: {inv[0]!r}" if inv else "")
    return out


def failed(items: list[Item]) -> list[Item]:
    return [i for i in items if not i.ok]
