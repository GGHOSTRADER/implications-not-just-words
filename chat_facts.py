"""The chat (tactical) agent's facts as words (2026-10-05, CHAT_MODE=words).

The numbers-and-tools design (CHAT_MODE=numbers, the original) hands the model raw figures through three tools and asks it
to do the arithmetic. Measured on one real evening (evals/chat_*): "how am I doing today" said 983 kcal remaining when the
day was 10 kcal over, "can I have a pizza" computed "1,950 - 263 = 1,687 remaining", a canned dump answered "is 500 kcal too
much". Same principle as the weekly review and the checkpoint comment (evals/REPORT.md): code does the judging and the
arithmetic, the model reads words and writes the answer.

  * calories, protein, steps: where the day stands as words, against the plan for THIS hour, and what it means in the phase;
  * time: how far into the day, which meals and walks still make sense;
  * a size in the question ("is 500 kcal too much") is judged by code against what is left;
  * exact figures appear only when the question asks for one, as complete sentences to quote;
  * the answer is checked: figures not in the facts, exercises, invented workouts, shouting, suggestions that do not fit
    the hour, claims of having logged.

Pure functions: no database, no model. agent.py gathers the inputs and calls the model.
"""
from __future__ import annotations

import re

import checkpoint_labels as cl
import checkpoint_labels_v3 as cl3
import label_context_v3 as ctx3

DAY_START_HOUR = 6          # the plan's cumulative curve starts at zero here
LAST_HOUR = 23              # from here on the day is over for planning purposes (the 23:59 check)

# ── the plan for this hour ───────────────────────────────────────────────────


def schedule_fractions(checkpoints: list, hour: int, minute: int = 0) -> dict:
    """Cumulative share of each metric the plan expects by now, from the checkpoint schedule
    [(label, hour, steps_frac, cal_frac, protein_frac)]: linear between checkpoints, zero at DAY_START_HOUR, one after the
    last. Without a schedule: None per metric (no pace judgment)."""
    if not checkpoints:
        return {"steps": None, "calories": None, "protein": None}
    t = hour + minute / 60
    first = min((float(r[1]) + (int(r[0].split(":")[1]) / 60 if isinstance(r[0], str) and ":" in r[0] else 0)) for r in checkpoints)
    if t < first:                      # before the day's first check no pace is judged (breakfast is not "far over plan")
        return {"steps": None, "calories": None, "protein": None, "early": True}
    pts = [(float(DAY_START_HOUR), 0.0, 0.0, 0.0)]
    for row in checkpoints:
        label, h = row[0], row[1]
        hh = float(h)
        if isinstance(label, str) and ":" in label:
            try:
                hh = int(label.split(":")[0]) + int(label.split(":")[1]) / 60
            except ValueError:
                pass
        sf, cf, pf = (row[2], row[3], row[4]) if len(row) >= 5 else (row[2], row[2], row[2])
        pts.append((hh, float(sf), float(cf), float(pf)))
    pts.sort()
    if t >= pts[-1][0]:
        return {"steps": 1.0, "calories": 1.0, "protein": 1.0}
    if t <= pts[0][0]:
        return {"steps": 0.0, "calories": 0.0, "protein": 0.0}
    for (h0, s0, c0, p0), (h1, s1, c1, p1) in zip(pts, pts[1:]):
        if h0 <= t <= h1:
            k = (t - h0) / (h1 - h0) if h1 > h0 else 1.0
            return {"steps": s0 + k * (s1 - s0), "calories": c0 + k * (c1 - c0), "protein": p0 + k * (p1 - p0)}
    return {"steps": 1.0, "calories": 1.0, "protein": 1.0}


_SHARES = [(0.05, "only a small part"), (0.15, "about a tenth"), (0.22, "about a fifth"), (0.3, "about a quarter"),
           (0.4, "about a third"), (0.6, "about half"), (0.85, "most"), (9.0, "nearly all")]


def share_words(f: float) -> str:
    f = max(0.0, f)
    return next(w for limit, w in _SHARES if f < limit)


def _over_size(frac_of_target: float) -> str:
    f = abs(frac_of_target)
    return "slightly" if f < 0.06 else "a little" if f < 0.12 else "well" if f < 0.25 else "far"


# ── the pieces ───────────────────────────────────────────────────────────────


def day_left_words(hour: int, minute: int) -> str:
    """How much of the waking day (06:00 to 23:00) is left, in words."""
    left = 23 - (hour + minute / 60)
    if hour >= LAST_HOUR:
        return ""
    if left > 15:
        return "Day left: the day has hardly started."
    if left > 11:
        return "Day left: most of the day is still ahead."
    if left > 7:
        return "Day left: more than half of the day is left."
    if left > 3.5:
        return "Day left: less than half of the day is left."
    if left > 1:
        return "Day left: only a little of the day is left."
    return "Day left: the day is nearly over."


def phase_line(phase: str) -> str:
    """What the phase's goal means for today's food, so no advice works against it."""
    phase = cl._phase(phase)
    if phase == cl.FAT_LOSS:
        return ("Phase: Fat Loss. The goal is a calorie deficit: staying at or under the calorie target helps and going over "
                "works against it. Being under the target is fine and is never a reason to eat more calories. Protein matters "
                "to protect muscle.")
    if phase == cl.MUSCLE_GAIN:
        return ("Phase: Muscle Gain. The goal is a calorie surplus: staying under the calorie target works against it, so the "
                "rest of the day needs enough food; a little over is fine. Never tell the user to eat less when the day is "
                "under its target. Protein drives growth.")
    return ("Phase: Maintenance. The goal is to stay level: ending well over or well under the calorie target both matter.")


def time_words(hour: int, minute: int) -> str:
    return ctx3.time_line(hour, minute, cl.hour_words(hour, minute), is_last=hour >= LAST_HOUR)


def calorie_words(inp: dict, sched: float | None) -> str:
    target, eaten = float(inp["cal_target"]), float(inp["cal_consumed"])
    phase = cl._phase(inp.get("phase", ""))
    if not inp.get("entries") and eaten <= 0:
        return "Calories: no food is logged yet today."
    room = target - eaten
    if room < -0.03 * target:
        head = (f"Calories: the day's calorie target is used up and the day is {_over_size(room / target)} over it; "
                "there is no room for more food today")
    elif abs(room) <= 0.03 * target:
        head = "Calories: the day is right at its calorie target; there is no room left for more food"
    else:
        head = f"Calories: {share_words(room / target)} of the day's calorie budget is still free"
    pace = ""
    if sched and sched > 0.05:
        dev = eaten / (target * sched) - 1
        pace = ", " + cl._day_words("calories", dev)
        mean = (cl._meaning_past if sched >= 1.0 else cl._meaning)("calories", phase, dev)
        if mean:
            pace += f" ({mean})"
    return head + pace + "."


def protein_words(inp: dict, sched: float | None) -> str:
    target, got = float(inp["protein_target"]), float(inp["protein_consumed"])
    phase = cl._phase(inp.get("phase", ""))
    if target <= 0:
        return ""
    q = got / target
    if q >= 1.15:
        head = "Protein: the goal is reached and then some"
    elif q >= 0.97:
        head = "Protein: the goal is reached"
    else:
        head = f"Protein: {share_words(1 - q)} of the day's goal is still missing"
    pace = ""
    if q < 0.97 and sched and sched > 0.05:
        dev = got / (target * sched) - 1
        pace = ", " + cl._day_words("protein", dev)
        mean = cl._meaning("protein", phase, dev)
        if mean and abs(dev) >= cl._ON_PLAN:
            pace += f" ({mean})"
    return head + pace + "."


def step_words(inp: dict, sched: float | None) -> str:
    goal, steps = float(inp["step_goal"]), float(inp["steps"])
    if goal <= 0:
        return ""
    s = steps / goal
    if s >= 1.15:
        head = "Steps: the goal is reached, and well past it"
    elif s >= 1.0:
        head = "Steps: the goal is reached"
    else:
        head = f"Steps: {share_words(1 - s)} of the day's goal is still missing"
    pace = ""
    if s < 1.0 and sched and sched > 0.05:
        dev = steps / (goal * sched) - 1
        pace = ", " + cl._day_words("steps", dev)
    age = inp.get("steps_age_min")
    stale = ""
    if age is not None and age > 90:
        stale = (" The step count is several hours old, so it may be behind the real number." if age >= 180
                 else " The step count is over an hour old, so it may be a little behind.")
    return head + pace + "." + stale


def overall_words(inp: dict, sched: dict) -> str:
    """One line with the day's verdict, so the answer opens with the truth and does not cheer a bad day (found by reading
    the first evaluation: "you're doing great! you're well behind on steps and protein")."""
    last = inp["now"].hour >= LAST_HOUR
    target = float(inp["cal_target"])
    eaten = float(inp["cal_consumed"])
    issues = []
    if eaten - target > 0.06 * target:
        issues.append("over on calories")
    elif sched.get("calories") and sched["calories"] > 0.2 and not last and eaten / (target * sched["calories"]) - 1 < -0.2:
        issues.append("well under on food so far")
    pt, pg = float(inp["protein_target"]), float(inp["protein_consumed"])
    if pt > 0 and pg < 0.9 * pt and (last or (sched.get("protein") and pg / (pt * sched["protein"]) - 1 < -0.2)):
        issues.append("short on protein" if last else "behind on protein")
    sg, ss = float(inp["step_goal"]), float(inp["steps"])
    if sg > 0 and ss < sg and (last or (sched.get("steps") and ss / (sg * sched["steps"]) - 1 < -0.2)):
        issues.append("short on steps" if last else "behind on steps")
    if not issues and sched.get("early"):
        return "Overall: the day has only just started, so nothing is behind yet."
    if not issues and last:
        return "Overall: the day ended on plan."
    if not issues:
        return "Overall: the day is on plan."
    if last:
        return ("Overall: the day ended " + (issues[0] if len(issues) == 1 else ", ".join(issues[:-1]) + " and " + issues[-1]) +
                "; say how it went in the past tense and do not call it great.")
    return ("Overall: " + (issues[0] if len(issues) == 1 else ", ".join(issues[:-1]) + " and " + issues[-1]) +
            "; open with that and do not call the day great.")


def log_words(inp: dict) -> str:
    """Whether the food log can be trusted for the hour: a long gap means meals may be missing."""
    last = inp.get("hours_since_last_entry")
    if last is None or not inp.get("entries"):
        return ""
    if last >= 4 and inp["now"].hour < LAST_HOUR:
        return "The food log has had nothing added for several hours, so a meal may be missing from it."
    return ""


# ── what the question asks ───────────────────────────────────────────────────

_KCAL_IN_Q = re.compile(r"\b(\d{2,4})\s*(?:kcal|cal|cals|calories)\b", re.I)
_FIGURE_Q = re.compile(r"\b(how (?:many|much|far)|exact|exactly|number|total|left|remain\w*|still need|need to|what(?:'s| is) my|"
                       r"show me|status|target|where am i on|where do i stand|how are my (?:steps|calories|protein))\b", re.I)
_LOG_Q = re.compile(r"\b(food log|what did i (?:eat|have)|what have i (?:eaten|had)|show (?:me )?(?:my )?(?:food|log|meals)|"
                    r"list (?:my )?(?:food|meals)|my meals|eaten today)\b", re.I)
_CAN_I = re.compile(r"\b(can|could|may|should|is it ok(?:ay)? to)\b[^.?!]{0,12}\b(i|we)\b[^.?!]{0,12}\b(have|eat|get|order|"
                    r"drink|snack|grab|add)\b", re.I)
_STEP_Q = re.compile(r"\b(steps?|walk\w*|move|movement|active|treadmill|stairs|pace)\b", re.I)
_CAL_Q = re.compile(r"\b(calor\w*|kcal|food|eat|meal|snack|budget|hungry|dinner|lunch|breakfast|macros?)\b", re.I)
_PROT_Q = re.compile(r"\b(protein|macros?)\b", re.I)
# "how is the day going" in its many forms (not "how's my protein" and not "where am I on steps")
_STATUS_Q = re.compile(r"\b(how am i doing|am i on track|how(?:'s| is) (?:my day|today|the day|it going)|how did i do|"
                       r"where am i(?! on\b))", re.I)
_DAYLEFT_Q = re.compile(r"\b(how much (?:of )?(?:the )?(?:day|time)|how long (?:is )?(?:left|until)|day (?:is )?left|hours? left|"
                        r"time left|how late)\b", re.I)
_OLD_WORDS = re.compile(r"\b(old|stale|hours|sync|delay|behind the real|may be|might not|out of date|not up to date)\b", re.I)


def wants_figures(question: str) -> bool:
    return bool(_FIGURE_Q.search(question or ""))


def wants_log(question: str) -> bool:
    return bool(_LOG_Q.search(question or ""))


def meal_fit(question: str, inp: dict) -> str:
    """A size in the question ('is 500 kcal too much for dinner') judged against what is left today, in words."""
    m = _KCAL_IN_Q.search(question or "")
    if not m:
        return ""
    x, target = float(m.group(1)), float(inp["cal_target"])
    room = target - float(inp["cal_consumed"])
    if room <= 0.03 * target:
        return ("The meal size in the question does not fit: the day is already at or over its calorie target, so it would "
                "push the day further over.")
    if x <= 0.6 * room:
        return "The meal size in the question fits comfortably in what is left of the day's calories."
    if x <= room:
        return "The meal size in the question fits, but it uses most of what is left of the day's calories."
    return (f"The meal size in the question is more than what is left of the day's calories: it would take the day "
            f"{_over_size((x - room) / target)} over its target.")


def question_notes(question: str, inp: dict) -> list[str]:
    notes = []
    fit = meal_fit(question, inp)
    if fit:
        notes.append(fit)
    elif _CAN_I.search(question or "") and not _KCAL_IN_Q.search(question or ""):
        notes.append("The question names a food but gives no calorie figure. Never invent one: answer from how much room "
                     "is left today (in words) and the hour, and say the user can tell you the amount if they want it judged.")
    return notes


def figure_lines(question: str, inp: dict) -> list[str]:
    """Complete sentences with the exact figures, only for the metrics the question is about."""
    q = question or ""
    want_steps, want_cal, want_prot = bool(_STEP_Q.search(q)), bool(_CAL_Q.search(q)), bool(_PROT_Q.search(q))
    if not (want_steps or want_cal or want_prot):
        want_steps = want_cal = want_prot = True
    out = []
    if want_cal:
        target, eaten = float(inp["cal_target"]), float(inp["cal_consumed"])
        room = round(target - eaten)
        out.append(f"Calories: {eaten:,.0f} eaten of a {target:,.0f} target, " +
                   (f"{room:,} left." if room > 0 else "none left." if room == 0 else f"{-room:,} over."))
    if want_prot:
        t, g = float(inp["protein_target"]), float(inp["protein_consumed"])
        left = round(t - g)
        out.append(f"Protein: {g:,.0f} g of a {t:,.0f} g goal, " + (f"{left:,} g to go." if left > 0 else "goal reached."))
    if want_steps:
        goal, steps = float(inp["step_goal"]), float(inp["steps"])
        left = round(goal - steps)
        age = inp.get("steps_age_min")
        old = (" This count is several hours old." if age is not None and age >= 180 else
               " This count is over an hour old." if age is not None and age > 90 else "")
        out.append(f"Steps: {steps:,.0f} of a {goal:,.0f} goal, " + (f"{left:,} to go." if left > 0 else "goal reached.") + old)
    return out


def food_log_lines(inp: dict) -> list[str]:
    entries = inp.get("entries") or []
    if not entries:
        return ["Food logged today: nothing yet."]
    return ["Food logged today (oldest first): " + "; ".join(
        f"{e.get('food', '?')} ({round(float(e.get('calories', 0)))} kcal at {e.get('time', '?')})" for e in entries) + "."]


# ── the whole block ──────────────────────────────────────────────────────────


def view(inp: dict, question: str, sched: dict | None = None) -> str:
    """The facts block for one question. `inp` holds now, phase, step_goal, steps, steps_age_min, cal_target,
    cal_consumed, protein_target, protein_consumed, entries, hours_since_last_entry, weather, foods (optional list of
    {food, calories, protein} that fit what is left)."""
    now = inp["now"]
    sched = sched or {}
    lines = [time_words(now.hour, now.minute),
             day_left_words(now.hour, now.minute),
             phase_line(inp.get("phase", "")),
             overall_words(inp, sched),
             calorie_words(inp, sched.get("calories")),
             protein_words(inp, sched.get("protein")),
             step_words(inp, sched.get("steps"))]
    note = log_words(inp)
    if note:
        lines.append(note)
    q = question or ""
    age = inp.get("steps_age_min")
    if age is not None and age > 90 and (_STEP_Q.search(q) or _STATUS_Q.search(q)):
        lines.append("The step count is old (see Steps): whenever you mention steps, say the count is old and that the user "
                     "may be further along than it shows.")
    if _DAYLEFT_Q.search(q):
        lines.append("The user asked how much of the day is left: answer from the Time line (the hours that remain) and the "
                     "Day left line, in a sentence.")
    if _STATUS_Q.search(q):
        lines.append("The user asked how the day is going: start with the Overall verdict, then say briefly where calories, "
                     "protein and steps each stand.")
    behind_steps = float(inp["steps"]) < float(inp["step_goal"])
    if _STEP_Q.search(q) and now.hour < LAST_HOUR:
        lines.append(ctx3.steps_line(inp.get("weather")) if behind_steps else
                     "The step goal is reached: there is nothing to add; do not push for more steps.")
    elif _STEP_Q.search(q):
        lines.append("The day is over for planning: do not suggest more steps.")
    lines.extend(question_notes(q, inp))
    foods = inp.get("foods") or []
    room = float(inp["cal_target"]) - float(inp["cal_consumed"])
    if foods and room > 0.1 * float(inp["cal_target"]) and now.hour < 22 and re.search(r"\b(eat|food|hungry|snack|meal|dinner|lunch|breakfast|have)\b", q, re.I):
        lines.append("Foods from the user's own list that fit what is left: " +
                     ", ".join(f"{f['food']} ({round(f['calories'])} kcal, {round(f['protein'])} g protein)" for f in foods[:3]) + ".")
    if wants_figures(q):
        lines.append("Exact figures (quote only these, and only because the user asked): " + " ".join(figure_lines(q, inp)))
    if wants_log(q):
        lines.extend(food_log_lines(inp))
    return "\n".join(line for line in lines if line)


WORDS_PROMPT = """You are the chat agent of a fitness assistant. Below is a FACTS block written by code from the user's own data for this moment. Answer the user's question from it.

How to answer:
- Answer the question that was asked, first. Two to four sentences, plain text, no markdown, no headings, no lists unless the user asked for their food log.
- Judge with the facts block's own words ("a little over", "well behind", "the goal is reached") and never contradict them. Do not calculate anything yourself.
- Write numbers only if they appear in the facts block or in the user's message. If the user asks for a figure and the block has no "Exact figures" line, say you do not have it to hand and that Home shows it.
- If what the user says disagrees with the facts (for example they say they are behind on steps and the facts say the goal is reached), say what the facts show.
- Respect the Phase line: never advise against the phase's goal (for example, more calories in Fat Loss, or eating less in Muscle Gain when the day is under its target). When asked how much of the day is left, answer from the Time and Day left lines.
- If the user names a food without a calorie figure, do not invent one.
- If the facts say there is no room left for food today, do not say it is fine to eat anyway: say the day is full and that it can wait for tomorrow.
- A greeting or small talk gets one short friendly sentence. Do not report the day unless the user asks.
- Never mention "the facts", "the block" or "the data" to the user: just say it.
- If the facts say the step count is old, say so when you give it.
- Workouts come only from the user's own plan: for a workout request, say your plan is where workouts live (/workout, /blueprint) and that you do not make up workouts.
- Suggest only what still fits the hour in the Time line: no breakfast ideas in the evening, no walks late at night, nothing about "later today" once the day is over.
- Calm tone. No shouting, no "LET'S GO", no pep talk or sign-off at the end: stop after the useful point.
- You cannot log, add, change or delete anything. If the user wants that, say where to go in one short sentence: Log Food / Log Workout / Log Weight in the Home mini app, or /food, /workout.
- """ + ctx3.NO_EXERCISES + """ Steps are walking steps only.
"""


# ── the answer checks ────────────────────────────────────────────────────────

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SHOUT = re.compile(r"\b(?:LET'?S GO|LETS GO|YOU GOT THIS|YOU'?VE GOT THIS|CRUSH IT|SMASH(?:ED)? IT)\b|[A-Z]{5,}(?:\s+[A-Z]{2,})+")
_CLAIM = re.compile(r"\b(?:i(?:'ve| have)? (?:added|recorded|saved|logged|updated|deleted)|logged (?:it|that|this|your)|"
                    r"added (?:it |that |this )?to your|saved (?:it )?to|i(?:'ll| will) (?:log|add|record|save))\b", re.I)
_FACTS_SAY = re.compile(r"^\s*(?:(?:the |my )?facts|the data|the block)\s+(?:say|says|show|shows|indicate|indicates)\s+(?:that\s+)?", re.I)
_META = re.compile(r"\b(?:the facts|facts say|facts block|the block|the data says|according to the data)\b", re.I)
_EAT_ANYWAY = re.compile(r"\b(?:you (?:can|could) (?:still )?(?:have|eat|enjoy)|go ahead and (?:have|eat|enjoy)|treat yourself|"
                         r"it'?s (?:fine|okay|ok) to (?:have|eat))\b", re.I)


def tidy(text: str) -> str:
    """Remove "The facts say (that)" from the start of an answer (the model sometimes narrates its source)."""
    out = _FACTS_SAY.sub("", text or "", count=1).lstrip(" ,:-")
    return out[:1].upper() + out[1:] if out else out


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text or "")}


def problems(text: str, inp: dict, question: str, facts: str) -> list[str]:
    """What is wrong with an answer: a figure that is not in the facts or the question, exercises, an invented workout,
    shouting, a claim of having logged, a suggestion that does not fit the hour, an outdoor-only step idea in bad weather."""
    text = (text or "").replace("’", "'").replace("‘", "'")
    out = []
    allowed = _numbers(facts) | _numbers(question)
    bad = sorted(n for n in _numbers(text) if n not in allowed and len(n.replace(".", "")) >= 2)
    if bad:
        out.append(f"it writes {', '.join(bad[:3])}, which is not in the facts or in the user's message; write only figures "
                   "from those, or use the facts' words")
    for m in cl3._EXERCISE.finditer(text):
        if not cl3._negated(text, m.start()):
            out.append(f"it names an exercise ({m.group(0)}); steps are walking steps, so suggest only walking, the stairs, "
                       "pacing or a treadmill, and never exercises")
            break
    for m in cl3._EXTRA_WORKOUT.finditer(text):
        if not cl3._negated(text, m.start()):
            out.append("it suggests a workout; training comes only from the user's plan, so do not add or invent one")
            break
    fit = meal_fit(question, inp)
    if fit.startswith("The meal size in the question fits"):
        if re.search(r"\b(?:too much|too big|too heavy|too large|a bit much|bit much|quite a lot|a lot for|exceed\w*|doesn'?t fit|"
                     r"does not fit|a bit over|go over|goes over|put(?:s)? you over|"
                     r"push(?:es)? (?:you|the day|it) (?:further |even )?over)\b", text, re.I):
            out.append("it says the meal is too big, but the facts say it fits; say that it fits")
    elif fit:
        if re.search(r"\b(?:fits comfortably|should fit|will fit|you'?re good to have|that works|go ahead|no problem)\b", text, re.I):
            out.append("it says the meal fits, but the facts say it does not; say that it does not fit")
    age = inp.get("steps_age_min")
    if age is not None and age > 90 and (_STEP_Q.search(question or "") or _STATUS_Q.search(question or "")) \
            and re.search(r"\b(steps?|walk\w*)\b", text, re.I) and not _OLD_WORDS.search(text):
        out.append("it talks about steps without saying the step count is old; say it is old and the user may be further along")
    if _STATUS_Q.search(question or ""):
        for what, pat in (("calories", r"calor|kcal|food|eat|budget|meal"), ("protein", r"protein"), ("steps", r"step|walk|move")):
            if not re.search(pat, text, re.I):
                out.append(f"it leaves out {what}; a how-is-the-day-going answer says where calories, protein and steps each stand")
    phase = cl._phase(inp.get("phase", ""))
    tgt, eaten_now = float(inp["cal_target"]), float(inp["cal_consumed"])
    if phase == cl.FAT_LOSS and any(not cl3._negated(text, m.start()) for m in re.finditer(
            r"\b(?:eat|have|add|take in|consume)\s+(?:a lot |much |some |a bit |significantly )?more (?:food|calories|carbs)\b|"
            r"\bincrease (?:your )?(?:calorie|food) intake\b|\bneed(?:s)? to eat more\b|\bshould eat more\b|\bcalorie surplus\b", text, re.I)):
        out.append("it tells the user to eat more calories, but the phase is Fat Loss: being under the target is fine; say so")
    if phase == cl.MUSCLE_GAIN and tgt - eaten_now > 0.1 * tgt and any(not cl3._negated(text, m.start()) for m in re.finditer(
            r"\b(?:eat less|cut (?:back|calories)|reduce (?:your )?(?:calories|intake)|lighter (?:meals?|portions?|dinner|lunch)|"
            r"skip (?:dinner|a meal|your snack)|calorie deficit|too many calories)\b", text, re.I)):
        out.append("it tells the user to eat less, but the phase is Muscle Gain and the day is under its target: more food is needed")
    if _META.search(text):
        out.append("it mentions the facts or the data to the user; state things directly instead")
    room = float(inp["cal_target"]) - float(inp["cal_consumed"])
    if room <= 0.03 * float(inp["cal_target"]) and inp.get("entries"):
        for m in _EAT_ANYWAY.finditer(text):
            if not cl3._negated(text, m.start()):
                out.append("it says the user can eat although the day has no room left; say the day is full and it can wait")
                break
    if _SHOUT.search(text):
        out.append("it shouts or cheers (capital letters, 'let's go'); use a calm tone")
    if _CLAIM.search(text):
        out.append("it says it logged or will log something; it cannot log, and should say where to go instead")
    now = inp["now"]
    hour = now.hour + (1 if now.minute >= 30 else 0)
    if now.hour >= LAST_HOUR:
        for m in cl3._LAST_FRAMING.finditer(text):
            if m.group(0).lower() != "tomorrow" and not cl3._negated(text, m.start()):     # "it can wait for tomorrow" is right
                out.append(f"it talks as if the day went on ({m.group(0)!r}); the day is over, so say only how it went")
                break
    for word, over_at in cl3._MEAL_WORDS.items():
        for m in re.finditer(rf"\b{word}s?\b", text, re.I):
            if hour >= over_at and not cl3._negated(text, m.start()) \
                    and not cl3._TOMORROW.search(text[max(0, m.start() - 40):m.end() + 40]) \
                    and re.search(r"\b(try|have|eat|grab|choose|opt|add|go for|plan|start|enjoy|make|prepare)\b",
                                  text[max(0, m.start() - 50):m.end() + 20], re.I):
                out.append(f"it suggests {word}, but at this hour that is over; suggest only what still fits")
                break
    if hour >= 22:
        for s in re.split(r"(?<=[.!?])\s+", text):
            if cl3._MOVE_WORD.search(s) and re.search(r"\b(try|take|go|get|add|do|fit|squeeze|aim|consider|maybe|could)\b", s, re.I) \
                    and not cl3._negated(s, len(s)) and not cl3._TOMORROW.search(s):
                out.append("it suggests walking, but it is too late for movement ideas")
                break
    verdict, reason = ctx3.weather_verdict(inp.get("weather"))
    if verdict != "fine" and cl.suggests_outside(text, allow_conditional=verdict == "unknown"):
        out.append(f"it sends the user outside ({reason}); suggest steps indoors: a treadmill, the stairs, pacing or a covered walk")
    return out
