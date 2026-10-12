"""Checklist for the chat agent's answers (version 1.0, 2026-10-05). Self-contained: it imports nothing from chat_facts
(the design under test), so it stays an independent judge. Every item is True (passed), False (failed) or None (does not
apply to this case). A case "passes" when no applicable item fails (every-item, as in the other message evaluations).

`truth` is what the fake day really is (evals/run_chat_eval.py computes it with the production functions):
hour, minute, steps, step_goal, steps_hit, steps_left, cal_target, cal_consumed, cal_room, protein_target,
protein_consumed, protein_left, entries (names), steps_age, allowed (every figure the day supports).
"""
from __future__ import annotations

import re

VERSION = "1.6"        # 1.1: figures the production system prints from the day allowed. 1.2 (flags read, 2026-10-05): clock
                       # times are not figures, "not yet logged" is not a claim of logging, a negation after "outdoor" counts,
                       # more ways to say no. 1.3 (flags read, round 2): "catch up tomorrow" is not talk of a day that goes on,
                       # more ways to say yes; new items no_meta (does not narrate "the facts") and no_cheer_on_bad_day.
                       # 1.4 (precision check of the original design's "invented figure" flags, AFTER the held-out run, 2026-10-05):
                       # figures the system itself prints were missing from the allowed set: the step-discount note, a negative
                       # protein remainder, the inventory foods' calories and protein. This only removes false flags from the
                       # original design; the new design never used those figures.
                       # 1.5 (before the fresh set was run): an old step count must be named in every answer that talks about
                       # steps (steps_left, steps_help, walk_now, and a status answer that mentions steps)
                       # 1.6 (before the third set was run): phase_sense (advice must not work against the phase's goal) and kind
                       # day_left (the answer states how much of the day is left, within an hour and a half)

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SHOUT = re.compile(r"\b(?:LET'?S GO|LETS GO|YOU GOT THIS|YOU'?VE GOT THIS|CRUSH IT)\b|[A-Z]{5,}(?:\s+[A-Z]{2,})+")
_CLAIM = re.compile(r"\b(?:i(?:'ve| have)? (?:added|recorded|saved|logged|updated|deleted)|logged (?:it|that|this|your)|"
                    r"added (?:it |that |this )?to your|saved (?:it )?to|i(?:'ll| will) (?:log|add|record|save))\b", re.I)
_EXERCISE = re.compile(r"\b(squats?|lunges?|push-?ups?|pull-?ups?|sit-?ups?|planks?|burpees?|jumping jacks?|crunch(?:es)?|"
                       r"calisthenics|hiit|circuit|yoga|pilates|kettlebells?|deadlifts?|resistance bands?|home workout|"
                       r"workout at home|sets? of|reps?)\b", re.I)
_EXTRA_WORKOUT = re.compile(r"\b(?:extra|another|additional|new|different|quick|short)\s+(?:workout|training session|"
                            r"gym session|strength session|cardio session)\b|\b(?:do|try|start|squeeze in|fit in)\b[^.!?]{0,20}"
                            r"\b(?:a|an|another)\s+(?:\w+\s+)?(?:workout|training session|gym session|hiit|cardio session)\b", re.I)
_NEG = re.compile(r"\b(?:no|not|never|without|avoid|skip|don'?t|can'?t|cannot|instead of|rather than|nothing|isn'?t|aren'?t)\b"
                  r"[^.!?]{0,25}$", re.I)
_MEAL_OVER = {"breakfast": 11, "lunch": 15, "dinner": 22}
_EAT_VERB = re.compile(r"\b(try|have|eat|grab|choose|opt|add|go for|plan|start|enjoy|make|prepare|snack on)\b", re.I)
_MOVE = re.compile(r"\b(walk\w*|treadmill|stairs|pacing|stroll\w*)\b", re.I)
_MOVE_VERB = re.compile(r"\b(try|take|go|get|add|do|fit|squeeze|aim|consider|head|hop|start)\b", re.I)
_LATE_FRAME = re.compile(r"\b(on track|still time|plenty of time|keep going|catch up|rest of the day|later today|next meal|"
                         r"tomorrow|from here)\b", re.I)
_OUTSIDE = re.compile(r"\b(go outside|head outside|get outside|outdoors?|outside|in the park|around the block)\b", re.I)
_INDOOR = re.compile(r"\b(treadmill|stairs|pacing|indoors?|inside|mall|covered|at home|in the house|around the house)\b", re.I)


def _neg(text: str, pos: int) -> bool:
    return bool(_NEG.search(text[max(0, pos - 40):pos]))


_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", re.I)


def _numbers(text: str) -> set[str]:
    text = _CLOCK.sub(" ", text or "")           # a clock time is not a figure
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def _sent_of(text: str, pos: int) -> str:
    """The sentence that contains position `pos`."""
    start = max(text.rfind(".", 0, pos), text.rfind("!", 0, pos), text.rfind("?", 0, pos)) + 1
    ends = [i for i in (text.find(".", pos), text.find("!", pos), text.find("?", pos)) if i != -1]
    return text[start:min(ends) + 1 if ends else len(text)]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]


def allowed_figures(truth: dict, question: str) -> set[str]:
    out = _numbers(question)
    vals = [truth["steps"], truth["step_goal"], truth["steps_left"], truth["cal_target"], truth["cal_consumed"],
            abs(truth["cal_room"]), truth["protein_target"], truth["protein_consumed"], truth["protein_left"],
            truth["hour"], truth["minute"], truth["hour"] % 12 or 12, truth.get("steps_age") or 0,
            truth.get("hours_left", 0)]
    for e in truth.get("entry_figures", []):
        vals.extend(e)
    # figures the production system itself prints from the day (the original design shows them, so they are not invented):
    # the base calories, the step bonus, the steps over the goal, the plan's checkpoint targets
    vals.extend(truth.get("system_figures", []))
    for v in vals:
        v = abs(float(v))
        out.update({str(int(round(v))), str(int(v)), (f"{v:.1f}").rstrip("0").rstrip(".")})
    return out


def score(answer: str, case: dict, truth: dict) -> dict:
    """{item: True|False|None} for one answer."""
    a = (answer or "").replace("’", "'").replace("‘", "'")
    kind, q = case["kind"], case["question"]
    hour = truth["hour"] + (1 if truth["minute"] >= 30 else 0)
    low = a.lower()
    room, target = truth["cal_room"], truth["cal_target"]
    full = room <= 0.03 * target                      # no room left for food
    roomy = room > 0.25 * target                      # plenty of room
    r: dict = {}

    # ── every answer ─────────────────────────────────────────────────────────
    bad = [n for n in _numbers(a) if n not in allowed_figures(truth, q) and len(n.replace(".", "")) >= 2]
    r["no_invented_figure"] = not bad
    r["calm"] = not _SHOUT.search(a)
    r["no_claim_logged"] = not _CLAIM.search(a)
    words = len(a.split())
    r["no_meta"] = not re.search(r"\b(?:the facts|facts say|facts block|the data says|according to the data)\b", a, re.I)
    r["no_cheer_on_bad_day"] = (not re.search(r"\b(?:doing (?:great|amazing|fantastic|awesome)|great (?:job|day|work)|well done|"
                                              r"crushing it|nailing it)\b", a, re.I)) if truth.get("bad_day") else None
    # the advice must not work against the phase's goal (Maintenance: no rule)
    phase = truth.get("phase", "")
    if phase == "Fat Loss":
        r["phase_sense"] = not any(not _neg(a, m.start()) for m in re.finditer(
            r"\b(?:eat|have|add|take in|consume)\s+(?:a lot |much |some |a bit |significantly )?more (?:food|calories|carbs)\b|"
            r"\bincrease (?:your )?(?:calorie|food) intake\b|\bneed(?:s)? to eat more\b|\bshould eat more\b|"
            r"\bhit your calorie (?:target|goal)\b|\bcalorie surplus\b|\bbulk\b", a, re.I))
    elif phase == "Muscle Gain":
        under = room > 0.1 * target
        r["phase_sense"] = (not any(not _neg(a, m.start()) for m in re.finditer(
            r"\b(?:eat less|cut (?:back|calories)|reduce (?:your )?(?:calories|intake)|lighter (?:meals?|portions?|dinner|lunch)|"
            r"skip (?:dinner|a meal|your snack)|calorie deficit|too many calories)\b", a, re.I))) if under else None
    else:
        r["phase_sense"] = None
    r["short"] = None if kind == "log_list" else (len(_sentences(a)) <= 6 and words <= 110)
    r["no_dump"] = None if kind == "log_list" else not (re.search(r"calorie status:|food log:|^\s*\d+\.\s", a, re.I | re.M)
                                                         or a.count("\n") >= 8)
    r["steps_are_walking"] = not any(not _neg(a, m.start()) for m in _EXERCISE.finditer(a))
    r["no_invented_workout"] = not any(not _neg(a, m.start()) for m in _EXTRA_WORKOUT.finditer(a))

    # time of day
    ok_time = True
    if hour >= 22:
        for w, over in _MEAL_OVER.items():
            for m in re.finditer(rf"\b{w}s?\b", a, re.I):
                if not _neg(a, m.start()) and _EAT_VERB.search(a[max(0, m.start() - 50):m.end() + 20]) \
                        and not re.search(r"tomorrow", a[max(0, m.start() - 40):m.end() + 40], re.I):
                    ok_time = False
        for s in _sentences(a):
            if _MOVE.search(s) and _MOVE_VERB.search(s) and not _neg(s, len(s)) and not re.search(r"tomorrow", s, re.I):
                ok_time = False
        if truth["hour"] >= 23:
            for m in _LATE_FRAME.finditer(a):
                if not _neg(a, m.start()) and "tomorrow" not in _sent_of(a, m.start()).lower():
                    ok_time = False
    else:
        for w, over in _MEAL_OVER.items():
            for m in re.finditer(rf"\b{w}s?\b", a, re.I):
                if hour >= over and not _neg(a, m.start()) and _EAT_VERB.search(a[max(0, m.start() - 50):m.end() + 20]) \
                        and not re.search(r"tomorrow", a[max(0, m.start() - 40):m.end() + 40], re.I):
                    ok_time = False
    r["time_fits"] = ok_time

    # the calorie status told truthfully
    ok_cal = True
    if full:
        # eating more is the wrong advice when the day's calories are used up (unless the answer says not to)
        for m in re.finditer(r"\b(?:you (?:can|could) (?:still )?(?:have|eat|enjoy)|room for|go ahead and (?:have|eat)|grab|snack on|treat yourself|"
                             r"have a (?:snack|shake)|add (?:a|some))\b", a, re.I):
            if not _neg(a, m.start()):
                ok_cal = False
        if re.search(r"\b[\d,]{3,}\s*(?:kcal|calories)\s*(?:remaining|left|to go)\b", a, re.I):
            ok_cal = False
    if roomy and re.search(r"\b(over (?:your|the|today'?s)|exceeded|already over|used up|no room)\b", a, re.I) \
            and not re.search(r"\b(not|won'?t|wouldn'?t|isn'?t)\b[^.]{0,25}\bover\b", a, re.I):
        ok_cal = False
    r["calorie_truth"] = ok_cal

    # the step status told truthfully
    ok_steps = True
    hit = truth["steps_hit"]
    if hit:
        for m in re.finditer(r"\b(?:still need|need (?:another|to walk|more)|more steps|steps? (?:short|behind)|behind on steps|"
                             r"missed (?:your )?step|haven'?t (?:hit|reached))\b", a, re.I):
            if not _neg(a, m.start()):
                ok_steps = False
    else:
        for m in re.finditer(r"\b(?:step goal (?:is |was )?(?:reached|hit|met|done)|reached (?:your )?step goal|hit (?:your )?step goal|"
                             r"goal (?:is )?(?:reached|hit|met))\b", a, re.I):
            if not _neg(a, m.start()):
                ok_steps = False
    r["step_truth"] = ok_steps

    # ── by kind ──────────────────────────────────────────────────────────────
    r["answers_question"] = None
    if kind == "status":
        says_cal = bool(re.search(r"calorie|kcal|food|eat|budget|fuel", low))
        says_steps = bool(re.search(r"step|walk|move", low))
        r["answers_question"] = says_cal and says_steps
    elif kind == "steps_left":
        if hit:
            r["answers_question"] = bool(re.search(r"reached|hit|met|done|past|over|exceed|complete|achieved", low))
        else:
            fig = {str(truth["steps_left"]), f"{truth['steps_left']:,}"}
            # the exact count, or a close one (rounded to the hundred), or the steps taken so far with the goal
            near = {str(round(truth["steps_left"], -2)), f"{round(truth['steps_left'], -2):,}"}
            r["answers_question"] = any(f in a for f in fig | near)
    elif kind == "cal_left":
        if full:
            r["answers_question"] = bool(re.search(r"\b(none|nothing|no (?:more|room|calories)|used up|reached|at (?:your|the) target|"
                                                    r"over|0 kcal)\b", low))
        else:
            fig = {str(int(round(room))), f"{int(round(room)):,}", str(int(room)), f"{int(room):,}"}
            near = {str(int(round(room, -1))), f"{int(round(room, -1)):,}", str(int(round(room, -2))), f"{int(round(room, -2)):,}"}
            r["answers_question"] = any(f in a for f in fig | near)
    elif kind == "protein":
        r["answers_question"] = bool(re.search(r"protein", low)) and (
            bool(re.search(r"behind|short|low|missing|far|well|need|more|reached|hit|goal|on (?:plan|track)", low)))
    elif kind == "fit":
        m = re.search(r"(\d{2,4})\s*kcal", q)
        x = float(m.group(1)) if m else 0
        fits = (not full) and x <= room
        no_words = r"too much|too big|too many|more than (?:what|you|the)|over (?:your|the) target|further over|even more over|exceed|doesn'?t fit|does not fit|won'?t fit|no more food|" \
                   r"not fit|no room|go over|push(?:es)? you over|put you over|take you over|goes over|not (?:a good idea|advisable)|better to skip|skip"
        yes_words = r"should fit|will fit|fit well|good to have|you'?re good|no problem|sure\b|fits|fit(?:s)? (?:within|in)|within (?:your|what)|yes|fine|okay|ok\b|room for|comfortably|can (?:have|fit)|go ahead|works|good to go|that works"
        if fits:
            r["answers_question"] = bool(re.search(yes_words, low)) and not re.search(r"\b(too much|doesn'?t fit|does not fit|exceed)\b", low)
        else:
            r["answers_question"] = bool(re.search(no_words, low)) and not re.search(r"\bfits comfortably\b", low)
    elif kind == "named_food":
        # no calorie figure invented for the food: any "N kcal" must come from the allowed set (covered above); the answer
        # must say what the day has room for, in some form
        r["answers_question"] = bool(re.search(r"room|left|budget|target|over|used up|fit|portion|slice|amount|how (?:big|much)|"
                                               r"tell me|size|calories?", low))
        if full:
            r["answers_question"] = r["answers_question"] and bool(re.search(r"over|used up|no room|at (?:your|the) target|"
                                                                            r"reached|already|tomorrow|not today|better to skip|skip", low))
    elif kind == "what_eat":
        if truth["hour"] >= 22 or full:
            r["answers_question"] = bool(re.search(r"nothing|no (?:more|room)|used up|over|at (?:your|the) target|reached|"
                                                    r"tomorrow|done for|enough|wind down|already", low))
        else:
            r["answers_question"] = bool(re.search(r"protein|yogurt|shake|egg|chicken|tuna|cottage|banana|almond|snack|meal|"
                                                    r"dinner|lunch|rice", low))
    elif kind in ("steps_help", "steps_help_rain"):
        if hit:
            r["answers_question"] = bool(re.search(r"reached|hit|met|done|past|over|exceed|not behind|ahead|already", low))
        elif truth["hour"] >= 23:
            r["answers_question"] = bool(re.search(r"short|missed|day is over|tomorrow|ended|fell", low))
        else:
            r["answers_question"] = bool(_MOVE.search(a))
        if kind == "steps_help_rain" and not hit and truth["hour"] < 23:
            outside = any(not _neg(a, m.start()) and not re.search(r"\b(?:not|no|never|cannot)\b|n't\b", _sent_of(a, m.start()), re.I)
                          for m in _OUTSIDE.finditer(a))
            r["answers_question"] = r["answers_question"] and bool(_INDOOR.search(a)) and not outside
        elif kind == "steps_help" and not hit and truth["hour"] < 23:
            # weather unknown: leans indoors, an outdoor walk only conditionally
            r["answers_question"] = r["answers_question"] and bool(_INDOOR.search(a))
    elif kind == "walk_now":
        if truth["hour"] >= 22:
            r["answers_question"] = bool(re.search(r"late|tomorrow|wind down|rest|sleep|too late|not (?:now|tonight)|short|"
                                                    r"day is over|indoor|treadmill|stairs|short", low))
        else:
            r["answers_question"] = bool(re.search(r"yes|walk|treadmill|stairs|go|good", low))
    elif kind == "phase_eat":
        r["answers_question"] = bool(re.search(r"budget|target|plan|room|left|protein|dinner|meal|snack|food|calor", low))
    elif kind == "day_left":
        hl = truth.get("hours_left", 0)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
                 "eighteen": 18, "nineteen": 19, "twenty": 20}
        nums = [float(n) for n in re.findall(r"\b\d{1,2}(?:\.\d)?\b", low)] + [float(v) for w, v in words.items() if re.search(rf"\b{w}\b", low)]
        near = any(abs(n - hl) <= 1.5 for n in nums)
        r["answers_question"] = near or (hl <= 1 and bool(re.search(r"nearly over|almost over|end of the day|over", low)))
    elif kind == "greeting":
        r["answers_question"] = len(_sentences(a)) <= 3 and not re.search(r"\b\d{3,}\b", a)
    elif kind == "workout":
        r["answers_question"] = bool(re.search(r"blueprint|/workout|your plan|your (?:training|workout) plan|can'?t|cannot|"
                                               r"only (?:your|the) plan|not able|walk|steps", low))
    elif kind == "log_list":
        names = [n.lower() for n in truth["entries"]]
        r["answers_question"] = sum(1 for n in names if n in low) >= max(1, len(names) - 1)
    # stale step data must be mentioned when the answer is about the step count
    talks_steps = kind in ("steps_left", "steps_help", "walk_now") or (kind == "status" and bool(re.search(r"step|walk", low)))
    r["stale_noted"] = (bool(re.search(r"\bold\b|stale|hours|sync|delay|behind the real|may be|might not|out of date|not up to date", low))
                        if (truth["steps_age"] >= 180 and talks_steps) else None)
    return r


def passed(scores: dict) -> bool:
    return all(v is not False for v in scores.values())


def failed_items(scores: dict) -> list[str]:
    return [k for k, v in scores.items() if v is False]
