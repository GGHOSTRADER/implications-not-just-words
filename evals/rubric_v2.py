"""Core rubric, version 2: ONE fixed checklist applied to every approach, including a SEMANTIC kind.

Why a second rubric. Rubric v1 (evals/weekly_rubric.py, unchanged) grew per-approach variants and reused the system's
own guard functions (weekly_analysis._grounding_check, ...). Two problems for measurement:
  1. approaches were scored by slightly different item sets, so totals were not comparable;
  2. it measured format and hallucination, not whether the advice was RIGHT for the goal: on the holdout weeks the
     labels-v3 approach scored below labels v1 (6/10 vs 8/10 all items passed) while winning 10/10 blind judgments.

This module is the evaluation instrument for the report. It is deliberately SELF-CONTAINED: it imports nothing from
the system under test (weekly_analysis, weekly_labels*), so changing a prompt, a label or a guard can never change
how answers are measured. Every rule depends only on the week's raw data and the answer text. If a rule must change,
bump RUBRIC_VERSION and re-score all saved answers (evals/report.py does that on every build).

Items are only added when they apply (e.g. "no claims without data" only for a week with a metric missing), so
compare pass RATES per item, not raw totals.

Kinds: hallucination, instruction, coverage, specificity, semantic. The semantic items encode a small, explicit domain
spec (see SEMANTIC SPEC below). Labels v3/v4 were designed with the same principles, which favours them on these
items: the report states this as a threat to validity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# core-v2.1: after reading the flags (evals/rubric_v2_reading_round1.json, rubric_v2_validation.json): contradictions
# read from what an increase word governs, negations inside the span, calorie trends vs intake, wider food vocabulary.
RUBRIC_VERSION = "core-v2.1 (2026-10-04)"
KINDS = ("hallucination", "instruction", "coverage", "specificity", "semantic")


@dataclass
class Item:
    kind: str
    id: str
    ok: bool
    detail: str = ""


# One line per item, used by the report's method section.
ITEM_DOCS = {
    "no_invented_numbers": "every figure about the week (>7) matches a data value or target within 2%; in the "
                           "recommendations, proposals such as '20 minutes' or 'at least 30g' are allowed",
    "respects_target_status": "no recommendation moves a metric the wrong way: more steps/protein when already met, "
                              "more calories when over, fewer when under (negations such as 'without adding' excluded)",
    "no_claims_without_data": "a metric with no data this week is not described",
    "two_paragraphs": "exactly two paragraphs (observations, recommendations)",
    "no_markdown": "no markdown syntax",
    "no_label_or_heading": "no 'Observations:'-style label or heading",
    "no_greeting": "does not open with a greeting",
    "no_sign_off": "does not close with a sign-off",
    "about_four_lines_each": "each paragraph 150-700 characters",
    "observations_hold_no_advice": "the observations paragraph contains no advice wording",
    "one_to_three_numbered_recommendations": "1-3 recommendations numbered from 1",
    "observations_cover_each_metric": "the observations discuss every metric that has data",
    "recommendations_address_every_off_target_metric": "every off-target metric is named in the recommendations",
    "every_recommendation_is_concrete": "each recommendation names an amount, time, place or habit",
    "no_filler_phrases": "no filler such as 'stay consistent' or 'keep up the good work'",
    "no_keep_only_recommendation": "no recommendation slot is spent only on keeping something as it is",
    "fix_matches_problem": "each problem gets an action of the matching kind: calories over -> eat less (a food "
                           "change, not exercise); calories under -> eat more; protein short -> a protein food; "
                           "steps short -> more movement; no workouts -> a training session",
    "first_recommendation_targets_a_problem": "when something is off target, recommendation 1 addresses a problem",
    "large_calorie_gap_first": "when calories are off by 15% or more, recommendation 1 is a calorie (food) change",
    "no_extra_movement_while_underfed": "when calories are under target, no added-movement recommendation comes "
                                        "before a food recommendation (more movement would deepen the deficit)",
    "no_unsupported_link": "the observations claim no connection the data cannot support: time of day or workouts "
                           "linked to eating, or two metrics linked causally unless the link follows from their verdicts",
    "consequence_matches_metric": "in the observations, a 'which means ...' consequence is attached to the metric it "
                                  "belongs to",
}

# ── reference state from the raw data (independent of every label module) ───

CALORIE_BAND = 0.08          # within 8% of the calorie target is on target
LARGE_CALORIE_GAP = 0.15
_GOAL_KEYS = {"steps": "step_goal", "protein": "protein_target", "calories": "calories_target"}


def reference_state(data: dict) -> dict:
    """{'steps': 'met'|'below'|'nodata', 'protein': ..., 'calories': 'on'|'over'|'under'|'nodata',
        'cal_gap': fraction, 'workouts': n}"""
    g = data["goals"]
    out = {}
    for key in ("steps", "protein"):
        m, goal = data[key], g.get(_GOAL_KEYS[key], 0)
        out[key] = "nodata" if (m.get("status") != "ok" or not goal) else ("met" if m["avg"] >= goal else "below")
    c, target = data["calories"], g.get("calories_target", 0)
    if c.get("status") != "ok" or not target:
        out["calories"], out["cal_gap"] = "nodata", 0.0
    else:
        rel = (c["avg"] - target) / target
        out["cal_gap"] = abs(rel)
        out["calories"] = "over" if rel > CALORIE_BAND else "under" if rel < -CALORIE_BAND else "on"
    out["workouts"] = len(data.get("workouts") or [])
    return out


def problems(state: dict) -> list[str]:
    """Everything that is off target, plus 'workouts' when there were none."""
    out = [k for k in ("steps", "protein") if state[k] == "below"]
    if state["calories"] in ("over", "under"):
        out.append("calories")
    if state["workouts"] == 0:
        out.append("workouts")
    return out


# ── text helpers ─────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_PROPOSAL_RE = re.compile(
    r"\d+(?:\.\d+)?\s*-?\s*(?:minutes?|mins?|hours?|hrs?|reps?|sets?|times|seconds?|secs?|days?|sessions?|weeks?)\b"
    r"|(?:at least|extra|additional|another|up to|around|about)\s+\d+(?:\.\d+)?\s*(?:g|grams?|steps?|kcal|calories)?\b"
    r"|\b(?:consume|add|eat|have|take|drink|include|aim for)\s+(?:an?\s+)?\d+(?:\.\d+)?\s*(?:g|grams?)\b"
    r"|\ba\s+\d+(?:\.\d+)?\s*(?:g|grams?|calories|kcal)\s+[a-z]"
    r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b", re.I)

_GREETING = re.compile(r"^\s*(hi|hey|hello|dear|good (morning|evening)|great (job|week))\b", re.I)
_SIGNOFF = re.compile(r"(let me know|good luck|keep (it up|crushing)|you('ve| have) got this|cheers|best,)", re.I)
_LABEL = re.compile(r"^\s*(\*\*)?(observations?|recommendations?|analysis|summary|stage \d)\s*[:\-]", re.I | re.M)
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|`|^\s*[-*]\s+\S)", re.M)
_ADVICE = re.compile(r"\b(recommend|should|consider|try to|aim to|i suggest|suggest that|you could|make sure|focus on)\b", re.I)
_GENERIC = re.compile(r"(stay consistent|keep it up|keep up the (good|great) work|do your best|keep pushing|"
                      r"stay (the course|motivated|focused)|maintain consistency|continue (to )?(do|doing) what)", re.I)
_ANCHOR = re.compile(r"(\d|minutes?|hours?|daily|each day|every day|per day|a day|weekly|each week|twice|once|"
                     r"morning|evening|afternoon|lunch|dinner|breakfast|snack|before|after|bedtime|"
                     r"gram|serving|portion|scoop|shake|walk|session|workout|"
                     r"meat|dairy|source|fish|egg|chicken|yogurt|tofu|vegetable|stairs|jog|run|cycle|swim|gym|lift|sleep|"
                     r"water|track|log|weigh|measure|plan|prep|schedule|reminder|break)", re.I)
# "Keep calories exactly as they are" spends a slot on nothing. "Keep a water bottle by the bed" or "keep a step goal
# visible" starts with keep but adds a concrete device or habit, so those are excluded (validated by reading flags).
_KEEP_ONLY = re.compile(r"^\W*(?:keep|maintain|continue|stick|hold)\b(?!.*\b(?:walk|snack|meal|shake|portion|session|"
                        r"breakfast|lunch|dinner|minutes?|bottle|tracker|reminder|visible|logging|log|breaks?|timer|"
                        r"alarm|counter|journal|app|scale|prep|stretch\w*|(?<!without )adding|(?<!without )add)\b)", re.I)

METRIC = {
    "steps": re.compile(r"\b(steps?|walk\w*|move|moving|movement|activity|active|stairs|cardio)\b", re.I),
    "protein": re.compile(r"\b(protein)\b", re.I),
    "calories": re.compile(r"\b(calor\w*|kcal|intake|portions?|eat\w*|deficit|surplus|food)\b", re.I),
    "workouts": re.compile(r"\b(workouts?|training|strength|gym|weights?|lifting|resistance|exercise sessions?)\b", re.I),
}
# names that identify a metric in a sentence (narrower than METRIC: 'eating' does not name calories)
_NAME = {
    "steps": re.compile(r"\b(steps?|step count|movement)\b", re.I),
    "protein": re.compile(r"\bprotein\b", re.I),
    "calories": re.compile(r"\b(calor\w*\b(?!\s+(?:burn\w*|expenditure|spent))|kcal)\b", re.I),   # burn = movement side
    "workouts": re.compile(r"\b(workouts?|training|sessions)\b", re.I),
}

# kind-matched actions (semantic spec). Validated by reading every flag on the saved answers (2026-10-04): the verb
# lists are broad on purpose ("consume 30g of protein", "prioritise high-protein snacks", "incorporating meals").
_FOOD_LESS = re.compile(r"\b(small(?:er)?|trim\w*|cut\w*|reduc\w*|swap\w*|replac\w*|skip\w*|less|light(?:er)?|half|"
                        r"limit\w*|fewer|drop\w*|remov\w*|instead of|lower-calorie|low-calorie|portion control|avoid\w*|"
                        r"adjust\w*|control\w*|curb\w*|in check)\b", re.I)
_FOOD_MORE = re.compile(r"\b(add\w*|extra|larger|bigger|more|another|second|calorie-dense|energy-dense|increas\w*|"
                        r"boost\w*|includ\w*|have|eat\w*|consum\w*|incorporat\w*)\b", re.I)
_ACTION = re.compile(r"\b(add\w*|have|eat\w*|consum\w*|includ\w*|prioriti[sz]\w*|swap\w*|replac\w*|choos\w*|start\w*|"
                     r"drink\w*|take|boost\w*|increas\w*|aim\w*|get|plan\w*|prepar\w*|make|incorporat\w*|serve|pack\w*|"
                     r"put|use|schedul\w*)\b", re.I)
# increase wording, for 'more of something already met' (protein, steps)
_INCREASE = re.compile(r"\b(add\w*|boost\w*|increas\w*|extra|more|additional|higher|another|rais\w*|rise|further)\b", re.I)
# An action only fixes a food problem when it GOVERNS a food word: "add a walk after dinner" mentions dinner but its
# action is the walk. So the verb and the food word must be close, with no movement word between them (2026-10-04,
# found when 'Add a 20-minute walk after dinner ... step count' passed as a food-side fix).
_MOVEMENT_WORD = r"(?:walk\w*|steps?|stairs|jog\w*|run\w*|cycl\w*|movement|activity|hike|stroll)"
_FOOD_OBJECT = (r"(?:calor\w*|kcal|intake|consumption|food|meals?|portions?|snack\w*|plates?|servings?|breakfast|lunch|"
                r"dinner|desserts?|treats?|(?:over)?eat\w*)")


def _governs(verbs: re.Pattern, objects: str, item: str, after: str | None = None) -> bool:
    """The verb comes BEFORE the food word it acts on. `after` optionally allows a describing adjective after the food
    word ('portions smaller'); an arbitrary later verb is not allowed ('dinner to boost steps' is not eating more)."""
    gap = rf"(?:(?!\b{_MOVEMENT_WORD}\b)[^.]){{0,45}}?"
    pat = rf"(?:{verbs.pattern}){gap}\b{objects}\b"
    if after:
        pat += rf"|\b{objects}\b\W+(?:\w+\W+){{0,2}}?\b(?:{after})\b"
    return bool(re.search(pat, item, re.I))


_FOOD_TRACK = re.compile(r"\b(track\w*|log\w*|monitor\w*|weigh\w*|measur\w*|count\w*)\b", re.I)   # food-side, either way
_FOOD_WORD = re.compile(r"\b(portions?|snacks?|meals?|breakfast|lunch|dinner|plate|serving|food|calorie\w*|kcal|dessert|"
                        r"drink\w*|shake|smoothie|nuts|bread|rice|pasta|oats|fruit|yogurt|cheese|eggs?|chicken|fish|tofu)\b", re.I)
_PROTEIN_FOOD = re.compile(r"\b(protein|shake|whey|eggs?|chicken|fish|tuna|salmon|yogurt|cottage|tofu|lean|beef|turkey|"
                           r"beans|lentils|cheese|milk|quark|tempeh|edamame)\b", re.I)
_MOVE = re.compile(r"\b(walk\w*|steps?|stairs|stroll|hike|jog\w*|run\w*|cycl\w*|bike|move|moving|movement|active break|"
                   r"pace|commute)\b", re.I)
_TRAIN = re.compile(r"\b(workouts?|session|strength|training|gym|weights?|lift\w*|resistance|bodyweight|circuit|"
                    r"push-?ups?|squats?)\b", re.I)
_MOVE_MORE = re.compile(r"\b(walk\w*|steps?|stairs|jog\w*|run\w*|cycl\w*|hike|stroll|cardio|movement|workout|session|"
                        r"exercise)\b", re.I)

# links (semantic spec): cues that a clause claims a connection. Only relational wording counts; interpretive words
# such as "suggests" or "reflects" do not (validated by reading flagged sentences, 2026-10-04).
_LINK_CUE = re.compile(r"\b(align\w* with|correlat\w*|link\w* (?:to|with|between)|connect\w*|contribut\w* to|"
                       r"influenc\w*|due to|because of|caus\w*|driv\w*|led to|leading to|result\w* (?:in|from)|tied to|"
                       r"associated with|mirror\w*|coincid\w*|as a result of|impact\w* on|affect\w*)\b", re.I)
# "aligns with the goal / the target / the schedule" restates a status; it is not a link between two things
_BENIGN_ALIGN = re.compile(r"\balign\w* with (?:the |your |their |a |its )?(?:\w+ ){0,2}?(?:goals?|targets?|frequency|plan|"
                           r"routine|schedule|split|days?|phase)\b", re.I)
_CLAUSE_SPLIT = re.compile(r";|,\s*(?:while|whereas|but|and yet|and)\b|\s+while\s+|\s+whereas\s+", re.I)
_NEGATED_CUE = re.compile(r"\b(?:no|not|n't|without|never)\b(?:\W+\w+){0,3}\W*$", re.I)
_TIME_OF_DAY = re.compile(r"\b(early evening|evening|morning|afternoon|time of day|peak (?:step )?hour|most active time|"
                          r"active time|\d{1,2}:\d{2})\b", re.I)
# eating, not burning: "calorie burn" / "calories burned" are about expenditure
_INTAKE = re.compile(r"\b(calor\w*\b(?!\s+(?:burn\w*|expenditure|spent))|kcal|intake|eat\w*|meals?|food|protein|nutrition|"
                     r"diet\w*)\b", re.I)          # the \b after calor\w* stops backtracking past the lookahead
_WORKOUT_MENTION = re.compile(r"\b(workouts?|training|sessions?|workout days?)\b", re.I)

# consequences (semantic spec): which metric each consequence belongs to
_MEANS_CUE = re.compile(r"\b(which means|this means|that means|meaning|means that|means)\b", re.I)
_CONSEQUENCE = {
    "steps": re.compile(r"\b(less movement|fewer calories burned|calories burned|burn\w*|activity level)\b", re.I),
    "protein": re.compile(r"\b(recovery|growth|repair)\b", re.I),
    "calories": re.compile(r"\b(surplus|fat gain|gain fat|weight (?:will|is likely to)|creep up|drift down)\b", re.I),
}


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


def numbered_items(rec_paragraph: str) -> list[tuple[int, str]]:
    parts = re.split(r"(?:(?<=\s)|^)(\d)[.)]\s+", rec_paragraph)
    return [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _numbers(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUMBER_RE.findall(text or "")}


def _data_numbers(data: dict) -> set[float]:
    """Every figure a correct answer could cite: averages, targets, gaps, slopes, CVs, counts, peak hour."""
    g, out = data["goals"], set()
    for k in ("step_goal", "calories_target", "protein_target"):
        if g.get(k):
            out.add(float(g[k]))
    for key, goal_key in _GOAL_KEYS.items():
        m = data[key]
        if m.get("status") == "ok":
            out |= {float(m["avg"]), abs(float(m["slope"])), float(m["coefficient_of_variation_pct"])}
            if g.get(goal_key):
                out.add(abs(float(m["avg"] - g[goal_key])))
    h = data.get("hourly") or {}
    if h.get("status") == "ok":
        out |= {float(h["peak_hour"]), float(h["peak_avg_steps"])}
    out |= {float(len(data.get("workouts") or [])), float(data.get("total_days") or 7)}
    if data.get("goal_hit_days") is not None:
        out.add(float(data["goal_hit_days"]))
    return out


def invented_numbers(text: str, data: dict, allow_proposals: bool) -> list[float]:
    real = _data_numbers(data)
    body = _PROPOSAL_RE.sub(" ", text) if allow_proposals else text
    bad = []
    for n in _numbers(body):
        n = abs(n)
        if n <= 7:
            continue
        if any(abs(n - r) <= 0.5 or (r and abs(n - r) / r <= 0.02) for r in real):
            continue
        bad.append(n)
    return sorted(set(bad))


_NEGATION = re.compile(r"\b(?:without|avoid\w*|not|never|no|don'?t|do not|instead of|rather than)\b[^.]{0,25}$", re.I)
_MORE = r"(?:increas\w*|rais\w*|boost\w*|add\w*\s+more|get\w*\s+more|more|higher|up\s+your|extra|additional)"
_CAL_MORE = re.compile(r"\b(?:increas\w*|rais\w*|boost\w*|more|higher|add\w*|extra)\b[^.]{0,40}\b(?:calories|kcal|calorie intake)\b", re.I)
_CAL_LESS = re.compile(r"\b(?:reduc\w*|cut\w*|lower\w*|less|decreas\w*|trim\w*|fewer)\b[^.]{0,40}\b(?:calories|kcal|calorie intake)\b", re.I)
# eating MORE dense food; merely mentioning it ("keep a small portion of a high-calorie food ... once a day") is not
_CAL_DENSE = re.compile(r"\b(?:add\w*|eat\w*|have|prioriti[sz]\w*|includ\w*|choos\w*|more)\b[^.]{0,30}"
                        r"\b(?:calorie-dense|energy-dense|high-calorie)\b", re.I)


def _asks(pattern: re.Pattern, text: str) -> bool:
    return any(not _NEGATION.search(text[:m.start()]) for m in pattern.finditer(text))


_TREND_OBJECT = re.compile(r"^[^.]{0,20}\b(slope|trend|decline|drop|variab\w*|fluctuat\w*|swings?)\b", re.I)


def _asks_more_of(key: str, text: str) -> bool:
    """An increase word that GOVERNS the metric ('add an extra serving of protein', 'further boost daily steps'), not
    negated ('without adding more food or protein', 'avoid increasing it')."""
    objects = (rf"(?:{_PROTEIN_FOOD.pattern[2:-2]}|high-protein|protein-rich)" if key == "protein"
               else r"(?:steps?|step count|walking)")
    # the increase word governs a word at most 4 words on, never across a full stop, semicolon or colon (round 2 found
    # '... leverage higher step counts. Maintain current protein' read as more protein)
    gap = r"[^\w.;:]+(?:[\w'-]+[^\w.;:]+){0,4}?"
    for m in re.finditer(rf"(?:{_INCREASE.pattern}){gap}{objects}\b", text, re.I):
        if _NEGATION.search(text[:m.start()]) or re.search(r"\b(without|not|no|avoid\w*|never)\b", m.group(0), re.I):
            continue
        return True
    return False


def _asks_less_calories(text: str) -> bool:
    """'cut calories' asks for less food; 'reduce the -40 kcal/day slope' asks to stop a decline, which is not."""
    for m in _CAL_LESS.finditer(text):
        if _NEGATION.search(text[:m.start()]) or _TREND_OBJECT.search(text[m.end():]):
            continue
        return True
    return False


def contradictions(recs: str, state: dict) -> list[str]:
    out = []
    for key in ("protein", "steps"):
        if state[key] == "met" and _asks_more_of(key, recs):
            out.append(f"asks for more {key}, already met")
    if state["calories"] == "over" and (_asks(_CAL_MORE, recs) or _asks(_CAL_DENSE, recs)):
        out.append("asks for more calories while over target")
    if state["calories"] == "under" and _asks_less_calories(recs):
        out.append("asks for fewer calories while under target")
    return out


def _is_food_action(item: str, direction: str) -> bool:
    """A food-side change in the given direction, or tracking what is eaten (either direction). The verb must govern a
    food word (see _governs)."""
    verbs = _FOOD_LESS if direction == "less" else _FOOD_MORE
    after = "smaller|lighter|less|fewer|halved" if direction == "less" else "larger|bigger|extra|doubled"
    return _governs(verbs, _FOOD_OBJECT, item, after) or _governs(_FOOD_TRACK, _FOOD_OBJECT, item)


def kind_matched(item: str, problem: str, state: dict) -> bool:
    """Does this recommendation fix `problem` with an action of the matching kind?"""
    if problem == "calories":
        return _is_food_action(item, "less" if state["calories"] == "over" else "more")
    if problem == "protein":
        return _governs(_ACTION, rf"(?:{_PROTEIN_FOOD.pattern[2:-2]}|high-protein \w+)", item)
    if problem == "steps":
        return bool(_MOVE.search(item))
    if problem == "workouts":
        return bool(_TRAIN.search(item))
    return False


def _allowed_link_pairs(state: dict) -> set[frozenset]:
    """Connections that follow from the verdicts themselves (no extra data needed)."""
    s, out = state, set()
    if s["calories"] == "over" and s["steps"] in ("met", "below"):
        out.add(frozenset({"steps", "calories"}))         # activity fine / both wrong: the food side is the gap
    if s["protein"] == "below" and s["calories"] in ("under", "on", "over"):
        out.add(frozenset({"protein", "calories"}))       # short because food is short / a swap, not more food
    if s["calories"] == "under" and (s["steps"] == "met" or s["workouts"] >= 5):
        out.add(frozenset({"steps", "calories"}))         # a lot of activity on too little food
        out.add(frozenset({"workouts", "calories"}))
    if s["steps"] == "below" and s["calories"] == "on":
        out.add(frozenset({"steps", "calories"}))         # food on plan: movement holds the deficit back
    if s["workouts"] == 0 and s["steps"] == "below":
        out.add(frozenset({"steps", "workouts"}))
    return out


def _claims_link(clause: str) -> bool:
    """A relational cue that is not just 'aligns with the goal/target' and is not negated ('no direct correlation')."""
    return any(not _BENIGN_ALIGN.match(clause, m.start()) and not _NEGATED_CUE.search(clause[:m.start()])
               for m in _LINK_CUE.finditer(clause))


def unsupported_links(text: str, state: dict) -> list[str]:
    """Sentences with a clause claiming a connection the data cannot support. Pass the OBSERVATIONS: advice such as
    'a snack in the early evening, to line up with your most active time' is a suggestion, not a claim about data."""
    allowed, bad = _allowed_link_pairs(state), []
    for s in sentences(text):
        for clause in _CLAUSE_SPLIT.split(s):
            if not _claims_link(clause):
                continue
            if _TIME_OF_DAY.search(clause) and _INTAKE.search(clause):
                bad.append(s)                               # we have no meal-timing data at all
                break
            if _WORKOUT_MENTION.search(clause) and re.search(r"\b(calor\w*|intake|protein|eating|meals?)\b", clause, re.I) \
                    and frozenset({"workouts", "calories"}) not in allowed:
                bad.append(s)                               # no per-day pairing of workouts with intake
                break
            named = [k for k in ("steps", "protein", "calories") if _NAME[k].search(clause)]
            if len(named) >= 2:
                pairs = {frozenset({a, b}) for i, a in enumerate(named) for b in named[i + 1:]}
                if not pairs & allowed:
                    bad.append(s)
                    break
    return bad


def misattributed_consequences(text: str) -> list[str]:
    """'Protein was under, which means less movement ...': a consequence that belongs to a different metric."""
    bad = []
    for s in sentences(text):
        cue = _MEANS_CUE.search(s)
        if not cue:
            continue
        before, after = s[:cue.start()], s[cue.start():]
        named = [k for k in ("steps", "protein", "calories") if _NAME[k].search(before)]   # the subject, before the cue
        if len(named) != 1:
            continue
        owners = [k for k, pat in _CONSEQUENCE.items() if pat.search(after)]
        if owners and named[0] not in owners:
            bad.append(s)
    return bad


# ── the checklist ────────────────────────────────────────────────────────────

def score(text: str, data: dict) -> list[Item]:
    """Every applicable core item for one answer (`text` is the full joined review)."""
    state = reference_state(data)
    paras = paragraphs(text)
    two = len(paras) == 2
    obs = paras[0] if paras else ""
    recs = paras[1] if len(paras) > 1 else ""
    items = numbered_items(recs)
    rec_texts = [t for _n, t in items]
    probs = problems(state)
    out: list[Item] = []
    add = lambda kind, iid, ok, detail="": out.append(Item(kind, iid, bool(ok), "" if ok else detail))

    # hallucination
    bad = (invented_numbers(obs, data, False) + invented_numbers(recs, data, True)) if two \
        else invented_numbers(text, data, False)
    add("hallucination", "no_invented_numbers", not bad, f"not in the data: {sorted(set(bad))}")
    con = contradictions(recs if two else text, state)
    add("hallucination", "respects_target_status", not con, "; ".join(con))
    for key in ("steps", "protein", "calories"):
        if state[key] == "nodata":
            hit = _NAME[key].search(text)
            add("hallucination", "no_claims_without_data", not hit, f"mentions {key}, which has no data")

    # instruction following
    add("instruction", "two_paragraphs", two, f"{len(paras)} paragraphs")
    add("instruction", "no_markdown", not _MARKDOWN.search(text))
    add("instruction", "no_label_or_heading", not _LABEL.search(text))
    add("instruction", "no_greeting", not _GREETING.search(text))
    add("instruction", "no_sign_off", not _SIGNOFF.search(paras[-1] if paras else text))
    lens = [len(p) for p in paras[:2]]
    add("instruction", "about_four_lines_each", lens and all(150 <= n <= 700 for n in lens), f"lengths {lens}")
    adv = _ADVICE.search(obs) if two else None
    add("instruction", "observations_hold_no_advice", not adv, adv.group(0) if adv else "")
    nums = [n for n, _t in items]
    add("instruction", "one_to_three_numbered_recommendations",
        nums and 1 <= len(nums) <= 3 and nums == list(range(1, len(nums) + 1)), f"found {nums}")

    # coverage
    missing = [k for k in ("steps", "calories", "protein") if state[k] != "nodata" and not METRIC[k].search(obs)]
    add("coverage", "observations_cover_each_metric", not missing, f"not discussed: {missing}")
    off = [k for k in ("steps", "protein", "calories") if k in probs]
    unaddressed = [k for k in off if not METRIC[k].search(recs)]
    add("coverage", "recommendations_address_every_off_target_metric", not unaddressed, f"not addressed: {unaddressed}")

    # specificity
    vague = [t[:50] for t in rec_texts if not _ANCHOR.search(t)]
    add("specificity", "every_recommendation_is_concrete", rec_texts and not vague,
        f"no concrete action in: {vague}" if rec_texts else "no numbered recommendations")
    filler = _GENERIC.search(text)
    add("specificity", "no_filler_phrases", not filler, filler.group(0) if filler else "")
    keep_only = [n for n, t in items if _KEEP_ONLY.search(t)]
    add("specificity", "no_keep_only_recommendation", not keep_only, f"keep-only items: {keep_only}")

    # semantic. These judge CONTENT, so an unnumbered recommendations paragraph is read sentence by sentence (its format
    # failure is already counted by the instruction items; it should not be counted twice here).
    units = rec_texts or [s for s in sentences(recs) if not re.fullmatch(r"\W*recommendations?\W*", s, re.I)]
    first = units[0] if units else ""
    if probs:
        # while under-eating, steps are not required: more movement would deepen the deficit (eat first)
        required = [p for p in probs if not (p == "steps" and state["calories"] == "under")]
        unfixed = [p for p in required if not any(kind_matched(t, p, state) for t in units)]
        add("semantic", "fix_matches_problem", not unfixed, f"no matching kind of fix for: {unfixed}")
        targets = first and not _KEEP_ONLY.search(first) and any(
            METRIC[p].search(first) or kind_matched(first, p, state) for p in probs)
        add("semantic", "first_recommendation_targets_a_problem", targets, f"recommendation 1: {first[:70]!r}")
    if state["calories"] in ("over", "under") and state["cal_gap"] >= LARGE_CALORIE_GAP:
        add("semantic", "large_calorie_gap_first", first and kind_matched(first, "calories", state),
            f"calories off by {state['cal_gap']:.0%}; recommendation 1: {first[:70]!r}")
    if state["calories"] == "under":
        order_ok, seen_food = True, False
        for t in units:
            if _is_food_action(t, "more") or (_PROTEIN_FOOD.search(t) and _FOOD_MORE.search(t)):
                seen_food = True
            elif _MOVE_MORE.search(t) and not seen_food and not re.search(r"\bonce\b|\bafter\b[^.]{0,30}\b(food|eating)", t, re.I):
                order_ok = False
                break
        add("semantic", "no_extra_movement_while_underfed", order_ok, "adds movement before fixing food")
    links = unsupported_links(obs if two else text, state)      # claims live in the observations
    add("semantic", "no_unsupported_link", not links, f"{len(links)} unsupported: {links[0][:90]!r}" if links else "")
    mis = misattributed_consequences(obs if two else text)
    add("semantic", "consequence_matches_metric", not mis, mis[0][:90] if mis else "")
    return out


def failed(items: list[Item]) -> list[Item]:
    return [i for i in items if not i.ok]


def summarize(items: list[Item]) -> dict:
    by = {k: [0, 0] for k in KINDS}
    for i in items:
        by[i.kind][1] += 1
        by[i.kind][0] += i.ok
    return {**{k: tuple(v) for k, v in by.items()}, "all": all(i.ok for i in items)}
