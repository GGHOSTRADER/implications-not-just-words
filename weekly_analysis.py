"""Weekly strategic analysis — backs the /analysis Telegram command.

Pipeline:
1. gather_weekly_analysis_data() — pure Python, reuses agent._compute_weekly_trend /
   agent.get_hourly_step_profile / agent._load_goals / db_store.get_recent_workouts
   so the numbers match what the Strategic Agent and weekly_trends_report.py see.
2. format_data_summary() — plain-text "Python print" block sent to the user first.
3. run_cot_analysis() — one CoT-structured LLM call (role "strategic", the big model)
   producing two short paragraphs, followed by a deterministic regex grounding-check
   and at most one redraft if the draft cites numbers not present in the data.
4. generate_report_charts() — (re)generates the three weekly_trends_*.png reports.
5. run_weekly_analysis() — orchestrates 1-4 for /analysis.
6. run_analysis_followup() — grounded multi-turn Q&A over the cached analysis context,
   using the 12-message (6-exchange) analysis_history.

Usage: python weekly_analysis.py
"""
from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parent

# "2,200" is ONE number (2200). Read as two, it never matched a model writing "2200", and "10,000" became 10 and 0.
_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def gather_weekly_analysis_data() -> dict:
    """Precompute the weekly metrics needed for the strategic review."""
    from agent import _compute_weekly_trend, _load_goals, _store, get_hourly_step_profile

    steps = _compute_weekly_trend(lambda e: e.steps)
    calories = _compute_weekly_trend(lambda e: e.calories)
    protein = _compute_weekly_trend(lambda e: e.protein)
    calorie_delta = _compute_weekly_trend(lambda e: e.calorie_delta)

    goal_hit_days = None
    total_days = None
    if calorie_delta["status"] == "ok":
        total_days = calorie_delta["days"]
        goal_hit_days = sum(1 for v in calorie_delta["values"] if abs(v) < 10)

    hourly = get_hourly_step_profile.func()
    goals = _load_goals()

    from agent import _local_now
    today = _local_now().date()
    cutoff_start = (today - timedelta(days=7)).isoformat()
    cutoff_end = (today - timedelta(days=1)).isoformat()
    workouts_raw = _store.get_recent_workouts(n=10)
    workouts = [
        {"date": w["date"], "type": w.get("type", "")}
        for w in workouts_raw
        if w.get("date") and cutoff_start <= w["date"] <= cutoff_end
    ]

    return {
        "steps": steps,
        "calories": calories,
        "protein": protein,
        "calorie_delta": calorie_delta,
        "goal_hit_days": goal_hit_days,
        "total_days": total_days,
        "hourly": hourly,
        "goals": goals,
        "workouts": workouts,
    }


def format_data_summary(data: dict) -> str:
    """Plain-text 'Python print' block — sent to the user before the LLM analysis."""
    goals = data["goals"]
    hourly = data["hourly"]
    steps, calories, protein = data["steps"], data["calories"], data["protein"]
    workouts = data["workouts"]
    goal_hit_days = data["goal_hit_days"]
    total_days = data["total_days"] or 7

    lines = [f"Phase: {goals.get('phase', 'unknown')}", ""]

    if hourly.get("status") == "ok":
        lines.append(f"Peak step hour: {hourly['peak_hour']:02d}:00 (avg {hourly['peak_avg_steps']:.0f} steps)")
    else:
        lines.append("Peak step hour: no hourly step data yet")
    lines.append("")

    lines.append("Stability this week (avg / CV% / slope per day):")
    if steps["status"] == "ok":
        lines.append(f"  Steps:    avg {steps['avg']:.0f}, CV {steps['coefficient_of_variation_pct']:.1f}%, slope {steps['slope']:+.1f}/day")
    if calories["status"] == "ok":
        lines.append(f"  Calories: avg {calories['avg']:.0f} kcal, CV {calories['coefficient_of_variation_pct']:.1f}%, slope {calories['slope']:+.1f} kcal/day")
    if protein["status"] == "ok":
        lines.append(f"  Protein:  avg {protein['avg']:.0f} g, CV {protein['coefficient_of_variation_pct']:.1f}%, slope {protein['slope']:+.1f} g/day")
    lines.append("")

    step_goal = goals.get("step_goal", 0)
    protein_target = goals.get("protein_target", 0)
    lines.append("Vs targets (precomputed — use as-is, do not recompute):")
    if steps["status"] == "ok" and step_goal:
        delta = steps["avg"] - step_goal
        status = "target met" if delta >= 0 else "below target"
        lines.append(f"  Steps:    {steps['avg']:.0f} vs {step_goal:,.0f} target ({delta:+.0f}, {status})")
    if protein["status"] == "ok" and protein_target:
        delta = protein["avg"] - protein_target
        status = "target met/exceeded" if delta >= 0 else "below target"
        lines.append(f"  Protein:  {protein['avg']:.0f}g vs {protein_target:.0f}g target ({delta:+.0f}g, {status})")
    lines.append("")

    if workouts:
        detail = ", ".join(f"{w['date']} ({w['type']})" for w in workouts)
        lines.append(f"Workouts this week: {len(workouts)}/{total_days} — {detail}")
    else:
        lines.append(f"Workouts this week: 0/{total_days}")
    lines.append("")

    if goal_hit_days is not None:
        lines.append(f"Caloric goal hit (within +/-10 kcal of target): {goal_hit_days}/{total_days} days")
    else:
        lines.append("Caloric goal hit: not enough data yet")

    return "\n".join(lines)


def _data_block(data: dict, summary_text: str) -> str:
    """The identity line, the framing and the week's numbers, shared by both sub-questions."""
    from agent import _identity_header

    return f"{_identity_header('strategic')}\n\n" + _facts_block(data, summary_text)


def _facts_block(data: dict, summary_text: str) -> str:
    """The framing and the week's numbers, with no identity line (pure: no database, no app imports). This is
    everything the model may cite, so it is also what the number check compares against."""
    goals = data["goals"]
    return (
        "This is the user's weekly strategic review. Below is precomputed data for the last "
        "7 full days (today excluded) — these numbers are correct, do not recompute, round "
        "differently, or invent any other numbers.\n\n"
        f"Goal phase: {goals.get('phase', 'unknown')}\n"
        f"Targets: {int(goals.get('step_goal', 0)):,} steps, {int(goals.get('calories_target', 0)):,} kcal, "
        f"{int(goals.get('protein_target', 0))}g protein\n\n"
        f"{summary_text}\n\n"
    )


_TARGET_RULE = (
    "The 'Vs targets' lines already state whether each metric is at/above or below "
    "target — use that status directly, do not recompute or contradict it. Never "
    "recommend increasing a metric whose status is 'target met' or 'target met/exceeded'.\n\n"
)
_NUMBER_RULE = (
    "Use ONLY the numbers given above — do not invent, estimate, or restate rounded numbers "
    "as different values. Plain text only, no markdown."
)


# The review used to be ONE prompt asking a small local model to reason through three stages and write two
# paragraphs at once. A big ask is where an 8B model drops a number or contradicts a target, so it is now two
# focused questions: first the observations, then the recommendations, which are shown those observations. Each
# part gets its own grounding/direction check and its own redraft, and the two paragraphs are joined.
def build_observations_prompt(data: dict, summary_text: str) -> str:
    return (
        _data_block(data, summary_text)
        + "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the OBSERVATIONS only: state the average steps, the average calories and the average protein "
        "(each one that has data) with its number, in plain language, and note any "
        "correlations or hypotheses connecting them (e.g. consistency vs adherence, workout "
        "frequency vs calorie stability, time-of-day activity vs intake timing). "
        "Do NOT give recommendations or advice yet.\n\n"
        + _TARGET_RULE + _NUMBER_RULE
    )


def build_recommendations_prompt(data: dict, summary_text: str, observations: str) -> str:
    return (
        _data_block(data, summary_text)
        + "The observations for this review have already been written:\n"
        f"\"{observations}\"\n\n"
        "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the RECOMMENDATIONS only: 1-3 numbered, specific strategies tied directly to those "
        "observations. Each one must name a concrete habit and when to do it (for example a 20 minute walk "
        "after dinner, or protein at breakfast), not general encouragement. You may suggest how long or how many "
        "for a habit, but any number that describes the user's week must come from the data above. "
        "Do not repeat the observations.\n\n"
        + _TARGET_RULE + _NUMBER_RULE
    )


def _extract_numbers(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUMBER_RE.findall(text)}


# A number in a RECOMMENDATION can be a proposal ("a 20 minute walk", "3 sets", "at least 25g of protein") rather than
# a claim about the user's week. Proposals are advice, not invented facts, so the number check skips them in the
# recommendations. Only time and rep units, or "at least/extra/up to N g", count: a bare "150g" still has to be real.
_PROPOSAL_RE = re.compile(
    r"\d+(?:\.\d+)?\s*-?\s*(?:minutes?|mins?|hours?|hrs?|reps?|sets?|times)\b"
    r"|(?:at least|extra|additional|another|up to)\s+\d+(?:\.\d+)?\s*(?:g|grams?|steps?)\b"
    r"|\b(?:consume|add|eat|have|take|drink|include)\s+(?:an?\s+)?\d+(?:\.\d+)?\s*(?:g|grams?)\b"
    r"|\ba\s+\d+(?:\.\d+)?\s*(?:g|grams?)\s+[a-z]"
    r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.I)


def _grounding_check(draft: str, summary_text: str, tolerance: float = 0.5,
                     allow_proposals: bool = False) -> list[float]:
    """Return numbers cited in `draft` that aren't supported by `summary_text` (within tolerance).

    Numbers with |n| <= 7 are skipped — day counts, list numbering, and hours are too common
    to be meaningful hallucination signals.
    """
    # Compare by MAGNITUDE. The data prints a shortfall as "-1880" and a falling trend as "-140.0", but a correct
    # sentence says "1880 below the target" / "falls about 140 a day"; matching the sign flagged those as invented,
    # which sent nearly every answer back for a pointless redraft (found 2026-10-04 comparing prompt designs).
    supported = {abs(s) for s in _extract_numbers(summary_text)}
    unsupported = []
    for n in _extract_numbers(_PROPOSAL_RE.sub(" ", draft) if allow_proposals else draft):
        n = abs(n)
        if n <= 7:
            continue
        if any(abs(n - s) <= tolerance or (s != 0 and abs(n - s) / abs(s) <= 0.02) for s in supported):
            continue
        unsupported.append(n)
    return unsupported


_INCREASE_RE = r'(?:increas\w*|rais\w*|boost\w*|add\w*\s+more|get\w*\s+more|more|higher|up\s+your)'


def _directional_check(draft: str, data: dict) -> list[str]:
    """Return descriptions of recommendations that contradict the precomputed vs-target status.

    Catches cases like recommending "increase protein to meet your target of 146g" when the
    actual average (e.g. 161g) already meets/exceeds that target.
    """
    issues = []
    goals = data["goals"]
    protein, steps = data["protein"], data["steps"]

    protein_target = goals.get("protein_target", 0)
    if protein["status"] == "ok" and protein_target and protein["avg"] >= protein_target:
        if re.search(rf'\bprotein\b[^.]{{0,60}}{_INCREASE_RE}|{_INCREASE_RE}[^.]{{0,60}}\bprotein\b', draft, re.IGNORECASE):
            issues.append(
                f"protein average ({protein['avg']:.0f}g) already meets/exceeds the target "
                f"({protein_target:.0f}g) — do not recommend increasing it"
            )

    step_goal = goals.get("step_goal", 0)
    if steps["status"] == "ok" and step_goal and steps["avg"] >= step_goal:
        if re.search(rf'\bsteps?\b[^.]{{0,60}}{_INCREASE_RE}[^.]{{0,30}}\b(target|goal)\b|{_INCREASE_RE}[^.]{{0,60}}\bsteps?\b[^.]{{0,30}}\b(target|goal)\b', draft, re.IGNORECASE):
            issues.append(
                f"step average ({steps['avg']:.0f}) already meets/exceeds the goal "
                f"({step_goal:,.0f}) — do not recommend increasing steps to reach the goal"
            )

    return issues


def _answer_with_checks(model, prompt: str, data: dict, summary_text: str, part: str = "observations") -> str:
    """Ask ONE focused question, check the answer against the data, and redraft once if it fails the checks.

    The two parts are checked for different things. The OBSERVATIONS only describe the week, so every number must be
    real, but describing a rising protein trend is not "recommending more protein". The RECOMMENDATIONS are checked
    for contradicting a target ("increase protein" when it is met), and may contain proposals such as "20 minutes"."""
    from agent import _clean_response

    resp = model.invoke([HumanMessage(content=prompt)])
    draft = _clean_response(getattr(resp, "content", "") or "")

    # What the model was shown is the data block (it includes the goal targets, e.g. "2,200 kcal") plus the summary;
    # a number from either is not invented. Checking against the summary alone flagged every mention of a target.
    shown = _facts_block(data, summary_text)
    recommending = part == "recommendations"
    unsupported = _grounding_check(draft, shown, allow_proposals=recommending)
    directional_issues = _directional_check(draft, data) if recommending else []
    if unsupported or directional_issues:
        feedback = []
        if unsupported:
            feedback.append(
                "Your previous draft mentioned these numbers, which do NOT appear in the "
                f"data above: {', '.join(f'{n:g}' for n in unsupported)}."
            )
        if directional_issues:
            feedback.append(
                "Your previous draft also contained recommendations that contradict the "
                "data: " + "; ".join(directional_issues) + "."
            )
        redraft_prompt = (
            f"{prompt}\n\n"
            + "\n".join(feedback) + "\n"
            "Redraft your response fixing these issues. Use ONLY the numbers given in the "
            "data above, and respect the 'Vs targets' status for each metric. Same format: "
            "one short paragraph of about 4 lines, no headings."
        )
        resp = model.invoke([HumanMessage(content=redraft_prompt)])
        draft = _clean_response(getattr(resp, "content", "") or "")
    return draft


# ── label mode: the model reads words, not numbers ───────────────────────────
# Same two questions, but weekly_labels.py has already compared every average with its target and turned the result
# (and the trends, the consistency, even the averages) into plain-language labels. The model never sees a figure to
# compare or misquote; the checks below then verify the answer against the same labels.

_LABEL_RULE = (
    "The descriptions above were worked out for you: use them as they are and do not contradict them. "
    "Write in words, never with digits. Never say a metric listed under 'On track' needs improving. "
    "Say nothing at all about a metric that has no data. Plain text only, no markdown."
)
_CALORIES_MORE_RE = re.compile(r"\b(?:increas\w*|rais\w*|boost\w*|more|higher|add\w*)\b[^.]{0,40}\b(?:calories|kcal|calorie intake)\b", re.I)
_CALORIES_LESS_RE = re.compile(r"\b(?:reduc\w*|cut\w*|lower\w*|less|decreas\w*|trim\w*|fewer)\b[^.]{0,40}\b(?:calories|kcal|calorie intake)\b", re.I)


def _label_facts_block(data: dict) -> str:
    """The framing and the week described in words (pure: no database, no app imports). It has no digits, so any
    number the model writes is, by definition, not from the data."""
    import weekly_labels
    return (
        "This is the user's weekly strategic review. Below is a description of the last seven full days. Every "
        "comparison with the targets has already been worked out for you, so there are no numbers on purpose.\n\n"
        + weekly_labels.label_summary(data) + "\n\n"
    )


def _label_data_block(data: dict) -> str:
    from agent import _identity_header
    return f"{_identity_header('strategic')}\n\n" + _label_facts_block(data)


def build_label_observations_prompt(data: dict) -> str:
    return (
        _label_data_block(data)
        + "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the OBSERVATIONS only: say how steps, calories and protein went (each one that has data) using the "
        "descriptions above, and note any link between them (for example consistency vs adherence, workouts vs "
        "calorie steadiness, time of day vs eating). Do NOT give recommendations or advice yet.\n\n"
        + _LABEL_RULE
    )


def build_label_recommendations_prompt(data: dict, observations: str) -> str:
    import weekly_labels
    needs = weekly_labels.off_target(weekly_labels.classify(data))
    must = (f"Every metric under 'Needs attention' ({', '.join(needs)}) must be addressed by at least one of them. "
            if needs else "Nothing needs attention, so the recommendations should protect what is working. ")
    return (
        _label_data_block(data)
        + "The observations for this review have already been written:\n"
        f"\"{observations}\"\n\n"
        "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the RECOMMENDATIONS only: 1-3 numbered, specific strategies tied directly to those observations. "
        + must
        + "Each one must name a concrete habit and the time of day it happens, not general encouragement. "
        "You may say how long a habit lasts. Do not repeat the observations.\n\n"
        + _LABEL_RULE
    )


_NEGATION_RE = re.compile(r"\b(?:without|avoid\w*|not|never|no|don'?t|do not|instead of)\b[^.]{0,25}$", re.I)


def _asks(pattern: re.Pattern, draft: str) -> bool:
    """True when the draft recommends what `pattern` describes. "Without adding extra calories" does not."""
    return any(not _NEGATION_RE.search(draft[:m.start()]) for m in pattern.finditer(draft))


def _label_directional_check(draft: str, data: dict) -> list[str]:
    """Contradictions with the verdicts: more of something already met, more calories when over, fewer when under,
    or any mention of a metric with no data."""
    import weekly_labels
    lab = weekly_labels.classify(data)
    issues = []
    for key, name in (("steps", "steps"), ("protein", "protein")):
        if lab[key]["status"] == "met":
            if re.search(rf'\b{name}\b[^.]{{0,60}}{_INCREASE_RE}|{_INCREASE_RE}[^.]{{0,60}}\b{name}\b', draft, re.I):
                issues.append(f"{name} is already {lab[key]['level']} — do not recommend increasing it")
    if lab["calories"]["status"] == "over" and _asks(_CALORIES_MORE_RE, draft):
        issues.append(f"calories are {lab['calories']['level']} — do not recommend eating more")
    if lab["calories"]["status"] == "under" and _asks(_CALORIES_LESS_RE, draft):
        issues.append(f"calories are {lab['calories']['level']} — do not recommend eating less")
    for key in weekly_labels.no_data(lab):
        if re.search(rf'\b{key}\b', draft, re.I):
            issues.append(f"there is no {key} data this week — do not mention {key}")
    return issues


def _answer_with_label_checks(model, prompt: str, data: dict, part: str, version: int = 1) -> str:
    """One focused question in label mode; the answer is checked against the labels and redrafted once if needed.
    Observations must be free of digits; recommendations may propose a duration but not state a figure about the week;
    recommendations must also cover every metric that needs attention. Version 2 also counts "no workouts" as
    something that needs attention (the observations must mention it and the recommendations must address it)."""
    import weekly_labels
    import weekly_labels_v2
    from agent import _clean_response

    resp = model.invoke([HumanMessage(content=prompt)])
    draft = _clean_response(getattr(resp, "content", "") or "")

    import weekly_labels_v3
    import weekly_labels_v4

    recommending = part == "recommendations"
    problems = []
    shown = {2: _label2_facts_block, 3: _label3_facts_block, 4: _label4_facts_block,
             5: _label4_facts_block}.get(version, _label_facts_block)(data)
    labels_module = {2: weekly_labels_v2, 3: weekly_labels_v3, 4: weekly_labels_v4, 5: weekly_labels_v4}.get(version)
    unsupported = _grounding_check(draft, shown, allow_proposals=recommending)
    if unsupported:
        problems.append("it contains numbers that are not in the description: " + ", ".join(f"{n:g}" for n in unsupported))
    if not recommending and re.search(r"\d", draft):
        problems.append("it uses digits; describe everything in words")
    problems += _label_directional_check(draft, data)
    if recommending:
        if labels_module:
            needs = labels_module.attention(labels_module.classify(data))
            missed = [k for k in needs if not labels_module.KEYWORDS[k].search(draft)]
        else:
            lab = weekly_labels.classify(data)
            needs = weekly_labels.off_target(lab)
            missed = [k for k in needs if not weekly_labels.KEYWORDS[k].search(draft)]
        if missed:
            problems.append("it does not address " + ", ".join(missed) + ", which needs attention")
        # v3+ rank the problems by what they do to the goal, so the FIRST recommendation must tackle the first one.
        first = _first_numbered_item(draft)
        if version >= 3 and needs and first and not labels_module.KEYWORDS[needs[0]].search(first):
            problems.append(f"recommendation one must address {needs[0]}, the first priority")
        if version >= 3:
            for n in _keep_only_items(draft):
                problems.append(f"recommendation {n} only says to keep something as it is; drop it or replace it "
                                "with a concrete habit")
        if version == 5:
            want = len(recommendation_plan(data))
            got = len(re.findall(r"(?:^|\s)\d[.)]\s+", draft))
            if got != want:
                problems.append(f"it has {got} numbered recommendations; write exactly {_COUNT_WORDS[want]}, "
                                "one for each line of the plan, in order")
    else:
        if labels_module and labels_module.classify(data)["no_workouts"] \
                and not labels_module.KEYWORDS["workouts"].search(draft):
            problems.append("it does not mention that there were no workouts this week")
        if version == 3 and _OBS_ADVICE_V3_RE.search(draft):        # v3's exact message, kept so v3 stays v3
            problems.append("it tells the user what to do; the observations only describe the week")
        if version in (4, 5) and _OBS_ADVICE_V4_RE.search(draft):
            problems.append(f"it tells the user what to do ('{_OBS_ADVICE_V4_RE.search(draft).group(0)}'); the "
                            "observations only describe the week, they give no advice")
    if problems:
        resp = model.invoke([HumanMessage(content=(
            f"{prompt}\n\nYour previous draft had these problems: " + "; ".join(problems) + ".\n"
            "Redraft it fixing them, using only the descriptions above. Same format: one short paragraph of about "
            "4 lines, no headings."))])
        draft = _clean_response(getattr(resp, "content", "") or "")
    return draft


# Advice wording that must not appear in the OBSERVATIONS. v3 caught only the bare imperatives; v3's observations still
# said "focus on" / "make sure" (measured 2026-10-04), so v4 uses the full list of advice wording.
_OBS_ADVICE_V3_RE = re.compile(r"\b(?:should|need to|needs to|must)\b", re.I)
_OBS_ADVICE_V4_RE = re.compile(r"\b(?:should|need to|needs to|must|focus on|make sure|try to|aim to|consider|"
                               r"recommend\w*|suggest that|you could|i suggest)\b", re.I)


def _first_numbered_item(text: str) -> str:
    """The text of recommendation 1 ("1. foo 2. bar" -> "foo"), or "" when the answer is not numbered."""
    m = re.search(r"(?:^|\s)1[.)]\s+(.*?)(?=\s2[.)]\s|$)", text, re.S)
    return m.group(1).strip() if m else ""


def _run_label_analysis(model, data: dict, version: int = 1) -> str:
    if version == 6:                      # v5's checks (guard version 5); only the plan's wording differs
        observations = _answer_with_label_checks(model, build_label4_observations_prompt(data), data, "observations", 5)
        recommendations = _answer_with_label_checks(
            model, build_label6_recommendations_prompt(data, observations), data, "recommendations", 5)
        return "\n\n".join(part for part in (observations, recommendations) if part)
    if version == 5:
        observations = _answer_with_label_checks(model, build_label4_observations_prompt(data), data, "observations", 5)
        recommendations = _answer_with_label_checks(
            model, build_label5_recommendations_prompt(data, observations), data, "recommendations", 5)
        return "\n\n".join(part for part in (observations, recommendations) if part)
    if version == 4:
        observations = _answer_with_label_checks(model, build_label4_observations_prompt(data), data, "observations", 4)
        recommendations = _answer_with_label_checks(
            model, build_label4_recommendations_prompt(data, observations), data, "recommendations", 4)
        return "\n\n".join(part for part in (observations, recommendations) if part)
    if version == 3:
        observations = _answer_with_label_checks(model, build_label3_observations_prompt(data), data, "observations", 3)
        recommendations = _answer_with_label_checks(
            model, build_label3_recommendations_prompt(data, observations), data, "recommendations", 3)
        return "\n\n".join(part for part in (observations, recommendations) if part)
    if version == 2:
        observations = _answer_with_label_checks(model, build_label2_observations_prompt(data), data, "observations", 2)
        recommendations = _answer_with_label_checks(
            model, build_label2_recommendations_prompt(data, observations), data, "recommendations", 2)
    else:
        observations = _answer_with_label_checks(model, build_label_observations_prompt(data), data, "observations")
        recommendations = _answer_with_label_checks(
            model, build_label_recommendations_prompt(data, observations), data, "recommendations")
    return "\n\n".join(part for part in (observations, recommendations) if part)


# ── label mode, second version (weekly_labels_v2.py) ─────────────────────────
# v1 stays selectable as mode "labels"; this is mode "labels2". It adds to the description the size of each gap, the
# direction to move, a "keep as they are" list, an advice goal and the workout verdict, and asks the model to vary its
# wording instead of copying the descriptions.

_LABEL2_RULE = (
    "The descriptions above were worked out for you: use them as they are and do not contradict them. "
    "Write in words, never with digits, and put them in your own words rather than copying the descriptions "
    "phrase for phrase. Everything under 'Keep as they are' stays as it is: never ask for more of it. "
    "Say nothing at all about a metric that has no data. Plain text only, no markdown."
)


def _label2_facts_block(data: dict) -> str:
    import weekly_labels_v2
    return (
        "This is the user's weekly strategic review. Below is a description of the last seven full days. Every "
        "comparison with the targets has already been worked out for you, so there are no numbers on purpose.\n\n"
        + weekly_labels_v2.label_summary(data) + "\n\n"
    )


def _label2_data_block(data: dict) -> str:
    from agent import _identity_header
    return f"{_identity_header('strategic')}\n\n" + _label2_facts_block(data)


def build_label2_observations_prompt(data: dict) -> str:
    return (
        _label2_data_block(data)
        + "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the OBSERVATIONS only: say how steps, calories and protein went (each one that has data) and what "
        "needs attention, in order of size, and mention the workouts. Note one link between them if there is a real "
        "one. Do NOT give recommendations or advice yet.\n\n"
        + _LABEL2_RULE
    )


def build_label2_recommendations_prompt(data: dict, observations: str) -> str:
    import weekly_labels_v2
    needs = weekly_labels_v2.attention(weekly_labels_v2.classify(data))
    must = (f"Every item under 'Needs attention' ({', '.join(needs)}) must be addressed by at least one of them, "
            "moving in the 'Direction for any advice'. " if needs
            else "Nothing needs attention, so the recommendations should protect what is working. ")
    return (
        _label2_data_block(data)
        + "The observations for this review have already been written:\n"
        f"\"{observations}\"\n\n"
        "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the RECOMMENDATIONS only: 1-3 numbered, specific strategies that follow the 'Advice goal'. "
        + must
        + "Each one must name a concrete habit and the time of day it happens, not general encouragement. "
        "You may say how long a habit lasts. Do not repeat the observations and do not invent extra problems.\n\n"
        + _LABEL2_RULE
    )


# ── label mode, third version (weekly_labels_v3.py) ──────────────────────────
# Mode "labels3". The description now carries MEANING: what each gap does to this phase's goal, whether each trend
# moves toward or away from the target, which links between metrics are real, priorities weighed by the phase, the kind
# of fix that fits, and what to keep an eye on. The model's job shrinks to explaining that in plain words.

# v2's "keep as they are" list made the model spend a recommendation slot on "keep calories exactly as they are" in
# 10 of 20 answers (measured 2026-10-04). In v3 the list only says what NOT to change, and a keep-only item is redrafted.
_KEEP_ONLY_RE = re.compile(r"^\W*(?:keep|maintain|continue|stick|hold)\b(?!.*\b(?:walk|snack|meal|shake|portion|session|"
                           r"breakfast|lunch|dinner|minutes?)\b)", re.I)


def _keep_only_items(recs: str) -> list[int]:
    """Numbers of the recommendations that only say to keep something as it is, with no concrete habit."""
    parts = re.split(r"(?:^|\s)(\d)[.)]\s+", recs)
    return [int(parts[i]) for i in range(1, len(parts) - 1, 2) if _KEEP_ONLY_RE.search(parts[i + 1])]


_LABEL3_RULE = (
    "Everything above was worked out for you from the user's data: use it as it is and do not contradict it. "
    "Write in words, never with digits, and in your own words rather than copying the descriptions. "
    "Claim no link between the metrics other than the ones listed under 'Links'. Everything under 'Keep as they are' "
    "stays as it is: never ask for more of it. Say nothing at all about a metric that has no data. "
    "Plain text only, no markdown."
)


def _label3_facts_block(data: dict) -> str:
    import weekly_labels_v3
    return (
        "This is the user's weekly strategic review. Below is a description of the last seven full days and what it "
        "means for their goal. Every comparison with the targets has already been worked out, so there are no numbers "
        "on purpose.\n\n" + weekly_labels_v3.label_summary(data) + "\n\n"
    )


def _label3_data_block(data: dict) -> str:
    from agent import _identity_header
    return f"{_identity_header('strategic')}\n\n" + _label3_facts_block(data)


def build_label3_observations_prompt(data: dict) -> str:
    return (
        _label3_data_block(data)
        + "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the OBSERVATIONS only. Open with the headline, then say how steps, calories and protein went (each one "
        "that has data) and, for anything that needs attention, what it means for the goal. Mention the workouts and "
        "anything to keep an eye on. Do NOT give recommendations or advice yet: describe, never tell the user what "
        "they should do or need to do.\n\n"
        + _LABEL3_RULE
    )


def build_label3_recommendations_prompt(data: dict, observations: str) -> str:
    import weekly_labels_v3
    needs = weekly_labels_v3.attention(weekly_labels_v3.classify(data))
    if needs:
        order = ", then ".join(needs[:3])
        must = (f"Follow the priorities in order ({order}): recommendation one addresses the first priority, and each "
                "one uses the 'Kind of fix' given for it. ")
    else:
        must = ("There are no priorities, so give one or two recommendations that protect the routine, and if there is "
                "anything to keep an eye on, one that guards against it. ")
    return (
        _label3_data_block(data)
        + "The observations for this review have already been written:\n"
        f"\"{observations}\"\n\n"
        "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the RECOMMENDATIONS only: 1-3 numbered, specific strategies (one or two are enough when there is only "
        "one priority). "
        + must
        + "Each one must name a concrete habit and the time of day it happens, not general encouragement. "
        "The 'Keep as they are' list only tells you what not to change: never spend a recommendation on keeping "
        "something as it is. You may say how long a habit lasts. Do not repeat the observations and do not invent "
        "extra problems.\n\n"
        + _LABEL3_RULE
    )


# ── label mode, fourth version (weekly_labels_v4.py) ─────────────────────────
# Mode "labels4". The same verdicts and meanings as v3, but each question sees only its own view: the observations
# question gets what happened and what it means (meaning inline per metric, nothing that asks for action); the
# recommendations question gets what to work on, in order, with the kind of fix, and what to leave unchanged.

_LABEL4_RULE = (
    "The description above was worked out for you from the user's data: use it as it is and do not contradict it. "
    "Write in words, never with digits, and in your own words rather than copying the description. "
    "Say nothing at all about a metric that has no data. Plain text only, no markdown."
)


def _label4_facts_block(data: dict) -> str:
    """Both views (what the two questions are shown, together): used by the number check."""
    import weekly_labels_v4
    return weekly_labels_v4.label_summary(data) + "\n\n"


def _label4_intro(data: dict, view: str) -> str:
    from agent import _identity_header
    return (
        f"{_identity_header('strategic')}\n\n"
        "This is the user's weekly strategic review. Below is a description of the last seven full days and what it "
        "means for their goal. Every comparison with the targets has already been worked out, so there are no numbers "
        "on purpose.\n\n" + view + "\n\n"
    )


def build_label4_observations_prompt(data: dict) -> str:
    import weekly_labels_v4
    return (
        _label4_intro(data, weekly_labels_v4.observations_view(data))
        + "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the OBSERVATIONS only. Start from the main point, then say how steps, calories and protein went (each "
        "one that has data), keeping each meaning with its own metric, and mention the workouts and anything worth a "
        "mention. Mention a connection between metrics only if it is listed under 'Links'. This paragraph describes; "
        "it gives no advice at all, so never write should, need to, focus on, make sure or keep.\n\n"
        + _LABEL4_RULE
    )


def build_label4_recommendations_prompt(data: dict, observations: str) -> str:
    import weekly_labels_v4
    needs = weekly_labels_v4.attention(weekly_labels_v4.classify(data))
    if needs:
        order = ", then ".join(needs[:3])
        must = (f"Work through 'What to work on' in order ({order}): recommendation one addresses the first item, and "
                "each recommendation uses the 'Kind of fix' given for its item. ")
    else:
        must = ("Nothing is off target, so give one or two recommendations that protect the routine, and if there is "
                "anything worth guarding against, one that guards against it. ")
    return (
        _label4_intro(data, weekly_labels_v4.recommendations_view(data))
        + "The observations for this review have already been written:\n"
        f"\"{observations}\"\n\n"
        "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        "It is the RECOMMENDATIONS only: 1-3 numbered, specific strategies (one or two are enough when there is only "
        "one item to work on). "
        + must
        + "Each one must name a concrete habit and the time of day it happens, not general encouragement. Make no "
        "recommendation about anything under 'Leave unchanged'. You may say how long a habit lasts. Do not repeat "
        "the observations and do not invent extra problems.\n\n"
        + _LABEL4_RULE
    )


# ── label mode, sixth version: v5's plan, stated as one sentence ─────────────────────────────────────────────────────
# Mode "labels6". v5 listed the plan as "Recommendation one: a habit for calories." and the model copied that label
# format ("Recommendation one: Make a habit of ...") instead of numbering, so every answer failed the count check and
# the numbered-format item (measured 2026-10-04). v6 states the same plan in one sentence and names the numbering
# explicitly; checks are v5's (version 5 in _answer_with_label_checks), so only the wording of the plan differs.

_ORDINAL_WORDS = ["first", "second", "third"]


def build_label6_recommendations_prompt(data: dict, observations: str) -> str:
    import weekly_labels_v4
    plan = recommendation_plan(data)
    n = len(plan)
    numbering = " and ".join(f"{i + 1}." for i in range(n)) if n == 2 else ", ".join(f"{i + 1}." for i in range(n))
    roles = "; ".join(f"the {_ORDINAL_WORDS[i]} is {p}" for i, p in enumerate(plan))
    return (
        _label4_intro(data, weekly_labels_v4.recommendations_view(data))
        + "The observations for this review have already been written:\n"
        f"\"{observations}\"\n\n"
        "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        f"It is the RECOMMENDATIONS only: exactly {_COUNT_WORDS[n]} recommendations, numbered {numbering} "
        f"Plan them like this: {roles}. "
        "Each one uses the 'Kind of fix' given for its item, names a concrete habit and the time of day it happens, "
        "and is not general encouragement. Never spend a recommendation on keeping something as it is, and make no "
        "recommendation about anything under 'Leave unchanged'. You may say how long a habit lasts. Do not repeat "
        "the observations and do not invent extra problems.\n\n"
        + _LABEL4_RULE
    )


# ── label mode, fifth version: v4 + the NUMBER of recommendations decided by code ──────────────────────────────────
# Mode "labels5". Labels and views are v4's (weekly_labels_v4.py); the one change is that code, not the model, decides
# how many recommendations to write and what each one is for. With v3/v4 the model padded to three slots with "keep X
# as it is" when there was a single problem (v3 tuning answers 4/20; a v4 holdout check 4/10).

def recommendation_plan(data: dict) -> list[str]:
    """What each recommendation is for, in order. One per item to work on (at most three), and never fewer than two:
    a single item gets two different habits; nothing to work on gets protect-the-routine (plus guard-against)."""
    import weekly_labels_v4
    lab = weekly_labels_v4.classify(data)
    needs = weekly_labels_v4.attention(lab)[:3]
    if len(needs) >= 2:
        return [f"a habit for {k}" for k in needs]
    if len(needs) == 1:
        return [f"a habit for {needs[0]}", f"a second, different habit for {needs[0]}"]
    early = [k for k, _why in weekly_labels_v4.watch(lab)]
    if early:
        return ["a habit that protects the routine that is working", f"a habit that guards against {early[0]} drifting"]
    return ["a habit that protects the routine that is working", "a second habit that protects the routine"]


_COUNT_WORDS = ["zero", "one", "two", "three"]


def build_label5_recommendations_prompt(data: dict, observations: str) -> str:
    import weekly_labels_v4
    plan = recommendation_plan(data)
    n = len(plan)
    lines = "\n".join(f"  Recommendation {_COUNT_WORDS[i + 1]}: {p}." for i, p in enumerate(plan))
    return (
        _label4_intro(data, weekly_labels_v4.recommendations_view(data))
        + "The observations for this review have already been written:\n"
        f"\"{observations}\"\n\n"
        "Write ONE short paragraph of about 4 lines (no label, no heading, no greeting, no sign-off). "
        f"It is the RECOMMENDATIONS only: exactly {_COUNT_WORDS[n]} numbered recommendations, in this order:\n"
        f"{lines}\n"
        "Each one uses the 'Kind of fix' given for its item, names a concrete habit and the time of day it happens, "
        "and is not general encouragement. Never spend a recommendation on keeping something as it is, and make no "
        "recommendation about anything under 'Leave unchanged'. You may say how long a habit lasts. Do not repeat "
        "the observations and do not invent extra problems.\n\n"
        + _LABEL4_RULE
    )


def run_cot_analysis(data: dict, summary_text: str, mode: str | None = None) -> str:
    """The weekly review as two paragraphs, built from two separate questions to the local model (observations,
    then recommendations that are shown the observations) instead of one big one. Costs a second call; the
    smaller asks are less likely to drop a number or contradict a target. Each part has its own checks.

    mode "numbers" (default) shows the model the figures; the label modes show it only words derived from them:
    "labels" (weekly_labels.py: where each metric sits), "labels2" (weekly_labels_v2.py: plus gap size, direction,
    keep-as-is, advice goal, workouts), "labels3" (weekly_labels_v3.py: plus what each gap means for the phase's
    goal, trends read against the target, the only real links, phase-weighted priorities, the kind of fix, and what
    to keep an eye on), "labels4" (weekly_labels_v4.py: v3's meaning, but each question sees only its own view) and
    "labels5" (v4, plus code decides how many recommendations to write and what each is for: recommendation_plan) and
    "labels6" (v5's plan stated as one sentence with explicit numbering; v5's list format was copied by the model).
    Every mode stays selectable so they can be compared (evals/ledger.md, evals/REPORT.md). Set
    WEEKLY_REVIEW_MODE in .env to choose when no mode is passed."""
    import os
    from agent import log
    import llm_registry

    mode = (mode or os.getenv("WEEKLY_REVIEW_MODE", "numbers")).strip().lower()
    if mode in ("labels", "labels2", "labels3", "labels4", "labels5", "labels6"):
        try:
            return _run_label_analysis(llm_registry.get("weekly_review"), data,
                                       {"labels2": 2, "labels3": 3, "labels4": 4, "labels5": 5, "labels6": 6}.get(mode, 1))
        except Exception as e:
            log.warning(f"[weekly_analysis] label analysis failed: {e}")
            return "Analysis unavailable right now (AI service error) — but the numbers above are accurate."

    try:
        # Its own role (default: fast local mode), not "strategic": on small focused questions thinking mode was no
        # more accurate, 3-100x slower, and once ran for 20 minutes (measured 2026-10-04, see llm_registry).
        model = llm_registry.get("weekly_review")
        observations = _answer_with_checks(model, build_observations_prompt(data, summary_text), data, summary_text)
        recommendations = _answer_with_checks(
            model, build_recommendations_prompt(data, summary_text, observations), data, summary_text,
            part="recommendations")
        return "\n\n".join(part for part in (observations, recommendations) if part)
    except Exception as e:
        log.warning(f"[weekly_analysis] CoT analysis failed: {e}")
        return "Analysis unavailable right now (AI service error) — but the numbers above are accurate."


def generate_report_charts() -> list[Path]:
    """(Re)generate the weekly trend charts and return their paths."""
    import sys
    sys.path.insert(0, str(ROOT / "reports"))
    import weekly_trends_report as wtr
    from db_store import Store

    store = Store()
    wtr.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    wtr._build_metrics_report(store)
    wtr._build_deltas_report(store)
    wtr._build_hourly_steps_report()
    return [wtr.OUTPUT_PATH_METRICS, wtr.OUTPUT_PATH_DELTAS, wtr.OUTPUT_PATH_HOURLY_STEPS]


# The mode the APP's /analysis uses. Chosen 2026-10-04 from the evaluation in evals/REPORT.md: labels v6 passed every
# core check in 87% of holdout answers (numbers: 3%) and every semantic check in 100% (numbers: 17%), at 5.2 s.
# run_cot_analysis's own default stays "numbers" for callers that pass no mode; WEEKLY_REVIEW_MODE in .env overrides
# this for the app (e.g. WEEKLY_REVIEW_MODE=numbers to go back).
APP_REVIEW_MODE = "labels6"


def app_review_mode() -> str:
    import os
    return (os.getenv("WEEKLY_REVIEW_MODE") or APP_REVIEW_MODE).strip().lower()


def run_weekly_analysis() -> tuple[str, str, list[Path]]:
    """Full /analysis pipeline. Returns (data_summary_text, cot_analysis_text, chart_paths)."""
    data = gather_weekly_analysis_data()
    summary_text = format_data_summary(data)
    cot_text = run_cot_analysis(data, summary_text, mode=app_review_mode())
    chart_paths = generate_report_charts()
    return summary_text, cot_text, chart_paths


def run_analysis_followup(user_message: str, history: list[dict], analysis_context: str) -> str:
    """Grounded follow-up Q&A over the cached weekly analysis (last 12 messages of history)."""
    from agent import _identity_header, _clean_response, log
    import llm_registry

    system_prompt = (
        f"{_identity_header('strategic')}\n\n"
        "You are continuing a conversation about the user's weekly strategic analysis below. "
        "Answer follow-up questions grounded in this data and your earlier analysis — do not "
        "invent new numbers. Plain text only, no markdown, be concise.\n\n"
        f"{analysis_context}"
    )
    messages = [SystemMessage(content=system_prompt)]
    for turn in history[-12:]:
        if turn.get("role") == "assistant":
            messages.append(AIMessage(content=turn.get("content", "")))
        else:
            messages.append(HumanMessage(content=turn.get("content", "")))
    messages.append(HumanMessage(content=user_message))

    try:
        model = llm_registry.get("strategic")
        resp = model.invoke(messages)
        return _clean_response(getattr(resp, "content", "") or "")
    except Exception as e:
        log.warning(f"[weekly_analysis] follow-up failed: {e}")
        return "Something went wrong — please try again."


if __name__ == "__main__":
    _data = gather_weekly_analysis_data()
    _summary = format_data_summary(_data)
    print(_summary)
    print("\n---\n")
    print(run_cot_analysis(_data, _summary))
    print("\n---\n")
    for _p in generate_report_charts():
        print(f"chart: {_p}")
