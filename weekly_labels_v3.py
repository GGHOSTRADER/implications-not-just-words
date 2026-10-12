"""Label approach, third version: labels that carry MEANING, not only position.

v1 (weekly_labels.py) said where each metric sat ("far over the target"). v2 (weekly_labels_v2.py) added the size of
the gap, the direction to move, a keep-as-is list and an advice goal. Both are untouched and still selectable. Reading
v1/v2 answers showed what the small model still had to guess, and guessed badly:

  - what a gap DOES to this user's goal: calories far over in a fat-loss phase erase the deficit; the same gap in a
    gain phase only adds some fat. Without that the model cannot tell an important problem from a cosmetic one;
  - whether a trend is GOOD or BAD: "falling" is good for calories that are over and bad for protein that is short.
    v1/v2 only said "falling";
  - which LINKS between the metrics are real. The model invented them ("early evening activity aligns with calorie
    intake"). Here code derives the few links that follow from the verdicts, and the model is told to use only those;
  - what MATTERS MOST: priority now weighs the gap by the phase (a calorie overshoot in a cut outranks a step shortfall
    in a bulk), not by size alone;
  - what KIND of fix fits: a food fix, a movement fix or a training fix, and for short protein whether it is a swap
    (enough food, too little of it protein) or more food (everything short);
  - whether a metric that is met is about to SLIP: "still meeting the goal, but at this pace it slips below next week".

It also drops v2's "days within ten kcal" line (the model read "never" as "under target").

Same contract as v1/v2: pure functions, no model, no database, and the text it returns has no digits.
"""
from __future__ import annotations

import weekly_labels as v1
import weekly_labels_v2 as v2

KEYWORDS = v2.KEYWORDS
_GOAL_KEY = {"steps": "step_goal", "protein": "protein_target", "calories": "calories_target"}
_TITLES = {"steps": "Steps", "calories": "Calories", "protein": "Protein"}
_FLAT = 0.003                                   # same "holding steady" cutoff as v1 (under ~2% change over a week)
_ORDINALS = ["First", "Then", "Last"]

# How much each kind of gap matters in each phase. Priority = size of the gap x this weight.
_WEIGHT = {
    "Fat Loss":    {"calories": 1.5, "protein": 1.2, "steps": 1.0},
    "Muscle Gain": {"calories": 1.5, "protein": 1.3, "steps": 0.6},
    "Maintenance": {"calories": 1.2, "protein": 1.0, "steps": 1.0},
}
# No workouts at all, as a priority score on the same scale (sizes are fractions of the goal, ~0.05 to 0.5).
_NO_WORKOUT_SCORE = {"Fat Loss": 0.2, "Muscle Gain": 0.6, "Maintenance": 0.12}


def _phase(lab: dict) -> str:
    return lab["phase"] if lab["phase"] in _WEIGHT else "Maintenance"


def _severity(size: float) -> str:
    if size < 0.05:
        return "small"
    if size < 0.15:
        return "moderate"
    if size < 0.3:
        return "large"
    return "very large"


def _meaning(phase: str, key: str, status: str, size: float) -> str:
    """What this gap does to this user's goal, in one clause."""
    big = size >= 0.15
    if key == "calories":
        if phase == "Fat Loss":
            if status == "over":
                return "this eats into the planned deficit, so fat loss slows" + (" or stalls" if big else "")
            return ("a deficit this deep risks losing muscle and running out of energy" if big
                    else "a deeper deficit than planned, which is hard to keep up")
        if phase == "Muscle Gain":
            if status == "under":
                return "without the planned surplus there is little to build muscle with" if big else "a smaller surplus than planned, so muscle gain slows"
            return "a surplus this large adds far more fat than muscle" if big else "a bigger surplus than planned, so more of the gain is fat"
        return "weight will likely creep up" if status == "over" else "weight will likely drift down"
    if key == "protein":
        return {"Fat Loss": "too little protein while dieting puts muscle at risk",
                "Muscle Gain": "too little protein limits recovery and growth"}.get(phase, "a little more protein would help recovery")
    # steps
    return {"Fat Loss": "less movement means fewer calories burned and a smaller deficit",
            "Muscle Gain": "less daily movement, which in a gain phase matters for health and appetite more than for the goal"}.get(
        phase, "less movement makes it harder to hold weight steady")


def _trend_meaning(key: str, status: str, avg: float, slope: float, goal: float) -> str:
    """The trend read against the target: is the week moving the right way?"""
    r = slope / avg if avg else 0.0
    target_word = "the target" if key == "calories" else "the goal"
    if status in ("below", "over", "under"):
        if abs(r) < _FLAT:
            return f"not moving toward {target_word} during the week"
        toward = (status in ("below", "under") and r > 0) or (status == "over" and r < 0)
        return (f"moving back toward {target_word} as the week went on" if toward
                else f"drifting further from {target_word} as the week went on")
    projected = avg * (1 + r * 7)                    # where a straight line puts next week's average
    if key == "calories":
        rel = (projected - goal) / goal
        if rel > v1.CALORIE_BAND:
            return "on target now, but rising fast enough to go over next week"
        if rel < -v1.CALORIE_BAND:
            return "on target now, but falling fast enough to go under next week"
        return "holding steady" if abs(r) < _FLAT else "moving a little but staying in range"
    if abs(r) < _FLAT:
        return "holding steady"
    if r > 0:
        return "still rising"
    return ("still meeting the goal, but at this pace it slips below it next week" if projected < goal
            else "easing a little, with room to spare")


def _consistency_meaning(key: str, cv: float, status: str) -> str:
    """Only said when the day-to-day swings change what the average means."""
    if cv < 20:
        return ""
    if key == "calories":
        return ("on target only on average: some days much heavier and some much lighter" if status == "on"
                else "big swings between heavier and lighter days")
    if key == "steps":
        return "the average hides some much quieter days"
    return "some days well short of the rest"


def _fix(key: str, lab: dict) -> str:
    """The kind of change that fits, without inventing a specific habit."""
    if key == "workouts":
        return ("training: put strength sessions in the calendar, the surplus needs them" if _phase(lab) == "Muscle Gain"
                else "training: put sessions in the calendar")
    status = lab[key]["status"]
    if key == "calories":
        return ("food: trim portions or cut a snack, rather than adding exercise" if status == "over"
                else "food: add a snack or a bigger portion at a regular meal")
    if key == "protein":
        if lab["calories"]["status"] == "under":
            return "food: add protein-rich food, which also raises the calories"
        return "food: swap part of a meal for a protein source, not more food overall"
    # steps
    if lab["calories"]["status"] == "under":
        return ("movement: only once food is back up to target, since more movement now would deepen a deficit "
                "that is already too deep")
    when = f", for example in {lab['peak']}, when the user already moves most" if lab.get("peak") else ""
    return f"movement: add walking{when}"


# Finer than v2's bands (v2 called a 15% shortfall "about a fifth").
_GAP_WORDS = [(0.05, "only a few percent"), (0.09, "a bit under a tenth"), (0.13, "about a tenth"),
              (0.18, "about a sixth"), (0.23, "about a fifth"), (0.29, "about a quarter"), (0.4, "about a third"),
              (0.6, "about half"), (9.9, "more than half")]


def _gap_words(size: float) -> str:
    return next(words for limit, words in _GAP_WORDS if size < limit)


def classify(data: dict) -> dict:
    """v2's verdicts plus meaning: phase impact, trend vs target, consistency meaning, fix, priority score."""
    lab = v2.classify(data)
    phase = _phase(lab)
    for key in ("steps", "protein", "calories"):
        m = lab[key]
        if m["status"] == "nodata":
            continue
        raw, goal = data[key], data["goals"].get(_GOAL_KEY[key], 0)
        off = m["status"] in ("below", "over", "under")
        m["severity"] = _severity(m["size"]) if off else ""
        m["gap"] = _gap_words(m["size"]) if off else ""
        m["meaning"] = _meaning(phase, key, m["status"], m["size"]) if off else ""
        m["trend_meaning"] = _trend_meaning(key, m["status"], raw["avg"], raw["slope"], goal)
        m["consistency_meaning"] = _consistency_meaning(key, raw["coefficient_of_variation_pct"], m["status"])
        m["priority"] = m["size"] * _WEIGHT[phase][key] if off else 0.0
    # Under-eating outranks under-moving: more steps on too little food only deepens the deficit.
    if lab["steps"]["status"] == "below" and lab["calories"]["status"] == "under":
        lab["steps"]["priority"] *= 0.3
    n_workouts = len(data.get("workouts") or [])
    lab["n_workouts"] = n_workouts
    lab["workout_meaning"] = _workout_meaning(phase, n_workouts)
    lab["workout_priority"] = _NO_WORKOUT_SCORE[phase] if n_workouts == 0 else 0.0
    return lab


def _workout_meaning(phase: str, n: int) -> str:
    if n == 0:
        return {"Muscle Gain": "no training at all, so the extra food has nothing to build muscle with",
                "Fat Loss": "no training at all, which makes it harder to keep muscle while dieting"}.get(
            phase, "no training at all this week")
    if n <= 2 and phase == "Muscle Gain":
        return "light training for a gain phase"
    return ""


def attention(lab: dict) -> list[str]:
    """What needs attention, most important first (the gap weighed by what it does in this phase)."""
    items = [(lab[k]["priority"], k) for k in v1.off_target(lab)]
    if lab["no_workouts"]:
        items.append((lab["workout_priority"], "workouts"))
    return [k for _score, k in sorted(items, key=lambda t: -t[0])]


def links(lab: dict) -> list[str]:
    """The only connections between the metrics that follow from the verdicts. Nothing else may be claimed."""
    s = {k: lab[k]["status"] for k in ("steps", "protein", "calories")}
    out = []
    if s["steps"] == "below" and s["calories"] == "over":
        out.append("both the movement side and the food side are pushing the same wrong way")
    if s["steps"] == "met" and s["calories"] == "over":
        out.append("movement is fine, so the gap is entirely on the food side")
    if s["calories"] == "under" and s["protein"] == "below":
        out.append("protein is short mostly because total food is short, so eating more overall helps both")
    if s["protein"] == "below" and s["calories"] in ("on", "over"):
        out.append("there is enough food but too little of it is protein, so the fix is a swap, not more food")
    if s["calories"] == "under" and (lab["steps"].get("level") == "well above the goal" or lab["n_workouts"] >= 5):
        out.append("a lot of activity on too little food")
    if (s["steps"] == "below" and s["calories"] == "on" and s["protein"] in ("met", "nodata")
            and lab["phase"] == "Fat Loss" and lab["n_workouts"] > 0 and not lab["calories"].get("consistency_meaning")):
        out.append("food is on plan, so movement is the only thing holding the deficit back")
    return out


def watch(lab: dict) -> list[tuple[str, str]]:
    """Metrics that are fine on average this week but deserve an eye: drifting toward a miss next week, or swinging so
    much from day to day that the average hides bad days. [(metric, why)]"""
    out = []
    for k in ("steps", "protein", "calories"):
        m = lab[k]
        if m["status"] not in ("met", "on"):
            continue
        if "next week" in m.get("trend_meaning", ""):
            out.append((k, m["trend_meaning"]))
        elif m.get("consistency_meaning") and m["consistency"].startswith(("very erratic", "quite variable")):
            out.append((k, m["consistency_meaning"]))
    return out


def _headline(lab: dict, needs: list[str]) -> str:
    if not needs:
        early = [k for k, _why in watch(lab)]
        if early:
            return f"everything is on track on average this week, but keep an eye on {' and '.join(early)}"
        return "a steady week with everything on track; the job now is to repeat it"
    top = needs[0]
    if top == "workouts":
        return f"the main issue this week is training: {lab['workout_meaning']}"
    m = lab[top]
    return f"the main issue this week is {top} ({m['severity']} gap): {m['meaning']}"


def label_summary(data: dict) -> str:
    """The text block the model reads. It has no digits at all."""
    lab = classify(data)
    needs = attention(lab)
    lines = [f"Phase: {lab['phase']}", f"Headline: {_headline(lab, needs)}.", ""]

    for key in ("steps", "calories", "protein"):
        m = lab[key]
        if m["status"] == "nodata":
            lines.append(f"{_TITLES[key]}: no data this week.")
            continue
        parts = [m["wording"] + (f", a {m['severity']} gap ({m['gap']} off)" if m["severity"] else ""), m["trend_meaning"]]
        if m["consistency_meaning"]:
            parts.append(m["consistency_meaning"])
        line = f"{_TITLES[key]}: " + "; ".join(parts) + "."
        if m["meaning"]:
            line += f" What it means: {m['meaning']}."
        lines.append(line)
    w = lab["workout_meaning"]
    lines.append(f"Workouts: {lab['workouts']}." + (f" What it means: {w}." if w else ""))
    if lab["calories"]["status"] == "on" and lab["goal_days"] not in ("not enough data yet",):
        lines.append(f"Calories landed close to the target {lab['goal_days']}.")
    lines.append("")

    if needs:
        lines.append("Priorities, most important first:")
        for i, k in enumerate(needs[:3]):
            lines.append(f"  {_ORDINALS[min(i, 2)]}: {k}. Kind of fix: {_fix(k, lab)}.")
    else:
        lines.append("Priorities: none. Protect the routine that is working; do not invent problems or new habits.")
    fine = v1.on_track(lab)
    lines.append("Keep as they are, do not ask for more: " + (", ".join(fine) if fine else "nothing") + ".")
    early = watch(lab)
    if early:
        lines.append("Keep an eye on (fine on average, but): " + "; ".join(f"{k}, {why}" for k, why in early) + ".")
    missing = v1.no_data(lab)
    if missing:
        lines.append("No data this week (say nothing about these): " + ", ".join(missing) + ".")
    found = links(lab)
    lines.append("")
    lines.append("Links between the metrics (use only these, claim no others): "
                 + ("; ".join(found) if found else "none worth mentioning") + ".")
    return "\n".join(lines)
