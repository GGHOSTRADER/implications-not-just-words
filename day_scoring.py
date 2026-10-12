"""day_scoring.py — the ONE place that decides whether a day was "on target".

Previously this logic was hand-duplicated in THREE places that could silently
drift apart: server.py's _day_score() (for today's live row), background_checks.py's
nightly save_daily_summary (a copy of the same math, for the cached row), and
stats_mini_app.html's renderDetail() (a THIRD reimplementation in JS for the
day-detail popup). All three now call score_day() below instead.

2026-07-28: calorie hit is PHASE-AWARE (delta = actual - target):
  Fat Loss ("cut"):    hit if delta <= CUT_OVER_BAND_KCAL (30). Under target by
                       any amount is always a hit (eating less is fine on a
                       cut) — only overeating by more than the band misses.
  Muscle Gain ("bulk"): hit if delta >= -BULK_UNDER_BAND_KCAL (30). Over target
                       by any amount is always a hit (extra calories are fine
                       on a bulk) — only undereating by more than the band
                       misses.
  Maintenance:         symmetric band, hit if |delta| <= MAINTENANCE_BAND_KCAL (40).
  Unknown/missing phase falls back to the Maintenance band (safest default —
  neither a cut's nor a bulk's one-sided rule fits a phase we don't know).

Protein: hit if actual >= PROTEIN_HIT_FRACTION (0.95) of target; above target
also counts as a hit (no upper bound).

Workout: workout_hit is BLUEPRINT-COMPLETION-aware (2026-07-29) when the
logged workout can be matched to a blueprint by name: hit if
>= WORKOUT_COMPLETION_FRACTION (0.80) of that blueprint's exercises were
logged that day, by exact (lowercased) name match. This is scoped to
Mini-App-logged workouts ONLY — the Mini App's "change today's workout"
button writes an exact blueprint name into workout_type, and its logging flow
round-trips exact blueprint exercise names, so name matching is reliable
there. Bot-logged (conversational) workouts don't yet round-trip exact names
reliably (pending refactor — see project_bot_workout_logging_refactor_todo in
memory) and are NOT matched against a blueprint here; they fall back to plain
workout_logged, same as when no blueprint match is found at all (blueprint
deleted/renamed since, or the workout_type doesn't match any current
blueprint name).
"""
from __future__ import annotations

CUT_OVER_BAND_KCAL = 30
BULK_UNDER_BAND_KCAL = 30
MAINTENANCE_BAND_KCAL = 40
PROTEIN_HIT_FRACTION = 0.95
WORKOUT_COMPLETION_FRACTION = 0.80

PHASE_FAT_LOSS = "Fat Loss"
PHASE_MUSCLE_GAIN = "Muscle Gain"
PHASE_MAINTENANCE = "Maintenance"


def _calorie_hit(delta: float, phase: str | None) -> bool:
    """delta = actual - target. Band shape depends on phase; unknown phase
    falls back to the symmetric Maintenance band."""
    if phase == PHASE_FAT_LOSS:
        return delta <= CUT_OVER_BAND_KCAL
    if phase == PHASE_MUSCLE_GAIN:
        return delta >= -BULK_UNDER_BAND_KCAL
    return -MAINTENANCE_BAND_KCAL <= delta <= MAINTENANCE_BAND_KCAL


def blueprint_completion_fraction(blueprint_exercises: list, logged_exercises: list) -> float | None:
    """Fraction of a blueprint's exercises that were logged, by exact
    (lowercased) name match. Returns None when the blueprint has no exercises
    to match against (nothing to compute a fraction of — the caller should
    fall back to plain workout_logged in that case, not treat 0 exercises as
    a trivial 100%)."""
    bp_names = {
        (ex.get("name", "") if isinstance(ex, dict) else str(ex)).strip().lower()
        for ex in (blueprint_exercises or [])
    }
    bp_names.discard("")
    if not bp_names:
        return None
    logged_names = {
        (ex.get("name", "") if isinstance(ex, dict) else str(ex)).strip().lower()
        for ex in (logged_exercises or [])
    }
    return len(bp_names & logged_names) / len(bp_names)


def score_day(row: dict) -> dict:
    """Compute per-metric hit flags + the overall green/yellow/red/grey score
    from a calendar-data-shaped row (steps, step_goal_hit, calories,
    calories_target, calorie_delta?, protein, protein_target, phase?,
    workout_logged?, workout_completion_fraction?, is_rest_day?).

    Returns {"score", "step_hit", "cal_hit", "protein_hit", "workout_hit"}.

    workout_hit: when the caller supplies workout_completion_fraction (from
    blueprint_completion_fraction() above, computed only for Mini-App-logged
    workouts matched to a current blueprint by name), hit if that fraction is
    >= WORKOUT_COMPLETION_FRACTION (0.80). Otherwise (no blueprint match, bot-
    logged, or nothing logged) falls back to plain bool(workout_logged) — a
    day with SOME log but no blueprint match still counts as a hit, same as
    the previous simple rule.
    """
    step_hit = bool(row.get("step_goal_hit"))

    cal_delta = row.get("calorie_delta")
    if cal_delta is None:
        cal_delta = (row.get("calories") or 0) - (row.get("calories_target") or 0)
    cal_hit = _calorie_hit(cal_delta or 0, row.get("phase"))

    prot = row.get("protein") or 0
    prot_tgt = row.get("protein_target") or 0
    protein_hit = prot_tgt > 0 and prot >= prot_tgt * PROTEIN_HIT_FRACTION

    completion = row.get("workout_completion_fraction")
    if completion is not None:
        workout_hit = completion >= WORKOUT_COMPLETION_FRACTION
    else:
        workout_hit = bool(row.get("workout_logged"))
    # workout_hit stays the plain "did they train" fact (summable across a
    # week, 2026-07-28). For the SCORE a planned rest day has no session to
    # attend — resting IS the plan — so it counts as met; otherwise a rest day
    # could never be green (best case 3 of 4 → yellow). Reported 2026-09-17.
    workout_ok = workout_hit or bool(row.get("is_rest_day"))

    hits = int(step_hit) + int(cal_hit) + int(protein_hit) + int(workout_ok)
    total = 4

    if hits == total:
        score = "green"
    elif hits >= total * 0.5:
        score = "yellow"
    else:
        score = "red"

    return {
        "score": score,
        "step_hit": step_hit,
        "cal_hit": cal_hit,
        "protein_hit": protein_hit,
        "workout_hit": workout_hit,
    }
