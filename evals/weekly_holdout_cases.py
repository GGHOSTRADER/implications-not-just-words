"""Ten more synthetic weeks, kept apart from weekly_cases.py (the 20 the prompts and labels were tuned on).

Why they exist: every prompt and label rule so far was adjusted while looking at the first twenty weeks, so a good
score there is optimistic. These ten were written as different situations (not variations of a tuned case) and nothing
may be tuned against them: they are the check that an improvement is real. If one is ever used to tune, retire it and
write a fresh one.

Same shape as weekly_cases.CASES; run with `python -m evals.run_weekly_eval --set holdout`.
"""
from __future__ import annotations

from evals.weekly_cases import _NO_DATA, _case

# (steps, calories, protein) tuples are (avg, coefficient_of_variation_pct, slope_per_day)
CASES = dict([
    _case("H01 only calories a bit over", "Maintenance", (9000, 2500, 150), (11500, 9.0, 5.0), (2750, 6.0, 8.0), (156, 6.0, 0.5), 4, 2),
    _case("H02 steps far above, protein a bit low", "Fat Loss", (8000, 2100, 140), (12600, 12.0, 60.0), (2090, 5.0, 0.0), (128, 9.0, -1.0), 3, 5),
    _case("H03 protein far above, steps short", "Muscle Gain", (9000, 3000, 150), (6800, 21.0, -70.0), (3020, 4.0, 2.0), (214, 7.0, 3.0), 5, 5),
    _case("H04 cutting too hard, nothing trained", "Fat Loss", (10000, 2200, 146), (6000, 26.0, -150.0), (1500, 9.0, -30.0), (100, 14.0, -3.0), 0, 0),
    _case("H05 calories on average but erratic", "Fat Loss", (10000, 2200, 146), (9400, 8.0, 10.0), (2210, 38.0, 4.0), (147, 6.0, 0.0), 4, 1),
    _case("H06 seven workouts, steps short, calories low", "Muscle Gain", (9000, 3100, 170), (7400, 15.0, -40.0), (2750, 8.0, -15.0), (168, 6.0, 0.0), 7, 2),
    _case("H07 no step data this week", "Fat Loss", (10000, 2200, 146), _NO_DATA, (2230, 6.0, 3.0), (150, 7.0, 1.0), 3, 4),
    _case("H08 maintenance slowly drifting down", "Maintenance", (9000, 2400, 140), (8700, 11.0, -55.0), (2390, 5.0, -20.0), (133, 7.0, -2.0), 3, 4),
    _case("H09 calories slightly over and protein low", "Fat Loss", (10000, 2000, 150), (10400, 7.0, 20.0), (2150, 7.0, 10.0), (118, 12.0, -2.0), 2, 1),
    _case("H10 muscle gain overshooting", "Muscle Gain", (9000, 3000, 160), (10200, 9.0, 10.0), (3900, 8.0, 40.0), (166, 5.0, 1.0), 5, 0),
])


def names() -> list[str]:
    return list(CASES)
