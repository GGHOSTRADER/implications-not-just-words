"""Twenty synthetic training weeks for the weekly review. No database: each case is the dict
`weekly_analysis.gather_weekly_analysis_data()` would return.

Cases are chosen to cover the situations where a small model goes wrong: a metric that is met (must not be told to
increase it), a metric far below target, every metric fine, no workouts, a metric with no data, high variability,
and each phase (Fat Loss / Maintenance / Muscle Gain).

A case is (name, data). What a GOOD answer must do is derived from the data by evals/weekly_rubric.py, so adding a
case needs no extra bookkeeping.
"""
from __future__ import annotations


def _stats(avg, cv, slope, status="ok"):
    return {"status": status, "avg": avg, "coefficient_of_variation_pct": cv, "slope": slope}


_NO_DATA = {"status": "insufficient_data"}
_TYPES = ["Upper", "Lower", "Push", "Pull", "Legs", "Upper", "Lower"]


def _case(name, phase, goals, steps, cal, prot, workouts, hit, peak=(18, 1450)):
    step_goal, cal_target, prot_target = goals
    return name, {
        "goals": {"phase": phase, "step_goal": step_goal, "calories_target": cal_target, "protein_target": prot_target},
        "steps": steps if isinstance(steps, dict) else _stats(*steps),
        "calories": cal if isinstance(cal, dict) else _stats(*cal),
        "protein": prot if isinstance(prot, dict) else _stats(*prot),
        "hourly": {"status": "ok", "peak_hour": peak[0], "peak_avg_steps": peak[1]},
        "workouts": [{"date": f"2026-09-{22 + i}", "type": _TYPES[i % len(_TYPES)]} for i in range(workouts)],
        "goal_hit_days": hit, "total_days": 7,
    }


# (steps, calories, protein) tuples are (avg, coefficient_of_variation_pct, slope_per_day)
CASES = dict([
    _case("01 everything on target", "Maintenance", (9000, 2500, 150), (9300, 8.0, 20.0), (2490, 4.0, 1.0), (155, 5.0, 0.5), 4, 6),
    _case("02 steps slightly below", "Fat Loss", (10000, 2200, 146), (9100, 12.0, -30.0), (2190, 6.0, 0.0), (150, 6.0, 1.0), 4, 5),
    _case("03 steps far below", "Fat Loss", (10000, 2200, 146), (5200, 35.0, -260.0), (2210, 7.0, 3.0), (149, 6.0, 0.0), 2, 4),
    _case("04 protein slightly below", "Fat Loss", (10000, 2200, 146), (10400, 9.0, 40.0), (2150, 5.0, -5.0), (132, 9.0, -1.0), 4, 6),
    _case("05 protein far below", "Muscle Gain", (9000, 3000, 180), (9400, 11.0, 10.0), (3040, 8.0, 12.0), (96, 22.0, -4.0), 5, 5),
    _case("06 steps and protein both below", "Fat Loss", (10000, 2200, 146), (6400, 31.0, -210.0), (2480, 20.0, 35.0), (101, 25.0, -6.0), 1, 1),
    _case("07 calories well over target", "Fat Loss", (10000, 2000, 150), (10200, 10.0, 15.0), (2650, 14.0, 60.0), (158, 8.0, 2.0), 3, 1),
    _case("08 calories far under target", "Fat Loss", (10000, 2400, 150), (11000, 9.0, 5.0), (1620, 12.0, -40.0), (118, 15.0, -3.0), 3, 0),
    _case("09 steps met but very erratic", "Maintenance", (9000, 2400, 140), (9600, 41.0, 5.0), (2410, 6.0, 2.0), (146, 7.0, 0.5), 3, 5),
    _case("10 everything trending down", "Fat Loss", (10000, 2200, 146), (8600, 18.0, -380.0), (2050, 11.0, -90.0), (131, 16.0, -9.0), 3, 2),
    _case("11 everything trending up", "Maintenance", (9000, 2500, 150), (9800, 10.0, 450.0), (2540, 9.0, 85.0), (157, 8.0, 8.0), 4, 4),
    _case("12 muscle gain, protein met, calories short", "Muscle Gain", (9000, 3100, 170), (9200, 8.0, 0.0), (2640, 6.0, -10.0), (176, 5.0, 1.0), 5, 1),
    _case("13 maintenance, all within range", "Maintenance", (8000, 2300, 130), (8150, 7.0, 10.0), (2310, 3.0, 0.0), (133, 4.0, 0.0), 3, 7),
    _case("14 no workouts at all", "Fat Loss", (10000, 2200, 146), (9900, 9.0, 12.0), (2180, 6.0, -2.0), (149, 7.0, 0.0), 0, 4),
    _case("15 six workouts", "Muscle Gain", (9000, 3000, 170), (9700, 8.0, 15.0), (3010, 5.0, 3.0), (172, 5.0, 1.0), 6, 6),
    _case("16 no protein data this week", "Fat Loss", (10000, 2200, 146), (9200, 12.0, -20.0), (2250, 7.0, 4.0), _NO_DATA, 3, 4),
    _case("17 steps exactly on goal", "Fat Loss", (10000, 2200, 146), (10000, 10.0, 0.0), (2200, 5.0, 0.0), (140, 8.0, -1.0), 3, 5),
    _case("18 protein just above target", "Fat Loss", (10000, 2200, 146), (7800, 19.0, -90.0), (2230, 6.0, 5.0), (148, 6.0, 0.5), 3, 4),
    _case("19 calorie goal never hit", "Fat Loss", (10000, 2200, 146), (10100, 8.0, 10.0), (2410, 17.0, 30.0), (150, 7.0, 0.0), 4, 0),
    _case("20 calorie goal hit every day", "Fat Loss", (10000, 2200, 146), (8900, 14.0, -40.0), (2205, 1.5, 0.0), (151, 4.0, 0.5), 4, 7),
])


def names() -> list[str]:
    return list(CASES)
