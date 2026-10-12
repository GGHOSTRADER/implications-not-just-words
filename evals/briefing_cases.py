"""Thirty synthetic morning briefings (20 tuning + 10 holdout) for comparing the briefing designs.

For the original design, the inputs are formatted exactly as background_checks._fire_morning_briefing formats them
(classifications via _classify_steps/_classify_calories, the adjusted/base value strings and carry-over lines of each
adjustment branch). For the labels design, the input is built by the production adapter
background_checks._briefing_label_input from the same values.

A spec: (name, phase, day, yesterday (steps, step_goal, cal, cal_target, protein, protein_target, workout_day,
workout_type, workout_logged, weight_logged), today (workout, weather key, base_cal, base_steps, food_likes),
adjustment). Adjustment: None | ("calories", amount, adjusted_target) | ("calories_boost", amount, adjusted_target)
| ("steps", amount, extra_steps) | ("steps_reduce", amount, reduced_steps) | ("pending", amount, "over"|"under").
"""
from __future__ import annotations

import copy
from datetime import datetime
from types import SimpleNamespace

from evals.checkpoint_cases import WEATHER

DAYS = {"Monday": "2026-09-14", "Wednesday": "2026-09-16", "Friday": "2026-09-18", "Saturday": "2026-09-19",
        "Sunday": "2026-09-20"}


def _original_inputs(base_cal, base_steps, adj):
    """The value strings and carry-over line exactly as _fire_morning_briefing's adjustment branches build them."""
    cal_value, steps_value = f"{base_cal:,} kcal", f"{base_steps:,}"
    cal_is_adjusted = steps_is_adjusted = False
    line = None
    if adj and adj[0] == "pending":
        _kind, amount, direction = adj
        line = f"- Carry Over Calories: {amount:.0f} kcal {'over' if direction == 'over' else 'under'} yesterday"
    elif adj:
        strategy, amount, value = adj
        if strategy == "calories":
            cal_value, line, cal_is_adjusted = f"{value:.0f} kcal", f"- Carry Over: {amount:.0f} kcal over yesterday", True
        elif strategy == "calories_boost":
            cal_value, line, cal_is_adjusted = f"{value:.0f} kcal", f"- Carry Over: {amount:.0f} kcal under yesterday", True
        elif strategy == "steps":
            steps_value, line, steps_is_adjusted = f"{base_steps + value:,}", f"- Carry Over: {amount:.0f} kcal over yesterday", True
        elif strategy == "steps_reduce":
            steps_value, line, steps_is_adjusted = f"{value:,}", f"- Carry Over: {amount:.0f} kcal under yesterday", True
    return cal_value, steps_value, cal_is_adjusted, steps_is_adjusted, line


def _adj_dict(adj, base_cal):
    """The today_calorie_adjustment state value the production code reads, for the labels adapter."""
    if not adj or adj[0] == "pending":
        return None
    strategy, amount, value = adj
    d = {"strategy": strategy, "amount": amount}
    if strategy in ("calories", "calories_boost"):
        d["adjusted_target"] = value
    elif strategy == "steps":
        d["extra_steps"] = value
    elif strategy == "steps_reduce":
        d["reduced_steps"] = value
    return d


def build(spec: tuple) -> dict:
    import background_checks as bc
    name, phase, day, y, t, adj = spec
    steps, step_goal, cal, cal_target, prot, prot_target, w_day, w_type, w_logged, weight_logged = y
    workout, weather_key, base_cal, base_steps, likes = t
    now = datetime.fromisoformat(DAYS[day] + "T06:00:00")
    weather = copy.deepcopy(WEATHER[weather_key])
    ys = SimpleNamespace(steps=steps, step_goal=step_goal, calories=cal, calories_target=cal_target, protein=prot,
                         protein_target=prot_target)
    cal_value, steps_value, cal_adj, steps_adj, line = _original_inputs(base_cal, base_steps, adj)
    original = {
        "yesterday": dict(phase=phase, yesterday_step_class=bc._classify_steps(steps, base_steps),
                          yesterday_cal_class=bc._classify_calories(cal, cal_target),
                          yesterday_protein_over=prot >= prot_target, yesterday_was_workout_day=w_day,
                          yesterday_workout_type=w_type, yesterday_workout_logged=w_logged,
                          yesterday_weight_logged=weight_logged),
        "today": dict(phase=phase, day=day, now=now, workout_today=workout, weather=weather, cal_is_adjusted=cal_adj,
                      cal_value=cal_value, steps_is_adjusted=steps_adj, steps_value=steps_value, food_likes=likes,
                      carryover_cal_line=line),
    }
    pending = {"amount": adj[1], "direction": adj[2]} if adj and adj[0] == "pending" else None
    label_input = bc._briefing_label_input(phase, day, now, ys, base_steps, base_cal, _adj_dict(adj, base_cal), pending,
                                           workout, weather, likes, w_day, w_type, w_logged, weight_logged)
    truth = {"steps": steps, "step_goal": step_goal, "calories": cal, "calories_target": cal_target, "protein": prot,
             "protein_target": prot_target, "adjustment": adj, "base_cal": base_cal, "base_steps": base_steps,
             "workout_today": workout, "weather": weather}
    return {"name": name, "phase": phase, "original": original, "label_input": label_input, "truth": truth}


_Y_UPPER = ("Upper Day", True)
SPECS = [
    ("01 cut, ate over, calories target lowered", "Fat Loss", "Wednesday",
     (8600, 10000, 2420, 2200, 152, 146, True, "Upper Day", True, True), ("Lower Day", "clear", 2200, 10000, "eggs, chicken"),
     ("calories", 220, 2010)),
    ("02 cut, ate under, target raised", "Fat Loss", "Wednesday",
     (10400, 10000, 1850, 2200, 140, 146, True, "Upper Day", True, True), ("Rest", "clear", 2200, 10000, "yogurt"),
     ("calories_boost", 350, 2350)),
    ("03 rain, steps offset carry-over", "Fat Loss", "Friday",
     (9800, 10000, 2380, 2200, 150, 146, False, "Rest", False, True), ("Push Day", "rain", 2200, 10000, ""),
     ("steps", 180, 3200)),
    ("04 pending carry-over over", "Fat Loss", "Monday",
     (7600, 10000, 2600, 2200, 120, 146, True, "Lower Day", False, False), ("Upper Day", "clear", 2200, 10000, "oats"),
     ("pending", 400, "over")),
    ("05 gain phase, everything hit", "Muscle Gain", "Wednesday",
     (9400, 9000, 3050, 3000, 175, 170, True, "Push Day", True, True), ("Pull Day", "clear", 3000, 9000, "rice, beef"),
     None),
    ("06 gain phase, under-ate, weather unknown", "Muscle Gain", "Saturday",
     (9100, 9000, 2500, 3000, 150, 170, True, "Legs", True, True), ("Rest", "none", 3000, 9000, "peanut butter"),
     ("calories_boost", 500, 3300)),
    ("07 maintenance, weekend, rest", "Maintenance", "Sunday",
     (12000, 9000, 2450, 2500, 140, 140, False, "Rest", False, True), ("Rest (scheduled)", "clear", 2500, 9000, ""), None),
    ("08 missed weigh-in and workout not logged", "Fat Loss", "Wednesday",
     (10100, 10000, 2190, 2200, 148, 146, True, "Upper Day", False, False), ("Lower Day", "rain", 2200, 10000, "eggs"),
     None),
    ("09 cut, far under on steps", "Fat Loss", "Monday",
     (4200, 10000, 2150, 2200, 150, 146, False, "Rest", False, True), ("Upper Day", "clear", 2200, 10000, "chicken"), None),
    ("10 steps reduced after under-eating", "Muscle Gain", "Friday",
     (8800, 9000, 2650, 3000, 160, 170, True, "Pull Day", True, True), ("Legs", "windy", 3000, 9000, ""),
     ("steps_reduce", 350, 7500)),
    ("11 protein far short", "Fat Loss", "Wednesday",
     (10300, 10000, 2180, 2200, 95, 146, True, "Push Day", True, True), ("Pull Day", "clear", 2200, 10000, "tuna, eggs"),
     None),
    ("12 windy cardio day", "Maintenance", "Saturday",
     (9500, 9000, 2550, 2500, 135, 140, False, "Rest", False, True), ("Cardio (scheduled)", "windy", 2500, 9000, ""), None),
    ("13 no workout plan set", "Fat Loss", "Friday",
     (9900, 10000, 2230, 2200, 147, 146, False, "Rest", False, True), ("None set", "clear", 2200, 10000, ""), None),
    ("14 pending carry-over under, rain", "Muscle Gain", "Monday",
     (9000, 9000, 2600, 3000, 172, 170, True, "Upper Day", True, True), ("Lower Day", "rain", 3000, 9000, "salmon"),
     ("pending", 400, "under")),
    ("15 cut, way over yesterday, big cut today", "Fat Loss", "Wednesday",
     (11000, 10000, 2900, 2200, 160, 146, True, "Lower Day", True, True), ("Upper Day", "clear", 2200, 10000, "eggs"),
     ("calories", 700, 1750)),
    ("16 everything on target, rest day", "Fat Loss", "Sunday",
     (10050, 10000, 2205, 2200, 150, 146, False, "Rest", False, True), ("Rest", "clear", 2200, 10000, "yogurt"), None),
    ("17 maintenance under on steps and protein", "Maintenance", "Wednesday",
     (6500, 9000, 2480, 2500, 110, 140, True, "Pull Day", False, True), ("Push Day", "windy", 2500, 9000, ""), None),
    ("18 gain phase over a lot", "Muscle Gain", "Friday",
     (9300, 9000, 3700, 3000, 180, 170, True, "Legs", True, True), ("Rest", "clear", 3000, 9000, "rice"),
     ("calories", 700, 2650)),
    ("19 rain, protein short, steps far ahead", "Fat Loss", "Saturday",
     (15200, 10000, 2150, 2200, 120, 146, False, "Rest", False, True), ("Rest", "rain", 2200, 10000, "cottage cheese"), None),
    ("20 maintenance, ate over, steps raised", "Maintenance", "Monday",
     (9100, 9000, 2800, 2500, 145, 140, True, "Upper Day", True, False), ("Lower Day", "clear", 2500, 9000, ""),
     ("steps", 300, 4000)),
]
HOLDOUT_SPECS = [
    ("H01 cut, under-ate, rain", "Fat Loss", "Monday",
     (9700, 10000, 1900, 2200, 150, 146, True, "Lower Day", True, True), ("Upper Day", "rain", 2200, 10000, "eggs"),
     ("calories_boost", 300, 2450)),
    ("H02 gain phase, missed weigh-in, everything else hit", "Muscle Gain", "Wednesday",
     (9600, 9000, 3100, 3000, 175, 170, True, "Pull Day", True, False), ("Legs", "clear", 3000, 9000, "oats"), None),
    ("H03 maintenance, pending over, weekend", "Maintenance", "Saturday",
     (8000, 9000, 2850, 2500, 130, 140, False, "Rest", False, True), ("Rest", "windy", 2500, 9000, ""),
     ("pending", 350, "over")),
    ("H04 cut, steps far short, weather unknown", "Fat Loss", "Friday",
     (5100, 10000, 2250, 2200, 140, 146, True, "Upper Day", True, True), ("Lower Day", "none", 2200, 10000, ""), None),
    ("H05 gain phase, under, steps reduced", "Muscle Gain", "Monday",
     (9200, 9000, 2400, 3000, 165, 170, True, "Push Day", False, True), ("Pull Day", "rain", 3000, 9000, "beef"),
     ("steps_reduce", 600, 7000)),
    ("H06 cut, slightly over, cardio", "Fat Loss", "Wednesday",
     (10200, 10000, 2300, 2200, 146, 146, False, "Rest", False, True), ("Cardio (scheduled)", "clear", 2200, 10000, "fruit"),
     ("calories", 100, 2100)),
    ("H07 maintenance, everything on target, workout logged", "Maintenance", "Friday",
     (9000, 9000, 2500, 2500, 140, 140, True, "Legs", True, True), ("Upper Day", "rain", 2500, 9000, "yogurt"), None),
    ("H08 cut, protein short, workout not logged, rain", "Fat Loss", "Monday",
     (8800, 10000, 2150, 2200, 105, 146, True, "Pull Day", False, True), ("Push Day", "rain", 2200, 10000, "eggs, tofu"),
     None),
    ("H09 gain phase, big surplus, steps raised", "Muscle Gain", "Sunday",
     (8000, 9000, 3600, 3000, 170, 170, False, "Rest", False, True), ("Rest", "clear", 3000, 9000, ""),
     ("steps", 600, 3000)),
    ("H10 maintenance, no plan, missed weigh-in", "Maintenance", "Wednesday",
     (7700, 9000, 2300, 2500, 120, 140, False, "Rest", False, False), ("None set", "windy", 2500, 9000, "pasta"), None),
]


# Regression cases for the steps-and-workout concepts found in a real checkpoint message (2026-10-05); see
# checkpoint_cases.INCIDENT_SPECS. Kept apart from "all" so earlier fingerprints stay valid (--set incident).
INCIDENT_SPECS = [
    ("I01 rain, training day (Lower body), steps raised", "Muscle Gain", "Monday",
     (3800, 4400, 3100, 2768, 220, 232, True, "Upper body", True, True), ("Lower body", "rain", 2768, 4400, "rice"),
     ("steps", 300, 5000)),
    ("I02 very hot, training day", "Fat Loss", "Wednesday",
     (9800, 10000, 2250, 2200, 148, 146, True, "Lower body", True, True), ("Upper body", "hot", 2200, 10000, "eggs"), None),
    ("I03 weather unknown, rest day, steps far behind", "Maintenance", "Friday",
     (4500, 9000, 2480, 2500, 138, 140, False, "Rest", False, True), ("Rest", "none", 2500, 9000, "yogurt"), None),
    ("I04 clear and sunny, training day", "Muscle Gain", "Saturday",
     (9100, 9000, 3050, 3000, 172, 170, True, "Legs", True, True), ("Push Day", "clear", 3000, 9000, "oats"), None),
    ("I05 strong wind, training day, steps lowered", "Muscle Gain", "Sunday",
     (8800, 9000, 2400, 3000, 165, 170, True, "Upper body", False, True), ("Lower body", "stormy", 3000, 9000, ""),
     ("steps_reduce", 400, 7600)),
]


def cases(case_set: str = "all") -> dict:
    specs = {"tuning": SPECS, "holdout": HOLDOUT_SPECS, "all": SPECS + HOLDOUT_SPECS, "incident": INCIDENT_SPECS}[case_set]
    return {s[0]: build(s) for s in specs}
