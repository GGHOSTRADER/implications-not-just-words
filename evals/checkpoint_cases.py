"""Thirty synthetic checkpoints (20 tuning + 10 holdout) for comparing the checkpoint comment designs.

Each case is built through the production code's own pure pieces, so both designs see exactly what they would see live:
  - the block figures come from background_checks._metric_block (the function _compute_checkpoint uses);
  - the original design's context text is background_checks._fc_format_context(context data);
  - the labels design's input is background_checks._checkpoint_label_input(...).
Schedule: the default checkpoint hours (12, 15, 19, 23:59), wake hour 7, an even four-way split.

A spec is (name, phase, checkpoint index 0-3, (cal, protein, steps) targets, cumulative BEFORE this block, logged IN
this block, context options). Holdout weeks (names starting with H) are never used to tune anything.
"""
from __future__ import annotations

import copy

HOURS = [12, 15, 19, 23]
WAKE = 7
CHECKPOINTS = [(f"{h:02d}:{'59' if h == HOURS[-1] else '00'}", h, (i + 1) / 4, (i + 1) / 4, (i + 1) / 4)
               for i, h in enumerate(HOURS)]
ORDINALS = ["first", "second", "third", "fourth"]

INVENTORY = [("Greek yogurt", 100, 10), ("Chicken breast", 165, 31), ("Peanut butter toast", 290, 9),
             ("Banana", 105, 1), ("Salmon fillet", 210, 23), ("Cottage cheese", 98, 11), ("Granola bar", 190, 4),
             ("Egg whites", 52, 11), ("Rice bowl", 340, 7), ("Protein shake", 120, 24), ("Apple", 95, 0),
             ("Trail mix", 280, 8)]
RAIN = {"description": "light rain", "temp_c": 21.6, "wind_kph": 14, "precipitation_mm": 1.4, "weather_code": 61}
CLEAR = {"description": "clear sky", "temp_c": 27.2, "wind_kph": 6, "precipitation_mm": 0, "weather_code": 0}
WINDY = {"description": "overcast", "temp_c": 16.0, "wind_kph": 34, "precipitation_mm": 0, "weather_code": 3}
HOT = {"description": "sunny", "temp_c": 34.0, "wind_kph": 8, "precipitation_mm": 0, "weather_code": 0}
STORMY = {"description": "overcast", "temp_c": 18.0, "wind_kph": 48, "precipitation_mm": 0, "weather_code": 3}
WEATHER = {"rain": RAIN, "clear": CLEAR, "windy": WINDY, "none": None, "hot": HOT, "stormy": STORMY}


def _entries(cal: float, prot: float) -> list[dict]:
    """Food log entries that add up to the cumulative calories and protein (nothing when both are zero)."""
    if round(cal) == 0 and round(prot) == 0:
        return []
    names = ["Oatmeal with milk", "Chicken rice", "Fruit and yogurt", "Pasta"]
    n = 2 if cal > 900 else 1
    out = []
    for i in range(n):
        out.append({"food": names[i], "calories": round(cal / n, 1), "protein": round(prot / n, 1)})
    return out


def build(spec: tuple) -> dict:
    """One case: the cp dict and context text for the original design, the input for the labels design, and truth."""
    import background_checks as bc
    name, phase, idx, targets, before, block, opts = spec
    cal_t, prot_t, steps_t = targets
    frac = (idx + 1) / 4
    keys = ("calories", "protein", "steps")
    metrics = {}
    for key, daily, b0, live in zip(keys, targets, before, block):
        expected = max(0, round(daily * frac - b0))
        metrics[key] = bc._metric_block(live, expected, daily, round(daily * frac), daily - b0, 1 - frac, 0.25)
    cum = {k: before[i] + block[i] for i, k in enumerate(keys)}
    cp = {"ordinal": ORDINALS[idx], "idx": idx, "label": CHECKPOINTS[idx][0], "hour": HOURS[idx], "frac": frac,
          **metrics, "scenario": max(m["status"] for m in metrics.values()), "cal_target": cal_t,
          "cal_cum_live": cum["calories"], "cal_cum_expected": cal_t * frac, "steps_cum_live": cum["steps"],
          "is_last": idx == 3, "block_start_hour": HOURS[idx - 1] if idx > 0 else WAKE}
    ctx = {"weather": copy.deepcopy(WEATHER[opts.get("weather", "clear")]),
           "food_entries": _entries(cum["calories"], cum["protein"]),
           "inventory": list(opts.get("inventory", INVENTORY)),
           "food_likes": opts.get("likes", ""), "food_dislikes": opts.get("dislikes", ""),
           "workout": opts.get("workout", {"kind": "session", "type": "Upper Day", "logged": False}),
           "morning_feedback": opts.get("feedback"), "phase": phase}
    goals = {"phase": phase, "calories_target": cal_t, "protein_target": prot_t, "step_goal": steps_t}
    return {
        "name": name, "phase": phase, "cp": cp, "context_data": ctx, "goals": goals, "cum": cum,
        "ctx_text": bc._fc_format_context(ctx),
        "label_input": bc._checkpoint_label_input(cp, goals, cum, CHECKPOINTS, ctx),
    }


# (name, phase, idx, (cal, protein, steps) targets, cumulative before, in this block, options)
_T = (2200, 150, 10000)
_MG = (3000, 170, 9000)
_MT = (2500, 140, 9000)
SPECS = [
    ("01 cut, way over on calories at noon", "Fat Loss", 0, _T, (0, 0, 0), (850, 30, 2300), {"weather": "clear"}),
    ("02 cut, under on calories early, fine", "Fat Loss", 0, _T, (0, 0, 0), (300, 25, 2600), {"weather": "clear"}),
    ("03 cut, protein far behind mid-afternoon", "Fat Loss", 1, _T, (550, 20, 2600), (500, 8, 2400),
     {"weather": "clear", "likes": "chicken, yogurt"}),
    ("04 nothing logged by noon", "Fat Loss", 0, _T, (0, 0, 0), (0, 0, 2400), {"weather": "clear"}),
    ("05 nothing logged by three", "Muscle Gain", 1, _MG, (0, 0, 2000), (0, 0, 2100), {"weather": "rain"}),
    ("06 steps behind in the rain", "Fat Loss", 1, _T, (560, 38, 1500), (540, 36, 1200), {"weather": "rain"}),
    ("07 steps behind, weather unknown", "Maintenance", 2, _MT, (1250, 70, 3000), (620, 35, 1100), {"weather": "none"}),
    ("08 gain phase short on calories", "Muscle Gain", 2, _MG, (1100, 80, 4500), (500, 40, 2300),
     {"weather": "clear", "likes": "rice, peanut butter"}),
    ("09 gain phase a little ahead", "Muscle Gain", 1, _MG, (800, 45, 2300), (900, 45, 2200), {"weather": "clear"}),
    ("10 maintenance all on plan", "Maintenance", 1, _MT, (620, 35, 2250), (630, 35, 2300), {"weather": "clear"}),
    ("11 last check, cut, ended over", "Fat Loss", 3, _T, (1800, 120, 7600), (650, 25, 1900), {"weather": "clear"}),
    ("12 last check, cut, ended under and short on protein", "Fat Loss", 3, _T, (1500, 90, 7000), (450, 20, 2600),
     {"weather": "clear"}),
    ("13 last check, gain phase short", "Muscle Gain", 3, _MG, (2100, 130, 7000), (500, 30, 1500), {"weather": "rain"}),
    ("14 dislikes fish while protein behind", "Fat Loss", 1, _T, (550, 30, 2500), (550, 10, 2500),
     {"weather": "clear", "dislikes": "salmon, fish"}),
    ("15 hungry and tired this morning", "Fat Loss", 1, _T, (560, 37, 2500), (600, 30, 2400),
     {"weather": "clear", "feedback": "tired, very hungry, busy day at work"}),
    ("16 day budget already used up by evening", "Fat Loss", 2, _T, (1500, 90, 5000), (900, 40, 2400),
     {"weather": "windy"}),
    ("17 rest day, everything fine", "Maintenance", 1, _MT, (620, 35, 2250), (600, 37, 2200),
     {"weather": "clear", "workout": {"kind": "rest"}}),
    ("18 workout done, protein behind", "Muscle Gain", 2, _MG, (1500, 70, 4600), (800, 25, 2200),
     {"weather": "clear", "workout": {"kind": "session", "type": "Lower Day", "logged": True}}),
    ("19 steps far ahead, calories over", "Fat Loss", 2, _T, (1100, 75, 6500), (750, 38, 4500), {"weather": "clear"}),
    ("20 cardio day, under on everything", "Maintenance", 0, _MT, (0, 0, 0), (400, 15, 1200),
     {"weather": "rain", "workout": {"kind": "cardio"}}),
]
HOLDOUT_SPECS = [
    ("H01 cut, over and protein short at three", "Fat Loss", 1, _T, (600, 25, 2400), (750, 12, 2300), {"weather": "rain"}),
    ("H02 gain phase, nothing logged at noon", "Muscle Gain", 0, _MG, (0, 0, 0), (0, 0, 1800), {"weather": "clear"}),
    ("H03 maintenance, over at evening", "Maintenance", 2, _MT, (1300, 70, 4500), (900, 40, 2200), {"weather": "clear"}),
    ("H04 last check, maintenance on plan", "Maintenance", 3, _MT, (1880, 105, 6800), (600, 35, 2300), {"weather": "windy"}),
    ("H05 cut, steps far behind, clear sky", "Fat Loss", 2, _T, (1100, 75, 3000), (540, 36, 900),
     {"weather": "clear", "likes": "eggs"}),
    ("H06 gain phase, protein and calories short, dislikes chicken", "Muscle Gain", 1, _MG, (700, 30, 2200),
     (600, 20, 2300), {"weather": "clear", "dislikes": "chicken"}),
    ("H07 cut, on plan, workout done", "Fat Loss", 1, _T, (550, 37, 2500), (560, 38, 2550),
     {"weather": "rain", "workout": {"kind": "session", "type": "Push Day", "logged": True}}),
    ("H08 nothing logged by evening", "Fat Loss", 2, _T, (0, 0, 4800), (0, 0, 2600), {"weather": "none"}),
    ("H09 last check, cut, protein met, calories far over", "Fat Loss", 3, _T, (2100, 140, 8200), (700, 20, 1500),
     {"weather": "clear"}),
    ("H10 tired, rest day, under on calories in a gain phase", "Muscle Gain", 2, _MG, (1200, 85, 4000), (500, 30, 1500),
     {"weather": "windy", "workout": {"kind": "rest"}, "feedback": "slept badly, not hungry"}),
]


# Regression cases from a real message (2026-10-05, the 12:00 check): a gain phase, calories a third over, steps far
# behind, rain, a lower-body session already done, and a food list that includes a supplement. Kept apart from
# "tuning"/"holdout"/"all" so the fingerprints of everything measured before stay valid; run with --set incident.
_INC = (2768, 232, 4400)
_INC_INVENTORY = [("Fish Oil", 45, 0), ("Fish Pasta", 459, 31), ("Roasted fish", 401, 23), ("chicken breast Costco", 165, 31),
                  ("Protein shake isolat", 120, 24), ("Greek yogurt", 100, 10), ("Banana", 105, 1), ("Rice bowl", 340, 7),
                  ("Creatine", 10, 0), ("Granola bar", 190, 4)]
_LOWER_DONE = {"kind": "session", "type": "Lower body", "logged": True}
INCIDENT_SPECS = [
    ("I01 real message: rain, calories over, steps far behind, lower body done", "Muscle Gain", 0, _INC, (0, 0, 0),
     (921, 59, 207), {"weather": "rain", "inventory": _INC_INVENTORY, "workout": _LOWER_DONE}),
    ("I02 same day, very hot", "Muscle Gain", 0, _INC, (0, 0, 0), (921, 59, 207),
     {"weather": "hot", "inventory": _INC_INVENTORY, "workout": _LOWER_DONE}),
    ("I03 same day, weather unknown", "Muscle Gain", 0, _INC, (0, 0, 0), (921, 59, 207),
     {"weather": "none", "inventory": _INC_INVENTORY, "workout": _LOWER_DONE}),
    ("I04 seven in the evening, steps far behind, clear and sunny", "Fat Loss", 2, _T, (1100, 75, 2000), (560, 36, 700),
     {"weather": "clear", "inventory": _INC_INVENTORY}),
    ("I05 three in the afternoon, calories far over, rain", "Maintenance", 1, _MT, (900, 50, 2300), (1100, 40, 2200),
     {"weather": "rain", "inventory": _INC_INVENTORY}),
    ("I06 seven in the evening, rest day, steps far behind, strong wind", "Muscle Gain", 2, _MG, (1500, 85, 2000),
     (800, 40, 700), {"weather": "stormy", "inventory": _INC_INVENTORY, "workout": {"kind": "rest"}}),
    ("I07 last check, steps far behind, rain", "Fat Loss", 3, _T, (1800, 115, 3000), (350, 25, 800),
     {"weather": "rain", "inventory": _INC_INVENTORY}),
]


def cases(case_set: str = "all") -> dict:
    specs = {"tuning": SPECS, "holdout": HOLDOUT_SPECS, "all": SPECS + HOLDOUT_SPECS, "incident": INCIDENT_SPECS}[case_set]
    return {s[0]: build(s) for s in specs}
