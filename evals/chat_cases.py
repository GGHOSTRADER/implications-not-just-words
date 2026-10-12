"""Fake days and typical questions for the chat agent (2026-10-05).

A case is one day plus one question. The days are made up (the harness cuts every route to the live database), the
questions are the ones people actually ask the chat. 20 tuning cases (used to build and fix the design) and 10 held-out
cases (H01-H10, never looked at while tuning). The 16 questions of the first real-model check (2026-10-05, a real
evening) are all here in some form.

`kind` tells the checklist what a good answer must contain (evals/chat_rubric.py).
"""
from __future__ import annotations

# the plan's cumulative curve, as background_checks._load_checkpoints returns it: (label, hour, steps, calories, protein)
SCHEDULE = [("12:00", 12, 0.10, 0.30, 0.30), ("15:00", 15, 0.25, 0.50, 0.50),
            ("19:00", 19, 0.40, 0.75, 0.75), ("23:59", 23, 1.0, 1.0, 1.0)]

# food entries are (name, kcal, protein, "HH:MM")
DAYS = {
    "late_over": dict(hour=23, minute=33, phase="Muscle Gain", step_goal=11000, steps=11571, steps_age=20, base_kcal=2769,
                      protein_target=130, entries=[("Oats", 380, 14, "08:10"), ("Chicken", 520, 62, "12:45"),
                                                   ("Rice bowl", 640, 18, "13:30"), ("Protein shake", 210, 35, "16:20"),
                                                   ("Pasta", 690, 24, "19:40"), ("Shrimp", 287, 67, "21:43"),
                                                   ("Banana", 72, 1, "23:28")]),
    "afternoon_behind": dict(hour=15, minute=10, phase="Muscle Gain", step_goal=11000, steps=2000, steps_age=12,
                             base_kcal=2783, protein_target=130,
                             entries=[("Oats", 380, 14, "08:10"), ("Banana", 105, 1, "10:30"), ("Chicken", 415, 40, "13:05")]),
    "evening_room": dict(hour=19, minute=45, phase="Fat Loss", step_goal=11000, steps=8500, steps_age=8, base_kcal=2400,
                         protein_target=150,
                         entries=[("Oats", 350, 12, "08:00"), ("Chicken salad", 600, 45, "13:00"), ("Yogurt", 150, 15, "16:00"),
                                  ("Eggs", 500, 36, "18:30")]),
    "morning_start": dict(hour=9, minute=5, phase="Fat Loss", step_goal=11000, steps=600, steps_age=5, base_kcal=2300,
                          protein_target=150, entries=[("Coffee with milk", 60, 3, "07:40"), ("Eggs", 290, 22, "08:20")]),
    "midday_on_track": dict(hour=12, minute=40, phase="Maintenance", step_goal=11000, steps=4500, steps_age=10,
                            base_kcal=2200, protein_target=140,
                            entries=[("Oats", 400, 15, "08:00"), ("Chicken", 480, 40, "12:15"), ("Banana", 120, 1, "10:30")]),
    "night_under": dict(hour=22, minute=15, phase="Fat Loss", step_goal=10000, steps=6000, steps_age=15, base_kcal=2200,
                        protein_target=150,
                        entries=[("Oats", 400, 15, "08:00"), ("Chicken", 520, 62, "13:00"), ("Rice", 450, 9, "13:10"),
                                 ("Yogurt", 330, 20, "18:30")]),
    "stale_steps": dict(hour=14, minute=0, phase="Muscle Gain", step_goal=11000, steps=1500, steps_age=240, base_kcal=2300,
                        protein_target=140, entries=[("Oats", 400, 15, "08:00"), ("Chicken", 700, 60, "13:00")]),
    # days for the FRESH set (written 2026-10-05 after the first held-out set was burned by reading its failures)
    "lunch_low": dict(hour=13, minute=20, phase="Fat Loss", step_goal=10000, steps=3000, steps_age=15, base_kcal=2000,
                      protein_target=140, entries=[("Oats", 300, 10, "07:50"), ("Coffee with milk", 50, 2, "09:30")]),
    "evening_over": dict(hour=20, minute=30, phase="Muscle Gain", step_goal=12000, steps=5000, steps_age=20, base_kcal=2600,
                         protein_target=130, entries=[("Oats", 450, 16, "08:00"), ("Chicken wrap", 650, 38, "12:30"),
                                                      ("Pasta", 750, 26, "16:00"), ("Protein shake", 200, 30, "17:30"),
                                                      ("Rice bowl", 250, 6, "19:00")]),
    "stale_evening": dict(hour=18, minute=15, phase="Maintenance", step_goal=11000, steps=9000, steps_age=200, base_kcal=2300,
                          protein_target=140, entries=[("Oats", 400, 15, "08:00"), ("Chicken", 520, 62, "13:00"),
                                                       ("Rice", 450, 9, "13:10"), ("Yogurt", 130, 13, "16:00")]),
    # days for the THIRD set (G01-G16), written 2026-10-05 before the fixes to the three weaknesses of the F set were checked
    "early_hungry": dict(hour=7, minute=50, phase="Fat Loss", step_goal=10000, steps=400, steps_age=5, base_kcal=1900,
                         protein_target=130, entries=[("Coffee with milk", 60, 3, "07:20")]),
    "midmorning_ahead": dict(hour=10, minute=20, phase="Muscle Gain", step_goal=12000, steps=2500, steps_age=10, base_kcal=2800,
                             protein_target=150, entries=[("Eggs", 300, 24, "07:30"), ("Toast", 300, 10, "07:35"),
                                                          ("Yogurt", 250, 20, "09:50")]),
    "afternoon_stale": dict(hour=16, minute=40, phase="Fat Loss", step_goal=10000, steps=4000, steps_age=190, base_kcal=2100,
                            protein_target=140, entries=[("Oats", 350, 12, "08:00"), ("Chicken salad", 450, 40, "12:40"),
                                                         ("Banana", 105, 1, "15:00"), ("Yogurt", 195, 17, "16:00")]),
    "evening_perfect": dict(hour=21, minute=10, phase="Maintenance", step_goal=10000, steps=10400, steps_age=12, base_kcal=2200,
                            protein_target=140, entries=[("Oats", 400, 15, "08:00"), ("Chicken", 600, 62, "13:00"),
                                                         ("Rice", 450, 9, "13:10"), ("Salmon", 500, 40, "19:30"),
                                                         ("Yogurt", 200, 16, "20:30")]),
    "post_goal_morning": dict(hour=9, minute=40, phase="Fat Loss", step_goal=10000, steps=11500, steps_age=10, base_kcal=2100,
                              protein_target=150, entries=[("Eggs", 300, 24, "07:30"), ("Toast", 300, 10, "07:35")]),
}

# (id, day, question, kind, set)
CASES = [
    ("01", "late_over", "how am I doing today", "status", "tuning"),
    ("02", "late_over", "is 500 kcal too much for my next meal?", "fit", "tuning"),
    ("03", "late_over", "how many steps do I still need today?", "steps_left", "tuning"),
    ("04", "late_over", "can I have a pizza?", "named_food", "tuning"),
    ("05", "late_over", "I'm behind on steps and it's raining, what can I do?", "steps_help", "tuning"),
    ("06", "afternoon_behind", "how am I doing today", "status", "tuning"),
    ("07", "afternoon_behind", "I'm behind on steps, what can I do?", "steps_help", "tuning"),
    ("08", "afternoon_behind", "I'm behind on steps and it's raining, what can I do?", "steps_help_rain", "tuning"),
    ("09", "afternoon_behind", "what should I eat next?", "what_eat", "tuning"),
    ("10", "afternoon_behind", "how's my protein?", "protein", "tuning"),
    ("11", "evening_room", "how many calories do I have left?", "cal_left", "tuning"),
    ("12", "evening_room", "is 500 kcal too much for my next meal?", "fit", "tuning"),
    ("13", "evening_room", "I'm still hungry, what can I eat without going over?", "what_eat", "tuning"),
    ("14", "evening_room", "should I go for a walk now?", "walk_now", "tuning"),
    ("15", "morning_start", "hi", "greeting", "tuning"),
    ("16", "morning_start", "what should I have for dinner?", "what_eat", "tuning"),
    ("17", "morning_start", "give me a workout to burn more calories", "workout", "tuning"),
    ("18", "night_under", "how am I doing today", "status", "tuning"),
    ("19", "night_under", "can I have a 300 kcal snack?", "fit", "tuning"),
    ("20", "stale_steps", "how many steps do I have?", "steps_left", "tuning"),
    ("H01", "midday_on_track", "am I on track?", "status", "holdout"),
    ("H02", "midday_on_track", "can I have some ice cream?", "named_food", "holdout"),
    ("H03", "midday_on_track", "show my food log", "log_list", "holdout"),
    ("H04", "evening_room", "how's my day looking?", "status", "holdout"),
    ("H05", "night_under", "should I go for a walk now?", "walk_now", "holdout"),
    ("H06", "afternoon_behind", "can I have a 250 kcal snack?", "fit", "holdout"),
    ("H07", "late_over", "what should I eat next?", "what_eat", "holdout"),
    ("H08", "morning_start", "how many calories can I eat today?", "cal_left", "holdout"),
    ("H09", "stale_steps", "I'm behind on steps, give me something to add steps", "steps_help", "holdout"),
    ("H10", "late_over", "give me a workout to burn more calories", "workout", "holdout"),
    # FRESH set (F01-F16): written after the H set was seen, never run while tuning. Two cases per weakness found in H
    # (an old step count that is not mentioned; a status answer that leaves out calories or steps) plus regression cases.
    ("F01", "lunch_low", "how am I doing?", "status", "fresh"),
    ("F02", "lunch_low", "how many calories do I have left?", "cal_left", "fresh"),
    ("F03", "lunch_low", "can I have a 600 kcal lunch?", "fit", "fresh"),
    ("F04", "evening_over", "how's today going?", "status", "fresh"),
    ("F05", "evening_over", "is 300 kcal too much for a snack?", "fit", "fresh"),
    ("F06", "evening_over", "I'm behind on steps, what can I do?", "steps_help", "fresh"),
    ("F07", "stale_evening", "where am I on steps?", "steps_left", "fresh"),
    ("F08", "stale_evening", "how am I doing today?", "status", "fresh"),
    ("F09", "stale_evening", "should I walk more tonight?", "walk_now", "fresh"),
    ("F10", "post_goal_morning", "am I on track?", "status", "fresh"),
    ("F11", "post_goal_morning", "what should I eat for breakfast?", "what_eat", "fresh"),
    ("F12", "post_goal_morning", "do I need more steps?", "steps_help", "fresh"),
    ("F13", "lunch_low", "give me a quick workout", "workout", "fresh"),
    ("F14", "evening_over", "can I have some chocolate?", "named_food", "fresh"),
    ("F15", "stale_evening", "how many steps do I have left?", "steps_left", "fresh"),
    ("F16", "lunch_low", "what's my protein like?", "protein", "fresh"),
    # THIRD set (G01-G16): written before the three weaknesses of the F set were fixed; run once after the fixes.
    ("G01", "midmorning_ahead", "how am I doing?", "status", "fresh2"),
    ("G02", "midmorning_ahead", "how many calories can I still eat today?", "cal_left", "fresh2"),
    ("G03", "midmorning_ahead", "where am I on calories?", "cal_left", "fresh2"),
    ("G04", "early_hungry", "what should I have for breakfast?", "what_eat", "fresh2"),
    ("G05", "early_hungry", "how's my day looking?", "status", "fresh2"),
    ("G06", "early_hungry", "hi", "greeting", "fresh2"),
    ("G07", "afternoon_stale", "where am I on steps?", "steps_left", "fresh2"),
    ("G08", "afternoon_stale", "how are my steps?", "steps_left", "fresh2"),
    ("G09", "afternoon_stale", "is 400 kcal too much for dinner?", "fit", "fresh2"),
    ("G10", "afternoon_stale", "am I on track?", "status", "fresh2"),
    ("G11", "evening_perfect", "is 200 kcal too much for a snack?", "fit", "fresh2"),
    ("G12", "evening_perfect", "how am I doing today?", "status", "fresh2"),
    ("G13", "evening_perfect", "can I have a 150 kcal snack?", "fit", "fresh2"),
    ("G14", "evening_perfect", "should I walk more tonight?", "walk_now", "fresh2"),
    ("G15", "midmorning_ahead", "I'm behind on steps, what can I do?", "steps_help", "fresh2"),
    ("G16", "early_hungry", "give me a workout to burn calories", "workout", "fresh2"),
    # phase and day-left probes (added 2026-10-05 after the user asked for a phase-aware, day-left-aware agent, still before G ran)
    ("G17", "afternoon_stale", "should I eat more today?", "phase_eat", "fresh2"),          # Fat Loss, under plan
    ("G18", "midmorning_ahead", "should I eat less today?", "phase_eat", "fresh2"),         # Muscle Gain, under plan
    ("G19", "early_hungry", "how much of the day is left?", "day_left", "fresh2"),
    ("G20", "evening_perfect", "how much of the day is left?", "day_left", "fresh2"),
]

INVENTORY = [("Greek yogurt", 120, 17, "snack"), ("Protein shake", 160, 30, "drink"), ("Banana", 105, 1, "snack"),
             ("Chicken breast", 165, 31, "meal"), ("Rice bowl", 350, 8, "meal"), ("Almonds", 160, 6, "snack"),
             ("Boiled egg", 78, 6, "snack"), ("Tuna can", 120, 26, "snack"), ("Cottage cheese", 180, 24, "snack")]
