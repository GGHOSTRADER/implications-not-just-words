"""Shared by labels v3 of the checkpoint comment and the morning briefing (2026-10-05).

Three things labels v2 lacked, found in a real message (the 12:00 check of 2026-10-05):
  * steps are WALKING steps: the model suggested "light squats or lunges at home to add to your steps". Exercises add no
    steps, and training comes only from the user's plan (blueprint), never from the model;
  * the weather rule was only rain/not rain. Rain, great heat and strong wind send the steps indoors (treadmill, stairs,
    pacing, a covered walk); a sunny day is not an automatic "go outside" (a walk or the treadmill are equals); unknown
    weather leans indoors;
  * a sense of time: how far into the day the message is, how much of it remains, which meals are still ahead, and
    when movement ideas stop making sense.
Pure functions, no database, no model.
"""
from __future__ import annotations

import re

HOT_C = 32            # at or above this, steps move indoors (or to after sunset)
WINDY_KPH = 40
_NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                 "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                 "twenty-one", "twenty-two", "twenty-three", "twenty-four"]

# things that are not food for calorie or protein purposes (the real inventory had "Fish Oil", 45 kcal, 0 g protein)
_SUPPLEMENT = re.compile(r"\b(oil|vitamin|vitamins|creatine|supplement|capsules?|tablets?|omega|probiotic|multivitamin|"
                         r"electrolytes?|bcaa)\b", re.I)


def is_supplement(name: str, kcal: float, protein: float) -> bool:
    return bool(_SUPPLEMENT.search(name or "")) or (kcal < 60 and protein < 3)


def weather_verdict(weather: dict | None) -> tuple[str, str]:
    """('fine' | 'indoors' | 'unknown', reason). Rain or snow (WMO code 51+, 0.5 mm), great heat and strong wind send
    the steps indoors; unknown weather leans indoors; otherwise a walk or a treadmill are equals."""
    if not weather:
        return "unknown", "no weather data"
    code, precip = weather.get("weather_code"), weather.get("precipitation_mm") or 0
    if (code is not None and code >= 51) or precip >= 0.5:
        return "indoors", "rain"
    if (weather.get("temp_c") or 0) >= HOT_C:
        return "indoors", "great heat"
    if (weather.get("wind_kph") or 0) >= WINDY_KPH:
        return "indoors", "strong wind"
    return "fine", ""


def steps_line(weather: dict | None) -> str:
    """The one line that tells the model how to suggest steps today. Walking only."""
    verdict, reason = weather_verdict(weather)
    ways = "a walk, the stairs, pacing during calls"
    if verdict == "fine":
        return (f"Ways to add steps: {ways}, a walk outdoors or on a treadmill. A sunny day is not an automatic reason "
                "to go outside: a treadmill is just as good.")
    if verdict == "indoors":
        return (f"Ways to add steps indoors ({reason} today): a treadmill, the stairs, pacing during calls, a walk through "
                "a covered place such as a mall. Do not send the user outside.")
    return ("Weather unknown. Ways to add steps: a treadmill, the stairs, pacing during calls, an indoor walk; "
            "an outdoor walk only if the weather allows.")


NO_EXERCISES = ("Steps are walking steps. Never name exercises, sets or reps (squats, lunges, circuits, yoga): they add no "
                "steps. Training comes only from the user's plan; never add, change or invent a workout.")


def hours_left(end_hour: int, end_minute: int = 0) -> int:
    """Whole hours left before midnight when the check fires (the 23:59 check fires at the end of the day)."""
    h = end_hour + (1 if end_minute >= 30 else 0)
    return max(0, 24 - h)


def time_line(end_hour: int, end_minute: int, hour_text: str, is_last: bool) -> str:
    """How far into the day we are, what is still ahead, and what no longer makes sense."""
    if is_last:
        return ("Time: it is the end of the day. Nothing is left to plan for today: no meals, no walks, no workouts.")
    h = end_hour + (1 if end_minute >= 30 else 0)
    left = hours_left(end_hour, end_minute)
    if h <= 12:
        ahead = "lunch, an afternoon snack and dinner are still ahead"
    elif h <= 16:
        ahead = "an afternoon snack and dinner are still ahead; lunch and breakfast are over"
    elif h <= 19:
        ahead = "dinner and perhaps a light evening snack are still ahead; breakfast and lunch are over"
    elif h <= 21:
        ahead = "at most a light evening snack is left; breakfast, lunch and dinner are over"
    else:
        ahead = "no more meals today; do not suggest food"
    if h <= 18:
        move = "there is still time to walk today"
    elif h <= 20:
        move = "a walk or a treadmill session is still possible this evening"
    elif h <= 21:
        move = "only a short indoor walk still makes sense"
    else:
        move = "it is too late for movement ideas; time to wind down"
    return (f"Time: it is about {hour_text}, and about {_NUMBER_WORDS[min(left, 24)]} hours of the day remain. "
            f"Meals: {ahead}. Movement: {move}. Suggest only what still fits the time of day.")
