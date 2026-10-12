"""The prompt builders and answer checks the evaluation calls, extracted verbatim from the application's
background_checks.py (commit 19077f5). The application is not part of this release; every other function of the original
file is left out. Names are unchanged so that the evaluation code runs as written and the version fingerprints match."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Default checkpoint hours and an equal cumulative split (25/50/75/100 for 4 checkpoints).
# The 4th (last) hour is always 23 — the fixed end-of-day block, not user-
# schedulable (see _load_checkpoints' label logic below, which renders it as
# "23:59"). dispatch_tick's fire-gate calls _checkpoint_target_minute(h) so
# hour 23 only fires at :59, not the top of the hour.
_DEFAULT_CHECKPOINT_HOURS = [12, 15, 19, 23]


def _load_checkpoints() -> list[tuple[str, int, float, float, float]]:
    """Return the checkpoint schedule as [(label, hour, steps_frac, cal_frac, prot_frac)].

    Hours come from preferences (checkpoint_hours, default 12/15/19/22).
    Per-metric fractions come from preferences:
      - checkpoint_distribution_steps / _calories / _protein: list of cumulative %
      - If any is missing or "equal", that metric gets an even split.
    Legacy single checkpoint_distribution key is also honoured for all three metrics.
    """
    hours = list(_DEFAULT_CHECKPOINT_HOURS)
    prefs = {}
    try:
        from define_agent import load_preferences
        prefs = load_preferences()
        configured = prefs.get("checkpoint_hours")
        if configured:
            hours = sorted(int(h) for h in configured)
    except Exception:
        pass

    n = len(hours)

    def _fracs(key):
        val = prefs.get(key, prefs.get("checkpoint_distribution", "equal"))
        if isinstance(val, list) and len(val) == n:
            # Stored as block % per checkpoint (e.g. [25,25,25,25]) — convert to cumulative fracs
            cum = 0.0
            result = []
            for pct in val:
                cum += pct / 100.0
                result.append(round(cum, 6))
            # Fire time closes a block — the LAST checkpoint's fire time is
            # what closes the final block, but nothing closes the window
            # after it. Force the last cumulative fraction to exactly 1.0
            # (100%) regardless of what the stored distribution summed to,
            # so anything logged after the last checkpoint fires through
            # midnight always falls inside that final block instead of a
            # dead zone no block ever accounts for.
            if result:
                result[-1] = 1.0
            return result
        return [(i + 1) / n for i in range(n)]

    steps_fracs = _fracs("checkpoint_distribution_steps")
    cal_fracs   = _fracs("checkpoint_distribution_calories")
    prot_fracs  = _fracs("checkpoint_distribution_protein")

    # The last configured checkpoint is the fixed final block — always fires
    # at :59 (scheduler.py registers its cron job at minute=59), not :00 like
    # every other checkpoint. Reflect that in the label everywhere it's used
    # (Telegram messages, LLM prompt context, home-screen/day-detail display)
    # rather than patching each call site individually.
    last_hour = hours[-1] if hours else None
    return [
        (f"{h:02d}:{'59' if h == last_hour else '00'}", h, sf, cf, pf)
        for h, sf, cf, pf in zip(hours, steps_fracs, cal_fracs, prot_fracs)
    ]


def _checkpoint_status(dev_abs: float) -> int:
    """1 if |dev| <= 5%, 2 if <= 20%, 3 if > 20%."""
    if dev_abs <= 0.05:
        return 1
    if dev_abs <= 0.20:
        return 2
    return 3


def _metric_block(block_live: float, block_expected: float, daily: float,
                  cumulative_expected: float, remaining_budget: float, remaining_frac: float = 0.0,
                  block_frac: float = 0.0) -> dict:
    """Compute one metric's deviation/status/direction/daily%/next-block carry.

    block_live       = live amount consumed/walked IN this block only
    block_expected   = what was expected for this block (remaining-budget-forward)
    cumulative_expected = daily * frac — used only for daily_pct display
    remaining_budget = daily_target - live_cumulative_BEFORE this block
                       nextb = remaining_budget - block_live (total left for all future blocks)
    remaining_frac   = unused, kept for call-site compatibility
    block_frac       = this block's own slice of the day's goal (e.g. 0.25 for
                       an even 4-way split) — the "% share of the daily goal"
                       shown on the home-screen card, distinct from daily_pct
                       (cumulative percent-through-the-day at this checkpoint).
    """
    if block_expected > 0:
        dev_raw = (block_live / block_expected) - 1.0
    elif block_live > 0:
        # Nothing expected but something happened — flag as over
        dev_raw = 1.0
    else:
        dev_raw = 0.0
    dev_abs = abs(dev_raw)

    # Carry = how much of this block's target was not consumed (block_expected - block_live)
    # Negative when user exceeded the block target
    nextb_raw = round(block_expected - block_live)
    nextb = nextb_raw

    return {
        "live": block_live,
        "expected": block_expected,
        "dev_raw": round(dev_raw, 4),
        "dev_abs": round(dev_abs, 4),
        "status": _checkpoint_status(dev_abs),
        "direction": "over target" if dev_raw > 0 else "under target",
        "daily_pct": round((cumulative_expected / daily), 4) if daily else 0.0,
        "block_pct": round(block_frac, 4),
        "nextb": nextb,
        "nextb_raw": nextb_raw,
    }


def _fc_format_context(d: dict) -> str:
    """The original ('numbers') context text, from _fc_context_data. Pinned by tests/golden/checkpoint_context.json."""
    parts: list[str] = []
    _w = d.get("weather")
    try:                                   # as in the original: a malformed weather dict skips the line, nothing more
        if _w:
            parts.append(
                f"Weather now: {_w['description']}, {_w['temp_c']}C, "
                f"wind {_w['wind_kph']} kph, precip {_w['precipitation_mm']}mm"
            )
    except Exception:
        pass
    # Outdoor-walking suitability decided here so the LLM never assumes good weather.
    _code = (_w or {}).get("weather_code")
    _precip = (_w or {}).get("precipitation_mm") or 0
    if _w is None:
        parts.append("Outdoor walking: UNKNOWN — weather unavailable; do NOT assume it's nice, keep any movement tip weather-neutral")
    elif (_code is not None and _code >= 51) or _precip >= 0.5:
        parts.append("Outdoor walking: NOT SUITABLE — rain/snow/wet; suggest INDOOR movement only, never outside")
    else:
        parts.append("Outdoor walking: SUITABLE — an outdoor walk is fine")

    if d.get("food_entries"):
        eaten = "; ".join(
            f"{e['food']} ({round(e['calories'])}kcal/{round(e['protein'])}gP)"
            for e in d["food_entries"]
        )
        parts.append(f"Food logged today: {eaten}")
    else:
        parts.append("Food logged today: NOTHING logged yet (could be forgot-to-log or simply hasn't eaten yet)")

    try:                                   # as in the original: a bad inventory row skips the line
        if d.get("inventory"):
            inv = "; ".join(f"{name} ({round(kcal)}kcal/{round(prot)}gP)" for name, kcal, prot in d["inventory"])
            parts.append(f"Inventory available for suggestions: {inv}")
    except Exception:
        pass

    if d.get("food_likes"):
        parts.append(f"Food likes: {d['food_likes']}")
    if d.get("food_dislikes"):
        parts.append(f"Food dislikes: {d['food_dislikes']}")

    w = d.get("workout")
    if w:
        if w["kind"] == "cardio":
            parts.append("Workout today: cardio day (logged through the Cardio tab)")
        elif w["kind"] == "session":
            parts.append(f"Workout today: {w['type']} ({'logged' if w['logged'] else 'not done/logged yet'})")
        else:
            parts.append("Workout today: rest day")

    if d.get("morning_feedback"):
        parts.append(f"Morning briefing feedback from user: {d['morning_feedback']}")

    parts.append(f"Phase: {d.get('phase', '')}")
    return "\n".join(parts)


def _checkpoint_prompt(cp: dict, ctx_text: str, phase: str) -> str:
    """The checkpoint comment's prompt as the original ('numbers') design builds it. Pure apart from the identity
    header, with the phase passed in, so the evaluation (evals/) sends exactly what production sends. The text is
    pinned by tests/golden/checkpoint_prompts.json (captured from the code before this was extracted)."""
    from agent import _identity_header
    cal, prot, stp = cp["calories"], cp["protein"], cp["steps"]
    ord_ = cp["ordinal"]
    is_last = cp.get("is_last", False)

    def _m(name: str, m: dict, unit: str) -> str:
        raw = m.get("nextb_raw", m["nextb"])
        carry_note = f"target next block {m['nextb']}{unit}"
        if raw < 0:
            carry_note += f" (raw carry {raw}{unit} — user is {abs(raw)}{unit} over daily budget)"
        return (f"{name}: live {round(m['live'])}{unit} vs expected {round(m['expected'])}{unit} "
                f"(status {m['status']}, {m['direction']}); {carry_note}")

    metrics = "\n".join([_m("Calories", cal, " kcal"), _m("Protein", prot, "g"), _m("Steps", stp, "")])

    # Exact-match against the shared constant — phase is a closed set of three
    # values the UI writes verbatim; see the note in
    # _create_checkpoint_strategy_decision for why the old substring matching
    # was removed (2026-07-30).
    from day_scoring import PHASE_FAT_LOSS as _PHASE_FAT_LOSS
    phase = phase or ""
    is_fat_loss = phase == _PHASE_FAT_LOSS
    no_food = round(cal["live"]) == 0 and round(prot["live"]) == 0

    notes = []
    if no_food:
        notes.append(
            "CRITICAL FRAMING: the user has logged NO food yet. The calorie/protein 'shortfalls' are almost "
            "certainly just that they haven't eaten yet OR forgot to log — this is NOT 'falling behind' or "
            "failing. Do NOT scold or say they're way off on nutrition. Instead ask plainly whether they "
            "forgot to log or simply haven't eaten yet, and nudge them to log or grab a meal."
        )
    if is_fat_loss:
        notes.append(
            "PHASE NOTE (fat loss): being UNDER on calories is consistent with — even good for — fat loss. "
            "Never tell the user low calories means they're 'not doing it right'. In fat loss the things that "
            "matter are hitting PROTEIN and not OVER-eating; being under early is fine."
        )
    extra = ("\nImportant context notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""

    # A checkpoint FIRES at the moment its block CLOSES — it is always
    # evaluating a block that just ended, never one that's beginning. Stating
    # only the fire hour (e.g. "checkpoint at 12:00") left the LLM to guess
    # whether that block was starting or ending, and it sometimes guessed
    # wrong (confirmed real bug: told a user they were "still in the early
    # stages" of a block that had, by definition, just closed). Spelling out
    # the actual window removes the ambiguity entirely.
    block_window = f"{cp['block_start_hour']:02d}:00–{cp['label']}"

    if is_last:
        # Final checkpoint of the day — short retrospective, morning briefing will revisit
        prompt = (
            f"{_identity_header()}\n\n"
            f"This is the user's {ord_} (last) checkpoint of the day. The block that just CLOSED "
            f"ran {block_window} — you are summarizing what happened during that block, which has "
            f"already ended, not describing something still in progress.\n\n"
            f"Metrics (status 1 = within 5% of pace, 2 = 5-20% off, 3 = over 20% off):\n{metrics}\n\n"
            f"Context:\n{ctx_text}\n"
            f"{extra}\n"
            "Write 2-3 sentences: a quick honest summary of how the day went on each metric. "
            "Keep it brief — the morning briefing will do the full day review tomorrow. "
            "No greeting, no sign-off, no headings, no bullet points."
        )
    else:
        prompt = (
            f"{_identity_header()}\n\n"
            f"This is the user's {ord_} checkpoint of the day. The block that just CLOSED ran "
            f"{block_window} — you are summarizing what happened during that block, which has "
            f"already ended, not describing something still in progress. Never say things like "
            f"\"you're still in the early stages of this block\" — the block is over; frame any "
            f"shortfall as what happened during {block_window}, and any forward-looking advice as "
            f"being for the NEXT block, not this one.\n\n"
            f"Metrics (status 1 = within 5% of pace, 2 = 5-20% off, 3 = over 20% off):\n{metrics}\n\n"
            f"Context:\n{ctx_text}\n"
            f"{extra}\n"
            "Write 1-2 short paragraphs (3-4 sentences each): a brief, honest analysis of where they stand, "
            "concrete recommendations, and a line of motivation. Guidance by status:\n"
            "- status 1: affirm they are close to target, keep it going.\n"
            "- status 2: correctable. If OVER target suggest lighter choices for the next meal (or an easier next "
            "block for steps); if UNDER target suggest richer choices — protein-rich foods for protein, more caloric "
            "foods for calories. Prefer specific items from the inventory that fit the budget and the user's preferences.\n"
            "- status 3: serious — the system will separately ask how to redistribute the amount; do NOT list options "
            "yourself, just explain honestly and motivate. BUT obey the context notes above (e.g. a calorie/protein "
            "status 3 caused only by not-yet-logged food is NOT serious — treat it as a logging/eating nudge).\n"
            "For ANY step or walking suggestion, STRICTLY follow the 'Outdoor walking' line in the context — never "
            "suggest going outside when it says NOT SUITABLE or UNKNOWN; suggest indoor movement instead.\n"
            "You may mention specific foods and rough amounts. No greeting, no sign-off, no headings."
        )
    return prompt


def _checkpoint_label_input(cp: dict, goals: dict, live: dict, checkpoints: list, ctx_data: dict) -> dict:
    """The input checkpoint_labels.py describes in words: this block's figures (from _compute_checkpoint), the day's
    cumulative figures and targets, the schedule's cumulative fractions at this checkpoint, and the context data."""
    row = checkpoints[cp["idx"]]
    sf, cf, pf = (row[2], row[3], row[4]) if len(row) == 5 else (row[2], row[2], row[2])
    label = cp.get("label") or ""
    end_minute = int(label.split(":")[1]) if ":" in label else 0
    m = lambda key: {"block_live": cp[key]["live"], "block_expected": cp[key]["expected"]}
    return {
        "ordinal": cp["ordinal"], "is_last": cp.get("is_last", False),
        "phase": ctx_data.get("phase") or goals.get("phase", ""),
        "block_start_hour": cp["block_start_hour"], "end_hour": cp["hour"], "end_minute": end_minute,
        "metrics": {
            "calories": {**m("calories"), "daily_target": cp.get("cal_target") or goals.get("calories_target", 0),
                         "cum_live": live["calories"], "cum_frac": cf},
            "protein": {**m("protein"), "daily_target": goals.get("protein_target", 150),
                        "cum_live": live["protein"], "cum_frac": pf},
            "steps": {**m("steps"), "daily_target": goals.get("step_goal", 17000),
                      "cum_live": live["steps"], "cum_frac": sf},
        },
        "context": ctx_data,
    }


def _checkpoint_comment_labels(inp: dict, guards: bool = True) -> str:
    """The labels design: the checkpoint described in words by checkpoint_labels.py; with guards, the answer is
    checked (numbers, outdoor rule, scolding, 'still early', direction, lists, dislikes) and redrafted once."""
    import checkpoint_labels
    from agent import _identity_header, _clean_response
    from langchain_core.messages import HumanMessage as _HM
    import llm_registry
    ord_ = inp.get("ordinal", "")
    prompt = checkpoint_labels.prompt(inp, _identity_header())
    try:
        model = llm_registry.get("briefing_nlg")
        raw = getattr(model.invoke([_HM(content=prompt)]), "content", "")
        result = _clean_response(raw if isinstance(raw, str) else "").strip()
        problems = checkpoint_labels.problems(result, inp) if guards else []
        if problems:
            redraft = (prompt + "\n\nYour previous draft had these problems: " + "; ".join(problems) + ".\n"
                       "Redraft it fixing them. Same format.")
            raw = getattr(model.invoke([_HM(content=redraft)]), "content", "")
            result = _clean_response(raw if isinstance(raw, str) else "").strip()
        log.info(f"[checkpoint {ord_}] LLM comment (labels) generated ({len(result)} chars, "
                 f"{'redrafted' if problems else 'first draft'})")
        return result
    except Exception as e:
        log.warning(f"[checkpoint {ord_}] comment LLM (labels) failed: {e}")
        return ""


def _labels_answer(prompt: str, check, model) -> str:
    """One answer over a labels prompt (the v2 designs). With `check` (answer -> list of problems), the answer is
    checked and redrafted once with the problems named. v1 keeps its own functions so its measured runs stay valid."""
    from agent import _clean_response
    from langchain_core.messages import HumanMessage as _HM
    text = _clean_response(getattr(model.invoke([_HM(content=prompt)]), "content", "") or "").strip()
    problems = check(text) if check else []
    if problems:
        redraft = (prompt + "\n\nYour previous draft had these problems: " + "; ".join(problems) + ".\n"
                   "Redraft it fixing them. Same format.")
        text = _clean_response(getattr(model.invoke([_HM(content=redraft)]), "content", "") or "").strip()
    return text


def _checkpoint_comment_labels_v3(inp: dict, guards: bool = True) -> str:
    """Labels v3 (CHECKPOINT_MODE=labels3 / labels3_noguard): steps are walking steps and no exercises are named,
    weather-aware indoor/outdoor ideas, a sense of the time of day, no supplements in the food list."""
    import checkpoint_labels_v3 as cl3
    from agent import _identity_header
    import llm_registry
    ord_ = inp.get("ordinal", "")
    prompt = cl3.prompt_v3(inp, _identity_header())
    try:
        result = _labels_answer(prompt, (lambda t: cl3.problems_v3(t, inp)) if guards else None,
                                llm_registry.get("briefing_nlg"))
        log.info(f"[checkpoint {ord_}] LLM comment (labels v3) generated ({len(result)} chars)")
        return result
    except Exception as e:
        log.warning(f"[checkpoint {ord_}] comment LLM failed (labels v3): {e}")
        return ""


def _checkpoint_comment_labels_v2(inp: dict, guards: bool = True) -> str:
    """Labels v2 (CHECKPOINT_MODE=labels2 / labels2_noguard): the day's last check holds only how the day went, the
    size words are kept as given, and the checks tell a request for more food from a description of one."""
    import checkpoint_labels
    from agent import _identity_header
    import llm_registry
    ord_ = inp.get("ordinal", "")
    prompt = checkpoint_labels.prompt_v2(inp, _identity_header())
    try:
        result = _labels_answer(prompt, (lambda t: checkpoint_labels.problems_v2(t, inp)) if guards else None,
                                llm_registry.get("briefing_nlg"))
        log.info(f"[checkpoint {ord_}] LLM comment (labels v2) generated ({len(result)} chars)")
        return result
    except Exception as e:
        log.warning(f"[checkpoint {ord_}] comment LLM failed (labels v2): {e}")
        return ""


def _checkpoint_comment(cp: dict, ctx_text: str) -> str:
    """LLM analysis/recommendation for any checkpoint. Prompt adapts to position in day."""
    from agent import _clean_response, _load_goals
    from langchain_core.messages import HumanMessage as _HM
    import llm_registry
    ord_ = cp["ordinal"]
    prompt = _checkpoint_prompt(cp, ctx_text, _load_goals().get("phase", "") or "")
    try:
        model = llm_registry.get("briefing_nlg")
        resp = model.invoke([_HM(content=prompt)])
        raw = getattr(resp, "content", "")
        result = _clean_response(raw if isinstance(raw, str) else "").strip()
        log.info(f"[checkpoint {ord_}] LLM comment generated ({len(result)} chars)")
        return result
    except Exception as e:
        log.warning(f"[checkpoint {ord_}] comment LLM failed: {e}")
        return ""


def _classify_steps(steps, goal):
    """Bucket yesterday's step performance."""
    if steps is None:
        return "no data"
    delta = steps - goal
    if delta >= 5000:   return "substantially above"
    if delta >= 3000:   return "above"
    if delta >= 1000:   return "a little above"
    if delta >= 0:      return "hit"
    if delta >= -1000:  return "a little below"
    if delta >= -3000:  return "below"
    if delta >= -5000:  return "substantially below"
    return "very below"


def _classify_calories(calories, target):
    """Bucket yesterday's calorie performance."""
    if not target:
        return "no target"
    pct = abs(calories - target) / target
    over = calories > target
    if pct <= 0.03:
        return "on target"
    if pct <= 0.08:
        return f"a bit {'over' if over else 'under'}"
    if pct <= 0.15:
        return f"off — needs focus today ({'over' if over else 'under'})"
    return f"very off ({'over' if over else 'under'}) — something happened, reset today"


def _briefing_yesterday_prompt(phase, yesterday_step_class, yesterday_cal_class, yesterday_protein_over,
                               yesterday_was_workout_day, yesterday_workout_type, yesterday_workout_logged,
                               yesterday_weight_logged) -> str:
    """The briefing's 'yesterday reflection' prompt as the original ('numbers') design builds it. Extracted unchanged
    from _fire_morning_briefing so the evaluation (evals/) sends exactly what production sends; pinned by
    tests/golden/briefing_prompts.json."""
    from agent import _identity_header
    _ys_prompt = (
        f"{_identity_header()}\n\n"
        f"Phase: {phase}\n\n"
        f"Yesterday's performance:\n"
        f"- Steps: {yesterday_step_class}\n"
        f"- Calories: {yesterday_cal_class}\n"
        f"- Protein: {'over' if yesterday_protein_over else 'under'} goal\n"
    )
    if yesterday_was_workout_day:
        _ys_prompt += f"- Workout ({yesterday_workout_type}): {'completed' if yesterday_workout_logged else 'not logged'}\n"
    # Unconditional now, matching the deterministic yesterday block above:
    # there is no such thing as a non-weigh-in day to stay silent about.
    _ys_prompt += f"- Weight log: {'recorded' if yesterday_weight_logged else 'missed'}\n"
    _ys_prompt += (
        "\nStart with a short, warm greeting that uses the user's name exactly once. "
        "Then write 2-3 short sentences reflecting on yesterday. "
        "Be specific about what went well or what needs improvement based on the classifications above. "
        "Tone should match the phase and be direct. "
        "Do NOT repeat any numbers — they are shown above. No bullet points, no sign-off."
    )
    return _ys_prompt


def _briefing_today_prompt(phase, day, now, workout_today, weather, cal_is_adjusted, cal_value,
                           steps_is_adjusted, steps_value, food_likes, carryover_cal_line) -> str:
    """The briefing's 'today plan' prompt as the original ('numbers') design builds it. Extracted unchanged from
    _fire_morning_briefing; pinned by tests/golden/briefing_prompts.json."""
    from agent import _identity_header
    # Decide outdoor suitability in Python so the LLM never assumes good weather.
    # Bad WMO codes: 51-67 drizzle/rain, 71-77 snow, 80-99 showers/thunderstorm.
    _weather_str = "unknown"
    if weather:
        _weather_str = f"{weather['description']}, {weather['temp_c']}°C"
        if weather.get("precipitation_mm"):
            _weather_str += f", {weather['precipitation_mm']}mm rain"
        if weather.get("wind_kph"):
            _weather_str += f", {weather['wind_kph']} km/h wind"
    _code = (weather or {}).get("weather_code")
    _precip = (weather or {}).get("precipitation_mm") or 0
    if weather is None:
        _outdoor = "UNKNOWN — weather data is unavailable, so do NOT assume it is nice; keep the movement tip weather-neutral (e.g. a walk if conditions allow) and do not specifically push outdoor activity"
    elif (_code is not None and _code >= 51) or _precip >= 0.5:
        _outdoor = "NOT SUITABLE — it is raining/snowing/poor; recommend INDOOR movement only (treadmill, mall walk, home workout). Do NOT suggest going outside"
    else:
        _outdoor = "SUITABLE — clear enough; an outdoor walk is fine"
    is_weekend = now.weekday() >= 5
    return (
        f"{_identity_header()}\n\n"
        f"Phase: {phase}\n"
        f"Today: {day} ({'weekend' if is_weekend else 'weekday'})\n"
        f"Workout: {workout_today}\n"
        f"Weather: {_weather_str}\n"
        f"Outdoor walking: {_outdoor}\n"
        f"Calories today: {'adjusted (' + cal_value + ')' if cal_is_adjusted else 'base (' + cal_value + ')'}\n"
        f"Steps today: {'adjusted (' + steps_value + ')' if steps_is_adjusted else 'base (' + steps_value + ')'}\n"
        + (f"Food likes: {food_likes}\n" if food_likes else "")
        + (f"Carryover: {carryover_cal_line}\n" if carryover_cal_line else "")
        + "\nWrite 2-3 short sentences with practical suggestions for today:\n"
        "1. A light food suggestion for starting the day (based on food likes if available, within phase goals)\n"
        "2. A step/movement tip that STRICTLY follows the 'Outdoor walking' guidance above — never suggest going outside when it says NOT SUITABLE or UNKNOWN\n"
        "3. A brief workout or rest day note\n"
        "Do NOT greet the user or use their name, and do NOT start with 'Hey'/'Hi'/'Hello' — this continues the briefing, so go straight into the suggestions. "
        "Do NOT repeat numbers. No bullet points, no sign-off."
    )


def _briefing_label_input(phase, day, now, ys, base_steps, base_cal, adj_today, pending_carryover, workout_today,
                          weather, food_likes, y_workout_day, y_workout_type, y_workout_logged, y_weight_logged) -> dict:
    """What briefing_labels.py describes in words: yesterday's raw results and today's base and adjusted targets,
    from the same values _fire_morning_briefing uses for its original prompts."""
    strategy = (adj_today or {}).get("strategy", "")
    cal_today = (adj_today or {}).get("adjusted_target", base_cal) if strategy in ("calories", "calories_boost") else base_cal
    if strategy == "steps":
        steps_today = base_steps + (adj_today or {}).get("extra_steps", 0)
    elif strategy == "steps_reduce":
        steps_today = (adj_today or {}).get("reduced_steps", base_steps)
    else:
        steps_today = base_steps
    if pending_carryover:
        carry = {"pending": True, "direction": pending_carryover.get("direction", "over")}
    elif strategy:
        carry = {"strategy": strategy}
    else:
        carry = None
    yesterday = None
    if ys:
        yesterday = {"steps": ys.steps, "step_goal": ys.step_goal or base_steps, "calories": ys.calories,
                     "calories_target": ys.calories_target or base_cal, "protein": ys.protein,
                     "protein_target": ys.protein_target, "workout_day": y_workout_day, "workout_type": y_workout_type,
                     "workout_logged": y_workout_logged, "weight_logged": y_weight_logged}
    else:
        yesterday = {"workout_day": y_workout_day, "workout_type": y_workout_type, "workout_logged": y_workout_logged,
                     "weight_logged": y_weight_logged}
    return {"phase": phase, "day": day, "weekend": now.weekday() >= 5, "yesterday": yesterday,
            "today": {"workout": workout_today, "weather": weather, "cal_base": base_cal, "cal_today": cal_today,
                      "steps_base": base_steps, "steps_today": steps_today, "carry": carry, "food_likes": food_likes}}


def _briefing_labels_part(part: str, inp: dict, model, guards: bool = True) -> str:
    """One briefing section in the labels design; with guards, checked (numbers, lists, greeting, outdoor rule) and
    redrafted once."""
    import briefing_labels
    from agent import _identity_header, _clean_response
    from langchain_core.messages import HumanMessage as _HM
    build = briefing_labels.yesterday_prompt if part == "yesterday" else briefing_labels.today_prompt
    prompt = build(inp, _identity_header())
    text = _clean_response(getattr(model.invoke([_HM(content=prompt)]), "content", "") or "")
    problems = briefing_labels.problems(part, text, inp) if guards else []
    if problems:
        redraft = (prompt + "\n\nYour previous draft had these problems: " + "; ".join(problems) + ".\n"
                   "Redraft it fixing them. Same format.")
        text = _clean_response(getattr(model.invoke([_HM(content=redraft)]), "content", "") or "")
    return text


def _briefing_labels_part_v4(part: str, inp: dict, model, guards: bool = True) -> str:
    """One briefing section in labels v4 (BRIEFING_MODE=labels4 / labels4_noguard): the yesterday part ranks the big
    misses by size and names them all in one sentence; the today part is v3's."""
    import briefing_labels_v3 as bl3
    import briefing_labels_v4 as bl4
    from agent import _identity_header
    if part == "yesterday":
        return _labels_answer(bl4.yesterday_prompt_v4(inp, _identity_header()),
                              (lambda t: bl4.problems_v4_yesterday(t, inp)) if guards else None, model)
    return _labels_answer(bl3.today_prompt_v3(inp, _identity_header()),
                          (lambda t: bl3.problems_v3("today", t, inp)) if guards else None, model)


def _briefing_labels_part_v3(part: str, inp: dict, model, guards: bool = True) -> str:
    """One briefing section in labels v3 (BRIEFING_MODE=labels3 / labels3_noguard): yesterday as in v1, today with
    walking-only steps, weather-aware indoor/outdoor ideas and no invented workouts."""
    import briefing_labels
    import briefing_labels_v3 as bl3
    from agent import _identity_header
    build = briefing_labels.yesterday_prompt if part == "yesterday" else bl3.today_prompt_v3
    return _labels_answer(build(inp, _identity_header()),
                          (lambda t: bl3.problems_v3(part, t, inp)) if guards else None, model)


def _briefing_labels_part_v2(part: str, inp: dict, model, guards: bool = True) -> str:
    """One briefing section in labels v2 (BRIEFING_MODE=labels2 / labels2_noguard): the yesterday plan names every
    to-do item, a day with no workout plan says so, and a suggested duration is allowed."""
    import briefing_labels
    from agent import _identity_header
    build = briefing_labels.yesterday_prompt_v2 if part == "yesterday" else briefing_labels.today_prompt_v2
    return _labels_answer(build(inp, _identity_header()),
                          (lambda t: briefing_labels.problems_v2(part, t, inp)) if guards else None, model)
