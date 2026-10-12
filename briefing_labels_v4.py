"""Labels v4 of the morning briefing's YESTERDAY part (BRIEFING_MODE=labels4 / labels4_noguard, 2026-10-05).

v1's plan asked for "the first thing to do better today and how", and the model wrote exactly one thing even when
yesterday had two or three big misses (4 of the 5 failing cases in the v3 measurement). v2 asked for every item and made
four-to-five-item reflections too long. v4 lets code decide the shape:
  * misses of 10% or more are ranked BY SIZE (v1 ranked by phase weight, which put a 4% protein miss above a 16% calorie
    miss) and named TOGETHER in one sentence, however many there are;
  * the "how" is asked for the biggest one only;
  * when nothing went well that sentence is dropped, instead of the cold "nothing stood out" the model copied.
The today part is v3's (briefing_labels_v3.py, unchanged).
"""
from __future__ import annotations

import re

import briefing_labels as bl
import checkpoint_labels as cpl

BIG = 0.10                    # a miss of this size or more must be named


def misses(inp: dict) -> tuple[list[tuple[float, str, str]], list[str]]:
    """([(size, metric, words)] biggest first for misses of BIG or more, small things worth a mention)."""
    phase, y, _sw, steps_short, _cw, cal_dev, _pw, prot_short = bl._results(inp)
    big = []
    if steps_short >= BIG:
        big.append((steps_short, "steps", f"steps fell short ({cpl.gap_words(steps_short)} short)"))
    if prot_short >= BIG:
        big.append((prot_short, "protein", f"protein fell short ({cpl.gap_words(prot_short)} short)"))
    if abs(cal_dev) >= BIG:
        side = "over" if cal_dev > 0 else "under"
        big.append((abs(cal_dev), "calories", f"calories came in {side} the target ({cpl.gap_words(cal_dev)} {side})"))
    big.sort(key=lambda t: -t[0])
    small = [k for k in bl._to_do(inp, 2) if k not in [b[1] for b in big]]
    return big, small


def yesterday_view_v4(inp: dict) -> str:
    phase, y, steps_w, steps_short, cal_w, cal_dev, prot_w, prot_short = bl._results(inp)
    link = bl._carry_link((inp.get("today") or {}).get("carry")).get("calories", "")
    lines = [f"Phase: {phase}", "", "Yesterday:"]

    def line(title, words, meaning, extra=""):
        tail = "; ".join(x for x in (meaning, extra) if x)
        return f"{title}: {words}." + (f" For the goal: {tail}." if tail else "")

    lines.append(line("Steps", steps_w, bl._meaning(phase, "steps", steps_short)))
    lines.append(line("Calories", cal_w, bl._meaning(phase, "calories", cal_dev), link if abs(cal_dev) > 0.03 else ""))
    lines.append(line("Protein", prot_w, bl._meaning(phase, "protein", prot_short)))
    lines.append(f"Workout: {y.get('workout_type')}, " + ("done." if y.get("workout_logged") else "planned but not logged.")
                 if y.get("workout_day") else "Workout: rest day.")
    lines.append("Weigh-in: recorded." if y.get("weight_logged") else
                 "Weigh-in: missed; a daily weigh-in keeps the progress estimate reliable.")
    lines.append("")
    well = [n for n, ok in (("steps", steps_short == 0 and steps_w != "no data"),
                            ("calories", abs(cal_dev) <= 0.03 and cal_w != "no data"),
                            ("protein", prot_short == 0 and prot_w != "no data"),
                            ("the workout", y.get("workout_day") and y.get("workout_logged")),
                            ("the weigh-in", y.get("weight_logged"))) if ok]
    big, small = misses(inp)
    if well:
        lines.append("Went well: " + ", ".join(well) + ".")
    if big:
        lines.append("Big misses, biggest first: " + "; ".join(w for _s, _k, w in big) + ".")
    if small:
        lines.append("Smaller things to do better: " + ", ".join(small) + ".")
    lines.append("")
    steps = ["start with a short, warm greeting that uses the user's name once (the name is in the first line above)"]
    if well:
        steps.append("one sentence on what went well")
    if len(big) == 1:
        steps.append(f"one sentence on the big miss, {big[0][1]}, and how to fix it")
    elif big:
        steps.append("one sentence that names every big miss together, leaving none out")
        steps.append(f"one sentence on how to fix the biggest one, {big[0][1]}")
    elif small:
        steps.append("one sentence on the smaller things to do better")
    else:
        steps.append("one sentence saying the routine is working")
    lines.append("Plan, in this order: " + "; then ".join(steps) + ". Four sentences at most, no numbers, no lists. "
                 "Stop after the last of these: no encouragement, no sign-off, and do not use the name again. Do not say "
                 "that nothing went well.")
    return "\n".join(lines)


def yesterday_prompt_v4(inp: dict, identity: str) -> str:
    return f"{identity}\n\n" + bl._INTRO.format(temp="") + yesterday_view_v4(inp)


_COLD = re.compile(r"\b(nothing (?:really )?stood out|nothing went well|nothing went right)\b", re.I)
_GREETING_NAME = re.compile(r"^\W*(?:good morning|good day|hi|hey|hello)\W+([A-Z][\w'-]*)", re.I)
# a closing pep line ("Keep up the great work, G.") is a sign-off and uses the name a second time (seen on days with no misses)
_PEP = re.compile(r"\b(keep (?:it )?up|keep going|keep the (?:momentum|streak)|you've got this|stay (?:consistent|focused))\b",
                  re.I)


def problems_v4_yesterday(text: str, inp: dict) -> list[str]:
    """v1's checks (figures, lists), plus every big miss named, at most five sentences, and no 'nothing stood out'."""
    text = (text or "").replace("’", "'").replace("‘", "'")
    out = list(bl.problems("yesterday", text, inp))
    big, _small = misses(inp)
    missing = [k for _s, k, _w in big if not bl._MENTIONS[k].search(text)]
    if missing:
        out.append("it leaves out " + " and ".join(missing) + ", which missed by 10% or more; name every big miss in one "
                   "sentence")
    n = len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])
    if n > 5:
        out.append(f"it has {n} sentences; keep it to four or five including the greeting")
    if _COLD.search(text):
        out.append("it says nothing went well; leave that out and go straight to what needs fixing")
    m = _GREETING_NAME.match(text)
    if m and len(re.findall(rf"\b{re.escape(m.group(1))}\b", text)) > 1:
        out.append("it uses the name more than once; use it only in the greeting")
    last = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()][-1:] or [""]
    if _PEP.search(last[0]):
        out.append("it ends with encouragement; stop after the last sentence of the plan, with no sign-off")
    return out
