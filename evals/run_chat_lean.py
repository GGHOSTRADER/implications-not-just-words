"""A lean chat agent for replication: the paper's "words" chat design without the application around it.

    python -m evals.run_chat_lean --set fresh2 --repeats 1
    python -m evals.run_chat_lean --cases G01,G05 --setups words,words_noguard

What the application's chat does for a user's message in the words design, and what this file does for a fake day:
  1. the day's figures become facts in words (chat_facts.view; exact figures only when the question asks for one),
  2. one call to the local model with the persona, the rules (chat_facts.WORDS_PROMPT), the facts and the question,
  3. the answer is checked (chat_facts.problems) and redrafted at most twice with the problems named,
  4. the answer is tidied (chat_facts.tidy).
No tools, no graph, no database. Each fake day is a case from evals/chat_cases.py; its calorie target (the application's
step-bonus rule is not part of this release) is read from evals/chat_targets.json. Scoring uses evals/chat_rubric.py.

Only the words design runs here. The original design (raw figures through tools) needs the application's agent graph; its
saved answers are in evals/results/ and are scored with the same checklist. Results go to evals/results/chat_<time>.json
in the same format as the saved runs. Needs Ollama running with the model in .env (default qwen3:8b).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = Path(__file__).parent / "results"

from evals import chat_cases as cc, chat_rubric as rub  # noqa: E402

TARGETS = json.loads((Path(__file__).parent / "chat_targets.json").read_text(encoding="utf-8"))
NOW = (2026, 10, 5)                      # the fake days all happen on this date
PERSONA = ("Your name is Base Agent. Tone: Warm and casual, like a knowledgeable friend texting. Relaxed, low-pressure, "
           "informal. Example: \"Hey, a bit behind on steps — maybe squeeze in a walk?\"")   # the application's default persona
SETUPS = {
    "words": "chat_facts.py with its answer checks and up to two redrafts",
    "words_noguard": "chat_facts.py: the day as words, no answer checks (isolates the representation)",
}


def day_inputs(day: dict, day_name: str) -> dict:
    """What chat_facts.view describes: the fake day's figures (what the application builds from its own loaders)."""
    import label_context_v3 as ctx3
    now = datetime(*NOW, day["hour"], day["minute"])
    entries = [{"food": n, "calories": k, "protein": p, "time": t} for n, k, p, t in day["entries"]]
    consumed, protein = sum(e["calories"] for e in entries), sum(e["protein"] for e in entries)
    since = None
    if entries:
        h, m = map(int, entries[-1]["time"].split(":"))
        since = round((now - now.replace(hour=h, minute=m)).total_seconds() / 3600, 1)
    target = TARGETS[day_name]["cal_target"]
    inp = {"now": now, "phase": day["phase"], "step_goal": int(day["step_goal"]), "steps": day["steps"],
           "steps_age_min": day["steps_age"], "cal_target": target, "cal_consumed": consumed,
           "protein_target": int(day["protein_target"]), "protein_consumed": protein, "entries": entries,
           "hours_since_last_entry": since, "weather": None, "foods": []}
    room = target - consumed
    if room > 0.1 * target:                    # the user's own foods that fit what is left, supplements out
        near = [SimpleNamespace(food_name=n, calories=k, protein_g=p) for n, k, p, _c in cc.INVENTORY
                if abs(k - room * 0.4) <= room * 0.3]
        near = sorted(near, key=lambda r: abs(r.calories - room * 0.4))[:8]
        fits = [{"food": f.food_name, "calories": f.calories, "protein": f.protein_g} for f in near
                if not ctx3.is_supplement(f.food_name, f.calories, f.protein_g)]
        inp["foods"] = sorted(fits, key=lambda f: -f["protein"])[:3]
    return inp


def facts_for(day: dict, day_name: str, question: str) -> tuple[dict, str]:
    import chat_facts as cf
    inp = day_inputs(day, day_name)
    now = inp["now"]
    sched = cf.schedule_fractions(list(cc.SCHEDULE), now.hour, now.minute)
    return inp, cf.view(inp, question, sched)


def answer(model, inp: dict, question: str, facts: str, guard: bool = True) -> str:
    """One answer: model call, then up to two checked redrafts (guard=False skips the checks)."""
    import chat_facts as cf
    from agent import _clean_response
    from langchain_core.messages import HumanMessage, SystemMessage

    system = f"{PERSONA}\n\n{cf.WORDS_PROMPT}\nFACTS (written by code from the user's data, now):\n{facts}"
    history = [HumanMessage(content=question)]
    text = _clean_response(getattr(model.invoke([SystemMessage(content=system)] + history), "content", "") or "").strip()
    if guard:
        for _attempt in range(2):
            found = cf.problems(text, inp, question, facts)
            if not found:
                break
            redraft = (system + "\n\nYour previous draft was:\n" + text + "\nIt had these problems: " + "; ".join(found) +
                       ".\nWrite the answer again, fixing them.")
            text = _clean_response(getattr(model.invoke([SystemMessage(content=redraft)] + history), "content", "") or "").strip()
    return cf.tidy(text) or "I could not put an answer together. Please try again."


def bad_day(day: dict, protein: float) -> bool:
    """Steps or protein more than a quarter behind the plan for this hour (own linear curve, not chat_facts')."""
    t = day["hour"] + day["minute"] / 60
    if t < cc.SCHEDULE[0][1]:
        return False
    pts = [(6.0, 0.0, 0.0)] + [(float(r[1]), r[2], r[4]) for r in cc.SCHEDULE]
    sf = pf = 1.0
    if t < pts[-1][0]:
        sf = pf = 0.0
        for (h0, s0, p0), (h1, s1, p1) in zip(pts, pts[1:]):
            if h0 <= t <= h1:
                k = (t - h0) / (h1 - h0)
                sf, pf = s0 + k * (s1 - s0), p0 + k * (p1 - p0)
                break
    steps_dev = day["steps"] / (day["step_goal"] * sf) - 1 if sf > 0.05 else 0
    prot_dev = protein / (day["protein_target"] * pf) - 1 if pf > 0.05 else 0
    return steps_dev < -0.25 or prot_dev < -0.25


def truth_for(day: dict, day_name: str) -> dict:
    """The day's truth for the checklist (what the application's evaluation harness computes for a fake day)."""
    inp = day_inputs(day, day_name)
    consumed, protein, target = inp["cal_consumed"], inp["protein_consumed"], inp["cal_target"]
    bonus = abs(TARGETS[day_name]["bonus_kcal"])
    entries = inp["entries"]
    return {
        "phase": day["phase"], "hours_left": round(24 - (day["hour"] + day["minute"] / 60), 1),
        "bad_day": bad_day(day, protein), "hour": day["hour"], "minute": day["minute"], "steps": day["steps"],
        "step_goal": day["step_goal"], "steps_hit": day["steps"] >= day["step_goal"],
        "steps_left": max(0, day["step_goal"] - day["steps"]), "steps_age": day["steps_age"],
        "cal_target": target, "cal_consumed": consumed, "cal_room": round(target - consumed, 1),
        "protein_target": day["protein_target"], "protein_consumed": protein,
        "protein_left": max(0, round(day["protein_target"] - protein)), "entries": [e["food"] for e in entries],
        "entry_figures": [(e["calories"], e["protein"]) for e in entries],
        "system_figures": [day["base_kcal"], target - day["base_kcal"], abs(day["steps"] - day["step_goal"]), bonus,
                           abs(day["protein_target"] - protein), abs(target - consumed)]
        + [v for row in cc.INVENTORY for v in row[1:3]]
        + [round(day["step_goal"] * row[2]) for row in cc.SCHEDULE]
        + [round(target * row[3]) for row in cc.SCHEDULE] + [round(day["protein_target"] * row[4]) for row in cc.SCHEDULE],
    }


def fingerprint() -> str:
    """Hash of the chat design's source (same recipe as the application's evaluation: a changed design does not pool)."""
    return hashlib.sha1((ROOT / "chat_facts.py").read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:10]


def selected_cases(case_set: str, ids: str | None) -> list[dict]:
    rows = [dict(id=i, day=d, question=q, kind=k, set=s) for i, d, q, k, s in cc.CASES]
    if ids:
        want = {x.strip().upper() for x in ids.split(",")}
        return [r for r in rows if r["id"].upper() in want]
    return rows if case_set == "all" else [r for r in rows if r["set"] == case_set]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", default="words")
    ap.add_argument("--set", default="fresh2", choices=["tuning", "holdout", "fresh", "fresh2", "all"])
    ap.add_argument("--cases", default=None, help="comma-separated case ids such as G01,G05 (overrides --set)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    setups = [s.strip() for s in args.setups.split(",") if s.strip()]
    for s in setups:
        if s not in SETUPS:
            ap.error(f"unknown setup {s}; choose from {list(SETUPS)}")
    import llm_registry
    model = llm_registry.get("briefing_nlg")
    cases = selected_cases(args.set, args.cases)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"chat_lean_{stamp}.json"
    meta = {"checklist": rub.VERSION, "set": args.set, "repeats": args.repeats, "engine": "lean",
            "model": os.getenv("MODEL_NAME", "qwen3:8b"), "fingerprints": {s: fingerprint() for s in setups},
            "started": stamp}
    results: list[dict] = []
    for rep in range(1, args.repeats + 1):
        for case in cases:
            day = cc.DAYS[case["day"]]
            truth = truth_for(day, case["day"])
            for setup in setups:
                t0 = time.time()
                inp, facts = facts_for(day, case["day"], case["question"])
                text = answer(model, inp, case["question"], facts, guard=(setup == "words"))
                scores = rub.score(text, case, truth)
                row = {"case": case["id"], "set": case["set"], "day": case["day"], "kind": case["kind"],
                       "question": case["question"], "setup": setup, "repeat": rep, "answer": text,
                       "seconds": round(time.time() - t0, 1), "scores": scores, "passed": rub.passed(scores),
                       "failed": rub.failed_items(scores), "fingerprint": meta["fingerprints"][setup]}
                results.append(row)
                path.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{case['id']:>3} {setup:14} {'PASS' if row['passed'] else 'FAIL'} {row['seconds']:5.1f}s "
                      f"{','.join(row['failed']) or '-'}", flush=True)
    print(f"\nsaved {path}")
    for s in setups:
        rows = [r for r in results if r["setup"] == s]
        print(f"{s}: {sum(r['passed'] for r in rows)} of {len(rows)} answers passed every applicable item")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
