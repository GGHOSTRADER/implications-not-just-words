"""Run the checkpoint comment and morning briefing designs over the synthetic cases, score every answer, save everything.

    python -m evals.run_message_eval --setups checkpoint_numbers,checkpoint_labels --set all --repeats 3
    python -m evals.run_message_eval --setups briefing_numbers,briefing_labels_noguard,briefing_labels --cases 01,H03

Each setup calls the production code path: checkpoint_numbers is background_checks._checkpoint_comment itself (with
the case's phase), the labels setups are background_checks._checkpoint_comment_labels / _briefing_labels_part, and
briefing_numbers sends the two extracted production prompts exactly as _fire_morning_briefing does. All of them use the
production model role (briefing_nlg). Needs Ollama running. Results are saved after every answer, so an interrupted run
loses nothing: evals/results/messages_<time>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = Path(__file__).parent / "results"
# The production code logs "[checkpoint first] LLM comment generated" and, on errors, "comment LLM failed", which
# health_monitor counts in agent.log and alerts on. agent.py reads CALVIS_LOG_FILE when it is first imported, so
# the evaluation's lines go to their own file instead (pytest's conftest already sets its own).
RESULTS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CALVIS_LOG_FILE", str(RESULTS / "message_eval.log"))
LEDGER = Path(__file__).parent / "ledger_messages.md"
IDENTITY = "You are Calvis, a friendly fitness coach. The user is G."

# name -> (description, message, mode)
SETUPS = {
    "checkpoint_numbers": ("checkpoint comment, the original prompt: figures, status codes 1-3, carry-over amounts and "
                           "the food list with kcal/protein", "checkpoint", "numbers"),
    "checkpoint_labels_noguard": ("checkpoint comment over text labels (checkpoint_labels.py), no answer checks: "
                                  "isolates the representation", "checkpoint", "labels_noguard"),
    "checkpoint_labels": ("checkpoint comment over text labels, with answer checks and one redraft", "checkpoint",
                          "labels"),
    "briefing_numbers": ("morning briefing's two sections, the original prompts: classifications, adjusted values and "
                         "carry-over amounts", "briefing", "numbers"),
    "briefing_labels_noguard": ("morning briefing over text labels (briefing_labels.py), no answer checks", "briefing",
                                "labels_noguard"),
    "briefing_labels": ("morning briefing over text labels, with answer checks and one redraft each", "briefing",
                        "labels"),
    # version 2 of the labels (2026-10-04), from v1's failures on the tuning cases
    "checkpoint_labels2_noguard": ("checkpoint labels v2: the day's last check holds only how the day went (no weather "
                                   "or movement lines), size words kept as given; no answer checks", "checkpoint",
                                   "labels2_noguard"),
    "checkpoint_labels2": ("checkpoint labels v2 with its answer checks (a request for more food told from a "
                           "description, the last check's length and weather talk) and one redraft", "checkpoint",
                           "labels2"),
    "briefing_labels2_noguard": ("briefing labels v2: only 5%+ misses on the to-do list and every item named, a plain "
                                 "note when no workout is planned, a suggested duration allowed; no answer checks",
                                 "briefing", "labels2_noguard"),
    "briefing_labels2": ("briefing labels v2 with its answer checks (every to-do item named, the no-plan note, "
                         "durations allowed) and one redraft each", "briefing", "labels2"),
    # version 3 (2026-10-05), from a real message: steps are walking, no exercises or invented workouts, weather-aware
    # indoor/outdoor ideas, a sense of the time of day, no supplements as food
    "checkpoint_labels3_noguard": ("checkpoint labels v3: walking-only steps, weather-aware ideas, time of day, no "
                                   "supplements; no answer checks", "checkpoint", "labels3_noguard"),
    "checkpoint_labels3": ("checkpoint labels v3 with its answer checks (exercises, invented workouts, meals and walks "
                           "that no longer fit the hour, outdoor-only ideas when the steps belong indoors)", "checkpoint",
                           "labels3"),
    "briefing_labels3_noguard": ("briefing labels v3 (yesterday as v1; today: walking-only steps, weather-aware ideas, "
                                 "planned session only); no answer checks", "briefing", "labels3_noguard"),
    "briefing_labels3": ("briefing labels v3 with its answer checks", "briefing", "labels3"),
    # version 4 (2026-10-05): the yesterday part ranks big misses by size and names them all in one sentence
    "briefing_labels4_noguard": ("briefing labels v4: yesterday names every big miss in one sentence, biggest first; "
                                 "today as v3; no answer checks", "briefing", "labels4_noguard"),
    "briefing_labels4": ("briefing labels v4 with its answer checks (every big miss named, at most five sentences, no "
                         "'nothing stood out')", "briefing", "labels4"),
}


def _cases(message: str, case_set: str) -> dict:
    from evals import briefing_cases, checkpoint_cases
    return (checkpoint_cases if message == "checkpoint" else briefing_cases).cases(case_set)


class _Patched:
    """The identity header, the phase _checkpoint_comment reads, and the model role, for one answer (no database)."""

    def __init__(self, model, goals: dict | None = None):
        self.model, self.goals = model, goals

    def __enter__(self):
        import agent
        import llm_registry
        self._saved = (agent._identity_header, agent._load_goals, llm_registry.get)
        agent._identity_header = lambda *a, **k: IDENTITY
        if self.goals is not None:
            goals = self.goals
            agent._load_goals = lambda *a, **k: dict(goals)
        llm_registry.get = lambda role: self.model
        return self

    def __exit__(self, *exc):
        import agent
        import llm_registry
        agent._identity_header, agent._load_goals, llm_registry.get = self._saved
        return False


class _Counting:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def invoke(self, *a, **k):
        self.calls += 1
        return self.inner.invoke(*a, **k)


def _briefing_numbers(case: dict, model) -> dict:
    """The original design, as _fire_morning_briefing sends it: one call per section, _clean_response on each."""
    import background_checks as bc
    from agent import _clean_response
    from langchain_core.messages import HumanMessage as _HM
    orig = case["original"]
    y = _clean_response(getattr(model.invoke([_HM(content=bc._briefing_yesterday_prompt(**orig["yesterday"]))]),
                                "content", "") or "")
    t = _clean_response(getattr(model.invoke([_HM(content=bc._briefing_today_prompt(**orig["today"]))]),
                                "content", "") or "")
    return {"yesterday": y, "today": t}


def run_one(setup: str, case: dict):
    """(answer, seconds, model calls). The answer is a string for a checkpoint, {'yesterday','today'} for a briefing."""
    import background_checks as bc
    import llm_registry
    _desc, message, mode = SETUPS[setup]
    model = _Counting(llm_registry.get("briefing_nlg"))
    t0 = time.time()
    with _Patched(model, case.get("goals")):
        guards = not mode.endswith("_noguard")
        if message == "checkpoint":
            if mode == "numbers":
                answer = bc._checkpoint_comment(case["cp"], case["ctx_text"])
            elif mode.startswith("labels3"):
                answer = bc._checkpoint_comment_labels_v3(case["label_input"], guards=guards)
            elif mode.startswith("labels2"):
                answer = bc._checkpoint_comment_labels_v2(case["label_input"], guards=guards)
            else:
                answer = bc._checkpoint_comment_labels(case["label_input"], guards=guards)
        elif mode == "numbers":
            answer = _briefing_numbers(case, model)
        else:
            part_fn = (bc._briefing_labels_part_v4 if mode.startswith("labels4")
                       else bc._briefing_labels_part_v3 if mode.startswith("labels3")
                       else bc._briefing_labels_part_v2 if mode.startswith("labels2") else bc._briefing_labels_part)
            answer = {part: part_fn(part, case["label_input"], model, guards=guards) for part in ("yesterday", "today")}
    return answer, time.time() - t0, model.calls


def prompts(setup: str, case: dict) -> list[str]:
    """Every prompt a setup sends first for a case (the redraft prompt is the same plus the problems found)."""
    import background_checks as bc
    import briefing_labels
    import checkpoint_labels
    _desc, message, mode = SETUPS[setup]
    with _Patched(None, case.get("goals")):
        if message == "checkpoint":
            if mode == "numbers":
                return [bc._checkpoint_prompt(case["cp"], case["ctx_text"], case["phase"])]
            if mode.startswith("labels3"):
                import checkpoint_labels_v3
                return [checkpoint_labels_v3.prompt_v3(case["label_input"], IDENTITY)]
            if mode.startswith("labels2"):
                return [checkpoint_labels.prompt_v2(case["label_input"], IDENTITY)]
            return [checkpoint_labels.prompt(case["label_input"], IDENTITY)]
        if mode == "numbers":
            return [bc._briefing_yesterday_prompt(**case["original"]["yesterday"]),
                    bc._briefing_today_prompt(**case["original"]["today"])]
        if mode.startswith("labels4"):
            import briefing_labels_v3
            import briefing_labels_v4
            return [briefing_labels_v4.yesterday_prompt_v4(case["label_input"], IDENTITY),
                    briefing_labels_v3.today_prompt_v3(case["label_input"], IDENTITY)]
        if mode.startswith("labels3"):
            import briefing_labels_v3
            return [briefing_labels.yesterday_prompt(case["label_input"], IDENTITY),
                    briefing_labels_v3.today_prompt_v3(case["label_input"], IDENTITY)]
        if mode.startswith("labels2"):
            return [briefing_labels.yesterday_prompt_v2(case["label_input"], IDENTITY),
                    briefing_labels.today_prompt_v2(case["label_input"], IDENTITY)]
        return [briefing_labels.yesterday_prompt(case["label_input"], IDENTITY),
                briefing_labels.today_prompt(case["label_input"], IDENTITY)]


def fingerprint(setup: str) -> str:
    """A short hash of everything that decides a setup's behaviour: the model behind the role, every first prompt for
    every case (tuning + holdout), and the source of the code that calls the model, cleans the answer and (with
    guards) checks and redrafts it. Same fingerprint = same version of the approach, so the report pools those runs."""
    import hashlib
    import inspect
    import os

    import agent
    import background_checks as bc
    import briefing_labels
    import checkpoint_labels
    _desc, message, mode = SETUPS[setup]
    h = hashlib.sha256(f"{message}|{mode}|{os.getenv('LLM_BRIEFING_NLG', 'qwen')}".encode())
    model = __import__("llm_registry").get("briefing_nlg")
    h.update(f"{getattr(model, 'model', '')}|{getattr(model, 'reasoning', '')}|{getattr(model, 'temperature', '')}".encode())
    for _name, case in sorted(_cases(message, "all").items()):
        for p in prompts(setup, case):
            h.update(p.encode("utf-8"))
    sources = [agent._clean_response]
    outdoor = ("_OUTSIDE", "_NOT_OUT_BEFORE", "_NOT_OUT_AFTER", "_ADVICE_CUE", "_CONDITIONAL")
    if mode.startswith("labels4"):              # v4 (briefing only) lives in its own module; v3's modules are hashed too
        import briefing_labels_v3
        import briefing_labels_v4
        import checkpoint_labels_v3
        import label_context_v3
        sources += [bc._briefing_labels_part_v4, bc._labels_answer, checkpoint_labels.suggests_outside]
        for mod in (label_context_v3, checkpoint_labels_v3, briefing_labels_v3, briefing_labels_v4):
            h.update(inspect.getsource(mod).encode("utf-8"))
        if mode == "labels4":
            sources += [briefing_labels_v4.problems_v4_yesterday, briefing_labels_v3.problems_v3, briefing_labels.problems_v2,
                        briefing_labels.problems]
            for pat in ("_LIST",) + outdoor:
                h.update(getattr(checkpoint_labels, pat).pattern.encode("utf-8"))
    elif mode.startswith("labels3"):              # v3 lives in its own modules; v1 and v2 branches stay as they were
        import briefing_labels_v3
        import checkpoint_labels_v3
        import label_context_v3
        sources += [bc._checkpoint_comment_labels_v3 if message == "checkpoint" else bc._briefing_labels_part_v3,
                    bc._labels_answer, checkpoint_labels.suggests_outside]
        h.update(inspect.getsource(label_context_v3).encode("utf-8"))
        h.update(inspect.getsource(checkpoint_labels_v3).encode("utf-8"))
        if message == "briefing":
            h.update(inspect.getsource(briefing_labels_v3).encode("utf-8"))
        if mode == "labels3":
            sources += [checkpoint_labels_v3.problems_v3, checkpoint_labels.problems_v2, briefing_labels_v3.problems_v3,
                        briefing_labels.problems_v2]
            for pat in ("_SCOLD", "_STILL_GOING", "_EAT_MORE_REQUEST", "_CATCH_UP", "_LIST", "_NUM", "_SLIGHT",
                        "_WEATHER_TALK") + outdoor:
                h.update(getattr(checkpoint_labels, pat).pattern.encode("utf-8"))
    elif mode.startswith("labels2"):              # v2 has its own functions; v1's branches below stay as they were
        sources += [bc._checkpoint_comment_labels_v2 if message == "checkpoint" else bc._briefing_labels_part_v2,
                    bc._labels_answer]
        if mode == "labels2" and message == "checkpoint":
            sources += [checkpoint_labels.problems_v2, checkpoint_labels.suggests_outside]
            for pat in ("_SCOLD", "_STILL_GOING", "_EAT_MORE_REQUEST", "_CATCH_UP", "_LIST", "_NUM", "_SLIGHT",
                        "_WEATHER_TALK") + outdoor:
                h.update(getattr(checkpoint_labels, pat).pattern.encode("utf-8"))
        elif mode == "labels2":
            sources += [briefing_labels.problems_v2, briefing_labels._to_do, checkpoint_labels.suggests_outside]
            for pat in ("_GREETING", "_NUM", "_DURATION"):
                h.update(getattr(briefing_labels, pat).pattern.encode("utf-8"))
            for key, pat in sorted(briefing_labels._MENTIONS.items()):
                h.update(f"{key}:{pat.pattern}".encode("utf-8"))
            for pat in ("_LIST",) + outdoor:
                h.update(getattr(checkpoint_labels, pat).pattern.encode("utf-8"))
    elif message == "checkpoint":
        sources.append(bc._checkpoint_comment if mode == "numbers" else bc._checkpoint_comment_labels)
        if mode == "labels":
            sources += [checkpoint_labels.problems, checkpoint_labels.suggests_outside]
            for pat in ("_SCOLD", "_STILL_GOING", "_EAT_MORE", "_CATCH_UP", "_LIST", "_NUM", "_SLIGHT") + outdoor:
                h.update(getattr(checkpoint_labels, pat).pattern.encode("utf-8"))
    else:
        sources.append(_briefing_numbers if mode == "numbers" else bc._briefing_labels_part)
        if mode == "labels":
            sources += [briefing_labels.problems, checkpoint_labels.suggests_outside]
            for pat in ("_GREETING", "_NUM"):
                h.update(getattr(briefing_labels, pat).pattern.encode("utf-8"))
            for pat in ("_LIST",) + outdoor:
                h.update(getattr(checkpoint_labels, pat).pattern.encode("utf-8"))
    for fn in sources:
        h.update(inspect.getsource(fn).encode("utf-8"))
    return h.hexdigest()[:12]


def score(setup: str, answer, case: dict):
    from evals import message_rubric as mr
    return mr.score_checkpoint(answer, case) if SETUPS[setup][1] == "checkpoint" else mr.score_briefing(answer, case)


_LEDGER_HEADER = ("| date | approach | cases | answers | every item passed | hallucination | instruction | semantic "
                  "| avg s | calls | code | note |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def _git_head() -> str:
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("+uncommitted" if dirty else "")
    except Exception:
        return "?"


def append_ledger(results: list[dict], setups: list[str], case_set: str, path: Path) -> None:
    """One row per setup in evals/ledger_messages.md, the running record of every checkpoint and briefing approach."""
    from evals import message_rubric as mr
    if not LEDGER.exists():
        LEDGER.write_text("# Checkpoint comment and morning briefing: approach ledger\n\nOne row per evaluation run, "
                          "newest last. Rows compare only within the same message and case set. `tuning` = the 20 cases "
                          "the label wording was written against; `holdout` = 10 untouched cases. Scores are as scored "
                          "at run time; the report re-scores every saved answer with the current checklist.\n\n"
                          + _LEDGER_HEADER, encoding="utf-8")
    rows = ""
    for setup in setups:
        rs = [r for r in results if r["setup"] == setup]
        if not rs:
            continue
        kinds = {k: [0, 0] for k in mr.KINDS}
        for r in rs:
            for i in r["items"]:
                kinds[i["kind"]][1] += 1
                kinds[i["kind"]][0] += i["ok"]
        clean = sum(all(i["ok"] for i in r["items"]) for r in rs)
        k = lambda name: f"{kinds[name][0]}/{kinds[name][1]}"
        reps = len({r.get("repeat", 0) for r in rs})
        note = (f"results file {path.name}; rubric {rs[0].get('rubric', '')}" + (f"; {reps} repeats" if reps > 1 else "")
                + f"; fingerprint {rs[0].get('fingerprint', '')}")
        rows += (f"| {datetime.now():%Y-%m-%d %H:%M} | {setup} | {case_set} | {len(rs)} | {clean}/{len(rs)} "
                 f"| {k('hallucination')} | {k('instruction')} | {k('semantic')} "
                 f"| {sum(r['seconds'] for r in rs) / len(rs):.1f} | {sum(r['calls'] for r in rs) / len(rs):.2f} "
                 f"| {_git_head()} | {note} |\n")
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", default="checkpoint_numbers,checkpoint_labels")
    ap.add_argument("--cases", default="all", help="'all' or comma-separated prefixes such as 01,06,H03")
    ap.add_argument("--set", default="all", choices=["tuning", "holdout", "all", "incident"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)

    from evals import message_rubric as mr
    setups = [s.strip() for s in args.setups.split(",") if s.strip()]
    unknown = [s for s in setups if s not in SETUPS]
    if unknown:
        print(f"unknown setup(s) {unknown}; choose from {list(SETUPS)}")
        return 2
    prefixes = None if args.cases == "all" else [c.strip() for c in args.cases.split(",")]
    pools = {}
    for s in setups:
        msg = SETUPS[s][1]
        if msg not in pools:
            pools[msg] = {n: c for n, c in _cases(msg, args.set).items() if prefixes is None or n.split()[0] in prefixes}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"messages_{datetime.now():%Y%m%d_%H%M%S}.json"
    prints = {s: fingerprint(s) for s in setups}
    meta = {"setups": {s: SETUPS[s][0] for s in setups}, "fingerprints": prints, "case_set": args.set,
            "repeats": args.repeats, "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "code": _git_head(),
            "rubric": mr.MESSAGE_RUBRIC_VERSION, "model": os.getenv("MODEL_NAME", "qwen3:8b")}
    results: list[dict] = []
    total = sum(len(pools[SETUPS[s][1]]) for s in setups) * args.repeats
    print(f"{total} answers ({', '.join(setups)}; {args.repeats} repeat(s)) -> {path}\n", flush=True)
    for rep in range(args.repeats):
        for msg, pool in pools.items():
            for name, case in pool.items():
                for setup in (s for s in setups if SETUPS[s][1] == msg):
                    try:
                        answer, secs, calls = run_one(setup, case)
                        error = ""
                    except Exception as e:                 # keep going; the failure is recorded and counted
                        answer, secs, calls, error = ("" if msg == "checkpoint" else {"yesterday": "", "today": ""}), 0.0, 0, repr(e)
                    items = score(setup, answer, case)
                    bad = mr.failed(items)
                    results.append({
                        "message": msg, "case": name, "setup": setup, "repeat": rep, "fingerprint": prints[setup],
                        "rubric": mr.MESSAGE_RUBRIC_VERSION, "seconds": round(secs, 1), "calls": calls,
                        "answer": answer, "error": error,
                        "items": [{"kind": i.kind, "id": i.id, "ok": i.ok, "detail": i.detail} for i in items],
                    })
                    path.write_text(json.dumps({**meta, "results": results}, indent=1, ensure_ascii=False),
                                    encoding="utf-8")
                    print(f"r{rep} {name[:36]:<36} {setup:<26} {secs:5.1f}s {calls} calls | "
                          + (f"ERROR {error}" if error else
                             "all items pass" if not bad else "FAILED: " + ", ".join(i.id for i in bad)), flush=True)

    print("\n===== SUMMARY by setup")
    for setup in setups:
        rows = [r for r in results if r["setup"] == setup]
        n = len(rows)
        by_kind = {k: [0, 0] for k in mr.KINDS}
        fails: dict[str, int] = {}
        for r in rows:
            for i in r["items"]:
                by_kind[i["kind"]][1] += 1
                by_kind[i["kind"]][0] += i["ok"]
                if not i["ok"]:
                    fails[i["id"]] = fails.get(i["id"], 0) + 1
        clean = sum(all(i["ok"] for i in r["items"]) for r in rows)
        top = ", ".join(f"{k} x{v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])[:6]) or "none"
        print(f"{setup:<26} {n} answers | avg {sum(r['seconds'] for r in rows) / max(n, 1):5.1f}s | "
              f"every item passed: {clean}/{n}\n    items: "
              + "  ".join(f"{k} {p}/{t}" for k, (p, t) in by_kind.items()) + f"\n    most common failures: {top}")
    print(f"\nsaved {path}")
    if not args.no_ledger:
        append_ledger(results, setups, args.set, path)
        print(f"added {len(setups)} row(s) to {LEDGER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
