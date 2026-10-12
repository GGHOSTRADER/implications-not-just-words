"""Run setups of the weekly review over the synthetic weeks, score every answer against the checklist, save everything.

    python -m evals.run_weekly_eval                         # split_fast and old_think over all 20 cases
    python -m evals.run_weekly_eval --setups split_fast    # just the current design
    python -m evals.run_weekly_eval --cases 01,06,16       # a few cases by their number prefix

Needs Ollama running with qwen3:8b. Takes minutes (the thinking setups take much longer). Results are saved after
EVERY answer, so an interrupted run loses nothing: evals/results/weekly_<time>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = Path(__file__).parent / "results"

# name -> (description, model attribute on llm_registry, design)
SETUPS = {
    "split_fast": ("two questions, fast mode (the current design)", "qwen", "split"),
    "split_think": ("two questions, thinking mode", "qwen_think", "split"),
    "labels_fast": ("two questions over text labels, no digits shown (weekly_labels.py), fast mode", "qwen", "labels"),
    "labels_think": ("two questions over text labels, thinking mode", "qwen_think", "labels"),
    "labels2_fast": ("labels v2: gap sizes, directions, keep-as-is list, advice goal, workout verdict, varied wording, fast mode",
                     "qwen", "labels2"),
    "labels3_fast": ("labels v3: what each gap means for the phase, trend vs target, only real links, phase-weighted "
                     "priorities, kind of fix, keep-an-eye; fast mode", "qwen", "labels3"),
    "labels4_fast": ("labels v4: v3's meaning, but each question sees only its own view (meaning inline per metric "
                     "for the observations; priorities, fixes and constraints for the recommendations); fast mode",
                     "qwen", "labels4"),
    "labels5_fast": ("labels v5: v4, plus code decides how many recommendations to write and what each one is for "
                     "(one per priority, at least two); fast mode", "qwen", "labels5"),
    "labels6_fast": ("labels v6: v5's plan stated as one sentence with explicit 1./2. numbering (v5's 'Recommendation "
                     "one:' list was copied into the answers); fast mode", "qwen", "labels6"),
    "old_fast": ("one big prompt, fast mode", "qwen", "single"),
    "old_think": ("one big prompt, thinking mode (the design before 2026-10-04)", "qwen_think", "single"),
    # Ablation for the paper (evals/hybrid_weekly.py): A1's prompts and checks, the numeric block kept, labels v1's
    # verdicts appended. Separates "complete verdicts" (A8 vs A1) from "no digits" (A8 vs A2).
    "hybrid_fast": ("numbers plus labels v1's verdicts, digits kept (the words-vs-coverage ablation), fast mode",
                    "qwen", "hybrid"),
    # Ablation for the paper (evals/matched_numeric_weekly.py): labels v6's description, priorities and plan, plus the
    # figures. Separates "the digits" (A9 vs A7) from "implications, priorities and plan" (A9 vs A1).
    "matched6_fast": ("labels v6's description, priorities and plan plus the figures, digits allowed "
                      "(the information-matched numeric ablation), fast mode", "qwen", "matched6"),
}

# The functions whose code decides how each design behaves (prompts aside). Part of the fingerprint below.
_GUARDS = {
    "single": ["_grounding_check", "_directional_check", "_extract_numbers"],
    "split": ["_answer_with_checks", "_grounding_check", "_directional_check", "_extract_numbers"],
    "labels": ["_answer_with_label_checks", "_label_directional_check", "_grounding_check", "_extract_numbers",
               "_keep_only_items", "_first_numbered_item", "_asks", "recommendation_plan"],
    "hybrid": ["_answer_with_checks", "_grounding_check", "_directional_check", "_extract_numbers"],   # = split
    "matched6": ["_grounding_check", "_label_directional_check", "_extract_numbers", "_keep_only_items",
                 "_first_numbered_item", "_asks", "recommendation_plan"],   # the checks themselves are in the module below
}


def fingerprint(setup: str) -> str:
    """A short hash of everything that defines a setup's behaviour: the model it uses, every prompt it would send for
    every case (tuning + holdout; the recommendations prompt with a fixed placeholder observation) and the source of
    its guard functions and patterns. Two runs with the same fingerprint ran the same version of the approach, so the
    report can pool them as repeats; any prompt, label or guard change gives a new fingerprint."""
    import hashlib
    import inspect

    import agent
    import weekly_analysis as wa
    from evals import legacy_weekly, weekly_cases, weekly_holdout_cases

    _desc, attr, design = SETUPS[setup]
    real_header = agent._identity_header
    agent._identity_header = lambda *a, **k: "You are Calvis, a friendly fitness coach. The user is G."
    try:
        h = hashlib.sha256(f"{attr}|{design}".encode())
        for _name, data in sorted({**weekly_cases.CASES, **weekly_holdout_cases.CASES}.items()):
            summary = wa.format_data_summary(data)
            if design == "single":
                prompts = [legacy_weekly.build_single_prompt(data, summary)]
            elif design == "split":
                prompts = [wa.build_observations_prompt(data, summary),
                           wa.build_recommendations_prompt(data, summary, "<OBS>")]
            elif design == "hybrid":
                from evals import hybrid_weekly
                hs = hybrid_weekly.summary(data)
                prompts = [wa.build_observations_prompt(data, hs), wa.build_recommendations_prompt(data, hs, "<OBS>")]
            elif design == "matched6":
                from evals import matched_numeric_weekly as mnw
                prompts = [mnw.observations_prompt(data), mnw.recommendations_prompt(data, "<OBS>")]
            else:
                obs_suffix = {"labels": "label", "labels2": "label2", "labels3": "label3", "labels4": "label4",
                              "labels5": "label4", "labels6": "label4"}[design]   # v5/v6 ask v4's observations question
                recs_suffix = {"labels5": "label5", "labels6": "label6"}.get(design, obs_suffix)
                prompts = [getattr(wa, f"build_{obs_suffix}_observations_prompt")(data),
                           getattr(wa, f"build_{recs_suffix}_recommendations_prompt")(data, "<OBS>")]
            for p in prompts:
                h.update(p.encode("utf-8"))
        for fn in _GUARDS["labels" if design.startswith("labels") else design]:
            h.update(inspect.getsource(getattr(wa, fn)).encode("utf-8"))
        for pattern in ("_PROPOSAL_RE", "_NUMBER_RE", "_INCREASE_RE", "_CALORIES_MORE_RE", "_CALORIES_LESS_RE",
                        "_KEEP_ONLY_RE", "_OBS_ADVICE_V3_RE", "_OBS_ADVICE_V4_RE"):
            obj = getattr(wa, pattern, None)
            h.update(str(getattr(obj, "pattern", obj)).encode("utf-8"))
        if design == "single":
            h.update(inspect.getsource(legacy_weekly.run).encode("utf-8"))
        if design == "hybrid":
            from evals import hybrid_weekly
            h.update(inspect.getsource(hybrid_weekly).encode("utf-8"))
        if design == "matched6":
            from evals import matched_numeric_weekly as mnw
            h.update(inspect.getsource(mnw).encode("utf-8"))
        return h.hexdigest()[:12]
    finally:
        agent._identity_header = real_header


LEDGER = Path(__file__).parent / "ledger.md"
_LEDGER_HEADER = (
    "| date | approach | cases | answers | every item passed | hallucination | instruction | coverage | specificity "
    "| avg s | code | note |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def ledger_row(date: str, approach: str, case_set: str, answers: int, clean: int, kinds: dict, avg_s: float,
               code: str, note: str = "") -> str:
    k = lambda name: f"{kinds[name][0]}/{kinds[name][1]}" if name in kinds else "-"
    return (f"| {date} | {approach} | {case_set} | {answers} | {clean}/{answers} | {k('hallucination')} | {k('instruction')} "
            f"| {k('coverage')} | {k('specificity')} | {avg_s:.1f} | {code} | {note} |\n")


def _git_head() -> str:
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("+uncommitted" if dirty else "")
    except Exception:
        return "?"


def append_ledger(results: list[dict], setups: list[str], case_set: str, path: Path) -> None:
    """Add one row per setup to evals/ledger.md: the running record of every approach, so improvements are tracked.
    Scores are only comparable between rows with the same case set; labels approaches use a slightly different
    checklist (see weekly_rubric.score), which the note says."""
    if not LEDGER.exists():
        LEDGER.write_text("# Weekly review: approach ledger\n\nOne row per evaluation run, newest last. Rows are "
                          "comparable only within the same case set. `tuning` = the 20 weeks prompts were tuned on "
                          "(optimistic); `holdout` = 10 untouched weeks. The labels approaches are scored with a "
                          "slightly different checklist (no digits instead of restating averages).\n\n"
                          + _LEDGER_HEADER, encoding="utf-8")
    from evals import weekly_rubric
    rows = ""
    for setup in setups:
        rs = [r for r in results if r["setup"] == setup]
        if not rs:
            continue
        kinds = {k: [0, 0] for k in weekly_rubric.KINDS}
        for r in rs:
            for i in r["items"]:
                kinds[i["kind"]][1] += 1
                kinds[i["kind"]][0] += i["ok"]
        clean = sum(all(i["ok"] for i in r["items"]) for r in rs)
        reps = len({r.get("repeat", 0) for r in rs})
        fp = rs[0].get("fingerprint", "")
        note = f"results file {path.name}" + (f"; {reps} repeats" if reps > 1 else "") + (f"; fingerprint {fp}" if fp else "")
        rows += ledger_row(f"{datetime.now():%Y-%m-%d %H:%M}", setup, case_set, len(rs), clean, kinds,
                           sum(r["seconds"] for r in rs) / len(rs), _git_head(), note)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(rows)


class _Counting:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def invoke(self, *a, **k):
        self.calls += 1
        return self.inner.invoke(*a, **k)


def run_one(setup: str, data: dict) -> tuple[str, float, int]:
    """(answer text, seconds, model calls) for one setup on one case."""
    import llm_registry
    import weekly_analysis as wa
    from evals import legacy_weekly

    _desc, attr, design = SETUPS[setup]
    model = _Counting(getattr(llm_registry, attr))
    summary = wa.format_data_summary(data)
    if design == "hybrid":                       # A1's path, with labels v1's verdicts appended to the summary
        from evals import hybrid_weekly
        summary = hybrid_weekly.summary(data)
    t = time.time()
    if design == "matched6":
        from evals import matched_numeric_weekly
        text = matched_numeric_weekly.run(data, model)
    elif design in ("split", "hybrid", "labels", "labels2", "labels3", "labels4", "labels5", "labels6"):
        real_get = llm_registry.get
        llm_registry.get = lambda role: model
        try:
            text = wa.run_cot_analysis(data, summary, mode=design if design not in ("split", "hybrid") else "numbers")
        finally:
            llm_registry.get = real_get
    else:
        text = legacy_weekly.run(data, summary, model)
    return text, time.time() - t, model.calls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setups", default="split_fast,old_think")
    ap.add_argument("--cases", default="all", help="'all' or comma-separated number prefixes such as 01,06,16")
    ap.add_argument("--set", default="tuning", choices=["tuning", "holdout", "all"],
                    help="tuning = the 20 weeks prompts were tuned on; holdout = 10 untouched weeks; all = both")
    ap.add_argument("--no-ledger", action="store_true", help="do not append this run to evals/ledger.md")
    ap.add_argument("--repeats", type=int, default=1,
                    help="answer every case this many times per setup (the model samples, so repeats measure variance)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)

    import agent
    agent._identity_header = lambda *a, **k: "You are Calvis, a friendly fitness coach. The user is G."   # no database
    from evals import weekly_cases, weekly_rubric

    setups = [s.strip() for s in args.setups.split(",") if s.strip()]
    unknown = [s for s in setups if s not in SETUPS]
    if unknown:
        print(f"unknown setup(s) {unknown}; choose from {list(SETUPS)}")
        return 2
    from evals import weekly_holdout_cases
    pool = {"tuning": weekly_cases.CASES, "holdout": weekly_holdout_cases.CASES,
            "all": {**weekly_cases.CASES, **weekly_holdout_cases.CASES}}[args.set]
    prefixes = None if args.cases == "all" else [c.strip() for c in args.cases.split(",")]
    cases = {n: d for n, d in pool.items() if prefixes is None or n.split()[0] in prefixes}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"weekly_{datetime.now():%Y%m%d_%H%M%S}.json"
    prints = {s: fingerprint(s) for s in setups}
    meta = {"setups": {s: SETUPS[s][0] for s in setups}, "fingerprints": prints, "case_set": args.set,
            "repeats": args.repeats, "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "code": _git_head(),
            "model": __import__("os").getenv("MODEL_NAME", "qwen3:8b")}
    results: list[dict] = []
    print(f"{len(cases)} cases x {len(setups)} setups x {args.repeats} repeat(s) -> {path}\n", flush=True)
    for rep in range(args.repeats):
        for name, data in cases.items():
            for setup in setups:
                text, secs, calls = run_one(setup, data)
                design = SETUPS[setup][2]
                items = weekly_rubric.score(text, data, mode=design if design.startswith(("labels", "matched")) else "numbers")
                bad = weekly_rubric.failed(items)
                results.append({
                    "case": name, "setup": setup, "repeat": rep, "fingerprint": prints[setup],
                    "seconds": round(secs, 1), "calls": calls, "text": text,
                    "items": [{"kind": i.kind, "id": i.id, "ok": i.ok, "detail": i.detail} for i in items],
                })
                path.write_text(json.dumps({**meta, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"r{rep} {name[:34]:<34} {setup:<12} {secs:6.1f}s {calls} calls | "
                      f"{'all items pass' if not bad else 'FAILED: ' + ', '.join(i.id for i in bad)}", flush=True)

    print("\n===== SUMMARY by setup")
    for setup in setups:
        rows = [r for r in results if r["setup"] == setup]
        n = len(rows)
        by_kind = {k: [0, 0] for k in weekly_rubric.KINDS}
        fails: dict[str, int] = {}
        clean = 0
        for r in rows:
            clean += all(i["ok"] for i in r["items"])
            for i in r["items"]:
                by_kind[i["kind"]][1] += 1
                by_kind[i["kind"]][0] += i["ok"]
                if not i["ok"]:
                    fails[i["id"]] = fails.get(i["id"], 0) + 1
        kinds = "  ".join(f"{k} {p}/{t}" for k, (p, t) in by_kind.items())
        top = ", ".join(f"{k} x{v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])[:5]) or "none"
        print(f"{setup:<12} {n} answers | avg {sum(r['seconds'] for r in rows) / n:5.1f}s | every item passed: {clean}/{n}\n"
              f"             items passed: {kinds}\n             most common failures: {top}")
    print(f"\nsaved {path}")
    if not args.no_ledger:
        append_ledger(results, setups, args.set, path)
        print(f"added {len(setups)} row(s) to {LEDGER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
