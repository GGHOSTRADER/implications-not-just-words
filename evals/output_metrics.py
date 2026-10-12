"""Output-side measures: what the model DID with each representation, computed from saved answers.

Every answer is re-scored with the frozen core rubric (evals/rubric_v2.py) at analysis time, so all approaches are
measured by the same instrument regardless of when they ran. Only runs whose fingerprint matches the approach's CURRENT
fingerprint (evals/run_weekly_eval.fingerprint) are pooled, so a number always describes one exact version.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = Path(__file__).parent / "results"

# setup -> (attempt id, short name, representation key in label_metrics, baseline model calls per review)
APPROACHES = {
    "old_think":    ("A0", "old single prompt (thinking)", "numbers", 1),
    "split_fast":   ("A1", "numbers, two questions", "numbers", 2),
    "labels_fast":  ("A2", "labels v1", "labels_v1", 2),
    "labels2_fast": ("A3", "labels v2", "labels_v2", 2),
    "labels3_fast": ("A4", "labels v3", "labels_v3", 2),
    "labels4_fast": ("A5", "labels v4", "labels_v4", 2),
    "labels5_fast": ("A6", "labels v5", "labels_v5", 2),
    "labels6_fast": ("A7", "labels v6", "labels_v6", 2),
    "hybrid_fast":  ("A8", "numbers + v1 verdicts (ablation)", "numbers", 2),
    "matched6_fast": ("A9", "labels v6 + figures (ablation)", "numbers", 2),
}
# the representation an approach's answers are compared with for the copy rate, when it is not APPROACHES[...][2]
COPY_REPRESENTATION = {"hybrid_fast": "numbers_verdicts"}

OUTPUT_DOCS = {
    "all_items": "share of answers passing every applicable core item (rubric v2)",
    "semantic_all": "share of answers passing every applicable semantic item",
    "item_rate": "share of applicable (answer, item) pairs passed, per kind",
    "redraft_rate": "share of answers where at least one part failed a guard and was redrafted",
    "latency": "seconds per review (mean / median), local qwen3:8b on an RTX 4060",
    "copy_rate": "share of the answer's word 4-grams copied verbatim from the representation it was shown",
    "distinct3": "distinct word trigrams / all word trigrams across one run's answers (one pass over the weeks), "
                 "averaged over runs; higher = less templated. Pooling more answers lowers it, so it is computed "
                 "per run to keep approaches comparable",
    "top_phrase": "share of answers containing the approach's most frequent recommendation 5-gram, and the phrase",
    "obs_digits": "digits per observations paragraph",
}


# ── loading ──────────────────────────────────────────────────────────────────

def load_rows(paths: list[Path] | None = None) -> list[dict]:
    rows = []
    for p in sorted(paths or RESULTS.glob("weekly_*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_set = doc.get("case_set", "")
        for r in doc.get("results", []):
            rows.append({**r, "file": p.name, "case_set_run": case_set})
    return rows


def current_rows(rows: list[dict], fingerprints: dict) -> dict:
    """{setup: [rows]} keeping only rows produced by the setup's current version."""
    out = defaultdict(list)
    for r in rows:
        if r["setup"] in fingerprints and r.get("fingerprint") == fingerprints[r["setup"]]:
            out[r["setup"]].append(r)
    return out


def case_set_of(case: str) -> str:
    """'incident' = the regression cases from a real message (checkpoint_cases.INCIDENT_SPECS), named I01...; weekly
    cases start with a digit or H, so they are unaffected."""
    return "holdout" if case.startswith("H") else "incident" if case.startswith("I") else "tuning"


# ── statistics ───────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(proportion, low, high) with the Wilson score interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def wilcoxon_p(xs: list[float], ys: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p for paired per-week values (None when not computable)."""
    try:
        from scipy.stats import wilcoxon
        diffs = [x - y for x, y in zip(xs, ys)]
        if all(d == 0 for d in diffs):
            return 1.0
        return float(wilcoxon(xs, ys, zero_method="zsplit").pvalue)
    except Exception:
        return None


# ── per-answer scoring ───────────────────────────────────────────────────────

def _cases() -> dict:
    from evals import weekly_cases, weekly_holdout_cases
    return {**weekly_cases.CASES, **weekly_holdout_cases.CASES}


def scored(rows: list[dict]) -> list[dict]:
    """Attach rubric-v2 items to each row (cached per text)."""
    from evals import rubric_v2
    cases = _cases()
    cache = {}
    out = []
    for r in rows:
        key = (r["case"], r["text"])
        if key not in cache:
            cache[key] = rubric_v2.score(r["text"], cases[r["case"]])
        out.append({**r, "core": cache[key]})
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def _ngrams(ws: list[str], n: int) -> list[tuple]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def copy_rate(answer: str, shown: str) -> float:
    a = _ngrams(_words(answer), 4)
    if not a:
        return 0.0
    s = set(_ngrams(_words(shown), 4))
    return sum(1 for g in a if g in s) / len(a)


# ── one approach on one case set ─────────────────────────────────────────────

def summarize(rows: list[dict], setup: str) -> dict:
    """All measures for one approach's rows (already filtered to one case set)."""
    from evals import label_metrics, rubric_v2
    if not rows:
        return {}
    rows = scored(rows)
    n = len(rows)
    rep_fn = {**label_metrics.representations(), **label_metrics.extra_representations()}[
        COPY_REPRESENTATION.get(setup, APPROACHES[setup][2])]
    baseline = APPROACHES[setup][3]
    cases = _cases()

    all_ok = sum(all(i.ok for i in r["core"]) for r in rows)
    sem_ok = sum(all(i.ok for i in r["core"] if i.kind == "semantic") for r in rows)
    kinds = {k: [0, 0] for k in rubric_v2.KINDS}
    items = defaultdict(lambda: [0, 0])
    for r in rows:
        for i in r["core"]:
            kinds[i.kind][1] += 1
            kinds[i.kind][0] += i.ok
            items[i.id][1] += 1
            items[i.id][0] += i.ok

    by_rep = defaultdict(list)
    for r in rows:
        by_rep[r.get("repeat", 0)].append(r)
    rep_all = [sum(all(i.ok for i in x["core"]) for x in rs) / len(rs) for rs in by_rep.values()]
    rep_sem = [sum(all(i.ok for i in x["core"] if i.kind == "semantic") for x in rs) / len(rs) for rs in by_rep.values()]

    obs = [rubric_v2.paragraphs(r["text"])[0] if rubric_v2.paragraphs(r["text"]) else "" for r in rows]
    recs = [rubric_v2.paragraphs(r["text"])[-1] if len(rubric_v2.paragraphs(r["text"])) > 1 else "" for r in rows]
    def distinct3(rs):
        tri = [g for r in rs for g in _ngrams(_words(r["text"]), 3)]
        return len(set(tri)) / len(tri) if tri else 0.0

    rec_5grams = Counter()
    for t in recs:
        rec_5grams.update(set(_ngrams(_words(t), 5)))
    top_phrase, top_count = rec_5grams.most_common(1)[0] if rec_5grams else ((), 0)

    return {
        "n": n,
        "repeats": len(by_rep),
        "weeks": len({r["case"] for r in rows}),
        "all_items": wilson(all_ok, n),
        "semantic_all": wilson(sem_ok, n),
        "all_items_by_repeat": (statistics.mean(rep_all), statistics.stdev(rep_all) if len(rep_all) > 1 else 0.0),
        "semantic_by_repeat": (statistics.mean(rep_sem), statistics.stdev(rep_sem) if len(rep_sem) > 1 else 0.0),
        "kinds": {k: (p, t) for k, (p, t) in kinds.items()},
        "items": {k: (p, t) for k, (p, t) in items.items()},
        "redraft_rate": sum(r["calls"] > baseline for r in rows) / n,
        "calls": statistics.mean(r["calls"] for r in rows),
        "latency": (statistics.mean(r["seconds"] for r in rows), statistics.median(r["seconds"] for r in rows)),
        "obs_words": statistics.mean(len(_words(t)) for t in obs),
        "recs_words": statistics.mean(len(_words(t)) for t in recs),
        "obs_digits": statistics.mean(len(re.findall(r"\d", t)) for t in obs),
        "copy_rate": statistics.mean(copy_rate(r["text"], rep_fn(cases[r["case"]])) for r in rows),
        "distinct3": statistics.mean(distinct3(rs) for rs in by_rep.values()),
        "top_phrase": (top_count / n, " ".join(top_phrase)),
    }


def per_week_scores(rows: list[dict]) -> dict:
    """{case: (mean core fraction, mean semantic fraction)} averaged over repeats, for paired tests."""
    rows = scored(rows)
    acc = defaultdict(lambda: [[], []])
    for r in rows:
        core = r["core"]
        sem = [i for i in core if i.kind == "semantic"]
        acc[r["case"]][0].append(sum(i.ok for i in core) / len(core))
        acc[r["case"]][1].append(sum(i.ok for i in sem) / len(sem) if sem else 1.0)
    return {c: (statistics.mean(a), statistics.mean(b)) for c, (a, b) in acc.items()}


def paired(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """A vs B on the weeks both answered: mean difference in core and semantic fractions, Wilcoxon p, wins/ties."""
    a, b = per_week_scores(rows_a), per_week_scores(rows_b)
    weeks = sorted(set(a) & set(b))
    out = {"weeks": len(weeks)}
    for idx, name in ((0, "core"), (1, "semantic")):
        xa, xb = [a[w][idx] for w in weeks], [b[w][idx] for w in weeks]
        diffs = [x - y for x, y in zip(xa, xb)]
        out[name] = {
            "diff": statistics.mean(diffs) if diffs else 0.0,
            "p": wilcoxon_p(xa, xb) if len(weeks) >= 5 else None,
            "wins": sum(d > 1e-9 for d in diffs), "ties": sum(abs(d) <= 1e-9 for d in diffs),
            "losses": sum(d < -1e-9 for d in diffs),
        }
    return out


# ── blind judgments: win rates and agreement with the rubric ─────────────────

def judgments() -> list[dict]:
    p = Path(__file__).parent / "judgments.json"
    return json.loads(p.read_text(encoding="utf-8"))["judgments"] if p.exists() else []


def judge_agreement(j: dict) -> dict:
    """How often the rubric (core / semantic fraction) prefers the answer the judge picked; ties reported apart."""
    from evals import rubric_v2
    cases = _cases()
    texts = {}
    for setup, fname in j["answers"].items():
        path = RESULTS / fname
        if not path.exists():
            return {}
        for r in json.loads(path.read_text(encoding="utf-8"))["results"]:
            if r["setup"] == setup and r.get("repeat", 0) == 0:
                texts[(setup, r["case"])] = r["text"]
    out = {}
    for name, kinds in (("core", None), ("semantic", ("semantic",))):
        agree = disagree = tie = 0
        for case, winner in j["winners"].items():
            scores = {}
            for s in j["answers"]:
                items = [i for i in rubric_v2.score(texts[(s, case)], cases[case]) if kinds is None or i.kind in kinds]
                scores[s] = sum(i.ok for i in items) / len(items) if items else 1.0
            best = max(scores.values())
            top = [s for s, v in scores.items() if v == best]
            if len(top) > 1:
                tie += 1
            elif top[0] == winner:
                agree += 1
            else:
                disagree += 1
        out[name] = (agree, disagree, tie)
    return out
