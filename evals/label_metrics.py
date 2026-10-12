"""Objective, model-independent measures of each REPRESENTATION the weekly review shows the model.

The same week can be shown as numbers (the original design) or as words made by code (labels v1 ... v5). This module
measures the representations themselves, with no model involved, so the paper can say what changed between attempts
in input terms, not only in output terms. Everything is deterministic (fixed seed) and cheap to recompute.

Representations (the data-dependent text only; the fixed framing sentence and instructions are measured separately
under "prompt load"):
  numbers      the target line + weekly_analysis.format_data_summary (attempts A0 and A1)
  labels_v1    weekly_labels.label_summary
  labels_v2    weekly_labels_v2.label_summary
  labels_v3    weekly_labels_v3.label_summary
  labels_v4    weekly_labels_v4 observations view + recommendations view (each question sees one of them)
  labels_v5    labels_v4 + the recommendation plan (weekly_analysis.recommendation_plan)

Measures (see MEASURE_DOCS for the definitions the report prints).
"""
from __future__ import annotations

import math
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SWEEP_N = 4000
SWEEP_SEED = 2026

MEASURE_DOCS = {
    "tokens": "tokens per week (tiktoken cl100k_base, a proxy: Qwen's own tokenizer is not available offline)",
    "words": "words per week",
    "numbers": "numeric tokens per week (any figure such as 2650, 18:00, 14.0%)",
    "statements": "statements per week: lines and clauses split on . ; and newlines, with at least two words",
    "verdict_coverage": "share of metrics with data whose status against its target is stated explicitly",
    "verdict_fidelity": "share of stated verdicts that agree with the ground truth computed from the raw numbers "
                        "(evals/rubric_v2.reference_state), over the 30 weeks and the random sweep",
    "capacity_bits": f"sum over the label's fields (per metric) of log2(distinct values the field takes) over a random "
                     f"sweep of {SWEEP_N} synthetic weeks (seed {SWEEP_SEED}): how finely the words can describe a "
                     "week. Raw numbers are continuous, so this does not apply to them",
    "discriminability": "mean pairwise distance between the 30 weeks' texts (1 - Jaccard similarity of word trigrams)",
    "flesch": "Flesch reading ease (higher = easier), sentences = lines and full stops",
    "fk_grade": "Flesch-Kincaid grade level",
    "redundancy": "share of repeated word trigrams within a week's text (1 - unique/total)",
    "semantic_elements": "statements of meaning per week: phase consequences, trend-vs-target judgements, links, "
                         "priorities, kinds of fix, watch items, constraints",
}


# ── the representations ──────────────────────────────────────────────────────

def _numbers(data: dict) -> str:
    import weekly_analysis as wa
    g = data["goals"]
    return (f"Goal phase: {g.get('phase', 'unknown')}\nTargets: {int(g.get('step_goal', 0)):,} steps, "
            f"{int(g.get('calories_target', 0)):,} kcal, {int(g.get('protein_target', 0))}g protein\n\n"
            + wa.format_data_summary(data))


def _v4(data: dict) -> str:
    import weekly_labels_v4
    return weekly_labels_v4.observations_view(data) + "\n\n" + weekly_labels_v4.recommendations_view(data)


def _v5(data: dict) -> str:
    import weekly_analysis as wa
    plan = wa.recommendation_plan(data)
    return _v4(data) + "\n\nPlan: " + "; ".join(plan) + "."


def _v6(data: dict) -> str:
    """Same information as v5; the plan is worded as one sentence (that wording lives in the prompt, measured under
    prompt load), so the representation is v5's."""
    return _v5(data)


def representations() -> dict:
    import weekly_labels
    import weekly_labels_v2
    import weekly_labels_v3
    return {
        "numbers": _numbers,
        "labels_v1": weekly_labels.label_summary,
        "labels_v2": weekly_labels_v2.label_summary,
        "labels_v3": weekly_labels_v3.label_summary,
        "labels_v4": _v4,
        "labels_v5": _v5,
        "labels_v6": _v6,
    }


def extra_representations() -> dict:
    """Representations measured and reported but kept out of representations(), whose tests pin that every entry
    except 'numbers' shows no digits. A8 (evals/hybrid_weekly.py): the numeric block plus labels v1's verdicts."""
    from evals import hybrid_weekly
    return {"numbers_verdicts": lambda d: _numbers(d) + "\n\n" + hybrid_weekly.verdict_section(d)}


# ── text measures ────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z']+|\d[\d,.:%]*")
_NUM_RE = re.compile(r"\d[\d,.:%]*")


def _tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s)), "cl100k_base"
    except Exception:
        return lambda s: max(1, round(len(s) / 4)), "chars/4 (tiktoken unavailable)"


TOKENS, TOKENIZER = _tokenizer()


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def statements(text: str) -> int:
    parts = re.split(r"[.;\n]+", text or "")
    return sum(1 for p in parts if len(words(p)) >= 2)


def trigrams(text: str) -> list[tuple]:
    w = [x for x in words(text) if not x[0].isdigit()] or words(text)
    return [tuple(w[i:i + 3]) for i in range(len(w) - 2)]


def redundancy(text: str) -> float:
    t = trigrams(text)
    return 1 - len(set(t)) / len(t) if t else 0.0


def _syllables(word: str) -> int:
    w = word.lower().strip("'")
    if not w or w[0].isdigit():
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def readability(text: str) -> tuple[float, float]:
    """(Flesch reading ease, Flesch-Kincaid grade). Lines count as sentences: label blocks are line-structured."""
    sents = [s for s in re.split(r"[.!?\n]+", text or "") if words(s)]
    ws = words(text)
    if not sents or not ws:
        return 0.0, 0.0
    wps = len(ws) / len(sents)
    spw = sum(_syllables(w) for w in ws) / len(ws)
    return 206.835 - 1.015 * wps - 84.6 * spw, 0.39 * wps + 11.8 * spw - 15.59


def jaccard_distance(a: str, b: str) -> float:
    ta, tb = set(trigrams(a)), set(trigrams(b))
    if not ta and not tb:
        return 0.0
    return 1 - len(ta & tb) / len(ta | tb)


# ── verdicts: coverage and fidelity against the ground truth ─────────────────

def _truth(data: dict) -> dict:
    from evals import rubric_v2
    return rubric_v2.reference_state(data)


def stated_verdicts(name: str, data: dict) -> dict:
    """{metric: status} as the representation states it; a metric missing from the dict has no stated verdict."""
    if name == "numbers":
        text = _numbers(data)
        block = text.split("Vs targets", 1)[1].split("\n\n", 1)[0] if "Vs targets" in text else ""
        out = {}
        for line in block.splitlines():
            m = re.match(r"\s*(Steps|Protein|Calories):.*\((.*?)\)\s*$", line)
            if m:
                key = m.group(1).lower()
                out[key] = "met" if "met" in m.group(2) else "below"
        return out
    import weekly_labels
    lab = weekly_labels.classify(data)          # every label version states v1's verdicts (v2-v5 extend them)
    return {k: lab[k]["status"] for k in ("steps", "protein", "calories") if lab[k]["status"] != "nodata"}


def verdict_scores(name: str, weeks: list[dict]) -> tuple[float, float]:
    """(coverage, fidelity) over the weeks."""
    have = stated = agree = 0
    for d in weeks:
        truth = _truth(d)
        verdicts = stated_verdicts(name, d)
        for k in ("steps", "protein", "calories"):
            if truth[k] == "nodata":
                continue
            have += 1
            if k in verdicts:
                stated += 1
                agree += verdicts[k] == truth[k]
    return (stated / have if have else 0.0), (agree / stated if stated else 0.0)


# ── semantic elements per week ───────────────────────────────────────────────

def semantic_elements(name: str, data: dict) -> int:
    """Statements of meaning (not position): consequences, trend-vs-target, links, priorities, fixes, watch items,
    constraints. Counted from each version's own classification, so the count is what the text carries."""
    if name in ("numbers", "numbers_verdicts", "labels_v1"):
        return 0
    if name == "labels_v2":
        import weekly_labels_v2 as v2
        lab = v2.classify(data)
        needs = v2.attention(lab)
        directions = sum(1 for k in ("steps", "calories", "protein") if lab[k]["status"] != "nodata")
        return len(needs) + directions + 1 + 1          # priorities, directions, advice goal, keep list
    import weekly_labels_v3 as v3
    lab = v3.classify(data)
    needs = v3.attention(lab)
    n = sum(1 for k in ("steps", "calories", "protein") if lab[k].get("meaning"))            # consequences
    n += sum(1 for k in ("steps", "calories", "protein") if lab[k].get("trend_meaning"))     # trend vs target
    n += sum(1 for k in ("steps", "calories", "protein") if lab[k].get("consistency_meaning"))
    n += 1 if lab.get("workout_meaning") else 0
    n += len(v3.links(lab)) + len(needs[:3]) * 2 + len(v3.watch(lab)) + 1                     # links, priority+fix, watch, keep
    if name in ("labels_v5", "labels_v6"):
        import weekly_analysis as wa
        n += len(wa.recommendation_plan(data))
    return n


# ── field resolution over a random sweep ─────────────────────────────────────

def sweep_weeks(n: int = SWEEP_N, seed: int = SWEEP_SEED) -> list[dict]:
    """Synthetic weeks covering the space far more densely than the 30 test weeks (fixed seed: reproducible)."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        phase = rnd.choice(["Fat Loss", "Maintenance", "Muscle Gain"])
        sg, ct, pt = rnd.choice([8000, 9000, 10000]), rnd.choice([2000, 2200, 2500, 2800, 3100]), rnd.choice([130, 146, 150, 170, 180])

        def metric(goal, lo, hi):
            if rnd.random() < 0.05:
                return {"status": "insufficient_data"}
            avg = goal * rnd.uniform(lo, hi)
            return {"status": "ok", "avg": round(avg), "coefficient_of_variation_pct": round(rnd.uniform(2, 45), 1),
                    "slope": round(avg * rnd.uniform(-0.04, 0.04), 1)}

        w = rnd.randint(0, 7)
        out.append({
            "goals": {"phase": phase, "step_goal": sg, "calories_target": ct, "protein_target": pt},
            "steps": metric(sg, 0.45, 1.35), "calories": metric(ct, 0.6, 1.4), "protein": metric(pt, 0.5, 1.5),
            "hourly": {"status": "ok", "peak_hour": rnd.randint(6, 22), "peak_avg_steps": rnd.randint(500, 2500)}
            if rnd.random() > 0.05 else {"status": "no_data"},
            "workouts": [{"date": f"2026-09-{22 + i}", "type": "Upper"} for i in range(w)],
            "goal_hit_days": rnd.randint(0, 7) if rnd.random() > 0.05 else None, "total_days": 7,
        })
    return out


# The fields each version's TEXT actually carries (v4/v5 present v3's fields; v5 adds the plan).
_FIELDS = {
    "labels_v1": ["level", "trend", "consistency"],
    "labels_v2": ["wording", "trend", "consistency", "gap", "direction"],
    "labels_v3": ["wording", "gap", "severity", "trend_meaning", "consistency_meaning", "meaning"],
}


def field_resolution(weeks: list[dict]) -> dict:
    """{version: {field: distinct values seen}} per metric field ('steps.level', ...) plus whole-week fields: how
    finely each version can describe a week. A metric with no data counts as one more value of its fields."""
    import weekly_analysis as wa
    import weekly_labels as v1
    import weekly_labels_v2 as v2
    import weekly_labels_v3 as v3
    mods = {"labels_v1": v1, "labels_v2": v2, "labels_v3": v3, "labels_v4": v3, "labels_v5": v3, "labels_v6": v3}
    res = {}
    for name, mod in mods.items():
        fields = _FIELDS.get(name, _FIELDS["labels_v3"])
        seen: dict[str, set] = {}
        for d in weeks:
            lab = mod.classify(d)
            for k in ("steps", "calories", "protein"):
                for f in fields:
                    seen.setdefault(f"{k}.{f}", set()).add(lab[k].get(f, "nodata" if lab[k]["status"] == "nodata" else ""))
            for f in ("workouts", "goal_days", "peak"):
                seen.setdefault(f, set()).add(str(lab.get(f)))
            if name != "labels_v1" and name != "labels_v2":
                seen.setdefault("workout_meaning", set()).add(str(lab.get("workout_meaning")))
                seen.setdefault("links", set()).add(tuple(v3.links(lab)))
                seen.setdefault("priority_order", set()).add(tuple(v3.attention(lab)[:3]))
                seen.setdefault("watch", set()).add(tuple(k for k, _ in v3.watch(lab)))
            if name == "labels_v2":
                seen.setdefault("priority_order", set()).add(tuple(v2.attention(lab)))
            if name in ("labels_v5", "labels_v6"):
                seen.setdefault("plan", set()).add(tuple(wa.recommendation_plan(d)))
        res[name] = {f: len(v) for f, v in seen.items()}
    return res


def capacity_bits(resolution: dict) -> float:
    return sum(math.log2(n) for n in resolution.values() if n > 1)


# ── prompt load (full prompts, instructions included) ────────────────────────

_DIRECTIVE_RE = re.compile(r"\b(never|do not|don't|must|only|always|exactly|no )\b", re.I)


def prompt_builders() -> dict:
    """{approach: (observations prompt fn, recommendations prompt fn)} as the model receives them."""
    import weekly_analysis as wa
    from evals import hybrid_weekly, legacy_weekly
    s = wa.format_data_summary
    return {
        "old (A0)": (lambda d: legacy_weekly.build_single_prompt(d, s(d)), None),
        "numbers (A1)": (lambda d: wa.build_observations_prompt(d, s(d)),
                         lambda d: wa.build_recommendations_prompt(d, s(d), "OBS")),
        "numbers + verdicts (A8)": (lambda d: wa.build_observations_prompt(d, hybrid_weekly.summary(d)),
                                    lambda d: wa.build_recommendations_prompt(d, hybrid_weekly.summary(d), "OBS")),
        "labels v1": (wa.build_label_observations_prompt, lambda d: wa.build_label_recommendations_prompt(d, "OBS")),
        "labels v2": (wa.build_label2_observations_prompt, lambda d: wa.build_label2_recommendations_prompt(d, "OBS")),
        "labels v3": (wa.build_label3_observations_prompt, lambda d: wa.build_label3_recommendations_prompt(d, "OBS")),
        "labels v4": (wa.build_label4_observations_prompt, lambda d: wa.build_label4_recommendations_prompt(d, "OBS")),
        "labels v5": (wa.build_label4_observations_prompt, lambda d: wa.build_label5_recommendations_prompt(d, "OBS")),
        "labels v6": (wa.build_label4_observations_prompt, lambda d: wa.build_label6_recommendations_prompt(d, "OBS")),
    }


def prompt_load(weeks: list[dict]) -> list[dict]:
    import agent
    real = agent._identity_header
    agent._identity_header = lambda *a, **k: "You are Calvis, a friendly fitness coach. The user is G."
    try:
        rows = []
        for name, (obs_fn, recs_fn) in prompt_builders().items():
            obs = [obs_fn(d) for d in weeks]
            recs = [recs_fn(d) for d in weeks] if recs_fn else []
            rows.append({
                "approach": name,
                "obs_tokens": statistics.mean(TOKENS(p) for p in obs),
                "recs_tokens": statistics.mean(TOKENS(p) for p in recs) if recs else 0.0,
                "directives": statistics.mean(len(_DIRECTIVE_RE.findall(p)) for p in obs + recs) if recs
                else statistics.mean(len(_DIRECTIVE_RE.findall(p)) for p in obs),
                "calls_per_review": 1 if not recs_fn else 2,
            })
        return rows
    finally:
        agent._identity_header = real


# ── everything ───────────────────────────────────────────────────────────────

def _mean_sd(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def measure_all(weeks: list[dict], sweep: list[dict] | None = None) -> dict:
    """{representation: {measure: value or (mean, sd)}} plus 'resolution' and 'prompt_load'."""
    sweep = sweep if sweep is not None else sweep_weeks()
    resolution = field_resolution(sweep)
    out = {}
    for name, fn in {**representations(), **extra_representations()}.items():
        texts = [fn(d) for d in weeks]
        flesch = [readability(t) for t in texts]
        pairs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
        coverage, fidelity = verdict_scores(name, weeks)
        _cov_s, fidelity_sweep = verdict_scores(name, sweep)
        out[name] = {
            "tokens": _mean_sd([TOKENS(t) for t in texts]),
            "words": _mean_sd([len(words(t)) for t in texts]),
            "numbers": _mean_sd([len(_NUM_RE.findall(t)) for t in texts]),
            "statements": _mean_sd([statements(t) for t in texts]),
            "verdict_coverage": coverage,
            "verdict_fidelity": (fidelity, fidelity_sweep),
            "capacity_bits": capacity_bits(resolution[name]) if name in resolution else None,
            "discriminability": statistics.mean(jaccard_distance(texts[i], texts[j]) for i, j in pairs) if pairs else 0.0,
            "flesch": statistics.mean(f for f, _g in flesch),
            "fk_grade": statistics.mean(g for _f, g in flesch),
            "redundancy": statistics.mean(redundancy(t) for t in texts),
            "semantic_elements": _mean_sd([semantic_elements(name, d) for d in weeks]),
        }
    out["resolution"] = resolution
    out["prompt_load"] = prompt_load(weeks)
    out["tokenizer"] = TOKENIZER
    return out


if __name__ == "__main__":
    from evals import weekly_cases, weekly_holdout_cases
    weeks = list({**weekly_cases.CASES, **weekly_holdout_cases.CASES}.values())
    res = measure_all(weeks)
    for name, m in res.items():
        if name in ("resolution", "prompt_load", "tokenizer"):
            continue
        print(name, {k: (tuple(round(x, 2) for x in v) if isinstance(v, tuple) else round(v, 3) if isinstance(v, float) else v)
                     for k, v in m.items()})
    print("resolution", res["resolution"])
    for row in res["prompt_load"]:
        print(row)
    print("tokenizer", res["tokenizer"])
