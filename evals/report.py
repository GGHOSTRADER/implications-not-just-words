"""Build the research report on the weekly-review approaches.

    python -m evals.report                        # evals/REPORT.md + evals/report_figures/*.png
    python -m evals.report --html <file.html>     # also a standalone HTML rendering of the same document

Nothing in the report is typed by hand except the narrative in evals/attempts.py: every number is recomputed on each
build, every saved answer is re-scored with the frozen core rubric (evals/rubric_v2.py), and only runs whose
fingerprint matches an approach's current version are pooled (older runs are used only as a labelled fallback).

The document is built once as a list of blocks and rendered twice (Markdown and HTML), so the two never differ.
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVALS = Path(__file__).parent
OUT = EVALS / "REPORT.md"
FIGS = EVALS / "report_figures"

EXAMPLE_WEEKS = ["07 calories well over target", "H04 cutting too hard, nothing trained"]
CASE_SETS = ("tuning", "holdout")
MSG_SETS = CASE_SETS + ("incident",)      # section 6 also reports the regression cases from a real message


# ── formatting helpers ───────────────────────────────────────────────────────

def pct(x: float | None, digits: int = 0) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def ci(w: tuple) -> str:
    p, lo, hi = w
    return f"{pct(p)} [{pct(lo)}–{pct(hi)}]"


def ms(t: tuple, digits: int = 1) -> str:
    m, s = t
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def table(header: list[str], rows: list[list]) -> str:
    """A Markdown table (kept for callers and tests)."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


# ── data gathering ───────────────────────────────────────────────────────────

def gather() -> dict:
    import agent
    from evals import label_metrics, output_metrics as om, run_weekly_eval as rw, weekly_cases, weekly_holdout_cases

    agent._identity_header = lambda *a, **k: "You are Calvis, a friendly fitness coach. The user is G."
    fps = {s: rw.fingerprint(s) for s in om.APPROACHES}
    rows = om.load_rows()
    current = om.current_rows(rows, fps)

    by_setup_file = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_setup_file[r["setup"]][r["file"]].append(r)
    used, provenance = {}, {}
    for s in om.APPROACHES:
        if current.get(s):
            used[s] = current[s]
            provenance[s] = ("current", sorted({r["file"] for r in current[s]}))
        else:                                       # labelled fallback: the most recent older run per case set
            files = sorted(by_setup_file[s])
            picked = []
            for case_set in CASE_SETS:
                for f in reversed(files):
                    part = [r for r in by_setup_file[s][f] if om.case_set_of(r["case"]) == case_set]
                    if part:
                        picked += part
                        break
            used[s] = picked
            provenance[s] = ("older version" if picked else "not run", sorted({r["file"] for r in picked}))

    summaries = {cs: {} for cs in CASE_SETS}
    for s, rs in used.items():
        for cs in CASE_SETS:
            part = [r for r in rs if om.case_set_of(r["case"]) == cs]
            if part:
                summaries[cs][s] = om.summarize(part, s)

    weeks = list({**weekly_cases.CASES, **weekly_holdout_cases.CASES}.values())
    try:                                            # the checkpoint and briefing study (section 6)
        from evals import message_metrics
        msg = message_metrics.gather()
    except Exception as e:                          # the weekly report must still build
        msg = {"error": repr(e)}
    return {"fps": fps, "used": used, "provenance": provenance, "summaries": summaries,
            "label": label_metrics.measure_all(weeks), "cases": {**weekly_cases.CASES, **weekly_holdout_cases.CASES},
            "msg": msg}


# ── document model ───────────────────────────────────────────────────────────
# blocks: ("h", level, text, anchor) ("p", text) ("ul", [text]) ("table", header, rows, note) ("figure", key, caption)
#         ("details", summary, preformatted text) ("code", text)

def _anchor(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _glance_rows(g: dict):
    """One line per approach: the headline numbers from both case sets."""
    from evals import output_metrics as om
    rows = []
    for s, (aid, name, _rep, _b) in om.APPROACHES.items():
        t, h = g["summaries"]["tuning"].get(s), g["summaries"]["holdout"].get(s)
        cell = lambda m, k: pct(m[k][0]) if m else "–"
        tag = "" if g["provenance"][s][0] == "current" else " ⚠"
        rows.append([aid, name + tag, cell(h, "all_items"), cell(h, "semantic_all"), cell(t, "all_items"),
                     cell(t, "semantic_all"), f"{h['latency'][0]:.1f}" if h else "–",
                     pct(h["redraft_rate"]) if h else "–"])
    return (["id", "approach", "holdout: every core item", "holdout: every semantic item", "tuning: every core item",
             "tuning: every semantic item", "s / review", "redrafted (holdout)"], rows)


def _main_rows(g: dict, cs: str):
    from evals import output_metrics as om
    rows = []
    for s, (aid, name, _rep, _b) in om.APPROACHES.items():
        m = g["summaries"][cs].get(s)
        if not m:
            rows.append([aid, name, "not run", "", "", "", "", "", "", "", ""])
            continue
        k = m["kinds"]
        kind = lambda x: pct(k[x][0] / k[x][1]) if k[x][1] else "–"
        tag = "" if g["provenance"][s][0] == "current" else " ⚠"
        rows.append([aid, name + tag, f"{m['n']} ({m['weeks']}×{m['repeats']})", ci(m["all_items"]),
                     ci(m["semantic_all"]), kind("hallucination"), kind("instruction"), kind("coverage"),
                     kind("specificity"), kind("semantic"), f"{m['latency'][0]:.1f}"])
    return (["id", "approach", "answers (weeks × runs)", "every core item", "every semantic item", "halluc.",
             "instruction", "coverage", "specificity", "semantic", "s / review"], rows)


def _behaviour_rows(g: dict, cs: str):
    from evals import output_metrics as om
    rows = []
    for s, (aid, name, _rep, _b) in om.APPROACHES.items():
        m = g["summaries"][cs].get(s)
        if not m:
            continue
        share, phrase = m["top_phrase"]
        rows.append([aid, name, ms(m["all_items_by_repeat"], 2) if m["repeats"] > 1 else "single run",
                     ms(m["semantic_by_repeat"], 2) if m["repeats"] > 1 else "single run",
                     pct(m["redraft_rate"]), f"{m['calls']:.2f}", f"{m['obs_words']:.0f} / {m['recs_words']:.0f}",
                     f"{m['obs_digits']:.1f}", pct(m["copy_rate"]), f"{m['distinct3']:.2f}", f"{pct(share)} “{phrase}”"])
    return (["id", "approach", "every core item (mean ± sd over runs)", "every semantic item (mean ± sd)",
             "redrafted", "calls", "words obs / recs", "digits in obs", "copied from labels", "distinct-3",
             "most repeated recommendation 5-gram"], rows)


def _item_rows(g: dict, cs: str):
    from evals import output_metrics as om, rubric_v2
    present = [s for s in om.APPROACHES if g["summaries"][cs].get(s)]
    items = [i for i in rubric_v2.ITEM_DOCS if any(i in g["summaries"][cs][s]["items"] for s in present)]
    rows = []
    for item in items:
        cells = []
        for s in present:
            p, t = g["summaries"][cs][s]["items"].get(item, (0, 0))
            cells.append(pct(p / t) if t else "–")
        rows.append([f"`{item}`"] + cells)
    return (["item"] + [om.APPROACHES[s][0] for s in present], rows)


PAIRS = [("split_fast", "old_think"), ("labels_fast", "split_fast"), ("labels2_fast", "labels_fast"),
         ("labels3_fast", "labels2_fast"), ("labels4_fast", "labels3_fast"), ("labels5_fast", "labels4_fast"),
         ("labels6_fast", "labels5_fast"), ("labels6_fast", "labels_fast"), ("labels6_fast", "split_fast"),
         ("hybrid_fast", "split_fast"), ("hybrid_fast", "labels_fast"),
         ("matched6_fast", "labels6_fast"), ("matched6_fast", "split_fast")]


def _paired_rows(g: dict, cs: str):
    from evals import output_metrics as om
    rows = []
    for a, b in PAIRS:
        ra = [r for r in g["used"].get(a, []) if om.case_set_of(r["case"]) == cs]
        rb = [r for r in g["used"].get(b, []) if om.case_set_of(r["case"]) == cs]
        if not ra or not rb:
            continue
        p = om.paired(ra, rb)
        fmt = lambda x: (f"{x['diff']:+.3f} ({x['wins']}/{x['ties']}/{x['losses']}"
                         + (f", p = {x['p']:.3f})" if x["p"] is not None else ")"))
        rows.append([f"{om.APPROACHES[a][0]} vs {om.APPROACHES[b][0]}", p["weeks"], fmt(p["core"]), fmt(p["semantic"])])
    return (["comparison", "weeks", "core items passed: mean difference (wins / ties / losses, Wilcoxon p)",
             "semantic items passed: same"], rows)


def _label_rows(g: dict):
    lab = g["label"]
    reps = [r for r in ("numbers", "numbers_verdicts", "labels_v1", "labels_v2", "labels_v3", "labels_v4", "labels_v5", "labels_v6") if r in lab]
    rows = []
    for r in reps:
        m = lab[r]
        rows.append([r.replace("labels_", "labels "), ms(m["tokens"], 0), ms(m["numbers"], 1), ms(m["statements"], 1),
                     pct(m["verdict_coverage"]), f"{pct(m['verdict_fidelity'][0])} / {pct(m['verdict_fidelity'][1])}",
                     "continuous" if m["capacity_bits"] is None else f"{m['capacity_bits']:.1f}",
                     f"{m['discriminability']:.2f}", f"{m['flesch']:.0f} / {m['fk_grade']:.1f}",
                     f"{m['redundancy']:.2f}", ms(m["semantic_elements"], 1)])
    return (["representation", "tokens", "numbers", "statements", "verdict coverage",
             "verdict fidelity (weeks / sweep)", "capacity (bits)", "discriminability", "Flesch / grade",
             "redundancy", "semantic elements"], rows)


def _resolution_rows(g: dict):
    res = g["label"]["resolution"]
    fields = []
    for v in res.values():
        for f in v:
            if f not in fields:
                fields.append(f)
    versions = list(res)
    return (["field"] + [v.replace("labels_", "") for v in versions],
            [[f"`{f}`"] + [res[v].get(f, "–") for v in versions] for f in fields])


def _prompt_rows(g: dict):
    rows = [[r["approach"], f"{r['obs_tokens']:.0f}", f"{r['recs_tokens']:.0f}" if r["recs_tokens"] else "–",
             f"{r['directives']:.1f}", r["calls_per_review"]] for r in g["label"]["prompt_load"]]
    return (["approach", "observations prompt tokens", "recommendations prompt tokens", "directives per prompt",
             "questions per review"], rows)


def _judgment_rows():
    from evals import output_metrics as om
    rows = []
    for j in om.judgments():
        wins = defaultdict(int)
        for w in j["winners"].values():
            wins[w] += 1
        agree = om.judge_agreement(j)
        fmt = lambda t: f"{t[0]} agree / {t[1]} disagree / {t[2]} tie" if t else "–"
        rows.append([j["id"], j["date"], j["judge"], j["case_set"], len(j["winners"]),
                     ", ".join(f"{om.APPROACHES.get(s, (s,))[0]} {n}" for s, n in sorted(wins.items(), key=lambda x: -x[1])),
                     fmt(agree.get("core")), fmt(agree.get("semantic"))])
    return (["id", "date", "judge", "weeks", "n", "wins", "rubric v2 (all items) agrees with the judge",
             "semantic items agree with the judge"], rows)


def _validation_rows():
    import json
    p = EVALS / "rubric_v2_validation.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    r1p = EVALS / "rubric_v2_reading_round1.json"
    before = json.loads(r1p.read_text(encoding="utf-8"))["precision"] if r1p.exists() else {}
    rows = []
    for k, v in doc["items"].items():
        b = before.get(k)
        rows.append([f"`{k}`", f"{b[0]}/{b[1]}" if b else "–", v["read"], v["correct"], pct(v["precision"])])
    return (["item", "round 1, before fixes", "flags read (final)", "judged correct", "precision"], rows), doc


def _provenance_rows(g: dict):
    from evals import output_metrics as om
    rows = []
    for s, (aid, name, _r, _b) in om.APPROACHES.items():
        kind, files = g["provenance"][s]
        rows.append([aid, name, f"`{g['fps'][s]}`", kind, ", ".join(files) or "–"])
    return (["id", "approach", "current fingerprint", "answers used", "result files"], rows)


def build_doc(g: dict) -> list[tuple]:
    from evals import attempts, label_metrics, output_metrics as om, rubric_v2
    head = _git("rev-parse", "--short", "HEAD")
    dirty = " + uncommitted changes" if _git("status", "--porcelain", "--", "*.py") else ""
    D: list[tuple] = []
    h = lambda level, text: D.append(("h", level, text, _anchor(text)))
    p = lambda text: D.append(("p", text))

    h(1, attempts.TITLE)
    p(f"*Working report, generated {datetime.now():%Y-%m-%d %H:%M} from code at `{head}`{dirty} by "
      f"`python -m evals.report`. Rubric {rubric_v2.RUBRIC_VERSION}; tokens counted with "
      f"{g['label']['tokenizer']}. Change the narrative in `evals/attempts.py`; every number is recomputed.*")
    h(2, "1. Question")
    p(attempts.QUESTION)
    p("**At a glance.** Share of answers passing every item of the core rubric v2, and every semantic item, per "
      "approach (current versions, all runs pooled; ⚠ = an older run until the current version has been run).")
    D.append(("table", *_glance_rows(g), ""))
    if getattr(attempts, "LESSONS", None):
        p("**What the attempts taught so far** (each point refers to the tables in section 5):")
        D.append(("ul", list(attempts.LESSONS)))
    h(2, "2. System, model and data")
    p(attempts.SYSTEM)
    p(attempts.DATA)
    h(2, "3. Attempts")
    D.append(("table", ["id", "setup", "date", "attempt", "representation"],
              [[a["id"], f"`{a['setup']}`", a["date"], a["name"], a["representation"]] for a in attempts.ATTEMPTS], ""))
    for a in attempts.ATTEMPTS:
        p(f"**{a['id']} · {a['name']}.** *What changed:* {a['change']} *Why:* {a['why']} *What happened:* {a['result']}")
    h(2, "4. Instruments")
    D.append(("ul", [f"**{name}.** {text}" for name, text in attempts.INSTRUMENTS]))
    kinds_of = {}
    for case in g["cases"].values():
        for it in rubric_v2.score("x\n\n1. y", case):
            kinds_of[it.id] = it.kind
    p("**Rubric v2 items** (kind, item, definition):")
    D.append(("ul", [f"*{kinds_of.get(i, 'semantic')}* · `{i}`: {d}" for i, d in rubric_v2.ITEM_DOCS.items()]))
    p("**Representation measures:**")
    D.append(("ul", [f"`{k}`: {v}" for k, v in label_metrics.MEASURE_DOCS.items()]))
    p("**Output measures:**")
    D.append(("ul", [f"`{k}`: {v}" for k, v in om.OUTPUT_DOCS.items()]))

    h(2, "5. Results")
    h(3, "5.1 What each approach shows the model (no model involved)")
    D.append(("figure", "representations", "Size, numbers shown and semantic elements per week, by representation."))
    D.append(("table", *_label_rows(g), ""))
    p("Field resolution: distinct values each label field took over the random sweep (per-metric fields are written "
      "`metric.field`). The capacity column above is the sum of log2 of these counts.")
    D.append(("table", *_resolution_rows(g), ""))
    p("Prompt load: the full prompts as sent, instructions included (mean over the 30 weeks).")
    D.append(("table", *_prompt_rows(g), ""))
    h(3, "5.2 Answers, scored with the core rubric v2")
    D.append(("figure", "pass_rates", "Share of answers passing every core item and every semantic item, with 95% "
                                      "Wilson intervals."))
    for cs in CASE_SETS:
        p(f"**{cs.capitalize()} weeks.** ⚠ marks an approach with no run at its current version yet (its latest older "
          "run is shown).")
        D.append(("table", *_main_rows(g, cs), ""))
    h(3, "5.3 Behaviour")
    for cs in CASE_SETS:
        p(f"**{cs.capitalize()} weeks.**")
        D.append(("table", *_behaviour_rows(g, cs), ""))
    h(3, "5.4 Item by item")
    p("Share passed where the item applies.")
    for cs in CASE_SETS:
        p(f"**{cs.capitalize()} weeks.**")
        D.append(("table", *_item_rows(g, cs), ""))
    h(3, "5.5 Paired comparisons")
    p("Same weeks, each week's pass fraction averaged over runs. Wins / ties / losses count weeks.")
    for cs in CASE_SETS:
        p(f"**{cs.capitalize()} weeks.**")
        D.append(("table", *_paired_rows(g, cs), ""))
    h(3, "5.6 Blind judgments and agreement with the rubric")
    D.append(("table", *_judgment_rows(), ""))
    h(3, "5.7 How far the rubric can be trusted")
    val = _validation_rows()
    if val:
        (header, rows), doc = val
        p(f"Precision of the semantic items, measured by reading a seeded sample of what each item flagged on the "
          f"current answers ({doc['date']}, read by {doc['reader']}). Recall (failures the rubric misses) is not "
          "measured. " + doc.get("notes", {}).get("method", ""))
        D.append(("table", header, rows, ""))
        for k in ("respects_target_status", "no_keep_only_recommendation"):
            if doc.get("notes", {}).get(k):
                p(f"`{k}`: {doc['notes'][k]}")
    else:
        p("Not measured yet: run `python -m evals.validate_rubric sample`, read the flags, then `record`.")
    h(3, "5.8 History")
    p("Every evaluation run is a row in `evals/ledger.md` (rubric v1 scores at the time of the run), including the "
      "iterations inside each attempt.")

    h(2, "6. The same principle on the checkpoint comment and the morning briefing")
    _messages_section(D, g)
    h(2, "7. The same comparison on other models")
    _models_section(D, g)
    h(2, "8. Human ratings")
    _human_section(D)
    h(2, "9. Worked examples")
    _examples(D, g)
    h(2, "10. Threats to validity")
    D.append(("ul", list(attempts.THREATS)))
    h(2, "11. Reproducing")
    D.append(("code", "python -m evals.run_weekly_eval --setups <setup,...> --set all --repeats 3   # answers + ledger\n"
                      "python -m evals.label_metrics                                               # representation measures\n"
                      "python -m evals.run_message_eval --setups <setup,...> --set tuning --repeats 3   # checkpoint / briefing\n"
                      "python -m evals.message_metrics                                             # their representation measures\n"
                      "scripts\\run_model_study.ps1                                                  # other models, overnight\n"
                      "python -m evals.report --html report.html                                   # this report"))
    p("Answers used (only runs whose fingerprint equals the approach's current version are pooled):")
    D.append(("table", *_provenance_rows(g), ""))
    return D


def _examples(D: list, g: dict) -> None:
    from evals import label_metrics, output_metrics as om
    reps = label_metrics.representations()
    for week in EXAMPLE_WEEKS:
        d = g["cases"][week]
        D.append(("h", 3, f"Week “{week}”", _anchor(week)))
        D.append(("p", f"Ground truth: phase {d['goals']['phase']}; steps {d['steps'].get('avg', 'no data')} of "
                       f"{d['goals']['step_goal']}; calories {d['calories'].get('avg', 'no data')} of "
                       f"{d['goals']['calories_target']}; protein {d['protein'].get('avg', 'no data')} of "
                       f"{d['goals']['protein_target']}; workouts {len(d['workouts'])}."))
        for name in ("numbers", "labels_v1", "labels_v3", "labels_v6"):
            if name in reps:
                D.append(("details", f"{name.replace('_', ' ')}: what the model is shown", reps[name](d)))
        for s, (aid, name, _r, _b) in om.APPROACHES.items():
            rows = [r for r in g["used"].get(s, []) if r["case"] == week]
            if rows:
                r = sorted(rows, key=lambda x: (x.get("repeat", 0), x["file"]))[0]
                D.append(("details", f"{aid} {name}: answer", r["text"]))


# ── section 6: checkpoint comment and morning briefing ───────────────────────

MESSAGE_EXAMPLES = {"checkpoint": ["01 cut, way over on calories at noon", "H08 nothing logged by evening"],
                    "briefing": ["01 cut, ate over, calories target lowered", "H05 gain phase, under, steps reduced"]}
MESSAGE_PAIRS = {"checkpoint": [("checkpoint_labels_noguard", "checkpoint_numbers"),
                                ("checkpoint_labels", "checkpoint_numbers"),
                                ("checkpoint_labels", "checkpoint_labels_noguard"),
                                ("checkpoint_labels2_noguard", "checkpoint_labels_noguard"),
                                ("checkpoint_labels2", "checkpoint_labels"),
                                ("checkpoint_labels2", "checkpoint_numbers")],
                 "briefing": [("briefing_labels_noguard", "briefing_numbers"), ("briefing_labels", "briefing_numbers"),
                              ("briefing_labels", "briefing_labels_noguard"),
                              ("briefing_labels2_noguard", "briefing_labels_noguard"),
                              ("briefing_labels2", "briefing_labels"),
                              ("briefing_labels2_noguard", "briefing_numbers")]}


def _msg_ids() -> dict:
    from evals import attempts
    return {a["setup"]: a["id"] for a in attempts.MESSAGE_ATTEMPTS}


def _msg_setups(message: str) -> list[str]:
    from evals import run_message_eval as rme
    return [s for s, (_d, m, _mode) in rme.SETUPS.items() if m == message]


def _msg_rep_rows(g: dict):
    from evals import message_metrics as mm
    rep = g["msg"]["rep"]
    rows = []
    for message in ("checkpoint", "briefing"):
        for design in mm.DESIGNS[message]:
            m = rep[message][design]
            s, n = m["coverage"]
            mis, mis_p, total = m["misleading"]
            rows.append([message, design, ms(m["tokens"], 0), ms(m["figures"], 1), f"{pct(s / n)} ({s}/{n})",
                         f"{mis} in {mis_p}/{total} prompts" if message == "checkpoint" and design == "numbers" else "0"])
    return (["message", "design", "prompt tokens", "figures shown", "verdicts stated in words", "misleading statements"],
            rows)


def _msg_main_rows(g: dict, message: str, cs: str):
    ids = _msg_ids()
    rows = []
    for s in _msg_setups(message):
        m = g["msg"]["summaries"][cs].get(s)
        if not m:
            rows.append([ids.get(s, s), f"`{s}`", "not run"] + [""] * 11)
            continue
        k = m["kinds"]
        kind = lambda x: pct(k[x][0] / k[x][1]) if k[x][1] else "–"
        rows.append([ids.get(s, s), f"`{s}`", f"{m['n']} ({m['cases']}×{m['repeats']})", ci(m["all_items"]),
                     ms(m["all_by_repeat"], 2) if m["repeats"] > 1 else "single run", kind("hallucination"),
                     kind("instruction"), kind("semantic"), f"{m['latency'][0]:.1f}", f"{m['calls']:.2f}",
                     pct(m["redraft_rate"]), f"{m['words']:.0f}", f"{m['digits']:.1f}", pct(m["copy_rate"])])
    return (["id", "setup", "answers (cases × runs)", "every item", "every item, mean ± sd over runs", "halluc.",
             "instruction", "semantic", "s / answer", "calls", "redrafted", "words", "figures", "copied from prompt"],
            rows)


def _msg_item_rows(g: dict, message: str, cs: str):
    from evals import message_rubric as mr
    ids = _msg_ids()
    present = [s for s in _msg_setups(message) if g["msg"]["summaries"][cs].get(s)]
    prefix = "cp_" if message == "checkpoint" else "b"
    items = [i for i in mr.ITEM_DOCS if i.startswith(prefix)
             and any(i in g["msg"]["summaries"][cs][s]["items"] for s in present)]
    rows = []
    for item in items:
        cells = []
        for s in present:
            ok, n = g["msg"]["summaries"][cs][s]["items"].get(item, (0, 0))
            cells.append(f"{pct(ok / n)} ({n})" if n else "–")
        rows.append([f"`{item}`"] + cells)
    return (["item (answers where it applies)"] + [ids.get(s, s) for s in present], rows)


def _msg_paired_rows(g: dict, message: str, cs: str):
    from evals import message_metrics as mm
    ids = _msg_ids()
    rows = []
    fmt = lambda p: (f"{p['diff']:+.3f} ({p['wins']}/{p['ties']}/{p['losses']}"
                     + (f", p = {p['p']:.3f})" if p["p"] is not None else ")"))
    for a, b in MESSAGE_PAIRS[message]:
        ra, rb = g["msg"]["by"].get((a, cs), []), g["msg"]["by"].get((b, cs), [])
        if not ra or not rb:
            continue
        p_all, p_items = mm.paired(ra, rb, "all"), mm.paired(ra, rb, "items")
        rows.append([f"{ids.get(a, a)} vs {ids.get(b, b)}", p_all["cases"], fmt(p_all), fmt(p_items)])
    return (["comparison", "cases", "answer passes every item: mean difference per case (wins / ties / losses, "
             "Wilcoxon p)", "share of items passed: same"], rows)


def _msg_validation():
    import json
    p = EVALS / "message_rubric_validation.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    rows = [[f"`{k}`", v["read"], v["correct"], pct(v["precision"]), v.get("note", "")] for k, v in doc["items"].items()]
    return (["item", "flags read", "judged correct", "precision", "what the wrong flags were"], rows), doc


def _messages_section(D: list, g: dict) -> None:
    from evals import attempts, message_metrics as mm, message_rubric as mr
    h = lambda level, text: D.append(("h", level, text, _anchor(text)))
    p = lambda text: D.append(("p", text))
    msg = g.get("msg") or {}
    if "error" in msg:
        p(f"*Not built: {msg['error']}*")
        return
    p(attempts.MESSAGES_INTRO)
    p(attempts.MESSAGES_DATA)
    D.append(("table", ["id", "setup", "attempt", "representation"],
              [[a["id"], f"`{a['setup']}`", a["name"], a["representation"]] for a in attempts.MESSAGE_ATTEMPTS], ""))
    for a in attempts.MESSAGE_ATTEMPTS:
        p(f"**{a['id']} · {a['name']}.** *What changed:* {a['change']} *Why:* {a['why']}"
          + (f" *What happened:* {a['result']}" if a["result"] else ""))
    if attempts.MESSAGE_LESSONS:
        p("**What this part showed:**")
        D.append(("ul", list(attempts.MESSAGE_LESSONS)))

    h(3, "6.1 Instruments")
    p(f"Checklist {mr.MESSAGE_RUBRIC_VERSION} (`evals/message_rubric.py`): items apply only where they mean something "
      "(the outdoor rule only on wet or unknown days), so compare per-item rates. Every saved answer is re-scored with "
      "the current version on each build, and only runs at a setup's current fingerprint are pooled.")
    from evals import briefing_cases, checkpoint_cases
    kinds = {}
    for c in checkpoint_cases.cases("all").values():
        for it in mr.score_checkpoint("x", c):
            kinds[it.id] = it.kind
    for c in briefing_cases.cases("all").values():
        for it in mr.score_briefing({"yesterday": "x", "today": "x"}, c):
            kinds[it.id] = it.kind
    D.append(("ul", [f"*{kinds.get(i, '?')}* · `{i}`: {d}" for i, d in mr.ITEM_DOCS.items()]))
    p("**Representation measures** (`evals/message_metrics.py`, no model involved):")
    D.append(("ul", [f"`{k}`: {v}" for k, v in mm.MEASURE_DOCS.items()]))
    p("**Verdicts each message needs:** checkpoint: " + mm.NEEDED["checkpoint"] + ". Briefing: " + mm.NEEDED["briefing"]
      + ".")

    h(3, "6.2 What each design shows the model (no model involved)")
    D.append(("table", *_msg_rep_rows(g), f"tokens: {msg['rep']['tokenizer']}"))
    for message in ("checkpoint", "briefing"):
        miss = msg["rep"][message]["numbers"]["missing"]
        if miss:
            p(f"**{message.capitalize()}, numbers design: verdicts left for the model to work out** (cases where the "
              "verdict was needed and not stated): "
              + "; ".join(f"{k} {v}" for k, v in sorted(miss.items(), key=lambda kv: -kv[1])) + ".")

    h(3, "6.3 Answers")
    D.append(("figure", "message_pass_rates", "Share of answers passing every applicable checklist item, per setup, "
                                              "with 95% Wilson intervals."))
    for message in ("checkpoint", "briefing"):
        for cs in MSG_SETS:
            p(f"**{message.capitalize()}, {cs} cases.**")
            D.append(("table", *_msg_main_rows(g, message, cs), ""))

    h(3, "6.4 Item by item")
    for message in ("checkpoint", "briefing"):
        for cs in MSG_SETS:
            p(f"**{message.capitalize()}, {cs} cases.** Share passed where the item applies (answers in brackets).")
            D.append(("table", *_msg_item_rows(g, message, cs), ""))

    h(3, "6.5 Paired comparisons")
    p("Same cases, each case's value averaged over its runs: whether the answer passed every item, and the share of "
      "applicable items it passed. Wins / ties / losses count cases. With 10 or 20 cases these tests have little "
      "power, and most items pass under every design, so read them with the item table above.")
    for message in ("checkpoint", "briefing"):
        for cs in MSG_SETS:
            rows = _msg_paired_rows(g, message, cs)
            if rows[1]:
                p(f"**{message.capitalize()}, {cs} cases.**")
                D.append(("table", *rows, ""))

    h(3, "6.6 How far the checklist can be trusted")
    val = _msg_validation()
    if val:
        (header, rows), doc = val
        p(f"Precision of each item, measured by reading a seeded sample of what it flagged on the saved answers "
          f"({doc['date']}, read by {doc['reader']}). Recall (failures the checklist misses) is not measured. "
          + doc.get("method", ""))
        D.append(("table", header, rows, ""))
    else:
        p("Not measured yet.")

    h(3, "6.7 Examples")
    from evals import run_message_eval as rme
    ids = _msg_ids()
    for message, names in MESSAGE_EXAMPLES.items():
        cases = rme._cases(message, "all")
        for name in names:
            case = cases[name]
            D.append(("h", 4, f"{message.capitalize()} “{name}”", _anchor(f"{message} {name}")))
            for s in [x for x in _msg_setups(message) if rme.SETUPS[x][2] in ("numbers", "labels", "labels2")]:
                D.append(("details", f"{ids.get(s, s)} {s}: what the model is shown",
                          "\n\n────────\n\n".join(rme.prompts(s, case))))
            for s in _msg_setups(message):
                rs = sorted((r for r in msg["rows"] if r["setup"] == s and r["case"] == name),
                            key=lambda r: (r.get("repeat", 0), r["file"]))
                if rs:
                    r = rs[0]
                    ans = r["answer"] if isinstance(r["answer"], str) else (
                        f"[yesterday]\n{r['answer'].get('yesterday', '')}\n\n[today]\n{r['answer'].get('today', '')}")
                    bad = [f"{i.id}: {i.detail}" for i in r["now"] if not i.ok]
                    D.append(("details", f"{ids.get(s, s)} {s}: answer (run {r.get('repeat', 0) + 1})"
                              + (f" · fails {len(bad)}" if bad else " · passes every item"),
                              ans + ("\n\nFailed items:\n" + "\n".join(bad) if bad else "")))
    p("Answers used: " + (", ".join(f"`{f}`" for f in msg["files"]) or "none yet") + ". Current fingerprints: "
      + ", ".join(f"{ids.get(s, s)} `{fp}`" for s, fp in msg["fps"].items()) + ".")


def _msg_pass_series(g: dict):
    msg = g.get("msg") or {}
    out = {}
    for message in ("checkpoint", "briefing"):
        setups = _msg_setups(message)
        out[message] = (setups, {cs: [msg.get("summaries", {}).get(cs, {}).get(s, {}).get("all_items") for s in setups]
                                 for cs in CASE_SETS})
    return out


def svg_message_pass_rates(g: dict) -> str:
    if "error" in (g.get("msg") or {"error": 1}):
        return ""
    ids = _msg_ids()
    W, H, top, bottom, left = 360, 230, 26, 34, 34
    panels = []
    for message, (setups, series) in _msg_pass_series(g).items():
        group = (W - left - 10) / max(len(setups), 1)
        bw = group * 0.34
        y = lambda v: top + (H - top - bottom) * (1 - v)
        parts = [f'<text x="{left}" y="16" class="ct">{message}</text>']
        for t in (0, 0.25, 0.5, 0.75, 1):
            parts.append(f'<line x1="{left}" x2="{W - 10}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
                         f'<text x="{left - 5}" y="{y(t) + 3:.1f}" class="tick" text-anchor="end">{int(t * 100)}%</text>')
        for i, s in enumerate(setups):
            cx = left + group * i + group / 2
            for k, cs in enumerate(CASE_SETS):
                v = series[cs][i]
                if not v:
                    continue
                x = cx + (k - 1) * bw + 1
                parts.append(f'<rect x="{x:.1f}" y="{y(v[0]):.1f}" width="{bw - 2:.1f}" '
                             f'height="{max(y(0) - y(v[0]), 0.5):.1f}" class="bar{k}"><title>{ids.get(s, s)} {cs}: '
                             f'{pct(v[0])} [{pct(v[1])}–{pct(v[2])}]</title></rect>')
                mx = x + (bw - 2) / 2
                parts.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{y(v[2]):.1f}" y2="{y(v[1]):.1f}" class="err"/>')
            parts.append(f'<text x="{cx:.1f}" y="{H - bottom + 14}" class="tick" text-anchor="middle">{ids.get(s, s)}</text>')
        panels.append(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{message} pass rates">{"".join(parts)}</svg>')
    legend = ('<div class="legend"><span><i class="sw bar0"></i>tuning cases</span>'
              '<span><i class="sw bar1"></i>holdout cases</span><span>whiskers: 95% Wilson interval</span></div>')
    return f'<div class="chart two">{"".join(panels)}</div>{legend}'


def png_message_figure(g: dict) -> list[str]:
    if "error" in (g.get("msg") or {"error": 1}):
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    ids = _msg_ids()
    FIGS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (message, (setups, series)) in zip(axes, _msg_pass_series(g).items()):
        for k, (cs, color) in enumerate(zip(CASE_SETS, ("#3E6E8E", "#C9853A"))):
            vals = series[cs]
            ps = [v[0] if v else 0 for v in vals]
            lo = [max(0.0, v[0] - v[1]) if v else 0 for v in vals]
            hi = [max(0.0, v[2] - v[0]) if v else 0 for v in vals]
            ax.bar([i + (k - 0.5) * 0.38 for i in range(len(setups))], ps, width=0.38, label=f"{cs} cases",
                   color=color, yerr=[lo, hi], capsize=2)
        ax.set_xticks(range(len(setups)), [ids.get(s, s) for s in setups])
        ax.set_title(message)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("share passing every item")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=False)   # above: bars reach 100%
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "message_pass_rates.png", dpi=130)
    plt.close(fig)
    return ["message_pass_rates.png"]


# ── section 7: other models ──────────────────────────────────────────────────

def _models_section(D: list, g: dict) -> None:
    from evals import model_study
    p = lambda text: D.append(("p", text))
    try:
        study = model_study.gather(g)
    except Exception as e:                      # the report must still build
        p(f"*Not built: {e!r}*")
        return
    others = [m for m in study if m != model_study.DEFAULT_MODEL]
    p("Does the gain from words come from this one model? `scripts/run_model_study.ps1` repeats the key comparisons on "
      "other models served by Ollama (the same cases, three runs each, the same checklists), writing each model to its own "
      "folder under `evals/results/models/` so nothing can mix into the qwen3:8b results above. Designs compared: weekly "
      "review A1 numbers, A2 labels v1, A7 labels v6, A8 numbers + verdicts; checkpoint C0 numbers and C6 labels v3 with "
      "checks; briefing B0 numbers, B1 labels v1 and B8 labels v4 with checks. The labels designs were written and tuned "
      "on qwen3:8b only.")
    if not others:
        p("**Not run yet.** Only qwen3:8b has results. Run `scripts/run_model_study.ps1` (about 4 hours for four models; "
          "see `paper/README.md`) and rebuild this report.")
    for cs in ("holdout", "tuning"):
        p(f"**Share of answers passing every item, {cs} cases** (95% Wilson interval, number of answers).")
        D.append(("table", *model_study.cells_table(study, cs), ""))
    p("**Labels minus numbers**, tuning and held-out cases together: mean difference in the per-case share of answers "
      "passing every item (cases better / equal / worse, Wilcoxon p).")
    D.append(("table", *model_study.pairs_table(study, "pooled"), ""))
    p("The same on the 10 held-out cases only.")
    D.append(("table", *model_study.pairs_table(study, "holdout"), ""))
    for m in others:
        man = study[m].get("manifest") or {}
        if man:
            p(f"`{m}`: run {man.get('started', '?')}, code `{man.get('code', '?')}`, model id "
              f"`{man.get('model_id', '?')}`, Ollama {man.get('ollama', '?')}.")
    for m, why in model_study.EXCLUDED.items():
        p(f"**Not in the tables: `{m}`** — {why}.")
    chat = model_study.chat_summary()
    if chat:
        p("**The chat agent on each model.** The original chat design (the raw figures through three tools, canned dumps, "
          "thinking mode in production) against `chat_facts.py` (the day as words, no tools, answer checked and redrafted twice "
          "at most), on 66 fake days and questions, one run each, scored by `evals/chat_rubric.py` (a case passes when no "
          "applicable item fails). Two caveats. The original uses thinking mode only on qwen3:8b; every other model runs it "
          "on the fast backend, because Ollama refuses thinking on them. And the original is a tool-using agent: gemma3:4b "
          "has no tool support in Ollama, so most of its original-design answers are API errors (counted apart; only the "
          "cases a Python shortcut answers without the model succeed), not a verdict on the design. The words design was written and tuned on qwen3:8b only.")
        D.append(("table", *model_study.chat_table(chat), ""))


# ── section 8: human ratings ─────────────────────────────────────────────────

# (key file, folder holding that kit's rating files). The keys and ratings are git-ignored (rater names, comments).
_RATING_KITS = [
    (ROOT / "evals" / "human_rating_kit" / "submissions" / "key_calvis-ratings-202610052124_rebuilt.json",
     ROOT / "evals" / "human_rating_kit" / "submissions"),
    (ROOT / "evals" / "human_rating_kit_v2" / "key.json", ROOT / "evals" / "human_rating_kit_v2" / "submissions"),
]


def human_tallies() -> list[dict]:
    """One tally per kit that has a key and ratings on this machine (empty list when the ignored files are absent)."""
    from evals import human_rating
    out = []
    for key, folder in _RATING_KITS:
        files = sorted(folder.glob("ratings_*.json")) if folder.exists() else []
        if key.exists() and files:
            out.append(human_rating.tally(key, files))
    return out


def _human_section(D: list) -> None:
    """Counts, shares, intervals and anonymous rater labels only: no nicknames and no comments leave the ignored folders."""
    from evals import human_rating
    p = lambda text: D.append(("p", text))
    try:
        tallies = human_tallies()
    except Exception as e:                      # the report must still build
        p(f"*Not built: {e!r}*")
        return
    if not tallies:
        p("*No rating files on this machine (they are git-ignored); the human-ratings section is empty.*")
        return
    names = sorted({r["rater"] for t in tallies for r in t["raters"]}, key=str.lower)
    anon = {n: f"P{i}" for i, n in enumerate(names, 1)}      # fixed by sorted nickname, so the label never changes between builds
    p(f"{len(names)} raters, {len(tallies)} kits (blind A/B pairs, order randomised; the second kit asks two questions, the first three). "
      "Only items where the rater chose a side are counted; ties and \"both poor\" are shown apart. Items from one rater are "
      "not independent, so the intervals are optimistic. Rater labels are anonymous.")
    questions = ["overall", "accuracy"]
    for q in questions:
        rows, total = [], [0, 0, 0, 0]
        for message in human_rating.PAIRS:
            w = o = e = b = 0
            for t in tallies:
                c = t["by_message"].get(message, {}).get(q, {}).get("counts", {})
                w, o, e, b = w + c.get("labels", 0), o + c.get("numbers", 0), e + c.get("equal", 0), b + c.get("both_poor", 0)
            total = [total[0] + w, total[1] + o, total[2] + e, total[3] + b]
            pr, lo, hi = human_rating._wilson(w, w + o)
            rows.append([message, w, o, e, b, f"{pr:.0%}" if w + o else "-", f"{lo:.0%}-{hi:.0%}" if w + o else "-"])
        pr, lo, hi = human_rating._wilson(total[0], total[0] + total[1])
        rows.append(["all four", *total, f"{pr:.0%}", f"{lo:.0%}-{hi:.0%}"])
        p(f"**Question `{q}`** ({human_rating.QUESTIONS[q]}) Counts of items, words design against the original prompt.")
        D.append(("table", ["message", "words", "original", "equal", "both poor", "words share of decisive", "95% interval"], rows, ""))
    alphas = [a for t in tallies for m in t["agreement"].values() for q, a in m.items() if q == "overall" and a is not None]
    if alphas:
        p(f"Agreement between raters (Krippendorff's alpha, nominal, per kit and message, `overall`): "
          f"{min(alphas):.2f} to {max(alphas):.2f}.")
    rows = []
    for k, t in enumerate(tallies, 1):
        for r in t["raters"]:
            same, tot = t["consistency"].get(r["rater"], (0, 0))
            rows.append([anon[r["rater"]], r["background"], f"kit {k}", f"{same} of {tot}"])
    rows.sort(key=lambda row: int(row[0][1:]))
    p("**Consistency on repeated items** (same pair shown twice with the sides swapped; same side chosen both times).")
    D.append(("table", ["rater", "background", "kit", "consistent"], rows, ""))
    p("The first kit's answer key was overwritten after it was sent and was rebuilt from the saved page and the saved evaluation answers "
      "(each rater item matched uniquely; checked against the first tally).")


# ── Markdown renderer ────────────────────────────────────────────────────────

def render_md(D: list[tuple], figs: list[str]) -> str:
    out = []
    for b in D:
        kind = b[0]
        if kind == "h":
            out.append(f"{'#' * b[1]} {b[2]}\n")
        elif kind == "p":
            out.append(b[1] + "\n")
        elif kind == "ul":
            out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif kind == "table":
            out.append(table(b[1], b[2]) + "\n")
        elif kind == "figure":
            if f"{b[1]}.png" in figs:
                out.append(f"![{b[2]}](report_figures/{b[1]}.png)\n")
        elif kind == "details":
            out.append(f"<details><summary>{html.escape(b[1])}</summary>\n\n```text\n{b[2]}\n```\n</details>\n")
        elif kind == "code":
            out.append(f"```text\n{b[1]}\n```\n")
    return "\n".join(out)


# ── PNG figures (Markdown) and SVG charts (HTML) share the same series ───────

def _pass_series(g: dict):
    from evals import output_metrics as om
    order = [s for s in om.APPROACHES if any(s in g["summaries"][cs] for cs in CASE_SETS)]
    series = {}
    for cs in CASE_SETS:
        for metric in ("all_items", "semantic_all"):
            series[(cs, metric)] = [g["summaries"][cs][s][metric] if s in g["summaries"][cs] else None for s in order]
    return [om.APPROACHES[s][0] for s in order], series


def _rep_series(g: dict):
    lab = g["label"]
    reps = [r for r in ("numbers", "numbers_verdicts", "labels_v1", "labels_v2", "labels_v3", "labels_v4", "labels_v5", "labels_v6") if r in lab]
    return ([r.replace("labels_", "") for r in reps], [lab[r]["tokens"][0] for r in reps],
            [lab[r]["semantic_elements"][0] for r in reps], [lab[r]["numbers"][0] for r in reps])


def png_figures(g: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    FIGS.mkdir(exist_ok=True)
    names, series = _pass_series(g)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, cs in zip(axes, CASE_SETS):
        for k, (metric, label, color) in enumerate((("all_items", "every core item", "#3E6E8E"),
                                                    ("semantic_all", "every semantic item", "#C9853A"))):
            vals = series[(cs, metric)]
            ps = [v[0] if v else 0 for v in vals]
            lo = [max(0.0, v[0] - v[1]) if v else 0 for v in vals]     # clamp float noise at p = 0 or 1
            hi = [max(0.0, v[2] - v[0]) if v else 0 for v in vals]
            ax.bar([i + (k - 0.5) * 0.38 for i in range(len(names))], ps, width=0.38, label=label, color=color,
                   yerr=[lo, hi], capsize=2)
        ax.set_xticks(range(len(names)), names)
        ax.set_title(f"{cs} weeks")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("share of answers")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "pass_rates.png", dpi=130)
    plt.close(fig)

    reps, tokens, sem, nums = _rep_series(g)
    fig, ax1 = plt.subplots(figsize=(7, 3.4))
    ax1.bar(range(len(reps)), tokens, color="#9DB4C8", label="tokens per week")
    ax1.set_ylabel("tokens per week")
    ax2 = ax1.twinx()
    ax2.plot(range(len(reps)), sem, "o-", color="#C9853A", label="semantic elements")
    ax2.plot(range(len(reps)), nums, "s--", color="#3E6E8E", label="numbers shown")
    ax2.set_ylabel("count per week")
    ax1.set_xticks(range(len(reps)), reps)
    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax1.legend(handles, [x.get_label() for x in handles], fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "representations.png", dpi=130)
    plt.close(fig)
    return ["pass_rates.png", "representations.png"] + png_message_figure(g)


def svg_pass_rates(g: dict) -> str:
    names, series = _pass_series(g)
    W, H, top, bottom, left, gap = 360, 230, 26, 34, 34, 18
    panels = []
    for cs in CASE_SETS:
        n = len(names)
        plot_w = W - left - 10
        group = plot_w / max(n, 1)
        bw = group * 0.34
        y = lambda v: top + (H - top - bottom) * (1 - v)
        parts = [f'<text x="{left}" y="16" class="ct">{cs} weeks</text>']
        for t in (0, 0.25, 0.5, 0.75, 1):
            parts.append(f'<line x1="{left}" x2="{W - 10}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
                         f'<text x="{left - 5}" y="{y(t) + 3:.1f}" class="tick" text-anchor="end">{int(t * 100)}%</text>')
        for i, name in enumerate(names):
            cx = left + group * i + group / 2
            for k, metric in enumerate(("all_items", "semantic_all")):
                v = series[(cs, metric)][i]
                if not v:
                    continue
                x = cx + (k - 1) * bw + 1
                parts.append(f'<rect x="{x:.1f}" y="{y(v[0]):.1f}" width="{bw - 2:.1f}" '
                             f'height="{max(y(0) - y(v[0]), 0.5):.1f}" class="bar{k}"><title>{name} {cs}: '
                             f'{pct(v[0])} [{pct(v[1])}–{pct(v[2])}]</title></rect>')
                mx = x + (bw - 2) / 2
                parts.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{y(v[2]):.1f}" y2="{y(v[1]):.1f}" class="err"/>')
            parts.append(f'<text x="{cx:.1f}" y="{H - bottom + 14}" class="tick" text-anchor="middle">{name}</text>')
        panels.append(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="pass rates, {cs} weeks">{"".join(parts)}</svg>')
    legend = ('<div class="legend"><span><i class="sw bar0"></i>every core item</span>'
              '<span><i class="sw bar1"></i>every semantic item</span><span>whiskers: 95% Wilson interval</span></div>')
    return f'<div class="chart two">{"".join(panels)}</div>{legend}'


def svg_representations(g: dict) -> str:
    reps, tokens, sem, nums = _rep_series(g)
    W, H, top, bottom, left, right = 560, 230, 20, 34, 40, 40
    n = len(reps)
    group = (W - left - right) / n
    tmax = max(tokens) * 1.15
    cmax = max(max(sem), max(nums)) * 1.15 or 1
    yt = lambda v: top + (H - top - bottom) * (1 - v / tmax)
    yc = lambda v: top + (H - top - bottom) * (1 - v / cmax)
    parts = []
    for frac in (0, 0.5, 1):
        v = tmax / 1.15 * frac
        parts.append(f'<line x1="{left}" x2="{W - right}" y1="{yt(v):.1f}" y2="{yt(v):.1f}" class="grid"/>'
                     f'<text x="{left - 5}" y="{yt(v) + 3:.1f}" class="tick" text-anchor="end">{v:.0f}</text>')
        c = cmax / 1.15 * frac
        parts.append(f'<text x="{W - right + 5}" y="{yc(c) + 3:.1f}" class="tick">{c:.0f}</text>')
    pts_sem, pts_num = [], []
    for i, name in enumerate(reps):
        cx = left + group * i + group / 2
        parts.append(f'<rect x="{cx - group * 0.3:.1f}" y="{yt(tokens[i]):.1f}" width="{group * 0.6:.1f}" '
                     f'height="{max(yt(0) - yt(tokens[i]), 0.5):.1f}" class="bar2"><title>{name}: {tokens[i]:.0f} '
                     f'tokens</title></rect>')
        parts.append(f'<text x="{cx:.1f}" y="{H - bottom + 14}" class="tick" text-anchor="middle">{name}</text>')
        pts_sem.append(f"{cx:.1f},{yc(sem[i]):.1f}")
        pts_num.append(f"{cx:.1f},{yc(nums[i]):.1f}")
    parts.append(f'<polyline points="{" ".join(pts_sem)}" class="line1"/>')
    parts.append(f'<polyline points="{" ".join(pts_num)}" class="line0"/>')
    for pt in pts_sem:
        x, yv = pt.split(",")
        parts.append(f'<circle cx="{x}" cy="{yv}" r="3" class="dot1"/>')
    for pt in pts_num:
        x, yv = pt.split(",")
        parts.append(f'<rect x="{float(x) - 3:.1f}" y="{float(yv) - 3:.1f}" width="6" height="6" class="dot0"/>')
    legend = ('<div class="legend"><span><i class="sw bar2"></i>tokens per week (left axis)</span>'
              '<span><i class="sw line1"></i>semantic elements per week (right)</span>'
              '<span><i class="sw line0"></i>numbers shown per week (right)</span></div>')
    return (f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="representations">'
            f'{"".join(parts)}</svg></div>{legend}')


# ── HTML renderer ────────────────────────────────────────────────────────────

def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    return t


_CSS = """
/* Layout: one reading column (about 70ch) for prose; tables and charts may run wider inside their own scroll boxes. */
:root {
  --paper: #f5f6f4; --ink: #1d2329; --muted: #5c6670; --rule: #d8dcd9; --panel: #ffffff;
  --accent: #2f6a86; --accent-2: #b8741f; --bar2: #9fb5c6; --warn: #9a5b00;
  --display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --body: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #14181b; --ink: #e5e8ea; --muted: #9aa5ad; --rule: #2c3338; --panel: #1b2024;
  --accent: #7fb3cf; --accent-2: #e0a65a; --bar2: #4c6577; --warn: #e3a74f; color-scheme: dark } }
:root[data-theme="dark"] {
  --paper: #14181b; --ink: #e5e8ea; --muted: #9aa5ad; --rule: #2c3338; --panel: #1b2024;
  --accent: #7fb3cf; --accent-2: #e0a65a; --bar2: #4c6577; --warn: #e3a74f; color-scheme: dark }
body { background: var(--paper); color: var(--ink); font: 16px/1.6 var(--body); padding-inline: 16px; padding-block: 28px 64px; }
main { max-width: 1100px; margin: 0 auto; display: grid; gap: 14px; }
main > p, main > ul, main > .meta, main > h1, main > h2, main > h3 { max-width: 72ch; }
h1, h2, h3 { font-family: var(--display); line-height: 1.2; text-wrap: balance; margin: 0; font-weight: 600; }
h1 { font-size: clamp(1.7rem, 4vw, 2.5rem); }
h2 { font-size: 1.5rem; margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--rule); }
h3 { font-size: 1.15rem; margin-top: 12px; }
p, ul { margin: 0; }
ul { padding-left: 1.2em; display: grid; gap: 6px; }
.meta { color: var(--muted); font-size: 0.9rem; }
code, pre, .tbl td, .tbl th, .tick, .ct { font-family: var(--mono); }
code { font-size: 0.86em; background: color-mix(in srgb, var(--accent) 10%, transparent); padding: 0 4px; border-radius: 3px; }
a { color: var(--accent); }
nav.toc { font-size: 0.92rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 16px; max-width: 72ch; }
nav.toc a { text-decoration: none; }
.scroll { overflow-x: auto; min-width: 0; border: 1px solid var(--rule); background: var(--panel); border-radius: 6px; }
.tbl { border-collapse: collapse; font-size: 0.8rem; width: max-content; min-width: 100%; font-variant-numeric: tabular-nums; }
.tbl th { text-align: left; font-weight: 600; color: var(--muted); background: color-mix(in srgb, var(--rule) 45%, var(--panel));
  text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.7rem; }
.tbl th, .tbl td { padding: 6px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
.tbl td { max-width: 46ch; }
.tbl tr:last-child td { border-bottom: 0; }
.warn { color: var(--warn); }
details { border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); max-width: 100%; }
details > summary { cursor: pointer; padding: 8px 12px; font-size: 0.92rem; }
details > summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
pre { margin: 0; padding: 10px 12px; overflow-x: auto; font-size: 0.8rem; line-height: 1.5; white-space: pre-wrap;
  border-top: 1px solid var(--rule); }
.chart { overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; }
.chart.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 8px; }
.chart svg { width: 100%; height: auto; display: block; min-width: 300px; }
.grid { stroke: var(--rule); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.ct { fill: var(--ink); font-size: 11px; font-weight: 600; }
.bar0 { fill: var(--accent); } .bar1 { fill: var(--accent-2); } .bar2 { fill: var(--bar2); }
.err { stroke: var(--ink); stroke-width: 1; opacity: 0.6; }
.line1 { fill: none; stroke: var(--accent-2); stroke-width: 2; } .line0 { fill: none; stroke: var(--accent); stroke-width: 2; stroke-dasharray: 5 3; }
.dot1 { fill: var(--accent-2); } .dot0 { fill: var(--accent); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.82rem; color: var(--muted); }
.legend .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.legend .sw.line1 { background: var(--accent-2); height: 3px; vertical-align: 3px; }
.legend .sw.line0 { background: var(--accent); height: 3px; vertical-align: 3px; }
.legend .sw.bar0 { background: var(--accent); } .legend .sw.bar1 { background: var(--accent-2); } .legend .sw.bar2 { background: var(--bar2); }
figcaption { font-size: 0.85rem; color: var(--muted); max-width: 72ch; }
figure { margin: 0; display: grid; gap: 6px; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
"""


def render_html(D: list[tuple], g: dict, title: str) -> str:
    charts = {"pass_rates": svg_pass_rates(g), "representations": svg_representations(g),
              "message_pass_rates": svg_message_pass_rates(g)}
    toc = [b for b in D if b[0] == "h" and b[1] == 2]
    out = [f"<title>{html.escape(title)}</title>",
           '<link rel="preconnect" href="https://fonts.googleapis.com">',
           '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
           'family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&display=swap">',
           f"<style>{_CSS}</style>", "<main>"]
    first_h1 = True
    for b in D:
        kind = b[0]
        if kind == "h":
            out.append(f'<h{b[1]} id="{b[3]}">{_inline(b[2])}</h{b[1]}>')
            if b[1] == 1 and first_h1:
                first_h1 = False
                out.append('<nav class="toc" aria-label="Sections">'
                           + "".join(f'<a href="#{t[3]}">{_inline(t[2])}</a>' for t in toc) + "</nav>")
        elif kind == "p":
            cls = ' class="meta"' if b[1].startswith("*Working report") else ""
            out.append(f"<p{cls}>{_inline(b[1])}</p>")
        elif kind == "ul":
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in b[1]) + "</ul>")
        elif kind == "table":
            head = "".join(f"<th>{_inline(str(c))}</th>" for c in b[1])
            body = "".join("<tr>" + "".join(
                f"<td{' class=\"warn\"' if '⚠' in str(c) else ''}>{_inline(str(c))}</td>" for c in r) + "</tr>"
                for r in b[2])
            out.append(f'<div class="scroll"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody>'
                       "</table></div>")
        elif kind == "figure":
            out.append(f"<figure>{charts.get(b[1], '')}<figcaption>{_inline(b[2])}</figcaption></figure>")
        elif kind == "details":
            out.append(f"<details><summary>{_inline(b[1])}</summary><pre>{html.escape(b[2])}</pre></details>")
        elif kind == "code":
            out.append(f'<div class="scroll"><pre>{html.escape(b[1])}</pre></div>')
    out.append("</main>")
    return "\n".join(out)


# ── entry point ──────────────────────────────────────────────────────────────

def build() -> str:
    """The Markdown report (kept for callers and tests)."""
    g = gather()
    return render_md(build_doc(g), png_figures(g))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", type=Path, help="also write a standalone HTML rendering here")
    args = ap.parse_args(argv)
    g = gather()
    D = build_doc(g)
    md = render_md(D, png_figures(g))
    OUT.write_text(md, encoding="utf-8")
    print(f"wrote {OUT} ({len(md):,} chars)")
    if args.html:
        page = render_html(D, g, "Words, Not Numbers")
        args.html.write_text(page, encoding="utf-8")
        print(f"wrote {args.html} ({len(page):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
