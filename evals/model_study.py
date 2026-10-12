"""The same comparison on other models (scripts/run_model_study.ps1 writes one folder per model).

    evals/results/models/<model-slug>/  manifest.json, weekly_*.json, messages_*.json

The main report pools only qwen3:8b runs at their current fingerprint; these folders are kept apart so a run on another
model can never leak into it. Here each folder is summarized with the same functions the main report uses (the weekly
core rubric v2, the message checklist) and shown next to qwen3:8b: for each model the share of answers passing every
item on the held-out cases, per design, and the paired difference "labels minus numbers" with a Wilcoxon p.

The designs compared (the ones that carry the paper's claim): weekly review A1 numbers / A2 labels v1 / A7 labels v6 /
A8 numbers + verdicts; checkpoint C0 numbers / C6 labels v3 with checks; briefing B0 numbers / B1 labels v1 / B8 labels v4
with checks.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS_DIR = Path(__file__).parent / "results" / "models"
DEFAULT_MODEL = "qwen3:8b"
# Models that were started and had to be dropped, with the reason. Their folders stay on disk but never reach a table.
EXCLUDED = {"olmo-3:7b": "a thinking model: one weekly case took 22 minutes (1339 s), so the 30 cases x 3 runs of one design "
                         "would take about 33 hours; only that one case was run"}

WEEKLY = {"split_fast": "A1 numbers", "labels_fast": "A2 labels v1", "labels6_fast": "A7 labels v6",
          "hybrid_fast": "A8 numbers + verdicts"}
MESSAGES = {"checkpoint_numbers": "C0 numbers", "checkpoint_labels3": "C6 labels v3 + checks",
            "briefing_numbers": "B0 numbers", "briefing_labels_noguard": "B1 labels v1",
            "briefing_labels4": "B8 labels v4 + checks"}
# (name, labels design, numbers design, kind)
CONTRASTS = [("weekly review: A7 labels v6 minus A1 numbers", "labels6_fast", "split_fast", "weekly"),
             ("checkpoint: C6 labels v3 minus C0 numbers", "checkpoint_labels3", "checkpoint_numbers", "messages"),
             ("briefing: B8 labels v4 minus B0 numbers", "briefing_labels4", "briefing_numbers", "messages")]


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def model_folders(root: Path = MODELS_DIR) -> dict[str, Path]:
    """{model name: folder}, the name taken from the folder's manifest.json when it has one."""
    out = {}
    for d in sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []:
        man = d / "manifest.json"
        name = d.name
        if man.exists():
            try:
                name = json.loads(man.read_text(encoding="utf-8-sig")).get("model") or name
            except Exception:
                pass
        out[name] = d
    return out


def _case_set(case: str) -> str:
    from evals import output_metrics as om
    return om.case_set_of(case)


def _cell(summary: dict) -> dict:
    p, lo, hi = summary["all_items"]
    return {"n": summary["n"], "p": p, "lo": lo, "hi": hi, "seconds": summary["latency"][0]}


def summarize_rows(weekly_rows: list[dict], message_rows: list[dict]) -> dict:
    """{'cells': {setup: {case set: cell}}, 'pairs': {contrast: {'pooled': {...}, 'holdout': {...}}}} for one model."""
    from collections import defaultdict
    from evals import message_metrics as mm, output_metrics as om
    by_w, by_m = defaultdict(list), defaultdict(list)
    for r in weekly_rows:
        by_w[(r["setup"], _case_set(r["case"]))].append(r)
    for r in message_rows:
        by_m[(r["setup"], _case_set(r["case"]))].append(r)
    cells = defaultdict(dict)
    for (setup, cs), rs in by_w.items():
        if setup in WEEKLY and cs in ("tuning", "holdout"):
            cells[setup][cs] = _cell(om.summarize(rs, setup))
    for (setup, cs), rs in by_m.items():
        if setup in MESSAGES and cs in ("tuning", "holdout", "incident"):
            cells[setup][cs] = _cell(mm.summarize(rs, setup))
    pairs = {}
    for name, lab, num, kind in CONTRASTS:
        src = by_w if kind == "weekly" else by_m
        out = {}
        for label, sets in (("pooled", ("tuning", "holdout")), ("holdout", ("holdout",))):
            ra = [r for cs in sets for r in src.get((lab, cs), [])]
            rb = [r for cs in sets for r in src.get((num, cs), [])]
            if not ra or not rb:
                continue
            if kind == "weekly":
                q = om.paired(ra, rb)["core"]
            else:
                q = mm.paired(ra, rb, "all")
            out[label] = {"diff": q["diff"], "p": q.get("p"), "wins": q["wins"], "ties": q["ties"], "losses": q["losses"]}
        if out:
            pairs[name] = out
    return {"cells": dict(cells), "pairs": pairs}


def collect(folder: Path) -> dict:
    """Summaries of one model's folder (every row in it, re-scored with the current checklists)."""
    from evals import message_metrics as mm, output_metrics as om
    # load_rows(paths) treats an EMPTY list as "use the main results" (qwen3:8b), so a model folder with no files of one kind
    # must not be passed through it: its row would show qwen3's numbers under the other model's name (found 2026-10-05).
    weekly_files = sorted(folder.glob("weekly_*.json"))
    message_files = sorted(folder.glob("messages_*.json"))
    weekly = om.load_rows(weekly_files) if weekly_files else []
    messages = mm.rescore(mm.load_rows(message_files)) if message_files else []
    manifest = {}
    if (folder / "manifest.json").exists():
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8-sig"))
    return {**summarize_rows(weekly, messages), "manifest": manifest}


def from_main(g: dict) -> dict:
    """qwen3:8b from the main report's data (current fingerprints only), in the same shape."""
    weekly = [r for s in WEEKLY for r in g.get("used", {}).get(s, [])]
    messages = [r for r in (g.get("msg") or {}).get("rows", []) if r["setup"] in MESSAGES]
    return {**summarize_rows(weekly, messages), "manifest": {"model": DEFAULT_MODEL}}


def gather(g: dict | None = None, root: Path = MODELS_DIR) -> dict:
    """{model: summary}, qwen3:8b first (when the main report's data is given), then every folder found."""
    out = {}
    if g is not None:
        out[DEFAULT_MODEL] = from_main(g)
    for name, folder in model_folders(root).items():
        if name != DEFAULT_MODEL and name not in EXCLUDED:
            out[name] = collect(folder)
    return out


# ── tables ───────────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def cells_table(study: dict, case_set: str = "holdout") -> tuple[list[str], list[list]]:
    setups = {**WEEKLY, **MESSAGES}
    header = ["model"] + list(setups.values())
    rows = []
    for model, data in study.items():
        row = [model]
        for s in setups:
            c = data["cells"].get(s, {}).get(case_set)
            row.append(f"{_pct(c['p'])} [{_pct(c['lo'])}-{_pct(c['hi'])}] n={c['n']}" if c else "-")
        rows.append(row)
    return header, rows


def pairs_table(study: dict, scope: str = "pooled") -> tuple[list[str], list[list]]:
    header = ["model"] + [name for name, *_ in CONTRASTS]
    rows = []
    for model, data in study.items():
        row = [model]
        for name, *_ in CONTRASTS:
            q = data["pairs"].get(name, {}).get(scope)
            row.append("-" if not q else f"{q['diff']:+.2f} ({q['wins']}/{q['ties']}/{q['losses']}"
                       + (f", p = {q['p']:.3f})" if q.get("p") is not None else ")"))
        rows.append(row)
    return header, rows


# ── the chat agent (evals/run_chat_eval.py; one file per run in each model's folder) ─────────────────────────────

def chat_summary(root: Path = MODELS_DIR) -> dict:
    """{model: {'numbers': cell, 'words': cell, 'pair': {...}}} from the chat_*.json files of every model folder
    (qwen3:8b's folder included). Rows are taken as scored when they were run (the checklist version is in each file's meta);
    only (case, repeat) pairs that both designs answered are compared."""
    from math import comb
    from evals import output_metrics as om
    out = {}
    for name, folder in model_folders(root).items():
        if name in EXCLUDED:
            continue
        rows = []
        for p in sorted(folder.glob("chat_*.json")):
            try:
                doc = json.loads(p.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            rows += [{**r, "version": (doc.get("meta") or {}).get("checklist", "?")} for r in doc.get("results", [])]
        if not rows:
            continue
        data = {}
        for setup in ("numbers", "words"):
            rs = [r for r in rows if r["setup"] == setup]
            if rs:
                k = sum(1 for r in rs if r["passed"])
                p, lo, hi = om.wilson(k, len(rs))
                # the original design is a tool-using agent: a model without tool support (or without thinking) answers every
                # request with an API error, which run_agent turns into this text. Counted apart so it is not read as a verdict.
                errors = sum(1 for r in rs if str(r.get("answer", "")).startswith("Something went wrong"))
                data[setup] = {"n": len(rs), "k": k, "p": p, "lo": lo, "hi": hi, "errors": errors,
                               "seconds": sum(r["seconds"] for r in rs) / len(rs)}
        by = {(r["case"], r["repeat"], r["setup"]): r["passed"] for r in rows}
        keys = sorted({(c, rp) for c, rp, s in by if (c, rp, "numbers") in by and (c, rp, "words") in by})
        w = sum(1 for k in keys if by[k + ("words",)] and not by[k + ("numbers",)])
        n = sum(1 for k in keys if by[k + ("numbers",)] and not by[k + ("words",)])
        m = w + n
        sign_p = min(1.0, 2 * sum(comb(m, i) for i in range(0, min(w, n) + 1)) / 2 ** m) if m else None
        data["pair"] = {"pairs": len(keys), "words_only": w, "numbers_only": n, "p": sign_p}
        out[name] = data
    return out


def chat_table(summary: dict) -> tuple[list[str], list[list]]:
    header = ["model", "chat original (numbers + tools)", "chat words (chat_facts)", "pairs: only words / only original (sign test)",
              "seconds per answer: original / words"]
    rows = []
    for model, d in summary.items():
        def cell(s):
            c = d.get(s)
            if not c:
                return "-"
            note = f", {c['errors']} API errors" if c.get("errors") else ""
            return f"{_pct(c['p'])} [{_pct(c['lo'])}-{_pct(c['hi'])}] n={c['n']}{note}"
        pr = d["pair"]
        pv = "-" if pr["p"] is None else f"p = {pr['p']:.2g}"
        sec = " / ".join(f"{d[s]['seconds']:.1f}" if s in d else "-" for s in ("numbers", "words"))
        rows.append([model, cell("numbers"), cell("words"), f"{pr['words_only']} / {pr['numbers_only']} of {pr['pairs']} ({pv})", sec])
    return header, rows


if __name__ == "__main__":
    study = gather()
    if not study:
        print(f"no model folders under {MODELS_DIR} yet; run scripts\\run_model_study.ps1")
    for cs in ("holdout", "tuning"):
        h, r = cells_table(study, cs)
        print(f"\n{cs}: share of answers passing every item\n" + " | ".join(h))
        for row in r:
            print(" | ".join(row))
    h, r = pairs_table(study)
    print("\nlabels minus numbers, all 30 cases (wins/ties/losses)\n" + " | ".join(h))
    for row in r:
        print(" | ".join(row))
    chat = chat_summary()
    if chat:
        h, r = chat_table(chat)
        print("\nchat agent, fake days (share of answers passing every checklist item)\n" + " | ".join(h))
        for row in r:
            print(" | ".join(row))
