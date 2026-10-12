"""Measures for the checkpoint comment and the morning briefing: what each design shows the model (no model involved)
and what came back (re-scored with the current evals/message_rubric.py on every build).

Representation measures, per design, over the 30 cases of each message:
  tokens               prompt tokens as sent, identity header included (cl100k_base, a proxy for Qwen's tokenizer)
  figures              numeric tokens in the prompt
  verdict coverage     decisions the message needs (listed in NEEDED) that the prompt states in words, case-specific,
                       over the decisions needed; a figure the model must interpret ("live 850 vs expected 550") counts
                       as not stated, a coded verdict ("status 3, over target") counts as stated
  misleading           statements in the prompt that are false as worded (checkpoint numbers design: "target next
                       block -300 kcal", a negative target, and "user is 300 kcal over daily budget" while the day's
                       total is still under the daily target)
Output measures per setup: share passing every item (Wilson 95%), per kind and per item, seconds, model calls,
redraft rate, words, digits, and the share of the answer's word 4-grams copied verbatim from the prompt.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = Path(__file__).parent / "results"

DESIGNS = {"checkpoint": {"numbers": "checkpoint_numbers", "labels": "checkpoint_labels",
                          "labels2": "checkpoint_labels2"},
           "briefing": {"numbers": "briefing_numbers", "labels": "briefing_labels", "labels2": "briefing_labels2"}}
MEASURE_DOCS = {
    "prompt tokens": "tokens of everything sent for one message (one checkpoint prompt; both briefing prompts), "
                     "identity header included (tiktoken cl100k_base, a proxy for Qwen's tokenizer)",
    "figures shown": "numeric tokens in those prompts (2,200 / 15:00 / 27°C each count once)",
    "verdicts stated in words": "decisions the message needs (listed below) that the prompt states for this case, "
                                "over the decisions needed. A figure the model has to interpret ('live 850 kcal vs "
                                "expected 550 kcal') counts as not stated; a coded verdict ('status 3, over target') "
                                "counts as stated; the fat-loss note counts as stating what calories and protein mean "
                                "in a cut",
    "misleading statements": "statements in the prompt that are false as worded: in the checkpoint's original prompt, "
                             "a negative 'target next block' (the figure is the gap to the plan so far), and 'user is "
                             "N over daily budget' while the day's total is still under that daily target",
}
NEEDED = {
    "checkpoint": "per metric shown: what the block did, where the day stands, what is left for the rest of the day "
                  "(not at the last check), what the gap means for the phase (when 5%+ off); what to act on first "
                  "(when anything is off); the outdoor verdict (not at the last check, when no movement idea is "
                  "wanted); foods that fit (when food advice is needed); the nothing-logged framing (when nothing is "
                  "logged); that the block has ended",
    "briefing": "yesterday: each metric's result, the workout (on a training day), the weigh-in, what each 5%+ miss "
                "meant for the phase, the link to today's changed target (when there is one), what went well, what to "
                "do better first (when anything was missed); today: the workout, the outdoor verdict, calories and "
                "steps against the usual targets, the reason for a change (when there is one), weekday or weekend, "
                "that a pending carry-over may still change the target",
}
_NUM = re.compile(r"\d[\d,.:%]*")


# ── representation ───────────────────────────────────────────────────────────

def _metric_line(prompt: str, title: str) -> str:
    m = re.search(rf"(?m)^{title}: .*$", prompt)
    return m.group(0) if m else ""


def checkpoint_verdicts(case: dict, prompt: str, design: str) -> list[tuple[str, bool]]:
    """[(verdict needed, stated in words?)] for one checkpoint prompt."""
    from evals import message_rubric as mr
    t = mr.checkpoint_truth(case)
    shown = ["steps"] if t["no_food"] else ["calories", "protein", "steps"]
    out = []
    fat_loss_note = "PHASE NOTE (fat loss)" in prompt
    for key in shown:
        line = _metric_line(prompt, key.capitalize())
        if design == "numbers":
            out.append((f"{key}: block", bool(re.search(r"status \d, (over|under) target", line))))
            out.append((f"{key}: day so far", False))
            if not t["is_last"]:
                out.append((f"{key}: rest of day", False))
            if abs(t["dev"][key]) >= 0.05:
                out.append((f"{key}: meaning", fat_loss_note and key in ("calories", "protein")))
        else:
            out.append((f"{key}: block", "in this block" in line))
            out.append((f"{key}: day so far", "the day so far" in line))
            if not t["is_last"]:
                out.append((f"{key}: rest of day", bool(re.search(r"the rest of the day|already used up|already reached",
                                                                  line))))
            if abs(t["dev"][key]) >= 0.05:
                out.append((f"{key}: meaning", "For the goal:" in line))
    if not t["is_last"] and t["problems"]:
        out.append(("priority", design != "numbers" and "most important first" in prompt))
    if not t["is_last"]:
        out.append(("outdoor", "Outdoor walking:" in prompt))
    if not t["is_last"] and (t["no_food"] or t["problems"]):
        out.append(("foods that fit", design != "numbers" and "Foods from the user's list that fit" in prompt))
    if t["no_food"]:
        out.append(("nothing logged", "CRITICAL FRAMING" in prompt or "nothing logged yet" in prompt))
    out.append(("block ended", bool(re.search(r"already ended|has ended|It is over", prompt))))
    return out


def briefing_verdicts(case: dict, prompts: list[str], design: str) -> list[tuple[str, bool]]:
    from evals import message_rubric as mr
    y_prompt, t_prompt = prompts
    tr, t = case["truth"], mr.briefing_truth(case)
    adj = tr["adjustment"]
    out = []
    steps_short = 1 - tr["steps"] / tr["step_goal"]
    prot_short = 1 - tr["protein"] / tr["protein_target"]
    cal_dev = (tr["calories"] - tr["calories_target"]) / tr["calories_target"]
    misses = [k for k, v in (("steps", steps_short >= 0.05), ("protein", prot_short >= 0.05),
                             ("calories", abs(cal_dev) > 0.03)) if v]
    if design == "numbers":
        out += [("y: steps", "- Steps:" in y_prompt), ("y: calories", "- Calories:" in y_prompt),
                ("y: protein", "- Protein:" in y_prompt)]
        if t["workout_day"]:
            out.append(("y: workout", "- Workout (" in y_prompt))
        out.append(("y: weigh-in", "- Weight log:" in y_prompt))
        out += [(f"y: meaning of {k}", False) for k in misses]
        if adj:
            out.append(("y: link to today's change", False))
        out.append(("y: went well", False))
        if misses or (t["workout_day"] and not t["workout_logged"]) or not t["weight_logged"]:
            out.append(("y: fix first", False))
        adjusted_cal = "Calories today: adjusted" in t_prompt
        adjusted_steps = "Steps today: adjusted" in t_prompt
        out += [("t: workout", "Workout:" in t_prompt), ("t: outdoor", "Outdoor walking:" in t_prompt),
                ("t: calories vs usual", not adjusted_cal), ("t: steps vs usual", not adjusted_steps),
                ("t: weekday or weekend", bool(re.search(r"\((weekday|weekend)\)", t_prompt)))]
        if adj:
            out.append(("t: reason for the change", bool(re.search(r"Carry Over[^\n]*\b(over|under) yesterday", t_prompt))))
        if adj and adj[0] == "pending":
            out.append(("t: target may still change", False))
    else:
        for k in ("Steps", "Calories", "Protein"):
            out.append((f"y: {k.lower()}", bool(_metric_line(y_prompt, k))))
        if t["workout_day"]:
            out.append(("y: workout", bool(_metric_line(y_prompt, "Workout"))))
        out.append(("y: weigh-in", bool(_metric_line(y_prompt, "Weigh-in"))))
        out += [(f"y: meaning of {k}", "For the goal:" in _metric_line(y_prompt, k.capitalize())) for k in misses]
        if adj:
            out.append(("y: link to today's change", bool(re.search(r"that is why today's|will be asked how to spread",
                                                                    y_prompt))))
        out.append(("y: went well", "Went well:" in y_prompt))
        if misses or (t["workout_day"] and not t["workout_logged"]) or not t["weight_logged"]:
            out.append(("y: fix first", "To do better today" in y_prompt))
        cal_line, steps_line = _metric_line(t_prompt, "Calories today"), _metric_line(t_prompt, "Steps today")
        out += [("t: workout", bool(_metric_line(t_prompt, "Workout"))), ("t: outdoor", "Outdoor walking:" in t_prompt),
                ("t: calories vs usual", bool(re.search(r"usual", cal_line))),
                ("t: steps vs usual", bool(re.search(r"usual", steps_line))),
                ("t: weekday or weekend", bool(re.search(r"a (weekday|weekend day)", t_prompt)))]
        if adj:
            out.append(("t: reason for the change", bool(re.search(r"to make up for|because you ate|to offset|"
                                                                   r"will be asked", t_prompt))))
        if adj and adj[0] == "pending":
            out.append(("t: target may still change", "may still change" in t_prompt))
    return out


def misleading_statements(case: dict, prompt: str) -> int:
    """Checkpoint numbers prompt: negative 'target next block' lines, and 'over daily budget' notes while the day's
    total is still under that metric's daily target."""
    n = 0
    daily = {"Calories": case["goals"]["calories_target"], "Protein": case["goals"]["protein_target"],
             "Steps": case["goals"]["step_goal"]}
    cum = {"Calories": case["cum"]["calories"], "Protein": case["cum"]["protein"], "Steps": case["cum"]["steps"]}
    for title in daily:
        line = _metric_line(prompt, title)
        if re.search(r"target next block -\d", line):
            n += 1
        if "over daily budget" in line and cum[title] < daily[title]:
            n += 1
    return n


def _mean_sd(xs):
    return (statistics.mean(xs), statistics.pstdev(xs)) if xs else (0.0, 0.0)


def representation_measures() -> dict:
    """{message: {design: {tokens, figures, coverage (stated, needed), misleading (statements, prompts with any)}}}"""
    from evals import label_metrics, run_message_eval as rme
    out = {}
    for message, designs in DESIGNS.items():
        cases = rme._cases(message, "all")
        out[message] = {}
        for design, setup in designs.items():
            toks, figs, stated, needed, mis, mis_prompts, missing = [], [], 0, 0, 0, 0, defaultdict(int)
            for case in cases.values():
                ps = rme.prompts(setup, case)
                toks.append(sum(label_metrics.TOKENS(p) for p in ps))
                figs.append(sum(len(_NUM.findall(p)) for p in ps))
                vs = (checkpoint_verdicts(case, ps[0], design) if message == "checkpoint"
                      else briefing_verdicts(case, ps, design))
                needed += len(vs)
                stated += sum(ok for _v, ok in vs)
                for v, ok in vs:
                    if not ok:
                        missing[v.split(": ")[-1] if message == "checkpoint" else v] += 1
                if message == "checkpoint" and design == "numbers":
                    k = misleading_statements(case, ps[0])
                    mis += k
                    mis_prompts += k > 0
            out[message][design] = {"tokens": _mean_sd(toks), "figures": _mean_sd(figs), "coverage": (stated, needed),
                                    "misleading": (mis, mis_prompts, len(cases)), "missing": dict(missing),
                                    "cases": len(cases)}
    out["tokenizer"] = label_metrics.TOKENIZER
    return out


# ── answers ──────────────────────────────────────────────────────────────────

def load_rows(paths=None) -> list[dict]:
    rows = []
    for p in sorted(paths or RESULTS.glob("messages_*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in doc.get("results", []):
            rows.append({**r, "file": p.name})
    return rows


def current_rows(rows: list[dict], fingerprints: dict) -> dict:
    out = defaultdict(list)
    for r in rows:
        if r["setup"] in fingerprints and r.get("fingerprint") == fingerprints[r["setup"]]:
            out[r["setup"]].append(r)
    return out


def rescore(rows: list[dict]) -> list[dict]:
    """Attach the CURRENT checklist's items to each row (the run-time items are kept as `items_at_run`)."""
    from evals import briefing_cases, checkpoint_cases, message_rubric as mr
    cases = {"checkpoint": {**checkpoint_cases.cases("all"), **checkpoint_cases.cases("incident")},
             "briefing": {**briefing_cases.cases("all"), **briefing_cases.cases("incident")}}
    out = []
    for r in rows:
        case = cases[r["message"]][r["case"]]
        items = (mr.score_checkpoint(r["answer"], case) if r["message"] == "checkpoint"
                 else mr.score_briefing(r["answer"], case))
        out.append({**r, "items_at_run": r.get("items"), "now": items})
    return out


def _text(answer) -> str:
    return answer if isinstance(answer, str) else "\n\n".join(answer.get(k, "") for k in ("yesterday", "today"))


def summarize(rows: list[dict], setup: str) -> dict:
    """All output measures for one setup's re-scored rows (one case set)."""
    from evals import briefing_cases, checkpoint_cases, message_rubric as mr, output_metrics as om, run_message_eval as rme
    if not rows:
        return {}
    message = rows[0]["message"]
    base_calls = 1 if message == "checkpoint" else 2
    kinds = {k: [0, 0] for k in mr.KINDS}
    items = defaultdict(lambda: [0, 0])
    clean = 0
    module = checkpoint_cases if message == "checkpoint" else briefing_cases
    pool = {**module.cases("all"), **module.cases("incident")}
    copy = []
    for r in rows:
        ok_all = True
        for i in r["now"]:
            kinds[i.kind][1] += 1
            kinds[i.kind][0] += i.ok
            items[i.id][1] += 1
            items[i.id][0] += i.ok
            ok_all &= i.ok
        clean += ok_all
        ps = rme.prompts(setup, pool[r["case"]])
        if message == "checkpoint":
            copy.append(om.copy_rate(r["answer"], ps[0]))
        else:
            copy.append(statistics.mean([om.copy_rate(r["answer"].get("yesterday", ""), ps[0]),
                                         om.copy_rate(r["answer"].get("today", ""), ps[1])]))
    texts = [_text(r["answer"]) for r in rows]
    by_rep = defaultdict(list)
    for r in rows:
        by_rep[r.get("repeat", 0)].append(all(i.ok for i in r["now"]))
    return {
        "n": len(rows), "cases": len({r["case"] for r in rows}), "repeats": len(by_rep),
        "all_items": om.wilson(clean, len(rows)),
        "all_by_repeat": _mean_sd([sum(v) / len(v) for v in by_rep.values()]),
        "kinds": {k: tuple(v) for k, v in kinds.items()}, "items": {k: tuple(v) for k, v in items.items()},
        "latency": _mean_sd([r["seconds"] for r in rows]), "calls": statistics.mean(r["calls"] for r in rows),
        "redraft_rate": sum(r["calls"] > base_calls for r in rows) / len(rows),
        "words": statistics.mean(len(re.findall(r"[A-Za-z']+", t)) for t in texts),
        "digits": statistics.mean(len(_NUM.findall(t)) for t in texts),
        "copy_rate": statistics.mean(copy), "errors": sum(bool(r.get("error")) for r in rows),
    }


def per_case(rows: list[dict], measure: str = "items") -> dict:
    """{case: value averaged over repeats}: the share of applicable items passed ('items'), or whether the answer
    passed every item ('all')."""
    acc = defaultdict(list)
    for r in rows:
        acc[r["case"]].append(sum(i.ok for i in r["now"]) / max(len(r["now"]), 1) if measure == "items"
                              else float(all(i.ok for i in r["now"])))
    return {c: statistics.mean(v) for c, v in acc.items()}


def paired(rows_a: list[dict], rows_b: list[dict], measure: str = "items") -> dict:
    from evals import output_metrics as om
    a, b = per_case(rows_a, measure), per_case(rows_b, measure)
    common = sorted(set(a) & set(b))
    xs, ys = [a[c] for c in common], [b[c] for c in common]
    diffs = [x - y for x, y in zip(xs, ys)]
    return {"cases": len(common), "diff": statistics.mean(diffs) if diffs else 0.0,
            "wins": sum(d > 1e-9 for d in diffs), "ties": sum(abs(d) <= 1e-9 for d in diffs),
            "losses": sum(d < -1e-9 for d in diffs), "p": om.wilcoxon_p(xs, ys) if common else None}


def gather() -> dict:
    """Everything the report's message section needs."""
    from evals import output_metrics as om, run_message_eval as rme
    fps = {s: rme.fingerprint(s) for s in rme.SETUPS}
    rows = rescore([r for rs in current_rows(load_rows(), fps).values() for r in rs])
    by = defaultdict(list)
    for r in rows:
        by[(r["setup"], om.case_set_of(r["case"]))].append(r)
    summaries = {cs: {s: summarize(by[(s, cs)], s) for s in rme.SETUPS if by[(s, cs)]} for cs in ("tuning", "holdout", "incident")}
    return {"fps": fps, "rows": rows, "by": by, "summaries": summaries, "rep": representation_measures(),
            "files": sorted({r["file"] for r in rows})}


if __name__ == "__main__":
    rep = representation_measures()
    for message in DESIGNS:
        for design, m in rep[message].items():
            s, n = m["coverage"]
            print(f"{message:<10} {design:<8} tokens {m['tokens'][0]:6.0f} ± {m['tokens'][1]:4.0f}  figures "
                  f"{m['figures'][0]:5.1f}  verdicts stated {s}/{n} ({s / n:.0%})  misleading {m['misleading'][0]} "
                  f"in {m['misleading'][1]}/{m['misleading'][2]} prompts  missing {m['missing']}")
