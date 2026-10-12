"""Human ratings for the paper: blind pairwise judgments of the original prompts vs. the labels designs.

    python -m evals.human_rating build      # -> evals/human_rating_kit/rate.html (send to raters) + key.json (keep)
    python -m evals.human_rating tally evals/human_rating_kit/key.json ratings_*.json   # -> evals/human_ratings.json

What raters see: 36 items (rebuilt 2026-10-05), one case each. Every item shows the facts of the case (the figures, so
a rater can check accuracy) and two messages for it, "Message A" and "Message B": one from the original prompt (A1 weekly
review, C0 checkpoint, B0 briefing, the original chat agent) and one from the design now in production (A7, C6 =
checkpoint labels v3, B8 = briefing labels v4, the words chat agent). Eight cases per message type, drawn at random
(seeded) from the held-out cases; the chat items come from the last fresh set, G01-G20, whose answers are from the
design now deployed. Raters answer three questions per item (accuracy, advice, overall preference: A, B, about the same,
both poor) and may comment. One item per message type is shown twice with the sides swapped, to measure consistency.

Blinding: in the page each answer is only "p" or "q"; which design wrote it is in key.json, which raters never get.
Each rater also gets a different item order and side assignment (seeded by their alias). Answers are the first
run (run 1) of each held-out case at the configuration's current fingerprint, so nothing is cherry-picked.
The page is one self-contained file: no network, no libraries; ratings are saved in the browser as the rater goes
and leave it in one of two ways: "Copy my ratings" (a block of text between BEGIN CALVIS RATINGS and END CALVIS RATINGS
that the rater pastes into any chat or email: the practical way on a phone) or "Download my ratings" (a small JSON file).
`tally` reads both, in any mix, and ignores whatever text surrounds a pasted block. Served by evals/rating_server.py the page also
has "Send my ratings", which saves the file on the server directly (the best way on a phone).
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KIT = Path(__file__).parent / "human_rating_kit"
SUMMARY = Path(__file__).parent / "human_ratings.json"

# message -> (baseline setup, labels setup)
PAIRS = {"weekly review": ("split_fast", "labels6_fast"),
         "checkpoint comment": ("checkpoint_numbers", "checkpoint_labels3"),
         "morning briefing": ("briefing_numbers", "briefing_labels4"),
         "chat question": ("chat_numbers", "chat_words")}
PER_MESSAGE = 8                    # cases drawn per message type (keeps the page to about 35 minutes)
CHAT_SET = "fresh2"                # the chat cases whose answers come from the design now deployed (G01-G20)
QUESTIONS = {
    "accuracy": "Which message is more accurate about the user's data shown above?",
    "advice": "Which message gives better advice for this user's goal?",
    "overall": "Which message would you rather receive?",
}
CHOICES = ("p", "q", "equal", "both_poor")
REPEATS = 3


# ── facts shown to the rater (figures, so accuracy can be checked) ──────────

def _n(x, unit=""):
    return "no data" if x is None else f"{x:,.0f}{unit}"


def weekly_facts(d: dict) -> str:
    g = d["goals"]
    avg = lambda k: (d.get(k) or {}).get("avg")
    return (f"Goal phase: {g.get('phase', '?')}. Weekly averages against targets: steps {_n(avg('steps'))} "
            f"(target {_n(g.get('step_goal'))}); calories {_n(avg('calories'), ' kcal')} (target "
            f"{_n(g.get('calories_target'), ' kcal')}); protein {_n(avg('protein'), ' g')} (target "
            f"{_n(g.get('protein_target'), ' g')}); workouts {len(d.get('workouts') or [])} this week.")


def checkpoint_facts(c: dict) -> str:
    cp, cum, g, ctx = c["cp"], c["cum"], c["goals"], c["context_data"]
    plan = lambda k, t: t * cp["frac"]
    w = ctx.get("weather")
    work = ctx.get("workout") or {}
    work_s = {"rest": "rest day", "cardio": "cardio day"}.get(work.get("kind"),
                                                             f"{work.get('type', '?')}, {'done' if work.get('logged') else 'not done yet'}")
    end = "23:59" if cp["is_last"] else f"{cp['hour']:02d}:00"
    return (f"Phase: {c['phase']}. Check at {end} ({'the last check of the day' if cp['is_last'] else 'a daytime check'}), "
            f"closing the block {cp['block_start_hour']:02d}:00-{end}. The day so far against the plan for this time: "
            f"calories {_n(cum['calories'])} of {_n(plan('c', g['calories_target']))} kcal planned (daily target "
            f"{_n(g['calories_target'])}); protein {_n(cum['protein'])} of {_n(plan('p', g['protein_target']))} g "
            f"(daily {_n(g['protein_target'])}); steps {_n(cum['steps'])} of {_n(plan('s', g['step_goal']))} "
            f"(daily {_n(g['step_goal'])}). Weather: {w['description'] + ', ' + str(round(w['temp_c'])) + ' C' if w else 'unknown'}. "
            f"Workout today: {work_s}."
            + (f" Dislikes: {ctx['food_dislikes']}." if ctx.get("food_dislikes") else "")
            + (f" Said this morning: \"{ctx['morning_feedback']}\"." if ctx.get("morning_feedback") else ""))


def briefing_facts(c: dict) -> str:
    t, y = c["truth"], c["original"]["yesterday"]
    adj = t["adjustment"]
    if not adj:
        change = "today's targets are the usual ones"
    elif adj[0] == "pending":
        change = f"yesterday ended {adj[1]} kcal {adj[2]} target; the user will be asked how to make up for it"
    else:
        strategy, amount, value = adj
        change = {"calories": f"today's calorie target is lowered to {value:,} (usual {t['base_cal']:,}) to make up "
                              f"for {amount} kcal over yesterday",
                  "calories_boost": f"today's calorie target is raised to {value:,} (usual {t['base_cal']:,}) after "
                                    f"{amount} kcal under yesterday",
                  "steps": f"today's step goal is raised to {t['base_steps'] + value:,} (usual {t['base_steps']:,}) "
                           f"to offset {amount} kcal over yesterday",
                  "steps_reduce": f"today's step goal is lowered to {value:,} (usual {t['base_steps']:,}) after "
                                  f"{amount} kcal under yesterday"}[strategy]
    w = t["weather"]
    return (f"Phase: {c['phase']}. Yesterday: steps {_n(t['steps'])} of {_n(t['step_goal'])}; calories "
            f"{_n(t['calories'])} of {_n(t['calories_target'])} kcal; protein {_n(t['protein'])} of "
            f"{_n(t['protein_target'])} g; workout "
            + (f"{y['yesterday_workout_type']} {'logged' if y['yesterday_workout_logged'] else 'NOT logged'}"
               if y["yesterday_was_workout_day"] else "rest day")
            + f"; weigh-in {'recorded' if y['yesterday_weight_logged'] else 'missed'}. Today: {change}; workout "
            f"{t['workout_today']}; weather {w['description'] + ', ' + str(round(w['temp_c'])) + ' C' if w else 'unknown'}.")


def chat_step_bonus_target(day: dict) -> int:
    """The calorie target the chat shows for a fake day: the base plus a bonus for steps over the goal (nothing is taken
    away for steps under it). Same arithmetic as agent.live_display_target with the evaluation's 80 kg walker (pinned by a test)."""
    over = day["steps"] - day["step_goal"]
    bonus = round(over * 0.0005 * (3.0 - 1) / 3.0 * 80.0) if over > 0 else 0
    return day["base_kcal"] + bonus


def chat_facts(day: dict, question: str) -> str:
    """What a rater sees about one fake day of the chat evaluation (evals/chat_cases.py), with the question asked."""
    eaten = sum(e[1] for e in day["entries"])
    protein = sum(e[2] for e in day["entries"])
    target = chat_step_bonus_target(day)
    left = target - eaten
    stale = f" The step count is {day['steps_age'] // 60} hours old." if day["steps_age"] >= 180 else ""
    food = "; ".join(f"{n} {k} kcal at {t}" for n, k, _p, t in day["entries"]) or "nothing yet"
    return (f"Phase: {day['phase']}. Time: {day['hour']:02d}:{day['minute']:02d}. Steps {day['steps']:,} of a "
            f"{day['step_goal']:,} goal.{stale} Calories eaten {eaten:,} of a {target:,} kcal target "
            f"({f'{left:,} left' if left > 0 else f'{-left:,} over' if left < 0 else 'none left'}). Protein {protein:,.0f} of a "
            f"{day['protein_target']:,} g goal. Food logged today: {food}. The chat assistant cannot log or change anything. "
            f"The user asked it: \"{question}\"")


# ── building the kit ─────────────────────────────────────────────────────────

def _answers() -> dict:
    """{(setup, case): first-run answer text} at each setup's current fingerprint, held-out cases only."""
    from evals import message_metrics as mm, output_metrics as om, run_message_eval as rme, run_weekly_eval as rw
    out = {}
    weekly = [s for pair in (PAIRS["weekly review"],) for s in pair]
    cur = om.current_rows(om.load_rows(), {s: rw.fingerprint(s) for s in weekly})
    for s in weekly:
        for r in cur.get(s, []):
            if r["case"].startswith("H") and r.get("repeat", 0) == 0:
                out[(s, r["case"])] = r["text"]
    msg = [s for m in ("checkpoint comment", "morning briefing") for s in PAIRS[m]]
    cur = mm.current_rows(mm.load_rows(), {s: rme.fingerprint(s) for s in msg})
    for s in msg:
        for r in cur.get(s, []):
            if r["case"].startswith("H") and r.get("repeat", 0) == 0:
                a = r["answer"]
                out[(s, r["case"])] = a if isinstance(a, str) else f"{a.get('yesterday', '')}\n\n{a.get('today', '')}"
    out.update(_chat_answers())
    return out


def _chat_answers() -> dict:
    """{('chat_numbers' | 'chat_words', case): first-run answer} for the CHAT_SET cases, only from runs of the design as it is
    now (the fingerprint of the code that decides the answers). A case with a stale or missing answer on either side is absent."""
    from evals import chat_cases as cc, run_chat_eval as rce
    ids = {c[0] for c in cc.CASES if c[4] == CHAT_SET}
    current = {s: rce.fingerprint(s) for s in ("numbers", "words")}
    out = {}
    for path in sorted((Path(__file__).parent / "results").glob("chat_*.json")):
        if path.name.endswith(".rescored.json"):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8")).get("results", [])
        except Exception:
            continue
        for r in rows:
            if r["case"] in ids and r["setup"] in current and r.get("repeat") == 1 and r.get("fingerprint") == current[r["setup"]]:
                out[(f"chat_{r['setup']}", r["case"])] = r["answer"]
    return out


def build(seed: int = 2026, out_dir: Path = KIT, per_message: int | None = None, questions=None,
          repeats: int | None = None) -> tuple[Path, Path]:
    """per_message: cases per message type (default PER_MESSAGE); questions: which of QUESTIONS to ask (default all three);
    repeats: how many items are shown a second time with the sides swapped (default one per message type). A shorter kit
    (for example 6 cases, the questions accuracy and overall, 2 repeats) takes about half the time of the first one."""
    from evals import briefing_cases, checkpoint_cases, weekly_holdout_cases
    n_per = per_message or PER_MESSAGE
    if any(k not in QUESTIONS for k in (questions or [])):
        raise SystemExit(f"questions must be some of {list(QUESTIONS)}")
    asked = {k: QUESTIONS[k] for k in (questions or QUESTIONS)}
    answers = _answers()
    from evals import chat_cases as cc
    facts = {"weekly review": {n: weekly_facts(d) for n, d in weekly_holdout_cases.CASES.items()},
             "checkpoint comment": {n: checkpoint_facts(c) for n, c in checkpoint_cases.cases("holdout").items()},
             "morning briefing": {n: briefing_facts(c) for n, c in briefing_cases.cases("holdout").items()},
             "chat question": {c[0]: chat_facts(cc.DAYS[c[1]], c[2]) for c in cc.CASES if c[4] == CHAT_SET}}
    rng = random.Random(seed)
    items, key = [], {}
    for message, (base, lab) in PAIRS.items():
        # only cases with a current answer on BOTH sides can be shown; PER_MESSAGE of them are drawn at random (seeded)
        usable = [c for c in sorted(facts[message]) if answers.get((base, c)) and answers.get((lab, c))]
        if len(usable) < n_per:
            raise SystemExit(f"only {len(usable)} cases of '{message}' have a current answer on both sides "
                             f"(need {n_per}): run the evaluation first")
        for case in sorted(rng.sample(usable, n_per)):
            a, b = answers[(base, case)], answers[(lab, case)]
            p_is_base = rng.random() < 0.5
            iid = f"i{len(items) + 1:02d}"
            items.append({"id": iid, "message": message, "facts": facts[message][case],
                          "p": a if p_is_base else b, "q": b if p_is_base else a})
            key[iid] = {"message": message, "case": case, "p": base if p_is_base else lab,
                        "q": lab if p_is_base else base, "repeat_of": None}
    repeated = list(PAIRS) if repeats is None else rng.sample(list(PAIRS), min(max(repeats, 0), len(PAIRS)))
    for message in repeated:                               # repeated items (default: one per message), sides swapped
        src = rng.choice([it for it in items if it["message"] == message and key[it["id"]]["repeat_of"] is None])
        iid = f"i{len(items) + 1:02d}"
        items.append({**src, "id": iid, "p": src["q"], "q": src["p"]})
        key[iid] = {**key[src["id"]], "p": key[src["id"]]["q"], "q": key[src["id"]]["p"], "repeat_of": src["id"]}
    import secrets
    # a random suffix: two kits built in the same minute (a rebuilt one beside one in use) must never share an id, or each
    # would accept the other's ratings
    kit_id = f"calvis-ratings-{datetime.now():%Y%m%d%H%M}-{secrets.token_hex(2)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / "rate.html"
    minutes = -(-int((0.55 + 0.15 * len(asked)) * len(items)) // 5) * 5          # about this long, to the next five minutes
    page.write_text(_page(items, kit_id, asked, minutes), encoding="utf-8")
    keyp = out_dir / "key.json"
    keyp.write_text(json.dumps({"kit_id": kit_id, "seed": seed, "pairs": PAIRS, "questions": list(asked), "per_message": n_per,
                                "items": key}, indent=1), encoding="utf-8")
    return page, keyp


def _page(items: list[dict], kit_id: str, questions=None, minutes: int = 35) -> str:
    data = json.dumps({"kit": kit_id, "items": items, "questions": questions or QUESTIONS, "minutes": minutes}, ensure_ascii=False)
    data = data.replace("</", "<\\/")                      # keep the JSON inside its script tag
    return _HTML.replace("__DATA__", data).replace("__TITLE__", html.escape("Coaching message ratings"))


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f6f7f5; --ink:#1c2127; --muted:#5d6770; --card:#fff; --rule:#d9ddd9; --accent:#2f6a86; --pick:#e3eef4; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#15191c; --ink:#e4e7ea; --muted:#9aa4ac;
  --card:#1c2125; --rule:#2d3439; --accent:#7fb3cf; --pick:#23343d; } }
:root[data-theme="dark"] { --bg:#15191c; --ink:#e4e7ea; --muted:#9aa4ac; --card:#1c2125; --rule:#2d3439; --accent:#7fb3cf; --pick:#23343d; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 20px 16px 80px; }
h1 { font-size: 1.4rem; margin: 0 0 8px; } h2 { font-size: 1.05rem; margin: 0 0 6px; }
p { margin: 0 0 10px; } .muted { color: var(--muted); font-size: .92rem; }
.card { background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 14px; margin: 12px 0; }
.facts { border-left: 4px solid var(--accent); }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 720px) { .pair { grid-template-columns: 1fr; } }
.msg { white-space: pre-wrap; }
fieldset { border: 0; padding: 0; margin: 12px 0 0; }
legend { font-weight: 600; margin-bottom: 6px; }
.opts { display: flex; flex-wrap: wrap; gap: 8px; }
.opts label { border: 1px solid var(--rule); border-radius: 6px; padding: 6px 10px; cursor: pointer; background: var(--card); }
.opts input { margin-right: 6px; }
.opts label:has(input:checked) { background: var(--pick); border-color: var(--accent); }
textarea, input[type=text], select { width: 100%; font: inherit; padding: 8px; border: 1px solid var(--rule);
  border-radius: 6px; background: var(--card); color: var(--ink); }
.bar { display: flex; gap: 8px; justify-content: space-between; align-items: center; margin-top: 14px; flex-wrap: wrap; }
button { font: inherit; padding: 8px 14px; border-radius: 6px; border: 1px solid var(--accent); background: var(--accent);
  color: #fff; cursor: pointer; } button.secondary { background: transparent; color: var(--accent); }
button:disabled { opacity: .45; cursor: default; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.progress { height: 6px; background: var(--rule); border-radius: 3px; overflow: hidden; margin: 8px 0 0; }
.progress > div { height: 100%; background: var(--accent); }
</style></head>
<body><main id="app"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const STORE = 'ratings:' + D.kit;
let S = { rater: '', background: '', started: '', answers: {}, order: [], pos: 0 };
try { const saved = JSON.parse(localStorage.getItem(STORE) || 'null'); if (saved) S = saved; } catch (e) {}
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (e) {} };
// a small deterministic RNG seeded by the rater's alias, so each rater gets their own order and sides
function rngFor(seedText) { let h = 2166136261; for (const c of seedText) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); }
  return () => { h ^= h << 13; h ^= h >>> 17; h ^= h << 5; return ((h >>> 0) % 1000000) / 1000000; }; }
function setup(alias) {
  const r = rngFor(alias.trim().toLowerCase());
  const ids = D.items.map(i => i.id);
  for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
  S.order = ids.map(id => ({ id, left: r() < 0.5 ? 'p' : 'q' }));
}
const el = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === 'text') e.textContent = v; else e.setAttribute(k, v); }
  for (const k of kids) e.append(k); return e; };
const app = document.getElementById('app');
function render() { app.replaceChildren(); if (!S.rater) return renderStart(); if (S.pos >= S.order.length) return renderEnd(); renderItem(); }
function renderStart() {
  app.append(el('h1', { text: 'Rating fitness coaching messages' }),
    el('p', { text: `You will see ${D.items.length} short items. Each shows the facts of one situation (a week, a check during the day, a morning, or a question asked in a chat) and two messages an assistant wrote for it. For each item, answer three questions. There are no right answers; judge each message against the facts shown.` }),
    el('p', { class: 'muted', text: `About ${D.minutes} minutes. You do not have to finish: after a few items a button lets you stop and send what you have, and partial answers are still useful. Your answers are saved in this browser as you go, so you can also come back on the same phone or computer. At the end you send your ratings with one tap (if the page was opened from a link) or copy them into a message to the person who asked you. Ratings are anonymous and used only for research.` }));
  const alias = el('input', { type: 'text', id: 'alias', placeholder: 'e.g. initials or a nickname', autocomplete: 'off' });
  const bg = el('select', { id: 'bg' }, el('option', { value: '', text: 'Choose one' }),
    el('option', { value: 'none', text: 'No particular fitness or nutrition background' }),
    el('option', { value: 'some', text: 'I track my own training or food' }),
    el('option', { value: 'professional', text: 'Coach, trainer, dietitian or similar' }));
  const go = el('button', { text: 'Start' });
  go.onclick = () => { if (!alias.value.trim() || !bg.value) { alert('Please enter an alias and choose your background.'); return; }
    S.rater = alias.value.trim(); S.background = bg.value; S.started = new Date().toISOString(); setup(S.rater); S.pos = 0; save(); render(); };
  app.append(el('div', { class: 'card' }, el('label', { for: 'alias', text: 'Your alias (no real name needed)' }), alias,
    el('div', { style: 'height:10px' }), el('label', { for: 'bg', text: 'Your background' }), bg, el('div', { class: 'bar' }, go)));
}
function renderItem() {
  const slot = S.order[S.pos], item = D.items.find(i => i.id === slot.id);
  const sides = slot.left === 'p' ? { A: 'p', B: 'q' } : { A: 'q', B: 'p' };
  const a = S.answers[slot.id] || { t0: Date.now() };
  if (!a.t0) a.t0 = Date.now();
  app.append(el('div', { class: 'muted', text: `Item ${S.pos + 1} of ${S.order.length} - ${item.message}` }),
    el('div', { class: 'progress' }, el('div', { style: `width:${100 * S.pos / S.order.length}%` })),
    el('div', { class: 'card facts' }, el('h2', { text: 'The facts' }), el('p', { text: item.facts })),
    el('div', { class: 'pair' },
      el('div', { class: 'card' }, el('h2', { text: 'Message A' }), el('div', { class: 'msg', text: item[sides.A] })),
      el('div', { class: 'card' }, el('h2', { text: 'Message B' }), el('div', { class: 'msg', text: item[sides.B] }))));
  const form = el('div', { class: 'card' });
  const labels = { A: 'A is better', B: 'B is better', equal: 'About the same', both_poor: 'Both are poor' };
  for (const [q, text] of Object.entries(D.questions)) {
    const fs = el('fieldset', {}, el('legend', { text }));
    const opts = el('div', { class: 'opts' });
    for (const choice of ['A', 'B', 'equal', 'both_poor']) {
      const value = choice === 'A' ? sides.A : choice === 'B' ? sides.B : choice;
      const input = el('input', { type: 'radio', name: q, value });
      if (a[q] === value) input.checked = true;
      input.onchange = () => { a[q] = value; S.answers[slot.id] = a; save(); next.disabled = !complete(); };
      opts.append(el('label', {}, input, labels[choice]));
    }
    fs.append(opts); form.append(fs);
  }
  const note = el('textarea', { rows: '2', placeholder: 'Optional comment (for example, an error you noticed)' });
  note.value = a.comment || ''; note.oninput = () => { a.comment = note.value; S.answers[slot.id] = a; save(); };
  form.append(el('fieldset', {}, el('legend', { text: 'Comment (optional)' }), note));
  const complete = () => Object.keys(D.questions).every(q => a[q]);
  const back = el('button', { class: 'secondary', text: 'Back' }); back.disabled = S.pos === 0;
  back.onclick = () => { S.pos--; save(); render(); window.scrollTo(0, 0); };
  const next = el('button', { text: S.pos === S.order.length - 1 ? 'Finish' : 'Next' }); next.disabled = !complete();
  next.onclick = () => { a.ms = (a.ms || 0) + (Date.now() - a.t0); a.t0 = 0; a.left = slot.left; S.answers[slot.id] = a;
    S.pos++; save(); render(); window.scrollTo(0, 0); };
  form.append(el('div', { class: 'bar' }, back, next));
  if (S.pos >= 8 && S.pos < S.order.length - 1) {            // after eight items a rater may stop and send what is done
    const stop = el('button', { class: 'secondary', id: 'stop', text: "Stop here and send what I've done" });
    stop.onclick = () => { S.pos = S.order.length; save(); render(); window.scrollTo(0, 0); };
    form.append(el('div', { class: 'bar' }, stop));
  }
  app.append(form);
}
function ratingOut() {
  const out = { kit: D.kit, rater: S.rater, background: S.background, started: S.started, finished: new Date().toISOString(),
    items: S.order.map(o => ({ id: o.id, left: o.left, ...(S.answers[o.id] || {}) }))
      .filter(i => Object.keys(D.questions).some(q => i[q])) };                 // only items with at least one answer
  out.items.forEach(i => delete i.t0);
  return out;
}
// the text a rater pastes into a message: the ratings as JSON, UTF-8, base64, in short lines between two marker lines
// (base64 survives chat apps, smart quotes and line wrapping; evals/human_rating.py reads it back)
function ratingBlock() {
  const out = ratingOut();
  const bytes = new TextEncoder().encode(JSON.stringify(out));
  let bin = ''; bytes.forEach(b => { bin += String.fromCharCode(b); });
  return 'BEGIN CALVIS RATINGS\nkit=' + D.kit + ' rater=' + S.rater + ' items=' + out.items.length + '\n' +
    btoa(bin).match(/.{1,64}/g).join('\n') + '\nEND CALVIS RATINGS';
}
async function copyBlock(box, status) {
  const text = box.value;
  try { await navigator.clipboard.writeText(text); status.textContent = 'Copied. Now paste it into a message to the person who asked you.'; return; } catch (e) {}
  box.focus(); box.select(); try { box.setSelectionRange(0, text.length); } catch (e) {}
  try { if (document.execCommand('copy')) { status.textContent = 'Copied. Now paste it into a message to the person who asked you.'; return; } } catch (e) {}
  status.textContent = 'Copying did not work here. Press and hold the text above, choose Select all, then Copy.';
}
function download() {
  const out = ratingOut();
  const blob = new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
  link.download = `ratings_${S.rater.replace(/[^A-Za-z0-9_-]+/g, '_')}.json`; link.click();
}
// sent to the small server that served this page (evals/rating_server.py); only when the page came from a web address
async function sendRatings(status, btn) {
  btn.disabled = true; status.textContent = 'Sending...';
  try {
    const r = await fetch('submit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ratingOut()) });
    const j = await r.json().catch(() => ({}));
    if (r.ok && j.ok) { status.textContent = 'Sent. Thank you! You can close this page.'; return; }
    status.textContent = 'Could not send (' + (j.error || r.status) + '). Please use "Copy my ratings" below and paste it into a message to the person who asked you.';
  } catch (e) {
    status.textContent = 'Could not reach the server. Please use "Copy my ratings" below and paste it into a message to the person who asked you.';
  }
  btn.disabled = false;
}
function renderEnd() {
  const online = location.protocol === 'http:' || location.protocol === 'https:';
  const box = el('textarea', { rows: '6', readonly: 'readonly', style: 'font:12px/1.3 monospace' });
  box.value = ratingBlock(); box.onfocus = () => box.select();
  const status = el('p', { class: 'muted', id: 'copystatus', text: '' });
  const copy = el('button', { id: 'copy', class: online ? 'secondary' : '', text: 'Copy my ratings' }); copy.onclick = () => copyBlock(box, status);
  const sendStatus = el('p', { class: 'muted', id: 'sendstatus', text: '' });
  const send = el('button', { id: 'send', text: 'Send my ratings' }); send.onclick = () => sendRatings(sendStatus, send);
  const dl = el('button', { class: 'secondary', text: 'Download as a file instead' }); dl.onclick = download;
  const back = el('button', { class: 'secondary', text: 'Review my answers' }); back.onclick = () => { S.pos = 0; save(); render(); };
  app.append(el('h1', { text: 'Thank you' }),
    el('p', { text: online ? 'Press "Send my ratings": that is all. If sending fails, copy them with the second button and paste the text into a message to the person who asked you.'
      : 'Press "Copy my ratings", then paste the text into a message (WhatsApp, Telegram or email) to the person who asked you. You do not need to type or edit anything.' }),
    ...(online ? [el('div', { class: 'bar' }, send), sendStatus] : []),
    el('div', { class: 'bar' }, copy), status, box,
    el('p', { class: 'muted', text: 'If copying does not work on your phone, press and hold the text above and choose Select all, then Copy. On a computer you can also save a file.' }),
    el('div', { class: 'bar' }, back, dl));
}
render();
</script></body></html>"""


# ── tallying the returned files ──────────────────────────────────────────────

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def krippendorff_alpha_nominal(units: list[list[str]]) -> float | None:
    """Krippendorff's alpha for nominal data. `units`: one list of values per item (values from different raters;
    items with fewer than two values are skipped). None when there is no disagreement to measure."""
    from itertools import permutations
    o = defaultdict(float)
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for i, j in permutations(range(m), 2):
            o[(vals[i], vals[j])] += 1 / (m - 1)
    n_c = defaultdict(float)
    for (c, _k), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    if n <= 1:
        return None
    d_o = sum(v for (c, k), v in o.items() if c != k)
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k) / (n - 1)
    return None if d_e == 0 else 1 - d_o / d_e


BEGIN_MARK, END_MARK = "BEGIN CALVIS RATINGS", "END CALVIS RATINGS"


def encode_block(rating: dict) -> str:
    """The text block the page's Copy button produces (kept here so the format is tested and a rater can be re-sent one)."""
    import base64
    raw = base64.b64encode(json.dumps(rating, ensure_ascii=False).encode("utf-8")).decode("ascii")
    lines = "\n".join(raw[i:i + 64] for i in range(0, len(raw), 64))
    return (f"{BEGIN_MARK}\nkit={rating.get('kit', '')} rater={rating.get('rater', '')} items={len(rating.get('items', []))}\n"
            f"{lines}\n{END_MARK}")


def decode_blocks(text: str) -> list[dict]:
    """Every rating block in `text` (a pasted message, a chat export, an email). Anything around a block is ignored, and so
    are the line breaks and stray characters inside it; a block that is cut off or damaged raises a ValueError."""
    import base64
    import re
    out = []
    for n, m in enumerate(re.finditer(re.escape(BEGIN_MARK) + r"(.*?)" + re.escape(END_MARK), text, re.S), 1):
        body = "\n".join(l for l in m.group(1).splitlines() if not l.strip().startswith("kit="))
        b64 = re.sub(r"[^A-Za-z0-9+/]", "", body)
        try:
            out.append(json.loads(base64.b64decode(b64 + "=" * (-len(b64) % 4)).decode("utf-8")))
        except Exception as e:
            raise ValueError(f"rating block {n} is damaged or incomplete ({type(e).__name__}); ask the rater to send it again") from e
    return out


def load_ratings(path) -> list[dict]:
    """The ratings in one file: a downloaded .json file, or any text with one or more pasted blocks (a .txt, a chat export)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if BEGIN_MARK in text:
        found = decode_blocks(text)
        if not found:
            raise ValueError(f"{path}: no complete rating block (the END line is missing)")
        return found
    return [json.loads(text)]


def tally(key_path: Path, rating_paths: list[Path]) -> dict:
    """Win rates of the labels design over the original prompt per message and question, rater agreement, rater
    consistency on the repeated items. Choices are mapped through the key: 'labels' / 'numbers' / 'equal' /
    'both_poor'."""
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    items, pairs = key["items"], key["pairs"]
    labels_setups = {lab for _base, lab in pairs.values()}
    every = [r for p in rating_paths for r in load_ratings(p)]
    skipped, latest = [], {}
    for r in every:
        if r.get("kit") not in (None, key["kit_id"]):          # a rating of an older kit has other item ids: never mix
            skipped.append(f"{r.get('rater')}: kit {r.get('kit')} is not {key['kit_id']}")
            continue
        who = str(r.get("rater", "")).strip().lower()           # a rater who sent twice counts once: the later one
        if who not in latest or str(r.get("finished", "")) >= str(latest[who].get("finished", "")):
            if who in latest:
                skipped.append(f"{r.get('rater')}: sent twice, the later one is used")
            latest[who] = r
    raters = list(latest.values())
    to_side = lambda iid, v: ("labels" if items[iid][v] in labels_setups else "numbers") if v in ("p", "q") else v
    asked = key.get("questions") or list(QUESTIONS)                   # a shorter kit asks fewer questions
    out = {"kit": key["kit_id"], "raters": [{"rater": r["rater"], "background": r["background"]} for r in raters],
           "questions": asked, "skipped": skipped, "by_message": {}, "agreement": {}, "consistency": {}}
    for message in pairs:
        ids = [i for i, k in items.items() if k["message"] == message and k["repeat_of"] is None]
        res = {}
        for q in asked:
            counts = defaultdict(int)
            for r in raters:
                for it in r["items"]:
                    if it["id"] in ids and it.get(q):
                        counts[to_side(it["id"], it[q])] += 1
            decisive = counts.get("labels", 0) + counts.get("numbers", 0)
            res[q] = {"counts": dict(counts), "labels_win_rate": _wilson(counts.get("labels", 0), decisive)}
            units = [[to_side(i, it[q]) for r in raters for it in r["items"] if it["id"] == i and it.get(q)] for i in ids]
            out["agreement"].setdefault(message, {})[q] = krippendorff_alpha_nominal(units)
        out["by_message"][message] = res
    for r in raters:
        got = {it["id"]: it for it in r["items"]}
        same = total = 0
        for iid, k in items.items():
            if k["repeat_of"] and iid in got and k["repeat_of"] in got:
                for q in asked:
                    a, b = got[iid].get(q), got[k["repeat_of"]].get(q)
                    if a and b:
                        total += 1
                        same += to_side(iid, a) == to_side(k["repeat_of"], b)
        out["consistency"][r["rater"]] = (same, total)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=2026)
    b.add_argument("--out", type=Path, default=KIT, help="folder for rate.html and key.json (a new kit goes in its own folder)")
    b.add_argument("--per-message", type=int, default=None, help=f"cases per message type (default {PER_MESSAGE})")
    b.add_argument("--questions", default=None, help=f"comma list from {','.join(QUESTIONS)} (default all)")
    b.add_argument("--repeats", type=int, default=None, help="items shown twice with the sides swapped (default one per type)")
    t = sub.add_parser("tally")
    t.add_argument("key", type=Path)
    t.add_argument("ratings", type=Path, nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        page, keyp = build(args.seed, args.out, args.per_message, args.questions.split(",") if args.questions else None, args.repeats)
        print(f"wrote {page} (send this to raters)\nwrote {keyp} (keep this; never send it)")
        return 0
    res = tally(args.key, args.ratings)
    SUMMARY.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{len(res['raters'])} raters: " + ", ".join(f"{r['rater']} ({r['background']})" for r in res["raters"]))
    for note in res["skipped"]:
        print("  note:", note)
    for message, qs in res["by_message"].items():
        for q, v in qs.items():
            p, lo, hi = v["labels_win_rate"]
            alpha = res["agreement"][message][q]
            print(f"{message:<20} {q:<9} labels preferred {p:.0%} [{lo:.0%}-{hi:.0%}] of decisive  counts {v['counts']}  "
                  f"alpha {'-' if alpha is None else f'{alpha:.2f}'}")
    print("consistency on repeated items:", res["consistency"])
    print(f"saved {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
