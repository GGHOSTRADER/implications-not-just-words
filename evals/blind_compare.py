"""Blind A/B comparison of two setups, judged by a person.

    python -m evals.blind_compare build evals/results/weekly_<time>.json --a split_fast --b old_think
        -> blind_<time>.html (open it in a browser) and blind_<time>_key.json (which side was which; do not open yet)
    ... in the page: pick the better answer for each week, press "Download my picks" ...
    python -m evals.blind_compare tally ~/Downloads/picks.json evals/results/blind_<time>_key.json

Why blind: when you know which answer came from the "new" design you tend to prefer it. The page shows only "Answer A"
and "Answer B", the sides are shuffled per week, and the key stays in a separate file until the tally.
Self-contained page (no network, no libraries); picks are kept in the browser as you go, so you can stop and resume.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pairs(results: list[dict], a: str, b: str) -> list[tuple[str, str, str]]:
    """[(case, text_a, text_b)] for cases where BOTH setups produced an answer."""
    by = {}
    for r in results:
        by.setdefault(r["case"], {})[r["setup"]] = r["text"]
    return [(c, s[a], s[b]) for c, s in by.items() if a in s and b in s]


def build(results_path: Path, a: str, b: str, out_dir: Path | None = None, seed: int = 7):
    """Write the blind page and its key. Returns (html_path, key_path)."""
    import weekly_analysis as wa
    from evals import weekly_cases

    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    pairs = _pairs(data["results"], a, b)
    if not pairs:
        raise SystemExit(f"no case has answers for both {a!r} and {b!r} in {results_path}")
    rng = random.Random(seed)
    key, cards = {}, []
    for case, text_a, text_b in pairs:
        left_is_a = rng.random() < 0.5                       # shuffle which setup appears as "Answer A"
        shown = (text_a, text_b) if left_is_a else (text_b, text_a)
        key[case] = {"A": a if left_is_a else b, "B": b if left_is_a else a}
        summary = wa.format_data_summary(weekly_cases.CASES[case]) if case in weekly_cases.CASES else ""
        cards.append({"case": case, "summary": summary, "A": shown[0], "B": shown[1]})

    out_dir = Path(out_dir or Path(results_path).parent)
    stem = Path(results_path).stem.replace("weekly_", "blind_")
    html_path, key_path = out_dir / f"{stem}.html", out_dir / f"{stem}_key.json"
    html_path.write_text(_page(cards), encoding="utf-8")
    key_path.write_text(json.dumps({"pairs": key, "source": str(results_path), "setups": [a, b]}, indent=1), encoding="utf-8")
    return html_path, key_path


def tally(picks_path: Path, key_path: Path) -> dict:
    """{setup: wins, 'tie': n, 'both_bad': n, 'judged': n} from the downloaded picks and the key."""
    picks = json.loads(Path(picks_path).read_text(encoding="utf-8")).get("picks", {})
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))["pairs"]
    out = {s: 0 for s in json.loads(Path(key_path).read_text(encoding="utf-8"))["setups"]}
    out.update(tie=0, both_bad=0, judged=0)
    for case, choice in picks.items():
        if case not in key:
            continue
        out["judged"] += 1
        if choice in ("A", "B"):
            out[key[case][choice]] += 1
        elif choice == "tie":
            out["tie"] += 1
        elif choice == "bad":
            out["both_bad"] += 1
    return out


def _page(cards: list[dict]) -> str:
    payload = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    e = html.escape
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind compare</title>
<style>
:root {{ --bg:#fff; --fg:#1b1b1f; --muted:#666; --card:#f4f4f7; --line:#d6d6de; --accent:#4a5bd4; --ok:#2a9d5c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16161a; --fg:#ececf1; --muted:#9a9aa6; --card:#202027; --line:#34343e; --accent:#8b98ff; --ok:#4cc388; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif; }}
main {{ max-width:880px; margin:0 auto; padding:16px; }}
h1 {{ font-size:20px; margin:8px 0; }} .muted {{ color:var(--muted); font-size:13px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:2; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
.case {{ border:1px solid var(--line); border-radius:12px; margin:16px 0; padding:12px; }}
.case h2 {{ font-size:16px; margin:0 0 6px; }}
details {{ margin:6px 0 10px; }} pre {{ white-space:pre-wrap; background:var(--card); padding:8px; border-radius:8px; font-size:12.5px; }}
.answers {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
@media (max-width:640px) {{ .answers {{ grid-template-columns:1fr; }} }}
.ans {{ background:var(--card); border-radius:10px; padding:10px; white-space:pre-wrap; }} .ans b {{ display:block; margin-bottom:4px; }}
.picks {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
.picks button {{ min-height:40px; padding:0 14px; border-radius:10px; border:1px solid var(--line); background:transparent; color:var(--fg); font:600 14px system-ui; cursor:pointer; }}
.picks button.on {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.dl {{ min-height:40px; padding:0 14px; border-radius:10px; border:0; background:var(--ok); color:#fff; font:600 14px system-ui; cursor:pointer; }}
</style></head><body><main>
<h1>Which weekly review is better?</h1>
<p class="muted">Each week has two answers for the same numbers. Judge the <b>usefulness</b>: is it accurate to the data, specific, and something you would act on? You are not told which design wrote which. Your picks are kept in this browser, so you can stop and come back.</p>
<div class="bar"><span id="count"></span><button class="dl" id="dl">Download my picks</button></div>
<div id="cases"></div>
<script id="data" type="application/json">{payload}</script>
<script>
const cards = JSON.parse(document.getElementById('data').textContent);
const KEY = 'blind-compare-' + cards.map(c => c.case).join('|').length + '-' + cards.length;
let picks = {{}};
try {{ picks = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(picks)); }} catch (e) {{}} }};
const esc = s => s.replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
function render() {{
  const root = document.getElementById('cases'); root.textContent = '';
  cards.forEach(c => {{
    const d = document.createElement('div'); d.className = 'case';
    d.innerHTML = '<h2>' + esc(c.case) + '</h2>' +
      '<details><summary class="muted">The week\\'s numbers</summary><pre>' + esc(c.summary) + '</pre></details>' +
      '<div class="answers"><div class="ans"><b>Answer A</b>' + esc(c.A) + '</div><div class="ans"><b>Answer B</b>' + esc(c.B) + '</div></div>' +
      '<div class="picks"></div>';
    const row = d.querySelector('.picks');
    [['A','A is better'],['B','B is better'],['tie','About the same'],['bad','Both are poor']].forEach(([v, label]) => {{
      const b = document.createElement('button'); b.textContent = label; if (picks[c.case] === v) b.className = 'on';
      b.onclick = () => {{ picks[c.case] = v; save(); render(); }}; row.appendChild(b);
    }});
    root.appendChild(d);
  }});
  document.getElementById('count').textContent = Object.keys(picks).length + ' of ' + cards.length + ' judged';
}}
document.getElementById('dl').onclick = () => {{
  const blob = new Blob([JSON.stringify({{picks}}, null, 1)], {{type: 'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'picks.json'; a.click();
}};
render();
</script></main></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("results", type=Path)
    b.add_argument("--a", default="split_fast")
    b.add_argument("--b", default="old_think")
    t = sub.add_parser("tally")
    t.add_argument("picks", type=Path)
    t.add_argument("key", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        h, k = build(args.results, args.a, args.b)
        print(f"page: {h}\nkey (keep closed until you have judged): {k}")
    else:
        r = tally(args.picks, args.key)
        n = r["judged"] or 1
        print(f"{r['judged']} weeks judged")
        for name, v in r.items():
            if name != "judged":
                print(f"  {name:<12} {v:>3}  ({100 * v / n:4.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
