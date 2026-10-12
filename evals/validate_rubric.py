"""Precision check of the rubric's semantic items, by reading.

    python -m evals.validate_rubric sample [--per-item 12]   # print a seeded sample of flags to read, with ids
    python -m evals.validate_rubric record <file.json>       # store the reading verdicts into rubric_v2_validation.json

A regular-expression rubric can be wrong in two ways: it flags a good answer (false positive) or misses a bad one
(false negative). Precision, the share of flags that are right, can be measured by reading what an item flags; that
is what this script supports. Recall is not measured (it would need every unflagged answer read).

The sample is drawn from the current-version answers of every approach (the ones the report pools), seeded, so the
same command shows the same flags. The verdicts file is written by the reader: {flag_id: true|false, ...} plus notes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VALIDATION = Path(__file__).parent / "rubric_v2_validation.json"
SEMANTIC = ["fix_matches_problem", "first_recommendation_targets_a_problem", "large_calorie_gap_first",
            "no_extra_movement_while_underfed", "no_unsupported_link", "consequence_matches_metric",
            "no_keep_only_recommendation", "respects_target_status"]


def flags() -> dict:
    """{item: [(flag_id, setup, case, detail, text)]} over the current-version answers."""
    import agent
    from evals import output_metrics as om, rubric_v2, run_weekly_eval as rw
    agent._identity_header = lambda *a, **k: "You are Calvis, a friendly fitness coach. The user is G."
    fps = {s: rw.fingerprint(s) for s in om.APPROACHES}
    cases = om._cases()
    out = defaultdict(list)
    seen = set()
    for s, rows in om.current_rows(om.load_rows(), fps).items():
        for r in rows:
            key = (s, r["case"], r["text"])
            if key in seen:
                continue
            seen.add(key)
            for i in rubric_v2.score(r["text"], cases[r["case"]]):
                if not i.ok and i.id in SEMANTIC:
                    fid = f"{i.id}:{s}:{r['case'].split()[0]}:{r.get('repeat', 0)}"
                    out[i.id].append((fid, s, r["case"], i.detail, r["text"]))
    return out


_OBS_ITEMS = {"no_unsupported_link", "consequence_matches_metric"}


ROUND1 = Path(__file__).parent / "rubric_v2_reading_round1.json"


def already_read() -> set:
    """Flag ids read in round 1, so a fresh sample never shows them again."""
    if not ROUND1.exists():
        return set()
    doc = json.loads(ROUND1.read_text(encoding="utf-8"))
    return {f"{item}:{fid}" for item, ids in doc.get("read", {}).items() for fid in ids}


def sample(per_item: int, seed: int = 7, only: str | None = None, fresh: bool = False) -> None:
    """Print a seeded sample of each item's flags, showing the paragraph the item judges (observations for links and
    consequences, recommendations for the rest) and the week's ground truth. fresh=True skips round-1 flags."""
    from evals import output_metrics as om, rubric_v2
    rnd = random.Random(seed)
    cases = om._cases()
    skip = already_read() if fresh else set()
    for item, fl in sorted(flags().items()):
        if only and item != only:
            continue
        fl = [f for f in fl if f[0] not in skip]
        picked = fl if len(fl) <= per_item else rnd.sample(fl, per_item)
        print(f"\n######## {item}: {len(fl)} flags, showing {len(picked)}")
        for fid, s, case, detail, text in picked:
            paras = rubric_v2.paragraphs(text)
            part = (paras[0] if item in _OBS_ITEMS else paras[-1]) if len(paras) == 2 else text
            st = rubric_v2.reference_state(cases[case])
            print(f"\n--- {fid}  truth={ {k: v for k, v in st.items() if k != 'cal_gap'} } cal_gap={st['cal_gap']:.0%}"
                  f"\n    detail: {detail}\n    " + part.replace("\n", "\n    "))


def record(path: Path) -> None:
    verdicts = json.loads(path.read_text(encoding="utf-8"))
    by_item = defaultdict(lambda: [0, 0])
    for fid, ok in verdicts["verdicts"].items():
        item = fid.split(":")[0]
        by_item[item][1] += 1
        by_item[item][0] += bool(ok)
    doc = {
        "_readme": "Precision of rubric v2 semantic items, measured by reading a seeded sample of flags "
                   "(python -m evals.validate_rubric sample). correct = flags the reader agreed with.",
        "date": f"{datetime.now():%Y-%m-%d %H:%M}", "reader": verdicts.get("reader", ""),
        "items": {k: {"read": t, "correct": c, "precision": round(c / t, 3) if t else None}
                  for k, (c, t) in sorted(by_item.items())},
        "notes": verdicts.get("notes", {}),
        "verdicts": verdicts["verdicts"],
    }
    VALIDATION.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(doc["items"], indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--per-item", type=int, default=12)
    s.add_argument("--item", help="only this item")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--fresh", action="store_true", help="exclude every flag read in round 1")
    r = sub.add_parser("record")
    r.add_argument("file", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        sample(args.per_item, seed=args.seed, only=args.item, fresh=args.fresh)
    else:
        record(args.file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
