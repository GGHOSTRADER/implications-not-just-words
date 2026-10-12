"""Measure the message checklist's precision by reading what each item flags.

    python -m evals.validate_message_rubric sample [--per-item 10] [--seed 7]   # print a seeded sample of flags
    (read them, then write evals/message_rubric_validation.json: per item, flags read and how many were right)

Only answers from runs at a setup's current fingerprint are sampled, re-scored with the current checklist, so the
precision belongs to the checklist version the report uses. Recall (failures the checklist misses) is not measured.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def flags() -> dict:
    """{item id: [(row, item)]} for every failed item on current-version answers."""
    from evals import message_metrics as mm, run_message_eval as rme
    fps = {s: rme.fingerprint(s) for s in rme.SETUPS}
    rows = mm.rescore([r for rs in mm.current_rows(mm.load_rows(), fps).values() for r in rs])
    out = defaultdict(list)
    for r in rows:
        for i in r["now"]:
            if not i.ok:
                out[i.id].append((r, i))
    return out


def _part(row: dict, item_id: str) -> str:
    a = row["answer"]
    if isinstance(a, str):
        return a
    return a.get("yesterday", "") if item_id.startswith("by_") else a.get("today", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample"])
    ap.add_argument("--per-item", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    from evals import message_rubric as mr
    fl = flags()
    print(f"checklist {mr.MESSAGE_RUBRIC_VERSION}; flags per item: "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(fl.items())))
    rng = random.Random(args.seed)
    for item_id in sorted(fl):
        picked = rng.sample(fl[item_id], min(args.per_item, len(fl[item_id])))
        print(f"\n======== {item_id}: {len(picked)} of {len(fl[item_id])} — {mr.ITEM_DOCS.get(item_id, '')}")
        for n, (r, i) in enumerate(picked, 1):
            print(f"--- [{n}] {r['setup']} | {r['case']} | run {r.get('repeat', 0) + 1} | {i.detail}")
            print("    " + _part(r, item_id).replace("\n", " ").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
