"""Checks that this release reproduces what the paper reports. No model and no network are needed.

    python -m pytest tests -q
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _saved_fingerprints(prefix):
    out = {}
    for f in glob.glob(str(ROOT / "evals" / "results" / f"{prefix}_2*.json")):
        for setup, fp in json.loads(Path(f).read_text(encoding="utf-8")).get("fingerprints", {}).items():
            out.setdefault(setup, set()).add(fp)
    return out


def test_every_saved_run_was_made_by_the_code_in_this_folder():
    """The version fingerprint of each approach (model settings, every prompt for every case, the checking code) must
    equal the one stored with its saved answers. A mismatch means the code differs from what produced the answers."""
    from evals import run_message_eval, run_weekly_eval
    checked = 0
    for module, prefix in ((run_weekly_eval, "weekly"), (run_message_eval, "messages")):
        saved = _saved_fingerprints(prefix)
        for setup in module.SETUPS:
            if setup in saved:
                assert module.fingerprint(setup) in saved[setup], f"{prefix}: {setup} differs from its saved runs"
                checked += 1
    assert checked >= 26


def test_the_weekly_headline_numbers_of_the_paper():
    """Share of held-out answers passing every core item: A1 numbers 3%, A7 labels v6 87%, A9 labels v6 + figures 63%."""
    import agent
    from evals import output_metrics as om, run_weekly_eval

    agent._identity_header = lambda *a, **k: "You are Calvis, a friendly fitness coach. The user is G."
    fps = {s: run_weekly_eval.fingerprint(s) for s in om.APPROACHES}
    current = om.current_rows(om.load_rows(), fps)
    share = {}
    for setup in ("split_fast", "labels6_fast", "matched6_fast"):
        rows = [r for r in current[setup] if om.case_set_of(r["case"]) == "holdout"]
        scored = om.scored(rows)
        share[setup] = (len(rows), round(100 * sum(all(i.ok for i in r["core"]) for r in scored) / len(scored)))
    assert share == {"split_fast": (30, 3), "labels6_fast": (30, 87), "matched6_fast": (30, 63)}


def test_the_lean_chat_agent_writes_the_same_facts_as_the_application_did():
    """evals/chat_facts_reference.json holds the facts text the application's chat path wrote for each of the 66 chat
    cases. The lean agent must reproduce it word for word, so a model sees the same prompt in both."""
    from evals import chat_cases as cc, run_chat_lean
    reference = json.loads((ROOT / "evals" / "chat_facts_reference.json").read_text(encoding="utf-8"))
    assert len(reference) == len(cc.CASES) == 66
    for case_id, day_name, question, _kind, _set in cc.CASES:
        _inp, facts = run_chat_lean.facts_for(cc.DAYS[day_name], day_name, question)
        assert facts == reference[case_id], case_id


def test_the_checklists_accept_a_good_answer_and_reject_an_invented_figure():
    from evals import weekly_cases, weekly_rubric
    name, data = sorted(weekly_cases.CASES.items())[0]
    good = ("Steps held steady near the goal, calories landed close to the target and protein was met, a steady week of "
            "training. Workouts were regular and nothing needs attention beyond keeping the routine going.\n\n"
            "1. Keep the evening walk after dinner at the same time each day. 2. Prepare the protein breakfast the night "
            "before so it is ready every morning.")
    items = {i.id: i for i in weekly_rubric.score(good, data, mode="labels6")}
    assert items["two_paragraphs"].ok and items["no_markdown"].ok
    bad = good.replace("Steps held steady", "Steps averaged 99999 and held steady")
    assert not {i.id: i for i in weekly_rubric.score(bad, data, mode="labels6")}["observations_use_words_not_digits"].ok
