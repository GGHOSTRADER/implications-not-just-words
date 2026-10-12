"""The OLD weekly review: one big prompt asking for three stages and two paragraphs at once. Replaced on 2026-10-04 by
two focused questions (weekly_analysis.run_cot_analysis). Kept ONLY as a baseline for evals/run_weekly_eval.py, so the
claim "the split is better" can be re-tested on any model. Not used by the app.

It is judged by the corrected number check, like the current design, so the comparison is fair.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

import weekly_analysis as wa


def build_single_prompt(data: dict, summary_text: str) -> str:
    return (
        wa._data_block(data, summary_text)
        + "Reason through this in three internal stages, but output ONLY two short paragraphs "
        "(no labels, no numbering of the paragraphs, no headings, no greeting, no sign-off):\n"
        "1. Observations: restate the key numbers above in plain language, and note any "
        "correlations or hypotheses connecting them (e.g. consistency vs adherence, workout "
        "frequency vs calorie stability, time-of-day activity vs intake timing).\n"
        "2. Recommendations: 1-3 numbered, specific strategies tied directly to the "
        "observations above.\n\n"
        "The 'Vs targets' lines already state whether each metric is at/above or below "
        "target — use that status directly, do not recompute or contradict it. Never "
        "recommend increasing a metric whose status is 'target met' or 'target met/exceeded'.\n\n"
        "Each paragraph should be about 4 lines. Use ONLY the numbers given above — do not "
        "invent, estimate, or restate rounded numbers as different values. Plain text only, "
        "no markdown."
    )


def run(data: dict, summary_text: str, model) -> str:
    """One call, the same checks, at most one redraft: exactly the old flow."""
    from agent import _clean_response

    prompt = build_single_prompt(data, summary_text)
    draft = _clean_response(getattr(model.invoke([HumanMessage(content=prompt)]), "content", "") or "")
    shown = wa._facts_block(data, summary_text)
    unsupported = wa._grounding_check(draft, shown)
    directional = wa._directional_check(draft, data)
    if unsupported or directional:
        feedback = []
        if unsupported:
            feedback.append("Your previous draft mentioned these numbers, which do NOT appear in the data above: "
                            + ", ".join(f"{n:g}" for n in unsupported) + ".")
        if directional:
            feedback.append("Your previous draft also contained recommendations that contradict the data: "
                            + "; ".join(directional) + ".")
        redraft = (f"{prompt}\n\n" + "\n".join(feedback) + "\n"
                   "Redraft your response fixing these issues. Use ONLY the numbers given in the data above, and "
                   "respect the 'Vs targets' status for each metric. Same format: two short paragraphs, about 4 "
                   "lines each, no headings.")
        draft = _clean_response(getattr(model.invoke([HumanMessage(content=redraft)]), "content", "") or "")
    return draft
