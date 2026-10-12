"""Draft ablation A9 for the paper: labels v6 WITH the figures added back (information-matched numeric baseline).

A reviewer's question: would a numeric prompt that has the same implications, priorities and answer plan as the label
design do nearly as well? A8 (numbers + v1 verdicts) does not answer it, because v1 has no implications, priorities or
plan. A9 keeps everything labels v6 gives the model (the per-metric meaning, the links, what to work on in order, the kind
of fix, what to leave unchanged, the one-sentence numbered plan, the same word limits) and changes only the metrics:
the production numeric block (weekly_analysis.format_data_summary) is added under the description, and the sentences that
forbid digits are replaced by "use only the figures given, as given".
  A9 vs A7  measures the digits, with the implications, priorities and plan identical in both;
  A9 vs A1  measures the implications, priorities and plan, with the figures identical in both.
Evaluation only: nothing in the app uses this. DRAFT: not yet scored or run; see the open question about the checklist.
"""
from __future__ import annotations

_OLD_INTRO = ("Every comparison with the targets has already been worked out, so there are no numbers "
              "on purpose.")
_NEW_INTRO = ("Every comparison with the targets has already been worked out for you in the description; the figures "
              "behind it are listed after it.")
_OLD_RULE = ("Write in words, never with digits, and in your own words rather than copying the description. ")
_NEW_RULE = ("When you give a figure, use one from the figures above exactly as given: never invent, estimate or round "
             "a figure differently. Write in your own words rather than copying the description. ")
_FIGURES_HEADER = "The figures behind the description (correct, do not recompute):\n"


def _swap(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"expected one occurrence of: {old[:50]}"
    return text.replace(old, new)


def _with_figures(prompt: str, data: dict, view: str) -> str:
    import weekly_analysis as wa
    prompt = _swap(prompt, _OLD_INTRO, _NEW_INTRO)
    prompt = _swap(prompt, _OLD_RULE, _NEW_RULE)
    return _swap(prompt, view + "\n\n", view + "\n\n" + _FIGURES_HEADER + wa.format_data_summary(data) + "\n\n")


def shown_block(data: dict) -> str:
    """Everything the model may cite: the description (both views) and the figures. The figure check compares against it."""
    import weekly_analysis as wa
    return wa._label4_facts_block(data) + wa._facts_block(data, wa.format_data_summary(data))


def answer_with_checks(model, prompt: str, data: dict, part: str) -> str:
    """One question, checked and redrafted at most once. These are the checks of the label design (weekly_analysis.
    _answer_with_label_checks, version 5) with one change: the 'no digits' check is replaced by the figure check the
    numeric designs use, because A9 is shown the figures and may state them."""
    import re

    import weekly_analysis as wa
    import weekly_labels_v4
    from agent import _clean_response
    from langchain_core.messages import HumanMessage

    resp = model.invoke([HumanMessage(content=prompt)])
    draft = _clean_response(getattr(resp, "content", "") or "")
    recommending = part == "recommendations"
    problems = []
    unsupported = wa._grounding_check(draft, shown_block(data), allow_proposals=recommending)
    if unsupported:
        problems.append("it contains figures that are not in the data: " + ", ".join(f"{n:g}" for n in unsupported))
    problems += wa._label_directional_check(draft, data)
    lab = weekly_labels_v4.classify(data)
    if recommending:
        needs = weekly_labels_v4.attention(lab)
        missed = [k for k in needs if not weekly_labels_v4.KEYWORDS[k].search(draft)]
        if missed:
            problems.append("it does not address " + ", ".join(missed) + ", which needs attention")
        first = wa._first_numbered_item(draft)
        if needs and first and not weekly_labels_v4.KEYWORDS[needs[0]].search(first):
            problems.append(f"recommendation one must address {needs[0]}, the first priority")
        for n in wa._keep_only_items(draft):
            problems.append(f"recommendation {n} only says to keep something as it is; drop it or replace it with a "
                            "concrete habit")
        want = len(wa.recommendation_plan(data))
        got = len(re.findall(r"(?:^|\s)\d[.)]\s+", draft))
        if got != want:
            problems.append(f"it has {got} numbered recommendations; write exactly {wa._COUNT_WORDS[want]}, one for "
                            "each line of the plan, in order")
    else:
        if lab["no_workouts"] and not weekly_labels_v4.KEYWORDS["workouts"].search(draft):
            problems.append("it does not mention that there were no workouts this week")
        m = wa._OBS_ADVICE_V4_RE.search(draft)
        if m:
            problems.append(f"it tells the user what to do ('{m.group(0)}'); the observations only describe the week, "
                            "they give no advice")
    if problems:
        resp = model.invoke([HumanMessage(content=(
            f"{prompt}\n\nYour previous draft had these problems: " + "; ".join(problems) + ".\n"
            "Redraft it fixing them, using only the description and figures above. Same format: one short paragraph of "
            "about 4 lines, no headings."))])
        draft = _clean_response(getattr(resp, "content", "") or "")
    return draft


def run(data: dict, model) -> str:
    """The A9 review: two checked questions, joined like every other design."""
    observations = answer_with_checks(model, observations_prompt(data), data, "observations")
    recommendations = answer_with_checks(model, recommendations_prompt(data, observations), data, "recommendations")
    return "\n\n".join(part for part in (observations, recommendations) if part)


def observations_prompt(data: dict) -> str:
    import weekly_analysis as wa
    import weekly_labels_v4
    return _with_figures(wa.build_label4_observations_prompt(data), data, weekly_labels_v4.observations_view(data))


def recommendations_prompt(data: dict, observations: str) -> str:
    import weekly_analysis as wa
    import weekly_labels_v4
    return _with_figures(wa.build_label6_recommendations_prompt(data, observations), data,
                         weekly_labels_v4.recommendations_view(data))
