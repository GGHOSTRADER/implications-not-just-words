"""Ablation A8 for the paper: the weekly numbers block WITH every verdict stated, digits kept.

Labels v1 (A2) changed two things at once: it removed the digits, and it stated the verdict the numeric block left out
(the block has a "Vs targets" line for steps and protein but none for calories: verdict coverage 66%). This
representation keeps the numeric block exactly as production builds it (weekly_analysis.format_data_summary) and
appends labels v1's verdicts under a "use as-is" header, in the style of the block's own precomputed lines. The model
therefore sees v1's verdicts plus every figure, through A1's unchanged prompts and checks:
  A8 vs A1  measures complete verdicts (the figures are the same in both);
  A8 vs A2  measures the digits themselves (the verdicts are the same in both; A2 also uses the label checks).
Evaluation only: nothing in the app uses this.
"""
from __future__ import annotations

HEADER = "Verdicts worked out from the numbers above (use as-is, do not recompute):"


def verdict_section(data: dict) -> str:
    """Labels v1's view without its phase line (the numeric block already states the phase)."""
    import weekly_labels
    lines = [ln for ln in weekly_labels.label_summary(data).splitlines() if ln.strip() and not ln.startswith("Phase:")]
    return HEADER + "\n" + "\n".join(f"  {ln}" for ln in lines)


def summary(data: dict) -> str:
    """What A8 passes to weekly_analysis.run_cot_analysis in place of the numeric summary."""
    import weekly_analysis as wa
    return wa.format_data_summary(data) + "\n\n" + verdict_section(data)
