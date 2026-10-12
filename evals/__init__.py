"""Evaluation of what the local model writes (not unit tests: these need Ollama and take minutes).

  evals/weekly_cases.py   synthetic weeks to run the weekly review on
  evals/weekly_rubric.py  the checklist each answer is scored against (invented facts, format rules, coverage, specificity)
  evals/legacy_weekly.py  the old one-big-prompt design, kept as a baseline to compare against
  evals/run_weekly_eval.py   runs setups over the cases, saves answers + scores
  evals/blind_compare.py  builds a blind A/B page for a human and tallies the picks

The rubric functions are unit-tested (tests/test_weekly_rubric.py); running against the model is manual:
    python -m evals.run_weekly_eval --setups split_fast,old_think
"""
