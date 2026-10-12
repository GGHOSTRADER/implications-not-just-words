# Implications, Not Just Words

Code, synthetic test cases, saved model answers and checklists for the paper *Implications, Not Just Words: Label
Engineering for a Small Local Language Model Writing Fitness Coaching Messages* (Guillermo Alfonso Medrano Sanchez).
The paper is in `paper/`. The arXiv link will be added here after submission.

The paper asks what a small local language model (qwen3:8b, 4-bit, run with Ollama) needs in order to write short fitness
coaching messages that are correct. It compares ways of showing the model a user's week: raw numbers, verbal verdicts,
verdicts with what they mean for the user's goal, and the same with the figures put back (ablation A9). Code decides
everything the message depends on, and the model only writes the sentences. Messages: a weekly review, a checkpoint
comment, a morning briefing and a chat answer.

## What is in this folder

| Path | What it is |
|---|---|
| `weekly_analysis.py`, `weekly_labels*.py` | the weekly review: prompts, answer checks and the label versions v1 to v6 |
| `checkpoint_labels*.py`, `briefing_labels*.py`, `label_context_v3.py` | the checkpoint comment and morning briefing labels |
| `background_checks.py` | only the prompt builders and answer checks the evaluation calls, copied verbatim from the application |
| `chat_facts.py` | the chat design that writes the day in words (code from 5 October 2026) |
| `agent.py`, `llm_registry.py` | small stand-ins for two application modules (see below) |
| `evals/` | cases, frozen checklists, runners, metrics and the report generator |
| `evals/results/` | every saved answer the paper reports (weekly, messages, chat, other models) |
| `evals/REPORT.md` | the report with every table in the paper, generated from the saved answers |
| `paper/` | the paper source, bibliography, figures and the PDF |
| `tests/test_release.py` | checks that this folder reproduces the paper's numbers |

## Recompute the tables from the saved answers (no model needed)

```
pip install -r requirements.txt
python -m pytest tests -q            # fingerprints and the headline numbers
python -m evals.report               # rewrites evals/REPORT.md
```

Every saved run carries a *fingerprint*: a hash of the model settings, every prompt for every case, and the source of
the checking code. `tests/test_release.py` recomputes it and compares it with the saved one, so you can see that the
code in this folder is the code that produced those answers (26 approach versions are checked).

Regenerating the report gives the same tables as the committed `evals/REPORT.md`, with two differences. The "most
frequent phrase" column in section 5.3 can differ when two phrases are tied. And section 8 (human ratings) is empty,
because the raw ratings are not released; the committed report keeps the aggregate table that is also in the paper.

## Generate new answers

Needs [Ollama](https://github.com/ollama/ollama) with `qwen3:8b` (the saved runs used model id 500a1f067a9f, Ollama
0.34.4, an 8 GB RTX 4060, temperature 0.1). Set `MODEL_NAME` and `MODEL_BASE_URL` in a `.env` file to use another model.

```
python -m evals.run_weekly_eval  --setups split_fast,labels6_fast,matched6_fast --set holdout --repeats 3
python -m evals.run_message_eval --setups checkpoint_numbers,checkpoint_labels3 --set holdout --repeats 3
python -m evals.run_chat_lean    --set fresh2 --repeats 1
```

`evals/run_chat_lean.py` is a small chat agent written for this release. For a fake day and a question it does what the
paper's words design does: write the day's facts in words (`chat_facts.py`), make one call to the model, check the answer
and redraft at most twice, tidy it, and score it with the chat checklist. It has no tools, no graph and no database. The
calorie target of each fake day comes from `evals/chat_targets.json`, because the application's step-bonus rule is not
released. `tests/test_release.py` checks that the facts it writes equal, word for word, the facts the application's chat
wrote for all 66 chat cases. On the `fresh2` set (20 cases) it passed 18 answers at its first run, against 54 of 60 in
the paper.

Answers are sampled, so a new run will not equal the saved one word for word; compare shares and intervals, not text.
The report pools only runs whose fingerprint equals the current code's.

## What is not here

- **The application.** The label code is the paper's subject and is released. The rest of the application (Telegram bot,
  database, scheduler, decision engine, other features) is not. The evaluation reaches into it in a few places, and
  `agent.py`, `llm_registry.py` and `background_checks.py` replace those: the first two keep only the answer cleaner and
  the local model settings, the third keeps the 22 functions the evaluation calls. Nothing was rewritten.
- **The application's chat agent and the original chat design.** The saved chat answers in `evals/results/` were produced
  through the application's agent graph, which is not released. Instead this folder has a lean chat agent
  (`evals/run_chat_lean.py`, see below) that runs the paper's words design on the same cases. The original design (raw
  figures through tools) cannot be rerun here; its saved answers are included and scored with the same checklist. The chat
  code is the version of 5 October 2026 that the paper evaluates; the application has changed since.
- **Human ratings.** The six raters' answers and comments are not released (the paper reports aggregates only). The kit
  builder is in `evals/human_rating.py`.
- **Later work** on the workout matrix and exercise knowledge, which is not part of this paper.

## Code versions

The label and prompt code is copied from the application at commit `19077f5` (5 October 2026), after every run the paper
reports. The additions for ablation A9 (`evals/matched_numeric_weekly.py` and small changes to the runner, the scorer and
the report) are from 12 October 2026.

## Licences

- Code: PolyForm Noncommercial License 1.0.0 (`LICENSE`). Free for research, study and other noncommercial use;
  commercial use needs a separate licence from the author.
- Synthetic cases, saved answers and the generated report: CC BY 4.0 (`LICENSE-DATA.md`).
- The model, Qwen3, has its own licence, and is not distributed here.

## AI assistance

An AI assistant (Claude, Anthropic) helped write the code, the checklists and the text of the paper. The secondary judge
in the paper's blind judgments is a model of the same family. The author is responsible for the content.
