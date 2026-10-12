"""The record of every attempt at the weekly review, for the report (evals/report.py) and the paper.

Each attempt is one approach that stays runnable (its setup in evals/run_weekly_eval.SETUPS). The narrative here is
written by hand when an attempt is made; every number the report shows is computed from saved answers, so update the
narrative when an attempt changes, never the numbers. Add a new attempt rather than editing an old one.
"""
from __future__ import annotations

TITLE = "Words, not numbers: label engineering for a small local language model writing weekly fitness reviews"

QUESTION = (
    "Can a small local language model (qwen3:8b, 4-bit, on an 8 GB consumer GPU) write a reliable weekly fitness "
    "review if deterministic code first turns the week's numbers into words (labels), and how far should that go: "
    "verdicts only, or also meaning, priorities and the shape of the answer?"
)

SYSTEM = (
    "CALVIS is a personal fitness assistant on Telegram. Once a week it writes a two-paragraph review: observations "
    "about the week (steps, calories, protein, workouts against the user's goals), then one to three numbered, "
    "concrete recommendations. Everything runs locally: qwen3:8b through Ollama in fast (non-thinking) mode unless "
    "stated, on an RTX 4060 with 8 GB. A review is two calls to the model (one per paragraph); a deterministic check "
    "after each call can send the draft back once with the problems named (a 'redraft')."
)

DATA = (
    "Thirty synthetic weeks, each the exact dictionary the production code builds from the database (averages, "
    "day-to-day variability, daily slope, goals, workouts, peak step hour, days within ten kcal of the calorie "
    "target). Twenty TUNING weeks (evals/weekly_cases.py) cover the situations where a small model goes wrong: a met "
    "target it must not ask to increase, far-off targets, no workouts, a metric with no data, erratic days, each phase. "
    "Ten HOLDOUT weeks (evals/weekly_holdout_cases.py) were written later as different situations (eight of ten are "
    "verdict combinations the tuning weeks lack) and are never used to tune anything."
)

# id, setup, date, name, representation, what changed, why, what happened (qualitative; numbers live in the tables)
ATTEMPTS = [
    {
        "id": "A0", "setup": "old_think", "date": "before 2026-10-04", "name": "One big prompt (thinking)",
        "representation": "numbers",
        "change": "The original design: the week as numbers (averages, CV%, slopes, a 'Vs targets' line for steps "
                  "and protein) and ONE prompt asking the model to reason in three internal stages and write both "
                  "paragraphs; thinking mode; one redraft if a number is not in the data.",
        "why": "Baseline.",
        "result": "Slow (about thirty seconds a review), recommendations often unnumbered prose, correlations "
                  "invented between the time of day, workouts and eating. Its content was often on the right side "
                  "(a food fix for a calorie problem), which the semantic items credit.",
    },
    {
        "id": "A1", "setup": "split_fast", "date": "2026-10-04 00:20-00:48", "name": "Two focused questions (numbers)",
        "representation": "numbers",
        "change": "Two smaller questions (observations; then recommendations, shown the observations) in fast mode, "
                  "each with its own number check and one redraft. The checks were then corrected: magnitudes "
                  "compared without sign, targets in the header counted as data, '2,200' read as one number, "
                  "proposals such as '20 minutes' allowed in recommendations, the direction check applied to "
                  "recommendations only.",
        "why": "A small model drops numbers and contradicts targets when asked for everything at once.",
        "result": "Format and speed improved. Two failures remained that the checks could not see: the model called "
                  "2650 kcal 'slightly below' a 2000 kcal target (the numbers block gave a verdict line for steps and "
                  "protein but none for calories), and it copied the prompt's example habit ('20-minute walk after "
                  "dinner') into 18 of 20 answers.",
    },
    {
        "id": "A2", "setup": "labels_fast", "date": "2026-10-04 02:55", "name": "Labels v1: verdicts as words",
        "representation": "labels_v1",
        "change": "Deterministic code (weekly_labels.py) turns every number into words before the model sees "
                  "anything: each average as its position against the goal ('a little below the goal'), the trend, "
                  "the day-to-day consistency, workouts, the active time of day, a 'needs attention / on track' "
                  "list. No digit reaches the model. Checks: no digits in observations, no invented figures, no "
                  "contradiction of a verdict (calories included), every off-target metric addressed.",
        "why": "The user's principle: a small model reasons well over text and badly over numbers, so numbers go to "
               "deterministic functions that produce labels, and the model reasons over the labels.",
        "result": "The calorie-direction errors and the parroted example disappeared; it won 15 of 20 blind "
                  "comparisons against A1 and A0. Remaining: it ignored 'no workouts', invented links between "
                  "metrics, read 'calorie target reached: never' as 'under target' (wording fixed the same night), "
                  "and invented generic habits for a good week.",
    },
    {
        "id": "A3", "setup": "labels2_fast", "date": "2026-10-04 03:11", "name": "Labels v2: size, direction, goal",
        "representation": "labels_v2",
        "change": "v1 plus the size of each gap in words, the direction to move ('eat less'), a 'keep as they are, "
                  "do not ask for more' list, an 'advice goal' line, no workouts as a problem, and several wordings "
                  "per verdict.",
        "why": "v1's failures were things the labels did not say.",
        "result": "Advice targeted the right metric more often, but the keep-as-is list made the model spend a "
                  "recommendation slot on 'keep calories exactly as they are' in about half the answers, so "
                  "specificity and format scores fell. More information is not automatically better.",
    },
    {
        "id": "A4", "setup": "labels3_fast", "date": "2026-10-04 03:22", "name": "Labels v3: meaning",
        "representation": "labels_v3",
        "change": "Labels that carry meaning, not only position: what each gap does to this phase's goal (calories "
                  "over in a cut erase the deficit; in a bulk they add fat), whether each trend moves toward or away "
                  "from the target, the only links between metrics that follow from the verdicts, priorities "
                  "weighted by phase (under-eating outranks under-moving), the kind of fix (food / movement / "
                  "training; a protein swap vs more food), and what to keep an eye on (drifting or erratic). Guards: "
                  "recommendation one must take the first priority; a keep-only recommendation is redrafted; "
                  "'should' in the observations is redrafted.",
        "why": "A judge reading v1/v2 answers saw that the advice could not weigh a problem it did not understand.",
        "result": "Best substance so far: it won all ten blind holdout comparisons against v1 and passed nearly every "
                  "semantic item. Its observations degraded: advice wording ('focus on'), a meaning attached to the "
                  "wrong metric ('protein was under, which means less movement'), headline copied verbatim.",
    },
    {
        "id": "A5", "setup": "labels4_fast", "date": "2026-10-04 03:59", "name": "Labels v4: one view per question",
        "representation": "labels_v4",
        "change": "The same verdicts and meanings as v3, presented as two views: the observations question sees what "
                  "happened and what it means (each meaning inline in its metric's sentence, nothing that asks for "
                  "action, an explicit ban when no link exists); the recommendations question sees what to work on, "
                  "in order, with the kind of fix, and what to leave unchanged as a constraint. The observations "
                  "guard covers all advice wording.",
        "why": "v3's observations were shown priorities and fixes and drifted into advice; separate 'What it means' "
               "sentences drifted to the wrong metric.",
        "result": "See the tables. In a first check on the holdout weeks the observations stopped giving advice and "
                  "every semantic item passed, but the model still padded single-problem weeks with a 'keep X as it "
                  "is' recommendation.",
    },
    {
        "id": "A6", "setup": "labels5_fast", "date": "2026-10-04 04:03", "name": "Labels v5: code decides the plan",
        "representation": "labels_v5",
        "change": "v4 plus one change: code decides how many recommendations to write and what each one is for "
                  "(one per priority, at most three, never fewer than two: a single problem gets two different "
                  "habits; a quiet week gets protect-the-routine and guard-against). A wrong count is redrafted.",
        "why": "With one problem the model filled three slots by padding (v3 tuning answers 4/20, v4 holdout check "
               "4/10). Another decision moved out of the model.",
        "result": "The content followed the plan (the right habits, in priority order), but the plan was listed as "
                  "'Recommendation one: a habit for calories.' and the model copied that label format into its "
                  "answer ('Recommendation one: Make a habit of ...') instead of numbering, so the count check "
                  "redrafted nearly every answer and the numbered-format item failed: the model mirrors the "
                  "structure of the instructions in its output.",
    },
    {
        "id": "A7", "setup": "labels6_fast", "date": "2026-10-04 04:58", "name": "Labels v6: the plan as a sentence",
        "representation": "labels_v6",
        "change": "v5 with one change: the same plan is stated as one sentence ('exactly two recommendations, "
                  "numbered 1. and 2. Plan them like this: the first is a habit for calories; the second ...'). "
                  "Labels, views and checks are v5's.",
        "why": "v5's list format leaked into the answers.",
        "result": "The plan's benefit stayed and the format failure disappeared: the best approach on every summary "
                  "measure, on the holdout weeks as on the tuning weeks, with the fewest redrafts among the label "
                  "versions after v1 and the shortest time per review.",
    },
    {
        "id": "A8", "setup": "hybrid_fast", "date": "2026-10-05 (run 2026-10-04 11:56-12:11)",
        "name": "Ablation: numbers plus v1's verdicts",
        "representation": "numbers_verdicts",
        "change": "A1's unchanged prompts and checks, with the numeric block kept in full and labels v1's verdict lines "
                  "appended under 'use as-is'. The model sees every figure and every verdict, including the calorie "
                  "verdict the numeric block lacked.",
        "why": "A2 changed two things at once: it removed the digits and it stated the missing calorie verdict. A8 vs "
               "A1 isolates complete verdicts; A8 vs A2 isolates the digits.",
        "result": "Complete verdicts alone did almost nothing: A8 matches A1 on every item (22% vs 20% of tuning "
                  "answers passing every item, 13% vs 3% on holdout weeks, both differences not significant) and its "
                  "hallucination items stay at 77-78%, against 89-93% for A2. So the drop in hallucinations comes "
                  "from removing the digits, not from the added verdict. Correct advice needed neither: the share "
                  "of answers passing every semantic item stayed at 40% / 37%, like A1 (40% / 17%).",
    },
    {
        "id": "A9", "setup": "matched6_fast", "date": "2026-10-12",
        "name": "Ablation: labels v6 plus the figures (information-matched numeric)",
        "representation": "numbers",
        "change": "A7's description, priorities, kind of fix, links, one-sentence numbered plan and word limits, unchanged. "
                  "The production numeric block is added under the description and the 'no digits' rules are replaced by "
                  "'use a figure from the figures above exactly as given'. The checks are A7's, with the no-digits check "
                  "replaced by the figure check of the numeric designs. Scored with A7's checklist, minus the no-digits item.",
        "why": "A8 kept the digits but added only v1's verdicts, which carry no implications, priorities or plan. A "
               "reviewer's question: would a numeric prompt with the same implications, priorities and plan do as well? "
               "A9 vs A7 isolates the digits; A9 vs A1 isolates the implications, priorities and plan.",
        "result": "The advice is as good as A7's: every semantic item passed in 100% of holdout and tuning answers (A7 "
                  "100% / 95%, A1 17% / 40%), so implications, priorities and plan make the advice correct with or "
                  "without the digits. The digits bring the factual errors back: invented figures in 23% of holdout "
                  "answers (A7 0%, A1 30%) and 10% of tuning answers (A7 0%, A1 12%). Answers passing every core item: "
                  "63% on holdout weeks (A7 87%, A1 3%) and 88% on tuning weeks (A7 92%, A1 20%). On holdout weeks A9 "
                  "vs A7 is -0.019 core items per week (0 better, 6 tied, 4 worse, p = 0.125), too few weeks to call; "
                  "A9 vs A1 is +0.091 (10 better, p = 0.002). It is also slower than A7 (6.3 s against 5.2 s) and "
                  "redrafts more often (50% against 33%).",
    },
]

# What the attempts taught, each tied to evidence in the report (section numbers refer to the report).
LESSONS = [
    "The numeric prompt stated a verdict for only two of three metrics (verdict coverage 66%, section 5.1), and the "
    "model inferred the missing one wrongly ('2650 kcal, slightly below the 2000 kcal target'). Stating every "
    "comparison in words removed that class of error (hallucination column, A1 vs A2, section 5.2).",
    "Words alone did not make the advice right: with labels v1 the semantic pass rate stayed as low as with numbers "
    "(section 5.2). The model could read where each metric sat but not what it meant for the goal.",
    "Meaning is what made the advice right. Labels that say what a gap does to this phase's goal, whether a trend "
    "moves toward or away from the target, which links are real, what matters most and what kind of fix fits (v3) "
    "raised the semantic pass rate to nearly every answer, on the untouched holdout weeks as well (sections 5.2, 5.5).",
    "More information can hurt when it reads like an item to act on. v2's 'keep as they are' list became wasted "
    "'keep X as it is' recommendations (sections 5.2, 5.4).",
    "Show each question only what it needs. One view per question (v4) shortened the prompts and cut the "
    "directives (section 5.1) and stopped the observations drifting into advice (instruction column, section 5.2).",
    "The model copies the structure of its instructions. A plan listed as 'Recommendation one: ...' came back in "
    "that format instead of numbered (v5); the same plan as one sentence (v6) kept the benefit and fixed the format.",
    "Each decision moved out of the model and into deterministic code (the verdicts, their meaning, the priority "
    "order, how many recommendations and what each is for) made the answers more correct; what is left to the "
    "model is the wording.",
    "Thinking is not a substitute for representation: the single thinking prompt (A0) was the slowest, failed the "
    "format, and was less often right than any meaning-bearing label version (section 5.2).",
    "Measure meaning, not only form. A format-and-hallucination checklist ranked v3 below v1 while blind reading "
    "preferred v3 in all ten holdout weeks; a semantic checklist, validated by reading what it flags (section 5.7), "
    "agreed with that reading (section 5.6).",
]

INSTRUMENTS = [
    ("Rubric v1 (evals/weekly_rubric.py)", "The first checklist, used for the ledger rows: hallucination, "
     "instruction following, coverage, specificity. Its label-mode variants differ slightly per approach, and it "
     "reuses the system's own guard functions, so rows are only roughly comparable."),
    ("Rubric v2, core (evals/rubric_v2.py)", "The instrument used by this report: one fixed item set for every "
     "approach, self-contained (imports nothing from the system under test), with a semantic kind. Every saved answer "
     "is re-scored with it on each build. Each semantic item was validated by reading what it flagged on the saved "
     "answers; the false positives found are pinned by tests (tests/test_rubric_v2.py)."),
    ("Blind judgments (evals/judgments.json)", "Answers compared week by week with sides shuffled and the key opened "
     "after all picks. The judge is Claude in the working session, so judgments are a second opinion, not a "
     "human evaluation; the blind page (evals/blind_compare.py) lets the user judge."),
    ("Representation measures (evals/label_metrics.py)", "Model-independent measures of what each approach shows the "
     "model: size, numeric content, statements, verdict coverage and fidelity, field resolution and capacity, "
     "discriminability, readability, redundancy, semantic elements, prompt load."),
    ("Output measures (evals/output_metrics.py)", "From the saved answers: pass rates with 95% Wilson intervals, "
     "spread across repeats, redraft rate, latency, copy rate from the labels, lexical diversity, the most repeated "
     "recommendation phrase; paired per-week Wilcoxon comparisons between approaches."),
    ("Ledger (evals/ledger.md)", "One row per evaluation run, appended automatically, plus the blind judgments: the "
     "chronological record."),
]

# ── The same principle on two more messages (2026-10-04) ─────────────────────
# Narrative for the report's section on the checkpoint comment and the morning briefing. Setups live in
# evals/run_message_eval.SETUPS; numbers come from the saved answers, re-scored with evals/message_rubric.py.

MESSAGES_INTRO = (
    "After the weekly review switched to labels, the same principle was applied to the two other messages the "
    "model writes every day. The CHECKPOINT COMMENT follows each of the day's four checks (12:00, 15:00, 19:00 and "
    "23:59 by default). It says how the block that just closed went for calories, protein and steps, what the gap "
    "leaves for the rest of the day (the carry-over), and one concrete action. The MORNING BRIEFING has two model-written "
    "parts: a reflection on yesterday, opening with a greeting, and three practical suggestions for today. Each "
    "message keeps its current prompt as the baseline (the 'numbers' design) and gets a labels design written by "
    "code (checkpoint_labels.py, briefing_labels.py). The labels design runs with and without its answer checks, so "
    "the effect of the representation can be told apart from the effect of the checks. The only figure the labels "
    "show is the temperature, kept as a number at the user's request. On 2026-10-04 the user switched production "
    "(CHECKPOINT_MODE, BRIEFING_MODE) to the winners on the holdout cases. After the review of a real message the "
    "checkpoint moved to labels v3 with checks (C6) and the briefing to labels v4 with checks (B8)."
)
MESSAGES_DATA = (
    "Thirty synthetic checkpoints and thirty synthetic mornings (20 tuning + 10 holdout each; evals/checkpoint_cases.py, "
    "evals/briefing_cases.py), built through the production code's own pure functions. The block figures come from "
    "background_checks._metric_block, the original context text from _fc_format_context, and the labels input from "
    "the production adapters. Before any of this, the original prompts were extracted from background_checks.py "
    "unchanged and pinned byte for byte by golden files captured from the code before the change "
    "(tests/golden/). So the baseline sends exactly what production sends. The checkpoints cover each "
    "phase, each check of the day, nothing logged, rain or unknown weather, disliked foods, a used-up budget, rest "
    "and cardio days and morning feedback. The mornings cover every carry-over strategy (target lowered or raised, "
    "steps raised or lowered, a pending choice), missed weigh-ins and workouts, rest and cardio days, and each "
    "weather type."
)
MESSAGE_ATTEMPTS = [
    {"id": "C0", "setup": "checkpoint_numbers", "name": "Checkpoint, original prompt", "representation": "numbers",
     "change": "Production's prompt: each metric as 'live 850 kcal vs expected 550 kcal (status 3, over target); "
               "target next block -300 kcal (raw carry -300 kcal — user is 300 kcal over daily budget)', the "
               "food list with kcal and protein per item, rules for decoding the status codes, and a fat-loss note.",
     "why": "Baseline.",
     "result": "Its carry-over line is misleading as worded: the figure it calls 'target next block' is the gap to the "
               "plan so far, and when that gap is negative the prompt says the user is over the DAILY budget, even at "
               "noon with most of the day's budget left (section 6.2). The answers repeated it ('200 kcal over your "
               "daily budget' at 15:00 with 1,700 of 3,000 kcal eaten). They also passed the status codes on to the "
               "user ('which is a status 3'). Once, in a cut that was already over plan, an answer suggested 'a rice "
               "bowl or granola bar for calories' (section 6.4)."},
    {"id": "C1", "setup": "checkpoint_labels_noguard", "name": "Checkpoint over labels, no checks",
     "representation": "labels",
     "change": "Code states what the block did against its share, where the day stands against plan, what the "
               "carry-over leaves for the rest of the day compared with a normal rest of day, what that means in "
               "this phase, what to act on first (phase-weighted), which foods from the user's list fit and why, "
               "the outdoor verdict, and a one-sentence plan for the message. No figures except the temperature.",
     "why": "The weekly review's lesson: state verdicts and meaning, and leave only the wording to the model.",
     "result": "No status codes, no false budget claims, almost no figures, and every hallucination item passed on "
               "the holdout cases. Answers were about 60% as long as C0's and took 2.0-2.3 s against 3.4-4.1 s "
               "(section 6.3). It failed in new ways. At the day's last check it repeated the weather and "
               "outdoor lines meant for daytime checks ('outdoor walking is fine' at 23:59), which made the comment "
               "too long. In one tuning case it softened 'far over plan' into 'slightly over' (section 6.4)."},
    {"id": "C2", "setup": "checkpoint_labels", "name": "Checkpoint over labels, with checks",
     "representation": "labels",
     "change": "C1 plus deterministic checks on the answer (figures, outdoor rule, scolding when nothing is logged, "
               "'still early', eat-more while over, catch-up in a cut, lists, disliked foods, a far-off gap called "
               "slight) and one redraft naming the problems.",
     "why": "Measures what the checks add on top of the representation.",
     "result": "The checks removed the softened gap and changed little else: 7% of tuning answers and no holdout "
               "answer were redrafted, at about 0.2 s. They do not catch the repetition at the last check, "
               "which is a flaw in the view itself (sections 6.3-6.5). The first version of its outdoor check "
               "fired on mentions such as 'without stepping outside'. It was fixed, and C2 was run again on both "
               "case sets (section 6.6)."},
    {"id": "B0", "setup": "briefing_numbers", "name": "Briefing, original prompts", "representation": "numbers",
     "change": "Production's two prompts. Yesterday is given as classifications ('Steps: a little below', "
               "'Calories: off — needs focus today (over)'). Today is given as figures: 'Calories today: adjusted "
               "(2010 kcal)' without the usual target, 'Carry Over: 220 kcal over yesterday', the weather in °C, mm "
               "and km/h, and a numbered list of what to write.",
     "why": "Baseline.",
     "result": "The yesterday prompt was already close to a labels design: classifications such as 'a little below' "
               "instead of figures. Its reflections were mostly right and mentioned the big misses more reliably "
               "than B1's. The today prompt was not. On a training day the answer often called it a rest day "
               "('Rest day is okay, but stay active'), in 22% of the applicable tuning answers and half of the "
               "holdout ones (section 6.4). This plausibly comes from item 3 of the prompt's numbered list, 'A brief "
               "workout or rest day note'."},
    {"id": "B1", "setup": "briefing_labels_noguard", "name": "Briefing over labels, no checks",
     "representation": "labels",
     "change": "Yesterday: each result in words with what it meant for the phase, the link to today's changed target "
               "('that is why today's calorie target is a little lower'), what went well and what to do better "
               "first. Today: targets in words against the usual ones, with a reason only when it matches the "
               "direction; the weather with the temperature as a figure; the outdoor verdict; the workout; the plan "
               "as one sentence instead of a list.",
     "why": "Same principle; and the weekly review showed that a listed plan gets copied into the answer.",
     "result": "No rest-day mistakes, and every hallucination item passed. The labels name today's session "
               "('a short note on today's Lower Day session'). Its failures came from the view's own wording. The "
               "plan says 'then the first thing to do better today', so the reflection named only that thing and "
               "dropped a second big miss (section 6.4). On a day with no workout plan, the plan's 'a short note on "
               "the day without a plan' produced a vague 'a quiet day'. Slower than B0: its answers are longer "
               "and its prompts carry more tokens (sections 6.2, 6.3)."},
    {"id": "B2", "setup": "briefing_labels", "name": "Briefing over labels, with checks", "representation": "labels",
     "change": "B1 plus checks on each part (figures, lists, a greeting in the today part, the outdoor rule) and one "
               "redraft each.",
     "why": "Measures what the checks add.",
     "result": "Half the briefings were redrafted, nearly all for a duration such as 'a 20-minute cycling session'. "
               "The labels prompt forbids it ('no numbers except the temperature'), but the product rule and the "
               "checklist allow it. The checks found no real error and added about a second. Where B2 differs from "
               "B1 it is on the yesterday section, whose prompt is identical in both and is almost never redrafted: "
               "that is run-to-run variation on a borderline behaviour, not an effect of the checks (sections 6.3, "
               "6.5)."},
    {"id": "C3", "setup": "checkpoint_labels2_noguard", "name": "Checkpoint over labels v2, no checks",
     "representation": "labels v2",
     "change": "C1's view with two changes. The day's last check holds only how the day went: no weather, outdoor, "
               "food or morning-feedback lines, and the workout as logged or not. The plan also asks to keep the "
               "size words as given ('far over stays far over').",
     "why": "C1 repeated the daytime weather lines at 23:59 and ran past the length limit, and once softened 'far "
            "over plan' into 'slightly over' (both on tuning cases).",
     "result": "Under checklist v1.4 it passed every item in all 60 tuning and all 30 holdout answers (C1: 93% and "
               "90%): the last check stopped repeating the weather and stayed within length, and no gap was softened. "
               "Checklist v1.5, written after a real message (C5), adds rules on steps, workouts, the hour and "
               "supplements, and under it C3 and C4 no longer pass everything (section 6.4)."},
    {"id": "C4", "setup": "checkpoint_labels2", "name": "Checkpoint over labels v2, with checks",
     "representation": "labels v2",
     "change": "C3 plus C2's checks, three of them changed. A request for more food is told from a description of "
               "one ('eat more' versus 'the extra calories came from the oatmeal'). The last check is flagged when "
               "it runs past four sentences or talks about the weather.",
     "why": "C2's 'eat more' check fired on descriptions of the surplus, the same false positive found in the "
            "checklist (section 6.6).",
     "result": "Under v1.4, as C3: every item passed on both case sets, with 5-10% of answers redrafted and no added "
               "time. Under the final checklist, 88% of tuning and 80% of held-out answers pass every item (90% "
               "held-out before the end-of-day rule), and only 43% of the incident cases (C2: 29%): it offered a "
               "supplement as food in 11 of 21 incident answers, named an exercise in 9 of 111 answers overall, and "
               "talked as if the day went on in 3 of 18 last-check answers (section 6.4)."},
    {"id": "B3", "setup": "briefing_labels2_noguard", "name": "Briefing over labels v2, no checks",
     "representation": "labels v2",
     "change": "B1's views with three changes. The to-do list holds only misses of 5% or more, and the plan asks to "
               "name every item on it. A day with no workout plan says so ('no workout planned today'). A suggested "
               "duration ('a twenty-minute walk') is allowed.",
     "why": "B1's plan ('the first thing to do better') dropped a second big miss, and its no-plan note became a "
            "vague 'a quiet day'. The strict number rule sent half of B2's briefings back for a duration (tuning "
            "cases).",
     "result": "Under checklist v1.4, better than B1 on the tuning cases (95% against 90%) and worse on the holdout "
               "cases (77% against 90%). "
               "Every to-do item was now named, but on held-out mornings with four or five items the reflection ran "
               "to six sentences (H05, H10). The instruction's example '(such as a twenty-minute walk)' came back as "
               "the movement idea in most answers, and twice as 'a brisk walk around the block' on a day with unknown "
               "weather (H04), where B1 had suggested an indoor circuit (section 6.4)."},
    {"id": "B4", "setup": "briefing_labels2", "name": "Briefing over labels v2, with checks",
     "representation": "labels v2",
     "change": "B3 plus checks: B2's, with durations allowed, every to-do item named (yesterday) and a plain note "
               "when no workout is planned (today), and one redraft each.",
     "why": "Measures what the checks add to v2.",
     "result": "Under v1.4, 83% on the holdout cases. The checks caught the outdoor walk but not the length: they have no length "
               "check for the yesterday section. The duration rule no longer sent briefings back (3% redrafted, "
               "against B2's 50%)."},
    {"id": "C5", "setup": "checkpoint_labels3_noguard", "name": "Checkpoint over labels v3, no checks",
     "representation": "labels v3",
     "change": "C3 plus what a real message showed was missing (the 12:00 check of 2026-10-05: 'light squats or lunges "
               "at home to add to your steps' in the rain, after a lower-body session, with 'fish oil' offered as food). "
               "Steps are walking steps and the prompt says no exercise is ever named. Training is only the plan's "
               "session. Rain, great heat (32 C and above) and strong wind (40 km/h) send the steps indoors "
               "(treadmill, stairs, pacing, a covered walk); a sunny day offers a walk or a treadmill as equals; "
               "unknown weather leans indoors. The view states the hour, the hours left, which meals are still ahead "
               "and when walking ideas stop. Supplements leave the food list, and when calories are far over plan "
               "the advice is lighter portions, not another food.",
     "why": "The user's reading of that message: steps and sets are different things, workouts must come from the "
            "blueprint, and the advice showed no sense of the time of day. None of this was in checklist v1.4.",
     "result": "All 21 incident answers pass, 97% of held-out and 96% of tuning answers (C4: 80% and 88%), under "
               "the final checklist (section 6.3). No exercise was named in any of the 131 checkpoint answers."},
    {"id": "C6", "setup": "checkpoint_labels3", "name": "Checkpoint over labels v3, with checks",
     "representation": "labels v3",
     "change": "C5 plus checks for exercises named, workouts suggested, meals and walks that no longer fit the hour, "
               "and outdoor-only ideas when the steps belong indoors; one redraft.",
     "why": "Measures what the checks add on top of the representation.",
     "result": "All held-out answers (30 of 30) and all incident answers pass; 13% of held-out answers were "
               "redrafted. The checks changed little for the checkpoint, as for C2 and C4 (section 6.3)."},
    {"id": "B5", "setup": "briefing_labels3_noguard", "name": "Briefing over labels v3, no checks",
     "representation": "labels v3",
     "change": "B1's yesterday part, unchanged (v2's version lost on held-out cases). The today part gets C5's rules: "
               "walking-only steps, no exercises, the planned session only, weather-aware indoor or outdoor ideas. "
               "A suggested duration is allowed, and the instruction has no example phrase that could be copied.",
     "why": "Under checklist v1.5, 27 of the 105 live (B1) briefings named an exercise as the step idea, 'an indoor "
            "bodyweight circuit' among them: the briefing had the same defect as the checkpoint.",
     "result": "83% of held-out answers pass (B1: 53%), 7 of 112 name an exercise (B1: 27 of 105). On unknown "
               "weather it still sent the user for a walk around the block in 3 of 3 answers (section 6.4)."},
    {"id": "B6", "setup": "briefing_labels3", "name": "Briefing over labels v3, with checks",
     "representation": "labels v3",
     "change": "B5 plus checks on the today part (v2's, with exercises, invented workouts and outdoor-only ideas when "
               "the steps belong indoors) and one redraft; the yesterday checks are B2's.",
     "why": "The checks were the difference between sending the user outside on unknown weather and not.",
     "result": "97% of held-out answers pass (B1: 53%, B5: 83%) and 11 of 15 incident answers. The misses are "
               "almost all the yesterday part leaving out a second or third large miss (14 answers), because its plan "
               "asked for 'the first thing to do better today'. One outdoor walk on the unknown-weather case "
               "survived the redraft. 10-20% of answers were redrafted (section 6.3)."},
    {"id": "B7", "setup": "briefing_labels4_noguard", "name": "Briefing over labels v4, no checks",
     "representation": "labels v4",
     "change": "B5/B6 with a new yesterday part. Code ranks the misses of 10% or more by size, biggest first (v1 ranked "
               "by phase weight, which put a 4% protein miss above a 16% calorie miss), the plan names all of them "
               "together in one sentence, asks how to fix the biggest only, and drops the 'went well' sentence when "
               "nothing did (the model had copied 'nothing stood out'). It stops after the last planned sentence. "
               "The today part is v3's.",
     "why": "B6 failed by naming only the first of several big misses (14 answers). v2's 'name every item' made "
            "four-to-five-item reflections too long; naming them together in one sentence keeps the length fixed.",
     "result": "90% of held-out, 96% of tuning and 80% of incident answers pass (B5: 83%, 78%, 60%). The omitted-miss "
               "failure disappeared. A new one appeared on days with no misses: a closing 'Keep up the great work, G.' "
               "used the name twice, which the first version of the checks did not look for (section 6.4)."},
    {"id": "B8", "setup": "briefing_labels4", "name": "Briefing over labels v4, with checks",
     "representation": "labels v4",
     "change": "B7 plus checks: every big miss named, at most five sentences, no 'nothing went well', the name used "
               "once, no closing encouragement; the today part's checks are v3's; one redraft each.",
     "why": "The first measurement of B8 showed the sign-off failure; the check and the prompt (stop after the last "
            "planned sentence) were added and the measurement repeated.",
     "result": "Every check passed in all 124 answers across the three sets (held-out 30 of 30, tuning 79 of 79, "
               "incident 15 of 15). 37% of held-out answers were redrafted, at the same average time as B6. The "
               "design was written against the failures of B6, so the tuning and incident scores are partly by "
               "construction; the held-out gain over B6 (97%) is within noise (sections 6.3, 6.5)."},
]
# What the checkpoint and briefing study taught; each point cites section 6.
MESSAGE_LESSONS = [
    "The principle carried over to both messages. Under the final checklist, the production designs (checkpoint v3, "
    "briefing v4, both with checks) passed every item in 100% of the held-out answers, against 33% (checkpoint) and "
    "37% (briefing) with the original prompts (section 6.3). The gain comes from removing specific errors (section "
    "6.4). Pooled over the 30 cases, each beats the original prompt in the per-case share of answers passing every "
    "item (checkpoint +0.46, briefing +0.44; 17 cases better and none worse for both; p < 0.001); on the 10 "
    "held-out cases alone p = 0.008 and p = 0.016 (section 6.5).",
    "A list the answer must cover is a decision for code. The yesterday part's plan said 'the first thing to do better "
    "today' and the model named exactly one thing, dropping a second or third big miss in 14 answers. Asking for 'every "
    "item' (v2) made long reflections. Version 4 has code rank the misses by size, name them together in one sentence "
    "and fix the biggest, and the omission disappeared (0 of 124 answers) at the same length (section 6.4).",
    "The end of the day needs its own wording. All 18 last-check answers of the original prompts talked as if the day "
    "went on ('on track', 'next block', 'keep going'), against 1 of 18 for labels v1, 3 of 18 for v2 and none for v3, "
    "which states that the day is over (section 6.4).",
    "Read the prompt before blaming the model. The checkpoint prompt was itself wrong as worded in 16 of 30 cases. "
    "It calls the gap to the plan so far 'target next block', and when that gap is negative it says the user is "
    "over the daily budget. The model repeated it. Stating the carry-over in words ('the rest of the day has a fair "
    "bit less food to work with than usual') removed the claim (sections 6.2, 6.4).",
    "Codes meant for the model reach the user. 'status 3, over target' was passed on to the user in 7% of tuning "
    "and 20% of holdout checkpoint answers. Labels have no codes to leak (section 6.4).",
    "A numbered instruction list is read as topics to cover. 'A brief workout or rest day note' plausibly became "
    "'Rest day is okay' on training days, in half of the holdout training days. The labels name the day's session, "
    "and that error disappeared (section 6.4).",
    "Where the original prompt already used words, labels added little. The briefing's yesterday prompt gave "
    "classifications, not figures, and its reflections were as good as or better than the labels version's. The "
    "gain came from the parts that still used figures, codes and lists: the checkpoint and the briefing's today "
    "section.",
    "Labels bring their own failure: the model repeats whatever the view contains. At the day's last check the "
    "weather and outdoor lines were repeated at 23:59. A plan saying 'the first thing to do better' produced "
    "reflections that dropped a second big miss. The view should hold only what this message needs, and the plan "
    "should say what to cover (section 6.4).",
    "Curating the view is the lever, but only against the failures the checklist knows. Version 2 of the checkpoint "
    "removed lines the last check does not need and passed every item in all 90 answers under checklist v1.4. A real "
    "message then told the user to do squats and lunges 'to add steps'. The checklist had no rule for it (recall was "
    "not measured). Checklist v1.5 added the missing rules, and under it version 2 names an exercise in 9 of 111 "
    "checkpoint answers, and the live briefing design in 27 of 105 (section 6.4).",
    "Domain concepts the model cannot know have to be stated, not inferred: steps are walking and exercises add none; "
    "training comes only from the user's plan; a sunny day is not an automatic reason to go outside; the hour decides "
    "which meals and walks still make sense. Version 3 states each in the view and in one rule, and the exercise "
    "problem disappeared from the checkpoint (0 of 131) and from the checked briefing (0 of 112) (sections 6.3, 6.4). "
    "The same review found that the day's last check said the plan was 'on track' (7 of 36 answers), as if the day "
    "went on; v3 now states that the day is over, and none of the 15 last-check answers tested afterwards did.",
    "A fix that works on the tuning cases can trade one failure for another on new ones. Asking the briefing to name "
    "every to-do item fixed the omissions it targeted, but made the reflection too long on held-out mornings with "
    "four or five items, and its example duration became the suggestion itself (section 6.4). The holdout set "
    "caught what the tuning set could not, so the briefing stays on version 1.",
    "Validate the answer checks as carefully as the rubric. The first outdoor check fired on mentions ('without "
    "stepping outside'), and its redrafts 'fixed' correct answers, which the equally naive first checklist then "
    "rewarded. That apparent gain on the outdoor item vanished once both were fixed (section 6.6). A stricter rule "
    "than the product needs ('no numbers except the temperature') redrafted half the briefings for harmless "
    "durations and cost about a second each.",
]

THREATS = [
    "Checklist v1.5 was written after a real message exposed rules the earlier checklists lacked, and labels v3 was then "
    "designed against those rules. The incident cases (21 checkpoint and 15 briefing answers per design) were written "
    "from that message, so the version 3 and 4 scores on them (100% for the checkpoint and for the briefing with checks) are partly by construction. The tuning and "
    "held-out cases predate the rules; version 3's score on them is the fairer evidence. Re-scoring with v1.5 also "
    "lowered every earlier design's score relative to what was first reported.",
    "Synthetic weeks: thirty hand-built weeks, not real users. They target known failure modes, which makes failures "
    "more frequent than in real use.",
    "The prompts and labels were tuned while reading answers to the twenty tuning weeks; tuning numbers are "
    "optimistic. The ten holdout weeks were not used for tuning, but one v4 holdout run was read before v5 was "
    "written (v5's change was also motivated by tuning-week evidence from v3).",
    "The semantic rubric encodes an explicit domain spec (fix on the matching side, large calorie gaps first, eat "
    "before moving more when under-eating, only data-supported links). Labels v3-v5 were designed with the same "
    "principles, so they are favoured on these items by construction; the blind judgments are an independent check.",
    "The rubric is regular-expression based. Each semantic item was validated by reading its flags, and false "
    "positives found were fixed and pinned by tests, but recall (failures it misses) was not measured.",
    "One judge, and it is the same assistant that built the approaches. Judgments are blind to the approach, not to "
    "the design ideas. A human judgment (the blind page) is the stronger evidence.",
    "One model (qwen3:8b Q4_K_M via Ollama, fast mode) and one task. The token counts use cl100k_base as a proxy for "
    "Qwen's tokenizer.",
    "Sampling: the model samples, so a single run varies; the main tables pool repeated runs of the identical "
    "version (same fingerprint) and show the spread across repeats.",
    "Paired tests are per week (n = 20 or 10) and several comparisons are made; treat p-values as descriptive.",
    "Checkpoint and briefing: the message checklist (evals/message_rubric.py) was written by the same assistant that "
    "wrote the label designs, from the same product rules (the original prompts' own instructions, the outdoor rule, "
    "never asking for more food while over plan). It imports nothing from the code under test, and its precision was "
    "measured by reading what it flags (section 6.6), but it is not independent of the design ideas. The labels "
    "designs were adjusted once after a two-case smoke run (a gap the model softened, a 'light' breakfast on a bigger "
    "budget), before any measured run, using tuning cases only. Their answer checks were fixed once during the "
    "measured runs, from tuning-run evidence: the outdoor rule had been reading mentions as suggestions, and curly "
    "apostrophes defeated its negations. Both checked setups were then run again on both case sets, and the earlier "
    "runs stay in evals/ledger_messages.md as an older version. The checklist went from v1 to v1.4 by the same "
    "read-and-fix process, and every saved answer is re-scored with the current version.",
    "Checkpoint and briefing: 30 cases per message and one model. Several failures are tied to one or two cases "
    "('status 3' in four cases, 'rest day' in nine), so case selection moves the headline rates.",
    "Labels v2 (C3, C4, B3, B4): each of its changes answers a failure seen on tuning cases. The same failures had "
    "also been seen on holdout cases while the checklist was being validated (the last-check repetition in H09, "
    "the vague no-plan note in H10). The v2 holdout result is therefore less clean than v1's, whose holdout cases "
    "were unseen when it was built.",
]
