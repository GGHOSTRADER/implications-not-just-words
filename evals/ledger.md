# Weekly review: approach ledger

One row per evaluation run, newest last. Rows are comparable only within the same case set. `tuning` = the 20 weeks prompts were tuned on (optimistic); `holdout` = 10 untouched weeks. The labels approaches are scored with a slightly different checklist (no digits instead of restating averages). Rows 2026-10-04 00:20 to 03:00 were backfilled from the saved result files, using the scores stored at the time (the checker changed between them, so early rows are stricter or looser in places).

| date | approach | cases | answers | every item passed | hallucination | instruction | coverage | specificity | avg s | code | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-04 00:20 | split_fast | tuning | 20 | 7/20 | 41/41 | 159/160 | 29/40 | 34/40 | 10.1 | backfilled | first run; prompts before tuning; old_think = the design before 2026-10-04 |
| 2026-10-04 00:20 | old_think | tuning | 20 | 0/20 | 37/41 | 140/160 | 28/40 | 20/40 | 29.7 | backfilled | first run; prompts before tuning; old_think = the design before 2026-10-04 |
| 2026-10-04 00:34 | split_fast | tuning | 20 | 0/20 | 17/41 | 160/160 | 40/40 | 40/40 | 8.8 | backfilled | tuning iteration (checker/prompt changes) |
| 2026-10-04 00:37 | split_fast | tuning | 20 | 9/20 | 29/41 | 160/160 | 40/40 | 39/40 | 8.0 | backfilled | tuning iteration |
| 2026-10-04 00:44 | split_fast | tuning | 20 | 14/20 | 36/41 | 159/160 | 40/40 | 38/40 | 7.7 | backfilled | after proposals-allowed + recs-only directional checks |
| 2026-10-04 00:48 | split_fast | tuning | 20 | 15/20 | 38/41 | 159/160 | 40/40 | 39/40 | 7.2 | backfilled | after widening the proposal pattern (30g, 500 steps) |
| 2026-10-04 02:55 | labels_fast | tuning | 20 | 18/20 | 60/61 | 160/160 | 40/40 | 39/40 | 4.6 | backfilled | labels v1, first run |
| 2026-10-04 02:58 | labels_fast | tuning | 20 | 17/20 | 60/61 | 160/160 | 40/40 | 38/40 | 4.7 | backfilled | labels v1 after clock-time + 'target reached' wording fixes |
| 2026-10-04 03:00 | labels_fast | tuning | 20 | 18/20 | 59/61 | 160/160 | 40/40 | 39/40 | 4.7 | backfilled | labels v1 after negation fix (code at commit 1b4af03 + uncommitted label work) |
| 2026-10-04 03:13 | labels2_fast | tuning | 20 | 6/20 | 60/61 | 154/160 | 41/41 | 29/40 | 5.4 | 1b4af03+uncommitted | results file weekly_20261004_0311.json |
| 2026-10-04 03:22 | split_fast | holdout | 10 | 6/10 | 18/20 | 80/80 | 20/20 | 18/20 | 8.9 | 1b4af03+uncommitted | results file weekly_20261004_0313.json |
| 2026-10-04 03:22 | labels_fast | holdout | 10 | 8/10 | 30/30 | 78/80 | 20/20 | 20/20 | 4.4 | 1b4af03+uncommitted | results file weekly_20261004_0313.json |
| 2026-10-04 03:22 | labels2_fast | holdout | 10 | 2/10 | 29/30 | 76/80 | 21/21 | 13/20 | 5.1 | 1b4af03+uncommitted | results file weekly_20261004_0313.json |
| 2026-10-04 03:22 | old_think | holdout | 10 | 0/10 | 16/20 | 70/80 | 14/20 | 10/20 | 32.0 | 1b4af03+uncommitted | results file weekly_20261004_0313.json |
| 2026-10-04 03:24 | labels3_fast | tuning | 20 | 13/20 | 61/61 | 155/160 | 61/61 | 37/40 | 5.4 | 1b4af03+uncommitted | results file weekly_20261004_0322.json |
| 2026-10-04 03:25 | labels3_fast | holdout | 10 | 6/10 | 30/30 | 78/80 | 29/31 | 19/20 | 6.7 | 1b4af03+uncommitted | results file weekly_20261004_0324.json |

## Blind judgments (substance: right priority, right kind of fix, no invented links, no wrong facts)

The checklist above scores format and hallucination; it cannot tell whether advice is right for the goal. These rows
are blind A/B or A/B/C picks of the better answer per week (sides shuffled, key opened after picking).

| date | judge | cases | compared | result | note |
|---|---|---|---|---|---|
| 2026-10-04 01:10 | Claude (in-session) | tuning 20 | split_fast vs old_think | split_fast 12, old_think 8 | split_fast misread calorie direction (wk07, wk19), parroted the prompt example |
| 2026-10-04 03:05 | Claude (in-session) | tuning 20 | labels_fast vs split_fast vs old_think | labels v1 15, split_fast 3, old_think 2 | labels fixed the calorie direction and the parroting |
| 2026-10-04 03:30 | Claude (in-session) | holdout 10 | labels3_fast vs labels_fast | labels v3 10, labels v1 0 | v3 picked the right priority and kind of fix every time (eat more not walk more in an over-cut, food fix not walking for calories over); it lost checklist points for 'should'/'focus on' in observations and for not pushing steps in H04, which v3 deliberately postpones |
| 2026-10-04 04:57 | split_fast | all | 90 | 63/90 | 164/183 | 714/720 | 180/180 | 175/180 | 7.6 | 1b4af03+uncommitted | results file weekly_20261004_040303.json; 3 repeats; fingerprint c5f7a93067b3 |
| 2026-10-04 04:57 | labels_fast | all | 90 | 69/90 | 267/273 | 716/720 | 180/180 | 168/180 | 5.6 | 1b4af03+uncommitted | results file weekly_20261004_040303.json; 3 repeats; fingerprint a6c3d529c57d |
| 2026-10-04 04:57 | labels2_fast | all | 90 | 33/90 | 267/273 | 701/720 | 186/186 | 129/180 | 6.1 | 1b4af03+uncommitted | results file weekly_20261004_040303.json; 3 repeats; fingerprint 476bc75daefe |
| 2026-10-04 04:57 | labels3_fast | all | 90 | 56/90 | 268/273 | 703/720 | 270/276 | 163/180 | 6.0 | 1b4af03+uncommitted | results file weekly_20261004_040303.json; 3 repeats; fingerprint 51bfc900e7a4 |
| 2026-10-04 04:57 | labels4_fast | all | 90 | 72/90 | 272/273 | 720/720 | 273/276 | 165/180 | 5.4 | 1b4af03+uncommitted | results file weekly_20261004_040303.json; 3 repeats; fingerprint 6088c69e42d5 |
| 2026-10-04 04:57 | labels5_fast | all | 90 | 10/90 | 267/273 | 641/720 | 204/276 | 101/180 | 5.9 | 1b4af03+uncommitted | results file weekly_20261004_040303.json; 3 repeats; fingerprint f570909804d9 |
| 2026-10-04 05:05 | labels6_fast | all | 90 | 84/90 | 271/273 | 720/720 | 273/276 | 179/180 | 4.7 | 1b4af03+uncommitted | results file weekly_20261004_045821.json; 3 repeats; fingerprint 7824a3178100; RESCORED after the run: rubric v1 did not know mode labels6 and scored it as 'numbers' (0/90) |
| 2026-10-04 05:58 | old_think | all | 90 | 3/90 | 162/183 | 634/720 | 127/180 | 96/180 | 35.5 | 1b4af03+uncommitted | results file weekly_20261004_050531.json; 3 repeats; fingerprint 3ba9334cde45 |
| 2026-10-04 12:04 | hybrid_fast | tuning | 60 | 36/60 | 108/123 | 473/480 | 120/120 | 114/120 | 7.4 | 1b4af03+uncommitted | results file weekly_20261004_115653.json; 3 repeats; fingerprint 146ea2414357 |
| 2026-10-04 12:08 | hybrid_fast | holdout | 30 | 14/30 | 49/60 | 235/240 | 60/60 | 57/60 | 7.8 | 1b4af03+uncommitted | results file weekly_20261004_120426.json; 3 repeats; fingerprint 146ea2414357 |
| 2026-10-12 10:00 | matched6_fast | all | 90 | 77/90 | 167/183 | 720/720 | 274/276 | 180/180 | 5.8 | ce0b843+uncommitted | results file weekly_20261012_095143.json; 3 repeats; fingerprint 38afc649418c |
