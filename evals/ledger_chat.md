# Chat agent: approach ledger

One row per evaluation run (evals/run_chat_eval.py). A case passes when no applicable checklist item fails.

| when | setup | fingerprint | set | cases | pass | file |
|---|---|---|---|---|---|---|
| 2026-10-05 12:12 | numbers | 37f1453ea7 | tuning | 20 | 1/20 | chat_20261005_120857.json |
| 2026-10-05 12:12 | words | 9285287779 | tuning | 20 | 13/20 | chat_20261005_120857.json |
| 2026-10-05 12:19 | numbers | 37f1453ea7 | tuning | 20 | 5/20 | chat_20261005_121442.json |
| 2026-10-05 12:19 | words | 1bd586b49d | tuning | 20 | 16/20 | chat_20261005_121442.json |
| 2026-10-05 12:24 | numbers | 37f1453ea7 | tuning | 20 | 3/20 | chat_20261005_122045.json |
| 2026-10-05 12:24 | words | fb787bb173 | tuning | 20 | 19/20 | chat_20261005_122045.json |
| 2026-10-05 12:31 | numbers | 37f1453ea7 | holdout | 30 | 3/30 | chat_20261005_122503.json |
| 2026-10-05 12:31 | words_noguard | fb787bb173 | holdout | 30 | 24/30 | chat_20261005_122503.json |
| 2026-10-05 12:31 | words | fb787bb173 | holdout | 30 | 24/30 | chat_20261005_122503.json |
| 2026-10-05 12:58 | numbers | 37f1453ea7 | fresh | 48 | 10/48 | chat_20261005_124658.json |
| 2026-10-05 12:58 | words_noguard | 30247cb884 | fresh | 48 | 42/48 | chat_20261005_124658.json |
| 2026-10-05 12:58 | words | 30247cb884 | fresh | 48 | 44/48 | chat_20261005_124658.json |
| 2026-10-05 13:37 | numbers | 37f1453ea7 | fresh2 | 60 | 9/60 | chat_20261005_131757.json |
| 2026-10-05 13:37 | words_noguard | 3a38dd58fe | fresh2 | 60 | 54/60 | chat_20261005_131757.json |
| 2026-10-05 13:37 | words | 3a38dd58fe | fresh2 | 60 | 54/60 | chat_20261005_131757.json |
| 2026-10-05 23:26 | words_noguard | 40fc0f506b | fresh3 | 48 | 36/48 | chat_20261005_232431.json |
| 2026-10-05 23:26 | words | 40fc0f506b | fresh3 | 48 | 36/48 | chat_20261005_232431.json |
| 2026-10-08 03:16 | words | 46aeda7892 | tuning | 60 | 57/60 | chat_20261008_031514.json |
| 2026-10-08 03:17 | words | 46aeda7892 | holdout | 30 | 29/30 | chat_20261008_031634.json |
| 2026-10-08 03:18 | words | 46aeda7892 | fresh | 48 | 45/48 | chat_20261008_031726.json |
| 2026-10-08 03:20 | words | 46aeda7892 | fresh2 | 60 | 60/60 | chat_20261008_031835.json |
| 2026-10-08 03:21 | words | 46aeda7892 | fresh3 | 48 | 48/48 | chat_20261008_032017.json |
| 2026-10-08 03:22 | words | 46aeda7892 | fresh4 | 33 | 33/33 | chat_20261008_032127.json |
| 2026-10-08 04:22 | words | 6054602b03 | tuning | 60 | 60/60 | chat_20261008_042116.json |
| 2026-10-08 04:23 | words | 6054602b03 | holdout | 30 | 30/30 | chat_20261008_042245.json |
| 2026-10-08 04:24 | words | 6054602b03 | fresh | 48 | 46/48 | chat_20261008_042331.json |
| 2026-10-08 04:26 | words | 6054602b03 | fresh2 | 60 | 57/60 | chat_20261008_042447.json |
| 2026-10-08 04:27 | words | 6054602b03 | fresh3 | 48 | 48/48 | chat_20261008_042618.json |
| 2026-10-08 04:28 | words | 6054602b03 | fresh4 | 33 | 33/33 | chat_20261008_042728.json |
| 2026-10-08 05:39 | words | d543400212 | tuning | 60 | 57/60 | chat_20261008_053745.json |
| 2026-10-08 05:39 | words | d543400212 | holdout | 30 | 28/30 | chat_20261008_053924.json |
| 2026-10-08 05:41 | words | d543400212 | fresh | 48 | 45/48 | chat_20261008_054009.json |
| 2026-10-08 05:42 | words | d543400212 | fresh2 | 60 | 58/60 | chat_20261008_054124.json |
| 2026-10-08 05:43 | words | d543400212 | fresh3 | 48 | 45/48 | chat_20261008_054250.json |
| 2026-10-08 05:44 | words | d543400212 | fresh4 | 33 | 30/33 | chat_20261008_054400.json |
| 2026-10-08 05:45 | words | d543400212 | fresh5 | 18 | 15/18 | chat_20261008_054458.json |
| 2026-10-08 05:57 | words | 40844c671d | tuning | 60 | 56/60 | chat_20261008_055625.json |
| 2026-10-08 05:58 | words | 40844c671d | holdout | 30 | 28/30 | chat_20261008_055758.json |
| 2026-10-08 05:59 | words | 40844c671d | fresh | 48 | 48/48 | chat_20261008_055845.json |
| 2026-10-08 06:01 | words | 40844c671d | fresh2 | 60 | 57/60 | chat_20261008_060000.json |
| 2026-10-08 06:02 | words | 40844c671d | fresh3 | 48 | 48/48 | chat_20261008_060134.json |
| 2026-10-08 06:03 | words | 40844c671d | fresh4 | 33 | 31/33 | chat_20261008_060244.json |
| 2026-10-08 06:04 | words | 40844c671d | fresh5 | 18 | 12/18 | chat_20261008_060343.json |
| 2026-10-08 06:06 | words | de84ec6009 | fresh5 | 18 | 15/18 | chat_20261008_060542.json |
| 2026-10-08 06:19 | words | 7f5c8567b3 | tuning | 60 | 53/60 | chat_20261008_061752.json |
| 2026-10-08 06:19 | words | 7f5c8567b3 | holdout | 30 | 29/30 | chat_20261008_061923.json |
| 2026-10-08 06:21 | words | 7f5c8567b3 | fresh | 48 | 45/48 | chat_20261008_062010.json |
| 2026-10-08 06:22 | words | 7f5c8567b3 | fresh2 | 60 | 57/60 | chat_20261008_062118.json |
| 2026-10-08 06:23 | words | 7f5c8567b3 | fresh3 | 48 | 48/48 | chat_20261008_062241.json |
| 2026-10-08 06:24 | words | 7f5c8567b3 | fresh4 | 33 | 32/33 | chat_20261008_062348.json |
| 2026-10-08 06:25 | words | 7f5c8567b3 | fresh5 | 18 | 15/18 | chat_20261008_062445.json |
