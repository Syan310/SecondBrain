## LC0 vs R0

### answerable (n=50)

| Metric | R0 mean (min–max) | LC0 mean (min–max) | Diff | 95% CI | Better / Same / Worse | Verdict |
|---|---|---|---|---|---|---|
| correctness_score | 0.887 (0.880–0.900) | 1.000 (1.000–1.000) | +0.113 | [+0.040, +0.207] | 7 / 43 / 0 | improved |

### original_20 (n=20)

| Metric | R0 mean (min–max) | LC0 mean (min–max) | Diff | 95% CI | Better / Same / Worse | Verdict |
|---|---|---|---|---|---|---|
| correctness_score | 0.983 (0.975–1.000) | 1.000 (1.000–1.000) | +0.017 | [+0.000, +0.050] | 1 / 19 / 0 | flat |

### By question type (Better / Same / Worse, no CI)

| Metric | term | paraphrase | heading |
|---|---|---|---|
| recall_at_k | — | — | — |
| rr | — | — | — |
| rr_raw | — | — | — |
| rr_reranked | — | — | — |
| reranker_lift | — | — | — |
| faithfulness | — | — | — |
| answer_relevancy | — | — | — |
| context_precision | — | — | — |
| context_recall | — | — | — |
| correctness_score | 0 / 10 / 0 | 6 / 4 / 0 | 0 / 5 / 0 |

### Refusal, latency, tokens and cost (no CI)

| | R0 | LC0 |
|---|---|---|
| refusal_accuracy | 1.000 (1.000–1.000) [10/10, 10/10, 10/10] | 1.000 (1.000–1.000) [10/10, 10/10, 10/10] |
| false_refusal_rate | 0.020 (0.020–0.020) [1/50, 1/50, 1/50] | 0.000 (0.000–0.000) [0/50, 0/50, 0/50] |
| fully_refused (per run) | 0, 0, 0 | 0, 0, 0 |
| latency_p50_ms | 1380 (1343–1454) | 1536 (1481–1636) |
| latency_p95_ms | 1984 (1932–2022) | 2054 (2002–2109) |
| input_tokens (per question) | 932 (932–932) | 20346 (20346–20346) |
| output_tokens (per question) | 104 (103–104) | 92 (91–95) |
| cost_usd_actual (per question) | 0.000124 (0.000093–0.000186) | 0.000303 (0.000300–0.000304) |
| cost_usd_no_cache (per question) | 0.000269 (0.000202–0.000404) | 0.006215 (0.006213–0.006218) |

