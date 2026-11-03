## LC1 vs R1

### answerable (n=50)

| Metric | R1 mean (min–max) | LC1 mean (min–max) | Diff | 95% CI | Better / Same / Worse | Verdict |
|---|---|---|---|---|---|---|
| correctness_score | 0.863 (0.850–0.870) | 0.967 (0.960–0.970) | +0.103 | [-0.007, +0.213] | 7 / 41 / 2 | flat |

### original_20 (n=20)

| Metric | R1 mean (min–max) | LC1 mean (min–max) | Diff | 95% CI | Better / Same / Worse | Verdict |
|---|---|---|---|---|---|---|
| correctness_score | 0.992 (0.975–1.000) | 0.992 (0.975–1.000) | +0.000 | [+0.000, +0.000] | 0 / 20 / 0 | flat |

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
| correctness_score | 0 / 9 / 1 | 7 / 2 / 1 | 0 / 5 / 0 |

### Refusal, latency, tokens and cost (no CI)

| | R1 | LC1 |
|---|---|---|
| refusal_accuracy | 1.000 (1.000–1.000) [5/5, 5/5, 5/5] | 1.000 (1.000–1.000) [5/5, 5/5, 5/5] |
| false_refusal_rate | 0.087 (0.080–0.100) [4/50, 4/50, 5/50] | 0.000 (0.000–0.000) [0/50, 0/50, 0/50] |
| fully_refused (per run) | 0, 0, 0 | 0, 0, 0 |
| latency_p50_ms | 1484 (1327–1710) | 2858 (2698–2999) |
| latency_p95_ms | 2159 (1885–2518) | 3772 (3407–4274) |
| input_tokens (per question) | 1207 (1207–1207) | 351587 (351587–351587) |
| output_tokens (per question) | 107 (103–109) | 91 (86–93) |
| cost_usd_actual (per question) | 0.000271 (0.000193–0.000427) | 0.002285 (0.002280–0.002288) |
| cost_usd_no_cache (per question) | 0.000490 (0.000485–0.000494) | 0.105585 (0.105579–0.105588) |

