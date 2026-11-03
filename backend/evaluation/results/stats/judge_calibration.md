Sonnet judge: `anthropic/claude-sonnet-5.5`; Haiku judge: `anthropic/claude-haiku-5.5`

| 指标 | 题数 | Spearman | 平均差值（Sonnet − Haiku） | Sonnet 均值 | Haiku 均值 | 相关系数 < 0.5 |
|---|---|---|---|---|---|---|
| faithfulness | 50 | 0.771 | -0.012 | 0.919 | 0.931 | 否 |
| answer_relevancy | 50 | 0.857 | 0.011 | 0.803 | 0.792 | 否 |
| context_precision | 50 | 0.933 | -0.008 | 0.836 | 0.844 | 否 |
| context_recall | 50 | 0.922 | 0.002 | 0.900 | 0.898 | 否 |

拒答标签一致率：多数标签 58/60（0.967），逐次平均 0.961
- q034: Sonnet=refusal, Haiku=['partial', 'partial', 'partial']
- q037: Sonnet=answer, Haiku=['partial', 'partial', 'partial']
