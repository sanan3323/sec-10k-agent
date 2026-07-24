# Eval report

- Run: `2026-07-24T16:05:45+00:00`
- Items: 26  ·  k=5  ·  filters=True  ·  judged=True
- Tokens: 63238 prompt + 3530 completion

| Bucket | n | Faithful | Correct | Ctx recall | Hit@k | MRR |
|---|---|---|---|---|---|---|
| **overall** | 26 | 0.94 | 0.62 | 92% | 92% | 0.92 |
| single_fact | 12 | 1.00 | 0.82 | 100% | 100% | 1.00 |
| synthesis | 8 | 0.90 | 0.53 | 100% | 100% | 1.00 |
| temporal | 6 | 0.87 | 0.37 | 67% | 67% | 0.67 |

_Faithful/Correct are judge scores in [0,1]; Ctx recall / Hit@k are retrieval._
