# Eval report

- Run: `2026-07-24T15:27:01+00:00`
- Items: 26  ·  k=5  ·  filters=True  ·  judged=True
- Tokens: 61947 prompt + 4068 completion

| Bucket | n | Faithful | Correct | Ctx recall | Hit@k | MRR |
|---|---|---|---|---|---|---|
| **overall** | 26 | 0.89 | 0.73 | 92% | 92% | 0.92 |
| single_fact | 12 | 0.90 | 0.83 | 100% | 100% | 1.00 |
| synthesis | 8 | 0.90 | 0.62 | 100% | 100% | 1.00 |
| temporal | 6 | 0.87 | 0.67 | 67% | 67% | 0.67 |

_Faithful/Correct are judge scores in [0,1]; Ctx recall / Hit@k are retrieval._
