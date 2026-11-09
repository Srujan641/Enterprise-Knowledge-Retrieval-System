# RAG benchmark results

Model: `openai/gpt-5-nano`. Scored automatically against known answers, in one run, with no tuning on these questions.

**Overall accuracy: 75.6%** (189/250), 95% CI 69.9% to 80.5%

| Question type | Questions | Correct | Accuracy | 95% CI |
|---|---|---|---|---|
| Single fact (SQuAD v2) | 100 | 86 | 86.0% | 77.9% to 91.5% |
| PDF tables (generated) | 50 | 50 | 100.0% | 92.9% to 100.0% |
| Multi-hop (HotpotQA) | 50 | 39 | 78.0% | 64.8% to 87.2% |
| No answer exists (SQuAD v2) | 50 | 14 | 28.0% | 17.5% to 41.7% |

| Other metric | Result |
|---|---|
| Search hit rate @1 | 84.5% |
| Search hit rate @5 | 97.0% |
| Both supporting documents in top 5 (multi-hop) | 76.0% |
| MRR | 0.900 |
| Search latency, p50 | 11 ms |
| Correct refusal when no answer exists | 28.0% |
| False refusal when an answer exists | 3.0% |
| Answers citing a correct source | 95.0% |
| Citation precision | 90.2% |
| Answer time, p50 / p95 (4 at once) | 3.5s / 7.2s |
| Total cost / per question | $0.0433 / $0.00017 |
| Restricted chunks leaked to a general user | 0 in 50 attempts |
| Restricted documents still found by HR | 50/50 |
| Requests that errored | 0 |

Every question, expected answer and system answer is in `results.csv`.
