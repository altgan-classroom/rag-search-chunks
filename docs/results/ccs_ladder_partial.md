| rung | recall@5 (Δ vs previous, 95% CI) | mrr (Δ vs previous, 95% CI) | ndcg@10 (Δ vs previous, 95% CI) | answer_in_context (Δ vs previous, 95% CI) | ctx tokens | p50 ms |
|---|---|---|---|---|---:|---:|
| 0-baseline | 0.510 | 0.396 | 0.434 | 0.837 | 4430 | 167 |
| 1+structure-chunks | 0.786 ▲ +0.276 [+0.143,+0.408] | 0.674 ▲ +0.278 [+0.176,+0.384] | 0.718 ▲ +0.284 [+0.187,+0.389] | 0.857 ≈ +0.020 [-0.082,+0.122] | 2299 | 483 |
| 2+parent-child | 0.786 ≈ +0.000 [-0.061,+0.061] | 0.650 ≈ -0.024 [-0.065,+0.009] | 0.701 ≈ -0.018 [-0.053,+0.014] | 0.980 ▲ +0.122 [+0.041,+0.224] | 7701 | 444 |

▲ better / ▼ worse: paired-bootstrap 95% CI of the per-question difference excludes 0. ≈: CI includes 0, i.e. **no measurable effect** at this n.
