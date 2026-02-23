# Experiment 020 — Cursor-Only Learning

**Date:** 2026-02-23
**Duration:** 1.5s

## Hypothesis

Global reward works if you only change a few synapses at a time. A cursor restricts perturbation to a local spatial neighbourhood. One scalar reward is enough to evaluate a small local change — like a genetic algorithm running inside a network.

## Configuration

| Parameter | Value |
|---|---|
| Clusters | 2 |
| Neurons per cluster | 30 |
| Total neurons | 70 |
| Total synapses | 2292 |
| Intra-cluster connectivity | 60% |
| Inter-cluster connectivity | 50% |
| Cursor radius | 4 |
| Perturbation std | 0.05 |
| Episodes | 500 |
| Steps per episode | 50 |
| Target fire rate | 0.2 |
| Threshold adjust rate | 0.01 |

## Results

| Metric | Value |
|---|---|
| Initial avg reward | 0.476 |
| Final avg reward | 0.584 |
| Best avg reward | 0.644 |
| Perturbation accept rate | 99.7% |
| Avg eligible synapses/step | 279.9 |

## Episode Log (sample)

| Episode | Avg Reward | Accepted | Rejected | Avg Eligible |
|---|---|---|---|---|
| 0 | 0.476 | 44 | 0 | 222.8 |
| 36 | 0.532 | 50 | 0 | 293.1 |
| 71 | 0.520 | 45 | 0 | 165.3 |
| 107 | 0.560 | 47 | 0 | 312.4 |
| 143 | 0.508 | 49 | 1 | 315.4 |
| 178 | 0.528 | 49 | 0 | 269.7 |
| 214 | 0.528 | 40 | 0 | 92.3 |
| 250 | 0.480 | 49 | 1 | 370.4 |
| 285 | 0.468 | 50 | 0 | 351.8 |
| 321 | 0.440 | 49 | 0 | 346.5 |
| 356 | 0.584 | 46 | 0 | 212.2 |
| 392 | 0.596 | 49 | 0 | 233.2 |
| 428 | 0.448 | 49 | 0 | 235.6 |
| 463 | 0.524 | 50 | 0 | 341.4 |
| 499 | 0.584 | 41 | 0 | 348.7 |

## Conclusion

Reward showed modest improvement (0.476 → 0.584). The cursor mechanism shows partial signal but did not reach strong performance. Further tuning of radius, perturbation scale, or task complexity may help.
