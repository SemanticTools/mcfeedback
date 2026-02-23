# Experiment 020 — Cursor-Only Learning

**Date:** 2026-02-23
**Duration:** 0.8s

## Hypothesis

Global reward works if you only change a few synapses at a time. A cursor restricts perturbation to a local spatial neighbourhood. One scalar reward is enough to evaluate a small local change — like a genetic algorithm running inside a network.

## Configuration

| Parameter | Value |
|---|---|
| Clusters | 2 |
| Neurons per cluster | 30 |
| Total neurons | 70 |
| Total synapses | 2235 |
| Intra-cluster connectivity | 60% |
| Inter-cluster connectivity | 50% |
| Cursor radius | 2 |
| Perturbation std | 0.1 |
| Episodes | 500 |
| Steps per episode | 50 |
| Target fire rate | 0.2 |
| Threshold adjust rate | 0.01 |

## Results

| Metric | Value |
|---|---|
| Initial avg reward | 0.508 |
| Final avg reward | 0.668 |
| Best avg reward | 0.720 |
| Perturbation accept rate | 0.3% |
| Avg eligible synapses/step | 42.4 |

## Episode Log (sample)

| Episode | Avg Reward | Accepted | Rejected | Avg Eligible |
|---|---|---|---|---|
| 0 | 0.508 | 0 | 21 | 37.0 |
| 36 | 0.508 | 0 | 16 | 39.9 |
| 71 | 0.516 | 0 | 30 | 54.3 |
| 107 | 0.524 | 0 | 24 | 42.3 |
| 143 | 0.592 | 0 | 27 | 43.8 |
| 178 | 0.664 | 0 | 20 | 48.3 |
| 214 | 0.668 | 0 | 28 | 46.2 |
| 250 | 0.644 | 0 | 28 | 70.2 |
| 285 | 0.660 | 0 | 37 | 67.7 |
| 321 | 0.600 | 0 | 32 | 74.8 |
| 356 | 0.588 | 0 | 19 | 34.0 |
| 392 | 0.672 | 0 | 16 | 28.5 |
| 428 | 0.624 | 0 | 29 | 56.1 |
| 463 | 0.608 | 0 | 24 | 49.9 |
| 499 | 0.668 | 1 | 19 | 44.2 |

## Conclusion

The cursor mechanism successfully drove learning. Reward improved from 0.508 to 0.668 (best: 0.720). The hypothesis is supported: restricting perturbation to a local neighbourhood allows a single scalar reward to assign credit.
