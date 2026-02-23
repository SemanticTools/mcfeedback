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
| Total synapses | 2292 |
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
| Initial avg reward | 0.468 |
| Final avg reward | 0.644 |
| Best avg reward | 0.808 |
| Perturbation accept rate | 3.1% |
| Avg eligible synapses/step | 43.0 |

## Episode Log (sample)

| Episode | Avg Reward | Accepted | Rejected | Avg Eligible |
|---|---|---|---|---|
| 0 | 0.468 | 0 | 23 | 37.0 |
| 36 | 0.612 | 2 | 21 | 36.8 |
| 71 | 0.608 | 0 | 24 | 47.5 |
| 107 | 0.660 | 0 | 30 | 48.0 |
| 143 | 0.620 | 0 | 26 | 38.9 |
| 178 | 0.616 | 7 | 22 | 48.0 |
| 214 | 0.652 | 0 | 24 | 41.4 |
| 250 | 0.712 | 2 | 18 | 30.1 |
| 285 | 0.668 | 1 | 25 | 41.4 |
| 321 | 0.632 | 0 | 31 | 66.2 |
| 356 | 0.684 | 0 | 10 | 13.6 |
| 392 | 0.648 | 0 | 14 | 19.6 |
| 428 | 0.808 | 0 | 12 | 14.2 |
| 463 | 0.672 | 0 | 22 | 34.4 |
| 499 | 0.644 | 0 | 15 | 21.7 |

## Conclusion

The cursor mechanism successfully drove learning. Reward improved from 0.468 to 0.644 (best: 0.808). The hypothesis is supported: restricting perturbation to a local neighbourhood allows a single scalar reward to assign credit.
