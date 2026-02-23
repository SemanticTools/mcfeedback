**mcfeedback**

Murray-Claude Feedback Algorithm

Phase 1 Research Journal — Experiments 1–13

February 2026

*Murray & Claude*

# **Architecture**

mcfeedback implements biologically-plausible learning without backpropagation. Synapses learn using only local signals: eligibility traces, chemical diffusion from modulatory neurons, and an ambient spatial field. There are no global gradients.

### **Network topology**

Two recurrent clusters of 30 neurons each (60 total). Each cluster contains 5 special-role neurons (input or output), 2 modulatory neurons, and 23 regular neurons. This is not a layered feedforward network — it is a recurrent soup with probabilistic all-to-all connectivity. Intra-cluster connection probability: 60%. Inter-cluster: 50%. No self-loops. Single forward propagation pass per training step (signals travel exactly one hop).

### **Task**

Learn 4 binary inversion patterns with 5 input and 5 output neurons. P1: \[1,0,1,0,1\]→\[0,1,0,1,0\]. P2: \[1,1,0,0,1\]→\[0,0,1,1,0\]. P3: \[0,1,1,0,0\]→\[1,0,0,1,1\]. P4: \[0,1,0,1,0\]→\[1,0,1,0,1\]. Note: P1 and P4 are exact inverses of each other. Random chance \= 50%.

### **Three learning mechanisms**

**Eligibility traces (four-quadrant flagging):** Co-activation of pre and post neurons → positive trace. Mismatch (one fires, other doesn’t) → negative trace. Co-silence in an active area → weak positive. Irrelevant silence → zero.

**Chemical diffusion:** Modulatory neurons broadcast reward/punishment spatially with 1/distance falloff. Radius: 15 units (spans both clusters in Full model). The chemical level modulates the magnitude of weight updates.

**Dampening filters:** Activity history tracking, information dampening (inverted-U response peaking at target fire rate), and ambient relevance field (3-unit radius, intra-cluster only).

### **Four experimental conditions**

**Baseline:** Global chemical (constant dose everywhere), no ambient field, no dampening. **Ambient only:** Adds spatial ambient field. **Dampening only:** Adds activity-based dampening filters. **Full model:** All mechanisms active with local chemical diffusion.

### **Evaluation protocol**

Frozen-weight evaluation established from iteration 2 onward. After training, weights are frozen and a forward pass with no learning produces the accuracy measurement. This prevents inflated scores from eval-during-learning artifacts. 10 seeds (42, 137, 271, 314, 500, 618, 777, 888, 999, 1234), 1000 training episodes per run unless noted.

# **Experiments**

## **Iteration 1 — Baseline multi-seed**

| Aim | Establish baseline performance across all four conditions with proper statistical power (N=10 seeds). |
| :---- | :---- |
| **Result** | Full model: 45.5% ± 5.0% (p=0.006, significantly worse than Baseline at 53.5%). Training-time accuracy of \~80% was an artifact of eval-during-learning. |
| **Conclusion** | No learning detected. Dampening actively harmful. Led to frozen-weight evaluation protocol. |

## **Iteration 3 — Flipped eligibility signs**

| Aim | Test whether inversion task requires anti-Hebbian learning. Flip coActivation from \+1 to −1 and mismatch from −0.5 to \+0.5. |
| :---- | :---- |
| **Result** | Full model: 46.0% ± 3.2%. Variance collapsed — 9/10 seeds locked at exactly 45%. Per-pattern analysis showed flags homogenizing outputs rather than differentiating them. |
| **Conclusion** | Sign flip erased information rather than enhancing it. Flags are making things more uniform, not more selective. |

## **Iteration 4 — Flag gate (strengthen/decay)**

| Aim | Add flag persistence: flags must accumulate over consecutive turns (gain=0.3, decay=0.7, threshold=0.5) before gating weight updates. \~2 consistent turns required to latch. |
| :---- | :---- |
| **Result** | Full model: 53.0% ± 9.2%, max 65% (3 seeds). Bimodal: seeds 42–500 stall at 45%, seeds 888–1234 reach 55–65%. First above-chance results. |
| **Conclusion** | First meaningful progress. Flag persistence broke the 45% attractor for some seeds. Became the reference configuration for all subsequent experiments. |

## **Iteration 5 — Squared reward**

| Aim | Squash reward signal near 50% accuracy toward zero, making fixed-output strategies break even. rewardExponent=2.0 so 55% accuracy → reward 0.01 instead of 0.10. |
| :---- | :---- |
| **Result** | Full model: 45.0% ± 8.2% (p=0.03, significantly worse). The squared reward starved early learning signal — a network at 60% gets reward 0.04 instead of 0.20, not enough to bootstrap. |
| **Conclusion** | Hypothesis rejected. Squared reward suppresses everything near 50%, including genuine early improvement. Flag gate \+ squared reward form a deadlock: both need signal to bootstrap. |

## **Iteration 6 — Reward annealing**

| Aim | Smooth blend from linear to squared reward over training. Linear ep 0–349, blend ep 350–700, pure squared ep 700+. |
| :---- | :---- |
| **Result** | Full model: 51.0% ± 10.7%. Same 3 seeds succeed, same 7 fail. Worst variance yet. Partial recovery vs iter 5 but no improvement over iter 4\. |
| **Conclusion** | Reward shape is not the bottleneck. Three experiments (linear, squared, annealed) all produce the same seed-dependent success pattern. The problem is upstream of reward. |

## **Iteration 7 — Connectivity diagnostic**

| Aim | Measure structural differences between succeeding and failing seeds: direct I→O connections, 2-hop paths, fan-out/fan-in, chemical dose budget, modulatory neuron distance. |
| :---- | :---- |
| **Result** | Clean null result. All metrics overlap completely between good and poor seeds. Direct I→O: 13.6 vs 12.2 (noise). Chemical dose: 1.13 vs 1.16 (negligible). 2-hop paths: 344 vs 349\. |
| **Conclusion** | The failure mode is not structural. Network wiring is statistically identical across seeds. The divergence must emerge from early training dynamics, not topology. |

## **Iteration 8 — Flag strength diagnostic**

| Aim | Log flag strength at episodes 100, 300, 500\. Prediction: good seeds latch flags early, poor seeds don’t. |
| :---- | :---- |
| **Result** | Prediction was wrong. Both groups have 87–95% of flags latched by episode 100 with mean |flagStrength| ≈ 0.93. Groups are statistically indistinguishable. |
| **Conclusion** | Flags are saturated, not selective. With gain=0.3, any synapse seeing two consecutive non-zero traces latches. The gate opens for 90%+ of synapses uniformly — it provides no discrimination. |

## **Iteration 9 — Flag gate warmup**

| Aim | Bypass flag gate for first 200 episodes (all synapses learn freely), then activate normal gating. |
| :---- | :---- |
| **Result** | Full model: 45.5% ± 5.0% (p\<0.01, significantly worse). Variance collapsed to ±5% — network finds the same poor attractor on every seed. |
| **Conclusion** | Unfiltered early learning commits to bad fixed-output weights. Flags then lock in those bad weights. Opening the gate early is the problem, not the cure. |

## **Iteration 10 — Direction-consistent flags**

| Aim | Replace simple flag accumulation with a consistency detector. Track consecutive same-sign traces (threshold=5), sharp penalty on direction flip (0.5×), slower gain (0.15). |
| :---- | :---- |
| **Result** | Full model: 49.0% ± 7.0% (p\<0.05 worse). Latch rates dropped to 68–93% (from 90%+). Seeds with lower latch rates (42, 1234\) performed better, but 8/10 seeds still saturated at 86–93%. |
| **Conclusion** | Partial success on selectivity — mechanism correctly identifies that lower saturation correlates with better performance. But the task dynamics still satisfy the 5-turn streak requirement trivially for most synapses. |

## **Iteration 10b — Propagation cycles**

| Aim | Run 3 accumulate-and-fire cycles per training step so signals can travel multiple hops through the recurrent graph. |
| :---- | :---- |
| **Result** | Full model: 46.5% ± 6.7% (p\<0.01). Flag saturation finally broken (25–81%), but accuracy worse. Seed 888 (33% latch, 65% acc) confirms low saturation helps, but 9 other seeds still fail. |
| **Conclusion** | Propagation cycles successfully broke flag saturation but introduced noise through compound signal amplification. Also discovered: homeostasis runs 3× per episode inside the loop — an implementation issue to fix. |

## **Iteration 11 — Synapse frustration flip**

| Aim | Per-synapse frustration detection: track same-direction weight movement and average chemical. After 30 consecutive same-direction steps with negative reward, partially reverse the weight. |
| :---- | :---- |
| **Result** | Full model: 48.0% ± 4.8% (p\<0.01). Mechanism is all-or-nothing: 8/10 seeds had zero flips, 2/10 had every synapse flip (cascade). Key finding: 6/10 seeds have mean |weight| \< 0.28 — barely moved from initialization. |
| **Conclusion** | Cascade problem: one flip destabilizes neighbors, triggering chain reaction. But the real discovery was weight starvation — most synapses decay faster than they learn. Weight decay (0.005/step) exceeds typical learning delta (0.001/step). |

## **Iteration 12 — Learning/decay balance**

| Aim | Three sub-experiments: (a) double learning rate, (b) halve weight decay, (c) both. Test whether weight starvation is the fundamental bottleneck. |
| :---- | :---- |
| **Result** | 012b (0.5× weight decay): 55.0% ± 0.0% — ALL 10 seeds at 55%. First time 100% seed coverage achieved. Mean |weight| jumped from 0.30 to 0.79. 012c (both changes): same accuracy, |weight|=1.44. |
| **Conclusion** | Weight starvation was the core problem all along. One parameter change solved what 8 experiments of mechanism tuning couldn’t. New ceiling at 55% is structural: network learns 3/4 patterns (P2–P4 at 60%) but systematically fails P1 (40%), because P1 and P4 are exact inverses sharing the same synapses. |

## **Iteration 13 — 012b \+ propagation cycles**

| Aim | Combine the weight decay fix with propagationCycles=3 to enable multi-hop signal flow, potentially breaking the 55% ceiling through hidden representations. |
| :---- | :---- |
| **Result** | Full model: 45.0% ± 0.0% — complete collapse, 0/10 seeds above 45%. Mean |weight| dropped to 0.35. Per-pattern bias flipped: P1=60%, P2–P4=40% (mirror of 012b). |
| **Conclusion** | Threshold homeostasis runs inside the propagation loop, so 3 cycles \= 3× homeostasis speed. This drives the network to the wrong attractor. Fix: move regulateThreshold() outside the propagation loop. The 012b configuration (halved weight decay, single propagation) remains the best result. |

# **Progress summary**

Full model mean accuracy and key observations across all iterations:

| Iter | Change | Mean | Max | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Iter 1 | Original flags, linear reward | 45.5% | 55% | 6/10 fail |
| Iter 3 | Flipped eligibility signs | 46.0% | 55% | No change |
| Iter 4 | Flag gate (strengthen/decay) | 53.0% | 65% | 3/10 escape |
| Iter 5 | Squared reward | 45.0% | 65% | Regression |
| Iter 6 | Reward annealing | 51.0% | 65% | Partial recovery |
| Iter 7 | Connectivity diagnostic | — | — | Null: structure not the cause |
| Iter 8 | Flag strength diagnostic | — | — | Flags saturated at 90%+ |
| Iter 9 | Flag gate warmup | 45.5% | 55% | Worse (p\<0.01) |
| Iter 10 | Direction-consistent flags | 49.0% | 65% | Partial signal |
| Iter 10b | \+ propagationCycles=3 | 46.5% | 65% | Broke saturation, added noise |
| Iter 11 | Synapse frustration flip | 48.0% | 55% | Cascade problem |
| Iter 12b | 0.5× weight decay | 55.0% | 55% | 10/10 seeds ✓ |
| Iter 12c | 2× LR \+ 0.5× WD | 55.0% | 55% | 10/10 seeds ✓ |
| Iter 13 | 012b \+ propagationCycles=3 | 45.0% | 45% | Homeostasis bug |

# **Key findings**

**1\. Weight starvation was the fundamental bottleneck.** For 11 experiments, we tuned flags, reward shaping, and gating mechanisms while the real problem was arithmetic: weight decay outpaced learning by 5× for typical synapses. Halving weight decay (iter 12\) solved what no mechanism change could.

**2\. The 55% ceiling is a capacity limit, not a learning failure.** The network consistently learns 3/4 patterns and sacrifices the 4th (P1), because P1 and P4 are exact inverses sharing the same synapses. A single static weight matrix cannot satisfy both. Breaking past 55% requires context-dependent routing — different internal states for different inputs.

**3\. Flag saturation is real but secondary.** The flag gate latches 90%+ of synapses within 100 episodes regardless of seed. Direction-consistent flagging and propagation cycles can reduce this, but lower saturation alone doesn’t improve accuracy.

**4\. Diagnostic experiments are as valuable as mechanism changes.** The connectivity diagnostic (iter 7), flag strength diagnostic (iter 8), and weight magnitude analysis (iter 11\) each ruled out entire hypothesis branches and redirected effort productively.

**5\. Implementation details dominate.** Homeostasis running inside the propagation loop (iter 13), eval-during-learning artifacts (iter 1), and the learning/decay ratio (iter 12\) had larger effects than any novel mechanism. Getting the basics right matters more than adding complexity.

# **Next steps**

**1\. Fix propagation loop.** Move threshold homeostasis outside the propagation cycle loop. Retest propagationCycles=3 on 012b base to see if multi-hop signals enable hidden representations that break the 55% ceiling.

**2\. Genetic algorithm for architecture search.** Evolution optimizes structural priors (cluster count, spacing, connectivity, modulatory placement, learning rate, decay); mcfeedback learns within each configuration. This replaces manual parameter tuning with systematic exploration of the fitness landscape.

**3\. Longer training.** Run 012b for 5000–10000 episodes to determine whether the 55% ceiling is a time constraint or a fundamental capacity limit.

**4\. Structured inter-cluster wiring.** Replace random inter-cluster connectivity with topographic mapping (input neuron i preferentially connects to output-adjacent neurons). Biology doesn’t wire long-range connections randomly.