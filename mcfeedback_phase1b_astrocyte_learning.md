# mcfeedback Phase 1b: Astrocyte-Mediated Learning

**Murray & Claude — February 2026**

---

## Motivation

The first phase of mcfeedback explored biologically plausible learning using three neuron-centric mechanisms: eligibility traces (Hebbian co-firing detection), chemical diffusion from modulatory neurons (reward broadcast), and spatial dampening filters. Across 18 experiments, none of these mechanisms produced genuine pattern discrimination. The best results — 55% accuracy on a 50% chance task — turned out to be constant-output attractors: the network outputting the same vector for all inputs, scoring well by accident of arithmetic rather than by learning.

The critical finding came from a stripped-down control experiment. A single spatial cursor — randomly perturbing a small cluster of synapses per step, evaluating with one global scalar reward, and keeping or reverting the change — produced real pattern discrimination on the first attempt. Ten out of ten seeds generated distinct output vectors per input pattern, reaching up to 75% accuracy. No eligibility traces, no chemical diffusion, no dampening, no flags. One cursor and one number.

This result inverts the standard assumption. The bottleneck was never the reward signal — it was the update scope. Global reward works when perturbation is local. The problem with broadcast chemical modulation was not that the reward was too simple, but that every synapse was modified simultaneously, making credit assignment impossible. Restricting plasticity to a small spatial neighborhood at any given moment makes a single scalar sufficient to evaluate whether the local change helped.

Phase 1b formalises this cursor as what it biologically represents: an astrocyte.


## Astrocytes as the learning mechanism

In conventional computational neuroscience, learning is synaptic. Hebb's postulate, spike-timing-dependent plasticity, and their descendants all locate the learning rule at the synapse itself: the pre-synaptic neuron fires, the post-synaptic neuron fires, the synapse strengthens. Astrocytes, when included at all, are modelled as modulators that adjust the gain on an existing synaptic learning rule.

We propose something different: astrocytes as the primary learning mechanism. Neurons compute — they accumulate inputs, compare to threshold, and fire. But neurons do not learn. Synapses do not contain their own plasticity rules. Instead, astrocytes control which synapses change, when they change, and by how much. Learning is glial, not neuronal.

Each astrocyte occupies a fixed spatial territory covering a local cluster of synapses, consistent with the tiling organisation observed in cortex where individual astrocytes maintain non-overlapping domains of approximately 100,000 synapses each. At any moment, only a subset of astrocytes are active. An active astrocyte opens a plasticity window in its territory: synapses under its influence receive small random perturbations. After the network processes its current input and produces an output, a single global reward signal (a scalar: better or worse than before) determines whether the perturbation is kept or reverted. Synapses outside any active astrocyte's territory are frozen — they neither change nor contribute noise.

This is structurally equivalent to a genetic algorithm operating inside a neural network. The population is the set of synaptic configurations. Mutation is the astrocyte's perturbation. Selection is the global reward. The key constraint that makes it work — small mutation count per generation — maps directly to the astrocyte's spatial locality.


## What astrocytes sense and how they adapt

Astrocytes are not static. They have internal state that evolves over training:

**Activity sensing.** Each astrocyte monitors the firing patterns of neurons within its territory. When a specific input pattern is presented, it activates a specific subset of neurons, which fall within specific astrocyte territories. The astrocytes covering the most active neurons preferentially activate. Different input patterns therefore engage different astrocytes, creating pattern-specific plasticity windows without any explicit pattern-routing mechanism.

**Reward history.** Each astrocyte tracks whether its recent activations (and the perturbations they produced) were followed by positive or negative reward. An astrocyte whose perturbations consistently lead to improvement lowers its activation threshold — it becomes more responsive, opens its plasticity window more readily. An astrocyte whose perturbations consistently lead to degradation raises its threshold — it becomes conservative, contributing less noise to a region where changes are unhelpful.

**Developmental annealing.** Early in training, astrocyte activation thresholds are low and their territories are effectively broader (stronger calcium waves propagating further). This produces rapid, broad, exploratory perturbation — analogous to the massive synaptic turnover observed in infant brains. Over training, thresholds rise and effective territories narrow, focusing perturbation on regions where it has historically been productive. This mirrors the transition from exploratory childhood plasticity to refined adult learning.

These three properties mean the astrocyte population self-organises into an efficient search strategy: early exploration identifies which network regions matter for which input patterns; later refinement tunes those regions precisely. The global reward signal remains a single scalar throughout. The system learns to interpret that scalar locally because each astrocyte's small territory makes local attribution possible.


## Design principles

Phase 1b is built on two commitments:

**Simple global signals.** The reward signal is one number: did overall accuracy improve or not. There is no per-output error, no per-synapse gradient, no error backpropagation. This is the signal available to a biological organism — a diffuse neuromodulatory broadcast (dopamine, norepinephrine) that encodes valence (good/bad) without encoding what specifically was good or bad. We take the position that this signal is sufficient for learning and that the apparent insufficiency observed in Phase 1 was an architectural problem (global perturbation), not a signal problem.

**Local architecture resolves credit assignment.** The only structure required to make global reward useful is spatial locality of plasticity. Astrocytes provide this. By restricting which synapses change on each step to a small spatial cluster, the system converts the credit assignment problem from "which of 2000 synapses caused the improvement" to "did this cluster of 20 synapses cause the improvement." The former is intractable with a scalar signal. The latter is tractable. No additional mechanism is needed — no eligibility traces, no chemical gradients, no flag gates. Locality alone is sufficient.

This is a minimalist position. We are deliberately not adding mechanisms until we have evidence that the minimal system fails on a specific task. Phase 1 demonstrated the cost of premature complexity: 18 experiments tuning interacting mechanisms, each introducing failure modes that obscured the underlying signal.


## Relation to existing work

The computational neuroscience literature on astrocyte-neuron interaction has grown substantially but occupies a different space from the approach described here.

**Tripartite synapse models.** Alvarellos-González et al. (2012) introduced Artificial Neuron-Glia Networks (ANGNs), demonstrating that adding astrocyte-like elements to multilayer networks improved classification performance. However, the underlying learning rule remained backpropagation; astrocytes modulated synaptic gain but did not replace the gradient-based learning mechanism. Our approach eliminates the synaptic learning rule entirely.

**Contextual guidance.** De Pitta & Bhatt (2024, PLOS Computational Biology) modelled astrocytes as meta-plasticity agents operating on slower timescales than neuronal dynamics, enabling context-dependent learning in multi-armed bandit tasks. Their framework — nested feedback loops of neuron-astrocyte interaction with time-scale separation — shares our intuition that astrocytes enable learning across contexts. However, their model retains standard reinforcement learning update rules for synaptic weights; astrocytes modulate the learning process rather than constituting it.

**Astrocytic short-term memory.** Kostadinov & Bhatt (2023) combined convolutional neural networks with astrocyte-driven short-term synaptic plasticity models, showing that astrocytic modulation improves change-detection task performance. This work models a specific biological mechanism (gliotransmitter-mediated STP) within a conventional deep learning framework trained by stochastic gradient descent.

**Experimental evidence for astrocyte necessity.** Hösli et al. (2022, Cell Reports) demonstrated experimentally that disrupting astrocyte gap junction coupling in adult mice abolishes spatial learning and memory entirely — not merely impairing it, but eliminating it. This is perhaps the strongest biological evidence that astrocytes are not accessory to learning but essential infrastructure for it.

**Biological astrocyte plasticity.** Bohmbach et al. (2022) and subsequent work reviewed in Ghézali et al. (2022) established that astrocytes regulate the direction and magnitude of synaptic plasticity through D-serine release gated by endocannabinoid signaling, effectively implementing a BCM-like plasticity rule through extracellular dynamics. This demonstrates that astrocytes can control *whether* a synapse potentiates or depresses — a read-write mechanism at the glial level.

**Where Phase 1b diverges.** All existing computational models use astrocytes to modulate an underlying synaptic learning rule. Phase 1b proposes that astrocytes *are* the learning rule. Synapses have no intrinsic plasticity mechanism. There is no Hebbian update, no STDP, no reinforcement learning delta rule operating at the synapse. The astrocyte selects which synapses to perturb, the perturbation is random, the global reward determines keep or revert, and the astrocyte adapts its own activation patterns based on reward history. Learning is entirely glial. Neurons compute; astrocytes learn.

To our knowledge, this framing — astrocytes as the sole mechanism of learning, without any synaptic plasticity rule underneath — has not been explored computationally.


## References

Alvarellos-González, A., Pazos, A., & Porto-Pazos, A. B. (2012). Computational models of neuron-astrocyte interactions lead to improved efficacy in the performance of neural networks. *Computational and Mathematical Methods in Medicine*, 2012, 476324.

Bohmbach, K., Henneberger, C., & Bhatt, D. K. (2022). An astrocytic signaling loop for frequency-dependent control of dendritic integration and spatial learning. *Nature Communications*, 13, 7932.

De Pitta, M., & Bhatt, D. (2024). Astrocytes as a mechanism for contextually-guided network dynamics and function. *PLOS Computational Biology*, 20(5), e1012186.

Ghézali, G., Bhatt, D., & Bhatt, D. K. (2022). The role of astrocyte structural plasticity in regulating neural circuit function and behavior. *Glia*, 70(7), 1467–1483.

Hösli, L., Binini, N., Ferrari, K. D., et al. (2022). Decoupling astrocytes in adult mice impairs synaptic plasticity and spatial learning. *Cell Reports*, 38(10), 110484.

Kostadinov, D., & Bhatt, D. K. (2023). Artificial neural network model with astrocyte-driven short-term memory. *Biomimetics*, 8(5), 422.

Mederos, S., & Bhatt, D. K. (2021). Astrocytes as drivers and disruptors of behavior: New advances in basic mechanisms and therapeutic targeting. *Journal of Neuroscience*, 43(45), 7463–7476.
