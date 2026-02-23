# mcfeedback
Current: Astrocyte-Mediated Learning via Local Perturbation with Global Scalar Reward

# Exploration in Biologically Inspired Computing
An Experimental Exploration in Biologically Inspired Computing
This is an experimental journey through biologically inspired local learning algorithms for neural networks. The repository documents the full path from initial hypothesis to working mechanism, including every failure along the way. It is not a polished library — it is a research log in code form.
See astrocyte_writeup.pdf in the root directory for the full writeup with results, analysis, and references.

# About
What this is about
Can a neural network learn using only a single scalar reward ("that was good" / "that was bad") and no backpropagation? The answer depends on where you apply changes. Modifying all synapses at once makes credit assignment impossible. Restricting changes to small local neighbourhoods — controlled by astrocyte-like entities — makes a single scalar sufficient.
Versions

# Phase 1a
Phase 1a — Neuron-centric learning (experiments 001–018)
Eligibility traces, chemical diffusion from modulatory neurons, flag gates, dampening filters. Eighteen experiments, none producing genuine pattern discrimination. The 55% accuracy ceiling turned out to be a constant-output attractor — the network outputting the same vector for every input. These experiments are valuable as negative results documenting what does not work and why.

# Phase 1b
Phase 1b — Cursor and astrocyte-mediated learning (experiments 020–024)
Stripped everything out. No eligibility traces, no chemical diffusion, no dampening, no flags. Replaced with spatial perturbation controlled by astrocyte-like entities: fixed territories, activity-dependent activation, random perturbation, global scalar keep/revert, self-adapting thresholds with maturity-scaled exploration. Produces genuine pattern discrimination (up to 100% on individual patterns, 67.7% mean across 8 patterns vs 50% chance) on all 10 seeds tested.

# Status
This is an active experiment, not a finished product. Code quality reflects rapid AI-assisted prototyping. Expect rough edges, dead ends, and commented-out experiments. That's the point — the repository is the lab notebook, not the published result.


# Author
Dusty Wilhelm Murray & Claude (Anthropic), February 2026
