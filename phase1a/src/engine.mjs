import { computeOutput, regulateThreshold } from './neuron.mjs';
import { updateWeight, updateActivityHistory, decayChemical, updateFlagStrength, updateFrustration } from './synapse.mjs';
import { computeEligibility } from './flagging.mjs';
import { combinedDampening } from './dampening.mjs';
import { computeAmbientFields } from './ambient.mjs';
import { diffuseChemical, diffuseChemicalPerBit } from './chemical.mjs';

// Frozen forward pass — no weight updates, no learning, no side effects.
// Saves and restores neuron state so training state is not disturbed.
export function evaluate(network, inputPattern, targetPattern) {
  const { neurons, neuronState, synapses } = network;

  // Save state
  const savedState = new Map();
  for (const [id, s] of neuronState) savedState.set(id, { ...s });
  const savedOutputs = synapses.map(s => s.output);

  // Clamp inputs (no stat tracking)
  const inputNeurons = [...neurons.values()].filter(n => n.type === 'input');
  for (let i = 0; i < inputNeurons.length; i++) {
    const state = neuronState.get(inputNeurons[i].id);
    state.output         = inputPattern[i] ?? 0;
    state.firedThisCycle = state.output === 1;
  }

  // Forward pass using current weights (repeated for propagationCycles)
  const accumulators = new Map();
  for (let _cycle = 0; _cycle < (network.config?.propagationCycles ?? 1); _cycle++) {
    for (const id of neurons.keys()) accumulators.set(id, 0);
    for (const s of synapses) {
      const preOutput = neuronState.get(s.from).output;
      accumulators.set(s.to, (accumulators.get(s.to) ?? 0) + s.weight * preOutput);
    }
    for (const [id, neuron] of neurons) {
      if (neuron.type === 'input') continue;
      const state = neuronState.get(id);
      const input = accumulators.get(id) ?? 0;
      state.output         = input >= state.threshold ? 1 : 0;
      state.firedThisCycle = state.output === 1;
    }
  }

  // Read outputs
  const outputNeurons = [...neurons.values()].filter(n => n.type === 'output');
  let correct = 0;
  const outputs = outputNeurons.map((n, i) => {
    const actual = neuronState.get(n.id).output;
    if (actual === (targetPattern[i] ?? 0)) correct++;
    return actual;
  });
  const accuracy = correct / outputNeurons.length;
  const loss     = outputNeurons.reduce((sum, n, i) => {
    const diff = neuronState.get(n.id).output - (targetPattern[i] ?? 0);
    return sum + diff * diff;
  }, 0);

  // Restore state — training is unaffected
  for (const [id, s] of savedState) {
    const live = neuronState.get(id);
    Object.assign(live, s);
  }

  return { outputs, accuracy, loss };
}

export function step(network, inputPattern, targetPattern, episode = 0) {
  const { neurons, neuronState, synapses, config } = network;

  // 1. Clamp input neurons to inputPattern
  const inputNeurons = [...neurons.values()].filter(n => n.type === 'input');
  for (let i = 0; i < inputNeurons.length; i++) {
    const state = neuronState.get(inputNeurons[i].id);
    state.output         = inputPattern[i] ?? 0;
    state.firedThisCycle = state.output === 1;
    state.cycleCount    += 1;
    if (state.firedThisCycle) state.fireCount += 1;
    state.fireRate = state.fireCount / state.cycleCount;
  }

  // 2. Forward pass for non-input neurons (repeated for propagationCycles)
  // Only accumulation + firing repeat; stat updates and threshold regulation run once.
  const accumulators = new Map();
  for (let _cycle = 0; _cycle < (config.propagationCycles ?? 1); _cycle++) {
    // 2a. Reset accumulators
    for (const id of neurons.keys()) accumulators.set(id, 0);

    // 2b. Accumulate weighted inputs
    for (const s of synapses) {
      const preOutput = neuronState.get(s.from).output;
      accumulators.set(s.to, (accumulators.get(s.to) ?? 0) + s.weight * preOutput);
    }

    // 2c. Fire only — no stat updates, no threshold regulation (those run once below)
    for (const [id, neuron] of neurons) {
      if (neuron.type === 'input') continue;
      const state = neuronState.get(id);
      const input = accumulators.get(id) ?? 0;
      state.output         = input >= state.threshold ? 1 : 0;
      state.firedThisCycle = state.output === 1;
    }
  }

  // 2d. Stat updates + threshold regulation — runs exactly once per episode
  // When config.fixedOutputThreshold is true, output neurons skip threshold regulation:
  // their thresholds are determined by the per-bit reward signal, not homeostasis.
  for (const [id, neuron] of neurons) {
    if (neuron.type === 'input') continue;
    const state = neuronState.get(id);
    state.cycleCount++;
    if (state.firedThisCycle) state.fireCount++;
    state.fireRate = state.fireCount / state.cycleCount;
    if (!(config.fixedOutputThreshold && neuron.type === 'output')) {
      regulateThreshold(state, config);
    }
  }

  // 3. Ambient field
  computeAmbientFields(neurons, neuronState);

  // 4. Flagging
  for (const s of synapses) {
    const preState  = neuronState.get(s.from);
    const postState = neuronState.get(s.to);
    s.eligibilityTrace = computeEligibility(preState, postState, config);
  }

  // 4.5. Activity history — updated from raw trace before dampening
  // This must happen here: dampening reads activityHistory, so history must
  // reflect raw participation, not the already-dampened result.
  for (const s of synapses) updateActivityHistory(s, config);

  // 5. Dampening
  if (!config._skipDampening) {
    for (const s of synapses) {
      const preState  = neuronState.get(s.from);
      const postState = neuronState.get(s.to);
      s.eligibilityTrace *= combinedDampening(s, preState, postState, config);
    }
  }

  // 5.5. Flag strength — reward consistent trace signal across turns
  if (config.flagStrengthThreshold > 0) {
    for (const s of synapses) updateFlagStrength(s, config);
  }

  // 6. Reward signal: compare output neurons to target
  const outputNeurons = [...neurons.values()].filter(n => n.type === 'output');
  let correct = 0;
  const outputs = outputNeurons.map((n, i) => {
    const actual = neuronState.get(n.id).output;
    if (actual === (targetPattern[i] ?? 0)) correct++;
    return actual;
  });
  const accuracy    = correct / outputNeurons.length;
  const _linear = (accuracy - 0.5) * 2; // range [-1, +1]
  let rewardSignal;
  if (config.rewardAnnealStart != null && config.rewardAnnealEnd != null) {
    // Reward annealing: blend from linear → squared over training.
    // Episodes < annealStart: pure linear (bootstrap phase).
    // Episodes annealStart–annealEnd: smooth blend.
    // Episodes > annealEnd: pure squared (refinement phase).
    const blend = Math.max(0, Math.min(1,
      (episode - config.rewardAnnealStart) / (config.rewardAnnealEnd - config.rewardAnnealStart)
    ));
    const _squared = Math.sign(_linear) * Math.pow(Math.abs(_linear), 2.0);
    rewardSignal = (1 - blend) * _linear + blend * _squared;
  } else {
    // No annealing: fixed exponent (defaults to 1 = linear for old experiments).
    rewardSignal = Math.sign(_linear) * Math.pow(Math.abs(_linear), config.rewardExponent ?? 1);
  }

  // 6.5. Provisional weight commit/revert — checked after this step's reward is known
  if (config.provisionalWeights && network._provisional != null) {
    if (accuracy >= network._provisional.reward) {
      // Improved or same: commit — weights already applied, nothing to do
    } else {
      // No improvement: revert to pre-update weights saved at end of last step
      const pw = network._provisional.preWeights;
      for (let i = 0; i < synapses.length; i++) synapses[i].weight = pw[i];
    }
    network._provisional = null;
  }

  // Fire modulatory neurons based on reward — all at once, or one at a time (cycling)
  const modulatoryNeurons = [...neurons.values()].filter(n => n.type === 'modulatory');
  if (config.modCycling) {
    if (network.modCursor == null) network.modCursor = 0;
    const activeIdx = network.modCursor % modulatoryNeurons.length;
    network.modCursor++;
    for (let i = 0; i < modulatoryNeurons.length; i++) {
      const state = neuronState.get(modulatoryNeurons[i].id);
      state.firedThisCycle = (i === activeIdx) && rewardSignal !== 0;
      state.output         = state.firedThisCycle ? 1 : 0;
    }
  } else {
    for (const mn of modulatoryNeurons) {
      const state = neuronState.get(mn.id);
      state.firedThisCycle = rewardSignal !== 0;
      state.output         = state.firedThisCycle ? 1 : 0;
    }
  }

  // 7. Chemical diffusion (with optional radius annealing)
  if (config.chemicalDiffusionRadiusMin != null) {
    if (network._chemRadiusStart == null) network._chemRadiusStart = config.chemicalDiffusionRadius;
    const totalEps = config.trainingEpisodes ?? 1000;
    const t = Math.min(1, (episode - 1) / Math.max(1, totalEps - 1));
    config.chemicalDiffusionRadius = network._chemRadiusStart
      + (config.chemicalDiffusionRadiusMin - network._chemRadiusStart) * t;
  }
  if (config.perBitReward) {
    // Each output neuron broadcasts its own correctness signal spatially.
    // Replaces the global modulatory broadcast.
    diffuseChemicalPerBit(network, outputNeurons, targetPattern);
  } else {
    diffuseChemical(network, rewardSignal);
  }

  // 8. Weight update + frustration detection
  // Save weights before update so the provisional mechanism can revert if needed next step
  const preWeights = config.provisionalWeights ? synapses.map(s => s.weight) : null;
  for (const s of synapses) {
    const delta = updateWeight(s, config, episode);
    updateFrustration(s, config, delta);
  }
  if (config.provisionalWeights) {
    network._provisional = { preWeights, reward: accuracy };
  }

  // 9. Bookkeeping
  for (const s of synapses) decayChemical(s, config);

  // 10. Compute metrics
  const loss = outputNeurons.reduce((sum, n, i) => {
    const diff = neuronState.get(n.id).output - (targetPattern[i] ?? 0);
    return sum + diff * diff;
  }, 0);

  const allStates       = [...neuronState.values()];
  const meanFireRate    = allStates.reduce((s, st) => s + st.fireRate, 0) / allStates.length;
  const meanThreshold   = allStates.reduce((s, st) => s + st.threshold, 0) / allStates.length;
  const meanWeight      = synapses.reduce((s, syn) => s + Math.abs(syn.weight), 0) / synapses.length;
  const activeSynFrac   = synapses.filter(syn => Math.abs(syn.eligibilityTrace) > 1e-6).length / synapses.length;

  return { outputs, accuracy, loss, meanWeight, meanFireRate, meanThreshold, activeSynFrac };
}
