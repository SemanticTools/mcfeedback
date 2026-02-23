import { distance3d } from './utils.mjs';

function falloff(distance, mode, radius) {
  if (distance <= 0) return 1;
  switch (mode) {
    case 'inverseSquare': return 1 / (distance * distance);
    case 'linear':        return Math.max(0, 1 - distance / radius);
    case 'constant':      return 1;
    case 'inverse':
    default:              return 1 / distance;
  }
}

// Per-output-bit chemical diffusion.
// Each output neuron acts as its own reward source: correct bit → positive signal,
// wrong bit → negative signal. Diffuses spatially from that neuron's position,
// reaching all synapses within chemicalDiffusionRadius.
// This replaces the global modulatory broadcast when config.perBitReward is true.
export function diffuseChemicalPerBit(network, outputNeurons, targetPattern) {
  const { neurons, neuronState, synapses, config } = network;

  for (let i = 0; i < outputNeurons.length; i++) {
    const on       = outputNeurons[i];
    const actual   = neuronState.get(on.id).output;
    const expected = targetPattern[i] ?? 0;
    const signal   = actual === expected
      ? config.positiveRewardStrength
      : config.negativeRewardStrength;

    for (const synapse of synapses) {
      const toNeuron = neurons.get(synapse.to);
      const d = distance3d(on, toNeuron);
      if (d <= config.chemicalDiffusionRadius) {
        synapse.chemicalLevel += signal * falloff(d, config.chemicalFalloff, config.chemicalDiffusionRadius);
      }
    }
  }
}

// Diffuses chemical from every modulatory neuron that fired this cycle.
// rewardSignal: positive (correct output) or negative (incorrect output).
export function diffuseChemical(network, rewardSignal) {
  const { neurons, neuronState, synapses, config } = network;

  for (const [id, neuron] of neurons) {
    if (neuron.type !== 'modulatory') continue;
    if (!neuronState.get(id).firedThisCycle) continue;

    const strength = rewardSignal > 0
      ? rewardSignal * config.positiveRewardStrength
      : rewardSignal * Math.abs(config.negativeRewardStrength);

    for (const synapse of synapses) {
      // Use the post-synaptic neuron's position as the synapse's spatial location
      const toNeuron = neurons.get(synapse.to);
      const d = distance3d(neuron, toNeuron);
      if (d <= config.chemicalDiffusionRadius) {
        synapse.chemicalLevel += strength * falloff(d, config.chemicalFalloff, config.chemicalDiffusionRadius);
      }
    }
  }
}
