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
