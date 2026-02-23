// Perturbation: add Gaussian noise to eligible synapses.
// Save weights before perturbation so they can be reverted if reward drops.

const PERTURBATION_STD = 0.05;
const MAX_WEIGHT = 2.0;

// Box-Muller transform: uniform → standard normal
function gaussianRandom() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
}

// Returns an array of { synapse, savedWeight } for later reverting
export function saveWeights(eligibleSynapses) {
  return eligibleSynapses.map(s => ({ synapse: s, savedWeight: s.weight }));
}

export function perturb(eligibleSynapses) {
  for (const s of eligibleSynapses) {
    s.weight += gaussianRandom() * PERTURBATION_STD;
    s.weight = Math.max(-MAX_WEIGHT, Math.min(MAX_WEIGHT, s.weight));
  }
}

export function revertWeights(saved) {
  for (const { synapse, savedWeight } of saved) {
    synapse.weight = savedWeight;
  }
}
