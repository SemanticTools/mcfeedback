import { distance3d } from './utils.mjs';

// Updates ambientField in neuronState for every neuron.
// Uses precomputed neighbourIds from Tier 1 neuron data.
export function computeAmbientFields(neurons, neuronState) {
  for (const [id, neuron] of neurons) {
    let field = 0;
    for (const nid of neuron.neighbourIds) {
      const neighbourNeuron = neurons.get(nid);
      const d = distance3d(neuron, neighbourNeuron);
      if (d > 0) {
        field += neuronState.get(nid).output / d;
      }
    }
    neuronState.get(id).ambientField = field;
  }
}
