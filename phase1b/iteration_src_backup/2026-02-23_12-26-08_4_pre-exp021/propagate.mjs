// Single propagation pass
// Cluster 0 input neurons (ids 0–4) are set from the input pattern.
// All other neurons compute weighted sum from fired pre-synaptic neurons,
// then fire if sum > threshold.
//
// Returns { binaryOutput, activations }:
//   binaryOutput  — array of 0/1 for cluster 0 output neurons (ids 5–9)
//   activations   — raw weighted sums for those same neurons (before thresholding)

export function propagate(network, inputPattern) {
  const { neurons, synapses } = network;

  // Reset fired state for all neurons
  for (const n of neurons) n.fired = false;

  // Drive cluster 0 input neurons from the pattern
  for (let i = 0; i < 5; i++) {
    neurons[i].fired = inputPattern[i] === 1;
  }

  // Accumulate activations (single pass — no recurrence)
  const activation = new Float64Array(neurons.length);
  for (const syn of synapses) {
    if (neurons[syn.pre].fired) {
      activation[syn.post] += syn.weight;
    }
  }

  // Fire non-input neurons; update fire rate counters
  for (const n of neurons) {
    if (n.type === 'input') continue;
    n.fired = activation[n.id] > n.threshold;
    n.stepCount++;
    if (n.fired) n.fireCount++;
  }

  const outputIds = [5, 6, 7, 8, 9];
  return {
    binaryOutput: outputIds.map(i => (neurons[i].fired ? 1 : 0)),
    activations:  outputIds.map(i => activation[i]),
  };
}
