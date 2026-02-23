// Inference: run all training patterns through a frozen network.
// No perturbation, no homeostasis, no weight changes.
// Returns per-pattern results plus aggregate stats.

import { propagate } from './propagate.mjs';
import { computeBinaryReward } from './reward.mjs';
import { trainingPatterns } from './task.mjs';

export function runInference(network) {
  const results = [];

  for (let i = 0; i < trainingPatterns.length; i++) {
    const { input, target } = trainingPatterns[i];
    const { binaryOutput } = propagate(network, input);
    const accuracy = computeBinaryReward(binaryOutput, target);
    results.push({
      label: `P${i + 1}`,
      input,
      target,
      output: binaryOutput,
      accuracy,
    });
  }

  const meanAccuracy = results.reduce((s, r) => s + r.accuracy, 0) / results.length;
  const distinctOutputs = new Set(results.map(r => r.output.join(''))).size;

  return { results, meanAccuracy, distinctOutputs };
}
