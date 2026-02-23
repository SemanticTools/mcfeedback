// Binary reward: fraction of output bits exactly matching the target (0.0 – 1.0)
// Used for reporting actual task performance.
export function computeBinaryReward(binaryOutput, target) {
  let matches = 0;
  for (let i = 0; i < target.length; i++) {
    if (binaryOutput[i] === target[i]) matches++;
  }
  return matches / target.length;
}

// Soft reward: sigmoid of (activation - threshold) × target direction (0.0 – 1.0)
// Provides a continuous learning signal even when no bits have yet crossed threshold.
// Used by the hill-climber to decide whether to accept a perturbation.
export function computeSoftReward(activations, target, network) {
  const outputIds = [5, 6, 7, 8, 9];
  let reward = 0;
  for (let i = 0; i < target.length; i++) {
    const threshold = network.neurons[outputIds[i]].threshold;
    const a = activations[i] - threshold;
    // target=1: want activation above threshold → sigmoid(+a)
    // target=0: want activation below threshold → sigmoid(-a)
    const sign = target[i] === 1 ? 1 : -1;
    reward += 1 / (1 + Math.exp(-sign * a));
  }
  return reward / target.length;
}
