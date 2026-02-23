// Reward: fraction of output bits matching the target (0.0 – 1.0)

export function computeReward(output, target) {
  let matches = 0;
  for (let i = 0; i < target.length; i++) {
    if (output[i] === target[i]) matches++;
  }
  return matches / target.length;
}
