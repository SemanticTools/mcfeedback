export function createSynapse(fromId, toId, initialWeight) {
  return {
    from:             fromId,
    to:               toId,
    weight:           initialWeight,
    eligibilityTrace: 0,
    activityHistory:  0,
    chemicalLevel:    0,
  };
}

export function updateWeight(synapse, config) {
  // Decay first: weights must be continuously earned
  synapse.weight *= (1 - config.weightDecay);

  const rawDelta = synapse.eligibilityTrace * synapse.chemicalLevel * config.learningRate;
  const clampedDelta = Math.max(-config.maxWeightDelta, Math.min(config.maxWeightDelta, rawDelta));
  synapse.weight = Math.max(
    -config.maxWeightMagnitude,
    Math.min(config.maxWeightMagnitude, synapse.weight + clampedDelta)
  );
}

export function updateActivityHistory(synapse, config) {
  // Running average: did this synapse participate this step?
  const participated = Math.abs(synapse.eligibilityTrace) > 0 ? 1 : 0;
  synapse.activityHistory =
    config.activityHistoryDecay * synapse.activityHistory +
    (1 - config.activityHistoryDecay) * participated;
}

export function decayChemical(synapse, config) {
  synapse.chemicalLevel *= config.chemicalDecayRate;
}
