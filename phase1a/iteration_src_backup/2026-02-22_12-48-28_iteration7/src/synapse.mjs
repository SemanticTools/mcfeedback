export function createSynapse(fromId, toId, initialWeight) {
  return {
    from:             fromId,
    to:               toId,
    weight:           initialWeight,
    eligibilityTrace: 0,
    activityHistory:  0,
    chemicalLevel:    0,
    flagStrength:     0,   // accumulated flag signal across consecutive turns
  };
}

// Accumulate flag strength over consecutive same-direction turns.
// Called after dampening, before weight update.
// Only active when config.flagStrengthThreshold > 0.
export function updateFlagStrength(synapse, config) {
  const trace = synapse.eligibilityTrace;
  if (trace === 0) {
    // No flag this cycle: decay toward zero
    synapse.flagStrength *= config.flagDecayRate;
  } else if (Math.sign(trace) === Math.sign(synapse.flagStrength) || synapse.flagStrength === 0) {
    // Same direction: step forward, clamp to ±1
    synapse.flagStrength = Math.max(-1, Math.min(1,
      synapse.flagStrength + Math.sign(trace) * config.flagStrengthGain
    ));
  } else {
    // Direction flip: wipe and start fresh
    synapse.flagStrength = Math.sign(trace) * config.flagStrengthGain;
  }
}

export function updateWeight(synapse, config) {
  // Decay first: weights must be continuously earned
  synapse.weight *= (1 - config.weightDecay);

  // If flagStrengthThreshold is set, gate learning on accumulated flag strength;
  // otherwise fall back to raw eligibilityTrace (backward-compatible).
  let trace = synapse.eligibilityTrace;
  if (config.flagStrengthThreshold > 0) {
    trace = Math.abs(synapse.flagStrength) >= config.flagStrengthThreshold
      ? synapse.flagStrength
      : 0;
  }

  const rawDelta = trace * synapse.chemicalLevel * config.learningRate;
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
