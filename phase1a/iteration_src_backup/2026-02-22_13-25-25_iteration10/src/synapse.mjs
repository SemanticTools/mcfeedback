export function createSynapse(fromId, toId, initialWeight) {
  return {
    from:                 fromId,
    to:                   toId,
    weight:               initialWeight,
    eligibilityTrace:     0,
    activityHistory:      0,
    chemicalLevel:        0,
    flagStrength:         0,   // accumulated flag signal
    lastTraceSign:        0,   // sign of last non-zero trace (+1, -1, or 0 = unseen)
    consecutiveConsistent: 0,  // how many consecutive same-sign traces in a row
  };
}

// Direction-consistent flag accumulation.
// A synapse must show consecutiveConsistent >= consistencyThreshold same-sign
// traces before its flagStrength begins to grow. A direction flip resets the
// counter and sharply penalises flagStrength. Silence (trace === 0) is neutral:
// it does not reset the counter or lastTraceSign — only an active contradiction does.
export function updateFlagStrength(synapse, config) {
  const trace = synapse.eligibilityTrace;
  if (trace !== 0) {
    const currentSign = Math.sign(trace);
    if (currentSign === synapse.lastTraceSign) {
      synapse.consecutiveConsistent++;
    } else {
      // Active direction flip: reset streak, apply sharp decay penalty
      synapse.consecutiveConsistent = 0;
      synapse.flagStrength *= config.flagDecayOnFlip ?? 0.5;
    }
    synapse.lastTraceSign = currentSign;

    if (synapse.consecutiveConsistent >= (config.consistencyThreshold ?? 5)) {
      synapse.flagStrength = Math.min(1.0, synapse.flagStrength + config.flagStrengthGain);
    }
  } else {
    // No trace this step: passive decay only — streak and sign are preserved
    synapse.flagStrength *= config.flagDecayRate;
  }
}

export function updateWeight(synapse, config, episode = 0) {
  // Decay first: weights must be continuously earned
  synapse.weight *= (1 - config.weightDecay);

  // If flagStrengthThreshold is set, gate learning on accumulated flag strength.
  // During flagGateWarmup episodes the gate is bypassed so early traces can drive
  // weight updates before flags have had time to accumulate. Backward-compatible:
  // if neither param is set, falls back to raw eligibilityTrace.
  let trace = synapse.eligibilityTrace;
  if (config.flagStrengthThreshold > 0) {
    const inWarmup = config.flagGateWarmup != null && episode < config.flagGateWarmup;
    if (!inWarmup) {
      trace = Math.abs(synapse.flagStrength) >= config.flagStrengthThreshold
        ? synapse.flagStrength
        : 0;
    }
    // inWarmup: keep trace = eligibilityTrace — gate is open for all non-zero traces
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
