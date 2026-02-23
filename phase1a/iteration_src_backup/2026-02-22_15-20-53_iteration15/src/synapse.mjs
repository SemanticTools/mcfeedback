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
    // Frustration tracking (exp-011+)
    adjustmentDirection:  0,   // +1 or -1: direction of most recent weight delta
    sameDirectionCount:   0,   // consecutive steps moving in adjustmentDirection
    rewardWhileAdjusting: 0,   // EMA of chemicalLevel while adjusting in current direction
    frustrationFlipCount: 0,   // total flips this synapse has undergone (diagnostic)
  };
}

// Flag strength accumulation.
// If config.consistencyThreshold is set (exp-010+): direction-consistent mode —
// requires N consecutive same-sign traces before flagStrength grows; flips reset
// the streak and apply a sharp flagDecayOnFlip penalty.
// Otherwise (exp-004 base): simple mode — any non-zero trace accumulates flagStrength.
export function updateFlagStrength(synapse, config) {
  const trace = synapse.eligibilityTrace;
  if (config.consistencyThreshold != null) {
    // Direction-consistent mode
    if (trace !== 0) {
      const currentSign = Math.sign(trace);
      if (currentSign === synapse.lastTraceSign) {
        synapse.consecutiveConsistent++;
      } else {
        synapse.consecutiveConsistent = 0;
        synapse.flagStrength *= config.flagDecayOnFlip ?? 0.5;
      }
      synapse.lastTraceSign = currentSign;
      if (synapse.consecutiveConsistent >= config.consistencyThreshold) {
        synapse.flagStrength = Math.min(1.0, synapse.flagStrength + config.flagStrengthGain);
      }
    } else {
      synapse.flagStrength *= config.flagDecayRate;
    }
  } else {
    // Simple mode (exp-004 original): any non-zero trace accumulates
    if (trace !== 0) {
      synapse.flagStrength = Math.min(1.0, synapse.flagStrength + config.flagStrengthGain);
    } else {
      synapse.flagStrength *= config.flagDecayRate;
    }
  }
}

// Returns the clamped weight delta so callers can use it for frustration tracking.
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
  return clampedDelta;
}

// Frustration detection and partial weight flip (exp-011+).
// Called after updateWeight with the delta it returned.
// No-op if config.frustrationWindow is not set.
export function updateFrustration(synapse, config, weightDelta) {
  if (!config.frustrationWindow || weightDelta === 0) return;

  const currentDir = Math.sign(weightDelta);
  if (currentDir === synapse.adjustmentDirection) {
    synapse.sameDirectionCount++;
    synapse.rewardWhileAdjusting =
      0.95 * synapse.rewardWhileAdjusting + 0.05 * synapse.chemicalLevel;
  } else {
    // Direction changed naturally — reset tracking
    synapse.adjustmentDirection  = currentDir;
    synapse.sameDirectionCount   = 1;
    synapse.rewardWhileAdjusting = synapse.chemicalLevel;
  }

  // Frustration check: sustained same-direction movement with persistent negative reward
  if (synapse.sameDirectionCount >= config.frustrationWindow &&
      synapse.rewardWhileAdjusting < config.frustrationThreshold) {
    synapse.weight        = synapse.weight * -1 * config.frustrationFlipStrength;
    synapse.adjustmentDirection  = 0;
    synapse.sameDirectionCount   = 0;
    synapse.rewardWhileAdjusting = 0;
    synapse.flagStrength         = 0;  // re-earn latch in the new direction
    synapse.frustrationFlipCount++;
  }
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
