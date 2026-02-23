// Tier 1: static position, set once at network creation
export function createNeuron(id, x, y, z, type, clusterId) {
  return { id, x, y, z, type, clusterId, neighbourIds: [] };
}

// Tier 2: dynamic state, updated each step
export function createNeuronState(neuronId, initialThreshold) {
  return {
    neuronId,
    output:         0,
    firedThisCycle: false,
    fireCount:      0,
    cycleCount:     0,
    fireRate:       0,
    threshold:      initialThreshold,
    ambientField:   0,
  };
}

export function updateFireRate(state) {
  if (state.cycleCount === 0) return;
  state.fireRate = state.fireCount / state.cycleCount;
}

// Homeostatic plasticity: push threshold toward targetFireRate
export function regulateThreshold(state, config) {
  if (state.fireRate > config.targetFireRate) {
    state.threshold += config.thresholdAdjustRate;
  } else if (state.fireRate < config.targetFireRate) {
    state.threshold -= config.thresholdAdjustRate;
  }
}

// Binary step function: 1 if weighted input meets threshold, else 0
export function computeOutput(neuronId, accumulators, neuronState) {
  const state = neuronState.get(neuronId);
  const input = accumulators.get(neuronId) ?? 0;
  const fired = input >= state.threshold ? 1 : 0;

  state.output         = fired;
  state.firedThisCycle = fired === 1;
  state.cycleCount    += 1;
  if (fired) state.fireCount += 1;

  updateFireRate(state);
}
