// All dampening functions return a multiplier in [0, 1].
// Each takes only local information.

// Synapses that rarely participate contribute noise — dampen them.
export function activityHistoryDampening(activityHistory, config) {
  if (activityHistory < config.activityHistoryMinimum) {
    return activityHistory / config.activityHistoryMinimum;
  }
  return 1;
}

// Inverted-U on fire rate: always-on and always-off neurons carry no information.
// Peak (1.0) at fireRate = 0.5; approaches 0 at extremes.
export function informationDampening(fireRate) {
  return 4 * fireRate * (1 - fireRate);
}

// Silence is only relevant when the neighbourhood is active.
// Activity is dampened when the neighbourhood is quiet.
export function ambientRelevanceDampening(ambientField, neuronOutput, config) {
  if (neuronOutput === 1) {
    // Firing neuron: relevant in an active field, weaker in a quiet one
    return ambientField > config.ambientThreshold ? 1 : 0.5;
  } else {
    // Silent neuron: relevant only if something is happening nearby
    return ambientField > config.ambientThreshold ? 1 : 0;
  }
}

export function combinedDampening(synapse, preState, postState, config) {
  const history  = activityHistoryDampening(synapse.activityHistory, config);
  const info     = informationDampening(postState.fireRate);
  const ambient  = ambientRelevanceDampening(postState.ambientField, postState.output, config);
  return history * info * ambient;
}
