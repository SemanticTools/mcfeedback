// Four-quadrant eligibility flagging.
// All inputs are local: pre-state, post-state, config.
// Returns a raw trace value (before dampening).

export function computeEligibility(preState, postState, config) {
  const preFired  = preState.firedThisCycle;
  const postFired = postState.firedThisCycle;

  if (preFired && postFired) {
    // Co-activation: Hebbian
    return config.coActivationStrength;
  }

  if (!preFired && !postFired) {
    // Co-silence: only meaningful in an active neighbourhood
    if (postState.ambientField > config.ambientThreshold) {
      return config.coSilenceStrength;
    }
    return 0;
  }

  // Mismatch (one fires, the other doesn't)
  return config.mismatchStrength;
}
