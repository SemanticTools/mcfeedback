// Homeostasis: adjust thresholds toward target fire rate.
// Runs once per episode, outside any step loops.
// Resets fire rate counters after adjustment.

const TARGET_FIRE_RATE = 0.2;
const THRESHOLD_ADJUST_RATE = 0.01;

export function runHomeostasis(network) {
  for (const n of network.neurons) {
    if (n.type === 'input') continue;
    if (n.stepCount === 0) continue;

    const fireRate = n.fireCount / n.stepCount;
    // Too active → raise threshold; too quiet → lower it
    n.threshold += THRESHOLD_ADJUST_RATE * (fireRate - TARGET_FIRE_RATE);

    // Reset counters for next episode
    n.fireCount = 0;
    n.stepCount = 0;
  }
}
