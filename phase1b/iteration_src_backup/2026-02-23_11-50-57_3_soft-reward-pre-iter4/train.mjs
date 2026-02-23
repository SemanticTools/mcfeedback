// Training loop: cursor-based perturbation (hill-climbing inside the network)
//
// Per step:
//   1. Present a random training pattern
//   2. Propagate → baseline output & reward
//   3. Move cursor; find eligible synapses within radius
//   4. Save weights, perturb eligible synapses
//   5. Propagate again → new reward
//   6. Keep perturbation if reward improved, revert otherwise
//
// After every episode: run homeostasis (outside all loops)

import { propagate } from './propagate.mjs';
import { runHomeostasis } from './homeostasis.mjs';
import { createCursor, moveCursor, getEligibleSynapses } from './cursor.mjs';
import { saveWeights, perturb, revertWeights } from './perturb.mjs';
import { computeBinaryReward, computeSoftReward } from './reward.mjs';
import { randomPattern } from './task.mjs';

export const EPISODES = 500;
export const STEPS_PER_EPISODE = 50;

export function train(network) {
  const cursor = createCursor(network);
  const log = [];

  for (let episode = 0; episode < EPISODES; episode++) {
    let rewardSum = 0;
    let accepted = 0;
    let rejected = 0;
    let eligibleSum = 0;

    for (let step = 0; step < STEPS_PER_EPISODE; step++) {
      const { input, target } = randomPattern();

      // Baseline evaluation
      const base = propagate(network, input);
      const baseSoftReward = computeSoftReward(base.activations, target, network);

      // Cursor step
      moveCursor(cursor);
      const eligible = getEligibleSynapses(cursor, network);
      eligibleSum += eligible.length;

      if (eligible.length === 0) {
        rewardSum += computeBinaryReward(base.binaryOutput, target);
        continue;
      }

      // Perturb and evaluate
      const saved = saveWeights(eligible);
      perturb(eligible);

      const perturbed = propagate(network, input);
      const newSoftReward = computeSoftReward(perturbed.activations, target, network);

      if (newSoftReward > baseSoftReward) {
        accepted++;
        rewardSum += computeBinaryReward(perturbed.binaryOutput, target);
      } else {
        revertWeights(saved);
        rejected++;
        rewardSum += computeBinaryReward(base.binaryOutput, target);
      }
    }

    // Homeostasis: runs once per episode, outside all step loops
    runHomeostasis(network);

    const entry = {
      episode,
      avgReward: rewardSum / STEPS_PER_EPISODE,
      accepted,
      rejected,
      avgEligible: eligibleSum / STEPS_PER_EPISODE,
    };
    log.push(entry);

    if (episode % 50 === 0 || episode === EPISODES - 1) {
      const pct = ((accepted / (accepted + rejected || 1)) * 100).toFixed(1);
      console.log(
        `Episode ${String(episode).padStart(3)}: ` +
        `reward=${entry.avgReward.toFixed(3)}  ` +
        `accept=${pct}%  ` +
        `eligible=${entry.avgEligible.toFixed(1)}`
      );
    }
  }

  return log;
}
