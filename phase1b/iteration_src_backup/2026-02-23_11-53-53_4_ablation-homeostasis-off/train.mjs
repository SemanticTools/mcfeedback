// Training loop: cursor-based perturbation, homeostasis OFF.
//
// Per step:
//   1. Present a random training pattern
//   2. Propagate → baseline activations & soft reward
//   3. Move cursor; find eligible synapses within radius
//   4. Save weights, perturb eligible synapses
//   5. Propagate again → new soft reward
//   6. Keep if strictly better (soft reward), revert otherwise
//
// Homeostasis is intentionally absent for this ablation.
// Thresholds remain at their initial value (0.5) for the entire run.

import { propagate } from './propagate.mjs';
import { createCursor, moveCursor, getEligibleSynapses } from './cursor.mjs';
import { saveWeights, perturb, revertWeights } from './perturb.mjs';
import { computeBinaryReward, computeSoftReward } from './reward.mjs';
import { randomPattern } from './task.mjs';

export const EPISODES = 5000;
export const STEPS_PER_EPISODE = 10;

const TRAJECTORY_INTERVAL = 250; // record a trajectory point every N episodes
const LOG_INTERVAL = 500;        // console print every N episodes

function meanAbsWeight(network) {
  return network.synapses.reduce((s, syn) => s + Math.abs(syn.weight), 0) / network.synapses.length;
}

export function train(network) {
  const cursor = createCursor(network);

  let totalAccepted = 0;
  let totalRejected = 0;
  const weightStart = meanAbsWeight(network);
  const trajectory = []; // { episode, avgReward, acceptRate }

  for (let episode = 0; episode < EPISODES; episode++) {
    let rewardSum = 0;
    let accepted = 0;
    let rejected = 0;
    let eligibleSum = 0;

    for (let step = 0; step < STEPS_PER_EPISODE; step++) {
      const { input, target } = randomPattern();

      const base = propagate(network, input);
      const baseSoft = computeSoftReward(base.activations, target, network);

      moveCursor(cursor);
      const eligible = getEligibleSynapses(cursor, network);
      eligibleSum += eligible.length;

      if (eligible.length === 0) {
        rewardSum += computeBinaryReward(base.binaryOutput, target);
        continue;
      }

      const saved = saveWeights(eligible);
      perturb(eligible);

      const after = propagate(network, input);
      const afterSoft = computeSoftReward(after.activations, target, network);

      if (afterSoft > baseSoft) {
        accepted++;
        totalAccepted++;
        rewardSum += computeBinaryReward(after.binaryOutput, target);
      } else {
        revertWeights(saved);
        rejected++;
        totalRejected++;
        rewardSum += computeBinaryReward(base.binaryOutput, target);
      }
    }

    const avgReward = rewardSum / STEPS_PER_EPISODE;
    const acceptRate = accepted / (accepted + rejected || 1);

    if (episode % TRAJECTORY_INTERVAL === 0 || episode === EPISODES - 1) {
      trajectory.push({ episode, avgReward, acceptRate });
    }

    if (episode % LOG_INTERVAL === 0 || episode === EPISODES - 1) {
      console.log(
        `  ep ${String(episode).padStart(4)}: ` +
        `reward=${avgReward.toFixed(3)}  ` +
        `accept=${(acceptRate * 100).toFixed(1)}%  ` +
        `eligible=${(eligibleSum / STEPS_PER_EPISODE).toFixed(1)}`
      );
    }
  }

  const weightEnd = meanAbsWeight(network);

  return { trajectory, totalAccepted, totalRejected, weightStart, weightEnd };
}
