// Training loop for Experiment 021.
// Dispatches to three conditions: 'astrocyte', 'cursor', 'control'.
//
// Return shape (all conditions):
//   { trajectory, totalAccepted, totalRejected, weightStart, weightEnd, astrocyteStats }
//
// astrocyteStats is null for cursor/control.
// trajectory is [] for control (no training).

import { propagate } from './propagate.mjs';
import { createCursor, moveCursor, getEligibleSynapses } from './cursor.mjs';
import {
  createAstrocytes, logCoverage,
  computeActivationScores, selectActiveAstrocytes,
  getEligibleSynapsesFromAstrocytes, adaptAstrocytes,
} from './astrocyte.mjs';
import { saveWeights, perturb, revertWeights } from './perturb.mjs';
import { computeBinaryReward, computeSoftReward } from './reward.mjs';
import { randomPattern, trainingPatterns } from './task.mjs';

export const EPISODES          = 5000;
export const STEPS_PER_EPISODE = 10;

const TRAJECTORY_INTERVAL = 250;
const LOG_INTERVAL        = 500;

function meanAbsWeight(network) {
  return network.synapses.reduce((s, syn) => s + Math.abs(syn.weight), 0) / network.synapses.length;
}

// ─── Public dispatcher ────────────────────────────────────────────────────────

export function train(network, condition) {
  if (condition === 'control') {
    const w = meanAbsWeight(network);
    return { trajectory: [], totalAccepted: 0, totalRejected: 0,
             weightStart: w, weightEnd: w, astrocyteStats: null };
  }
  if (condition === 'cursor')    return trainCursor(network);
  if (condition === 'astrocyte') return trainAstrocyte(network);
  throw new Error(`Unknown condition: ${condition}`);
}

// ─── Cursor condition (iteration 4 baseline) ─────────────────────────────────

function trainCursor(network) {
  const cursor = createCursor(network);
  const trajectory = [];
  let totalAccepted = 0, totalRejected = 0;
  const weightStart = meanAbsWeight(network);

  for (let episode = 0; episode < EPISODES; episode++) {
    let rewardSum = 0, accepted = 0, rejected = 0, eligibleSum = 0;

    for (let step = 0; step < STEPS_PER_EPISODE; step++) {
      const { input, target } = randomPattern();

      const base     = propagate(network, input);
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

      const after     = propagate(network, input);
      const afterSoft = computeSoftReward(after.activations, target, network);

      if (afterSoft > baseSoft) {
        accepted++; totalAccepted++;
        rewardSum += computeBinaryReward(after.binaryOutput, target);
      } else {
        revertWeights(saved);
        rejected++; totalRejected++;
        rewardSum += computeBinaryReward(base.binaryOutput, target);
      }
    }

    const avgReward  = rewardSum / STEPS_PER_EPISODE;
    const acceptRate = accepted / (accepted + rejected || 1);

    if (episode % TRAJECTORY_INTERVAL === 0 || episode === EPISODES - 1)
      trajectory.push({ episode, avgReward, acceptRate });

    if (episode % LOG_INTERVAL === 0 || episode === EPISODES - 1)
      console.log(`  [cursor] ep ${String(episode).padStart(4)}: ` +
        `reward=${avgReward.toFixed(3)}  accept=${(acceptRate * 100).toFixed(1)}%  ` +
        `eligible=${(eligibleSum / STEPS_PER_EPISODE).toFixed(1)}`);
  }

  return {
    trajectory, totalAccepted, totalRejected,
    weightStart, weightEnd: meanAbsWeight(network),
    astrocyteStats: null,
  };
}

// ─── Astrocyte condition ──────────────────────────────────────────────────────

function trainAstrocyte(network) {
  const numPatterns = trainingPatterns.length;
  const astrocytes  = createAstrocytes(network, numPatterns);

  const coverage = logCoverage(astrocytes, network);
  console.log(`  [astro] coverage: ${coverage.covered}/${coverage.total} neurons ` +
    `(${(coverage.covered / coverage.total * 100).toFixed(0)}%)`);

  const trajectory = [];
  let totalAccepted = 0, totalRejected = 0, totalSteps = 0;
  const weightStart = meanAbsWeight(network);

  // prevFiredState: neuron id → fired (1/0) on previous step
  const prevFiredState   = new Uint8Array(network.neurons.length);
  const baselineFiredSnap = new Uint8Array(network.neurons.length);

  for (let episode = 0; episode < EPISODES; episode++) {
    let rewardSum = 0, accepted = 0, rejected = 0, eligibleSum = 0;

    for (let step = 0; step < STEPS_PER_EPISODE; step++) {
      const { input, target, patternIdx } = randomPattern();

      // Step 0: seed prevFiredState from input clamping (only input neurons fire)
      if (step === 0) {
        prevFiredState.fill(0);
        for (let i = 0; i < 5; i++) prevFiredState[i] = input[i];
      }

      // 1. SENSE
      const scores = computeActivationScores(astrocytes, prevFiredState);

      // 2. ACTIVATE
      const active = selectActiveAstrocytes(astrocytes, scores);
      for (const ast of active) ast.activationsByPattern[patternIdx]++;

      // 3. COLLECT
      const eligible = getEligibleSynapsesFromAstrocytes(active);
      eligibleSum += eligible.length;

      // Baseline forward pass
      const base     = propagate(network, input);
      const baseSoft = computeSoftReward(base.activations, target, network);

      // Snapshot baseline fired state (for prevFiredState update if reverted)
      for (const n of network.neurons) baselineFiredSnap[n.id] = n.fired ? 1 : 0;

      // 4. SNAPSHOT  5. PERTURB
      const saved = saveWeights(eligible);
      perturb(eligible);

      // 6. EVALUATE
      const after     = propagate(network, input);
      const afterSoft = computeSoftReward(after.activations, target, network);

      // 7. DECIDE
      const kept = afterSoft > baseSoft;
      if (kept) {
        accepted++; totalAccepted++;
        rewardSum += computeBinaryReward(after.binaryOutput, target);
        // prevFiredState = perturbed network's firing (current network.neurons[*].fired)
        for (const n of network.neurons) prevFiredState[n.id] = n.fired ? 1 : 0;
      } else {
        revertWeights(saved);
        rejected++; totalRejected++;
        rewardSum += computeBinaryReward(base.binaryOutput, target);
        // prevFiredState = baseline firing (captured before perturbation)
        prevFiredState.set(baselineFiredSnap);
      }

      // 8. ADAPT
      totalSteps++;
      adaptAstrocytes(astrocytes, active, kept, totalSteps);
    }

    const avgReward  = rewardSum / STEPS_PER_EPISODE;
    const acceptRate = accepted / (accepted + rejected || 1);

    if (episode % TRAJECTORY_INTERVAL === 0 || episode === EPISODES - 1)
      trajectory.push({ episode, avgReward, acceptRate });

    if (episode % LOG_INTERVAL === 0 || episode === EPISODES - 1)
      console.log(`  [astro] ep ${String(episode).padStart(4)}: ` +
        `reward=${avgReward.toFixed(3)}  accept=${(acceptRate * 100).toFixed(1)}%  ` +
        `eligible=${(eligibleSum / STEPS_PER_EPISODE).toFixed(1)}`);
  }

  return {
    trajectory, totalAccepted, totalRejected,
    weightStart, weightEnd: meanAbsWeight(network),
    astrocyteStats: astrocytes.map(ast => ({
      id:                  ast.id,
      cluster:             ast.cluster,
      position:            ast.position,
      activationCount:     ast.activationCount,
      successCount:        ast.successCount,
      finalThreshold:      ast.activationThreshold,
      activationsByPattern: ast.activationsByPattern.slice(),
      neuronCount:         ast.neuronIds.length,
      synapseCount:        ast.ownedSynapses.length,
    })),
  };
}
