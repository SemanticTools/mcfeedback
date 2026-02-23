// Training loop for Experiments 022–023.
// Dispatches to conditions: 'epsilon', 'baseline', 'astrocyte-traffic', 'astrocyte-firing', 'control'.
// Also retains 'cursor' for backward compatibility.
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
  computeActivationScores, computeTrafficScores,
  selectActiveAstrocytes,
  getEligibleSynapsesFromAstrocytes, adaptAstrocytes,
} from './astrocyte.mjs';
import { saveWeights, perturb, revertWeights } from './perturb.mjs';
import { computeBinaryReward, computeSoftReward } from './reward.mjs';
import { randomPattern, trainingPatterns } from './task.mjs';

export const EPISODES          = 20000;
export const STEPS_PER_EPISODE = 10;

const TRAJECTORY_INTERVAL = 250;
const LOG_INTERVAL        = 500;
// Episodes at which to snapshot per-astrocyte mean activation score
const SCORE_SAMPLE_EPS = new Set([100, 1000, EPISODES - 1]);

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
  if (condition === 'cursor')            return trainCursor(network);
  if (condition === 'epsilon')           return trainAstrocyte(network, 'firing',   true);
  if (condition === 'baseline')          return trainAstrocyte(network, 'firing',   false);
  if (condition === 'astrocyte-firing')  return trainAstrocyte(network, 'firing',   false);
  if (condition === 'astrocyte-traffic') return trainAstrocyte(network, 'traffic',  false);
  // Legacy aliases
  if (condition === 'astrocyte')         return trainAstrocyte(network, 'firing',   false);
  throw new Error(`Unknown condition: ${condition}`);
}

// ─── Cursor condition (iter-4 baseline, retained for compatibility) ───────────

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
// scoringMode: 'firing' | 'traffic'
// useEpsilon: if true, dormant astrocytes have EPSILON chance of activating each step

function trainAstrocyte(network, scoringMode, useEpsilon = false) {
  const numPatterns = trainingPatterns.length;
  const astrocytes  = createAstrocytes(network, numPatterns);

  const coverage = logCoverage(astrocytes, network);
  const label = scoringMode === 'traffic' ? 'traffic' : (useEpsilon ? 'epsilon ' : 'baseline');
  console.log(`  [${label}] coverage: ${coverage.covered}/${coverage.total} neurons ` +
    `(${(coverage.covered / coverage.total * 100).toFixed(0)}%)`);

  const trajectory = [];
  let totalAccepted = 0, totalRejected = 0, totalSteps = 0;
  const weightStart = meanAbsWeight(network);

  const prevFiredState    = new Uint8Array(network.neurons.length);
  const baselineFiredSnap = new Uint8Array(network.neurons.length);

  // Per-astrocyte score tracking: accumulate within each episode, snapshot at checkpoints
  const epScoreSum   = new Float64Array(astrocytes.length);
  const scoreSamples = astrocytes.map(() => ({ ep100: 0, ep1000: 0, epFinal: 0 }));

  for (let episode = 0; episode < EPISODES; episode++) {
    let rewardSum = 0, accepted = 0, rejected = 0, eligibleSum = 0;
    epScoreSum.fill(0);

    for (let step = 0; step < STEPS_PER_EPISODE; step++) {
      const { input, target, patternIdx } = randomPattern();

      if (step === 0) {
        prevFiredState.fill(0);
        for (let i = 0; i < 5; i++) prevFiredState[i] = input[i];
      }

      // 1. SENSE
      const scores = scoringMode === 'traffic'
        ? computeTrafficScores(astrocytes, prevFiredState)
        : computeActivationScores(astrocytes, prevFiredState);

      for (let j = 0; j < astrocytes.length; j++) epScoreSum[j] += scores[j];

      // 2. ACTIVATE
      const active = selectActiveAstrocytes(astrocytes, scores, useEpsilon);
      for (const ast of active) ast.activationsByPattern[patternIdx]++;

      // 3. COLLECT
      const eligible = getEligibleSynapsesFromAstrocytes(active);
      eligibleSum += eligible.length;

      // Baseline forward pass
      const base     = propagate(network, input);
      const baseSoft = computeSoftReward(base.activations, target, network);

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
        for (const n of network.neurons) prevFiredState[n.id] = n.fired ? 1 : 0;
      } else {
        revertWeights(saved);
        rejected++; totalRejected++;
        rewardSum += computeBinaryReward(base.binaryOutput, target);
        prevFiredState.set(baselineFiredSnap);
      }

      // 8. ADAPT
      totalSteps++;
      adaptAstrocytes(astrocytes, active, kept, totalSteps);
    }

    // Snapshot per-astrocyte mean score at checkpoint episodes
    if (SCORE_SAMPLE_EPS.has(episode)) {
      for (let j = 0; j < astrocytes.length; j++) {
        const epMean = epScoreSum[j] / STEPS_PER_EPISODE;
        if (episode === 100)          scoreSamples[j].ep100  = epMean;
        if (episode === 1000)         scoreSamples[j].ep1000 = epMean;
        if (episode === EPISODES - 1) scoreSamples[j].epFinal = epMean;
      }
    }

    const avgReward  = rewardSum / STEPS_PER_EPISODE;
    const acceptRate = accepted / (accepted + rejected || 1);

    if (episode % TRAJECTORY_INTERVAL === 0 || episode === EPISODES - 1)
      trajectory.push({ episode, avgReward, acceptRate });

    if (episode % LOG_INTERVAL === 0 || episode === EPISODES - 1)
      console.log(`  [${label}] ep ${String(episode).padStart(4)}: ` +
        `reward=${avgReward.toFixed(3)}  accept=${(acceptRate * 100).toFixed(1)}%  ` +
        `eligible=${(eligibleSum / STEPS_PER_EPISODE).toFixed(1)}`);
  }

  return {
    trajectory, totalAccepted, totalRejected,
    weightStart, weightEnd: meanAbsWeight(network),
    astrocyteStats: astrocytes.map((ast, j) => ({
      id:                   ast.id,
      cluster:              ast.cluster,
      position:             ast.position,
      activationCount:      ast.activationCount,
      successCount:         ast.successCount,
      epsilonCount:         ast.epsilonCount,
      finalThreshold:       ast.activationThreshold,
      activationsByPattern: ast.activationsByPattern.slice(),
      neuronCount:          ast.neuronIds.length,
      synapseCount:         ast.ownedSynapses.length,
      scoreSamples:         scoreSamples[j],
    })),
  };
}
