// Experiment 024 — Maturity-Scaled Exploration
// Three conditions per seed: maturity | epsilon (flat) | control
//
// maturity: per-astrocyte exploration rate decays with activation count
//           explorationRate = 0.30 / (1 + activationCount / 2000)
// epsilon:  flat 1% exploration (experiment 023 baseline)
// control:  no training, random init, fixed thresholds

import { seedRandom }              from './rng.mjs';
import { createNetwork }           from './network.mjs';
import { train, EPISODES, STEPS_PER_EPISODE } from './train.mjs';
import { runInference }            from './inference.mjs';
import { generateReport }          from './report.mjs';
import { trainingPatterns }        from './task.mjs';
import { EPSILON, BASE_EPSILON, MATURITY_HORIZON } from './astrocyte.mjs';

const SEEDS = [42, 137, 271, 314, 500, 618, 777, 888, 999, 1234];

const config = {
  experiment: '024',
  clusters: 2,
  neuronsPerCluster: 30,
  intraProb: 0.6,
  interProb: 0.5,
  astrocytesPerCluster: 4,
  territoryRadius: 3.0,
  perturbStd: 0.1,
  epsilon: EPSILON,
  baseEpsilon: BASE_EPSILON,
  maturityHorizon: MATURITY_HORIZON,
  episodes: EPISODES,
  stepsPerEpisode: STEPS_PER_EPISODE,
  homeostasis: false,
  seeds: SEEDS,
  numPatterns: trainingPatterns.length,
};

console.log('=== Experiment 024: Maturity-Scaled Exploration ===');
console.log(`Architecture: ${config.clusters} clusters × ${config.neuronsPerCluster} neurons`);
console.log(`Training: ${config.episodes} ep × ${config.stepsPerEpisode} steps`);
console.log(`Maturity: baseEpsilon=${BASE_EPSILON}, horizon=${MATURITY_HORIZON}`);
console.log(`Flat epsilon: ${(EPSILON * 100).toFixed(0)}% per astrocyte per step`);
console.log(`Conditions: maturity | epsilon | control`);
console.log(`Seeds: ${SEEDS.join(', ')}\n`);

const startTime = Date.now();
const allResults = [];

for (const seed of SEEDS) {
  console.log(`\n[Seed ${seed}]`);

  // ── Maturity condition (firing scoring + maturity-scaled exploration) ─────
  seedRandom(seed);
  const maturityNet = createNetwork();
  console.log(`  Training maturity...`);
  const maturityResult = train(maturityNet, 'maturity');
  const maturityInf    = runInference(maturityNet);

  // ── Epsilon condition (firing scoring + flat epsilon — exp023 baseline) ───
  seedRandom(seed);
  const epsilonNet = createNetwork();
  console.log(`  Training epsilon...`);
  const epsilonResult = train(epsilonNet, 'epsilon');
  const epsilonInf    = runInference(epsilonNet);

  // ── Control condition (no training) ──────────────────────────────────────
  seedRandom(seed);
  const controlNet    = createNetwork();
  const controlResult = train(controlNet, 'control');
  const controlInf    = runInference(controlNet);

  console.log(
    `  maturity acc=${maturityInf.meanAccuracy.toFixed(3)}  distinct=${maturityInf.distinctOutputs}  ` +
    `accepted=${maturityResult.totalAccepted}  reverted=${maturityResult.totalRejected}`
  );
  console.log(
    `  epsilon  acc=${epsilonInf.meanAccuracy.toFixed(3)}  distinct=${epsilonInf.distinctOutputs}  ` +
    `accepted=${epsilonResult.totalAccepted}  reverted=${epsilonResult.totalRejected}`
  );
  console.log(
    `  control  acc=${controlInf.meanAccuracy.toFixed(3)}  distinct=${controlInf.distinctOutputs}`
  );

  allResults.push({
    seed,
    maturity: { ...maturityResult, inference: maturityInf },
    epsilon:  { ...epsilonResult,  inference: epsilonInf },
    control:  { ...controlResult,  inference: controlInf },
  });
}

const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(`\nTotal time: ${elapsed}s`);

generateReport(allResults, config, startTime);
