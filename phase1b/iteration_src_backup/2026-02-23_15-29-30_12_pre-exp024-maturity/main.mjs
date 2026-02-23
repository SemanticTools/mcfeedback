// Experiment 023 — Epsilon-Exploration Astrocytes
// Three conditions per seed: epsilon | baseline | control
//
// epsilon:  firing-based scoring + epsilon exploration (EPSILON chance per astrocyte per step)
// baseline: firing-based scoring, no epsilon (exp021 mechanism)
// control:  no training, random init, fixed thresholds

import { seedRandom }              from './rng.mjs';
import { createNetwork }           from './network.mjs';
import { train, EPISODES, STEPS_PER_EPISODE } from './train.mjs';
import { runInference }            from './inference.mjs';
import { generateReport }          from './report.mjs';
import { trainingPatterns }        from './task.mjs';
import { EPSILON }                 from './astrocyte.mjs';

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
  episodes: EPISODES,
  stepsPerEpisode: STEPS_PER_EPISODE,
  homeostasis: false,
  seeds: SEEDS,
  numPatterns: trainingPatterns.length,
};

console.log('=== Experiment 024: Epsilon-Exploration Astrocytes (20k episodes) ===');
console.log(`Architecture: ${config.clusters} clusters × ${config.neuronsPerCluster} neurons`);
console.log(`Training: ${config.episodes} ep × ${config.stepsPerEpisode} steps`);
console.log(`Epsilon: ${(EPSILON * 100).toFixed(0)}% per astrocyte per step`);
console.log(`Conditions: epsilon | baseline | control`);
console.log(`Seeds: ${SEEDS.join(', ')}\n`);

const startTime = Date.now();
const allResults = [];

for (const seed of SEEDS) {
  console.log(`\n[Seed ${seed}]`);

  // ── Epsilon condition (firing scoring + epsilon exploration) ──────────────
  seedRandom(seed);
  const epsilonNet = createNetwork();
  console.log(`  Training epsilon...`);
  const epsilonResult = train(epsilonNet, 'epsilon');
  const epsilonInf    = runInference(epsilonNet);

  // ── Baseline condition (firing scoring only, no epsilon — exp021) ─────────
  seedRandom(seed);
  const baselineNet = createNetwork();
  console.log(`  Training baseline...`);
  const baselineResult = train(baselineNet, 'baseline');
  const baselineInf    = runInference(baselineNet);

  // ── Control condition (no training) ──────────────────────────────────────
  seedRandom(seed);
  const controlNet    = createNetwork();
  const controlResult = train(controlNet, 'control');
  const controlInf    = runInference(controlNet);

  console.log(
    `  epsilon  acc=${epsilonInf.meanAccuracy.toFixed(3)}  distinct=${epsilonInf.distinctOutputs}  ` +
    `accepted=${epsilonResult.totalAccepted}  reverted=${epsilonResult.totalRejected}`
  );
  console.log(
    `  baseline acc=${baselineInf.meanAccuracy.toFixed(3)}  distinct=${baselineInf.distinctOutputs}  ` +
    `accepted=${baselineResult.totalAccepted}  reverted=${baselineResult.totalRejected}`
  );
  console.log(
    `  control  acc=${controlInf.meanAccuracy.toFixed(3)}  distinct=${controlInf.distinctOutputs}`
  );

  allResults.push({
    seed,
    epsilon:  { ...epsilonResult,  inference: epsilonInf },
    baseline: { ...baselineResult, inference: baselineInf },
    control:  { ...controlResult,  inference: controlInf },
  });
}

const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(`\nTotal time: ${elapsed}s`);

generateReport(allResults, config, startTime);
