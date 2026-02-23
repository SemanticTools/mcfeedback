// Experiment 021 — Astrocyte-Mediated Learning
// Three conditions per seed: astrocyte | cursor (iter-4 baseline) | control (no training)

import { seedRandom }              from './rng.mjs';
import { createNetwork }           from './network.mjs';
import { train, EPISODES, STEPS_PER_EPISODE } from './train.mjs';
import { runInference }            from './inference.mjs';
import { generateReport }          from './report.mjs';
import { trainingPatterns }        from './task.mjs';

const SEEDS = [42, 137, 271, 314, 500, 618, 777, 888, 999, 1234];

const config = {
  experiment: '021',
  clusters: 2,
  neuronsPerCluster: 30,
  intraProb: 0.6,
  interProb: 0.5,
  astrocytesPerCluster: 4,
  territoryRadius: 3.0,
  perturbStd: 0.1,
  cursorRadius: 2,
  episodes: EPISODES,
  stepsPerEpisode: STEPS_PER_EPISODE,
  homeostasis: false,
  seeds: SEEDS,
  numPatterns: trainingPatterns.length,
};

console.log('=== Experiment 021: Astrocyte-Mediated Learning ===');
console.log(`Architecture: ${config.clusters} clusters × ${config.neuronsPerCluster} neurons`);
console.log(`Training: ${config.episodes} ep × ${config.stepsPerEpisode} steps`);
console.log(`Conditions: astrocyte | cursor | control`);
console.log(`Seeds: ${SEEDS.join(', ')}\n`);

const startTime = Date.now();
const allResults = [];

for (const seed of SEEDS) {
  console.log(`\n[Seed ${seed}]`);

  // ── Astrocyte condition ──────────────────────────────────────────────────
  seedRandom(seed);
  const astNet = createNetwork();
  console.log(`  Training astrocyte...`);
  const astResult = train(astNet, 'astrocyte');
  const astInf    = runInference(astNet);

  // ── Cursor condition ─────────────────────────────────────────────────────
  seedRandom(seed);
  const cursorNet = createNetwork();
  console.log(`  Training cursor...`);
  const cursorResult = train(cursorNet, 'cursor');
  const cursorInf    = runInference(cursorNet);

  // ── Control condition (no training) ──────────────────────────────────────
  seedRandom(seed);
  const controlNet  = createNetwork();
  const controlResult = train(controlNet, 'control');
  const controlInf    = runInference(controlNet);

  console.log(
    `  astrocyte acc=${astInf.meanAccuracy.toFixed(3)}  distinct=${astInf.distinctOutputs}  ` +
    `accepted=${astResult.totalAccepted}  reverted=${astResult.totalRejected}`
  );
  console.log(
    `  cursor    acc=${cursorInf.meanAccuracy.toFixed(3)}  distinct=${cursorInf.distinctOutputs}  ` +
    `accepted=${cursorResult.totalAccepted}  reverted=${cursorResult.totalRejected}`
  );
  console.log(
    `  control   acc=${controlInf.meanAccuracy.toFixed(3)}  distinct=${controlInf.distinctOutputs}`
  );

  allResults.push({
    seed,
    astrocyte: { ...astResult,    inference: astInf },
    cursor:    { ...cursorResult, inference: cursorInf },
    control:   { ...controlResult, inference: controlInf },
  });
}

const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(`\nTotal time: ${elapsed}s`);

generateReport(allResults, config, startTime);
