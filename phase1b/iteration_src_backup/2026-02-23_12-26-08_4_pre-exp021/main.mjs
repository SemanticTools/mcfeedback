// Experiment 020 — Iteration 4: Cursor Ablation (Homeostasis OFF)
//
// Two conditions per seed:
//   cursor  — cursor perturbation, soft-reward acceptance, fixed thresholds
//   control — same initial network, zero training, fixed thresholds
//
// 10 seeds for statistical coverage.

import { seedRandom } from './rng.mjs';
import { createNetwork } from './network.mjs';
import { train, EPISODES, STEPS_PER_EPISODE } from './train.mjs';
import { runInference } from './inference.mjs';
import { generateReport } from './report.mjs';

const SEEDS = [42, 137, 271, 314, 500, 618, 777, 888, 999, 1234];

const config = {
  clusters: 2,
  neuronsPerCluster: 30,
  intraProb: 0.6,
  interProb: 0.5,
  cursorRadius: 2,
  perturbStd: 0.1,
  episodes: EPISODES,
  stepsPerEpisode: STEPS_PER_EPISODE,
  homeostasis: false,
  seeds: SEEDS,
};

console.log('=== Experiment 020 — Iteration 4: Cursor Ablation (Homeostasis OFF) ===');
console.log(`Architecture: ${config.clusters} clusters × ${config.neuronsPerCluster} neurons`);
console.log(`Training: ${config.episodes} ep × ${config.stepsPerEpisode} steps`);
console.log(`Seeds: ${SEEDS.join(', ')}\n`);

const startTime = Date.now();
const allResults = [];

for (const seed of SEEDS) {
  console.log(`\n[Seed ${seed}]`);

  // ---- Cursor condition ----
  seedRandom(seed);
  const cursorNet = createNetwork();
  console.log(`  Training cursor condition...`);
  const trainResult = train(cursorNet);
  const cursorInf = runInference(cursorNet);

  // ---- Control condition (same seed = identical initial network, no training) ----
  seedRandom(seed);
  const controlNet = createNetwork();
  const controlInf = runInference(controlNet);

  console.log(
    `  cursor  acc=${cursorInf.meanAccuracy.toFixed(3)}  distinct=${cursorInf.distinctOutputs}` +
    `  accepted=${trainResult.totalAccepted}  reverted=${trainResult.totalRejected}`
  );
  console.log(
    `  control acc=${controlInf.meanAccuracy.toFixed(3)}  distinct=${controlInf.distinctOutputs}`
  );

  allResults.push({
    seed,
    cursor: { ...trainResult, inference: cursorInf },
    control: { inference: controlInf },
  });
}

const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(`\nTotal time: ${elapsed}s`);

generateReport(allResults, config, startTime);
