// Entry point for experiment-020: cursor-only learning

import { createNetwork } from './network.mjs';
import { train, EPISODES, STEPS_PER_EPISODE } from './train.mjs';
import { generateReport } from './report.mjs';

const config = {
  clusters: 2,
  neuronsPerCluster: 30,
  intraProb: 0.6,
  interProb: 0.5,
  cursorRadius: 4,
  perturbStd: 0.05,
  episodes: EPISODES,
  stepsPerEpisode: STEPS_PER_EPISODE,
  targetFireRate: 0.2,
  thresholdAdjustRate: 0.01,
};

console.log('=== Experiment 020: Cursor-Only Learning ===');
console.log(`Architecture: ${config.clusters} clusters × ${config.neuronsPerCluster} neurons`);
console.log(`Training: ${config.episodes} episodes × ${config.stepsPerEpisode} steps\n`);

const network = createNetwork();
console.log(`Network: ${network.neurons.length} neurons, ${network.synapses.length} synapses\n`);

const startTime = Date.now();
const log = train(network);
const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

console.log(`\nTraining complete in ${elapsed}s`);

generateReport(log, network, config, startTime);
