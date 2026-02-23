import { createNetwork } from '../src/network.mjs';
import { step, evaluate } from '../src/engine.mjs';

const BASE_CONFIG = {
  // Network structure
  clustersCount:               2,
  neuronsPerCluster:           30,
  modulatoryPerCluster:        2,
  intraClusterConnectionProb:  0.6,
  interClusterConnectionProb:  0.5,
  clusterSpacing:              10.0,
  neuronSpread:                2.0,

  // Neuron dynamics
  initialThreshold:    0.5,
  targetFireRate:      0.2,
  thresholdAdjustRate: 0.01,
  ambientRadius:       3.0,
  ambientFalloff:      'inverse',

  // Synapse initialization
  initialWeightRange: [-0.1, 0.1],

  // Flagging
  coActivationStrength: 1.0,
  coSilenceStrength:    0.5,
  mismatchStrength:    -0.5,
  ambientThreshold:     0.3,

  // Dampening
  activityHistoryDecay:   0.95,
  activityHistoryMinimum: 0.1,

  // Chemical / Reward
  chemicalDiffusionRadius: 5.0,
  chemicalFalloff:         'inverse',
  chemicalDecayRate:       0.5,
  positiveRewardStrength:  1.0,
  negativeRewardStrength: -1.0,

  // Learning
  learningRate:       0.01,
  maxWeightDelta:     0.1,
  maxWeightMagnitude: 2.0,
  weightDecay:        0.005,

  // Experiment
  inputSize:        5,
  outputSize:       5,
  trainingEpisodes: 1000,
  reportInterval:   100,
};

const PATTERNS = [
  { input: [1,0,1,0,1], target: [0,1,0,1,0] },
  { input: [1,1,0,0,0], target: [0,0,1,1,1] },
  { input: [1,0,0,0,1], target: [0,1,1,1,0] },
  { input: [0,1,0,1,0], target: [1,0,1,0,1] },
];

function runCondition(label, configOverrides) {
  const config  = { ...BASE_CONFIG, ...configOverrides };
  const network = createNetwork(config);

  let episodesToConverge = null;
  const CONVERGENCE_THRESHOLD = 0.9;

  for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
    const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
    const metrics = step(network, pattern.input, pattern.target);

    if (episodesToConverge === null && metrics.accuracy >= CONVERGENCE_THRESHOLD) {
      episodesToConverge = ep;
    }

    if (ep % config.reportInterval === 0) {
      console.log(
        `  Episode ${String(ep).padStart(4)}/${config.trainingEpisodes}` +
        ` | Acc: ${metrics.accuracy.toFixed(2)}` +
        ` | Loss: ${metrics.loss.toFixed(3)}` +
        ` | MeanW: ${metrics.meanWeight.toFixed(4)}` +
        ` | FireRate: ${metrics.meanFireRate.toFixed(3)}` +
        ` | Threshold: ${metrics.meanThreshold.toFixed(3)}`
      );
    }
  }

  // Final evaluation — frozen weights, no learning, one pass per pattern
  let totalCorrect = 0;
  for (const pattern of PATTERNS) {
    const result = evaluate(network, pattern.input, pattern.target);
    totalCorrect += result.accuracy;
  }
  const finalAccuracy = totalCorrect / PATTERNS.length;

  return { label, finalAccuracy, episodesToConverge };
}

// Four conditions
const conditions = [
  {
    label:   'Baseline',
    overrides: {
      ambientRadius:           0,       // no ambient field
      _skipDampening:          true,    // no dampening filters
      chemicalDiffusionRadius: 1000,    // global reward
      chemicalFalloff:         'constant',
    },
  },
  {
    label:   'Ambient only',
    overrides: {
      _skipDampening:          true,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:   'Dampening only',
    overrides: {
      ambientRadius:           0,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:   'Full model',
    overrides: {
      // Radius must span inter-cluster distance (spacing=10, spread=2 → max ~14)
      chemicalDiffusionRadius: 15,
    },
  },
];

console.log('=== mcfeedback — Experiment 001: Pattern Association ===\n');

const results = [];
for (const cond of conditions) {
  console.log(`--- ${cond.label} ---`);
  const result = runCondition(cond.label, cond.overrides ?? {});
  results.push(result);
  console.log();
}

// Summary table
console.log('=== Results ===');
console.log('Condition       | Final Accuracy | Episodes to 90% | Converged?');
console.log('----------------|----------------|-----------------|----------');
for (const r of results) {
  const acc  = (r.finalAccuracy * 100).toFixed(1).padStart(6) + '%';
  const eps  = r.episodesToConverge !== null ? String(r.episodesToConverge).padStart(7) : '    N/A';
  const conv = r.episodesToConverge !== null ? 'Yes' : 'No';
  console.log(`${r.label.padEnd(16)}| ${acc.padEnd(15)}| ${eps.padEnd(16)}| ${conv}`);
}
