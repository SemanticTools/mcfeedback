/**
 * Experiment 011: Synapse frustration detection and direction flip.
 * Base: experiment-004 (flag gate + linear reward, original flag gain/threshold).
 * propagationCycles=1, direction-consistent flags NOT carried over.
 *
 * Problem: synapses get stuck adjusting indefinitely in one direction while
 * receiving persistent negative chemical signal. They have no mechanism to
 * detect "I've been trying this for a long time and it keeps failing."
 *
 * Fix: per-synapse frustration tracking. Each synapse monitors:
 *   adjustmentDirection   — sign of the most recent non-zero weight delta
 *   sameDirectionCount    — consecutive steps moving that direction
 *   rewardWhileAdjusting  — EMA(0.95) of chemicalLevel while adjusting
 *
 * If sameDirectionCount >= frustrationWindow AND rewardWhileAdjusting < frustrationThreshold,
 * the synapse is frustrated: partially reverse its weight and reset all tracking.
 *   weight = weight * -1 * frustrationFlipStrength  (partial, not full inversion)
 *   flagStrength = 0  (must re-earn latch in the new direction)
 *
 * frustrationWindow:       30   — consecutive same-direction steps before check
 * frustrationThreshold:   -0.1  — minimum rewardWhileAdjusting to avoid flip
 * frustrationFlipStrength: 0.5  — weight * -0.5 on flip (e.g. +0.8 → -0.4)
 *
 * Diagnostic: at ep 1000 per seed, report total flip count and mean |weight|
 * of flipped vs non-flipped synapses in the Full model condition.
 */

import { createNetwork } from '../src/network.mjs';
import { step, evaluate } from '../src/engine.mjs';

// ─── Seedable PRNG (mulberry32) ───────────────────────────────────────────
function makePRNG(seed) {
  let s = seed >>> 0;
  return function () {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function withSeed(seed, fn) {
  const orig = Math.random;
  Math.random = makePRNG(seed);
  try { return fn(); } finally { Math.random = orig; }
}

// ─── Config ───────────────────────────────────────────────────────────────
const BASE_CONFIG = {
  clustersCount:               2,
  neuronsPerCluster:           30,
  modulatoryPerCluster:        2,
  intraClusterConnectionProb:  0.6,
  interClusterConnectionProb:  0.5,
  clusterSpacing:              10.0,
  neuronSpread:                2.0,
  initialThreshold:            0.5,
  targetFireRate:              0.2,
  thresholdAdjustRate:         0.01,
  ambientRadius:               3.0,
  ambientFalloff:              'inverse',
  initialWeightRange:          [-0.1, 0.1],
  coActivationStrength:        1.0,
  coSilenceStrength:           0.5,
  mismatchStrength:           -0.5,
  flagStrengthGain:            0.3,   // per-turn step toward ±1 (exp-004 original)
  flagDecayRate:               0.7,   // multiplier when trace is zero (exp-004 original)
  flagStrengthThreshold:       0.5,   // |flagStrength| needed to unlock learning
  // frustrationWindow, frustrationThreshold, frustrationFlipStrength not set in base
  // so frustration is inactive — allows clean Baseline/Ambient/Dampening conditions
  ambientThreshold:            0.3,
  activityHistoryDecay:        0.95,
  activityHistoryMinimum:      0.1,
  chemicalDiffusionRadius:     15.0,
  chemicalFalloff:             'inverse',
  chemicalDecayRate:           0.5,
  positiveRewardStrength:      1.0,
  negativeRewardStrength:     -1.0,
  learningRate:                0.01,
  maxWeightDelta:              0.1,
  maxWeightMagnitude:          2.0,
  weightDecay:                 0.005,
  inputSize:                   5,
  outputSize:                  5,
  trainingEpisodes:            1000,
};

// Frustration params — only applied to conditions that use them
const FRUSTRATION_PARAMS = {
  frustrationWindow:       30,   // consecutive same-direction steps before check
  frustrationThreshold:   -0.1,  // rewardWhileAdjusting below this triggers flip
  frustrationFlipStrength: 0.5,  // weight *= -frustrationFlipStrength on flip
};

const PATTERNS = [
  { input: [1,0,1,0,1], target: [0,1,0,1,0] },
  { input: [1,1,0,0,0], target: [0,0,1,1,1] },
  { input: [1,0,0,0,1], target: [0,1,1,1,0] },
  { input: [0,1,0,1,0], target: [1,0,1,0,1] },
];

const CONDITIONS = [
  {
    label:    'Baseline',
    overrides: {
      ambientRadius:           0,
      _skipDampening:          true,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Ambient only',
    overrides: {
      _skipDampening:          true,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Dampening only',
    overrides: {
      ambientRadius:           0,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Full model',
    overrides: {
      chemicalDiffusionRadius: 15,
      ...FRUSTRATION_PARAMS,
    },
  },
];

const N_SEEDS = 10;
const SEEDS = [42, 137, 271, 314, 500, 618, 777, 888, 999, 1234];

// ─── Run one condition with one seed ─────────────────────────────────────
function runOne(configOverrides, seed) {
  return withSeed(seed, () => {
    const config  = { ...BASE_CONFIG, ...configOverrides };
    const network = createNetwork(config);

    for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
      const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
      step(network, pattern.input, pattern.target, ep);
    }

    // Frozen-weight evaluation
    let total = 0;
    const perPattern = PATTERNS.map(p => {
      const r = evaluate(network, p.input, p.target);
      total += r.accuracy;
      return r.accuracy;
    });
    return { mean: total / PATTERNS.length, perPattern };
  });
}

// ─── Statistics ───────────────────────────────────────────────────────────
function mean(arr)   { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr)    {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

// Two-tailed paired t-test. Returns { t, p }.
function pairedT(a, b) {
  const diffs = a.map((x, i) => x - b[i]);
  const n     = diffs.length;
  const m     = mean(diffs);
  const s     = std(diffs);
  const t     = m / (s / Math.sqrt(n));
  const df    = n - 1;

  function betaInc(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const lbeta = lgamma(a + b) - lgamma(a) - lgamma(b);
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
    let f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d; f = d;
    for (let m2 = 1; m2 <= 100; m2++) {
      let num = m2 * (b - m2) * x / ((a + 2*m2 - 1) * (a + 2*m2));
      d = 1 + num * d; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1/d;
      c = 1 + num / c; if (Math.abs(c) < 1e-30) c = 1e-30;
      f *= d * c;
      num = -(a + m2) * (a + b + m2) * x / ((a + 2*m2) * (a + 2*m2 + 1));
      d = 1 + num * d; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1/d;
      c = 1 + num / c; if (Math.abs(c) < 1e-30) c = 1e-30;
      const delta = d * c;
      f *= delta;
      if (Math.abs(delta - 1) < 1e-10) break;
    }
    return front * f;
  }

  function lgamma(z) {
    const g = 7;
    const c = [0.99999999999980993,676.5203681218851,-1259.1392167224028,
      771.32342877765313,-176.61502916214059,12.507343278686905,
      -0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    z -= 1;
    let x = c[0];
    for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  const x  = df / (df + t * t);
  const p  = betaInc(x, df / 2, 0.5);
  return { t: +t.toFixed(4), p: +p.toFixed(4) };
}

// ─── Main ─────────────────────────────────────────────────────────────────
console.log(`mcfeedback — Multi-seed experiment (N=${N_SEEDS} seeds)\n`);
console.log('Seeds:', SEEDS.join(', '));
console.log('Training episodes per run:', BASE_CONFIG.trainingEpisodes);
console.log('Evaluation: frozen weights, 4 patterns\n');

const allResults = CONDITIONS.map(() => []);

for (let si = 0; si < N_SEEDS; si++) {
  const seed = SEEDS[si];
  process.stdout.write(`Seed ${seed} (${si + 1}/${N_SEEDS}): `);
  for (let ci = 0; ci < CONDITIONS.length; ci++) {
    const { mean: acc } = runOne(CONDITIONS[ci].overrides, seed);
    allResults[ci].push(acc);
    process.stdout.write(`${CONDITIONS[ci].label.split(' ')[0]}=${(acc * 100).toFixed(0)}%  `);
  }
  process.stdout.write('\n');
}

// ─── Distribution table ───────────────────────────────────────────────────
console.log('\n─── Distribution (frozen-weight accuracy) ───────────────────────────────');
console.log(
  'Condition       '.padEnd(18) +
  'Mean'.padEnd(8)  +
  'Std'.padEnd(8)   +
  'Min'.padEnd(8)   +
  'Max'.padEnd(8)   +
  'All values'
);
console.log('─'.repeat(80));
for (let ci = 0; ci < CONDITIONS.length; ci++) {
  const vals = allResults[ci];
  const m    = mean(vals);
  const s    = std(vals);
  const mn   = Math.min(...vals);
  const mx   = Math.max(...vals);
  const pcts = vals.map(v => (v * 100).toFixed(0) + '%').join(' ');
  console.log(
    CONDITIONS[ci].label.padEnd(18) +
    (m * 100).toFixed(1).padEnd(8) + '%' +
    (s * 100).toFixed(1).padEnd(8) + '%' +
    (mn * 100).toFixed(0).padEnd(8) + '%' +
    (mx * 100).toFixed(0).padEnd(8) + '%' +
    pcts
  );
}

// ─── Paired t-tests vs Baseline ──────────────────────────────────────────
console.log('\n─── Paired t-tests vs Baseline ──────────────────────────────────────────');
console.log(
  'Comparison'.padEnd(36) +
  'Mean diff'.padEnd(12) +
  't'.padEnd(10) +
  'p'.padEnd(10) +
  'Significant?'
);
console.log('─'.repeat(80));

const baselineVals = allResults[0];
for (let ci = 1; ci < CONDITIONS.length; ci++) {
  const vals     = allResults[ci];
  const diffs    = vals.map((v, i) => v - baselineVals[i]);
  const mdiff    = mean(diffs);
  const { t, p } = pairedT(vals, baselineVals);
  const label    = `${CONDITIONS[ci].label} vs Baseline`;
  const sig      = p < 0.05 ? (p < 0.01 ? '** (p<0.01)' : '* (p<0.05)') : 'ns';
  console.log(
    label.padEnd(36) +
    ((mdiff >= 0 ? '+' : '') + (mdiff * 100).toFixed(1) + '%').padEnd(12) +
    String(t).padEnd(10) +
    String(p).padEnd(10) +
    sig
  );
}

// ─── Per-pattern breakdown ────────────────────────────────────────────────
console.log('\n─── Per-pattern mean accuracy (across seeds) ────────────────────────────');
const patternResults = CONDITIONS.map(() => PATTERNS.map(() => []));

for (let si = 0; si < N_SEEDS; si++) {
  const seed = SEEDS[si];
  for (let ci = 0; ci < CONDITIONS.length; ci++) {
    const { perPattern } = withSeed(seed, () => {
      const config  = { ...BASE_CONFIG, ...CONDITIONS[ci].overrides };
      const network = createNetwork(config);
      for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
        const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
        step(network, pattern.input, pattern.target, ep);
      }
      let total = 0;
      const perPattern = PATTERNS.map(p => {
        const r = evaluate(network, p.input, p.target);
        total += r.accuracy;
        return r.accuracy;
      });
      return { mean: total / PATTERNS.length, perPattern };
    });
    perPattern.forEach((acc, pi) => patternResults[ci][pi].push(acc));
  }
}

const header = 'Condition       '.padEnd(18) +
  PATTERNS.map((_, i) => `P${i+1}`.padEnd(8)).join('');
console.log(header);
console.log('─'.repeat(18 + 8 * PATTERNS.length));
for (let ci = 0; ci < CONDITIONS.length; ci++) {
  const row = CONDITIONS[ci].label.padEnd(18) +
    patternResults[ci].map(vals => ((mean(vals) * 100).toFixed(1) + '%').padEnd(8)).join('');
  console.log(row);
}

// ─── Frustration flip diagnostic (Full model only) ───────────────────────
console.log('\n─── Frustration flip diagnostic — Full model ────────────────────────────');
console.log(
  'Seed'.padEnd(8) +
  'TotalFlips'.padEnd(12) +
  'FlippedSyn'.padEnd(12) +
  '|w| flipped'.padEnd(14) +
  '|w| stable'.padEnd(14) +
  'FinalAcc'
);
console.log('─'.repeat(72));

const fullOverrides = CONDITIONS[CONDITIONS.length - 1].overrides;
for (let si = 0; si < N_SEEDS; si++) {
  const seed = SEEDS[si];
  const diag = withSeed(seed, () => {
    const config  = { ...BASE_CONFIG, ...fullOverrides };
    const network = createNetwork(config);
    for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
      const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
      step(network, pattern.input, pattern.target, ep);
    }
    const flipped  = network.synapses.filter(s => s.frustrationFlipCount > 0);
    const stable   = network.synapses.filter(s => s.frustrationFlipCount === 0);
    const totalFlips = network.synapses.reduce((sum, s) => sum + s.frustrationFlipCount, 0);
    const meanWFlipped = flipped.length > 0
      ? flipped.reduce((sum, s) => sum + Math.abs(s.weight), 0) / flipped.length
      : 0;
    const meanWStable = stable.length > 0
      ? stable.reduce((sum, s) => sum + Math.abs(s.weight), 0) / stable.length
      : 0;
    return { totalFlips, flippedCount: flipped.length, meanWFlipped, meanWStable };
  });
  const finalAcc = (allResults[CONDITIONS.length - 1][si] * 100).toFixed(0);
  console.log(
    String(seed).padEnd(8) +
    String(diag.totalFlips).padEnd(12) +
    String(diag.flippedCount).padEnd(12) +
    diag.meanWFlipped.toFixed(4).padEnd(14) +
    diag.meanWStable.toFixed(4).padEnd(14) +
    `${finalAcc}%`
  );
}

console.log('\nDone.');
