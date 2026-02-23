/**
 * Experiment 010b: Direction-consistent flag gating + propagationCycles: 3.
 * Base: experiment-010. Adds multi-step signal propagation per training step.
 *
 * Motivation: with propagationCycles=1 (all prior experiments), signals can
 * only travel one synapse per step. The recurrent graph has ~2-hop paths from
 * input to output — so the input pattern may never fully reach the output
 * neurons within a single cycle. Running 3 accumulate-and-fire cycles per step
 * lets signals propagate further into the network before learning is applied.
 *
 * propagationCycles: 3  — run the forward pass 3× per training step
 * All other params unchanged from experiment-010.
 *
 * Same 10-seed harness. Reports ep-500 latch % per seed for Full model.
 */

import { createNetwork } from '../src/network.mjs';
import { step, evaluate } from '../src/engine.mjs';

// ─── Seedable PRNG (mulberry32) ───────────────────────────────────────────
// Replaces Math.random for the duration of each run so results are reproducible.
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
  propagationCycles:           3,     // forward-pass cycles per training step
  consistencyThreshold:        5,     // consecutive same-sign traces needed before growth
  flagStrengthGain:            0.15,  // per-turn step toward ±1 (slower than exp-004's 0.3)
  flagDecayOnFlip:             0.5,   // sharp flagStrength penalty on direction flip
  flagDecayRate:               0.9,   // passive multiplier when trace is zero (was 0.7)
  flagStrengthThreshold:       0.5,   // |flagStrength| needed to unlock learning
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
    },
  },
];

const N_SEEDS = 10;
// Fixed seed list so the experiment is fully reproducible
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

// ─── Flag latch stats for output-bound synapses ───────────────────────────
function flagStats(synapses, outputIds) {
  const ob = synapses.filter(s => outputIds.has(s.to));
  if (ob.length === 0) return { n: 0, latched: 0, mean: 0 };
  const latched = ob.filter(s => Math.abs(s.flagStrength) >= 0.5).length;
  const meanF   = ob.reduce((sum, s) => sum + Math.abs(s.flagStrength), 0) / ob.length;
  return { n: ob.length, latched, mean: +meanF.toFixed(3) };
}

// ─── Statistics ───────────────────────────────────────────────────────────
function mean(arr)   { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr)    {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

// Two-tailed paired t-test. Returns { t, p }.
// p approximated via regularised incomplete beta (Abramowitz & Stegun 26.7.8).
function pairedT(a, b) {
  const diffs = a.map((x, i) => x - b[i]);
  const n     = diffs.length;
  const m     = mean(diffs);
  const s     = std(diffs);
  const t     = m / (s / Math.sqrt(n));
  const df    = n - 1;

  // Approximate two-tailed p-value via regularised incomplete beta
  // I_x(a,b) where x = df/(df+t²), a = df/2, b = 0.5
  function betaInc(x, a, b) {
    // Continued fraction (Lentz method, enough terms for df=9)
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const lbeta = lgamma(a + b) - lgamma(a) - lgamma(b);
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
    // Modified Lentz
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
    // Lanczos approximation
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
  const p  = betaInc(x, df / 2, 0.5);   // two-tailed
  return { t: +t.toFixed(4), p: +p.toFixed(4) };
}

// ─── Main ─────────────────────────────────────────────────────────────────
console.log(`mcfeedback — Multi-seed experiment (N=${N_SEEDS} seeds)\n`);
console.log('Seeds:', SEEDS.join(', '));
console.log('Training episodes per run:', BASE_CONFIG.trainingEpisodes);
console.log('Evaluation: frozen weights, 4 patterns\n');

// results[conditionIndex][seedIndex] = mean accuracy
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

// ─── Ep-500 flag latch diagnostic (Full model only) ──────────────────────
console.log('\n─── Ep-500 flag latch % — Full model (output-bound synapses) ───────────');
console.log(
  'Seed'.padEnd(8) +
  'Latched'.padEnd(12) +
  'Latch%'.padEnd(10) +
  'Mean|f|'.padEnd(10) +
  'FinalAcc'
);
console.log('─'.repeat(50));

const fullOverrides = CONDITIONS[CONDITIONS.length - 1].overrides; // Full model
for (let si = 0; si < N_SEEDS; si++) {
  const seed = SEEDS[si];
  const snap = withSeed(seed, () => {
    const config    = { ...BASE_CONFIG, ...fullOverrides };
    const network   = createNetwork(config);
    const outputIds = new Set([...network.neurons.values()]
      .filter(n => n.type === 'output').map(n => n.id));
    let snap500;
    for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
      const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
      step(network, pattern.input, pattern.target, ep);
      if (ep === 500) snap500 = flagStats(network.synapses, outputIds);
    }
    return snap500;
  });
  const finalAcc = (allResults[CONDITIONS.length - 1][si] * 100).toFixed(0);
  const latchPct = ((snap.latched / snap.n) * 100).toFixed(1);
  console.log(
    String(seed).padEnd(8) +
    `${snap.latched}/${snap.n}`.padEnd(12) +
    `${latchPct}%`.padEnd(10) +
    String(snap.mean).padEnd(10) +
    `${finalAcc}%`
  );
}

console.log('\nDone.');
