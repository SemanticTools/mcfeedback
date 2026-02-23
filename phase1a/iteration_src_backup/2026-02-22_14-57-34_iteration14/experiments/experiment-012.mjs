/**
 * Experiment 012: Learning signal vs weight decay balance.
 * Base: experiment-004 (flag gate, linear reward, original flag gain/threshold).
 * No frustration flip, no direction-consistent flags, no reward annealing.
 *
 * Hypothesis: weightDecay (0.005/step) outpaces learning for most synapses.
 * Typical inter-cluster synapse: chemical ~0.1, trace ~1.0, lr 0.01
 *   → delta = 0.001/step vs decay 0.005/step = 5× weaker than decay.
 * Only synapses with unusually high chemical dose can accumulate weight.
 * This is why 6/10 seeds in exp-011 ended with mean |weight| < 0.28.
 *
 * Three variants tested against exp-004 reference:
 *   012a — learningRate: 0.02  (2×LR, everything else unchanged)
 *   012b — weightDecay:  0.0025 (0.5×WD, everything else unchanged)
 *   012c — learningRate: 0.02, weightDecay: 0.0025  (both)
 *
 * Diagnostic: mean |weight| per seed at ep 1000 for Full model.
 * Grouped by good (≥55%) vs poor (<55%) seeds.
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
  flagStrengthGain:            0.3,
  flagDecayRate:               0.7,
  flagStrengthThreshold:       0.5,
  ambientThreshold:            0.3,
  activityHistoryDecay:        0.95,
  activityHistoryMinimum:      0.1,
  chemicalDiffusionRadius:     15.0,
  chemicalFalloff:             'inverse',
  chemicalDecayRate:           0.5,
  positiveRewardStrength:      1.0,
  negativeRewardStrength:     -1.0,
  learningRate:                0.01,   // varied in 012a/012c
  maxWeightDelta:              0.1,
  maxWeightMagnitude:          2.0,
  weightDecay:                 0.005,  // varied in 012b/012c
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
    label: 'Baseline',
    overrides: {
      ambientRadius:           0,
      _skipDampening:          true,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label: 'Ambient only',
    overrides: {
      _skipDampening:          true,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label: 'Dampening only',
    overrides: {
      ambientRadius:           0,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label: 'Full model',
    overrides: {
      chemicalDiffusionRadius: 15,
    },
  },
];

const VARIANTS = [
  { label: 'exp-004',  overrides: {} },
  { label: '012a',     overrides: { learningRate: 0.02 } },
  { label: '012b',     overrides: { weightDecay: 0.0025 } },
  { label: '012c',     overrides: { learningRate: 0.02, weightDecay: 0.0025 } },
];

const N_SEEDS = 10;
const SEEDS = [42, 137, 271, 314, 500, 618, 777, 888, 999, 1234];

// ─── Statistics ───────────────────────────────────────────────────────────
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

function pairedT(a, b) {
  const diffs = a.map((x, i) => x - b[i]);
  const n = diffs.length;
  const m = mean(diffs);
  const s = std(diffs);
  const t = m / (s / Math.sqrt(n));
  const df = n - 1;

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
      const delta = d * c; f *= delta;
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

  const x = df / (df + t * t);
  const p = betaInc(x, df / 2, 0.5);
  return { t: +t.toFixed(4), p: +p.toFixed(4) };
}

// ─── Run one condition+variant with one seed, return accuracy + mean|w| ──
function runOne(condOverrides, variantOverrides, seed) {
  return withSeed(seed, () => {
    const config  = { ...BASE_CONFIG, ...condOverrides, ...variantOverrides };
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
    const meanAbsW = network.synapses.reduce((s, syn) => s + Math.abs(syn.weight), 0)
                   / network.synapses.length;
    return { mean: total / PATTERNS.length, perPattern, meanAbsW };
  });
}

// ─── Collect all results ───────────────────────────────────────────────────
// variantResults[vi][ci][si] = { mean, perPattern, meanAbsW }
const variantResults = VARIANTS.map(() => CONDITIONS.map(() => []));

console.log('mcfeedback — Experiment 012: Learning signal vs weight decay\n');
console.log('Seeds:', SEEDS.join(', '));
console.log(`Variants: ${VARIANTS.map(v => v.label).join(' | ')}\n`);

for (let vi = 0; vi < VARIANTS.length; vi++) {
  console.log(`── ${VARIANTS[vi].label} ─────────────────────────────────────────`);
  for (let si = 0; si < N_SEEDS; si++) {
    const seed = SEEDS[si];
    process.stdout.write(`  Seed ${String(seed).padEnd(5)}: `);
    for (let ci = 0; ci < CONDITIONS.length; ci++) {
      const r = runOne(CONDITIONS[ci].overrides, VARIANTS[vi].overrides, seed);
      variantResults[vi][ci].push(r);
      process.stdout.write(`${CONDITIONS[ci].label.split(' ')[0]}=${(r.mean * 100).toFixed(0)}%  `);
    }
    process.stdout.write('\n');
  }
  console.log('');
}

// ─── Combined summary table ───────────────────────────────────────────────
console.log('═'.repeat(80));
console.log('COMBINED SUMMARY — Full model condition');
console.log('═'.repeat(80));

const col = 14;
console.log(
  ''.padEnd(20) +
  VARIANTS.map(v => v.label.padEnd(col)).join('')
);
console.log('─'.repeat(20 + col * VARIANTS.length));

function fullRow(label, fn) {
  return label.padEnd(20) + VARIANTS.map((_, vi) => {
    const vals = variantResults[vi][3]; // Full model = index 3
    return fn(vals).padEnd(col);
  }).join('');
}

const FM_CI = 3; // Full model condition index
console.log(fullRow('Mean accuracy',    vals => (mean(vals.map(r => r.mean)) * 100).toFixed(1) + '%'));
console.log(fullRow('Std',              vals => (std(vals.map(r => r.mean))  * 100).toFixed(1) + '%'));
console.log(fullRow('Min',              vals => (Math.min(...vals.map(r => r.mean)) * 100).toFixed(0) + '%'));
console.log(fullRow('Max',              vals => (Math.max(...vals.map(r => r.mean)) * 100).toFixed(0) + '%'));
console.log(fullRow('Seeds ≥ 55%',      vals => vals.filter(r => r.mean >= 0.55).length + '/10'));
console.log(fullRow('Mean |weight|',    vals => mean(vals.map(r => r.meanAbsW)).toFixed(4)));

console.log('');

// ─── Per-variant: paired t-test vs Baseline ───────────────────────────────
console.log('─── Paired t-test: Full model vs Baseline (per variant) ─────────────────');
console.log(
  ''.padEnd(12) +
  VARIANTS.map(v => v.label.padEnd(col)).join('')
);
console.log('─'.repeat(12 + col * VARIANTS.length));

for (const stat of ['Mean diff', 't', 'p', 'Sig']) {
  const line = stat.padEnd(12) + VARIANTS.map((_, vi) => {
    const full     = variantResults[vi][3].map(r => r.mean);
    const baseline = variantResults[vi][0].map(r => r.mean);
    const diffs    = full.map((v, i) => v - baseline[i]);
    const md       = mean(diffs);
    const { t, p } = pairedT(full, baseline);
    const sig = p < 0.05 ? (p < 0.01 ? '**' : '*') : 'ns';
    const val = stat === 'Mean diff' ? ((md >= 0 ? '+' : '') + (md * 100).toFixed(1) + '%')
              : stat === 't'         ? String(t)
              : stat === 'p'         ? String(p)
              :                        sig;
    return val.padEnd(col);
  }).join('');
  console.log(line);
}

// ─── Per-seed weight diagnostic — Full model ─────────────────────────────
console.log('\n─── Mean |weight| per seed — Full model ─────────────────────────────────');
console.log(
  'Seed'.padEnd(8) +
  VARIANTS.map(v => (v.label + ' acc').padEnd(12) + (v.label + ' |w|').padEnd(10)).join('')
);
console.log('─'.repeat(8 + 22 * VARIANTS.length));

for (let si = 0; si < N_SEEDS; si++) {
  let line = String(SEEDS[si]).padEnd(8);
  for (let vi = 0; vi < VARIANTS.length; vi++) {
    const r   = variantResults[vi][FM_CI][si];
    const acc = (r.mean * 100).toFixed(0) + '%';
    line += acc.padEnd(12) + r.meanAbsW.toFixed(4).padEnd(10);
  }
  console.log(line);
}

// ─── Good vs poor mean |weight| ───────────────────────────────────────────
console.log('\n─── Mean |weight|: good seeds (≥55%) vs poor seeds (<55%) — Full model ──');
console.log(''.padEnd(12) + VARIANTS.map(v => v.label.padEnd(col)).join(''));
console.log('─'.repeat(12 + col * VARIANTS.length));

for (const grp of ['Good ≥55%', 'Poor <55%', 'Δ (G−P)']) {
  const line = grp.padEnd(12) + VARIANTS.map((_, vi) => {
    const vals = variantResults[vi][FM_CI];
    const good = vals.filter(r => r.mean >= 0.55).map(r => r.meanAbsW);
    const poor = vals.filter(r => r.mean <  0.55).map(r => r.meanAbsW);
    if (grp === 'Good ≥55%') return (good.length ? mean(good).toFixed(4) : '—').padEnd(col);
    if (grp === 'Poor <55%') return (poor.length ? mean(poor).toFixed(4) : '—').padEnd(col);
    const g = good.length ? mean(good) : 0;
    const p = poor.length ? mean(poor) : 0;
    return ((g - p >= 0 ? '+' : '') + (g - p).toFixed(4)).padEnd(col);
  }).join('');
  console.log(line);
}

// ─── Per-pattern breakdown ────────────────────────────────────────────────
console.log('\n─── Per-pattern accuracy — Full model (mean across seeds) ───────────────');
console.log(
  'Pattern'.padEnd(12) +
  VARIANTS.map(v => v.label.padEnd(col)).join('')
);
console.log('─'.repeat(12 + col * VARIANTS.length));
for (let pi = 0; pi < PATTERNS.length; pi++) {
  const line = `P${pi + 1}`.padEnd(12) + VARIANTS.map((_, vi) => {
    const acc = mean(variantResults[vi][FM_CI].map(r => r.perPattern[pi]));
    return ((acc * 100).toFixed(1) + '%').padEnd(col);
  }).join('');
  console.log(line);
}

console.log('\nDone.');
