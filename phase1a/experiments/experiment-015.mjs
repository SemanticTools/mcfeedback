/**
 * Experiment 015: 012b parameters + three new mechanisms.
 * Base: experiment-012b (exp-004 + weightDecay: 0.0025).
 *
 * New additions:
 *   1. modCycling: true
 *      Only one modulatory neuron fires per step, cycling through all 4
 *      (2 clusters × 2 per cluster) in round-robin order. Previously all
 *      4 fired simultaneously whenever rewardSignal ≠ 0.
 *
 *   2. chemicalDiffusionRadiusMin: 5
 *      Radius anneals from 15 → 5 linearly over 1000 episodes. Broadcast
 *      starts broad (global signal) and narrows (local, targeted).
 *
 *   3. provisionalWeights: true
 *      After each weight update, the new weights are flagged as provisional.
 *      On the next step: if accuracy improved (or stayed the same), commit.
 *      If accuracy dropped, revert all synapses to their pre-update values.
 *
 * Full config:
 *   learningRate:                0.01    (exp-004 original)
 *   weightDecay:                 0.0025  (012b fix)
 *   flagStrengthGain:            0.3
 *   flagDecayRate:               0.7
 *   flagStrengthThreshold:       0.5
 *   modCycling:                  true    ← new
 *   chemicalDiffusionRadius:     15      (start)
 *   chemicalDiffusionRadiusMin:  5       ← new (end of anneal)
 *   provisionalWeights:          true    ← new
 *
 * Same 10-seed harness, 4 conditions, 1000 episodes, frozen eval.
 * Diagnostic: mean |weight| and provisional revert rate per seed (Full model).
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
  chemicalDiffusionRadius:     15.0,   // anneal start
  chemicalDiffusionRadiusMin:  5.0,    // anneal end  ← new
  chemicalFalloff:             'inverse',
  chemicalDecayRate:           0.5,
  positiveRewardStrength:      1.0,
  negativeRewardStrength:     -1.0,
  learningRate:                0.01,
  maxWeightDelta:              0.1,
  maxWeightMagnitude:          2.0,
  weightDecay:                 0.0025,
  modCycling:                  true,   // ← new
  provisionalWeights:          true,   // ← new
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
      chemicalDiffusionRadiusMin: null,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Ambient only',
    overrides: {
      _skipDampening:          true,
      chemicalDiffusionRadius: 1000,
      chemicalDiffusionRadiusMin: null,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Dampening only',
    overrides: {
      ambientRadius:           0,
      chemicalDiffusionRadius: 1000,
      chemicalDiffusionRadiusMin: null,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Full model',
    overrides: {
      chemicalDiffusionRadius:    15,
      chemicalDiffusionRadiusMin: 5,
    },
  },
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
  const n = diffs.length; const m = mean(diffs); const s = std(diffs);
  const t = m / (s / Math.sqrt(n)); const df = n - 1;
  function betaInc(x, a, b) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const lbeta = lgamma(a + b) - lgamma(a) - lgamma(b);
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
    let f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
    if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d; f = d;
    for (let m2 = 1; m2 <= 100; m2++) {
      let num = m2 * (b - m2) * x / ((a + 2*m2 - 1) * (a + 2*m2));
      d = 1 + num * d; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1/d;
      c = 1 + num / c; if (Math.abs(c) < 1e-30) c = 1e-30; f *= d * c;
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
    z -= 1; let x = c[0];
    for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  const x = df / (df + t * t);
  const p = betaInc(x, df / 2, 0.5);
  return { t: +t.toFixed(4), p: +p.toFixed(4) };
}

// ─── Run one condition with one seed ─────────────────────────────────────
function runOne(configOverrides, seed) {
  return withSeed(seed, () => {
    const config  = { ...BASE_CONFIG, ...configOverrides };
    const network = createNetwork(config);
    let revertTotal = 0;

    for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
      const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
      // Revert proxy: a revert triggers if reward dropped vs previous step
      const prevReward = network._provisional?.reward ?? null;
      step(network, pattern.input, pattern.target, ep);
      if (prevReward != null && network._provisional != null
          && network._provisional.reward < prevReward) {
        revertTotal++;
      }
    }

    let total = 0;
    const perPattern = PATTERNS.map(p => {
      const r = evaluate(network, p.input, p.target);
      total += r.accuracy;
      return r.accuracy;
    });
    const meanAbsW = network.synapses.reduce((s, syn) => s + Math.abs(syn.weight), 0)
                   / network.synapses.length;
    return {
      mean: total / PATTERNS.length,
      perPattern,
      meanAbsW,
      revertRate: revertTotal / config.trainingEpisodes,
    };
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────
console.log('mcfeedback — Experiment 015: 012b + modCycling + radius anneal + provisional weights\n');
console.log('Seeds:', SEEDS.join(', '));
console.log('modCycling: true  chemRadius: 15→5  provisionalWeights: true\n');

const allResults = CONDITIONS.map(() => []);

for (let si = 0; si < N_SEEDS; si++) {
  const seed = SEEDS[si];
  process.stdout.write(`Seed ${seed} (${si + 1}/${N_SEEDS}): `);
  for (let ci = 0; ci < CONDITIONS.length; ci++) {
    const r = runOne(CONDITIONS[ci].overrides, seed);
    allResults[ci].push(r);
    process.stdout.write(`${CONDITIONS[ci].label.split(' ')[0]}=${(r.mean * 100).toFixed(0)}%  `);
  }
  process.stdout.write('\n');
}

// ─── Distribution table ───────────────────────────────────────────────────
console.log('\n─── Distribution (frozen-weight accuracy) ───────────────────────────────');
console.log('Condition       '.padEnd(18) + 'Mean'.padEnd(8) + 'Std'.padEnd(8) + 'Min'.padEnd(8) + 'Max'.padEnd(8) + 'All values');
console.log('─'.repeat(80));
for (let ci = 0; ci < CONDITIONS.length; ci++) {
  const vals = allResults[ci].map(r => r.mean);
  const m = mean(vals); const s = std(vals);
  const mn = Math.min(...vals); const mx = Math.max(...vals);
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
console.log('Comparison'.padEnd(36) + 'Mean diff'.padEnd(12) + 't'.padEnd(10) + 'p'.padEnd(10) + 'Significant?');
console.log('─'.repeat(80));
const baselineVals = allResults[0].map(r => r.mean);
for (let ci = 1; ci < CONDITIONS.length; ci++) {
  const vals  = allResults[ci].map(r => r.mean);
  const diffs = vals.map((v, i) => v - baselineVals[i]);
  const mdiff = mean(diffs);
  const { t, p } = pairedT(vals, baselineVals);
  const label = `${CONDITIONS[ci].label} vs Baseline`;
  const sig   = p < 0.05 ? (p < 0.01 ? '** (p<0.01)' : '* (p<0.05)') : 'ns';
  console.log(
    label.padEnd(36) +
    ((mdiff >= 0 ? '+' : '') + (mdiff * 100).toFixed(1) + '%').padEnd(12) +
    String(t).padEnd(10) + String(p).padEnd(10) + sig
  );
}

// ─── Per-pattern breakdown ────────────────────────────────────────────────
console.log('\n─── Per-pattern mean accuracy (across seeds) ────────────────────────────');
const header = 'Condition       '.padEnd(18) + PATTERNS.map((_, i) => `P${i+1}`.padEnd(8)).join('');
console.log(header);
console.log('─'.repeat(18 + 8 * PATTERNS.length));
for (let ci = 0; ci < CONDITIONS.length; ci++) {
  const row = CONDITIONS[ci].label.padEnd(18) +
    PATTERNS.map((_, pi) => {
      const acc = mean(allResults[ci].map(r => r.perPattern[pi]));
      return ((acc * 100).toFixed(1) + '%').padEnd(8);
    }).join('');
  console.log(row);
}

// ─── Mean |weight| + revert rate — Full model ────────────────────────────
console.log('\n─── Full model: per-seed weight and provisional revert stats ────────────');
console.log('Seed'.padEnd(8) + 'Accuracy'.padEnd(12) + 'Mean |weight|'.padEnd(16) + 'Revert rate');
console.log('─'.repeat(52));
const FM = allResults[3];
for (let si = 0; si < N_SEEDS; si++) {
  const r = FM[si];
  console.log(
    String(SEEDS[si]).padEnd(8) +
    ((r.mean * 100).toFixed(0) + '%').padEnd(12) +
    r.meanAbsW.toFixed(4).padEnd(16) +
    (r.revertRate * 100).toFixed(1) + '%'
  );
}
const goodW = FM.filter(r => r.mean >= 0.55).map(r => r.meanAbsW);
const poorW = FM.filter(r => r.mean <  0.55).map(r => r.meanAbsW);
console.log('');
const goodReverts = FM.filter(r => r.mean >= 0.55).map(r => r.revertRate);
const poorReverts = FM.filter(r => r.mean <  0.55).map(r => r.revertRate);
console.log(`Good seeds (≥55%): ${FM.filter(r=>r.mean>=0.55).length}/10  mean |w| = ${goodW.length ? mean(goodW).toFixed(4) : '—'}  mean revert = ${goodReverts.length ? (mean(goodReverts)*100).toFixed(1) : '—'}%`);
console.log(`Poor seeds (<55%): ${FM.filter(r=>r.mean< 0.55).length}/10  mean |w| = ${poorW.length ? mean(poorW).toFixed(4) : '—'}  mean revert = ${poorReverts.length ? (mean(poorReverts)*100).toFixed(1) : '—'}%`);

// ─── Comparison vs 012b ───────────────────────────────────────────────────
console.log('\n─── Comparison: 012b vs 015 (Full model) ────────────────────────────────');
const E12B_REF = { mean: 55.0, std: 0.0, seeds55: 10, meanW: 0.7898 };
const fm015Mean = mean(FM.map(r => r.mean)) * 100;
const fm015Std  = std(FM.map(r => r.mean)) * 100;
const fm015Good = FM.filter(r => r.mean >= 0.55).length;
const fm015W    = mean(FM.map(r => r.meanAbsW));
console.log(`${''.padEnd(20)} ${'012b'.padEnd(14)} ${'015 (3 mechanisms)'.padEnd(22)}`);
console.log('─'.repeat(56));
console.log(`${'Mean accuracy'.padEnd(20)} ${(E12B_REF.mean.toFixed(1) + '%').padEnd(14)} ${(fm015Mean.toFixed(1) + '%').padEnd(22)}`);
console.log(`${'Std'.padEnd(20)} ${(E12B_REF.std.toFixed(1) + '%').padEnd(14)} ${(fm015Std.toFixed(1) + '%').padEnd(22)}`);
console.log(`${'Seeds ≥ 55%'.padEnd(20)} ${(E12B_REF.seeds55 + '/10').padEnd(14)} ${(fm015Good + '/10').padEnd(22)}`);
console.log(`${'Mean |weight|'.padEnd(20)} ${E12B_REF.meanW.toFixed(4).padEnd(14)} ${fm015W.toFixed(4).padEnd(22)}`);

console.log('\nDone.');
