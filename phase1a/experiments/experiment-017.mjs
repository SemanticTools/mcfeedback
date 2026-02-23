/**
 * Experiment 017: 016 + weightDecay reduced 5× (0.0025 → 0.0005).
 *
 * Problem identified in 016: weight decay dominates the per-bit learning signal.
 *
 * Per-cycle balance for a bit that must fire in k/4 patterns:
 *   wrong-silence delta = mismatch(−0.5) × chemical(−1) × lr(0.01) = +0.005/occurrence
 *   decay per cycle (4 steps × 0.0025) = −0.010
 *
 *   k=1: +0.005 − 0.010 = −0.005  → decay wins → stays silent → [00000] attractor
 *   k=2: +0.010 − 0.010 =  0.000  → neutral
 *   k=3: +0.015 − 0.010 = +0.005  → signal wins
 *
 * Fix: weightDecay: 0.0005 (decay per cycle = 0.002).
 *   k=1: +0.005 − 0.002 = +0.003  → signal wins (minority bits can grow)
 *   k=2: +0.010 − 0.002 = +0.008
 *   k=3: +0.015 − 0.002 = +0.013
 *
 * Wrong-fire penalty: co-activation(+1.0) × chemical(−1) × lr = −0.010 per wrong fire.
 * This should cap growth: bits that overshoot and fire wrongly are quickly suppressed.
 * Expected dynamic: minority bits slowly grow, fire, get corrected → stable discrimination.
 *
 * Training extended to 5000 episodes. Mid-training eval at ep1000 to detect
 * the peak-then-collapse pattern seen in exploratory runs.
 *
 * Same 10-seed harness, 4 conditions, 5000 episodes.
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
  flagStrengthThreshold:       0,      // bypass flag gate: use raw eligibilityTrace
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
  weightDecay:                 0.0005,  // ← reduced 5× from 016 (was 0.0025)
  perBitReward:                true,
  inputSize:                   5,
  outputSize:                  5,
  trainingEpisodes:            5000,
};

const PATTERNS = [
  { input: [1,0,1,0,1], target: [0,1,0,1,0] },
  { input: [1,1,0,0,0], target: [0,0,1,1,1] },
  { input: [1,0,0,0,1], target: [0,1,1,1,0] },
  { input: [0,1,0,1,0], target: [1,0,1,0,1] },
];
const TARGET_STRS = PATTERNS.map(p => p.target.join(''));

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

// ─── Run one condition with one seed, snapshotting mid-training ───────────
function runOne(configOverrides, seed) {
  return withSeed(seed, () => {
    const config  = { ...BASE_CONFIG, ...configOverrides };
    const network = createNetwork(config);

    const MIDPOINT = 1000;
    let mid = null;

    for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
      const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
      step(network, pattern.input, pattern.target, ep);

      // Snapshot at midpoint
      if (ep === MIDPOINT) {
        let mt = 0;
        const mvecs = PATTERNS.map(p => {
          const r = evaluate(network, p.input, p.target);
          mt += r.accuracy;
          return r.outputs.join('');
        });
        mid = { mean: mt / PATTERNS.length, outputVectors: mvecs };
      }
    }

    let total = 0;
    const perPattern = PATTERNS.map(p => {
      const r = evaluate(network, p.input, p.target);
      total += r.accuracy;
      return r.accuracy;
    });
    const outputVectors = PATTERNS.map(p => evaluate(network, p.input, p.target).outputs.join(''));
    const allSame = outputVectors.every(v => v === outputVectors[0]);
    const exactMatches = outputVectors.filter((v, i) => v === TARGET_STRS[i]).length;
    const meanAbsW = network.synapses.reduce((s, syn) => s + Math.abs(syn.weight), 0)
                   / network.synapses.length;
    return {
      mean: total / PATTERNS.length,
      perPattern,
      outputVectors,
      allSame,
      exactMatches,
      meanAbsW,
      mid,
    };
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────
console.log('mcfeedback — Experiment 017: perBitReward + weightDecay=0.0005\n');
console.log('Seeds:', SEEDS.join(', '));
console.log('weightDecay: 0.0005  perBitReward: true  flagStrengthThreshold: 0  trainingEpisodes: 5000\n');

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
console.log('\n─── Distribution (frozen-weight accuracy at ep5000) ─────────────────────');
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

// ─── Output vector diagnostic — Full model ───────────────────────────────
console.log('\n─── Output vectors — Full model (ep1000 mid vs ep5000 final) ────────────');
console.log('Targets:'.padEnd(10) +
  PATTERNS.map((p,i) => `P${i+1}[${p.target.join('')}]`).join('  '));

console.log('\n  Mid-training (ep1000):');
console.log('Seed'.padEnd(8) + 'Acc'.padEnd(6) + PATTERNS.map((_,i) => `P${i+1}-out`.padEnd(10)).join('') + 'Varies?');
console.log('─'.repeat(65));
const FM = allResults[3];
for (let si = 0; si < N_SEEDS; si++) {
  const r = FM[si];
  const m = r.mid;
  if (m) {
    const allSameMid = m.outputVectors.every(v => v === m.outputVectors[0]);
    const vecStr = m.outputVectors.map(v => ('['+v+']').padEnd(10)).join('');
    console.log(
      String(SEEDS[si]).padEnd(8) +
      ((m.mean*100).toFixed(0)+'%').padEnd(6) +
      vecStr +
      (allSameMid ? 'constant' : 'VARIES')
    );
  }
}

console.log('\n  Final (ep5000):');
console.log('Seed'.padEnd(8) + 'Acc'.padEnd(6) + '|w|'.padEnd(8) +
  PATTERNS.map((_,i) => `P${i+1}-out`.padEnd(10)).join('') + 'Varies?  Exact');
console.log('─'.repeat(75));
for (let si = 0; si < N_SEEDS; si++) {
  const r = FM[si];
  const vecStr = r.outputVectors.map(v => ('['+v+']').padEnd(10)).join('');
  console.log(
    String(SEEDS[si]).padEnd(8) +
    ((r.mean*100).toFixed(0)+'%').padEnd(6) +
    r.meanAbsW.toFixed(3).padEnd(8) +
    vecStr +
    (r.allSame ? 'constant ' : 'VARIES   ') +
    r.exactMatches + '/4'
  );
}

const varying = FM.filter(r => !r.allSame).length;
const anyExact = FM.filter(r => r.exactMatches > 0).length;
const meanExact = mean(FM.map(r => r.exactMatches));
console.log(`\nVarying outputs: ${varying}/10   Seeds with ≥1 exact match: ${anyExact}/10   Mean exact patterns: ${meanExact.toFixed(1)}/4`);

// ─── Comparison table ─────────────────────────────────────────────────────
console.log('\n─── Progression: 012b → 016 → 017 ──────────────────────────────────────');
const fm017Mean = mean(FM.map(r => r.mean)) * 100;
const fm017Std  = std(FM.map(r => r.mean)) * 100;
console.log(`${''.padEnd(24)} ${'012b'.padEnd(14)} ${'016'.padEnd(14)} ${'017 (this)'.padEnd(14)}`);
console.log('─'.repeat(66));
console.log(`${'Mean accuracy'.padEnd(24)} ${'55.0%'.padEnd(14)} ${'45.5%'.padEnd(14)} ${(fm017Mean.toFixed(1)+'%').padEnd(14)}`);
console.log(`${'Std'.padEnd(24)} ${'0.0%'.padEnd(14)} ${'1.6%'.padEnd(14)} ${(fm017Std.toFixed(1)+'%').padEnd(14)}`);
console.log(`${'Attractor'.padEnd(24)} ${'[11111]'.padEnd(14)} ${'[00000]'.padEnd(14)} ${'varying/'+varying+'/10'}`);
console.log(`${'weightDecay'.padEnd(24)} ${'0.0025'.padEnd(14)} ${'0.0025'.padEnd(14)} ${'0.0005'}`);
console.log(`${'True discrimination'.padEnd(24)} ${'NONE'.padEnd(14)} ${'1/10'.padEnd(14)} ${varying+'/10 seeds'}`);

console.log('\nDone.');
