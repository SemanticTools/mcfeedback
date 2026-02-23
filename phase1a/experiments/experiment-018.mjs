/**
 * Experiment 018: 017 + fixedOutputThreshold=true.
 *
 * Problem identified in 017: homeostasis and per-bit reward conflict.
 *
 * targetFireRate=0.2 means homeostasis suppresses any output neuron that fires >20%
 * of training steps. But per-bit reward demands bits fire 25–75% of the time
 * (depending on how many target patterns require that bit). Example:
 *
 *   Bit 2 target sequence: [0,1,1,1] → should fire 3/4 = 75% of training steps.
 *   Homeostasis: "you're firing >20%, raise threshold" → eventual suppression.
 *   Net effect: per-bit reward pushes up, homeostasis pushes threshold up.
 *   Winner: homeostasis (threshold regulation is fast: thresholdAdjustRate=0.01/step).
 *   Result: all output bits driven to silence → [00000] attractor.
 *
 * Fix: config.fixedOutputThreshold = true
 *   Output neurons skip regulateThreshold() entirely. Thresholds stay at
 *   initialThreshold (0.5) forever. The per-bit reward drives weight evolution
 *   without homeostatic interference. Hidden/modulatory neurons retain homeostasis.
 *
 * Weight dynamics for a discriminating attractor:
 *   - Wrong fire (active input, output fires, target=0): co-activation×neg_chem → -0.01/occurrence
 *     → selectively weakens pathways active during wrong-pattern presentations
 *   - Correct fire (active input, output fires, target=1): co-activation×pos_chem → +0.01/occurrence
 *     → selectively strengthens pathways active during correct-pattern presentations
 *   - Net: input-specific weight signs emerge naturally (positive from P4-only inputs,
 *     negative from P1/P2/P3 inputs for an output bit that should fire only for P4).
 *
 * Same 10-seed harness, 4 conditions, 10000 episodes.
 * Mid-training eval at ep2000, ep5000 to observe convergence trajectory.
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
  fixedOutputThreshold:        true,  // ← output neurons skip threshold regulation
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
  weightDecay:                 0.0005,
  perBitReward:                true,
  inputSize:                   5,
  outputSize:                  5,
  trainingEpisodes:            10000,
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
const SNAPSHOTS = [2000, 5000, 10000];

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

// ─── Run one condition with one seed, snapshotting at SNAPSHOTS ───────────
function runOne(configOverrides, seed) {
  return withSeed(seed, () => {
    const config  = { ...BASE_CONFIG, ...configOverrides };
    const network = createNetwork(config);
    const snaps = {};

    for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
      const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
      step(network, pattern.input, pattern.target, ep);

      if (SNAPSHOTS.includes(ep)) {
        let t = 0;
        const vecs = PATTERNS.map(p => {
          const r = evaluate(network, p.input, p.target);
          t += r.accuracy;
          return r.outputs.join('');
        });
        snaps[ep] = { mean: t / PATTERNS.length, outputVectors: vecs };
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
    return { mean: total / PATTERNS.length, perPattern, outputVectors, allSame, exactMatches, meanAbsW, snaps };
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────
console.log('mcfeedback — Experiment 018: perBitReward + fixedOutputThreshold + weightDecay=0.0005\n');
console.log('Seeds:', SEEDS.join(', '));
console.log('fixedOutputThreshold: true  weightDecay: 0.0005  perBitReward: true  trainingEpisodes: 10000\n');

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
console.log('\n─── Distribution (frozen-weight accuracy at ep10000) ────────────────────');
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

// ─── Paired t-tests ───────────────────────────────────────────────────────
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

// ─── Training trajectory — Full model ────────────────────────────────────
console.log('\n─── Training trajectory — Full model (do outputs vary? when?) ───────────');
console.log('Targets:'.padEnd(10) +
  PATTERNS.map((p,i) => `P${i+1}[${p.target.join('')}]`).join('  '));
const FM = allResults[3];

for (const ep of SNAPSHOTS) {
  const label = ep === BASE_CONFIG.trainingEpisodes ? `ep${ep} (FINAL)` : `ep${ep}`;
  console.log(`\n  ${label}:`);
  console.log('Seed'.padEnd(8) + 'Acc'.padEnd(6) + PATTERNS.map((_,i) => `P${i+1}-out`.padEnd(10)).join('') + 'Varies?');
  console.log('─'.repeat(65));
  for (let si = 0; si < N_SEEDS; si++) {
    const snap = FM[si].snaps[ep];
    if (snap) {
      const allSameSnap = snap.outputVectors.every(v => v === snap.outputVectors[0]);
      const vecStr = snap.outputVectors.map(v => ('['+v+']').padEnd(10)).join('');
      console.log(
        String(SEEDS[si]).padEnd(8) +
        ((snap.mean*100).toFixed(0)+'%').padEnd(6) +
        vecStr +
        (allSameSnap ? 'constant' : 'VARIES')
      );
    }
  }
}

const varying = FM.filter(r => !r.allSame).length;
const anyExact = FM.filter(r => r.exactMatches > 0).length;
const meanExact = mean(FM.map(r => r.exactMatches));
const maxAcc = Math.max(...FM.map(r => r.mean));
console.log(`\nFinal: Varying outputs: ${varying}/10   Seeds with ≥1 exact match: ${anyExact}/10`);
console.log(`Mean exact: ${meanExact.toFixed(1)}/4   Best seed accuracy: ${(maxAcc*100).toFixed(0)}%`);

// ─── Comparison table ─────────────────────────────────────────────────────
console.log('\n─── Progression: 012b → 016 → 017 → 018 ────────────────────────────────');
const fm018Mean = mean(FM.map(r => r.mean)) * 100;
const fm018Std  = std(FM.map(r => r.mean)) * 100;
console.log(`${''.padEnd(26)} ${'012b'.padEnd(12)} ${'016'.padEnd(12)} ${'017'.padEnd(12)} ${'018 (this)'.padEnd(12)}`);
console.log('─'.repeat(74));
console.log(`${'Mean accuracy'.padEnd(26)} ${'55.0%'.padEnd(12)} ${'45.5%'.padEnd(12)} ${'46.0%'.padEnd(12)} ${(fm018Mean.toFixed(1)+'%').padEnd(12)}`);
console.log(`${'Std'.padEnd(26)} ${'0.0%'.padEnd(12)} ${'1.6%'.padEnd(12)} ${'3.2%'.padEnd(12)} ${(fm018Std.toFixed(1)+'%').padEnd(12)}`);
console.log(`${'Attractor type'.padEnd(26)} ${'[11111]'.padEnd(12)} ${'[00000]'.padEnd(12)} ${'[00000]'.padEnd(12)} ${varying+'/10 vary'}`);
console.log(`${'Homeostasis on outputs'.padEnd(26)} ${'yes'.padEnd(12)} ${'yes'.padEnd(12)} ${'yes'.padEnd(12)} ${'NO'}`);
console.log(`${'True discrimination'.padEnd(26)} ${'NONE'.padEnd(12)} ${'1/10'.padEnd(12)} ${'1/10'.padEnd(12)} ${varying+'/10 seeds'}`);

console.log('\nDone.');
