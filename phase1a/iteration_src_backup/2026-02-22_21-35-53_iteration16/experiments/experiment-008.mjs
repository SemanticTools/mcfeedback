/**
 * Experiment 008: Flag strength diagnostic.
 * No mechanism changes. Base: experiment-004 (flag gate + linear reward).
 *
 * For each seed, runs Full model training and snapshots flag strength
 * statistics on output-bound synapses at episodes 100, 300, and 500.
 *
 * Prediction: good seeds (≥55% final acc) will have latched flags
 * (|flagStrength| >= 0.5) by episode 100; poor seeds (<55%) won't.
 *
 * Final accuracy at episode 1000 is reported to confirm results
 * match experiment-004.
 */

import { createNetwork } from '../src/network.mjs';
import { step, evaluate } from '../src/engine.mjs';

// ─── Seedable PRNG ────────────────────────────────────────────────────────
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

// ─── Config (experiment-004 base) ────────────────────────────────────────
const CONFIG = {
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

const SEEDS      = [42, 137, 271, 314, 500, 618, 777, 888, 999, 1234];
const CHECKPOINTS = [100, 300, 500];

// ─── Flag stats for output-bound synapses ─────────────────────────────────
function flagStats(synapses, outputIds) {
  const ob = synapses.filter(s => outputIds.has(s.to));
  if (ob.length === 0) return { n: 0, latched: 0, building: 0, mean: 0, max: 0 };
  const latched  = ob.filter(s => Math.abs(s.flagStrength) >= 0.5).length;
  const building = ob.filter(s => Math.abs(s.flagStrength) > 0 && Math.abs(s.flagStrength) < 0.5).length;
  const mean = ob.reduce((sum, s) => sum + Math.abs(s.flagStrength), 0) / ob.length;
  const max  = Math.max(...ob.map(s => Math.abs(s.flagStrength)));
  return { n: ob.length, latched, building, mean: +mean.toFixed(4), max: +max.toFixed(4) };
}

// ─── Run one seed with mid-training snapshots ────────────────────────────
function runWithSnapshots(seed) {
  return withSeed(seed, () => {
    const network   = createNetwork(CONFIG);
    const outputIds = new Set([...network.neurons.values()]
      .filter(n => n.type === 'output').map(n => n.id));

    const snapshots = {};
    for (let ep = 1; ep <= CONFIG.trainingEpisodes; ep++) {
      const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
      step(network, pattern.input, pattern.target, ep);
      if (CHECKPOINTS.includes(ep)) {
        snapshots[ep] = flagStats(network.synapses, outputIds);
      }
    }

    let total = 0;
    for (const p of PATTERNS) total += evaluate(network, p.input, p.target).accuracy;
    const finalAcc = +(total / PATTERNS.length * 100).toFixed(0);

    return { snapshots, finalAcc };
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────
console.log('mcfeedback — Experiment 008: Flag Strength Diagnostic');
console.log('Config: experiment-004 base (flagStrengthThreshold=0.5, gain=0.3)\n');

const results = [];

for (const seed of SEEDS) {
  const { snapshots, finalAcc } = runWithSnapshots(seed);
  results.push({ seed, snapshots, finalAcc });

  const n = snapshots[CHECKPOINTS[0]].n;
  console.log(`Seed ${String(seed).padEnd(5)}  Final: ${finalAcc}%`);
  console.log(`  ${'ep'.padEnd(6)} ${'latched'.padEnd(10)} ${'building'.padEnd(11)} ${'mean|f|'.padEnd(10)} ${'max|f|'.padEnd(8)} (of ${n} output-bound synapses)`);
  for (const ep of CHECKPOINTS) {
    const s = snapshots[ep];
    const lPct = ((s.latched  / s.n) * 100).toFixed(0).padStart(3);
    const bPct = ((s.building / s.n) * 100).toFixed(0).padStart(3);
    console.log(`  ep${String(ep).padEnd(4)} ${s.latched} (${lPct}%)  ${s.building} (${bPct}%)     ${String(s.mean).padEnd(10)} ${s.max}`);
  }
  console.log('');
}

// ─── Group comparison ────────────────────────────────────────────────────
console.log('═'.repeat(70));
console.log('GROUP COMPARISON — good seeds (≥55%) vs poor seeds (<55%)\n');

const good = results.filter(r => r.finalAcc >= 55);
const poor = results.filter(r => r.finalAcc <  55);

function groupMean(rows, ep, key) {
  return (rows.reduce((s, r) => s + r.snapshots[ep][key], 0) / rows.length).toFixed(3);
}
function groupMeanPct(rows, ep, key) {
  const val = rows.reduce((s, r) => {
    const snap = r.snapshots[ep];
    return s + snap[key] / snap.n;
  }, 0) / rows.length;
  return (val * 100).toFixed(1) + '%';
}

console.log(`${''.padEnd(28)} ${CHECKPOINTS.map(ep => `ep ${ep}`.padEnd(22)).join('')}`);
console.log(`${'Metric'.padEnd(28)} ${CHECKPOINTS.map(() => 'Good     Poor     Δ'.padEnd(22)).join('')}`);
console.log('─'.repeat(28 + 22 * CHECKPOINTS.length));

const metrics = [
  { label: 'Latched (% of synapses)', fn: (rows, ep) => groupMeanPct(rows, ep, 'latched') },
  { label: 'Building (% of synapses)', fn: (rows, ep) => groupMeanPct(rows, ep, 'building') },
  { label: 'Mean |flagStrength|',      fn: (rows, ep) => groupMean(rows, ep, 'mean') },
  { label: 'Max |flagStrength|',       fn: (rows, ep) => groupMean(rows, ep, 'max') },
];

for (const m of metrics) {
  let line = m.label.padEnd(28);
  for (const ep of CHECKPOINTS) {
    const g = m.fn(good, ep);
    const p = m.fn(poor, ep);
    const gNum = parseFloat(g);
    const pNum = parseFloat(p);
    const delta = (gNum - pNum).toFixed(3);
    const sign = delta >= 0 ? '+' : '';
    line += `${g.padEnd(9)} ${p.padEnd(9)} ${sign}${delta}`.padEnd(22);
  }
  console.log(line);
}

console.log('\nFinal accuracy:');
console.log(`  Good seeds (≥55%): ${good.map(r => `${r.seed}→${r.finalAcc}%`).join('  ')}`);
console.log(`  Poor seeds (<55%): ${poor.map(r => `${r.seed}→${r.finalAcc}%`).join('  ')}`);
console.log('\nDone.');
