/**
 * Experiment 007: Connectivity diagnostic.
 * No mechanism changes — pure structural analysis before training.
 *
 * For each seed, before any training, measures:
 *   - Direct input→output synapses (count + which pairs)
 *   - Input fan-out (synapses leaving each input neuron)
 *   - Output fan-in (synapses arriving at each output neuron)
 *   - 2-hop input→hidden→output path count
 *   - Chemical dose budget per output-bound synapse:
 *       sum of falloff(1/distance) from every modulatory neuron
 *       (reflects how much reward signal each synapse can receive per step)
 *   - Modulatory neuron positions and distances to cluster centroids
 *
 * Then runs Full model training (exp-004 config) and correlates
 * structural metrics with final accuracy.
 */

import { createNetwork } from '../src/network.mjs';
import { step, evaluate } from '../src/engine.mjs';
import { distance3d } from '../src/utils.mjs';

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

// ─── Config (experiment-004 base — flag gate + linear reward) ─────────────
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

const SEEDS = [42, 137, 271, 314, 500, 618, 777, 888, 999, 1234];

// ─── Chemical falloff (mirrors chemical.mjs) ──────────────────────────────
function chemFalloff(d) {
  if (d <= 0) return 1;
  return 1 / d; // 'inverse' mode
}

// ─── Structural analysis (no training) ───────────────────────────────────
function analyzeStructure(network) {
  const { neurons, synapses, config } = network;

  const inputNeurons  = [...neurons.values()].filter(n => n.type === 'input');
  const outputNeurons = [...neurons.values()].filter(n => n.type === 'output');
  const hiddenNeurons = [...neurons.values()].filter(n => n.type === 'regular');
  const modNeurons    = [...neurons.values()].filter(n => n.type === 'modulatory');

  const inputIds  = new Set(inputNeurons.map(n => n.id));
  const outputIds = new Set(outputNeurons.map(n => n.id));
  const hiddenIds = new Set(hiddenNeurons.map(n => n.id));

  // ── Direct I→O synapses ─────────────────────────────────────────────────
  const directIO = synapses.filter(s => inputIds.has(s.from) && outputIds.has(s.to));
  const ioPairs  = directIO.map(s => ({
    fromIdx: inputNeurons.findIndex(n => n.id === s.from),
    toIdx:   outputNeurons.findIndex(n => n.id === s.to),
  }));

  // ── Input fan-out (to any neuron) ───────────────────────────────────────
  const inputFanOut = inputNeurons.map(n =>
    synapses.filter(s => s.from === n.id).length
  );

  // ── Output fan-in (from any neuron) ─────────────────────────────────────
  const outputFanIn = outputNeurons.map(n =>
    synapses.filter(s => s.to === n.id).length
  );

  // ── 2-hop I→H→O path count ──────────────────────────────────────────────
  // For each hidden neuron: does it have at least one input coming from an
  // input neuron AND at least one output going to an output neuron?
  let twoHopPaths = 0;
  for (const h of hiddenNeurons) {
    const hasInputFeed  = synapses.some(s => s.to === h.id && inputIds.has(s.from));
    const feedsOutput   = synapses.some(s => s.from === h.id && outputIds.has(s.to));
    if (hasInputFeed && feedsOutput) {
      // Count the actual pairs
      const feedingInputs  = synapses.filter(s => s.to === h.id && inputIds.has(s.from));
      const fedOutputs     = synapses.filter(s => s.from === h.id && outputIds.has(s.to));
      twoHopPaths += feedingInputs.length * fedOutputs.length;
    }
  }

  // ── Chemical dose budget for output-bound synapses ───────────────────────
  // For each synapse whose post-synaptic neuron is an output neuron,
  // compute the total 1/distance chemical dose it would receive from all
  // modulatory neurons per step (when all mod neurons fire).
  const outputBoundSynapses = synapses.filter(s => outputIds.has(s.to));
  const doseBudgets = outputBoundSynapses.map(s => {
    const toNeuron = neurons.get(s.to);
    return modNeurons.reduce((sum, mod) => {
      const d = distance3d(mod, toNeuron);
      return d <= config.chemicalDiffusionRadius ? sum + chemFalloff(d) : sum;
    }, 0);
  });
  const meanOutputDose = doseBudgets.length > 0
    ? doseBudgets.reduce((a, b) => a + b, 0) / doseBudgets.length
    : 0;
  const minOutputDose  = doseBudgets.length > 0 ? Math.min(...doseBudgets) : 0;

  // ── Modulatory neuron positions ──────────────────────────────────────────
  // Cluster centroids
  const inputCentroid  = {
    x: inputNeurons.reduce((s, n) => s + n.x, 0) / inputNeurons.length,
    y: inputNeurons.reduce((s, n) => s + n.y, 0) / inputNeurons.length,
    z: inputNeurons.reduce((s, n) => s + n.z, 0) / inputNeurons.length,
  };
  const outputCentroid = {
    x: outputNeurons.reduce((s, n) => s + n.x, 0) / outputNeurons.length,
    y: outputNeurons.reduce((s, n) => s + n.y, 0) / outputNeurons.length,
    z: outputNeurons.reduce((s, n) => s + n.z, 0) / outputNeurons.length,
  };

  const modInfo = modNeurons.map(mod => ({
    cluster:     mod.clusterId,
    x: +mod.x.toFixed(2), y: +mod.y.toFixed(2), z: +mod.z.toFixed(2),
    distToInput:  +distance3d(mod, inputCentroid).toFixed(2),
    distToOutput: +distance3d(mod, outputCentroid).toFixed(2),
  }));

  return {
    nInput:         inputNeurons.length,
    nOutput:        outputNeurons.length,
    nHidden:        hiddenNeurons.length,
    nMod:           modNeurons.length,
    totalSynapses:  synapses.length,
    directIOCount:  directIO.length,
    directIOMax:    inputNeurons.length * outputNeurons.length,
    ioPairs,
    inputFanOut,
    outputFanIn,
    twoHopPaths,
    meanOutputDose: +meanOutputDose.toFixed(3),
    minOutputDose:  +minOutputDose.toFixed(3),
    modInfo,
  };
}

// ─── Training run (Full model) ────────────────────────────────────────────
function runFullModel(seed) {
  return withSeed(seed, () => {
    const network = createNetwork(CONFIG);
    for (let ep = 1; ep <= CONFIG.trainingEpisodes; ep++) {
      const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
      step(network, pattern.input, pattern.target, ep);
    }
    let total = 0;
    for (const p of PATTERNS) {
      total += evaluate(network, p.input, p.target).accuracy;
    }
    return +(total / PATTERNS.length * 100).toFixed(0);
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────
console.log('mcfeedback — Experiment 007: Connectivity Diagnostic');
console.log('Config: experiment-004 base (flag gate, linear reward)\n');

const rows = [];

for (const seed of SEEDS) {
  // Build network with this seed (no training) for structural analysis
  const network    = withSeed(seed, () => createNetwork(CONFIG));
  const struct     = analyzeStructure(network);
  const finalAccPct = runFullModel(seed);

  rows.push({ seed, struct, finalAccPct });

  // ── Per-seed report ──────────────────────────────────────────────────────
  const pairsStr = struct.ioPairs.map(p => `I${p.fromIdx}→O${p.toIdx}`).join(', ') || 'none';
  const fanOutStr = struct.inputFanOut.map((v, i) => `I${i}:${v}`).join('  ');
  const fanInStr  = struct.outputFanIn.map((v, i) => `O${i}:${v}`).join('  ');

  console.log(`${'─'.repeat(70)}`);
  console.log(`Seed ${seed}   Final accuracy (Full model): ${finalAccPct}%`);
  console.log(`  Neurons     input=${struct.nInput}  output=${struct.nOutput}  hidden=${struct.nHidden}  mod=${struct.nMod}`);
  console.log(`  Synapses    total=${struct.totalSynapses}`);
  console.log(`  Direct I→O  ${struct.directIOCount} / ${struct.directIOMax} possible  [${pairsStr}]`);
  console.log(`  2-hop paths ${struct.twoHopPaths}`);
  console.log(`  Fan-out     ${fanOutStr}`);
  console.log(`  Fan-in      ${fanInStr}`);
  console.log(`  Chem dose   mean=${struct.meanOutputDose}  min=${struct.minOutputDose}  (output-bound synapses)`);
  for (const m of struct.modInfo) {
    console.log(`  Mod [${m.cluster}]  pos=(${m.x}, ${m.y}, ${m.z})  d→inputs=${m.distToInput}  d→outputs=${m.distToOutput}`);
  }
}

// ─── Correlation summary ──────────────────────────────────────────────────
console.log(`\n${'═'.repeat(70)}`);
console.log('CORRELATION SUMMARY — sorted by Full model accuracy (desc)\n');

const sorted = [...rows].sort((a, b) => b.finalAccPct - a.finalAccPct);

const hdr = [
  'Seed'.padEnd(6),
  'Acc%'.padEnd(6),
  'DirIO'.padEnd(7),
  '2-hop'.padEnd(7),
  'FanOut'.padEnd(8),
  'FanIn'.padEnd(8),
  'ChemDose'.padEnd(10),
  'ModDist→Out',
].join('');
console.log(hdr);
console.log('─'.repeat(70));

for (const { seed, struct, finalAccPct } of sorted) {
  const meanFanOut = (struct.inputFanOut.reduce((a, b) => a + b, 0) / struct.inputFanOut.length).toFixed(1);
  const meanFanIn  = (struct.outputFanIn.reduce((a, b) => a + b, 0) / struct.outputFanIn.length).toFixed(1);
  const modDistOut = (struct.modInfo.reduce((s, m) => s + m.distToOutput, 0) / struct.modInfo.length).toFixed(2);

  console.log([
    String(seed).padEnd(6),
    `${finalAccPct}%`.padEnd(6),
    String(struct.directIOCount).padEnd(7),
    String(struct.twoHopPaths).padEnd(7),
    meanFanOut.padEnd(8),
    meanFanIn.padEnd(8),
    String(struct.meanOutputDose).padEnd(10),
    modDistOut,
  ].join(''));
}

// ─── Group stats: good seeds (≥55%) vs poor seeds (<55%) ─────────────────
console.log(`\n${'─'.repeat(70)}`);
console.log('GROUP COMPARISON — good seeds (≥55%) vs poor seeds (<55%)\n');

function groupStats(rows) {
  const n = rows.length;
  if (n === 0) return null;
  const mean = k => rows.reduce((s, r) => s + r[k], 0) / n;
  return {
    n,
    directIO:    mean('directIO'),
    twoHop:      mean('twoHop'),
    fanOut:      mean('fanOut'),
    fanIn:       mean('fanIn'),
    chemDose:    mean('chemDose'),
    modDistOut:  mean('modDistOut'),
  };
}

const statsRows = rows.map(({ seed, struct, finalAccPct }) => ({
  seed,
  acc: finalAccPct,
  directIO:   struct.directIOCount,
  twoHop:     struct.twoHopPaths,
  fanOut:     struct.inputFanOut.reduce((a, b) => a + b, 0) / struct.inputFanOut.length,
  fanIn:      struct.outputFanIn.reduce((a, b) => a + b, 0) / struct.outputFanIn.length,
  chemDose:   struct.meanOutputDose,
  modDistOut: struct.modInfo.reduce((s, m) => s + m.distToOutput, 0) / struct.modInfo.length,
}));

const good = groupStats(statsRows.filter(r => r.acc >= 55));
const poor = groupStats(statsRows.filter(r => r.acc < 55));

const metrics = ['directIO','twoHop','fanOut','fanIn','chemDose','modDistOut'];
const labels  = ['Direct I→O','2-hop paths','Input fan-out','Output fan-in','Chem dose','Mod dist→output'];

console.log(`${'Metric'.padEnd(18)} ${'Good (≥55%)'.padEnd(14)} ${'Poor (<55%)'.padEnd(14)} Delta`);
console.log('─'.repeat(60));
for (let i = 0; i < metrics.length; i++) {
  const k = metrics[i];
  const g = good  ? good[k].toFixed(2)  : 'n/a';
  const p = poor  ? poor[k].toFixed(2)  : 'n/a';
  const d = (good && poor) ? (good[k] - poor[k]).toFixed(2) : 'n/a';
  const sign = parseFloat(d) > 0 ? '+' : '';
  console.log(`${labels[i].padEnd(18)} ${g.padEnd(14)} ${p.padEnd(14)} ${sign}${d}`);
}

console.log('\nDone.');
