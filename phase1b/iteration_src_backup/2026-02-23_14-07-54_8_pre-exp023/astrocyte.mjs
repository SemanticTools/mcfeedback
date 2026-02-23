// Astrocyte module for Experiment 021.
//
// 8 astrocytes total (4 per cluster), each with:
//   - Fixed 2D territory (x-y plane, radius 3.0)
//   - Activity-dependent activation (sense previous-step neuron firing)
//   - Self-adapting activation threshold (meta-learning)
//
// Territory distance uses 2D Euclidean (x-y only), matching the {x, y} position spec.
// Pre-computes owned neurons and synapses at initialisation for efficiency.

const TERRITORY_RADIUS  = 3.0;
const INITIAL_THRESHOLD = 0.5;
const REWARD_HISTORY_SIZE = 20;
const ADAPT_INTERVAL    = 50;   // steps (= 5 episodes at 10 steps/ep)
const SUCCESS_HIGH      = 0.15; // lower threshold if recent success > this
const SUCCESS_LOW       = 0.05; // raise threshold if recent success < this
const THRESHOLD_MIN     = 0.1;
const THRESHOLD_MAX     = 0.9;
const MAX_ACTIVE        = 3;

// ─── Placement ────────────────────────────────────────────────────────────────

export function createAstrocytes(network, numPatterns) {
  const { neurons, synapses } = network;
  const astrocytes = [];

  for (let c = 0; c < 2; c++) {
    const clusterNeurons = neurons.filter(n => n.cluster === c);

    // Actual x/y bounds of this cluster's neurons
    const xs = clusterNeurons.map(n => n.x);
    const ys = clusterNeurons.map(n => n.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    const qx   = (maxX - minX) / 4;  // quarter-range offset
    const qy   = (maxY - minY) / 4;

    // 2×2 grid at quartile positions
    const positions = [
      { x: midX - qx, y: midY - qy },
      { x: midX + qx, y: midY - qy },
      { x: midX - qx, y: midY + qy },
      { x: midX + qx, y: midY + qy },
    ];

    for (let i = 0; i < 4; i++) {
      const pos = positions[i];

      // Neurons within territory (2D distance, z ignored)
      const neuronIds = clusterNeurons
        .filter(n => {
          const dx = n.x - pos.x, dy = n.y - pos.y;
          return Math.sqrt(dx * dx + dy * dy) <= TERRITORY_RADIUS;
        })
        .map(n => n.id);

      // Pre-compute owned synapses (pre OR post neuron in territory)
      const inTerritory = new Set(neuronIds);
      const ownedSynapses = synapses.filter(
        s => inTerritory.has(s.pre) || inTerritory.has(s.post)
      );

      astrocytes.push({
        id: astrocytes.length,
        cluster: c,
        position: { x: pos.x, y: pos.y },
        territoryRadius: TERRITORY_RADIUS,
        activationThreshold: INITIAL_THRESHOLD,
        rewardHistory: [],
        activationCount: 0,
        successCount: 0,
        neuronIds,
        ownedSynapses,
        activationsByPattern: Array(numPatterns).fill(0),
      });
    }
  }

  return astrocytes;
}

// Log territorial coverage for diagnostics
export function logCoverage(astrocytes, network) {
  const covered = new Set(astrocytes.flatMap(a => a.neuronIds));
  const nonInput = network.neurons.filter(n => n.type !== 'input');
  return { covered: covered.size, total: network.neurons.length, nonInputTotal: nonInput.length };
}

// ─── Per-step sensing ─────────────────────────────────────────────────────────

// Experiment 021: score = fraction of territory neurons that fired last step.
// prevFiredState: Uint8Array of length network.neurons.length (1 = fired last step)
export function computeActivationScores(astrocytes, prevFiredState) {
  return astrocytes.map(ast => {
    if (ast.neuronIds.length === 0) return 0;
    let fired = 0;
    for (const id of ast.neuronIds) {
      if (prevFiredState[id]) fired++;
    }
    return fired / ast.neuronIds.length;
  });
}

// Experiment 022: score = mean synaptic traffic in territory.
// Traffic for synapse s = abs(syn.weight) if pre-synaptic neuron fired last step, else 0.
// Detects incoming signal regardless of whether post-synaptic neuron crossed threshold.
export function computeTrafficScores(astrocytes, prevFiredState) {
  return astrocytes.map(ast => {
    if (ast.ownedSynapses.length === 0) return 0;
    let trafficSum = 0;
    for (const syn of ast.ownedSynapses) {
      if (prevFiredState[syn.pre]) {
        trafficSum += Math.abs(syn.weight);
      }
    }
    return trafficSum / ast.ownedSynapses.length;
  });
}

// ─── Activation selection ─────────────────────────────────────────────────────

export function selectActiveAstrocytes(astrocytes, scores) {
  // Pair each astrocyte with its score, sort descending
  const ranked = scores
    .map((score, i) => ({ ast: astrocytes[i], score }))
    .sort((a, b) => b.score - a.score);

  // Find those above their own threshold
  let active = ranked.filter(({ ast, score }) => score > ast.activationThreshold);

  // Minimum 1 always active
  if (active.length === 0) active = [ranked[0]];

  // Maximum MAX_ACTIVE
  if (active.length > MAX_ACTIVE) active = active.slice(0, MAX_ACTIVE);

  return active.map(({ ast }) => ast);
}

// ─── Eligible synapse collection ──────────────────────────────────────────────

export function getEligibleSynapsesFromAstrocytes(activeAstrocytes) {
  const seen = new Set();
  const eligible = [];
  for (const ast of activeAstrocytes) {
    for (const s of ast.ownedSynapses) {
      if (!seen.has(s.id)) {
        seen.add(s.id);
        eligible.push(s);
      }
    }
  }
  return eligible;
}

// ─── Threshold adaptation ─────────────────────────────────────────────────────

export function adaptAstrocytes(astrocytes, activeAstrocytes, kept, totalSteps) {
  // Update each active astrocyte's history and counters
  for (const ast of activeAstrocytes) {
    ast.rewardHistory.push(kept ? 1 : 0);
    if (ast.rewardHistory.length > REWARD_HISTORY_SIZE) ast.rewardHistory.shift();
    ast.activationCount++;
    if (kept) ast.successCount++;
  }

  // Every ADAPT_INTERVAL steps, adjust all astrocyte thresholds
  if (totalSteps % ADAPT_INTERVAL === 0) {
    for (const ast of astrocytes) {
      if (ast.rewardHistory.length === 0) continue;
      const rate = ast.rewardHistory.reduce((s, v) => s + v, 0) / ast.rewardHistory.length;
      if (rate > SUCCESS_HIGH) {
        ast.activationThreshold = Math.max(THRESHOLD_MIN, ast.activationThreshold - 0.01);
      } else if (rate < SUCCESS_LOW) {
        ast.activationThreshold = Math.min(THRESHOLD_MAX, ast.activationThreshold + 0.01);
      }
    }
  }
}
