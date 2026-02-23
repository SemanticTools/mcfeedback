// Network creation: 2 clusters × 30 neurons
// Per cluster: 5 input, 5 output, 2 modulatory, 23 regular
// Intra-cluster connectivity: 60%, Inter-cluster: 50%

const NEURONS_PER_CLUSTER = 30;
const CLUSTERS = 2;
const INPUT_COUNT = 5;
const OUTPUT_COUNT = 5;
const MODULATORY_COUNT = 2;
const REGULAR_COUNT = 23;

const CLUSTER_SPREAD = 5;
const CLUSTER_SPACING = 15;
const INITIAL_THRESHOLD = 0.5;
const INTRA_PROB = 0.6;
const INTER_PROB = 0.5;

function randomInRange(min, max) {
  return min + Math.random() * (max - min);
}

function createNeuron(id, type, cluster) {
  const cx = cluster * CLUSTER_SPACING;
  return {
    id,
    type,
    cluster,
    x: cx + randomInRange(-CLUSTER_SPREAD, CLUSTER_SPREAD),
    y: randomInRange(-CLUSTER_SPREAD, CLUSTER_SPREAD),
    z: randomInRange(-CLUSTER_SPREAD, CLUSTER_SPREAD),
    threshold: INITIAL_THRESHOLD,
    fired: false,
    fireCount: 0,
    stepCount: 0,
  };
}

export function createNetwork() {
  const neurons = [];

  for (let c = 0; c < CLUSTERS; c++) {
    const base = c * NEURONS_PER_CLUSTER;
    for (let i = 0; i < INPUT_COUNT; i++)
      neurons.push(createNeuron(base + i, 'input', c));
    for (let i = 0; i < OUTPUT_COUNT; i++)
      neurons.push(createNeuron(base + INPUT_COUNT + i, 'output', c));
    for (let i = 0; i < MODULATORY_COUNT; i++)
      neurons.push(createNeuron(base + INPUT_COUNT + OUTPUT_COUNT + i, 'modulatory', c));
    for (let i = 0; i < REGULAR_COUNT; i++)
      neurons.push(createNeuron(base + INPUT_COUNT + OUTPUT_COUNT + MODULATORY_COUNT + i, 'regular', c));
  }

  const synapses = [];
  const total = neurons.length;

  for (let pre = 0; pre < total; pre++) {
    for (let post = 0; post < total; post++) {
      if (pre === post) continue;
      // Input neurons are driven externally — skip connections to them
      if (neurons[post].type === 'input') continue;

      const sameCluster = neurons[pre].cluster === neurons[post].cluster;
      const prob = sameCluster ? INTRA_PROB : INTER_PROB;

      if (Math.random() < prob) {
        synapses.push({
          id: synapses.length,
          pre,
          post,
          weight: (Math.random() - 0.5) * 0.2,
        });
      }
    }
  }

  return { neurons, synapses };
}
