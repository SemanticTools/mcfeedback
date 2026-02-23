import { createNeuron, createNeuronState } from './neuron.mjs';
import { createSynapse } from './synapse.mjs';
import { distance3d, randomInRange, generateId } from './utils.mjs';

function createCluster(clusterId, centerX, centerY, centerZ, config, isHidden = false) {
  const neurons = [];
  const spread  = config.neuronSpread;
  const total   = isHidden
    ? (config.hiddenNeuronsPerCluster ?? config.neuronsPerCluster)
    : config.neuronsPerCluster;
  const modCount = config.modulatoryPerCluster;

  for (let i = 0; i < total; i++) {
    const id   = generateId(`n_${clusterId}`);
    const x    = centerX + randomInRange(-spread, spread);
    const y    = centerY + randomInRange(-spread, spread);
    const z    = centerZ + randomInRange(-spread, spread);
    const type = i < modCount ? 'modulatory' : 'regular';
    neurons.push(createNeuron(id, x, y, z, type, clusterId));
  }
  return neurons;
}

function assignInputOutput(neurons, config) {
  // First config.inputSize regular neurons of cluster 0 become input
  // Last config.outputSize regular neurons of cluster 1 become output
  const regulars0 = neurons.filter(n => n.clusterId === 'cluster_0' && n.type === 'regular');
  const regulars1 = neurons.filter(n => n.clusterId === 'cluster_1' && n.type === 'regular');

  for (let i = 0; i < config.inputSize && i < regulars0.length; i++) {
    regulars0[i].type = 'input';
  }
  for (let i = 0; i < config.outputSize && i < regulars1.length; i++) {
    regulars1[regulars1.length - 1 - i].type = 'output';
  }
}

function buildSynapses(neurons, config) {
  const synapses = [];
  const [wMin, wMax] = config.initialWeightRange;

  for (let i = 0; i < neurons.length; i++) {
    for (let j = 0; j < neurons.length; j++) {
      if (i === j) continue;
      const from = neurons[i];
      const to   = neurons[j];

      // No synapses from or to modulatory neurons (they receive chemical cues, don't drive weights)
      if (from.type === 'modulatory' || to.type === 'modulatory') continue;

      const sameCluster = from.clusterId === to.clusterId;
      const prob = sameCluster
        ? config.intraClusterConnectionProb
        : config.interClusterConnectionProb;

      if (Math.random() < prob) {
        synapses.push(createSynapse(from.id, to.id, randomInRange(wMin, wMax)));
      }
    }
  }
  return synapses;
}

function precomputeNeighbours(neurons, config) {
  for (const neuron of neurons) {
    neuron.neighbourIds = neurons
      .filter(other => other.id !== neuron.id && distance3d(neuron, other) <= config.ambientRadius)
      .map(other => other.id);
  }
}

export function createNetwork(config) {
  const spacing = config.clusterSpacing;

  // Lay clusters out along the X axis
  const clusterCenters = [];
  for (let i = 0; i < config.clustersCount; i++) {
    clusterCenters.push({ x: i * spacing, y: 0, z: 0, id: `cluster_${i}` });
  }

  const allNeurons = [];
  for (let i = 0; i < clusterCenters.length; i++) {
    const center   = clusterCenters[i];
    const isHidden = i >= 2; // clusters 0 and 1 hold input/output neurons
    allNeurons.push(...createCluster(center.id, center.x, center.y, center.z, config, isHidden));
  }

  assignInputOutput(allNeurons, config);
  precomputeNeighbours(allNeurons, config);

  const neurons    = new Map(allNeurons.map(n => [n.id, n]));
  const neuronState = new Map(
    allNeurons.map(n => [n.id, createNeuronState(n.id, config.initialThreshold)])
  );

  const synapses = buildSynapses(allNeurons, config);

  return { neurons, neuronState, synapses, config };
}
