import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { diffuseChemical } from '../src/chemical.mjs';

function makeNetwork({ modFired = true, synapsePositions = [[0, 0, 0]], rewardStrength = 1 } = {}) {
  const config = {
    chemicalDiffusionRadius: 5.0,
    chemicalFalloff:         'inverse',
    chemicalDecayRate:       0.5,
    positiveRewardStrength:  1.0,
    negativeRewardStrength: -1.0,
  };

  const modNeuron = { id: 'mod', x: 0, y: 0, z: 0, type: 'modulatory' };
  const neurons   = new Map([['mod', modNeuron]]);
  const neuronState = new Map([['mod', { firedThisCycle: modFired }]]);

  const synapses = synapsePositions.map((pos, i) => {
    const toId = `n${i}`;
    neurons.set(toId, { id: toId, x: pos[0], y: pos[1], z: pos[2], type: 'regular' });
    neuronState.set(toId, { firedThisCycle: false });
    return { from: 'mod', to: toId, chemicalLevel: 0 };
  });

  return { network: { neurons, neuronState, synapses, config }, synapses };
}

describe('diffuseChemical', () => {
  it('positive reward increases chemical on nearby synapses', () => {
    const { network, synapses } = makeNetwork({ synapsePositions: [[1, 0, 0]] });
    diffuseChemical(network, 1);
    assert.ok(synapses[0].chemicalLevel > 0);
  });

  it('negative reward decreases chemical (negative level)', () => {
    const { network, synapses } = makeNetwork({ synapsePositions: [[1, 0, 0]] });
    diffuseChemical(network, -1);
    assert.ok(synapses[0].chemicalLevel < 0);
  });

  it('synapse beyond diffusion radius → no chemical', () => {
    const { network, synapses } = makeNetwork({ synapsePositions: [[100, 0, 0]] });
    diffuseChemical(network, 1);
    assert.equal(synapses[0].chemicalLevel, 0);
  });

  it('closer synapse gets more chemical than distant one', () => {
    const { network, synapses } = makeNetwork({
      synapsePositions: [[1, 0, 0], [4, 0, 0]],
    });
    diffuseChemical(network, 1);
    assert.ok(synapses[0].chemicalLevel > synapses[1].chemicalLevel);
  });

  it('modulatory neuron that did not fire → no diffusion', () => {
    const { network, synapses } = makeNetwork({ modFired: false, synapsePositions: [[1, 0, 0]] });
    diffuseChemical(network, 1);
    assert.equal(synapses[0].chemicalLevel, 0);
  });
});
