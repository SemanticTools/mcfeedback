import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computeAmbientFields } from '../src/ambient.mjs';

function makeNeuron(id, x, y, z, neighbourIds = []) {
  return { id, x, y, z, neighbourIds };
}

function makeState(id, output) {
  return [id, { output, ambientField: 0 }];
}

describe('computeAmbientFields', () => {
  it('all-silent neighbours → zero ambient field', () => {
    const n0 = makeNeuron('n0', 0, 0, 0, ['n1', 'n2']);
    const n1 = makeNeuron('n1', 1, 0, 0, []);
    const n2 = makeNeuron('n2', 2, 0, 0, []);
    const neurons    = new Map([['n0', n0], ['n1', n1], ['n2', n2]]);
    const neuronState = new Map([makeState('n0', 0), makeState('n1', 0), makeState('n2', 0)]);

    computeAmbientFields(neurons, neuronState);
    assert.equal(neuronState.get('n0').ambientField, 0);
  });

  it('all-active neighbours → positive ambient field', () => {
    const n0 = makeNeuron('n0', 0, 0, 0, ['n1', 'n2']);
    const n1 = makeNeuron('n1', 1, 0, 0, []);
    const n2 = makeNeuron('n2', 2, 0, 0, []);
    const neurons    = new Map([['n0', n0], ['n1', n1], ['n2', n2]]);
    const neuronState = new Map([makeState('n0', 0), makeState('n1', 1), makeState('n2', 1)]);

    computeAmbientFields(neurons, neuronState);
    assert.ok(neuronState.get('n0').ambientField > 0);
  });

  it('closer neighbours contribute more than distant ones', () => {
    // n0 has two neighbours: n1 at distance 1, n2 at distance 10
    const n0 = makeNeuron('n0', 0, 0, 0, ['n1', 'n2']);
    const n1 = makeNeuron('n1', 1, 0, 0, []);
    const n2 = makeNeuron('n2', 10, 0, 0, []);
    const neurons    = new Map([['n0', n0], ['n1', n1], ['n2', n2]]);

    // Only n1 active
    const stateA = new Map([makeState('n0', 0), makeState('n1', 1), makeState('n2', 0)]);
    computeAmbientFields(neurons, stateA);
    const fieldFromClose = stateA.get('n0').ambientField;

    // Only n2 active
    const stateB = new Map([makeState('n0', 0), makeState('n1', 0), makeState('n2', 1)]);
    computeAmbientFields(neurons, stateB);
    const fieldFromFar = stateB.get('n0').ambientField;

    assert.ok(fieldFromClose > fieldFromFar);
  });

  it('neuron with no neighbours → zero ambient field', () => {
    const n0 = makeNeuron('n0', 0, 0, 0, []);  // no neighbours
    const n1 = makeNeuron('n1', 1, 0, 0, []);
    const neurons    = new Map([['n0', n0], ['n1', n1]]);
    const neuronState = new Map([makeState('n0', 0), makeState('n1', 1)]);

    computeAmbientFields(neurons, neuronState);
    assert.equal(neuronState.get('n0').ambientField, 0);
  });

  it('field is inverse of distance for single active neighbour', () => {
    const n0 = makeNeuron('n0', 0, 0, 0, ['n1']);
    const n1 = makeNeuron('n1', 4, 0, 0, []);
    const neurons    = new Map([['n0', n0], ['n1', n1]]);
    const neuronState = new Map([makeState('n0', 0), makeState('n1', 1)]);

    computeAmbientFields(neurons, neuronState);
    assert.ok(Math.abs(neuronState.get('n0').ambientField - 0.25) < 1e-10);
  });
});
