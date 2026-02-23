import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createNeuronState, updateFireRate, regulateThreshold } from '../src/neuron.mjs';

const config = {
  targetFireRate:      0.2,
  thresholdAdjustRate: 0.01,
};

describe('updateFireRate', () => {
  it('fireRate is zero when cycleCount is zero', () => {
    const s = createNeuronState('n', 0.5);
    updateFireRate(s);
    assert.equal(s.fireRate, 0);
  });

  it('fireRate equals fireCount / cycleCount', () => {
    const s = createNeuronState('n', 0.5);
    s.fireCount  = 3;
    s.cycleCount = 10;
    updateFireRate(s);
    assert.ok(Math.abs(s.fireRate - 0.3) < 1e-10);
  });
});

describe('regulateThreshold', () => {
  it('fireRate above target → threshold increases', () => {
    const s = createNeuronState('n', 0.5);
    s.fireRate = 0.5; // above targetFireRate=0.2
    const before = s.threshold;
    regulateThreshold(s, config);
    assert.ok(s.threshold > before);
  });

  it('fireRate below target → threshold decreases', () => {
    const s = createNeuronState('n', 0.5);
    s.fireRate = 0.05; // below targetFireRate=0.2
    const before = s.threshold;
    regulateThreshold(s, config);
    assert.ok(s.threshold < before);
  });

  it('fireRate at target → threshold unchanged', () => {
    const s = createNeuronState('n', 0.5);
    s.fireRate = config.targetFireRate;
    const before = s.threshold;
    regulateThreshold(s, config);
    assert.equal(s.threshold, before);
  });

  it('threshold adjusts by exactly thresholdAdjustRate per step', () => {
    const s = createNeuronState('n', 0.5);
    s.fireRate = 0.5;
    regulateThreshold(s, config);
    assert.ok(Math.abs(s.threshold - (0.5 + config.thresholdAdjustRate)) < 1e-10);
  });

  it('threshold converges toward a stable value over many steps', () => {
    // Simulate a neuron that fires at a fixed rate of 0.5.
    // Threshold should climb until the neuron can no longer fire at that rate.
    // We just verify threshold moves in the right direction consistently.
    const s = createNeuronState('n', 0.1);
    s.fireRate = 0.5; // always too high

    const initial = s.threshold;
    for (let i = 0; i < 100; i++) regulateThreshold(s, config);

    assert.ok(s.threshold > initial);
  });
});
