import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  activityHistoryDampening,
  informationDampening,
  ambientRelevanceDampening,
  combinedDampening,
} from '../src/dampening.mjs';

const config = {
  activityHistoryMinimum: 0.1,
  ambientThreshold:       0.3,
};

describe('activityHistoryDampening', () => {
  it('history below minimum → dampened (< 1)', () => {
    const m = activityHistoryDampening(0.05, config);
    assert.ok(m < 1);
  });

  it('history at zero → zero multiplier', () => {
    assert.equal(activityHistoryDampening(0, config), 0);
  });

  it('history above minimum → full strength (1)', () => {
    assert.equal(activityHistoryDampening(0.5, config), 1);
    assert.equal(activityHistoryDampening(1.0, config), 1);
  });

  it('history at minimum → 1', () => {
    assert.equal(activityHistoryDampening(config.activityHistoryMinimum, config), 1);
  });
});

describe('informationDampening', () => {
  it('never fires (0.0) → near zero', () => {
    assert.ok(informationDampening(0) < 0.01);
  });

  it('always fires (1.0) → near zero', () => {
    assert.ok(informationDampening(1) < 0.01);
  });

  it('peak at 0.5 → 1.0', () => {
    assert.equal(informationDampening(0.5), 1.0);
  });

  it('symmetric around 0.5', () => {
    assert.ok(Math.abs(informationDampening(0.2) - informationDampening(0.8)) < 1e-10);
  });

  it('monotonically increases from 0 to 0.5', () => {
    for (let r = 0; r < 0.49; r += 0.1) {
      assert.ok(informationDampening(r) < informationDampening(r + 0.1));
    }
  });
});

describe('ambientRelevanceDampening', () => {
  it('active neuron in high ambient → 1', () => {
    assert.equal(ambientRelevanceDampening(0.5, 1, config), 1);
  });

  it('active neuron in low ambient → 0.5', () => {
    assert.equal(ambientRelevanceDampening(0.1, 1, config), 0.5);
  });

  it('silent neuron in high ambient → 1', () => {
    assert.equal(ambientRelevanceDampening(0.5, 0, config), 1);
  });

  it('silent neuron in low ambient → 0', () => {
    assert.equal(ambientRelevanceDampening(0.1, 0, config), 0);
  });
});

describe('combinedDampening', () => {
  it('returns product of all three multipliers', () => {
    const synapse   = { activityHistory: 1.0 };
    const preState  = {};
    const postState = { fireRate: 0.5, ambientField: 0.5, output: 1 };
    const m = combinedDampening(synapse, preState, postState, config);
    // history=1, information=1, ambient=1 → product=1
    assert.equal(m, 1);
  });

  it('zero activity history zeroes the combined multiplier', () => {
    const synapse   = { activityHistory: 0 };
    const preState  = {};
    const postState = { fireRate: 0.5, ambientField: 0.5, output: 1 };
    assert.equal(combinedDampening(synapse, preState, postState, config), 0);
  });

  it('always-on neuron (fireRate=1) zeroes the combined multiplier', () => {
    const synapse   = { activityHistory: 1.0 };
    const preState  = {};
    const postState = { fireRate: 1.0, ambientField: 0.5, output: 1 };
    assert.ok(combinedDampening(synapse, preState, postState, config) < 0.01);
  });
});
