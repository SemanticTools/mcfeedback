import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createSynapse, updateWeight, updateActivityHistory, decayChemical } from '../src/synapse.mjs';

const config = {
  learningRate:           0.1,
  maxWeightDelta:         1.0,
  maxWeightMagnitude:     2.0,
  weightDecay:            0.0,   // 0 for most tests so weight decay doesn't interfere
  activityHistoryDecay:   0.9,
  chemicalDecayRate:      0.5,
};

describe('createSynapse', () => {
  it('initializes with correct fields', () => {
    const s = createSynapse('a', 'b', 0.05);
    assert.equal(s.from, 'a');
    assert.equal(s.to, 'b');
    assert.equal(s.weight, 0.05);
    assert.equal(s.eligibilityTrace, 0);
    assert.equal(s.activityHistory, 0);
    assert.equal(s.chemicalLevel, 0);
  });
});

describe('updateWeight', () => {
  it('no update when trace is zero', () => {
    const s = { ...createSynapse('a', 'b', 0.5), eligibilityTrace: 0, chemicalLevel: 1 };
    updateWeight(s, config);
    assert.equal(s.weight, 0.5);
  });

  it('no update when chemical is zero', () => {
    const s = { ...createSynapse('a', 'b', 0.5), eligibilityTrace: 1, chemicalLevel: 0 };
    updateWeight(s, config);
    assert.equal(s.weight, 0.5);
  });

  it('positive trace + positive chemical → weight increases', () => {
    const s = { ...createSynapse('a', 'b', 0.0), eligibilityTrace: 1, chemicalLevel: 1 };
    updateWeight(s, config);
    assert.ok(s.weight > 0);
  });

  it('positive trace + negative chemical → weight decreases', () => {
    const s = { ...createSynapse('a', 'b', 0.0), eligibilityTrace: 1, chemicalLevel: -1 };
    updateWeight(s, config);
    assert.ok(s.weight < 0);
  });

  it('weight clamped at +maxWeightMagnitude', () => {
    const s = { ...createSynapse('a', 'b', 1.9), eligibilityTrace: 1, chemicalLevel: 1 };
    updateWeight(s, config);
    assert.ok(s.weight <= config.maxWeightMagnitude);
  });

  it('weight clamped at -maxWeightMagnitude', () => {
    const s = { ...createSynapse('a', 'b', -1.9), eligibilityTrace: 1, chemicalLevel: -1 };
    updateWeight(s, config);
    assert.ok(s.weight >= -config.maxWeightMagnitude);
  });

  it('delta clamped at maxWeightDelta', () => {
    // trace=100, chemical=100 → rawDelta = 100*100*0.1 = 1000, but clamped to maxWeightDelta=1.0
    const s = { ...createSynapse('a', 'b', 0.0), eligibilityTrace: 100, chemicalLevel: 100 };
    updateWeight(s, config);
    assert.ok(s.weight <= config.maxWeightDelta);
  });

  it('weight decays when weightDecay > 0 and no learning signal', () => {
    const s = { ...createSynapse('a', 'b', 1.0), eligibilityTrace: 0, chemicalLevel: 0 };
    updateWeight(s, { ...config, weightDecay: 0.1 });
    assert.ok(s.weight < 1.0);
  });
});

describe('updateActivityHistory', () => {
  it('non-zero trace increases activity history', () => {
    const s = { ...createSynapse('a', 'b', 0), eligibilityTrace: 1, activityHistory: 0 };
    updateActivityHistory(s, config);
    assert.ok(s.activityHistory > 0);
  });

  it('zero trace slowly decays activity history', () => {
    const s = { ...createSynapse('a', 'b', 0), eligibilityTrace: 0, activityHistory: 0.5 };
    updateActivityHistory(s, config);
    assert.ok(s.activityHistory < 0.5);
  });

  it('history converges toward 1 when always active', () => {
    const s = { ...createSynapse('a', 'b', 0), eligibilityTrace: 1, activityHistory: 0 };
    for (let i = 0; i < 200; i++) updateActivityHistory(s, config);
    assert.ok(s.activityHistory > 0.99);
  });
});

describe('decayChemical', () => {
  it('chemical decays by chemicalDecayRate each step', () => {
    const s = { ...createSynapse('a', 'b', 0), chemicalLevel: 1.0 };
    decayChemical(s, config);
    assert.ok(Math.abs(s.chemicalLevel - config.chemicalDecayRate) < 1e-10);
  });

  it('chemical reaches near-zero after many steps', () => {
    const s = { ...createSynapse('a', 'b', 0), chemicalLevel: 1.0 };
    for (let i = 0; i < 50; i++) decayChemical(s, config);
    assert.ok(s.chemicalLevel < 1e-10);
  });

  it('negative chemical also decays toward zero', () => {
    const s = { ...createSynapse('a', 'b', 0), chemicalLevel: -1.0 };
    decayChemical(s, config);
    assert.ok(s.chemicalLevel > -1.0);
  });
});
