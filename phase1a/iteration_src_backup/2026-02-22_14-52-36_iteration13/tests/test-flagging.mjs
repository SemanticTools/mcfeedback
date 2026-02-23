import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computeEligibility } from '../src/flagging.mjs';

const config = {
  coActivationStrength: 1.0,
  coSilenceStrength:    0.5,
  mismatchStrength:    -0.5,
  ambientThreshold:     0.3,
};

const state = (fired, ambientField = 0) => ({ firedThisCycle: fired, ambientField });

describe('computeEligibility', () => {
  it('co-activation → positive trace', () => {
    const trace = computeEligibility(state(true), state(true), config);
    assert.equal(trace, config.coActivationStrength);
  });

  it('co-silence with high ambient → positive trace', () => {
    const trace = computeEligibility(state(false), state(false, 0.5), config);
    assert.equal(trace, config.coSilenceStrength);
  });

  it('co-silence with low ambient → zero', () => {
    const trace = computeEligibility(state(false), state(false, 0.1), config);
    assert.equal(trace, 0);
  });

  it('co-silence at exact threshold → zero (not strictly greater)', () => {
    const trace = computeEligibility(state(false), state(false, config.ambientThreshold), config);
    assert.equal(trace, 0);
  });

  it('pre fires, post silent → negative trace', () => {
    const trace = computeEligibility(state(true), state(false), config);
    assert.equal(trace, config.mismatchStrength);
  });

  it('pre silent, post fires → negative trace', () => {
    const trace = computeEligibility(state(false), state(true), config);
    assert.equal(trace, config.mismatchStrength);
  });

  it('trace magnitudes match config strengths', () => {
    assert.equal(Math.abs(computeEligibility(state(true), state(true), config)), config.coActivationStrength);
    assert.equal(Math.abs(computeEligibility(state(true), state(false), config)), Math.abs(config.mismatchStrength));
    assert.equal(Math.abs(computeEligibility(state(false), state(false, 0.5), config)), config.coSilenceStrength);
  });
});
