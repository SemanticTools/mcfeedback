import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { distance3d, randomInRange, shuffleArray, generateId } from '../src/utils.mjs';

describe('distance3d', () => {
  it('3-4-5 triangle', () => {
    assert.equal(distance3d({ x: 0, y: 0, z: 0 }, { x: 3, y: 4, z: 0 }), 5);
  });

  it('same point is zero', () => {
    assert.equal(distance3d({ x: 1, y: 2, z: 3 }, { x: 1, y: 2, z: 3 }), 0);
  });

  it('uses all three axes', () => {
    const d = distance3d({ x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 1 });
    assert.ok(Math.abs(d - Math.sqrt(3)) < 1e-10);
  });
});

describe('randomInRange', () => {
  it('stays within bounds over many samples', () => {
    for (let i = 0; i < 1000; i++) {
      const v = randomInRange(-2, 5);
      assert.ok(v >= -2 && v <= 5);
    }
  });
});

describe('shuffleArray', () => {
  it('returns same length', () => {
    const a = [1, 2, 3, 4, 5];
    assert.equal(shuffleArray(a).length, a.length);
  });

  it('does not mutate original', () => {
    const a = [1, 2, 3];
    shuffleArray(a);
    assert.deepEqual(a, [1, 2, 3]);
  });

  it('contains same elements', () => {
    const a = [1, 2, 3, 4, 5];
    assert.deepEqual(shuffleArray(a).sort((x, y) => x - y), a);
  });
});

describe('generateId', () => {
  it('includes prefix', () => {
    assert.ok(generateId('foo').startsWith('foo'));
  });

  it('generates unique ids', () => {
    const ids = new Set(Array.from({ length: 100 }, () => generateId('x')));
    assert.equal(ids.size, 100);
  });
});
