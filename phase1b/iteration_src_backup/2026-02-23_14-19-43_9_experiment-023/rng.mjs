// Mulberry32 PRNG — patches Math.random() globally for reproducible runs.
// Call seedRandom(seed) before any operation that must be reproducible.

export function seedRandom(seed) {
  let s = (seed >>> 0) + 1; // +1 avoids degenerate 0-seed behaviour
  Math.random = function () {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
  };
}
