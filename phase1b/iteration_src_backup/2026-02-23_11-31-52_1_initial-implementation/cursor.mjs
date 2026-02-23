// Cursor: a 3D position in network space with a radius.
// Eligible synapses are those whose pre OR post neuron falls within the radius.
// Moves by random walk; occasionally jumps to a new random location.

const CURSOR_RADIUS = 4;
const WALK_STEP = 1.5;
const JUMP_PROBABILITY = 0.05;

export function createCursor(network) {
  const { neurons } = network;

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (const n of neurons) {
    if (n.x < minX) minX = n.x;
    if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.y > maxY) maxY = n.y;
    if (n.z < minZ) minZ = n.z;
    if (n.z > maxZ) maxZ = n.z;
  }

  const bounds = { minX, maxX, minY, maxY, minZ, maxZ };

  return {
    x: (minX + maxX) / 2,
    y: (minY + maxY) / 2,
    z: (minZ + maxZ) / 2,
    radius: CURSOR_RADIUS,
    bounds,
  };
}

export function moveCursor(cursor) {
  const { bounds } = cursor;

  if (Math.random() < JUMP_PROBABILITY) {
    cursor.x = bounds.minX + Math.random() * (bounds.maxX - bounds.minX);
    cursor.y = bounds.minY + Math.random() * (bounds.maxY - bounds.minY);
    cursor.z = bounds.minZ + Math.random() * (bounds.maxZ - bounds.minZ);
  } else {
    cursor.x += (Math.random() - 0.5) * 2 * WALK_STEP;
    cursor.y += (Math.random() - 0.5) * 2 * WALK_STEP;
    cursor.z += (Math.random() - 0.5) * 2 * WALK_STEP;

    cursor.x = Math.max(bounds.minX, Math.min(bounds.maxX, cursor.x));
    cursor.y = Math.max(bounds.minY, Math.min(bounds.maxY, cursor.y));
    cursor.z = Math.max(bounds.minZ, Math.min(bounds.maxZ, cursor.z));
  }
}

export function getEligibleSynapses(cursor, network) {
  const { neurons, synapses } = network;
  const { x, y, z, radius } = cursor;
  const r2 = radius * radius;

  const inRange = new Uint8Array(neurons.length);
  for (const n of neurons) {
    const dx = n.x - x, dy = n.y - y, dz = n.z - z;
    inRange[n.id] = (dx * dx + dy * dy + dz * dz) <= r2 ? 1 : 0;
  }

  return synapses.filter(s => inRange[s.pre] || inRange[s.post]);
}
