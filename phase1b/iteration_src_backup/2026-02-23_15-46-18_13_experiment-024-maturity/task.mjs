// Training patterns: 5-bit input → 5-bit target output
// Fixed arbitrary mappings the network must learn

export const trainingPatterns = [
  { input: [0,0,0,0,0], target: [1,0,1,0,1] },
  { input: [1,0,0,0,0], target: [0,1,0,1,0] },
  { input: [0,1,0,0,0], target: [1,1,0,0,1] },
  { input: [0,0,1,0,0], target: [0,0,1,1,0] },
  { input: [0,0,0,1,0], target: [1,0,0,1,1] },
  { input: [0,0,0,0,1], target: [0,1,1,0,0] },
  { input: [1,1,0,0,0], target: [0,0,1,0,1] },
  { input: [0,0,1,1,0], target: [1,1,0,1,0] },
];

export function randomPattern() {
  const idx = Math.floor(Math.random() * trainingPatterns.length);
  return { ...trainingPatterns[idx], patternIdx: idx };
}
