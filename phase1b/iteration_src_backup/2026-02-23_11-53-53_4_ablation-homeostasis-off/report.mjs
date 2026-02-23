// Generate ablation report in Markdown and HTML formats

import { writeFileSync, mkdirSync } from 'fs';

export function generateReport(allResults, config, startTime) {
  mkdirSync('./reports', { recursive: true });

  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  const stats = computeStats(allResults);

  writeFileSync('./reports/experiment-020.md', buildMarkdown(allResults, config, stats, duration));
  writeFileSync('./reports/experiment-020.html', buildHtml(allResults, config, stats, duration));

  console.log('\nReports written to ./reports/experiment-020.md and ./reports/experiment-020.html');
}

// ─── Statistics ───────────────────────────────────────────────────────────────

function computeStats(results) {
  const cursorAccs  = results.map(r => r.cursor.inference.meanAccuracy);
  const controlAccs = results.map(r => r.control.inference.meanAccuracy);

  const meanCursorAcc  = mean(cursorAccs);
  const meanControlAcc = mean(controlAccs);
  const cursorWins = results.filter(r =>
    r.cursor.inference.meanAccuracy > r.control.inference.meanAccuracy
  ).length;
  const seedsWithDistinct = results.filter(r => r.cursor.inference.distinctOutputs >= 2).length;

  // Average trajectory across seeds (align by episode index)
  const trajLength = results[0].cursor.trajectory.length;
  const avgTrajectory = [];
  for (let i = 0; i < trajLength; i++) {
    const ep = results[0].cursor.trajectory[i].episode;
    avgTrajectory.push({
      episode: ep,
      avgReward:   mean(results.map(r => r.cursor.trajectory[i].avgReward)),
      avgAcceptRate: mean(results.map(r => r.cursor.trajectory[i].acceptRate)),
    });
  }

  // Accept rate across all seeds and all trajectory points
  const allAcceptRates = results.flatMap(r => r.cursor.trajectory.map(t => t.acceptRate));
  const overallAcceptRate = mean(allAcceptRates);

  // Success criteria
  const success = {
    cursorBeatsControl:   meanCursorAcc > meanControlAcc,
    someDistinctOutputs:  seedsWithDistinct >= 1,
    acceptRateInRange:    overallAcceptRate >= 0.05 && overallAcceptRate <= 0.50,
  };

  return {
    meanCursorAcc,
    meanControlAcc,
    cursorWins,
    seedsWithDistinct,
    avgTrajectory,
    overallAcceptRate,
    success,
  };
}

function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

// ─── Markdown ─────────────────────────────────────────────────────────────────

function buildMarkdown(allResults, config, stats, duration) {
  const n = allResults.length;

  const perSeedRows = allResults.map(r => {
    const totalPert = r.cursor.totalAccepted + r.cursor.totalRejected;
    const acceptPct = totalPert > 0
      ? ((r.cursor.totalAccepted / totalPert) * 100).toFixed(1)
      : '—';
    return `| ${r.seed} ` +
      `| ${r.cursor.inference.meanAccuracy.toFixed(3)} ` +
      `| ${r.control.inference.meanAccuracy.toFixed(3)} ` +
      `| ${r.cursor.inference.distinctOutputs} ` +
      `| ${r.control.inference.distinctOutputs} ` +
      `| ${r.cursor.totalAccepted}/${r.cursor.totalRejected} ` +
      `| ${acceptPct}% ` +
      `| ${r.cursor.weightStart.toFixed(4)} → ${r.cursor.weightEnd.toFixed(4)} |`;
  }).join('\n');

  const trajRows = stats.avgTrajectory.map(t =>
    `| ${t.episode} | ${t.avgReward.toFixed(3)} | ${(t.avgAcceptRate * 100).toFixed(1)}% |`
  ).join('\n');

  const inferRows = allResults.map(r => {
    const patRows = r.cursor.inference.results.map((res, i) => {
      const ctrl = r.control.inference.results[i];
      return `| ${res.label} ` +
        `| [${res.input.join('')}] ` +
        `| [${res.target.join('')}] ` +
        `| [${res.output.join('')}] ${res.accuracy.toFixed(2)} ` +
        `| [${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)} |`;
    }).join('\n');
    return `\n**Seed ${r.seed}** (cursor distinct=${r.cursor.inference.distinctOutputs}, ` +
      `control distinct=${r.control.inference.distinctOutputs})\n\n` +
      `| Pat | Input | Target | Cursor out / acc | Control out / acc |\n` +
      `|-----|-------|--------|------------------|-------------------|\n` +
      patRows;
  }).join('\n');

  const sc = stats.success;

  return `# Experiment 020 — Iteration 4: Cursor Ablation (Homeostasis OFF)

**Date:** ${new Date().toISOString().slice(0, 10)}
**Duration:** ${duration}s

## Purpose

Determine whether the cursor mechanism learns independently of homeostasis.
Iterations 1–3 showed reward climbing to ~0.68, but iteration 2 (0% acceptance, cursor idle) reached the same level.
Homeostasis is the suspected driver. This run removes it entirely.

## Configuration

| Parameter | Value |
|---|---|
| Clusters | ${config.clusters} × ${config.neuronsPerCluster} neurons |
| Cursor radius | ${config.cursorRadius} |
| Perturbation std | ${config.perturbStd} |
| Acceptance | soft reward, strict > |
| Homeostasis | OFF |
| Episodes | ${config.episodes} |
| Steps / episode | ${config.stepsPerEpisode} |
| Seeds | ${config.seeds.join(', ')} |

## Summary

| Metric | Value |
|---|---|
| Mean cursor accuracy | ${stats.meanCursorAcc.toFixed(3)} |
| Mean control accuracy | ${stats.meanControlAcc.toFixed(3)} |
| Seeds where cursor > control | ${stats.cursorWins} / ${n} |
| Seeds with 2+ distinct outputs (cursor) | ${stats.seedsWithDistinct} / ${n} |
| Overall accept rate | ${(stats.overallAcceptRate * 100).toFixed(1)}% |

## Success Criteria

| Criterion | Result | Pass |
|---|---|---|
| Cursor > Control mean accuracy | ${stats.meanCursorAcc.toFixed(3)} vs ${stats.meanControlAcc.toFixed(3)} | ${sc.cursorBeatsControl ? '✅' : '❌'} |
| ≥1 seed with 2+ distinct outputs | ${stats.seedsWithDistinct} seed(s) | ${sc.someDistinctOutputs ? '✅' : '❌'} |
| Accept rate 5–50% | ${(stats.overallAcceptRate * 100).toFixed(1)}% | ${sc.acceptRateInRange ? '✅' : '❌'} |

## Per-Seed Results

| Seed | Cursor acc | Control acc | Distinct (cursor) | Distinct (ctrl) | Accepted/Reverted | Accept% | Mean |w| start→end |
|------|------------|-------------|-------------------|-----------------|-------------------|---------|----------------------|
${perSeedRows}

## Reward & Accept Rate Trajectory (averaged across ${n} seeds)

| Episode | Avg Reward | Accept Rate |
|---------|------------|-------------|
${trajRows}

## Inference Detail (per seed)

${inferRows}

## Conclusion

${buildConclusion(stats)}
`;
}

// ─── HTML ─────────────────────────────────────────────────────────────────────

function buildHtml(allResults, config, stats, duration) {
  const n = allResults.length;
  const sc = stats.success;

  // SVG reward + accept chart
  const chartW = 700, chartH = 280;
  const padL = 55, padR = 20, padT = 20, padB = 35;
  const w = chartW - padL - padR, h = chartH - padT - padB;

  const traj = stats.avgTrajectory;
  const rewardPoints = traj.map((t, i) => {
    const px = padL + (i / (traj.length - 1 || 1)) * w;
    const py = padT + (1 - t.avgReward) * h;
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');

  const acceptPoints = traj.map((t, i) => {
    const px = padL + (i / (traj.length - 1 || 1)) * w;
    const py = padT + (1 - t.avgAcceptRate) * h;
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');

  const perSeedRows = allResults.map(r => {
    const totalPert = r.cursor.totalAccepted + r.cursor.totalRejected;
    const acceptPct = totalPert > 0
      ? ((r.cursor.totalAccepted / totalPert) * 100).toFixed(1)
      : '—';
    const beats = r.cursor.inference.meanAccuracy > r.control.inference.meanAccuracy;
    return `<tr>
      <td>${r.seed}</td>
      <td>${r.cursor.inference.meanAccuracy.toFixed(3)}</td>
      <td>${r.control.inference.meanAccuracy.toFixed(3)}</td>
      <td>${beats ? '✅' : '—'}</td>
      <td>${r.cursor.inference.distinctOutputs}</td>
      <td>${r.control.inference.distinctOutputs}</td>
      <td>${r.cursor.totalAccepted} / ${r.cursor.totalRejected}</td>
      <td>${acceptPct}%</td>
      <td>${r.cursor.weightStart.toFixed(4)} → ${r.cursor.weightEnd.toFixed(4)}</td>
    </tr>`;
  }).join('\n');

  const inferSections = allResults.map(r => {
    const rows = r.cursor.inference.results.map((res, i) => {
      const ctrl = r.control.inference.results[i];
      return `<tr>
        <td>${res.label}</td>
        <td>[${res.input.join('')}]</td>
        <td>[${res.target.join('')}]</td>
        <td>[${res.output.join('')}]</td>
        <td>${res.accuracy.toFixed(2)}</td>
        <td>[${ctrl.output.join('')}]</td>
        <td>${ctrl.accuracy.toFixed(2)}</td>
      </tr>`;
    }).join('\n');
    return `<details>
      <summary><strong>Seed ${r.seed}</strong> — cursor distinct=${r.cursor.inference.distinctOutputs}, control distinct=${r.control.inference.distinctOutputs}, cursor acc=${r.cursor.inference.meanAccuracy.toFixed(3)}</summary>
      <table>
        <tr><th>Pat</th><th>Input</th><th>Target</th><th>Cursor out</th><th>Cursor acc</th><th>Control out</th><th>Control acc</th></tr>
        ${rows}
      </table>
    </details>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 020 — Iteration 4: Cursor Ablation</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 24px; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { margin-top: 2em; color: #444; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
  th, td { border: 1px solid #ccc; padding: 5px 10px; text-align: left; }
  th { background: #f0f0f0; }
  .meta { color: #666; font-size: 0.9em; }
  .conclusion { background: #f8f8f8; border-left: 4px solid #555; padding: 12px 16px; }
  .pass { color: #080; font-weight: bold; }
  .fail { color: #c00; font-weight: bold; }
  details { margin: 0.5em 0; border: 1px solid #ddd; padding: 8px 12px; border-radius: 4px; }
  summary { cursor: pointer; font-size: 0.95em; }
  svg text { font-family: system-ui, sans-serif; font-size: 11px; }
  .legend { display: flex; gap: 20px; font-size: 0.85em; margin-top: 4px; }
  .legend span::before { content: ''; display: inline-block; width: 20px; height: 3px; margin-right: 6px; vertical-align: middle; }
  .leg-reward::before { background: #2255cc; }
  .leg-accept::before { background: #cc5522; }
</style>
</head>
<body>
<h1>Experiment 020 — Iteration 4: Cursor Ablation (Homeostasis OFF)</h1>
<p class="meta">Date: ${new Date().toISOString().slice(0, 10)} &nbsp;|&nbsp; Duration: ${duration}s</p>

<h2>Purpose</h2>
<p>Determine whether the cursor mechanism learns independently of homeostasis.
Iterations 1–3 showed reward climbing to ~0.68, but iteration 2 (0% acceptance, cursor idle) reached the same level.
Homeostasis is the suspected driver. This run removes it entirely.</p>

<h2>Configuration</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Architecture</td><td>${config.clusters} clusters × ${config.neuronsPerCluster} neurons</td></tr>
  <tr><td>Cursor radius</td><td>${config.cursorRadius}</td></tr>
  <tr><td>Perturbation std</td><td>${config.perturbStd}</td></tr>
  <tr><td>Acceptance</td><td>soft reward, strict &gt;</td></tr>
  <tr><td>Homeostasis</td><td>OFF</td></tr>
  <tr><td>Episodes</td><td>${config.episodes}</td></tr>
  <tr><td>Steps / episode</td><td>${config.stepsPerEpisode}</td></tr>
  <tr><td>Seeds</td><td>${config.seeds.join(', ')}</td></tr>
</table>

<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Mean cursor accuracy</td><td>${stats.meanCursorAcc.toFixed(3)}</td></tr>
  <tr><td>Mean control accuracy</td><td>${stats.meanControlAcc.toFixed(3)}</td></tr>
  <tr><td>Seeds where cursor &gt; control</td><td>${stats.cursorWins} / ${n}</td></tr>
  <tr><td>Seeds with 2+ distinct outputs (cursor)</td><td>${stats.seedsWithDistinct} / ${n}</td></tr>
  <tr><td>Overall accept rate</td><td>${(stats.overallAcceptRate * 100).toFixed(1)}%</td></tr>
</table>

<h2>Success Criteria</h2>
<table>
  <tr><th>Criterion</th><th>Result</th><th>Pass</th></tr>
  <tr>
    <td>Cursor &gt; Control mean accuracy</td>
    <td>${stats.meanCursorAcc.toFixed(3)} vs ${stats.meanControlAcc.toFixed(3)}</td>
    <td class="${sc.cursorBeatsControl ? 'pass' : 'fail'}">${sc.cursorBeatsControl ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>≥1 seed with 2+ distinct outputs</td>
    <td>${stats.seedsWithDistinct} seed(s)</td>
    <td class="${sc.someDistinctOutputs ? 'pass' : 'fail'}">${sc.someDistinctOutputs ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>Accept rate 5–50%</td>
    <td>${(stats.overallAcceptRate * 100).toFixed(1)}%</td>
    <td class="${sc.acceptRateInRange ? 'pass' : 'fail'}">${sc.acceptRateInRange ? '✅ Yes' : '❌ No'}</td>
  </tr>
</table>

<h2>Reward &amp; Accept Rate Trajectory (avg across ${n} seeds)</h2>
<svg width="${chartW}" height="${chartH}" style="border:1px solid #ddd; background:#fafafa; display:block;">
  <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT+h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT+h}" x2="${padL+w}" y2="${padT+h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT+h/2}" x2="${padL+w}" y2="${padT+h/2}" stroke="#eee" stroke-dasharray="4"/>
  <text x="${padL-5}" y="${padT+4}" text-anchor="end">1.0</text>
  <text x="${padL-5}" y="${padT+h/2+4}" text-anchor="end">0.5</text>
  <text x="${padL-5}" y="${padT+h+4}" text-anchor="end">0.0</text>
  <text x="${padL}" y="${padT+h+20}" text-anchor="middle">0</text>
  <text x="${padL+w/2}" y="${padT+h+20}" text-anchor="middle">${Math.floor(config.episodes/2)}</text>
  <text x="${padL+w}" y="${padT+h+20}" text-anchor="middle">${config.episodes}</text>
  <text x="${padL+w/2}" y="${chartH}" text-anchor="middle" fill="#666">Episode</text>
  <polyline points="${rewardPoints}" fill="none" stroke="#2255cc" stroke-width="2"/>
  <polyline points="${acceptPoints}" fill="none" stroke="#cc5522" stroke-width="2" stroke-dasharray="6,3"/>
</svg>
<div class="legend">
  <span class="leg-reward">Avg reward (binary)</span>
  <span class="leg-accept">Accept rate (dashed)</span>
</div>

<h2>Per-Seed Results</h2>
<table>
  <tr>
    <th>Seed</th><th>Cursor acc</th><th>Control acc</th><th>Cursor wins</th>
    <th>Distinct (cursor)</th><th>Distinct (ctrl)</th>
    <th>Accepted / Reverted</th><th>Accept%</th><th>Mean |w| start→end</th>
  </tr>
  ${perSeedRows}
</table>

<h2>Inference Detail (per seed)</h2>
${inferSections}

<h2>Conclusion</h2>
<div class="conclusion">${buildConclusion(stats)}</div>

</body>
</html>`;
}

// ─── Conclusion ───────────────────────────────────────────────────────────────

function buildConclusion(stats) {
  const sc = stats.success;
  const passed = Object.values(sc).filter(Boolean).length;
  const total = Object.values(sc).length;

  if (passed === total) {
    return `All ${total} success criteria passed. The cursor mechanism demonstrates independent learning ` +
      `without homeostasis. Mean cursor accuracy (${stats.meanCursorAcc.toFixed(3)}) exceeds the random ` +
      `initialisation baseline (${stats.meanControlAcc.toFixed(3)}), the accept rate ` +
      `(${(stats.overallAcceptRate * 100).toFixed(1)}%) is within the meaningful 5–50% range, and ` +
      `at least ${stats.seedsWithDistinct} seed(s) produced discriminative outputs. ` +
      `The cursor perturbation mechanism works independently of homeostasis.`;
  } else if (!sc.cursorBeatsControl) {
    return `Cursor (${stats.meanCursorAcc.toFixed(3)}) did not exceed Control ` +
      `(${stats.meanControlAcc.toFixed(3)}). The cursor mechanism is not producing useful weight ` +
      `changes on this task in ${5000} episodes. The perturbation strategy may need rethinking ` +
      `— consider larger perturbation magnitude, task simplification (e.g. identity mapping), ` +
      `or a different cursor movement strategy.`;
  } else if (!sc.acceptRateInRange) {
    const rate = (stats.overallAcceptRate * 100).toFixed(1);
    if (stats.overallAcceptRate < 0.05) {
      return `Cursor beat Control but accept rate (${rate}%) is below 5% — selection pressure is too high ` +
        `or perturbations are too destructive. Most accepted steps happen by chance in favourable ` +
        `cursor positions. Reduce perturbation std or use a larger cursor radius.`;
    } else {
      return `Cursor beat Control but accept rate (${rate}%) exceeds 50% — acceptance is too permissive ` +
        `and most accepted perturbations are neutral. Tighten the acceptance criterion or reduce ` +
        `cursor radius to force more focused perturbations.`;
    }
  } else {
    return `Partial success (${passed}/${total} criteria). Cursor showed some improvement over Control ` +
      `but not across all dimensions. Review inference detail for per-seed patterns.`;
  }
}
