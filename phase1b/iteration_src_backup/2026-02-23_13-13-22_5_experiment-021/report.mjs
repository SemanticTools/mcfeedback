// Experiment 021 report generator — Markdown + HTML
// Three conditions: astrocyte | cursor | control
// Includes per-astrocyte diagnostics and pattern-specialisation analysis

import { writeFileSync, mkdirSync } from 'fs';
import { trainingPatterns } from './task.mjs';

export function generateReport(allResults, config, startTime) {
  mkdirSync('./reports', { recursive: true });

  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  const stats    = computeStats(allResults, config);

  writeFileSync('./reports/experiment-021.md',   buildMarkdown(allResults, config, stats, duration));
  writeFileSync('./reports/experiment-021.html',  buildHtml(allResults, config, stats, duration));

  console.log('\nReports written to ./reports/experiment-021.md and ./reports/experiment-021.html');
}

// ─── Statistics ───────────────────────────────────────────────────────────────

function computeStats(results, config) {
  const n = results.length;

  const astAccs     = results.map(r => r.astrocyte.inference.meanAccuracy);
  const cursorAccs  = results.map(r => r.cursor.inference.meanAccuracy);
  const controlAccs = results.map(r => r.control.inference.meanAccuracy);

  const meanAst     = mean(astAccs);
  const meanCursor  = mean(cursorAccs);
  const meanControl = mean(controlAccs);

  const varAst    = variance(astAccs);
  const varCursor = variance(cursorAccs);

  const astWinsCursor  = results.filter(r => r.astrocyte.inference.meanAccuracy > r.cursor.inference.meanAccuracy).length;
  const astWinsControl = results.filter(r => r.astrocyte.inference.meanAccuracy > r.control.inference.meanAccuracy).length;

  // Accept rates
  const astAcceptRates    = results.map(r => {
    const t = r.astrocyte.totalAccepted + r.astrocyte.totalRejected;
    return t > 0 ? r.astrocyte.totalAccepted / t : 0;
  });
  const cursorAcceptRates = results.map(r => {
    const t = r.cursor.totalAccepted + r.cursor.totalRejected;
    return t > 0 ? r.cursor.totalAccepted / t : 0;
  });
  const meanAstAccept    = mean(astAcceptRates);
  const meanCursorAccept = mean(cursorAcceptRates);

  // Pattern specialisation: for each seed, check if any astrocyte activates >2× mean for any pattern
  let specialisationCount = 0; // seeds with ≥1 specialised astrocyte
  let globalSpecialisedAstrocytes = 0;
  for (const r of results) {
    if (!r.astrocyte.astrocyteStats) continue;
    let found = false;
    for (const ast of r.astrocyte.astrocyteStats) {
      const pActs = ast.activationsByPattern;
      const total = pActs.reduce((s, v) => s + v, 0);
      if (total === 0) continue;
      const m = total / pActs.length;
      const maxAct = Math.max(...pActs);
      if (maxAct > 2 * m) { found = true; globalSpecialisedAstrocytes++; }
    }
    if (found) specialisationCount++;
  }

  // Average trajectory across seeds (aligned by index)
  const astTraj    = averageTrajectory(results.map(r => r.astrocyte.trajectory));
  const cursorTraj = averageTrajectory(results.map(r => r.cursor.trajectory));

  // Success criteria
  const success = {
    astBeatsCursor:   meanAst > meanCursor,
    lowerVariance:    varAst < varCursor,
    specialisation:   specialisationCount >= 1,
    higherAcceptRate: meanAstAccept > meanCursorAccept,
  };

  return {
    n, meanAst, meanCursor, meanControl,
    varAst, varCursor,
    astWinsCursor, astWinsControl,
    meanAstAccept, meanCursorAccept,
    specialisationCount, globalSpecialisedAstrocytes,
    astTraj, cursorTraj,
    success,
  };
}

function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function variance(arr) {
  const m = mean(arr);
  return mean(arr.map(v => (v - m) ** 2));
}

function averageTrajectory(trajectories) {
  // Filter out empty trajectories (control)
  const valid = trajectories.filter(t => t.length > 0);
  if (valid.length === 0) return [];
  const len = valid[0].length;
  return Array.from({ length: len }, (_, i) => ({
    episode:     valid[0][i].episode,
    avgReward:   mean(valid.map(t => t[i].avgReward)),
    avgAcceptRate: mean(valid.map(t => t[i].acceptRate)),
  }));
}

// ─── Markdown ─────────────────────────────────────────────────────────────────

function buildMarkdown(allResults, config, stats, duration) {
  const n = stats.n;
  const sc = stats.success;

  const perSeedRows = allResults.map(r => {
    const astAcc  = (r.astrocyte.totalAccepted / (r.astrocyte.totalAccepted + r.astrocyte.totalRejected || 1) * 100).toFixed(1);
    const curAcc  = (r.cursor.totalAccepted    / (r.cursor.totalAccepted    + r.cursor.totalRejected    || 1) * 100).toFixed(1);
    const astBeat = r.astrocyte.inference.meanAccuracy > r.cursor.inference.meanAccuracy ? '✅' : '—';
    return `| ${r.seed} ` +
      `| ${r.astrocyte.inference.meanAccuracy.toFixed(3)} (${r.astrocyte.inference.distinctOutputs}d) ` +
      `| ${r.cursor.inference.meanAccuracy.toFixed(3)} (${r.cursor.inference.distinctOutputs}d) ` +
      `| ${r.control.inference.meanAccuracy.toFixed(3)} ` +
      `| ${astBeat} ` +
      `| ${r.astrocyte.totalAccepted}/${r.astrocyte.totalRejected} (${astAcc}%) ` +
      `| ${r.cursor.totalAccepted}/${r.cursor.totalRejected} (${curAcc}%) |`;
  }).join('\n');

  const trajRows = stats.astTraj.map((t, i) => {
    const ct = stats.cursorTraj[i];
    return `| ${t.episode} | ${t.avgReward.toFixed(3)} | ${(t.avgAcceptRate * 100).toFixed(1)}% ` +
      `| ${ct ? ct.avgReward.toFixed(3) : '—'} | ${ct ? (ct.avgAcceptRate * 100).toFixed(1) + '%' : '—'} |`;
  }).join('\n');

  const patLabels = trainingPatterns.map((_, i) => `P${i + 1}`).join(' | ');

  const astroSections = allResults.map(r => {
    if (!r.astrocyte.astrocyteStats) return '';
    const rows = r.astrocyte.astrocyteStats.map(ast => {
      const sr = ast.activationCount > 0
        ? (ast.successCount / ast.activationCount * 100).toFixed(1) + '%'
        : '—';
      const pCounts = ast.activationsByPattern.join(' | ');
      const total = ast.activationsByPattern.reduce((s, v) => s + v, 0);
      const m = total > 0 ? total / ast.activationsByPattern.length : 0;
      const maxAct = Math.max(...ast.activationsByPattern);
      const specialised = maxAct > 2 * m ? ' ★' : '';
      return `| ${ast.id} | C${ast.cluster} | (${ast.position.x.toFixed(1)}, ${ast.position.y.toFixed(1)}) ` +
        `| ${ast.neuronCount}n / ${ast.synapseCount}s ` +
        `| ${ast.activationCount} | ${sr} | ${ast.finalThreshold.toFixed(3)}${specialised} ` +
        `| ${pCounts} |`;
    }).join('\n');

    return `\n**Seed ${r.seed}**\n\n` +
      `| ID | Cluster | Position | Territory | Activations | Success% | Final thresh | ${patLabels} |\n` +
      `|----|---------|----|---------|-------------|----------|--------------|${trainingPatterns.map(() => '---').join('|')}|\n` +
      rows;
  }).join('\n');

  return `# Experiment 021 — Astrocyte-Mediated Learning

**Date:** ${new Date().toISOString().slice(0, 10)}
**Duration:** ${duration}s

## Hypothesis

Replacing the random wandering cursor with biologically-motivated astrocytes (fixed territories, activity-dependent activation, self-adapting thresholds) should improve over random spatial perturbation. Input-pattern-driven activation should route perturbations to the synapses actually involved in each pattern's representation.

## Configuration

| Parameter | Value |
|---|---|
| Clusters | ${config.clusters} × ${config.neuronsPerCluster} neurons |
| Astrocytes | ${config.astrocytesPerCluster} per cluster (${config.astrocytesPerCluster * 2} total) |
| Territory radius | ${config.territoryRadius} (2D x-y) |
| Cursor radius (baseline) | ${config.cursorRadius} |
| Perturbation std | ${config.perturbStd} |
| Acceptance | soft reward, strict > |
| Homeostasis | OFF |
| Episodes | ${config.episodes} × ${config.stepsPerEpisode} steps |
| Seeds | ${config.seeds.join(', ')} |

## Summary

| Metric | Astrocyte | Cursor | Control |
|---|---|---|---|
| Mean accuracy | ${stats.meanAst.toFixed(3)} | ${stats.meanCursor.toFixed(3)} | ${stats.meanControl.toFixed(3)} |
| Variance | ${stats.varAst.toFixed(5)} | ${stats.varCursor.toFixed(5)} | — |
| Mean accept rate | ${(stats.meanAstAccept * 100).toFixed(1)}% | ${(stats.meanCursorAccept * 100).toFixed(1)}% | — |
| Seeds beating cursor | ${stats.astWinsCursor} / ${n} | — | — |

## Success Criteria

| Criterion | Result | Pass |
|---|---|---|
| Astrocyte > Cursor mean accuracy | ${stats.meanAst.toFixed(3)} vs ${stats.meanCursor.toFixed(3)} | ${sc.astBeatsCursor ? '✅' : '❌'} |
| Astrocyte lower variance than Cursor | ${stats.varAst.toFixed(5)} vs ${stats.varCursor.toFixed(5)} | ${sc.lowerVariance ? '✅' : '❌'} |
| ≥1 astrocyte shows pattern specialisation (>2× activation) | ${stats.specialisationCount} seed(s) | ${sc.specialisation ? '✅' : '❌'} |
| Astrocyte accept rate > Cursor accept rate | ${(stats.meanAstAccept * 100).toFixed(1)}% vs ${(stats.meanCursorAccept * 100).toFixed(1)}% | ${sc.higherAcceptRate ? '✅' : '❌'} |

## Per-Seed Results

| Seed | Astrocyte acc (distinct) | Cursor acc (distinct) | Control acc | Ast beats cur | Ast keep/revert (%) | Cursor keep/revert (%) |
|------|--------------------------|----------------------|-------------|---------------|---------------------|------------------------|
${perSeedRows}

## Trajectory (avg across ${n} seeds)

| Episode | Ast reward | Ast accept% | Cursor reward | Cursor accept% |
|---------|-----------|-------------|--------------|----------------|
${trajRows}

## Per-Astrocyte Diagnostics (★ = pattern specialisation >2× mean)

${astroSections}

## Inference Detail

${allResults.map(r => {
  const rows = r.astrocyte.inference.results.map((res, i) => {
    const cur  = r.cursor.inference.results[i];
    const ctrl = r.control.inference.results[i];
    return `| ${res.label} | [${res.input.join('')}] | [${res.target.join('')}] ` +
      `| [${res.output.join('')}] ${res.accuracy.toFixed(2)} ` +
      `| [${cur.output.join('')}] ${cur.accuracy.toFixed(2)} ` +
      `| [${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)} |`;
  }).join('\n');
  return `\n**Seed ${r.seed}**\n\n` +
    `| Pat | Input | Target | Astrocyte out/acc | Cursor out/acc | Control out/acc |\n` +
    `|-----|-------|--------|-------------------|----------------|----------------|\n` + rows;
}).join('\n')}

## Conclusion

${buildConclusion(stats)}
`;
}

// ─── HTML ─────────────────────────────────────────────────────────────────────

function buildHtml(allResults, config, stats, duration) {
  const n  = stats.n;
  const sc = stats.success;

  // SVG chart (reward trajectory)
  const chartW = 720, chartH = 280;
  const padL = 55, padR = 20, padT = 20, padB = 35;
  const w = chartW - padL - padR, h = chartH - padT - padB;

  function toPoly(traj) {
    if (!traj || traj.length === 0) return '';
    return traj.map((t, i) => {
      const px = padL + (i / (traj.length - 1 || 1)) * w;
      const py = padT + (1 - t.avgReward) * h;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');
  }
  function toAcceptPoly(traj) {
    if (!traj || traj.length === 0) return '';
    return traj.map((t, i) => {
      const px = padL + (i / (traj.length - 1 || 1)) * w;
      const py = padT + (1 - t.avgAcceptRate) * h;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');
  }

  const astReward   = toPoly(stats.astTraj);
  const curReward   = toPoly(stats.cursorTraj);
  const astAccept   = toAcceptPoly(stats.astTraj);
  const curAccept   = toAcceptPoly(stats.cursorTraj);

  const perSeedRows = allResults.map(r => {
    const astRate = (r.astrocyte.totalAccepted / (r.astrocyte.totalAccepted + r.astrocyte.totalRejected || 1) * 100).toFixed(1);
    const curRate = (r.cursor.totalAccepted    / (r.cursor.totalAccepted    + r.cursor.totalRejected    || 1) * 100).toFixed(1);
    const beat = r.astrocyte.inference.meanAccuracy > r.cursor.inference.meanAccuracy;
    return `<tr>
      <td>${r.seed}</td>
      <td>${r.astrocyte.inference.meanAccuracy.toFixed(3)} (${r.astrocyte.inference.distinctOutputs}d)</td>
      <td>${r.cursor.inference.meanAccuracy.toFixed(3)} (${r.cursor.inference.distinctOutputs}d)</td>
      <td>${r.control.inference.meanAccuracy.toFixed(3)}</td>
      <td>${beat ? '✅' : '—'}</td>
      <td>${r.astrocyte.totalAccepted}/${r.astrocyte.totalRejected} (${astRate}%)</td>
      <td>${r.cursor.totalAccepted}/${r.cursor.totalRejected} (${curRate}%)</td>
    </tr>`;
  }).join('\n');

  const patLabels = trainingPatterns.map((_, i) => `<th>P${i + 1}</th>`).join('');

  const astroDiagnostics = allResults.map(r => {
    if (!r.astrocyte.astrocyteStats) return '';
    const rows = r.astrocyte.astrocyteStats.map(ast => {
      const sr = ast.activationCount > 0
        ? (ast.successCount / ast.activationCount * 100).toFixed(1) + '%'
        : '—';
      const total = ast.activationsByPattern.reduce((s, v) => s + v, 0);
      const m = total > 0 ? total / ast.activationsByPattern.length : 0;
      const maxAct = Math.max(...ast.activationsByPattern);
      const spec = maxAct > 2 * m;
      const pCells = ast.activationsByPattern.map(v => {
        const highlight = (total > 0 && v > 2 * m) ? ' style="background:#ffe0b2;font-weight:bold"' : '';
        return `<td${highlight}>${v}</td>`;
      }).join('');
      return `<tr${spec ? ' style="background:#fff3e0"' : ''}>
        <td>${ast.id}${spec ? ' ★' : ''}</td>
        <td>C${ast.cluster}</td>
        <td>(${ast.position.x.toFixed(1)}, ${ast.position.y.toFixed(1)})</td>
        <td>${ast.neuronCount}n / ${ast.synapseCount}s</td>
        <td>${ast.activationCount}</td>
        <td>${sr}</td>
        <td>${ast.finalThreshold.toFixed(3)}</td>
        ${pCells}
      </tr>`;
    }).join('\n');
    return `<details>
      <summary><strong>Seed ${r.seed}</strong> — astrocyte acc=${r.astrocyte.inference.meanAccuracy.toFixed(3)}, distinct=${r.astrocyte.inference.distinctOutputs}</summary>
      <table>
        <tr><th>ID</th><th>Cluster</th><th>Position</th><th>Territory</th><th>Activations</th><th>Success%</th><th>Final thresh</th>${patLabels}</tr>
        ${rows}
      </table>
    </details>`;
  }).join('\n');

  const inferenceDetail = allResults.map(r => {
    const rows = r.astrocyte.inference.results.map((res, i) => {
      const cur  = r.cursor.inference.results[i];
      const ctrl = r.control.inference.results[i];
      return `<tr>
        <td>${res.label}</td>
        <td>[${res.input.join('')}]</td>
        <td>[${res.target.join('')}]</td>
        <td>[${res.output.join('')}] ${res.accuracy.toFixed(2)}</td>
        <td>[${cur.output.join('')}] ${cur.accuracy.toFixed(2)}</td>
        <td>[${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)}</td>
      </tr>`;
    }).join('\n');
    return `<details>
      <summary><strong>Seed ${r.seed}</strong> — ast=${r.astrocyte.inference.meanAccuracy.toFixed(3)}, cursor=${r.cursor.inference.meanAccuracy.toFixed(3)}, control=${r.control.inference.meanAccuracy.toFixed(3)}</summary>
      <table>
        <tr><th>Pat</th><th>Input</th><th>Target</th><th>Astrocyte out/acc</th><th>Cursor out/acc</th><th>Control out/acc</th></tr>
        ${rows}
      </table>
    </details>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 021 — Astrocyte-Mediated Learning</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 980px; margin: 40px auto; padding: 0 24px; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { margin-top: 2em; color: #444; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }
  th, td { border: 1px solid #ccc; padding: 4px 9px; text-align: left; }
  th { background: #f0f0f0; }
  .meta { color: #666; font-size: 0.9em; }
  .pass { color: #080; font-weight: bold; }
  .fail { color: #c00; font-weight: bold; }
  .conclusion { background: #f8f8f8; border-left: 4px solid #555; padding: 12px 16px; }
  details { margin: 0.4em 0; border: 1px solid #ddd; padding: 6px 12px; border-radius: 4px; }
  summary { cursor: pointer; }
  svg text { font-family: system-ui, sans-serif; font-size: 11px; }
  .legend { display: flex; gap: 24px; font-size: 0.85em; margin-top: 4px; }
  .legend span::before { content: ''; display: inline-block; width: 22px; height: 3px; margin-right: 6px; vertical-align: middle; }
  .leg-ast::before    { background: #2255cc; }
  .leg-cursor::before { background: #cc5522; }
</style>
</head>
<body>
<h1>Experiment 021 — Astrocyte-Mediated Learning</h1>
<p class="meta">Date: ${new Date().toISOString().slice(0, 10)} &nbsp;|&nbsp; Duration: ${duration}s</p>

<h2>Hypothesis</h2>
<p>Replacing the random cursor with biologically-motivated astrocytes (fixed territories, activity-dependent activation, adaptive thresholds) should outperform random spatial perturbation. Input-driven activation routes perturbations to synapses involved in each pattern's representation, resolving pattern-conflict issues inherent to random search.</p>

<h2>Configuration</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Architecture</td><td>${config.clusters} clusters × ${config.neuronsPerCluster} neurons</td></tr>
  <tr><td>Astrocytes</td><td>${config.astrocytesPerCluster} per cluster (${config.astrocytesPerCluster * 2} total), territory radius ${config.territoryRadius}</td></tr>
  <tr><td>Cursor radius (baseline)</td><td>${config.cursorRadius}</td></tr>
  <tr><td>Perturbation std</td><td>${config.perturbStd}</td></tr>
  <tr><td>Acceptance</td><td>soft reward, strict &gt;</td></tr>
  <tr><td>Homeostasis</td><td>OFF</td></tr>
  <tr><td>Episodes</td><td>${config.episodes} × ${config.stepsPerEpisode} steps</td></tr>
  <tr><td>Seeds</td><td>${config.seeds.join(', ')}</td></tr>
</table>

<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Astrocyte</th><th>Cursor</th><th>Control</th></tr>
  <tr><td>Mean accuracy</td><td>${stats.meanAst.toFixed(3)}</td><td>${stats.meanCursor.toFixed(3)}</td><td>${stats.meanControl.toFixed(3)}</td></tr>
  <tr><td>Variance</td><td>${stats.varAst.toFixed(5)}</td><td>${stats.varCursor.toFixed(5)}</td><td>—</td></tr>
  <tr><td>Mean accept rate</td><td>${(stats.meanAstAccept * 100).toFixed(1)}%</td><td>${(stats.meanCursorAccept * 100).toFixed(1)}%</td><td>—</td></tr>
  <tr><td>Seeds beating cursor</td><td>${stats.astWinsCursor} / ${n}</td><td>—</td><td>—</td></tr>
</table>

<h2>Success Criteria</h2>
<table>
  <tr><th>Criterion</th><th>Result</th><th>Pass</th></tr>
  <tr>
    <td>Astrocyte &gt; Cursor mean accuracy</td>
    <td>${stats.meanAst.toFixed(3)} vs ${stats.meanCursor.toFixed(3)}</td>
    <td class="${sc.astBeatsCursor ? 'pass' : 'fail'}">${sc.astBeatsCursor ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>Astrocyte lower variance than Cursor</td>
    <td>${stats.varAst.toFixed(5)} vs ${stats.varCursor.toFixed(5)}</td>
    <td class="${sc.lowerVariance ? 'pass' : 'fail'}">${sc.lowerVariance ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>≥1 astrocyte shows pattern specialisation (&gt;2× activation)</td>
    <td>${stats.specialisationCount} seed(s) with specialised astrocytes</td>
    <td class="${sc.specialisation ? 'pass' : 'fail'}">${sc.specialisation ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>Astrocyte accept rate &gt; Cursor accept rate</td>
    <td>${(stats.meanAstAccept * 100).toFixed(1)}% vs ${(stats.meanCursorAccept * 100).toFixed(1)}%</td>
    <td class="${sc.higherAcceptRate ? 'pass' : 'fail'}">${sc.higherAcceptRate ? '✅ Yes' : '❌ No'}</td>
  </tr>
</table>

<h2>Reward &amp; Accept Rate Trajectories (avg across ${n} seeds)</h2>
<svg width="${chartW}" height="${chartH}" style="border:1px solid #ddd;background:#fafafa;display:block;">
  <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT+h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT+h}" x2="${padL+w}" y2="${padT+h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT+h/2}" x2="${padL+w}" y2="${padT+h/2}" stroke="#eee" stroke-dasharray="4"/>
  <text x="${padL-5}" y="${padT+4}"      text-anchor="end">1.0</text>
  <text x="${padL-5}" y="${padT+h/2+4}" text-anchor="end">0.5</text>
  <text x="${padL-5}" y="${padT+h+4}"   text-anchor="end">0.0</text>
  <text x="${padL}"     y="${padT+h+20}" text-anchor="middle">0</text>
  <text x="${padL+w/2}" y="${padT+h+20}" text-anchor="middle">${Math.floor(config.episodes / 2)}</text>
  <text x="${padL+w}"   y="${padT+h+20}" text-anchor="middle">${config.episodes}</text>
  <text x="${padL+w/2}" y="${chartH}"    text-anchor="middle" fill="#666">Episode</text>
  ${astReward   ? `<polyline points="${astReward}"  fill="none" stroke="#2255cc" stroke-width="2"/>` : ''}
  ${curReward   ? `<polyline points="${curReward}"  fill="none" stroke="#cc5522" stroke-width="1.5" stroke-dasharray="6,3"/>` : ''}
  ${astAccept   ? `<polyline points="${astAccept}"  fill="none" stroke="#2255cc" stroke-width="1.5" opacity="0.4"/>` : ''}
  ${curAccept   ? `<polyline points="${curAccept}"  fill="none" stroke="#cc5522" stroke-width="1"   opacity="0.4" stroke-dasharray="6,3"/>` : ''}
</svg>
<div class="legend">
  <span class="leg-ast">Astrocyte reward (solid) / accept rate (faded)</span>
  <span class="leg-cursor">Cursor reward (dashed) / accept rate (faded dashed)</span>
</div>

<h2>Per-Seed Results</h2>
<table>
  <tr>
    <th>Seed</th>
    <th>Astrocyte acc (d)</th><th>Cursor acc (d)</th><th>Control acc</th>
    <th>Ast&gt;Cur</th>
    <th>Ast keep/revert (%)</th><th>Cursor keep/revert (%)</th>
  </tr>
  ${perSeedRows}
</table>

<h2>Per-Astrocyte Diagnostics <small>(★ = pattern specialisation &gt;2× mean; highlighted cells = dominant pattern)</small></h2>
${astroDiagnostics}

<h2>Inference Detail</h2>
${inferenceDetail}

<h2>Conclusion</h2>
<div class="conclusion">${buildConclusion(stats)}</div>

</body>
</html>`;
}

// ─── Conclusion ───────────────────────────────────────────────────────────────

function buildConclusion(stats) {
  const sc = stats.success;
  const passed = Object.values(sc).filter(Boolean).length;
  const total  = Object.values(sc).length;

  const lines = [];

  if (sc.astBeatsCursor) {
    lines.push(`Astrocyte condition (${stats.meanAst.toFixed(3)}) outperformed the random cursor baseline ` +
      `(${stats.meanCursor.toFixed(3)}) in mean inference accuracy across ${stats.n} seeds ` +
      `(${stats.astWinsCursor}/${stats.n} individual seeds). ` +
      `Activity-sensing and fixed territories add value beyond random spatial perturbation.`);
  } else {
    lines.push(`Astrocyte condition (${stats.meanAst.toFixed(3)}) did not surpass the random cursor ` +
      `(${stats.meanCursor.toFixed(3)}). Activity-sensing is not yet contributing sufficient advantage. ` +
      `Examine whether astrocytes are differentiating between patterns or activating uniformly.`);
  }

  if (sc.lowerVariance) {
    lines.push(`Variance was lower for astrocyte (${stats.varAst.toFixed(5)}) than cursor ` +
      `(${stats.varCursor.toFixed(5)}), indicating more consistent learning across seeds.`);
  } else {
    lines.push(`Variance was higher for astrocyte (${stats.varAst.toFixed(5)}) than cursor ` +
      `(${stats.varCursor.toFixed(5)}), suggesting sensitivity to network initialisation.`);
  }

  if (sc.specialisation) {
    lines.push(`Pattern specialisation emerged: ${stats.specialisationCount} seed(s) had astrocytes ` +
      `with >2× activation bias for specific patterns. The biological routing hypothesis is supported.`);
  } else {
    lines.push(`No clear pattern specialisation detected. Astrocytes activated roughly uniformly ` +
      `across patterns, suggesting the activity-sensing signal may be too weak or ` +
      `territory coverage too coarse to differentiate input patterns.`);
  }

  if (sc.higherAcceptRate) {
    lines.push(`Accept rate was higher for astrocyte (${(stats.meanAstAccept * 100).toFixed(1)}%) ` +
      `than cursor (${(stats.meanCursorAccept * 100).toFixed(1)}%), confirming that ` +
      `activity-guided perturbation selects more productive synapses.`);
  } else {
    lines.push(`Accept rate for astrocyte (${(stats.meanAstAccept * 100).toFixed(1)}%) did not ` +
      `exceed cursor (${(stats.meanCursorAccept * 100).toFixed(1)}%). ` +
      `Perturbation placement is not yet more productive than random.`);
  }

  lines.push(`Overall: ${passed}/${total} success criteria passed.`);

  return lines.join(' ');
}
