// Experiment 024 report generator — Markdown + HTML
// Three conditions: epsilon | baseline | control
// Key diagnostics: cluster activation breakdown, epsilon vs regular activations per astrocyte,
// threshold evolution (did epsilon-explored astrocytes earn regular activation?)

import { writeFileSync, mkdirSync } from 'fs';
import { trainingPatterns } from './task.mjs';

export function generateReport(allResults, config, startTime) {
  mkdirSync('./reports', { recursive: true });

  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  const stats    = computeStats(allResults, config);

  writeFileSync('./reports/experiment-024.md',   buildMarkdown(allResults, config, stats, duration));
  writeFileSync('./reports/experiment-024.html',  buildHtml(allResults, config, stats, duration));

  console.log('\nReports written to ./reports/experiment-024.md and ./reports/experiment-024.html');
}

// ─── Statistics ───────────────────────────────────────────────────────────────

function computeStats(results, config) {
  const n = results.length;

  const epsAccs  = results.map(r => r.epsilon.inference.meanAccuracy);
  const baseAccs = results.map(r => r.baseline.inference.meanAccuracy);
  const ctrlAccs = results.map(r => r.control.inference.meanAccuracy);

  const meanEps  = mean(epsAccs);
  const meanBase = mean(baseAccs);
  const meanCtrl = mean(ctrlAccs);

  const varEps  = variance(epsAccs);
  const varBase = variance(baseAccs);

  const epsWinsBase = results.filter(
    r => r.epsilon.inference.meanAccuracy > r.baseline.inference.meanAccuracy
  ).length;

  // Accept rates
  const epsAcceptRates = results.map(r => {
    const t = r.epsilon.totalAccepted + r.epsilon.totalRejected;
    return t > 0 ? r.epsilon.totalAccepted / t : 0;
  });
  const baseAcceptRates = results.map(r => {
    const t = r.baseline.totalAccepted + r.baseline.totalRejected;
    return t > 0 ? r.baseline.totalAccepted / t : 0;
  });
  const meanEpsAccept  = mean(epsAcceptRates);
  const meanBaseAccept = mean(baseAcceptRates);

  // Cluster activation breakdown (epsilon condition)
  const clusterStats = results.map(r => {
    function clusterMeans(asts) {
      if (!asts) return { c0MeanAct: 0, c1MeanAct: 0, c0SuccRate: 0, c1SuccRate: 0,
                          c0EpsMean: 0, c1EpsMean: 0, c0Total: 0, c1Total: 0 };
      const c0 = asts.filter(a => a.cluster === 0);
      const c1 = asts.filter(a => a.cluster === 1);
      const meanAct = arr => arr.length > 0
        ? arr.reduce((s, a) => s + a.activationCount, 0) / arr.length : 0;
      const meanEpsAct = arr => arr.length > 0
        ? arr.reduce((s, a) => s + (a.epsilonCount || 0), 0) / arr.length : 0;
      const meanSuccRate = arr => {
        const totalAct  = arr.reduce((s, a) => s + a.activationCount, 0);
        const totalSucc = arr.reduce((s, a) => s + a.successCount, 0);
        return totalAct > 0 ? totalSucc / totalAct : 0;
      };
      return {
        c0MeanAct:  meanAct(c0),      c1MeanAct:  meanAct(c1),
        c0SuccRate: meanSuccRate(c0),  c1SuccRate: meanSuccRate(c1),
        c0EpsMean:  meanEpsAct(c0),   c1EpsMean:  meanEpsAct(c1),
        c0Total:    c0.reduce((s, a) => s + a.activationCount, 0),
        c1Total:    c1.reduce((s, a) => s + a.activationCount, 0),
      };
    }
    return {
      seed: r.seed,
      epsilon:  clusterMeans(r.epsilon.astrocyteStats),
      baseline: clusterMeans(r.baseline.astrocyteStats),
    };
  });

  // Success criterion 2: mean cluster-1 epsilon activations per astrocyte across seeds > 500
  const meanC1EpsAct = mean(clusterStats.map(s => s.epsilon.c1MeanAct));

  // Success criterion 3: ≥1 seed where ≥1 cluster-1 epsilon astrocyte has success rate > 10%
  let c1SuccessSeeds = 0;
  for (const r of results) {
    const asts = r.epsilon.astrocyteStats || [];
    const c1   = asts.filter(a => a.cluster === 1);
    if (c1.some(a => a.activationCount > 0 && a.successCount / a.activationCount > 0.10))
      c1SuccessSeeds++;
  }

  // Count seeds where any cluster-1 epsilon astrocyte dropped its threshold (< 0.5)
  let c1ThresholdDropped = 0;
  for (const r of results) {
    const asts = r.epsilon.astrocyteStats || [];
    if (asts.filter(a => a.cluster === 1).some(a => a.finalThreshold < 0.5))
      c1ThresholdDropped++;
  }

  // Average trajectories
  const epsTraj  = averageTrajectory(results.map(r => r.epsilon.trajectory));
  const baseTraj = averageTrajectory(results.map(r => r.baseline.trajectory));

  const success = {
    epsBeatBaseline:       meanEps > meanBase,
    c1ActivationsAbove500: meanC1EpsAct > 500,
    c1SuccessAbove10pct:   c1SuccessSeeds >= 1,
    acceptRateInRange:     meanEpsAccept >= 0.20 && meanEpsAccept <= 0.50,
  };

  return {
    n, meanEps, meanBase, meanCtrl,
    varEps, varBase,
    epsWinsBase,
    meanEpsAccept, meanBaseAccept,
    meanC1EpsAct, c1SuccessSeeds, c1ThresholdDropped,
    clusterStats,
    epsTraj, baseTraj,
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
  const valid = trajectories.filter(t => t.length > 0);
  if (valid.length === 0) return [];
  const len = valid[0].length;
  return Array.from({ length: len }, (_, i) => ({
    episode:       valid[0][i].episode,
    avgReward:     mean(valid.map(t => t[i].avgReward)),
    avgAcceptRate: mean(valid.map(t => t[i].acceptRate)),
  }));
}

// ─── Markdown ─────────────────────────────────────────────────────────────────

function buildMarkdown(allResults, config, stats, duration) {
  const n  = stats.n;
  const sc = stats.success;

  const perSeedRows = allResults.map(r => {
    const eRate = (r.epsilon.totalAccepted  / (r.epsilon.totalAccepted  + r.epsilon.totalRejected  || 1) * 100).toFixed(1);
    const bRate = (r.baseline.totalAccepted / (r.baseline.totalAccepted + r.baseline.totalRejected || 1) * 100).toFixed(1);
    const beat  = r.epsilon.inference.meanAccuracy > r.baseline.inference.meanAccuracy ? '✅' : '—';
    return `| ${r.seed} ` +
      `| ${r.epsilon.inference.meanAccuracy.toFixed(3)} (${r.epsilon.inference.distinctOutputs}d) ` +
      `| ${r.baseline.inference.meanAccuracy.toFixed(3)} (${r.baseline.inference.distinctOutputs}d) ` +
      `| ${r.control.inference.meanAccuracy.toFixed(3)} ` +
      `| ${beat} ` +
      `| ${r.epsilon.totalAccepted}/${r.epsilon.totalRejected} (${eRate}%) ` +
      `| ${r.baseline.totalAccepted}/${r.baseline.totalRejected} (${bRate}%) |`;
  }).join('\n');

  const trajRows = stats.epsTraj.map((t, i) => {
    const b = stats.baseTraj[i];
    return `| ${t.episode} | ${t.avgReward.toFixed(3)} | ${(t.avgAcceptRate * 100).toFixed(1)}% ` +
      `| ${b ? b.avgReward.toFixed(3) : '—'} | ${b ? (b.avgAcceptRate * 100).toFixed(1) + '%' : '—'} |`;
  }).join('\n');

  const clusterRows = stats.clusterStats.map(s => {
    const e = s.epsilon, b = s.baseline;
    return `| ${s.seed} ` +
      `| ${e.c0Total.toLocaleString()} (${(e.c0SuccRate * 100).toFixed(0)}%) ` +
      `| **${e.c1Total.toLocaleString()}** (${(e.c1SuccRate * 100).toFixed(0)}%) ` +
      `| ${b.c0Total.toLocaleString()} (${(b.c0SuccRate * 100).toFixed(0)}%) ` +
      `| ${b.c1Total.toLocaleString()} (${(b.c1SuccRate * 100).toFixed(0)}%) |`;
  }).join('\n');

  const patLabels = trainingPatterns.map((_, i) => `P${i + 1}`).join(' | ');

  function astroSection(r, condKey, condLabel) {
    const asts = r[condKey].astrocyteStats;
    if (!asts) return '';
    const rows = asts.map(ast => {
      const sr     = ast.activationCount > 0
        ? (ast.successCount / ast.activationCount * 100).toFixed(1) + '%' : '—';
      const regAct = ast.activationCount - (ast.epsilonCount || 0);
      const total  = ast.activationsByPattern.reduce((s, v) => s + v, 0);
      const m      = total > 0 ? total / ast.activationsByPattern.length : 0;
      const maxAct = Math.max(...ast.activationsByPattern);
      const spec   = maxAct > 2 * m ? ' ★' : '';
      const ss     = ast.scoreSamples;
      const ssFmt  = ss
        ? `${ss.ep100.toFixed(4)} / ${ss.ep1000.toFixed(4)} / ${ss.epFinal.toFixed(4)}`
        : '—';
      const pCounts = ast.activationsByPattern.join(' | ');
      return `| ${ast.id} | C${ast.cluster} | (${ast.position.x.toFixed(1)}, ${ast.position.y.toFixed(1)}) ` +
        `| ${ast.neuronCount}n / ${ast.synapseCount}s ` +
        `| ${regAct} + ${ast.epsilonCount || 0}ε | ${sr} | ${ast.finalThreshold.toFixed(3)}${spec} ` +
        `| ${ssFmt} | ${pCounts} |`;
    }).join('\n');
    return `\n**Seed ${r.seed} — ${condLabel}**\n\n` +
      `| ID | Cl | Pos | Territory | Act (reg + ε) | Success% | Thresh | Score ep100/1000/5000 | ${patLabels} |\n` +
      `|----|----|-----|-----------|---------------|----------|--------|----------------------|${trainingPatterns.map(() => '---').join('|')}|\n` +
      rows;
  }

  const astroSections = allResults.map(r =>
    astroSection(r, 'epsilon', 'epsilon') + astroSection(r, 'baseline', 'baseline')
  ).join('\n');

  const inferenceDetail = allResults.map(r => {
    const rows = r.epsilon.inference.results.map((res, i) => {
      const base = r.baseline.inference.results[i];
      const ctrl = r.control.inference.results[i];
      return `| ${res.label} | [${res.input.join('')}] | [${res.target.join('')}] ` +
        `| [${res.output.join('')}] ${res.accuracy.toFixed(2)} ` +
        `| [${base.output.join('')}] ${base.accuracy.toFixed(2)} ` +
        `| [${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)} |`;
    }).join('\n');
    return `\n**Seed ${r.seed}**\n\n` +
      `| Pat | Input | Target | Epsilon out/acc | Baseline out/acc | Control out/acc |\n` +
      `|-----|-------|--------|-----------------|------------------|----------------|\n` + rows;
  }).join('\n');

  return `# Experiment 024 — Epsilon-Exploration Astrocytes

**Date:** ${new Date().toISOString().slice(0, 10)}
**Duration:** ${duration}s

## Hypothesis

Experiments 021–022 showed that cluster 1 (output-side) astrocytes remain dormant because their
activation scores never exceed threshold — not because the scoring signal is wrong, but because
they have no way to bootstrap. The fix: epsilon exploration. Every astrocyte has a ${(config.epsilon * 100).toFixed(0)}%
chance of activating each step regardless of score. Over ${(config.episodes * config.stepsPerEpisode).toLocaleString()} steps, that's
~${Math.round(config.episodes * config.stepsPerEpisode * config.epsilon)} forced activations per astrocyte. If perturbations in that territory are useful, the
success rate rises, the threshold drops, and regular activation takes over. If not, the astrocyte
stays at background exploration rate — quiet, not dead.

## Configuration

| Parameter | Value |
|---|---|
| Clusters | ${config.clusters} × ${config.neuronsPerCluster} neurons |
| Astrocytes | ${config.astrocytesPerCluster} per cluster (${config.astrocytesPerCluster * 2} total) |
| Territory radius | ${config.territoryRadius} (2D x-y) |
| Perturbation std | ${config.perturbStd} |
| Epsilon | ${(config.epsilon * 100).toFixed(0)}% per astrocyte per step |
| Acceptance | soft reward, strict > |
| Homeostasis | OFF |
| Episodes | ${config.episodes} × ${config.stepsPerEpisode} steps |
| Seeds | ${config.seeds.join(', ')} |

## Summary

| Metric | Epsilon | Baseline | Control |
|---|---|---|---|
| Mean accuracy | ${stats.meanEps.toFixed(3)} | ${stats.meanBase.toFixed(3)} | ${stats.meanCtrl.toFixed(3)} |
| Variance | ${stats.varEps.toFixed(5)} | ${stats.varBase.toFixed(5)} | — |
| Mean accept rate | ${(stats.meanEpsAccept * 100).toFixed(1)}% | ${(stats.meanBaseAccept * 100).toFixed(1)}% | — |
| Seeds beating baseline | ${stats.epsWinsBase} / ${n} | — | — |
| Mean cluster-1 activations (epsilon) | ${stats.meanC1EpsAct.toFixed(0)} | — | — |
| Seeds where C1 threshold dropped | ${stats.c1ThresholdDropped} / ${n} | — | — |

## Success Criteria

| Criterion | Result | Pass |
|---|---|---|
| Epsilon > Baseline mean accuracy | ${stats.meanEps.toFixed(3)} vs ${stats.meanBase.toFixed(3)} | ${sc.epsBeatBaseline ? '✅' : '❌'} |
| Cluster 1 epsilon activations mean > 500 | ${stats.meanC1EpsAct.toFixed(0)} | ${sc.c1ActivationsAbove500 ? '✅' : '❌'} |
| ≥1 cluster-1 epsilon astrocyte success rate > 10% | ${stats.c1SuccessSeeds} seed(s) | ${sc.c1SuccessAbove10pct ? '✅' : '❌'} |
| Epsilon accept rate 20–50% | ${(stats.meanEpsAccept * 100).toFixed(1)}% | ${sc.acceptRateInRange ? '✅' : '❌'} |

## Per-Seed Results

| Seed | Epsilon acc (d) | Baseline acc (d) | Control acc | E>B | Epsilon keep/revert (%) | Baseline keep/revert (%) |
|------|-----------------|------------------|-------------|-----|--------------------------|--------------------------|
${perSeedRows}

## Trajectory (avg across ${n} seeds)

| Episode | Epsilon reward | Epsilon accept% | Baseline reward | Baseline accept% |
|---------|---------------|-----------------|----------------|------------------|
${trajRows}

## Cluster Activation Breakdown

Format: total activations (success rate %). **Bold** = cluster 1 epsilon (key column).
A value in the hundreds-to-thousands here means epsilon woke up cluster 1.

| Seed | Epsilon C0 | **Epsilon C1** | Baseline C0 | Baseline C1 |
|------|-----------|---------------|------------|------------|
${clusterRows}

## Per-Astrocyte Diagnostics (Act = regular + epsilon; ★ = pattern specialisation >2× mean; Score = mean per episode at ep100/1000/5000)

${astroSections}

## Inference Detail

${inferenceDetail}

## Conclusion

${buildConclusion(stats, config)}
`;
}

// ─── HTML ─────────────────────────────────────────────────────────────────────

function buildHtml(allResults, config, stats, duration) {
  const n  = stats.n;
  const sc = stats.success;

  const chartW = 720, chartH = 280;
  const padL = 55, padR = 20, padT = 20, padB = 35;
  const w = chartW - padL - padR, h = chartH - padT - padB;

  function toPoly(traj, key) {
    if (!traj || traj.length === 0) return '';
    return traj.map((t, i) => {
      const px = padL + (i / (traj.length - 1 || 1)) * w;
      const py = padT + (1 - t[key]) * h;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');
  }

  const epsReward  = toPoly(stats.epsTraj,  'avgReward');
  const baseReward = toPoly(stats.baseTraj, 'avgReward');
  const epsAccept  = toPoly(stats.epsTraj,  'avgAcceptRate');
  const baseAccept = toPoly(stats.baseTraj, 'avgAcceptRate');

  const perSeedRows = allResults.map(r => {
    const eRate = (r.epsilon.totalAccepted  / (r.epsilon.totalAccepted  + r.epsilon.totalRejected  || 1) * 100).toFixed(1);
    const bRate = (r.baseline.totalAccepted / (r.baseline.totalAccepted + r.baseline.totalRejected || 1) * 100).toFixed(1);
    const beat  = r.epsilon.inference.meanAccuracy > r.baseline.inference.meanAccuracy;
    return `<tr>
      <td>${r.seed}</td>
      <td>${r.epsilon.inference.meanAccuracy.toFixed(3)} (${r.epsilon.inference.distinctOutputs}d)</td>
      <td>${r.baseline.inference.meanAccuracy.toFixed(3)} (${r.baseline.inference.distinctOutputs}d)</td>
      <td>${r.control.inference.meanAccuracy.toFixed(3)}</td>
      <td>${beat ? '✅' : '—'}</td>
      <td>${r.epsilon.totalAccepted}/${r.epsilon.totalRejected} (${eRate}%)</td>
      <td>${r.baseline.totalAccepted}/${r.baseline.totalRejected} (${bRate}%)</td>
    </tr>`;
  }).join('\n');

  const clusterRows = stats.clusterStats.map(s => {
    const e = s.epsilon, b = s.baseline;
    const c1Active = e.c1Total > 500;
    return `<tr>
      <td>${s.seed}</td>
      <td>${e.c0Total.toLocaleString()} (${(e.c0SuccRate * 100).toFixed(0)}%)</td>
      <td${c1Active ? ' style="background:#e8f5e9;font-weight:bold"' : ' style="background:#ffebee"'}>${e.c1Total.toLocaleString()} (${(e.c1SuccRate * 100).toFixed(0)}%)</td>
      <td>${b.c0Total.toLocaleString()} (${(b.c0SuccRate * 100).toFixed(0)}%)</td>
      <td style="color:#999">${b.c1Total.toLocaleString()} (${(b.c1SuccRate * 100).toFixed(0)}%)</td>
    </tr>`;
  }).join('\n');

  const patLabels = trainingPatterns.map((_, i) => `<th>P${i + 1}</th>`).join('');

  function astroDetailHtml(r, condKey, condLabel) {
    const asts = r[condKey].astrocyteStats;
    if (!asts) return '';
    const rows = asts.map(ast => {
      const sr     = ast.activationCount > 0
        ? (ast.successCount / ast.activationCount * 100).toFixed(1) + '%' : '—';
      const regAct = ast.activationCount - (ast.epsilonCount || 0);
      const total  = ast.activationsByPattern.reduce((s, v) => s + v, 0);
      const m      = total > 0 ? total / ast.activationsByPattern.length : 0;
      const maxAct = Math.max(...ast.activationsByPattern);
      const spec   = maxAct > 2 * m;
      const ss     = ast.scoreSamples;
      const threshDropped = ast.finalThreshold < 0.5;
      const isC1   = ast.cluster === 1;
      const pCells = ast.activationsByPattern.map(v => {
        const hi = total > 0 && v > 2 * m ? ' style="background:#ffe0b2;font-weight:bold"' : '';
        return `<td${hi}>${v}</td>`;
      }).join('');
      const rowBg = isC1
        ? (threshDropped ? ' style="background:#c8e6c9"' : ' style="background:#e3f2fd"')
        : (spec ? ' style="background:#fff3e0"' : '');
      return `<tr${rowBg}>
        <td>${ast.id}${spec ? ' ★' : ''}</td>
        <td>C${ast.cluster}</td>
        <td>(${ast.position.x.toFixed(1)}, ${ast.position.y.toFixed(1)})</td>
        <td>${ast.neuronCount}n / ${ast.synapseCount}s</td>
        <td>${regAct} + ${ast.epsilonCount || 0}ε</td>
        <td>${sr}</td>
        <td${threshDropped ? ' style="color:#080;font-weight:bold"' : ''}>${ast.finalThreshold.toFixed(3)}</td>
        <td>${ss ? ss.ep100.toFixed(4) : '—'}</td>
        <td>${ss ? ss.ep1000.toFixed(4) : '—'}</td>
        <td>${ss ? ss.epFinal.toFixed(4) : '—'}</td>
        ${pCells}
      </tr>`;
    }).join('\n');
    return `<details>
      <summary><strong>Seed ${r.seed} — ${condLabel}</strong> acc=${r[condKey].inference.meanAccuracy.toFixed(3)}, distinct=${r[condKey].inference.distinctOutputs}</summary>
      <table>
        <tr><th>ID</th><th>Cl</th><th>Pos</th><th>Territory</th><th>Act (reg+ε)</th><th>Success%</th><th>Thresh</th><th>Score ep100</th><th>Score ep1000</th><th>Score epFinal</th>${patLabels}</tr>
        ${rows}
      </table>
      <p style="font-size:0.82em;color:#666">
        Blue rows = cluster 1 (output-side).
        Green rows = cluster 1 astrocyte whose threshold dropped below 0.5 (earned regular activation via epsilon).
        ★ = pattern specialisation &gt;2× mean activation.
        Thresh in green = dropped from initial 0.5.
      </p>
    </details>`;
  }

  const astroDiagnostics = allResults.map(r =>
    astroDetailHtml(r, 'epsilon', 'epsilon') + '\n' + astroDetailHtml(r, 'baseline', 'baseline')
  ).join('\n');

  const inferenceDetail = allResults.map(r => {
    const rows = r.epsilon.inference.results.map((res, i) => {
      const base = r.baseline.inference.results[i];
      const ctrl = r.control.inference.results[i];
      return `<tr>
        <td>${res.label}</td>
        <td>[${res.input.join('')}]</td>
        <td>[${res.target.join('')}]</td>
        <td>[${res.output.join('')}] ${res.accuracy.toFixed(2)}</td>
        <td>[${base.output.join('')}] ${base.accuracy.toFixed(2)}</td>
        <td>[${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)}</td>
      </tr>`;
    }).join('\n');
    return `<details>
      <summary><strong>Seed ${r.seed}</strong> — epsilon=${r.epsilon.inference.meanAccuracy.toFixed(3)}, baseline=${r.baseline.inference.meanAccuracy.toFixed(3)}, control=${r.control.inference.meanAccuracy.toFixed(3)}</summary>
      <table>
        <tr><th>Pat</th><th>Input</th><th>Target</th><th>Epsilon out/acc</th><th>Baseline out/acc</th><th>Control out/acc</th></tr>
        ${rows}
      </table>
    </details>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 024 — Epsilon-Exploration Astrocytes</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 24px; color: #222; }
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
  .leg-eps::before  { background: #2255cc; }
  .leg-base::before { background: #cc5522; }
</style>
</head>
<body>
<h1>Experiment 024 — Epsilon-Exploration Astrocytes</h1>
<p class="meta">Date: ${new Date().toISOString().slice(0, 10)} &nbsp;|&nbsp; Duration: ${duration}s &nbsp;|&nbsp; ε = ${(config.epsilon * 100).toFixed(0)}%</p>

<h2>Hypothesis</h2>
<p>Experiments 021–022 showed cluster 1 (output-side) astrocytes go dormant regardless of scoring
mechanism: they lack any way to bootstrap because their activation scores are always below threshold,
and they can only learn that their territory is productive by participating in learning first.
Epsilon exploration breaks this deadlock: each astrocyte fires with probability ε=${(config.epsilon * 100).toFixed(0)}% every step,
independent of score and threshold. Over ${(config.episodes * config.stepsPerEpisode).toLocaleString()} steps, each dormant astrocyte receives ~${Math.round(config.episodes * config.stepsPerEpisode * config.epsilon)} activations.
If its territory is useful, the success rate rises, the threshold drops, and regular activation
takes over. If not, it stays at background exploration rate — quiet, not dead.</p>

<h2>Configuration</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Architecture</td><td>${config.clusters} clusters × ${config.neuronsPerCluster} neurons</td></tr>
  <tr><td>Astrocytes</td><td>${config.astrocytesPerCluster} per cluster (${config.astrocytesPerCluster * 2} total), territory radius ${config.territoryRadius}</td></tr>
  <tr><td>Epsilon</td><td>${(config.epsilon * 100).toFixed(0)}% per astrocyte per step</td></tr>
  <tr><td>Perturbation std</td><td>${config.perturbStd}</td></tr>
  <tr><td>Acceptance</td><td>soft reward, strict &gt;</td></tr>
  <tr><td>Homeostasis</td><td>OFF</td></tr>
  <tr><td>Episodes</td><td>${config.episodes} × ${config.stepsPerEpisode} steps</td></tr>
  <tr><td>Seeds</td><td>${config.seeds.join(', ')}</td></tr>
</table>

<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Epsilon</th><th>Baseline</th><th>Control</th></tr>
  <tr><td>Mean accuracy</td><td>${stats.meanEps.toFixed(3)}</td><td>${stats.meanBase.toFixed(3)}</td><td>${stats.meanCtrl.toFixed(3)}</td></tr>
  <tr><td>Variance</td><td>${stats.varEps.toFixed(5)}</td><td>${stats.varBase.toFixed(5)}</td><td>—</td></tr>
  <tr><td>Mean accept rate</td><td>${(stats.meanEpsAccept * 100).toFixed(1)}%</td><td>${(stats.meanBaseAccept * 100).toFixed(1)}%</td><td>—</td></tr>
  <tr><td>Seeds beating baseline</td><td>${stats.epsWinsBase} / ${n}</td><td>—</td><td>—</td></tr>
  <tr><td>Mean cluster-1 activations (epsilon)</td><td>${stats.meanC1EpsAct.toFixed(0)}</td><td>—</td><td>—</td></tr>
  <tr><td>Seeds where C1 threshold dropped</td><td>${stats.c1ThresholdDropped} / ${n}</td><td>—</td><td>—</td></tr>
</table>

<h2>Success Criteria</h2>
<table>
  <tr><th>Criterion</th><th>Result</th><th>Pass</th></tr>
  <tr>
    <td>Epsilon &gt; Baseline mean accuracy</td>
    <td>${stats.meanEps.toFixed(3)} vs ${stats.meanBase.toFixed(3)}</td>
    <td class="${sc.epsBeatBaseline ? 'pass' : 'fail'}">${sc.epsBeatBaseline ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>Cluster 1 epsilon activations mean &gt; 500</td>
    <td>${stats.meanC1EpsAct.toFixed(0)}</td>
    <td class="${sc.c1ActivationsAbove500 ? 'pass' : 'fail'}">${sc.c1ActivationsAbove500 ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>≥1 cluster-1 epsilon astrocyte success rate &gt; 10%</td>
    <td>${stats.c1SuccessSeeds} seed(s)</td>
    <td class="${sc.c1SuccessAbove10pct ? 'pass' : 'fail'}">${sc.c1SuccessAbove10pct ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>Epsilon accept rate 20–50%</td>
    <td>${(stats.meanEpsAccept * 100).toFixed(1)}%</td>
    <td class="${sc.acceptRateInRange ? 'pass' : 'fail'}">${sc.acceptRateInRange ? '✅ Yes' : '❌ No'}</td>
  </tr>
</table>

<h2>Reward &amp; Accept Rate Trajectories (avg across ${n} seeds)</h2>
<svg width="${chartW}" height="${chartH}" style="border:1px solid #ddd;background:#fafafa;display:block;">
  <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT + h}" x2="${padL + w}" y2="${padT + h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT + h / 2}" x2="${padL + w}" y2="${padT + h / 2}" stroke="#eee" stroke-dasharray="4"/>
  <text x="${padL - 5}" y="${padT + 4}"         text-anchor="end">1.0</text>
  <text x="${padL - 5}" y="${padT + h / 2 + 4}" text-anchor="end">0.5</text>
  <text x="${padL - 5}" y="${padT + h + 4}"      text-anchor="end">0.0</text>
  <text x="${padL}"       y="${padT + h + 20}" text-anchor="middle">0</text>
  <text x="${padL + w / 2}" y="${padT + h + 20}" text-anchor="middle">${Math.floor(config.episodes / 2)}</text>
  <text x="${padL + w}"   y="${padT + h + 20}" text-anchor="middle">${config.episodes}</text>
  <text x="${padL + w / 2}" y="${chartH}"       text-anchor="middle" fill="#666">Episode</text>
  ${epsReward  ? `<polyline points="${epsReward}"  fill="none" stroke="#2255cc" stroke-width="2"/>` : ''}
  ${baseReward ? `<polyline points="${baseReward}" fill="none" stroke="#cc5522" stroke-width="1.5" stroke-dasharray="6,3"/>` : ''}
  ${epsAccept  ? `<polyline points="${epsAccept}"  fill="none" stroke="#2255cc" stroke-width="1.5" opacity="0.4"/>` : ''}
  ${baseAccept ? `<polyline points="${baseAccept}" fill="none" stroke="#cc5522" stroke-width="1"   opacity="0.4" stroke-dasharray="6,3"/>` : ''}
</svg>
<div class="legend">
  <span class="leg-eps">Epsilon reward (solid) / accept rate (faded)</span>
  <span class="leg-base">Baseline reward (dashed) / accept rate (faded dashed)</span>
</div>

<h2>Per-Seed Results</h2>
<table>
  <tr>
    <th>Seed</th>
    <th>Epsilon acc (d)</th><th>Baseline acc (d)</th><th>Control acc</th>
    <th>E&gt;B</th>
    <th>Epsilon keep/revert (%)</th><th>Baseline keep/revert (%)</th>
  </tr>
  ${perSeedRows}
</table>

<h2>Cluster Activation Breakdown</h2>
<p>Key diagnostic: is cluster 1 participating in the epsilon condition?
Format: total activations (success rate %). Green = C1 epsilon &gt; 500, Red = still dormant.</p>
<table>
  <tr><th>Seed</th><th>Epsilon C0</th><th>Epsilon C1</th><th>Baseline C0</th><th>Baseline C1</th></tr>
  ${clusterRows}
</table>

<h2>Per-Astrocyte Diagnostics <small>(Act = regular + epsilon; blue = C1; green = C1 threshold dropped; ★ = specialisation)</small></h2>
${astroDiagnostics}

<h2>Inference Detail</h2>
${inferenceDetail}

<h2>Conclusion</h2>
<div class="conclusion">${buildConclusion(stats, config)}</div>

</body>
</html>`;
}

// ─── Conclusion ───────────────────────────────────────────────────────────────

function buildConclusion(stats, config) {
  const sc     = stats.success;
  const passed = Object.values(sc).filter(Boolean).length;
  const total  = Object.values(sc).length;
  const lines  = [];

  if (sc.epsBeatBaseline) {
    lines.push(`Epsilon exploration (${stats.meanEps.toFixed(3)}) outperformed the baseline ` +
      `(${stats.meanBase.toFixed(3)}) in mean inference accuracy (${stats.epsWinsBase}/${stats.n} seeds). ` +
      `Forcing dormant territories to participate improved overall learning.`);
  } else {
    lines.push(`Epsilon exploration (${stats.meanEps.toFixed(3)}) did not surpass the baseline ` +
      `(${stats.meanBase.toFixed(3)}). The accuracy change is within noise; ` +
      `cluster 1 participation may not yet be producing net positive perturbations.`);
  }

  if (sc.c1ActivationsAbove500) {
    lines.push(`Cluster 1 astrocytes woke up: mean ${stats.meanC1EpsAct.toFixed(0)} activations ` +
      `in the epsilon condition, driven by the ${(config.epsilon * 100).toFixed(0)}% exploration rate. ` +
      `${stats.c1ThresholdDropped}/${stats.n} seeds saw cluster 1 thresholds drop below 0.5, ` +
      `indicating earned regular participation.`);
  } else {
    lines.push(`Cluster 1 activations remain low (mean ${stats.meanC1EpsAct.toFixed(0)}). ` +
      `Even with epsilon exploration, cluster 1 perturbations are being filtered out by MAX_ACTIVE ` +
      `when cluster 0 astrocytes compete for the same slots. Consider increasing epsilon or ` +
      `guaranteeing a per-cluster activation slot.`);
  }

  if (sc.c1SuccessAbove10pct) {
    lines.push(`Output-side perturbation is productive: ${stats.c1SuccessSeeds} seed(s) had ` +
      `cluster 1 epsilon astrocytes exceeding 10% success rate. ` +
      `The output pathway can be trained once it gets participation.`);
  } else {
    lines.push(`No cluster 1 epsilon astrocyte exceeded 10% success rate. ` +
      `Output-side perturbations are not yet improving soft reward. ` +
      `The output synaptic landscape may be too flat for std=0.1 perturbations to find improvements.`);
  }

  if (sc.acceptRateInRange) {
    lines.push(`Accept rate ${(stats.meanEpsAccept * 100).toFixed(1)}% is within the 20–50% target. ` +
      `Epsilon exploration did not flood the system with unproductive perturbations.`);
  } else {
    const rate = (stats.meanEpsAccept * 100).toFixed(1);
    lines.push(`Accept rate ${rate}% is outside target range. ` +
      (stats.meanEpsAccept < 0.20
        ? `Below 20%: epsilon activations may be introducing too many unproductive perturbations.`
        : `Above 50%: selection pressure may be insufficient.`));
  }

  lines.push(`Overall: ${passed}/${total} success criteria passed.`);
  return lines.join(' ');
}
