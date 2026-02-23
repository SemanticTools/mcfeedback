// Experiment 022 report generator — Markdown + HTML
// Three conditions: astrocyte-traffic | astrocyte-firing | control
// Key new diagnostic: cluster activation breakdown (are cluster 1 astrocytes awake?)
// Also: per-astrocyte score samples at ep100, ep1000, ep5000.

import { writeFileSync, mkdirSync } from 'fs';
import { trainingPatterns } from './task.mjs';

export function generateReport(allResults, config, startTime) {
  mkdirSync('./reports', { recursive: true });

  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  const stats    = computeStats(allResults, config);

  writeFileSync('./reports/experiment-022.md',   buildMarkdown(allResults, config, stats, duration));
  writeFileSync('./reports/experiment-022.html',  buildHtml(allResults, config, stats, duration));

  console.log('\nReports written to ./reports/experiment-022.md and ./reports/experiment-022.html');
}

// ─── Statistics ───────────────────────────────────────────────────────────────

function computeStats(results, config) {
  const n = results.length;

  const trafficAccs = results.map(r => r.traffic.inference.meanAccuracy);
  const firingAccs  = results.map(r => r.firing.inference.meanAccuracy);
  const controlAccs = results.map(r => r.control.inference.meanAccuracy);

  const meanTraffic  = mean(trafficAccs);
  const meanFiring   = mean(firingAccs);
  const meanControl  = mean(controlAccs);

  const varTraffic = variance(trafficAccs);
  const varFiring  = variance(firingAccs);

  const trafficWinsFiring = results.filter(
    r => r.traffic.inference.meanAccuracy > r.firing.inference.meanAccuracy
  ).length;

  // Accept rates
  const trafficAcceptRates = results.map(r => {
    const t = r.traffic.totalAccepted + r.traffic.totalRejected;
    return t > 0 ? r.traffic.totalAccepted / t : 0;
  });
  const firingAcceptRates = results.map(r => {
    const t = r.firing.totalAccepted + r.firing.totalRejected;
    return t > 0 ? r.firing.totalAccepted / t : 0;
  });
  const meanTrafficAccept = mean(trafficAcceptRates);
  const meanFiringAccept  = mean(firingAcceptRates);

  // Cluster activation breakdown
  const clusterStats = results.map(r => {
    function clusterMeans(asts) {
      if (!asts) return { c0MeanAct: 0, c1MeanAct: 0, c0SuccRate: 0, c1SuccRate: 0 };
      const c0 = asts.filter(a => a.cluster === 0);
      const c1 = asts.filter(a => a.cluster === 1);
      const meanAct = arr => arr.length > 0
        ? arr.reduce((s, a) => s + a.activationCount, 0) / arr.length : 0;
      const meanSuccRate = arr => {
        const totalAct  = arr.reduce((s, a) => s + a.activationCount, 0);
        const totalSucc = arr.reduce((s, a) => s + a.successCount, 0);
        return totalAct > 0 ? totalSucc / totalAct : 0;
      };
      return {
        c0MeanAct:  meanAct(c0),      c1MeanAct:  meanAct(c1),
        c0SuccRate: meanSuccRate(c0),  c1SuccRate: meanSuccRate(c1),
        c0Total:    c0.reduce((s, a) => s + a.activationCount, 0),
        c1Total:    c1.reduce((s, a) => s + a.activationCount, 0),
      };
    }
    return {
      seed: r.seed,
      traffic: clusterMeans(r.traffic.astrocyteStats),
      firing:  clusterMeans(r.firing.astrocyteStats),
    };
  });

  // Success criterion 2: mean cluster 1 traffic activation per astrocyte across seeds > 1000
  const meanC1TrafficAct = mean(clusterStats.map(s => s.traffic.c1MeanAct));

  // Success criterion 3: ≥1 seed where ≥1 cluster-1 traffic astrocyte has success rate > 10%
  let c1SuccessSeeds = 0;
  for (const r of results) {
    const asts = r.traffic.astrocyteStats || [];
    const c1   = asts.filter(a => a.cluster === 1);
    if (c1.some(a => a.activationCount > 0 && a.successCount / a.activationCount > 0.10))
      c1SuccessSeeds++;
  }

  // Average trajectories
  const trafficTraj = averageTrajectory(results.map(r => r.traffic.trajectory));
  const firingTraj  = averageTrajectory(results.map(r => r.firing.trajectory));

  const success = {
    trafficBeatsFiring:    meanTraffic > meanFiring,
    c1ActivationThousands: meanC1TrafficAct > 1000,
    c1SuccessAbove10pct:   c1SuccessSeeds >= 1,
    acceptRateInRange:     meanTrafficAccept >= 0.20 && meanTrafficAccept <= 0.50,
  };

  return {
    n, meanTraffic, meanFiring, meanControl,
    varTraffic, varFiring,
    trafficWinsFiring,
    meanTrafficAccept, meanFiringAccept,
    meanC1TrafficAct, c1SuccessSeeds,
    clusterStats,
    trafficTraj, firingTraj,
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

  // Per-seed results
  const perSeedRows = allResults.map(r => {
    const tRate = (r.traffic.totalAccepted / (r.traffic.totalAccepted + r.traffic.totalRejected || 1) * 100).toFixed(1);
    const fRate = (r.firing.totalAccepted  / (r.firing.totalAccepted  + r.firing.totalRejected  || 1) * 100).toFixed(1);
    const beat  = r.traffic.inference.meanAccuracy > r.firing.inference.meanAccuracy ? '✅' : '—';
    return `| ${r.seed} ` +
      `| ${r.traffic.inference.meanAccuracy.toFixed(3)} (${r.traffic.inference.distinctOutputs}d) ` +
      `| ${r.firing.inference.meanAccuracy.toFixed(3)} (${r.firing.inference.distinctOutputs}d) ` +
      `| ${r.control.inference.meanAccuracy.toFixed(3)} ` +
      `| ${beat} ` +
      `| ${r.traffic.totalAccepted}/${r.traffic.totalRejected} (${tRate}%) ` +
      `| ${r.firing.totalAccepted}/${r.firing.totalRejected} (${fRate}%) |`;
  }).join('\n');

  // Trajectory
  const trajRows = stats.trafficTraj.map((t, i) => {
    const f = stats.firingTraj[i];
    return `| ${t.episode} | ${t.avgReward.toFixed(3)} | ${(t.avgAcceptRate * 100).toFixed(1)}% ` +
      `| ${f ? f.avgReward.toFixed(3) : '—'} | ${f ? (f.avgAcceptRate * 100).toFixed(1) + '%' : '—'} |`;
  }).join('\n');

  // Cluster activation breakdown
  const clusterRows = stats.clusterStats.map(s => {
    const t = s.traffic, f = s.firing;
    return `| ${s.seed} ` +
      `| ${t.c0Total.toLocaleString()} (${(t.c0SuccRate * 100).toFixed(0)}%) ` +
      `| **${t.c1Total.toLocaleString()}** (${(t.c1SuccRate * 100).toFixed(0)}%) ` +
      `| ${f.c0Total.toLocaleString()} (${(f.c0SuccRate * 100).toFixed(0)}%) ` +
      `| ${f.c1Total.toLocaleString()} (${(f.c1SuccRate * 100).toFixed(0)}%) |`;
  }).join('\n');

  // Per-astrocyte diagnostics
  const patLabels = trainingPatterns.map((_, i) => `P${i + 1}`).join(' | ');

  function astroSection(r, condKey, condLabel) {
    const asts = r[condKey].astrocyteStats;
    if (!asts) return '';
    const rows = asts.map(ast => {
      const sr = ast.activationCount > 0
        ? (ast.successCount / ast.activationCount * 100).toFixed(1) + '%' : '—';
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
        `| ${ast.activationCount} | ${sr} | ${ast.finalThreshold.toFixed(3)}${spec} ` +
        `| ${ssFmt} | ${pCounts} |`;
    }).join('\n');

    return `\n**Seed ${r.seed} — ${condLabel}**\n\n` +
      `| ID | Cl | Pos | Territory | Activations | Success% | Thresh | Score ep100/1000/5000 | ${patLabels} |\n` +
      `|----|----|-----|-----------|-------------|----------|--------|----------------------|${trainingPatterns.map(() => '---').join('|')}|\n` +
      rows;
  }

  const astroSections = allResults.map(r =>
    astroSection(r, 'traffic', 'traffic') + astroSection(r, 'firing', 'firing')
  ).join('\n');

  // Inference detail
  const inferenceDetail = allResults.map(r => {
    const rows = r.traffic.inference.results.map((res, i) => {
      const fir  = r.firing.inference.results[i];
      const ctrl = r.control.inference.results[i];
      return `| ${res.label} | [${res.input.join('')}] | [${res.target.join('')}] ` +
        `| [${res.output.join('')}] ${res.accuracy.toFixed(2)} ` +
        `| [${fir.output.join('')}] ${fir.accuracy.toFixed(2)} ` +
        `| [${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)} |`;
    }).join('\n');
    return `\n**Seed ${r.seed}**\n\n` +
      `| Pat | Input | Target | Traffic out/acc | Firing out/acc | Control out/acc |\n` +
      `|-----|-------|--------|-----------------|----------------|----------------|\n` + rows;
  }).join('\n');

  return `# Experiment 022 — Astrocyte Synaptic Traffic Sensing

**Date:** ${new Date().toISOString().slice(0, 10)}
**Duration:** ${duration}s

## Hypothesis

Experiment 021 revealed that cluster 1 (output-side) astrocytes never activated: they sensed
neuron firing, but output neurons rarely fire with random initial weights and fixed thresholds.
The fix: astrocytes sense synaptic traffic (∑ abs(pre_output × weight) per owned synapse)
rather than neuron firing. This detects incoming signals even when the post-synaptic neuron
hasn't crossed threshold — the biological analogue of glutamate spillover detection.

## Configuration

| Parameter | Value |
|---|---|
| Clusters | ${config.clusters} × ${config.neuronsPerCluster} neurons |
| Astrocytes | ${config.astrocytesPerCluster} per cluster (${config.astrocytesPerCluster * 2} total) |
| Territory radius | ${config.territoryRadius} (2D x-y) |
| Perturbation std | ${config.perturbStd} |
| Acceptance | soft reward, strict > |
| Homeostasis | OFF |
| Episodes | ${config.episodes} × ${config.stepsPerEpisode} steps |
| Seeds | ${config.seeds.join(', ')} |

## Summary

| Metric | Traffic | Firing | Control |
|---|---|---|---|
| Mean accuracy | ${stats.meanTraffic.toFixed(3)} | ${stats.meanFiring.toFixed(3)} | ${stats.meanControl.toFixed(3)} |
| Variance | ${stats.varTraffic.toFixed(5)} | ${stats.varFiring.toFixed(5)} | — |
| Mean accept rate | ${(stats.meanTrafficAccept * 100).toFixed(1)}% | ${(stats.meanFiringAccept * 100).toFixed(1)}% | — |
| Seeds beating firing | ${stats.trafficWinsFiring} / ${n} | — | — |
| Mean cluster-1 activations (traffic) | ${stats.meanC1TrafficAct.toFixed(0)} | — | — |

## Success Criteria

| Criterion | Result | Pass |
|---|---|---|
| Traffic > Firing mean accuracy | ${stats.meanTraffic.toFixed(3)} vs ${stats.meanFiring.toFixed(3)} | ${sc.trafficBeatsFiring ? '✅' : '❌'} |
| Cluster 1 traffic activations mean > 1000 | ${stats.meanC1TrafficAct.toFixed(0)} | ${sc.c1ActivationThousands ? '✅' : '❌'} |
| ≥1 cluster-1 traffic astrocyte success rate > 10% | ${stats.c1SuccessSeeds} seed(s) | ${sc.c1SuccessAbove10pct ? '✅' : '❌'} |
| Traffic accept rate 20–50% | ${(stats.meanTrafficAccept * 100).toFixed(1)}% | ${sc.acceptRateInRange ? '✅' : '❌'} |

## Per-Seed Results

| Seed | Traffic acc (d) | Firing acc (d) | Control acc | T>F | Traffic keep/revert (%) | Firing keep/revert (%) |
|------|-----------------|----------------|-------------|-----|--------------------------|------------------------|
${perSeedRows}

## Trajectory (avg across ${n} seeds)

| Episode | Traffic reward | Traffic accept% | Firing reward | Firing accept% |
|---------|---------------|-----------------|--------------|----------------|
${trajRows}

## Cluster Activation Breakdown

The key diagnostic: are cluster 1 astrocytes participating in the traffic condition?
Format: total activations (success rate %). **Bold** = cluster 1 traffic (the column to watch).

| Seed | Traffic C0 | **Traffic C1** | Firing C0 | Firing C1 |
|------|-----------|---------------|-----------|-----------|
${clusterRows}

## Per-Astrocyte Diagnostics (★ = pattern specialisation >2× mean; Score = mean activationScore at ep100 / ep1000 / ep5000)

${astroSections}

## Inference Detail

${inferenceDetail}

## Conclusion

${buildConclusion(stats)}
`;
}

// ─── HTML ─────────────────────────────────────────────────────────────────────

function buildHtml(allResults, config, stats, duration) {
  const n  = stats.n;
  const sc = stats.success;

  // SVG trajectory chart
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

  const trafficReward = toPoly(stats.trafficTraj, 'avgReward');
  const firingReward  = toPoly(stats.firingTraj,  'avgReward');
  const trafficAccept = toPoly(stats.trafficTraj, 'avgAcceptRate');
  const firingAccept  = toPoly(stats.firingTraj,  'avgAcceptRate');

  // Per-seed results table
  const perSeedRows = allResults.map(r => {
    const tRate = (r.traffic.totalAccepted / (r.traffic.totalAccepted + r.traffic.totalRejected || 1) * 100).toFixed(1);
    const fRate = (r.firing.totalAccepted  / (r.firing.totalAccepted  + r.firing.totalRejected  || 1) * 100).toFixed(1);
    const beat  = r.traffic.inference.meanAccuracy > r.firing.inference.meanAccuracy;
    return `<tr>
      <td>${r.seed}</td>
      <td>${r.traffic.inference.meanAccuracy.toFixed(3)} (${r.traffic.inference.distinctOutputs}d)</td>
      <td>${r.firing.inference.meanAccuracy.toFixed(3)} (${r.firing.inference.distinctOutputs}d)</td>
      <td>${r.control.inference.meanAccuracy.toFixed(3)}</td>
      <td>${beat ? '✅' : '—'}</td>
      <td>${r.traffic.totalAccepted}/${r.traffic.totalRejected} (${tRate}%)</td>
      <td>${r.firing.totalAccepted}/${r.firing.totalRejected} (${fRate}%)</td>
    </tr>`;
  }).join('\n');

  // Cluster breakdown table
  const clusterRows = stats.clusterStats.map(s => {
    const t = s.traffic, f = s.firing;
    const c1Awake = t.c1Total > 1000;
    return `<tr>
      <td>${s.seed}</td>
      <td>${t.c0Total.toLocaleString()} (${(t.c0SuccRate * 100).toFixed(0)}%)</td>
      <td${c1Awake ? ' style="background:#e8f5e9;font-weight:bold"' : ' style="background:#ffebee"'}>${t.c1Total.toLocaleString()} (${(t.c1SuccRate * 100).toFixed(0)}%)</td>
      <td>${f.c0Total.toLocaleString()} (${(f.c0SuccRate * 100).toFixed(0)}%)</td>
      <td style="color:#999">${f.c1Total.toLocaleString()} (${(f.c1SuccRate * 100).toFixed(0)}%)</td>
    </tr>`;
  }).join('\n');

  // Per-astrocyte diagnostics (collapsible)
  const patLabels = trainingPatterns.map((_, i) => `<th>P${i + 1}</th>`).join('');

  function astroDetailHtml(r, condKey, condLabel) {
    const asts = r[condKey].astrocyteStats;
    if (!asts) return '';
    const rows = asts.map(ast => {
      const sr = ast.activationCount > 0
        ? (ast.successCount / ast.activationCount * 100).toFixed(1) + '%' : '—';
      const total  = ast.activationsByPattern.reduce((s, v) => s + v, 0);
      const m      = total > 0 ? total / ast.activationsByPattern.length : 0;
      const maxAct = Math.max(...ast.activationsByPattern);
      const spec   = maxAct > 2 * m;
      const isC1   = ast.cluster === 1;
      const ss     = ast.scoreSamples;
      const pCells = ast.activationsByPattern.map(v => {
        const hi = total > 0 && v > 2 * m ? ' style="background:#ffe0b2;font-weight:bold"' : '';
        return `<td${hi}>${v}</td>`;
      }).join('');
      const rowStyle = isC1 ? ' style="background:#e3f2fd"' : (spec ? ' style="background:#fff3e0"' : '');
      return `<tr${rowStyle}>
        <td>${ast.id}${spec ? ' ★' : ''}</td>
        <td>C${ast.cluster}</td>
        <td>(${ast.position.x.toFixed(1)}, ${ast.position.y.toFixed(1)})</td>
        <td>${ast.neuronCount}n / ${ast.synapseCount}s</td>
        <td>${ast.activationCount}</td>
        <td>${sr}</td>
        <td>${ast.finalThreshold.toFixed(3)}</td>
        <td>${ss ? ss.ep100.toFixed(4) : '—'}</td>
        <td>${ss ? ss.ep1000.toFixed(4) : '—'}</td>
        <td>${ss ? ss.epFinal.toFixed(4) : '—'}</td>
        ${pCells}
      </tr>`;
    }).join('\n');

    return `<details>
      <summary><strong>Seed ${r.seed} — ${condLabel}</strong> acc=${r[condKey].inference.meanAccuracy.toFixed(3)}, distinct=${r[condKey].inference.distinctOutputs}</summary>
      <table>
        <tr><th>ID</th><th>Cl</th><th>Pos</th><th>Territory</th><th>Activations</th><th>Success%</th><th>Thresh</th><th>Score ep100</th><th>Score ep1000</th><th>Score epFinal</th>${patLabels}</tr>
        ${rows}
      </table>
      <p style="font-size:0.82em;color:#666">Blue rows = cluster 1 (output-side). ★ = pattern specialisation >2× mean activation. Score columns show mean activation score for that episode.</p>
    </details>`;
  }

  const astroDiagnostics = allResults.map(r =>
    astroDetailHtml(r, 'traffic', 'traffic') + '\n' + astroDetailHtml(r, 'firing', 'firing')
  ).join('\n');

  // Inference detail (collapsible)
  const inferenceDetail = allResults.map(r => {
    const rows = r.traffic.inference.results.map((res, i) => {
      const fir  = r.firing.inference.results[i];
      const ctrl = r.control.inference.results[i];
      return `<tr>
        <td>${res.label}</td>
        <td>[${res.input.join('')}]</td>
        <td>[${res.target.join('')}]</td>
        <td>[${res.output.join('')}] ${res.accuracy.toFixed(2)}</td>
        <td>[${fir.output.join('')}] ${fir.accuracy.toFixed(2)}</td>
        <td>[${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)}</td>
      </tr>`;
    }).join('\n');
    return `<details>
      <summary><strong>Seed ${r.seed}</strong> — traffic=${r.traffic.inference.meanAccuracy.toFixed(3)}, firing=${r.firing.inference.meanAccuracy.toFixed(3)}, control=${r.control.inference.meanAccuracy.toFixed(3)}</summary>
      <table>
        <tr><th>Pat</th><th>Input</th><th>Target</th><th>Traffic out/acc</th><th>Firing out/acc</th><th>Control out/acc</th></tr>
        ${rows}
      </table>
    </details>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 022 — Astrocyte Synaptic Traffic Sensing</title>
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
  .leg-traffic::before { background: #2255cc; }
  .leg-firing::before  { background: #cc5522; }
</style>
</head>
<body>
<h1>Experiment 022 — Astrocyte Synaptic Traffic Sensing</h1>
<p class="meta">Date: ${new Date().toISOString().slice(0, 10)} &nbsp;|&nbsp; Duration: ${duration}s</p>

<h2>Hypothesis</h2>
<p>Experiment 021 revealed cluster 1 (output-side) astrocytes were effectively dead — they sensed
neuron firing, but output neurons rarely fire with random initial weights. This experiment switches
astrocyte activation scoring to <strong>synaptic traffic</strong>:
<code>score = ∑ abs(weight) for fired pre-synaptic synapses / total synapses</code>.
Traffic is non-zero whenever input signals propagate through a territory, even if the receiving
neuron hasn't fired. Cluster 1 astrocytes should now detect incoming inter-cluster signals and
participate in learning.</p>

<h2>Configuration</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Architecture</td><td>${config.clusters} clusters × ${config.neuronsPerCluster} neurons</td></tr>
  <tr><td>Astrocytes</td><td>${config.astrocytesPerCluster} per cluster (${config.astrocytesPerCluster * 2} total), territory radius ${config.territoryRadius}</td></tr>
  <tr><td>Perturbation std</td><td>${config.perturbStd}</td></tr>
  <tr><td>Acceptance</td><td>soft reward, strict &gt;</td></tr>
  <tr><td>Homeostasis</td><td>OFF</td></tr>
  <tr><td>Episodes</td><td>${config.episodes} × ${config.stepsPerEpisode} steps</td></tr>
  <tr><td>Seeds</td><td>${config.seeds.join(', ')}</td></tr>
</table>

<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Traffic</th><th>Firing</th><th>Control</th></tr>
  <tr><td>Mean accuracy</td><td>${stats.meanTraffic.toFixed(3)}</td><td>${stats.meanFiring.toFixed(3)}</td><td>${stats.meanControl.toFixed(3)}</td></tr>
  <tr><td>Variance</td><td>${stats.varTraffic.toFixed(5)}</td><td>${stats.varFiring.toFixed(5)}</td><td>—</td></tr>
  <tr><td>Mean accept rate</td><td>${(stats.meanTrafficAccept * 100).toFixed(1)}%</td><td>${(stats.meanFiringAccept * 100).toFixed(1)}%</td><td>—</td></tr>
  <tr><td>Seeds beating firing</td><td>${stats.trafficWinsFiring} / ${n}</td><td>—</td><td>—</td></tr>
  <tr><td>Mean cluster-1 activations (traffic)</td><td>${stats.meanC1TrafficAct.toFixed(0)}</td><td>—</td><td>—</td></tr>
</table>

<h2>Success Criteria</h2>
<table>
  <tr><th>Criterion</th><th>Result</th><th>Pass</th></tr>
  <tr>
    <td>Traffic &gt; Firing mean accuracy</td>
    <td>${stats.meanTraffic.toFixed(3)} vs ${stats.meanFiring.toFixed(3)}</td>
    <td class="${sc.trafficBeatsFiring ? 'pass' : 'fail'}">${sc.trafficBeatsFiring ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>Cluster 1 traffic activations mean &gt; 1000</td>
    <td>${stats.meanC1TrafficAct.toFixed(0)}</td>
    <td class="${sc.c1ActivationThousands ? 'pass' : 'fail'}">${sc.c1ActivationThousands ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>≥1 cluster-1 traffic astrocyte success rate &gt; 10%</td>
    <td>${stats.c1SuccessSeeds} seed(s)</td>
    <td class="${sc.c1SuccessAbove10pct ? 'pass' : 'fail'}">${sc.c1SuccessAbove10pct ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>Traffic accept rate 20–50%</td>
    <td>${(stats.meanTrafficAccept * 100).toFixed(1)}%</td>
    <td class="${sc.acceptRateInRange ? 'pass' : 'fail'}">${sc.acceptRateInRange ? '✅ Yes' : '❌ No'}</td>
  </tr>
</table>

<h2>Reward &amp; Accept Rate Trajectories (avg across ${n} seeds)</h2>
<svg width="${chartW}" height="${chartH}" style="border:1px solid #ddd;background:#fafafa;display:block;">
  <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT + h}" x2="${padL + w}" y2="${padT + h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT + h / 2}" x2="${padL + w}" y2="${padT + h / 2}" stroke="#eee" stroke-dasharray="4"/>
  <text x="${padL - 5}" y="${padT + 4}"          text-anchor="end">1.0</text>
  <text x="${padL - 5}" y="${padT + h / 2 + 4}"  text-anchor="end">0.5</text>
  <text x="${padL - 5}" y="${padT + h + 4}"       text-anchor="end">0.0</text>
  <text x="${padL}"       y="${padT + h + 20}" text-anchor="middle">0</text>
  <text x="${padL + w / 2}" y="${padT + h + 20}" text-anchor="middle">${Math.floor(config.episodes / 2)}</text>
  <text x="${padL + w}"   y="${padT + h + 20}" text-anchor="middle">${config.episodes}</text>
  <text x="${padL + w / 2}" y="${chartH}"       text-anchor="middle" fill="#666">Episode</text>
  ${trafficReward ? `<polyline points="${trafficReward}" fill="none" stroke="#2255cc" stroke-width="2"/>` : ''}
  ${firingReward  ? `<polyline points="${firingReward}"  fill="none" stroke="#cc5522" stroke-width="1.5" stroke-dasharray="6,3"/>` : ''}
  ${trafficAccept ? `<polyline points="${trafficAccept}" fill="none" stroke="#2255cc" stroke-width="1.5" opacity="0.4"/>` : ''}
  ${firingAccept  ? `<polyline points="${firingAccept}"  fill="none" stroke="#cc5522" stroke-width="1"   opacity="0.4" stroke-dasharray="6,3"/>` : ''}
</svg>
<div class="legend">
  <span class="leg-traffic">Traffic reward (solid) / accept rate (faded)</span>
  <span class="leg-firing">Firing reward (dashed) / accept rate (faded dashed)</span>
</div>

<h2>Per-Seed Results</h2>
<table>
  <tr>
    <th>Seed</th>
    <th>Traffic acc (d)</th><th>Firing acc (d)</th><th>Control acc</th>
    <th>T&gt;F</th>
    <th>Traffic keep/revert (%)</th><th>Firing keep/revert (%)</th>
  </tr>
  ${perSeedRows}
</table>

<h2>Cluster Activation Breakdown</h2>
<p>Key diagnostic: are cluster 1 (output-side) astrocytes activating in the traffic condition?
Format: total activations (success rate %). Green = C1 traffic &gt; 1000, Red = dormant.</p>
<table>
  <tr><th>Seed</th><th>Traffic C0</th><th>Traffic C1</th><th>Firing C0</th><th>Firing C1</th></tr>
  ${clusterRows}
</table>

<h2>Per-Astrocyte Diagnostics <small>(blue rows = cluster 1; ★ = specialisation; Score = mean per episode)</small></h2>
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
  const lines  = [];

  if (sc.trafficBeatsFiring) {
    lines.push(`Synaptic traffic sensing (${stats.meanTraffic.toFixed(3)}) outperformed ` +
      `neuron-firing sensing (${stats.meanFiring.toFixed(3)}) in mean inference accuracy ` +
      `(${stats.trafficWinsFiring}/${stats.n} seeds). Sensing traffic rather than firing ` +
      `improved learning quality.`);
  } else {
    lines.push(`Synaptic traffic sensing (${stats.meanTraffic.toFixed(3)}) did not surpass ` +
      `neuron-firing sensing (${stats.meanFiring.toFixed(3)}). ` +
      `The activation signal change alone was insufficient for an accuracy improvement.`);
  }

  if (sc.c1ActivationThousands) {
    lines.push(`Cluster 1 astrocytes woke up: mean activation count ${stats.meanC1TrafficAct.toFixed(0)} ` +
      `in the traffic condition (vs near-zero in experiment 021). ` +
      `Output-side perturbation is now occurring.`);
  } else {
    lines.push(`Cluster 1 astrocytes remain under-active: mean activation count ` +
      `${stats.meanC1TrafficAct.toFixed(0)}, below the 1000-activation target. ` +
      `Traffic scores may still be too weak to push cluster 1 thresholds.`);
  }

  if (sc.c1SuccessAbove10pct) {
    lines.push(`Output-side perturbation is productive: ${stats.c1SuccessSeeds} seed(s) had ` +
      `at least one cluster 1 traffic astrocyte achieving >10% success rate. ` +
      `This confirms cluster 1 perturbations can improve network output.`);
  } else {
    lines.push(`No cluster 1 traffic astrocyte exceeded 10% success rate. ` +
      `When output-side astrocytes activate, their perturbations are rarely beneficial. ` +
      `Perturbation strength may need tuning for the cold-start regime.`);
  }

  if (sc.acceptRateInRange) {
    lines.push(`Overall accept rate ${(stats.meanTrafficAccept * 100).toFixed(1)}% is within the ` +
      `20–50% target range, indicating healthy selection pressure without over-fitting.`);
  } else {
    lines.push(`Accept rate ${(stats.meanTrafficAccept * 100).toFixed(1)}% is outside the 20–50% ` +
      `target range. ` +
      (stats.meanTrafficAccept < 0.20
        ? `Too low: perturbations are rarely productive. Consider larger std or smaller territories.`
        : `Too high: almost all perturbations are kept. Selection pressure may be insufficient.`));
  }

  lines.push(`Overall: ${passed}/${total} success criteria passed.`);
  return lines.join(' ');
}
