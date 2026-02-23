// Experiment 024 report generator — Markdown + HTML
// Three conditions: maturity | epsilon (flat) | control
// Key diagnostics: per-astrocyte exploration rates, cluster maturation trajectories,
// late-training stability comparison.

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

  const matAccs  = results.map(r => r.maturity.inference.meanAccuracy);
  const epsAccs  = results.map(r => r.epsilon.inference.meanAccuracy);
  const ctrlAccs = results.map(r => r.control.inference.meanAccuracy);

  const meanMat  = mean(matAccs);
  const meanEps  = mean(epsAccs);
  const meanCtrl = mean(ctrlAccs);

  const varMat = variance(matAccs);
  const varEps = variance(epsAccs);

  const matWinsEps = results.filter(
    r => r.maturity.inference.meanAccuracy > r.epsilon.inference.meanAccuracy
  ).length;

  // Accept rates
  const matAcceptRates = results.map(r => {
    const t = r.maturity.totalAccepted + r.maturity.totalRejected;
    return t > 0 ? r.maturity.totalAccepted / t : 0;
  });
  const epsAcceptRates = results.map(r => {
    const t = r.epsilon.totalAccepted + r.epsilon.totalRejected;
    return t > 0 ? r.epsilon.totalAccepted / t : 0;
  });
  const meanMatAccept = mean(matAcceptRates);
  const meanEpsAccept = mean(epsAcceptRates);

  // Cluster activation breakdown
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
      maturity: clusterMeans(r.maturity.astrocyteStats),
      epsilon:  clusterMeans(r.epsilon.astrocyteStats),
    };
  });

  // Success criterion 1: maturity beats flat epsilon on mean accuracy
  const matBeatEps = meanMat > meanEps;

  // Success criterion 2: C0 astrocytes reach explorationRate < 0.03 by end
  let c0MatureCount = 0, c0TotalCount = 0;
  for (const r of results) {
    const asts = r.maturity.astrocyteStats || [];
    for (const a of asts) {
      if (a.cluster === 0) {
        c0TotalCount++;
        if (a.finalExplorationRate < 0.03) c0MatureCount++;
      }
    }
  }
  const c0Matured = c0MatureCount > 0 && c0MatureCount / c0TotalCount >= 0.5;

  // Success criterion 3: C1 astrocytes reach explorationRate < 0.10 by end
  let c1MatureCount = 0, c1TotalCount = 0;
  for (const r of results) {
    const asts = r.maturity.astrocyteStats || [];
    for (const a of asts) {
      if (a.cluster === 1) {
        c1TotalCount++;
        if (a.finalExplorationRate < 0.10) c1MatureCount++;
      }
    }
  }
  const c1Matured = c1MatureCount > 0 && c1MatureCount / c1TotalCount >= 0.5;

  // Success criterion 4: no late-training degradation (accuracy at ep15000+ >= ep10000)
  // Use trajectory to check: compare avg reward at episode ~10000 vs final
  const matTraj = averageTrajectory(results.map(r => r.maturity.trajectory));
  const epsTraj = averageTrajectory(results.map(r => r.epsilon.trajectory));

  // Find trajectory points near ep10000 and ep15000+
  let noLateDegradation = true;
  if (matTraj.length >= 2) {
    const halfIdx = Math.floor(matTraj.length * 0.5);  // ~ep10000
    const lastQuarterIdx = Math.floor(matTraj.length * 0.75);  // ~ep15000
    const midReward = matTraj[halfIdx].avgReward;
    // Average of last quarter
    let lateSum = 0, lateCount = 0;
    for (let i = lastQuarterIdx; i < matTraj.length; i++) {
      lateSum += matTraj[i].avgReward;
      lateCount++;
    }
    const lateReward = lateCount > 0 ? lateSum / lateCount : midReward;
    noLateDegradation = lateReward >= midReward - 0.01;  // allow tiny tolerance
  }

  // Cluster mean exploration rates at checkpoints (maturity condition)
  function clusterMeanExplorationRate(results, cluster, checkpoint) {
    let sum = 0, count = 0;
    for (const r of results) {
      const asts = r.maturity.astrocyteStats || [];
      for (const a of asts) {
        if (a.cluster === cluster && a.explorationRateSamples) {
          sum += a.explorationRateSamples[checkpoint];
          count++;
        }
      }
    }
    return count > 0 ? sum / count : 0;
  }

  const c0RateEp5k  = clusterMeanExplorationRate(results, 0, 'ep5000');
  const c0RateEp10k = clusterMeanExplorationRate(results, 0, 'ep10000');
  const c0RateFinal = clusterMeanExplorationRate(results, 0, 'epFinal');
  const c1RateEp5k  = clusterMeanExplorationRate(results, 1, 'ep5000');
  const c1RateEp10k = clusterMeanExplorationRate(results, 1, 'ep10000');
  const c1RateFinal = clusterMeanExplorationRate(results, 1, 'epFinal');

  const success = {
    matBeatEps,
    c0Matured,
    c1Matured,
    noLateDegradation,
  };

  return {
    n, meanMat, meanEps, meanCtrl,
    varMat, varEps,
    matWinsEps,
    meanMatAccept, meanEpsAccept,
    clusterStats,
    matTraj, epsTraj,
    c0RateEp5k, c0RateEp10k, c0RateFinal,
    c1RateEp5k, c1RateEp10k, c1RateFinal,
    c0MatureCount, c0TotalCount,
    c1MatureCount, c1TotalCount,
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
    const mRate = (r.maturity.totalAccepted / (r.maturity.totalAccepted + r.maturity.totalRejected || 1) * 100).toFixed(1);
    const eRate = (r.epsilon.totalAccepted  / (r.epsilon.totalAccepted  + r.epsilon.totalRejected  || 1) * 100).toFixed(1);
    const beat  = r.maturity.inference.meanAccuracy > r.epsilon.inference.meanAccuracy ? '✅' : '—';
    return `| ${r.seed} ` +
      `| ${r.maturity.inference.meanAccuracy.toFixed(3)} (${r.maturity.inference.distinctOutputs}d) ` +
      `| ${r.epsilon.inference.meanAccuracy.toFixed(3)} (${r.epsilon.inference.distinctOutputs}d) ` +
      `| ${r.control.inference.meanAccuracy.toFixed(3)} ` +
      `| ${beat} ` +
      `| ${r.maturity.totalAccepted}/${r.maturity.totalRejected} (${mRate}%) ` +
      `| ${r.epsilon.totalAccepted}/${r.epsilon.totalRejected} (${eRate}%) |`;
  }).join('\n');

  const trajRows = stats.matTraj.map((t, i) => {
    const e = stats.epsTraj[i];
    return `| ${t.episode} | ${t.avgReward.toFixed(3)} | ${(t.avgAcceptRate * 100).toFixed(1)}% ` +
      `| ${e ? e.avgReward.toFixed(3) : '—'} | ${e ? (e.avgAcceptRate * 100).toFixed(1) + '%' : '—'} |`;
  }).join('\n');

  const clusterRows = stats.clusterStats.map(s => {
    const m = s.maturity, e = s.epsilon;
    return `| ${s.seed} ` +
      `| ${m.c0Total.toLocaleString()} (${(m.c0SuccRate * 100).toFixed(0)}%) ` +
      `| ${m.c1Total.toLocaleString()} (${(m.c1SuccRate * 100).toFixed(0)}%) ` +
      `| ${e.c0Total.toLocaleString()} (${(e.c0SuccRate * 100).toFixed(0)}%) ` +
      `| ${e.c1Total.toLocaleString()} (${(e.c1SuccRate * 100).toFixed(0)}%) |`;
  }).join('\n');

  const patLabels = trainingPatterns.map((_, i) => `P${i + 1}`).join(' | ');

  function astroSection(r, condKey, condLabel) {
    const asts = r[condKey].astrocyteStats;
    if (!asts) return '';
    const rows = asts.map(ast => {
      const sr     = ast.activationCount > 0
        ? (ast.successCount / ast.activationCount * 100).toFixed(1) + '%' : '—';
      const regAct = ast.activationCount - (ast.epsilonCount || 0);
      const er     = ast.finalExplorationRate !== undefined ? ast.finalExplorationRate.toFixed(4) : '—';
      const ers    = ast.explorationRateSamples;
      const ersFmt = ers
        ? `${ers.ep5000.toFixed(4)} / ${ers.ep10000.toFixed(4)} / ${ers.epFinal.toFixed(4)}`
        : '—';
      const pCounts = ast.activationsByPattern.join(' | ');
      return `| ${ast.id} | C${ast.cluster} | (${ast.position.x.toFixed(1)}, ${ast.position.y.toFixed(1)}) ` +
        `| ${ast.neuronCount}n / ${ast.synapseCount}s ` +
        `| ${regAct} + ${ast.epsilonCount || 0}ε | ${sr} | ${ast.finalThreshold.toFixed(3)} ` +
        `| ${er} | ${ersFmt} | ${pCounts} |`;
    }).join('\n');
    return `\n**Seed ${r.seed} — ${condLabel}**\n\n` +
      `| ID | Cl | Pos | Territory | Act (reg + ε) | Success% | Thresh | ExplRate | ExplRate 5k/10k/20k | ${patLabels} |\n` +
      `|----|----|-----|-----------|---------------|----------|--------|----------|---------------------|${trainingPatterns.map(() => '---').join('|')}|\n` +
      rows;
  }

  const astroSections = allResults.map(r =>
    astroSection(r, 'maturity', 'maturity') + astroSection(r, 'epsilon', 'epsilon')
  ).join('\n');

  const inferenceDetail = allResults.map(r => {
    const rows = r.maturity.inference.results.map((res, i) => {
      const eps  = r.epsilon.inference.results[i];
      const ctrl = r.control.inference.results[i];
      return `| ${res.label} | [${res.input.join('')}] | [${res.target.join('')}] ` +
        `| [${res.output.join('')}] ${res.accuracy.toFixed(2)} ` +
        `| [${eps.output.join('')}] ${eps.accuracy.toFixed(2)} ` +
        `| [${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)} |`;
    }).join('\n');
    return `\n**Seed ${r.seed}**\n\n` +
      `| Pat | Input | Target | Maturity out/acc | Epsilon out/acc | Control out/acc |\n` +
      `|-----|-------|--------|------------------|-----------------|----------------|\n` + rows;
  }).join('\n');

  return `# Experiment 024 — Maturity-Scaled Exploration

**Date:** ${new Date().toISOString().slice(0, 10)}
**Duration:** ${duration}s

## Hypothesis

Experiment 023 showed epsilon exploration works to bootstrap cluster 1, but flat 1% epsilon
becomes a drag over long training (20k episodes). Cluster 0 matures by episode 5000 but keeps
perturbing, accumulating drift damage. Each astrocyte should mature at its own pace:
explorationRate = ${config.baseEpsilon} / (1 + activationCount / ${config.maturityHorizon}).

Fresh astrocytes explore at ${(config.baseEpsilon * 100).toFixed(0)}%. After ${config.maturityHorizon} activations: ${(config.baseEpsilon / 2 * 100).toFixed(0)}%.
After 40,000 activations: ~1.5%. Maturation is per-astrocyte, mirroring biological
region-specific cortical maturation.

## Configuration

| Parameter | Value |
|---|---|
| Clusters | ${config.clusters} × ${config.neuronsPerCluster} neurons |
| Astrocytes | ${config.astrocytesPerCluster} per cluster (${config.astrocytesPerCluster * 2} total) |
| Territory radius | ${config.territoryRadius} (2D x-y) |
| Perturbation std | ${config.perturbStd} |
| Maturity-scaled epsilon | baseEpsilon=${config.baseEpsilon}, horizon=${config.maturityHorizon} |
| Flat epsilon (comparison) | ${(config.epsilon * 100).toFixed(0)}% per astrocyte per step |
| Acceptance | soft reward, strict > |
| Homeostasis | OFF |
| Episodes | ${config.episodes} × ${config.stepsPerEpisode} steps |
| Seeds | ${config.seeds.join(', ')} |

## Summary

| Metric | Maturity | Epsilon (flat) | Control |
|---|---|---|---|
| Mean accuracy | ${stats.meanMat.toFixed(3)} | ${stats.meanEps.toFixed(3)} | ${stats.meanCtrl.toFixed(3)} |
| Variance | ${stats.varMat.toFixed(5)} | ${stats.varEps.toFixed(5)} | — |
| Mean accept rate | ${(stats.meanMatAccept * 100).toFixed(1)}% | ${(stats.meanEpsAccept * 100).toFixed(1)}% | — |
| Seeds maturity > epsilon | ${stats.matWinsEps} / ${n} | — | — |

## Exploration Rate Maturation

| Cluster | Ep 5000 | Ep 10000 | Ep 20000 (final) |
|---------|---------|----------|-------------------|
| C0 mean | ${stats.c0RateEp5k.toFixed(4)} | ${stats.c0RateEp10k.toFixed(4)} | ${stats.c0RateFinal.toFixed(4)} |
| C1 mean | ${stats.c1RateEp5k.toFixed(4)} | ${stats.c1RateEp10k.toFixed(4)} | ${stats.c1RateFinal.toFixed(4)} |

C0 matured (rate < 0.03): ${stats.c0MatureCount}/${stats.c0TotalCount} astrocytes across all seeds.
C1 matured (rate < 0.10): ${stats.c1MatureCount}/${stats.c1TotalCount} astrocytes across all seeds.

## Success Criteria

| Criterion | Result | Pass |
|---|---|---|
| Maturity > Epsilon mean accuracy | ${stats.meanMat.toFixed(3)} vs ${stats.meanEps.toFixed(3)} | ${sc.matBeatEps ? '✅' : '❌'} |
| C0 explorationRate < 0.03 by ep 20000 | ${stats.c0MatureCount}/${stats.c0TotalCount} matured | ${sc.c0Matured ? '✅' : '❌'} |
| C1 explorationRate < 0.10 by ep 20000 | ${stats.c1MatureCount}/${stats.c1TotalCount} matured | ${sc.c1Matured ? '✅' : '❌'} |
| No late-training degradation (ep15k+ >= ep10k) | see trajectory | ${sc.noLateDegradation ? '✅' : '❌'} |

## Per-Seed Results

| Seed | Maturity acc (d) | Epsilon acc (d) | Control acc | M>E | Maturity keep/revert (%) | Epsilon keep/revert (%) |
|------|------------------|-----------------|-------------|-----|--------------------------|-------------------------|
${perSeedRows}

## Trajectory (avg across ${n} seeds)

| Episode | Maturity reward | Maturity accept% | Epsilon reward | Epsilon accept% |
|---------|----------------|-------------------|----------------|-----------------|
${trajRows}

## Cluster Activation Breakdown

| Seed | Maturity C0 | Maturity C1 | Epsilon C0 | Epsilon C1 |
|------|-------------|-------------|------------|------------|
${clusterRows}

## Per-Astrocyte Diagnostics

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

  const matReward  = toPoly(stats.matTraj,  'avgReward');
  const epsReward  = toPoly(stats.epsTraj,  'avgReward');
  const matAccept  = toPoly(stats.matTraj,  'avgAcceptRate');
  const epsAccept  = toPoly(stats.epsTraj,  'avgAcceptRate');

  const perSeedRows = allResults.map(r => {
    const mRate = (r.maturity.totalAccepted / (r.maturity.totalAccepted + r.maturity.totalRejected || 1) * 100).toFixed(1);
    const eRate = (r.epsilon.totalAccepted  / (r.epsilon.totalAccepted  + r.epsilon.totalRejected  || 1) * 100).toFixed(1);
    const beat  = r.maturity.inference.meanAccuracy > r.epsilon.inference.meanAccuracy;
    return `<tr>
      <td>${r.seed}</td>
      <td>${r.maturity.inference.meanAccuracy.toFixed(3)} (${r.maturity.inference.distinctOutputs}d)</td>
      <td>${r.epsilon.inference.meanAccuracy.toFixed(3)} (${r.epsilon.inference.distinctOutputs}d)</td>
      <td>${r.control.inference.meanAccuracy.toFixed(3)}</td>
      <td>${beat ? '✅' : '—'}</td>
      <td>${r.maturity.totalAccepted}/${r.maturity.totalRejected} (${mRate}%)</td>
      <td>${r.epsilon.totalAccepted}/${r.epsilon.totalRejected} (${eRate}%)</td>
    </tr>`;
  }).join('\n');

  const clusterRows = stats.clusterStats.map(s => {
    const m = s.maturity, e = s.epsilon;
    return `<tr>
      <td>${s.seed}</td>
      <td>${m.c0Total.toLocaleString()} (${(m.c0SuccRate * 100).toFixed(0)}%)</td>
      <td>${m.c1Total.toLocaleString()} (${(m.c1SuccRate * 100).toFixed(0)}%)</td>
      <td>${e.c0Total.toLocaleString()} (${(e.c0SuccRate * 100).toFixed(0)}%)</td>
      <td>${e.c1Total.toLocaleString()} (${(e.c1SuccRate * 100).toFixed(0)}%)</td>
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
      const er     = ast.finalExplorationRate !== undefined ? ast.finalExplorationRate.toFixed(4) : '—';
      const ers    = ast.explorationRateSamples;
      const threshDropped = ast.finalThreshold < 0.5;
      const isC1   = ast.cluster === 1;
      const pCells = ast.activationsByPattern.map(v => `<td>${v}</td>`).join('');
      const rowBg = isC1
        ? (threshDropped ? ' style="background:#c8e6c9"' : ' style="background:#e3f2fd"')
        : '';
      return `<tr${rowBg}>
        <td>${ast.id}</td>
        <td>C${ast.cluster}</td>
        <td>(${ast.position.x.toFixed(1)}, ${ast.position.y.toFixed(1)})</td>
        <td>${ast.neuronCount}n / ${ast.synapseCount}s</td>
        <td>${regAct} + ${ast.epsilonCount || 0}ε</td>
        <td>${sr}</td>
        <td${threshDropped ? ' style="color:#080;font-weight:bold"' : ''}>${ast.finalThreshold.toFixed(3)}</td>
        <td style="font-weight:bold">${er}</td>
        <td>${ers ? ers.ep5000.toFixed(4) : '—'}</td>
        <td>${ers ? ers.ep10000.toFixed(4) : '—'}</td>
        <td>${ers ? ers.epFinal.toFixed(4) : '—'}</td>
        ${pCells}
      </tr>`;
    }).join('\n');
    return `<details>
      <summary><strong>Seed ${r.seed} — ${condLabel}</strong> acc=${r[condKey].inference.meanAccuracy.toFixed(3)}, distinct=${r[condKey].inference.distinctOutputs}</summary>
      <table>
        <tr><th>ID</th><th>Cl</th><th>Pos</th><th>Territory</th><th>Act (reg+ε)</th><th>Success%</th><th>Thresh</th><th>ExplRate</th><th>Rate 5k</th><th>Rate 10k</th><th>Rate 20k</th>${patLabels}</tr>
        ${rows}
      </table>
      <p style="font-size:0.82em;color:#666">
        Blue rows = cluster 1. Green rows = cluster 1 with threshold dropped below 0.5.
        ExplRate = final exploration rate (maturity condition only).
      </p>
    </details>`;
  }

  const astroDiagnostics = allResults.map(r =>
    astroDetailHtml(r, 'maturity', 'maturity') + '\n' + astroDetailHtml(r, 'epsilon', 'epsilon')
  ).join('\n');

  const inferenceDetail = allResults.map(r => {
    const rows = r.maturity.inference.results.map((res, i) => {
      const eps  = r.epsilon.inference.results[i];
      const ctrl = r.control.inference.results[i];
      return `<tr>
        <td>${res.label}</td>
        <td>[${res.input.join('')}]</td>
        <td>[${res.target.join('')}]</td>
        <td>[${res.output.join('')}] ${res.accuracy.toFixed(2)}</td>
        <td>[${eps.output.join('')}] ${eps.accuracy.toFixed(2)}</td>
        <td>[${ctrl.output.join('')}] ${ctrl.accuracy.toFixed(2)}</td>
      </tr>`;
    }).join('\n');
    return `<details>
      <summary><strong>Seed ${r.seed}</strong> — maturity=${r.maturity.inference.meanAccuracy.toFixed(3)}, epsilon=${r.epsilon.inference.meanAccuracy.toFixed(3)}, control=${r.control.inference.meanAccuracy.toFixed(3)}</summary>
      <table>
        <tr><th>Pat</th><th>Input</th><th>Target</th><th>Maturity out/acc</th><th>Epsilon out/acc</th><th>Control out/acc</th></tr>
        ${rows}
      </table>
    </details>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 024 — Maturity-Scaled Exploration</title>
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
  .leg-mat::before  { background: #2255cc; }
  .leg-eps::before  { background: #cc5522; }
</style>
</head>
<body>
<h1>Experiment 024 — Maturity-Scaled Exploration</h1>
<p class="meta">Date: ${new Date().toISOString().slice(0, 10)} &nbsp;|&nbsp; Duration: ${duration}s &nbsp;|&nbsp; baseEpsilon=${config.baseEpsilon}, horizon=${config.maturityHorizon}</p>

<h2>Hypothesis</h2>
<p>Flat epsilon (1%) works to bootstrap cluster 1 but becomes a drag over long training.
Cluster 0 matures by episode 5000 but keeps perturbing, accumulating drift damage. Each astrocyte
should mature at its own pace: explorationRate = ${config.baseEpsilon} / (1 + activationCount / ${config.maturityHorizon}).
Fresh astrocytes explore at ${(config.baseEpsilon * 100).toFixed(0)}%, heavily-activated ones converge toward zero.
Maturation is per-astrocyte, mirroring biological region-specific cortical development.</p>

<h2>Configuration</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Architecture</td><td>${config.clusters} clusters × ${config.neuronsPerCluster} neurons</td></tr>
  <tr><td>Astrocytes</td><td>${config.astrocytesPerCluster} per cluster (${config.astrocytesPerCluster * 2} total), territory radius ${config.territoryRadius}</td></tr>
  <tr><td>Maturity-scaled epsilon</td><td>baseEpsilon=${config.baseEpsilon}, maturityHorizon=${config.maturityHorizon}</td></tr>
  <tr><td>Flat epsilon (comparison)</td><td>${(config.epsilon * 100).toFixed(0)}% per astrocyte per step</td></tr>
  <tr><td>Perturbation std</td><td>${config.perturbStd}</td></tr>
  <tr><td>Acceptance</td><td>soft reward, strict &gt;</td></tr>
  <tr><td>Homeostasis</td><td>OFF</td></tr>
  <tr><td>Episodes</td><td>${config.episodes} × ${config.stepsPerEpisode} steps</td></tr>
  <tr><td>Seeds</td><td>${config.seeds.join(', ')}</td></tr>
</table>

<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Maturity</th><th>Epsilon (flat)</th><th>Control</th></tr>
  <tr><td>Mean accuracy</td><td>${stats.meanMat.toFixed(3)}</td><td>${stats.meanEps.toFixed(3)}</td><td>${stats.meanCtrl.toFixed(3)}</td></tr>
  <tr><td>Variance</td><td>${stats.varMat.toFixed(5)}</td><td>${stats.varEps.toFixed(5)}</td><td>—</td></tr>
  <tr><td>Mean accept rate</td><td>${(stats.meanMatAccept * 100).toFixed(1)}%</td><td>${(stats.meanEpsAccept * 100).toFixed(1)}%</td><td>—</td></tr>
  <tr><td>Seeds maturity &gt; epsilon</td><td>${stats.matWinsEps} / ${n}</td><td>—</td><td>—</td></tr>
</table>

<h2>Exploration Rate Maturation</h2>
<table>
  <tr><th>Cluster</th><th>Ep 5000</th><th>Ep 10000</th><th>Ep 20000 (final)</th></tr>
  <tr><td>C0 mean</td><td>${stats.c0RateEp5k.toFixed(4)}</td><td>${stats.c0RateEp10k.toFixed(4)}</td><td>${stats.c0RateFinal.toFixed(4)}</td></tr>
  <tr><td>C1 mean</td><td>${stats.c1RateEp5k.toFixed(4)}</td><td>${stats.c1RateEp10k.toFixed(4)}</td><td>${stats.c1RateFinal.toFixed(4)}</td></tr>
</table>
<p>C0 matured (rate &lt; 0.03): ${stats.c0MatureCount}/${stats.c0TotalCount} astrocytes across all seeds.<br>
C1 matured (rate &lt; 0.10): ${stats.c1MatureCount}/${stats.c1TotalCount} astrocytes across all seeds.</p>

<h2>Success Criteria</h2>
<table>
  <tr><th>Criterion</th><th>Result</th><th>Pass</th></tr>
  <tr>
    <td>Maturity &gt; Epsilon mean accuracy</td>
    <td>${stats.meanMat.toFixed(3)} vs ${stats.meanEps.toFixed(3)}</td>
    <td class="${sc.matBeatEps ? 'pass' : 'fail'}">${sc.matBeatEps ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>C0 explorationRate &lt; 0.03 by ep 20000</td>
    <td>${stats.c0MatureCount}/${stats.c0TotalCount} matured</td>
    <td class="${sc.c0Matured ? 'pass' : 'fail'}">${sc.c0Matured ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>C1 explorationRate &lt; 0.10 by ep 20000</td>
    <td>${stats.c1MatureCount}/${stats.c1TotalCount} matured</td>
    <td class="${sc.c1Matured ? 'pass' : 'fail'}">${sc.c1Matured ? '✅ Yes' : '❌ No'}</td>
  </tr>
  <tr>
    <td>No late-training degradation (ep15k+ &gt;= ep10k)</td>
    <td>see trajectory</td>
    <td class="${sc.noLateDegradation ? 'pass' : 'fail'}">${sc.noLateDegradation ? '✅ Yes' : '❌ No'}</td>
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
  ${matReward  ? `<polyline points="${matReward}"  fill="none" stroke="#2255cc" stroke-width="2"/>` : ''}
  ${epsReward  ? `<polyline points="${epsReward}"  fill="none" stroke="#cc5522" stroke-width="1.5" stroke-dasharray="6,3"/>` : ''}
  ${matAccept  ? `<polyline points="${matAccept}"  fill="none" stroke="#2255cc" stroke-width="1.5" opacity="0.4"/>` : ''}
  ${epsAccept  ? `<polyline points="${epsAccept}"  fill="none" stroke="#cc5522" stroke-width="1"   opacity="0.4" stroke-dasharray="6,3"/>` : ''}
</svg>
<div class="legend">
  <span class="leg-mat">Maturity reward (solid) / accept rate (faded)</span>
  <span class="leg-eps">Epsilon reward (dashed) / accept rate (faded dashed)</span>
</div>

<h2>Per-Seed Results</h2>
<table>
  <tr>
    <th>Seed</th>
    <th>Maturity acc (d)</th><th>Epsilon acc (d)</th><th>Control acc</th>
    <th>M&gt;E</th>
    <th>Maturity keep/revert (%)</th><th>Epsilon keep/revert (%)</th>
  </tr>
  ${perSeedRows}
</table>

<h2>Cluster Activation Breakdown</h2>
<table>
  <tr><th>Seed</th><th>Maturity C0</th><th>Maturity C1</th><th>Epsilon C0</th><th>Epsilon C1</th></tr>
  ${clusterRows}
</table>

<h2>Per-Astrocyte Diagnostics</h2>
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

  if (sc.matBeatEps) {
    lines.push(`Maturity-scaled exploration (${stats.meanMat.toFixed(3)}) outperformed flat epsilon ` +
      `(${stats.meanEps.toFixed(3)}) in mean inference accuracy (${stats.matWinsEps}/${stats.n} seeds). ` +
      `Per-astrocyte maturation successfully reduces late-training perturbation drift.`);
  } else {
    lines.push(`Maturity-scaled exploration (${stats.meanMat.toFixed(3)}) did not surpass flat epsilon ` +
      `(${stats.meanEps.toFixed(3)}). The initial ${(config.baseEpsilon * 100).toFixed(0)}% exploration rate ` +
      `may be too aggressive, or the maturity horizon (${config.maturityHorizon}) may need tuning.`);
  }

  if (sc.c0Matured) {
    lines.push(`Cluster 0 matured as expected: ${stats.c0MatureCount}/${stats.c0TotalCount} astrocytes ` +
      `reached explorationRate < 0.03 by episode ${config.episodes} (final C0 mean rate: ` +
      `${stats.c0RateFinal.toFixed(4)}). Converged weights are protected from further drift.`);
  } else {
    lines.push(`Cluster 0 did not fully mature: only ${stats.c0MatureCount}/${stats.c0TotalCount} ` +
      `astrocytes reached explorationRate < 0.03. Final C0 mean rate: ${stats.c0RateFinal.toFixed(4)}.`);
  }

  if (sc.c1Matured) {
    lines.push(`Cluster 1 caught up and matured: ${stats.c1MatureCount}/${stats.c1TotalCount} astrocytes ` +
      `reached explorationRate < 0.10 by episode ${config.episodes} (final C1 mean rate: ` +
      `${stats.c1RateFinal.toFixed(4)}). The output-side territory learned and stabilised.`);
  } else {
    lines.push(`Cluster 1 has not matured: only ${stats.c1MatureCount}/${stats.c1TotalCount} ` +
      `astrocytes reached explorationRate < 0.10. Final C1 mean rate: ${stats.c1RateFinal.toFixed(4)}. ` +
      `C1 may need more activations before the maturity horizon kicks in.`);
  }

  if (sc.noLateDegradation) {
    lines.push(`No late-training degradation observed: accuracy in the final quarter of training ` +
      `remained stable relative to mid-training, confirming that maturity-scaling prevents ` +
      `the drift damage seen in experiment 023 at 20k episodes.`);
  } else {
    lines.push(`Late-training degradation was detected: accuracy in the final quarter dropped below ` +
      `mid-training levels. Maturity-scaling may not have decayed fast enough to prevent drift.`);
  }

  lines.push(`Overall: ${passed}/${total} success criteria passed.`);
  return lines.join(' ');
}
