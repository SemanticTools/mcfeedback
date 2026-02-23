// Generate experiment report in both Markdown and HTML formats

import { writeFileSync, mkdirSync } from 'fs';

export function generateReport(log, network, config, startTime) {
  mkdirSync('./reports', { recursive: true });

  const duration = ((Date.now() - startTime) / 1000).toFixed(1);
  const initial = log[0].avgReward;
  const final = log[log.length - 1].avgReward;
  const best = Math.max(...log.map(e => e.avgReward));
  const totalAccepted = log.reduce((s, e) => s + e.accepted, 0);
  const totalRejected = log.reduce((s, e) => s + e.rejected, 0);
  const acceptRate = (totalAccepted / (totalAccepted + totalRejected) * 100).toFixed(1);
  const avgEligible = (log.reduce((s, e) => s + e.avgEligible, 0) / log.length).toFixed(1);

  const summary = { duration, initial, final, best, totalAccepted, totalRejected, acceptRate, avgEligible };

  writeFileSync('./reports/experiment-020.md', buildMarkdown(log, network, config, summary));
  writeFileSync('./reports/experiment-020.html', buildHtml(log, network, config, summary));

  console.log('\nReports written to ./reports/experiment-020.md and ./reports/experiment-020.html');
}

function buildMarkdown(log, network, config, s) {
  const sampleRows = sampleLog(log);

  const rows = sampleRows.map(e =>
    `| ${e.episode} | ${e.avgReward.toFixed(3)} | ${e.accepted} | ${e.rejected} | ${e.avgEligible.toFixed(1)} |`
  ).join('\n');

  return `# Experiment 020 — Cursor-Only Learning

**Date:** ${new Date().toISOString().slice(0, 10)}
**Duration:** ${s.duration}s

## Hypothesis

Global reward works if you only change a few synapses at a time. A cursor restricts perturbation to a local spatial neighbourhood. One scalar reward is enough to evaluate a small local change — like a genetic algorithm running inside a network.

## Configuration

| Parameter | Value |
|---|---|
| Clusters | ${config.clusters} |
| Neurons per cluster | ${config.neuronsPerCluster} |
| Total neurons | ${network.neurons.length} |
| Total synapses | ${network.synapses.length} |
| Intra-cluster connectivity | ${config.intraProb * 100}% |
| Inter-cluster connectivity | ${config.interProb * 100}% |
| Cursor radius | ${config.cursorRadius} |
| Perturbation std | ${config.perturbStd} |
| Episodes | ${config.episodes} |
| Steps per episode | ${config.stepsPerEpisode} |
| Target fire rate | ${config.targetFireRate} |
| Threshold adjust rate | ${config.thresholdAdjustRate} |

## Results

| Metric | Value |
|---|---|
| Initial avg reward | ${s.initial.toFixed(3)} |
| Final avg reward | ${s.final.toFixed(3)} |
| Best avg reward | ${s.best.toFixed(3)} |
| Perturbation accept rate | ${s.acceptRate}% |
| Avg eligible synapses/step | ${s.avgEligible} |

## Episode Log (sample)

| Episode | Avg Reward | Accepted | Rejected | Avg Eligible |
|---|---|---|---|---|
${rows}

## Conclusion

${buildConclusion(s)}
`;
}

function buildConclusion(s) {
  const improved = s.final > s.initial + 0.02;
  const learned = s.best > 0.7;

  if (learned && improved) {
    return `The cursor mechanism successfully drove learning. Reward improved from ` +
      `${s.initial.toFixed(3)} to ${s.final.toFixed(3)} (best: ${s.best.toFixed(3)}). ` +
      `The hypothesis is supported: restricting perturbation to a local neighbourhood ` +
      `allows a single scalar reward to assign credit.`;
  } else if (improved) {
    return `Reward showed modest improvement (${s.initial.toFixed(3)} → ${s.final.toFixed(3)}). ` +
      `The cursor mechanism shows partial signal but did not reach strong performance. ` +
      `Further tuning of radius, perturbation scale, or task complexity may help.`;
  } else {
    return `Reward did not improve significantly (${s.initial.toFixed(3)} → ${s.final.toFixed(3)}). ` +
      `The cursor mechanism as configured did not produce reliable learning on this task. ` +
      `Possible causes: cursor radius too large/small, perturbation scale mismatched, ` +
      `or the single-pass propagation lacks sufficient expressivity for the mapping.`;
  }
}

function buildHtml(log, network, config, s) {
  const chartWidth = 700;
  const chartHeight = 250;
  const padL = 50, padR = 20, padT = 20, padB = 30;
  const w = chartWidth - padL - padR;
  const h = chartHeight - padT - padB;

  const points = log.map((e, i) => {
    const px = padL + (i / (log.length - 1)) * w;
    const py = padT + (1 - e.avgReward) * h;
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');

  const sampleRows = sampleLog(log).map(e =>
    `<tr><td>${e.episode}</td><td>${e.avgReward.toFixed(3)}</td>` +
    `<td>${e.accepted}</td><td>${e.rejected}</td><td>${e.avgEligible.toFixed(1)}</td></tr>`
  ).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 020 — Cursor-Only Learning</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { margin-top: 2em; color: #444; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; }
  th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }
  th { background: #f0f0f0; }
  .meta { color: #666; font-size: 0.9em; }
  .conclusion { background: #f8f8f8; border-left: 4px solid #555; padding: 12px 16px; }
  svg text { font-family: system-ui, sans-serif; font-size: 11px; }
</style>
</head>
<body>
<h1>Experiment 020 — Cursor-Only Learning</h1>
<p class="meta">Date: ${new Date().toISOString().slice(0, 10)} &nbsp;|&nbsp; Duration: ${s.duration}s</p>

<h2>Hypothesis</h2>
<p>Global reward works if you only change a few synapses at a time. A cursor restricts perturbation to a local spatial neighbourhood — one scalar reward is enough to evaluate that small local change. Like a genetic algorithm running inside a network.</p>

<h2>Configuration</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Clusters</td><td>${config.clusters}</td></tr>
  <tr><td>Neurons per cluster</td><td>${config.neuronsPerCluster}</td></tr>
  <tr><td>Total neurons</td><td>${network.neurons.length}</td></tr>
  <tr><td>Total synapses</td><td>${network.synapses.length}</td></tr>
  <tr><td>Intra-cluster connectivity</td><td>${config.intraProb * 100}%</td></tr>
  <tr><td>Inter-cluster connectivity</td><td>${config.interProb * 100}%</td></tr>
  <tr><td>Cursor radius</td><td>${config.cursorRadius}</td></tr>
  <tr><td>Perturbation std</td><td>${config.perturbStd}</td></tr>
  <tr><td>Episodes</td><td>${config.episodes}</td></tr>
  <tr><td>Steps per episode</td><td>${config.stepsPerEpisode}</td></tr>
  <tr><td>Target fire rate</td><td>${config.targetFireRate}</td></tr>
  <tr><td>Threshold adjust rate</td><td>${config.thresholdAdjustRate}</td></tr>
</table>

<h2>Results</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Initial avg reward</td><td>${s.initial.toFixed(3)}</td></tr>
  <tr><td>Final avg reward</td><td>${s.final.toFixed(3)}</td></tr>
  <tr><td>Best avg reward</td><td>${s.best.toFixed(3)}</td></tr>
  <tr><td>Perturbation accept rate</td><td>${s.acceptRate}%</td></tr>
  <tr><td>Avg eligible synapses/step</td><td>${s.avgEligible}</td></tr>
</table>

<h2>Reward Over Time</h2>
<svg width="${chartWidth}" height="${chartHeight}" style="border:1px solid #ddd; background:#fafafa;">
  <!-- Axes -->
  <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + h}" stroke="#999" stroke-width="1"/>
  <line x1="${padL}" y1="${padT + h}" x2="${padL + w}" y2="${padT + h}" stroke="#999" stroke-width="1"/>
  <!-- Y labels -->
  <text x="${padL - 5}" y="${padT + 4}" text-anchor="end">1.0</text>
  <text x="${padL - 5}" y="${padT + h / 2 + 4}" text-anchor="end">0.5</text>
  <text x="${padL - 5}" y="${padT + h + 4}" text-anchor="end">0.0</text>
  <!-- X labels -->
  <text x="${padL}" y="${padT + h + 20}" text-anchor="middle">0</text>
  <text x="${padL + w / 2}" y="${padT + h + 20}" text-anchor="middle">${Math.floor(log.length / 2)}</text>
  <text x="${padL + w}" y="${padT + h + 20}" text-anchor="middle">${log.length}</text>
  <text x="${padL + w / 2}" y="${chartHeight}" text-anchor="middle" fill="#666">Episode</text>
  <!-- Grid lines -->
  <line x1="${padL}" y1="${padT + h / 2}" x2="${padL + w}" y2="${padT + h / 2}" stroke="#eee" stroke-width="1" stroke-dasharray="4"/>
  <!-- Reward line -->
  <polyline points="${points}" fill="none" stroke="#2255cc" stroke-width="1.5"/>
</svg>

<h2>Episode Log (sample)</h2>
<table>
  <tr><th>Episode</th><th>Avg Reward</th><th>Accepted</th><th>Rejected</th><th>Avg Eligible</th></tr>
  ${sampleRows}
</table>

<h2>Conclusion</h2>
<div class="conclusion">${buildConclusion(s)}</div>
</body>
</html>`;
}

// Return ~15 evenly spaced episodes from the log
function sampleLog(log) {
  const n = Math.min(15, log.length);
  const result = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.round((i / (n - 1)) * (log.length - 1));
    result.push(log[idx]);
  }
  return result;
}
