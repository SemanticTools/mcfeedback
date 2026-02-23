import { writeFileSync } from 'node:fs';
import { createNetwork } from '../src/network.mjs';
import { step } from '../src/engine.mjs';

const BASE_CONFIG = {
  clustersCount:               2,
  neuronsPerCluster:           30,
  modulatoryPerCluster:        2,
  intraClusterConnectionProb:  0.6,
  interClusterConnectionProb:  0.5,
  clusterSpacing:              10.0,
  neuronSpread:                2.0,
  initialThreshold:            0.5,
  targetFireRate:              0.2,
  thresholdAdjustRate:         0.01,
  ambientRadius:               3.0,
  ambientFalloff:              'inverse',
  initialWeightRange:          [-0.1, 0.1],
  coActivationStrength:        1.0,
  coSilenceStrength:           0.5,
  mismatchStrength:           -0.5,
  ambientThreshold:            0.3,
  activityHistoryDecay:        0.95,
  activityHistoryMinimum:      0.1,
  chemicalDiffusionRadius:     15.0,
  chemicalFalloff:             'inverse',
  chemicalDecayRate:           0.5,
  positiveRewardStrength:      1.0,
  negativeRewardStrength:     -1.0,
  learningRate:                0.01,
  maxWeightDelta:              0.1,
  maxWeightMagnitude:          2.0,
  weightDecay:                 0.005,
  inputSize:                   5,
  outputSize:                  5,
  trainingEpisodes:            1000,
  reportInterval:              10,
};

const PATTERNS = [
  { input: [1,0,1,0,1], target: [0,1,0,1,0] },
  { input: [1,1,0,0,0], target: [0,0,1,1,1] },
  { input: [1,0,0,0,1], target: [0,1,1,1,0] },
  { input: [0,1,0,1,0], target: [1,0,1,0,1] },
];

const CONDITIONS = [
  {
    label:    'Baseline',
    overrides: {
      ambientRadius:           0,
      _skipDampening:          true,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Ambient only',
    overrides: {
      _skipDampening:          true,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Dampening only',
    overrides: {
      ambientRadius:           0,
      chemicalDiffusionRadius: 1000,
      chemicalFalloff:         'constant',
    },
  },
  {
    label:    'Full model',
    overrides: {
      chemicalDiffusionRadius: 15,
    },
  },
];

function runCondition(label, configOverrides) {
  const config  = { ...BASE_CONFIG, ...configOverrides };
  const network = createNetwork(config);
  const history = [];   // one entry per reportInterval

  let acc10sum   = 0;
  let loss10sum  = 0;
  let mw10sum    = 0;
  let fr10sum    = 0;
  let thr10sum   = 0;
  let asf10sum   = 0;
  let count      = 0;

  for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
    const pattern = PATTERNS[(ep - 1) % PATTERNS.length];
    const m = step(network, pattern.input, pattern.target);

    acc10sum  += m.accuracy;
    loss10sum += m.loss;
    mw10sum   += m.meanWeight;
    fr10sum   += m.meanFireRate;
    thr10sum  += m.meanThreshold;
    asf10sum  += m.activeSynFrac;
    count++;

    if (ep % config.reportInterval === 0) {
      history.push({
        episode:        ep,
        accuracy:       +(acc10sum  / count).toFixed(4),
        loss:           +(loss10sum / count).toFixed(4),
        meanWeight:     +(mw10sum   / count).toFixed(4),
        meanFireRate:   +(fr10sum   / count).toFixed(4),
        meanThreshold:  +(thr10sum  / count).toFixed(4),
        activeSynFrac:  +(asf10sum  / count).toFixed(4),
      });
      acc10sum = loss10sum = mw10sum = fr10sum = thr10sum = asf10sum = count = 0;
    }
  }

  // Final accuracy across all patterns
  let totalCorrect = 0;
  for (const pattern of PATTERNS) {
    const m = step(network, pattern.input, pattern.target);
    totalCorrect += m.accuracy;
  }
  const finalAccuracy = +(totalCorrect / PATTERNS.length).toFixed(4);

  process.stdout.write(`  ${label} done\n`);
  return { label, config: { ...config }, history, finalAccuracy };
}

process.stdout.write('Running conditions...\n');
const results = CONDITIONS.map(c => runCondition(c.label, c.overrides ?? {}));

// ─── HTML generation ───────────────────────────────────────────────────────

const COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'];
const METRICS = [
  { key: 'accuracy',      label: 'Accuracy',           yMin: 0,    yMax: 1 },
  { key: 'loss',          label: 'Loss',                yMin: 0,    yMax: null },
  { key: 'meanWeight',    label: 'Mean |Weight|',       yMin: 0,    yMax: null },
  { key: 'meanFireRate',  label: 'Mean Fire Rate',      yMin: 0,    yMax: 1 },
  { key: 'meanThreshold', label: 'Mean Threshold',      yMin: null, yMax: null },
  { key: 'activeSynFrac', label: 'Active Synapse Frac', yMin: 0,    yMax: 1 },
];

// Build chart datasets for each metric
function datasetsForMetric(key) {
  return results.map((r, i) => ({
    label:       r.label,
    data:        r.history.map(h => ({ x: h.episode, y: h[key] })),
    borderColor: COLORS[i],
    backgroundColor: COLORS[i] + '22',
    tension:     0.3,
    pointRadius: 0,
    borderWidth: 2,
  }));
}

const chartConfigs = METRICS.map(m => ({
  title: m.label,
  key:   m.key,
  yMin:  m.yMin,
  yMax:  m.yMax,
  datasets: datasetsForMetric(m.key),
}));

// Parameters table: show keys that differ between conditions
const paramKeys = [
  'interClusterConnectionProb', 'intraClusterConnectionProb',
  'chemicalDiffusionRadius', 'chemicalFalloff',
  'ambientRadius', 'weightDecay', 'learningRate',
  'activityHistoryMinimum', 'targetFireRate', 'thresholdAdjustRate',
  '_skipDampening',
];

function paramRows() {
  return paramKeys.map(k => {
    const vals = results.map(r => r.config[k] === undefined ? '—' : String(r.config[k]));
    const allSame = vals.every(v => v === vals[0]);
    return { key: k, vals, allSame };
  });
}

const finalTable = results.map(r => ({
  label:         r.label,
  finalAccuracy: (r.finalAccuracy * 100).toFixed(1) + '%',
}));

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>mcfeedback — Experiment 001 Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; color: #fff; }
  .subtitle { font-size: 0.85rem; color: #888; margin-bottom: 28px; }
  h2 { font-size: 1rem; font-weight: 600; color: #ccc; margin: 32px 0 12px; border-bottom: 1px solid #2a2a2a; padding-bottom: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 20px; }
  .chart-card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px; padding: 16px; }
  .chart-title { font-size: 0.78rem; font-weight: 600; color: #aaa; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
  canvas { max-height: 220px; }
  table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
  th { background: #1e2130; color: #aaa; font-weight: 600; text-align: left; padding: 8px 12px; border-bottom: 2px solid #2a2d3a; }
  td { padding: 7px 12px; border-bottom: 1px solid #1e2130; color: #ccc; }
  tr:hover td { background: #1e2130; }
  .diff { color: #f28e2b; }
  .same { color: #555; }
  .good { color: #76b7b2; font-weight: 600; }
  .legend { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.82rem; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .params-note { font-size: 0.75rem; color: #555; margin-top: 6px; }
</style>
</head>
<body>
<h1>mcfeedback — Experiment 001: Pattern Association</h1>
<div class="subtitle">Murray-Claude Feedback Algorithm · Phase 1 · ${new Date().toISOString().slice(0,10)}</div>

<div class="legend">
${results.map((r,i) => `  <div class="legend-item"><div class="legend-dot" style="background:${COLORS[i]}"></div>${r.label}</div>`).join('\n')}
</div>

<div class="grid" id="charts"></div>

<h2>Final Accuracy</h2>
<table>
  <tr><th>Condition</th><th>Final Accuracy (avg over 4 patterns)</th></tr>
  ${finalTable.map(r => `<tr><td>${r.label}</td><td class="good">${r.finalAccuracy}</td></tr>`).join('\n  ')}
</table>

<h2>Parameters</h2>
<p class="params-note">Orange = differs between conditions · Grey = same across all</p>
<table>
  <tr><th>Parameter</th>${results.map(r => `<th>${r.label}</th>`).join('')}</tr>
  ${paramRows().map(row =>
    `<tr><td>${row.key}</td>${row.vals.map(v => `<td class="${row.allSame ? 'same' : 'diff'}">${v}</td>`).join('')}</tr>`
  ).join('\n  ')}
</table>

<h2>Base Config</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  ${Object.entries(BASE_CONFIG)
    .filter(([k]) => !['trainingEpisodes','reportInterval'].includes(k))
    .map(([k,v]) => `<tr><td>${k}</td><td class="same">${JSON.stringify(v)}</td></tr>`)
    .join('\n  ')}
</table>

<script>
const chartConfigs = ${JSON.stringify(chartConfigs, null, 2)};

const container = document.getElementById('charts');
chartConfigs.forEach(cfg => {
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.innerHTML = \`<div class="chart-title">\${cfg.title}</div><canvas></canvas>\`;
  container.appendChild(card);

  new Chart(card.querySelector('canvas'), {
    type: 'line',
    data: { datasets: cfg.datasets },
    options: {
      animation:   false,
      responsive:  true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => \` \${ctx.dataset.label}: \${ctx.parsed.y.toFixed(3)}\`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Episode', color: '#666', font: { size: 11 } },
          ticks: { color: '#555', maxTicksLimit: 8 },
          grid:  { color: '#1e2130' },
        },
        y: {
          min:   cfg.yMin  ?? undefined,
          max:   cfg.yMax  ?? undefined,
          ticks: { color: '#555', maxTicksLimit: 6 },
          grid:  { color: '#1e2130' },
        }
      }
    }
  });
});
</script>
</body>
</html>`;

const outPath = new URL('../results.html', import.meta.url).pathname;
writeFileSync(outPath, html, 'utf8');
process.stdout.write(`\nReport written to: ${outPath}\n`);
