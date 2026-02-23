import { writeFileSync } from 'node:fs';
import { createNetwork } from '../src/network.mjs';
import { step, evaluate } from '../src/engine.mjs';

// ─── Seedable PRNG (mulberry32) ───────────────────────────────────────────
function makePRNG(seed) {
  let s = seed >>> 0;
  return function () {
    s |= 0; s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function withSeed(seed, fn) {
  const orig = Math.random;
  Math.random = makePRNG(seed);
  try { return fn(); } finally { Math.random = orig; }
}

// ─── Config ───────────────────────────────────────────────────────────────
const BASE_CONFIG = {
  clustersCount: 2, neuronsPerCluster: 30, modulatoryPerCluster: 2,
  intraClusterConnectionProb: 0.6, interClusterConnectionProb: 0.5,
  clusterSpacing: 10.0, neuronSpread: 2.0,
  initialThreshold: 0.5, targetFireRate: 0.2, thresholdAdjustRate: 0.01,
  ambientRadius: 3.0, ambientFalloff: 'inverse',
  initialWeightRange: [-0.1, 0.1],
  coActivationStrength: 1.0, coSilenceStrength: 0.5, mismatchStrength: -0.5,
  ambientThreshold: 0.3, activityHistoryDecay: 0.95, activityHistoryMinimum: 0.1,
  chemicalDiffusionRadius: 15.0, chemicalFalloff: 'inverse', chemicalDecayRate: 0.5,
  positiveRewardStrength: 1.0, negativeRewardStrength: -1.0,
  learningRate: 0.01, maxWeightDelta: 0.1, maxWeightMagnitude: 2.0, weightDecay: 0.005,
  inputSize: 5, outputSize: 5, trainingEpisodes: 1000,
};

const PATTERNS = [
  { input: [1,0,1,0,1], target: [0,1,0,1,0], label: 'P1' },
  { input: [1,1,0,0,0], target: [0,0,1,1,1], label: 'P2' },
  { input: [1,0,0,0,1], target: [0,1,1,1,0], label: 'P3' },
  { input: [0,1,0,1,0], target: [1,0,1,0,1], label: 'P4' },
];

const CONDITIONS = [
  { label: 'Baseline',      color: '#4e79a7', overrides: { ambientRadius: 0, _skipDampening: true, chemicalDiffusionRadius: 1000, chemicalFalloff: 'constant' } },
  { label: 'Ambient only',  color: '#f28e2b', overrides: { _skipDampening: true, chemicalDiffusionRadius: 1000, chemicalFalloff: 'constant' } },
  { label: 'Dampening only',color: '#e15759', overrides: { ambientRadius: 0, chemicalDiffusionRadius: 1000, chemicalFalloff: 'constant' } },
  { label: 'Full model',    color: '#76b7b2', overrides: { chemicalDiffusionRadius: 15 } },
];

const SEEDS = [42, 137, 271, 314, 500, 618, 777, 888, 999, 1234];

// ─── Run ──────────────────────────────────────────────────────────────────
function runOne(overrides, seed) {
  return withSeed(seed, () => {
    const config  = { ...BASE_CONFIG, ...overrides };
    const network = createNetwork(config);
    for (let ep = 1; ep <= config.trainingEpisodes; ep++) {
      step(network, PATTERNS[(ep-1) % PATTERNS.length].input, PATTERNS[(ep-1) % PATTERNS.length].target);
    }
    let total = 0;
    const perPattern = PATTERNS.map(p => {
      const r = evaluate(network, p.input, p.target);
      total += r.accuracy;
      return r.accuracy;
    });
    return { mean: total / PATTERNS.length, perPattern };
  });
}

process.stdout.write('Running...\n');
// condResults[ci][si] = { mean, perPattern }
const condResults = CONDITIONS.map(() => []);
for (let si = 0; si < SEEDS.length; si++) {
  process.stdout.write('  seed ' + SEEDS[si] + ': ');
  for (let ci = 0; ci < CONDITIONS.length; ci++) {
    const r = runOne(CONDITIONS[ci].overrides, SEEDS[si]);
    condResults[ci].push(r);
    process.stdout.write(CONDITIONS[ci].label.split(' ')[0] + '=' + (r.mean*100).toFixed(0) + '%  ');
  }
  process.stdout.write('\n');
}

// ─── Stats ────────────────────────────────────────────────────────────────
function mean(a)  { return a.reduce((s,x) => s+x, 0) / a.length; }
function std(a)   { const m = mean(a); return Math.sqrt(a.reduce((s,x) => s+(x-m)**2, 0) / (a.length-1)); }

function pairedT(a, b) {
  const diffs = a.map((x,i) => x - b[i]);
  const n = diffs.length, m = mean(diffs), s = std(diffs);
  const t = m / (s / Math.sqrt(n)), df = n - 1;
  function lgamma(z) {
    const c = [0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,
      -176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI*z)) - lgamma(1-z);
    z -= 1; let x = c[0];
    for (let i = 1; i < 9; i++) x += c[i]/(z+i);
    const tt = z + 7.5;
    return 0.5*Math.log(2*Math.PI) + (z+0.5)*Math.log(tt) - tt + Math.log(x);
  }
  function betaInc(x, a, b) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const lbeta = lgamma(a+b) - lgamma(a) - lgamma(b);
    const front = Math.exp(Math.log(x)*a + Math.log(1-x)*b - lbeta) / a;
    let f=1,c=1,d=1-(a+b)*x/(a+1);
    if (Math.abs(d)<1e-30) d=1e-30; d=1/d; f=d;
    for (let m2=1; m2<=100; m2++) {
      let num=m2*(b-m2)*x/((a+2*m2-1)*(a+2*m2));
      d=1+num*d; if(Math.abs(d)<1e-30)d=1e-30; d=1/d; c=1+num/c; if(Math.abs(c)<1e-30)c=1e-30; f*=d*c;
      num=-(a+m2)*(a+b+m2)*x/((a+2*m2)*(a+2*m2+1));
      d=1+num*d; if(Math.abs(d)<1e-30)d=1e-30; d=1/d; c=1+num/c; if(Math.abs(c)<1e-30)c=1e-30;
      const delta=d*c; f*=delta; if(Math.abs(delta-1)<1e-10) break;
    }
    return front*f;
  }
  const p = betaInc(df/(df+t*t), df/2, 0.5);
  return { t: +t.toFixed(3), p: +p.toFixed(4), meanDiff: m };
}

const means       = condResults.map(r => mean(r.map(x => x.mean)));
const stds        = condResults.map(r => std(r.map(x => x.mean)));
const allMeans    = condResults.map(r => r.map(x => x.mean));
const tTests      = CONDITIONS.slice(1).map((_, i) => pairedT(allMeans[i+1], allMeans[0]));
const patMeans    = CONDITIONS.map((_, ci) =>
  PATTERNS.map((_, pi) => mean(condResults[ci].map(r => r.perPattern[pi])))
);

// ─── HTML ─────────────────────────────────────────────────────────────────
function sigLabel(p) {
  if (p < 0.01) return '** p&lt;0.01';
  if (p < 0.05) return '* p&lt;0.05';
  return 'ns';
}
function sigColor(p, isNeg) {
  if (p >= 0.05) return '#555';
  return isNeg ? '#e15759' : '#76b7b2';
}

// Strip + mean chart data
const stripData = CONDITIONS.map((cond, ci) => ({
  label:  cond.label,
  color:  cond.color,
  values: allMeans[ci].map(v => +(v*100).toFixed(1)),
  mean:   +(means[ci]*100).toFixed(1),
  std:    +(stds[ci]*100).toFixed(1),
}));

// Per-pattern grouped bar data
const patternChartData = {
  labels:   PATTERNS.map(p => p.label),
  datasets: CONDITIONS.map((cond, ci) => ({
    label:           cond.label,
    data:            patMeans[ci].map(v => +(v*100).toFixed(1)),
    backgroundColor: cond.color + 'cc',
    borderColor:     cond.color,
    borderWidth:     1,
  })),
};

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>mcfeedback — Multi-seed Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #d0d0d0; padding: 32px 28px; max-width: 1000px; margin: 0 auto; }
  h1 { font-size: 1.5rem; font-weight: 700; color: #fff; margin-bottom: 4px; }
  .subtitle { font-size: 0.85rem; color: #555; margin-bottom: 36px; }
  h2 { font-size: 0.95rem; font-weight: 700; color: #aaa; margin: 36px 0 14px; border-bottom: 1px solid #1e2130; padding-bottom: 7px; text-transform: uppercase; letter-spacing: 0.06em; }
  .card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px; }
  canvas { width: 100% !important; }
  table { border-collapse: collapse; width: 100%; font-size: 0.83rem; }
  th { background: #1e2130; color: #777; font-weight: 600; text-align: left; padding: 9px 14px; border-bottom: 2px solid #2a2d3a; }
  td { padding: 8px 14px; border-bottom: 1px solid #161920; }
  .meta { font-size: 0.78rem; color: #444; margin-top: 10px; }
  .verdict { border: 1px solid #2a2d3a; border-radius: 6px; padding: 18px 22px; margin-bottom: 20px; font-size: 0.9rem; line-height: 1.7; }
  .verdict strong { color: #fff; }
  .verdict.red    { border-color: #3a2222; background: #1a1215; }
  .verdict.yellow { border-color: #3a3020; background: #181510; }
  .legend { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 18px; }
  .leg { display: flex; align-items: center; gap: 7px; font-size: 0.82rem; color: #aaa; }
  .dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 680px) { .two-col { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<h1>mcfeedback — Multi-seed Experiment</h1>
<div class="subtitle">
  N = ${SEEDS.length} seeds &nbsp;·&nbsp;
  Seeds: ${SEEDS.join(', ')} &nbsp;·&nbsp;
  ${BASE_CONFIG.trainingEpisodes} training episodes &nbsp;·&nbsp;
  Frozen-weight evaluation &nbsp;·&nbsp;
  Random chance = 50%
</div>

<!-- Verdict -->
<div class="verdict red">
  <strong>Verdict: no reliable learning detected.</strong><br>
  No condition beats random chance (50%) consistently across seeds.
  Dampening only and Full model are statistically <em>significantly worse</em> than Baseline (p&lt;0.05, p&lt;0.01).
  The novel components are hurting, not helping, in the current parameter regime.
  Results are highly seed-dependent (Baseline std = ${(stds[0]*100).toFixed(1)}%), indicating
  the network is settling into different fixed-output biases rather than learning the associations.
</div>

<div class="legend">
  ${CONDITIONS.map(c => '<div class="leg"><div class="dot" style="background:' + c.color + '"></div>' + c.label + '</div>').join('')}
</div>

<h2>1 — Accuracy distribution across seeds</h2>
<div class="card">
  <canvas id="stripChart" height="320"></canvas>
  <p class="meta">Each dot = one seed. Horizontal line = mean. Dashed line = 50% random chance.</p>
</div>

<h2>2 — Mean ± 1 std</h2>
<div class="card">
  <canvas id="barChart" height="260"></canvas>
  <p class="meta">Error bars show ±1 standard deviation across ${SEEDS.length} seeds.</p>
</div>

<div class="two-col">
  <div>
    <h2>3 — Paired t-tests vs Baseline</h2>
    <div class="card" style="padding:0">
    <table>
      <tr><th>Comparison</th><th>Mean diff</th><th>t</th><th>p</th><th>Result</th></tr>
      ${CONDITIONS.slice(1).map((cond, i) => {
        const tt = tTests[i];
        const isNeg = tt.meanDiff < 0;
        const color = sigColor(tt.p, isNeg);
        return '<tr>' +
          '<td>' + cond.label + ' vs Baseline</td>' +
          '<td style="color:' + (isNeg ? '#e15759' : '#76b7b2') + '">' +
            (isNeg ? '' : '+') + (tt.meanDiff*100).toFixed(1) + '%</td>' +
          '<td style="color:#aaa">' + tt.t + '</td>' +
          '<td style="color:#aaa">' + tt.p + '</td>' +
          '<td style="color:' + color + ';font-weight:600">' + sigLabel(tt.p) + '</td>' +
        '</tr>';
      }).join('')}
    </table>
    </div>
    <p class="meta" style="margin-top:8px">Two-tailed paired t-test, df=9. ** p&lt;0.01 &nbsp; * p&lt;0.05 &nbsp; ns = not significant.</p>
  </div>

  <div>
    <h2>4 — Raw data (all seeds)</h2>
    <div class="card" style="padding:0">
    <table>
      <tr><th>Seed</th>${CONDITIONS.map(c => '<th>' + c.label.split(' ')[0] + '</th>').join('')}</tr>
      ${SEEDS.map((seed, si) =>
        '<tr><td style="color:#555">' + seed + '</td>' +
        CONDITIONS.map((_, ci) => {
          const v = allMeans[ci][si];
          const color = v > 0.55 ? '#76b7b2' : v < 0.45 ? '#e15759' : '#888';
          return '<td style="color:' + color + '">' + (v*100).toFixed(0) + '%</td>';
        }).join('') +
        '</tr>'
      ).join('')}
      <tr style="border-top:2px solid #2a2d3a">
        <td style="color:#aaa;font-weight:600">Mean</td>
        ${CONDITIONS.map((_, ci) => '<td style="color:#fff;font-weight:600">' + (means[ci]*100).toFixed(1) + '%</td>').join('')}
      </tr>
      <tr>
        <td style="color:#555">Std</td>
        ${CONDITIONS.map((_, ci) => '<td style="color:#555">±' + (stds[ci]*100).toFixed(1) + '%</td>').join('')}
      </tr>
    </table>
    </div>
  </div>
</div>

<h2>5 — Per-pattern accuracy (mean across seeds)</h2>
<div class="card">
  <canvas id="patternChart" height="260"></canvas>
  <p class="meta">
    If the network learned the associations, each condition would show consistent accuracy per pattern.
    Uneven bars indicate the network is outputting a fixed bias rather than responding to input.
  </p>
</div>

<div class="verdict yellow" style="margin-top:28px">
  <strong>What the per-pattern data shows:</strong><br>
  Each condition does well on different patterns — there is no single pattern all conditions agree on.
  This is the signature of a <em>fixed output bias</em>: the network picks one output vector and sticks with it,
  which happens to match some patterns better than others depending on the random seed.
  A genuinely learning network would show uniform or systematically high accuracy across all patterns.
</div>

<script>
const COLORS = ${JSON.stringify(CONDITIONS.map(c => c.color))};
const LABELS = ${JSON.stringify(CONDITIONS.map(c => c.label))};
const stripData = ${JSON.stringify(stripData)};
const patternChartData = ${JSON.stringify(patternChartData)};

// ── Strip chart (custom via scatter) ─────────────────────────────────────
const stripCtx = document.getElementById('stripChart').getContext('2d');

// One scatter dataset per condition (the dots)
const scatterSets = stripData.map((d, ci) => ({
  type: 'scatter',
  label: d.label,
  data: d.values.map((v, si) => ({ x: ci + 1 + (Math.random()-0.5)*0.18, y: v })),
  backgroundColor: COLORS[ci] + 'bb',
  borderColor:     COLORS[ci],
  borderWidth: 1,
  pointRadius: 6,
  pointHoverRadius: 8,
  showLine: false,
}));

// Mean line segments as scatter with showLine
const meanSets = stripData.map((d, ci) => ({
  type: 'scatter',
  label: '_mean_' + ci,
  data: [{ x: ci + 0.7, y: d.mean }, { x: ci + 1.3, y: d.mean }],
  showLine: true,
  borderColor: COLORS[ci],
  borderWidth: 3,
  pointRadius: 0,
  fill: false,
}));

new Chart(stripCtx, {
  type: 'scatter',
  data: { datasets: [...scatterSets, ...meanSets] },
  options: {
    animation: false,
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        filter: item => !item.dataset.label.startsWith('_'),
        callbacks: {
          label: ctx => ' ' + ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%'
        }
      },
      annotation: {
        annotations: {
          chance: {
            type: 'line', yMin: 50, yMax: 50,
            borderColor: '#444', borderWidth: 1, borderDash: [6, 4],
            label: { content: '50% chance', display: true, color: '#555', font: { size: 10 }, position: 'end', yAdjust: -8 }
          }
        }
      }
    },
    scales: {
      x: {
        min: 0.3, max: ${CONDITIONS.length} + 0.7,
        ticks: {
          color: '#aaa',
          callback: (v) => {
            const idx = Math.round(v) - 1;
            return idx >= 0 && idx < LABELS.length ? LABELS[idx] : '';
          },
          stepSize: 1,
          maxTicksLimit: 6,
        },
        grid: { color: '#1a1d27' },
      },
      y: {
        min: 25, max: 80,
        title: { display: true, text: 'Accuracy (%)', color: '#555', font: { size: 11 } },
        ticks: { color: '#555', callback: v => v + '%' },
        grid: { color: '#1e2130' },
      }
    }
  }
});

// ── Bar chart with error bars ──────────────────────────────────────────────
// Chart.js doesn't have built-in error bars; draw them as a custom plugin.
const barCtx = document.getElementById('barChart').getContext('2d');
const barPlugin = {
  id: 'errorbars',
  afterDatasetDraw(chart) {
    const { ctx, scales: { x, y } } = chart;
    chart.data.datasets.forEach((ds, di) => {
      if (!ds._stds) return;
      const meta = chart.getDatasetMeta(di);
      meta.data.forEach((bar, i) => {
        const std = ds._stds[i];
        if (std == null) return;
        const cx = bar.x, cy = bar.y;
        const top    = y.getPixelForValue(ds.data[i] + std);
        const bottom = y.getPixelForValue(ds.data[i] - std);
        const w = 6;
        ctx.save();
        ctx.strokeStyle = ds.borderColor[i] || ds.borderColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, top); ctx.lineTo(cx, bottom);
        ctx.moveTo(cx-w, top); ctx.lineTo(cx+w, top);
        ctx.moveTo(cx-w, bottom); ctx.lineTo(cx+w, bottom);
        ctx.stroke();
        ctx.restore();
      });
    });
  }
};

new Chart(barCtx, {
  type: 'bar',
  plugins: [barPlugin],
  data: {
    labels: LABELS,
    datasets: [{
      label: 'Mean accuracy',
      data: stripData.map(d => d.mean),
      _stds: stripData.map(d => d.std),
      backgroundColor: COLORS.map(c => c + '88'),
      borderColor: COLORS,
      borderWidth: 2,
      borderRadius: 4,
    }]
  },
  options: {
    animation: false,
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => ' Mean: ' + ctx.parsed.y.toFixed(1) + '%' } },
    },
    scales: {
      x: { ticks: { color: '#aaa' }, grid: { color: '#1a1d27' } },
      y: {
        min: 30, max: 70,
        title: { display: true, text: 'Mean accuracy (%)', color: '#555', font: { size: 11 } },
        ticks: { color: '#555', callback: v => v + '%' },
        grid: { color: '#1e2130' },
      }
    }
  }
});

// ── Per-pattern grouped bar ────────────────────────────────────────────────
new Chart(document.getElementById('patternChart'), {
  type: 'bar',
  data: patternChartData,
  options: {
    animation: false,
    responsive: true,
    plugins: {
      legend: {
        labels: { color: '#aaa', font: { size: 11 }, boxWidth: 12 }
      },
      tooltip: { callbacks: { label: ctx => ' ' + ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%' } }
    },
    scales: {
      x: { ticks: { color: '#aaa' }, grid: { color: '#1a1d27' } },
      y: {
        min: 30, max: 75,
        title: { display: true, text: 'Mean accuracy (%)', color: '#555', font: { size: 11 } },
        ticks: { color: '#555', callback: v => v + '%' },
        grid: { color: '#1e2130' },
      }
    }
  }
});
</script>
</body>
</html>`;

const outPath = new URL('../multiseed-results.html', import.meta.url).pathname;
writeFileSync(outPath, html, 'utf8');
process.stdout.write('\nReport written to: ' + outPath + '\n');
