/* Multi-Model Observatory — dashboard logic */

const $ = (sel) => document.querySelector(sel);

const queryInput = $('#queryInput');
const runBtn = $('#runBtn');
const reportBtn = $('#reportBtn');
const routingToggle = $('#routingToggle');
const refreshRoutingBtn = $('#refreshRoutingBtn');
const forceReexploreAllBtn = $('#forceReexploreAllBtn');
const seedBootstrapBtn = $('#seedBootstrapBtn');
const seedVerifyBtn = $('#seedVerifyBtn');
const seedStatusEl = $('#seedStatus');
const statusEl = $('#status');
const clientIdInput = $('#clientIdInput');
const reqTone = $('#reqTone');
const reqDetail = $('#reqDetail');
const reqInclude = $('#reqInclude');
const reqAvoid = $('#reqAvoid');
const reqFormat = $('#reqFormat');
const reqDomain = $('#reqDomain');
const saveRequirementsBtn = $('#saveRequirementsBtn');
const loadRequirementsBtn = $('#loadRequirementsBtn');
const requirementsStatusEl = $('#requirementsStatus');
const selfCorrectionToggle = $('#selfCorrectionToggle');
const selfCorrectionSection = $('#selfCorrectionSection');
const scSummary = $('#scSummary');
const scAttempts = $('#scAttempts');
const scOutcomeBadge = $('#scOutcomeBadge');
const modeBadge = $('#modeBadge');
const reportSummary = $('#reportSummary');
const clusterGrid = $('#clusterGrid');

const emptyState = $('#emptyState');
const resultsZone = $('#resultsZone');
const reportModal = $('#reportModal');
const reportModalBody = $('#reportModalBody');
const reportModalBackdrop = $('#reportModalBackdrop');
const closeReportModalBtn = $('#closeReportModalBtn');
const routingSection = $('#routingSection');
const winnerSection = $('#winnerSection');
const metricsSection = $('#metricsSection');
const responsesSection = $('#responsesSection');

const routingFlow = $('#routingFlow');
const routingDetails = $('#routingDetails');
const routingTimestamp = $('#routingTimestamp');
const bestResponse = $('#bestResponse');
const winnerModel = $('#winnerModel');
const winnerPrompt = $('#winnerPrompt');
const winnerDomain = $('#winnerDomain');
const kpiGrid = $('#kpiGrid');
const blueChart = $('#blueChart');
const modelGroups = $('#modelGroups');
const responseCount = $('#responseCount');

let currentLogId = null;
let lastResult = null;
let lastResultMode = null; // 'standard' | 'self_correct' | null
let routingStatus = null;

const MODEL_META = {
  organic_model: { label: 'Organic Chemistry', abbr: 'ORG', color: '#5b8def' },
  python_model: { label: 'Python FAQ', abbr: 'PY', color: '#22d3b8' },
  gita_model: { label: 'Bhagavad Gita', abbr: 'GITA', color: '#c084fc' },
  groq_model: { label: 'Groq (General)', abbr: 'LLM', color: '#f0b429' },
};

const REASON_LABELS = {
  confident_cluster_direct_route: 'Cluster is confident — routed to best model',
  insufficient_cluster_samples: 'Not enough samples — full exploration',
  re_exploration_triggered: 'Re-exploration triggered',
  cluster_not_in_direct_mode: 'Cluster not in direct mode yet',
  routing_disabled_by_user: 'Adaptive routing disabled — all models called',
};

function fmt(n, digits = 2) {
  if (n == null || Number.isNaN(n)) return '—';
  return typeof n === 'number' ? n.toFixed(digits) : String(n);
}

function pct(n) {
  if (n == null) return '—';
  return `${(n * 100).toFixed(1)}%`;
}

function setStatus(msg, type = '') {
  statusEl.textContent = msg;
  statusEl.className = `status ${type}`;
}

function showResults(show, isSelfCorrect = false) {
  emptyState.classList.toggle('hidden', show);
  resultsZone.classList.toggle('hidden', !show);
  selfCorrectionSection.classList.toggle('hidden', !show || !isSelfCorrect);
  routingSection.classList.toggle('hidden', !show || isSelfCorrect);
  winnerSection.classList.toggle('hidden', !show);
  metricsSection.classList.toggle('hidden', !show || isSelfCorrect);
  responsesSection.classList.toggle('hidden', !show || isSelfCorrect);
  if (show) {
    requestAnimationFrame(() => {
      resultsZone.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
}

function updateModeBadge(routing, routingEnabled) {
  const compact = 'mode-badge mode-badge-compact';
  if (!routingEnabled) {
    modeBadge.textContent = 'Routing OFF — all 4 models × 3 prompts';
    modeBadge.className = `${compact} mode-off`;
    return;
  }
  if (routing?.mode === 'direct') {
    modeBadge.textContent = `Direct route → ${routing.cluster_best_model || routing.models?.[0]}`;
    modeBadge.className = `${compact} mode-direct`;
  } else {
    modeBadge.textContent = 'Full exploration — 4 models × 3 prompts';
    modeBadge.className = `${compact} mode-explore`;
  }
}

function renderRoutingFlow(data) {
  const r = data.routing || {};
  const analysis = data.analysis || {};
  const routingEnabled = r.routing_enabled !== false;
  const allModels = ['organic_model', 'python_model', 'gita_model', 'groq_model'];
  const called = new Set(r.models || []);
  const isDirect = r.mode === 'direct' && routingEnabled;

  const steps = [
    {
      label: 'Query',
      value: (data.query || queryInput.value.trim()).slice(0, 40) + (queryInput.value.length > 40 ? '…' : ''),
      sub: analysis.domain ? `Domain: ${analysis.domain}` : '',
      active: true,
    },
    {
      label: 'Classify',
      value: r.cluster_id || '—',
      sub: r.classification_method?.replace(/_/g, ' ') || '',
      active: true,
    },
    {
      label: 'Decision',
      value: isDirect ? 'Direct' : 'Explore all',
      sub: REASON_LABELS[r.reason] || r.reason || '',
      active: true,
    },
    {
      label: 'Models called',
      value: `${called.size} / 4`,
      sub: [...called].map((m) => MODEL_META[m]?.abbr || m).join(', '),
      active: called.size > 0,
    },
  ];

  routingFlow.innerHTML = steps
    .map(
      (s) => `
    <div class="flow-step">
      <div class="flow-node ${s.active ? 'active' : 'skipped'}">
        <span class="flow-node-label">${s.label}</span>
        <span class="flow-node-value">${s.value}</span>
        <span class="flow-node-sub">${s.sub}</span>
      </div>
    </div>`
    )
    .join('');

  const sims = r.classification_similarities || {};
  const simBars = Object.entries(sims)
    .sort((a, b) => b[1] - a[1])
    .map(
      ([id, score]) => `
    <div class="sim-bar-row">
      <span class="sim-bar-label">${id}</span>
      <div class="sim-bar-track"><div class="sim-bar-fill" style="width:${Math.max(0, Math.min(100, score * 100))}%"></div></div>
      <span class="sim-bar-val">${fmt(score, 2)}</span>
    </div>`
    )
    .join('');

  const reasons = (r.reexploration_reasons || [])
    .map((reason) => `<span class="reason-tag">${reason}</span>`)
    .join('');

  routingDetails.innerHTML = `
    <dl class="detail-card"><dt>Cluster</dt><dd>${r.cluster_id || '—'}</dd></dl>
    <dl class="detail-card"><dt>Classification confidence</dt><dd>${pct(r.classification_confidence)}</dd></dl>
    <dl class="detail-card"><dt>Cluster samples</dt><dd>${r.cluster_count ?? 0} / ${routingStatus?.min_samples_for_direct ?? 20} for direct</dd></dl>
    <dl class="detail-card"><dt>Best model (cluster)</dt><dd>${r.cluster_best_model || '—'}</dd></dl>
    <dl class="detail-card"><dt>Routing enabled</dt><dd>${routingEnabled ? 'Yes' : 'No (toggle off)'}</dd></dl>
    <dl class="detail-card"><dt>Ready for direct</dt><dd>${r.ready_for_direct ? 'Yes' : 'No'}</dd></dl>
    ${
      simBars
        ? `<div class="detail-card" style="grid-column:1/-1"><dt>Embedding similarities</dt><div class="sim-bar-wrap">${simBars || '<span class="muted">No centroids yet</span>'}</div></div>`
        : ''
    }
    ${reasons ? `<div class="detail-card" style="grid-column:1/-1"><dt>Re-exploration triggers</dt><dd class="reason-list">${reasons}</dd></div>` : ''}
  `;

  routingTimestamp.textContent = new Date().toLocaleString();
}

function renderWinner(data) {
  winnerModel.textContent = MODEL_META[data.best_model]?.label || data.best_model;
  winnerPrompt.textContent = `Prompt ${data.best_prompt}`;
  winnerDomain.textContent = data.analysis?.domain || '';
  bestResponse.textContent = data.best_answer || 'No answer';
}

function renderMetricsKpis(metrics, blue) {
  const m = metrics || {};
  const sem = m.semantic || {};
  const kpis = [
    { label: 'Accuracy', value: pct(m.accuracyScore) },
    { label: 'Confidence', value: pct(m.confidenceScore) },
    { label: 'Hallucination', value: pct(m.hallucinationScore) },
    { label: 'Semantic acc', value: sem.accuracy != null ? pct(sem.accuracy) : '—' },
    { label: 'Relevance', value: pct(m.relevanceScore) },
    { label: 'Latency', value: `${fmt(m.latency, 0)} ms` },
    { label: 'Cost', value: `$${fmt(m.cost, 6)}` },
    { label: 'Tokens', value: m.tokenUsage ?? '—' },
  ];

  let html = `<div class="report-kpi-grid">${kpis
    .map((k) => `<div class="report-kpi"><div class="report-kpi-value">${k.value}</div><div class="report-kpi-label">${k.label}</div></div>`)
    .join('')}</div>`;

  if (blue && Object.keys(blue).length) {
    html += `<table class="report-table"><thead><tr><th>BLUE metric</th><th>Value</th></tr></thead><tbody>
      <tr><td>Behavior stability</td><td>${fmt(blue.behaviorStability, 4)}</td></tr>
      <tr><td>Avg latency</td><td>${fmt(blue.latency, 0)} ms</td></tr>
      <tr><td>Usage cost</td><td>$${fmt(blue.usageCost, 6)}</td></tr>
      <tr><td>Error rate</td><td>${fmt(blue.errorRate, 4)}</td></tr>
    </tbody></table>`;
  }
  return html;
}

function renderPlatformReportBlock(data) {
  const domains = data.domain_stats || {};
  const domainRows = Object.keys(domains)
    .sort()
    .map((d) => {
      const s = domains[d];
      return `<tr>
        <td>${escapeHtml(d)}</td>
        <td>${escapeHtml(s.best_model || '—')}</td>
        <td>${pct(s.avg_accuracy)}</td>
        <td>${fmt(s.avg_latency, 0)} ms</td>
        <td>${s.count ?? 0}</td>
      </tr>`;
    })
    .join('');

  return `
    <div class="report-block">
      <h3>Platform learning (all queries)</h3>
      <div class="report-kpi-grid">
        <div class="report-kpi"><div class="report-kpi-value">${data.total_queries ?? 0}</div><div class="report-kpi-label">Total queries</div></div>
        <div class="report-kpi"><div class="report-kpi-value">${fmt(data.average_latency, 0)} ms</div><div class="report-kpi-label">Avg latency</div></div>
        <div class="report-kpi"><div class="report-kpi-value">${Object.keys(domains).length}</div><div class="report-kpi-label">Domains</div></div>
        <div class="report-kpi"><div class="report-kpi-value">${Object.values(domains).reduce((n, d) => n + (d.count || 0), 0)}</div><div class="report-kpi-label">Domain samples</div></div>
      </div>
      ${
        domainRows
          ? `<table class="report-table"><thead><tr><th>Domain</th><th>Best model</th><th>Avg accuracy</th><th>Avg latency</th><th>Queries</th></tr></thead><tbody>${domainRows}</tbody></table>`
          : '<p class="report-empty">No domain learning data yet — run a standard query first.</p>'
      }
    </div>`;
}

function renderLastQueryReportBlock() {
  if (!lastResult) {
    return `
      <div class="report-block">
        <h3>Last query</h3>
        <p class="report-empty">No query run yet in this session. Run a query, then click Report again.</p>
      </div>`;
  }

  const query = escapeHtml(lastResult.query || queryInput.value.trim() || '—');

  if (lastResultMode === 'self_correct') {
    const attempts = lastResult.attempts || [];
    const rows = attempts
      .map((a) => {
        const j = a.judge || {};
        const m = a.metrics || {};
        const sem = m.semantic || {};
        return `<tr>
          <td>${a.attempt}</td>
          <td>${escapeHtml(a.model || '—')}</td>
          <td>${j.pass ? 'PASS' : 'FAIL'}</td>
          <td>${fmt(j.score, 2)}</td>
          <td>${pct(m.accuracyScore)}</td>
          <td>${sem.accuracy != null ? pct(sem.accuracy) : '—'}</td>
          <td>${fmt(m.latency, 0)} ms</td>
          <td>${escapeHtml(a.selection_mode || '')}</td>
        </tr>`;
      })
      .join('');

    const finalM = attempts.find((a) => a.model === lastResult.final_model && a.response === lastResult.final_answer)?.metrics
      || attempts.reduce((best, a) => ((a.judge?.score || 0) > (best?.judge?.score || 0) ? a : best), attempts[0])?.metrics;

    return `
      <div class="report-block">
        <h3>Last query — self-correction</h3>
        <p class="muted" style="margin-bottom:12px;font-size:0.85rem"><strong>Query:</strong> ${query}</p>
        <div class="report-kpi-grid">
          <div class="report-kpi"><div class="report-kpi-value">${lastResult.requirements_met ? 'PASS' : 'FAIL'}</div><div class="report-kpi-label">Requirements</div></div>
          <div class="report-kpi"><div class="report-kpi-value">${fmt(lastResult.final_judge?.score, 2)}</div><div class="report-kpi-label">Judge score</div></div>
          <div class="report-kpi"><div class="report-kpi-value">${lastResult.passed_on_attempt ?? '—'}</div><div class="report-kpi-label">Passed attempt</div></div>
          <div class="report-kpi"><div class="report-kpi-value">${escapeHtml(lastResult.final_model || '—')}</div><div class="report-kpi-label">Final model</div></div>
        </div>
        ${finalM ? renderMetricsKpis(finalM) : ''}
        ${
          rows
            ? `<table class="report-table"><thead><tr><th>#</th><th>Model</th><th>Judge</th><th>Score</th><th>Accuracy</th><th>Semantic</th><th>Latency</th><th>Mode</th></tr></thead><tbody>${rows}</tbody></table>`
            : ''
        }
      </div>`;
  }

  const rows = (lastResult.all_responses || [])
    .map((r) => {
      const m = r.metrics || {};
      const sem = m.semantic || {};
      return `<tr>
        <td>${escapeHtml(r.model || '—')}</td>
        <td>${escapeHtml(r.prompt || '—')}</td>
        <td>${pct(m.accuracyScore)}</td>
        <td>${pct(m.confidenceScore)}</td>
        <td>${sem.accuracy != null ? pct(sem.accuracy) : '—'}</td>
        <td>${fmt(m.latency, 0)} ms</td>
        <td>$${fmt(m.cost, 6)}</td>
      </tr>`;
    })
    .join('');

  return `
    <div class="report-block">
      <h3>Last query — standard pipeline</h3>
      <p class="muted" style="margin-bottom:12px;font-size:0.85rem"><strong>Query:</strong> ${query} · <strong>Winner:</strong> ${escapeHtml(lastResult.best_model || '—')}</p>
      ${renderMetricsKpis(lastResult.metrics, lastResult.blue_metrics)}
      ${
        rows
          ? `<table class="report-table"><thead><tr><th>Model</th><th>Prompt</th><th>Accuracy</th><th>Confidence</th><th>Semantic</th><th>Latency</th><th>Cost</th></tr></thead><tbody>${rows}</tbody></table>`
          : '<p class="report-empty">No per-response metrics in last result.</p>'
      }
    </div>`;
}

function openReportModal() {
  reportModal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeReportModal() {
  reportModal.classList.add('hidden');
  document.body.style.overflow = '';
}

async function showReportModal() {
  reportModalBody.innerHTML = '<p class="muted">Loading metrics…</p>';
  openReportModal();

  try {
    const [reportRes] = await Promise.all([
      fetch('/report'),
      loadReport(),
      loadRoutingStatus(),
      loadSeedStatus(),
    ]);
    const reportData = await reportRes.json();
    reportModalBody.innerHTML = renderPlatformReportBlock(reportData) + renderLastQueryReportBlock();
    setStatus('Metrics report opened', 'success');
  } catch (err) {
    reportModalBody.innerHTML = `<p class="report-empty">Failed to load report: ${escapeHtml(err.message)}</p>`;
    setStatus(`Report error: ${err.message}`, 'error');
  }
}

function renderMetrics(data) {
  const m = data.metrics || {};
  const sem = m.semantic || {};
  const blue = data.blue_metrics || {};

  const kpis = [
    { label: 'Accuracy', value: pct(m.accuracyScore) },
    { label: 'Confidence', value: pct(m.confidenceScore) },
    { label: 'Hallucination', value: pct(m.hallucinationScore) },
    { label: 'Semantic acc', value: sem.accuracy != null ? pct(sem.accuracy) : '—' },
    { label: 'Latency', value: `${fmt(m.latency, 0)} ms` },
    { label: 'Cost', value: `$${fmt(m.cost, 4)}` },
  ];

  kpiGrid.innerHTML = kpis
    .map((k) => `<div class="kpi"><div class="kpi-value">${k.value}</div><div class="kpi-label">${k.label}</div></div>`)
    .join('');

  const bars = [
    { label: 'Stability', value: blue.behaviorStability, max: 1 },
    { label: 'Avg latency', value: blue.latency, max: Math.max(blue.latency, 500) },
    { label: 'Usage cost', value: blue.usageCost, max: Math.max(blue.usageCost, 0.01) },
    { label: 'Error rate', value: blue.errorRate, max: 1 },
  ];

  blueChart.innerHTML = bars
    .map((b) => {
      const width = b.max ? Math.min(100, (b.value / b.max) * 100) : 0;
      return `
      <div class="bar-row">
        <span class="bar-label">${b.label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
        <span class="bar-val">${typeof b.value === 'number' ? fmt(b.value, 4) : b.value}</span>
      </div>`;
    })
    .join('');
}

function renderResponses(data) {
  const rows = data.all_responses || [];
  responseCount.textContent = `${rows.length} responses (${new Set(rows.map((r) => r.model)).size} models)`;

  const byModel = {};
  for (const row of rows) {
    if (!byModel[row.model]) byModel[row.model] = [];
    byModel[row.model].push(row);
  }

  modelGroups.innerHTML = Object.entries(byModel)
    .map(([model, items]) => {
      const meta = MODEL_META[model] || { label: model, abbr: '?', color: '#888' };
      const avgAcc =
        items.reduce((s, i) => s + (i.metrics?.accuracyScore || 0), 0) / items.length;
      const wasCalled = (data.routing?.models || []).includes(model);

      return `
      <div class="model-group open" data-model="${model}">
        <div class="model-group-header" onclick="this.parentElement.classList.toggle('open')">
          <div class="model-group-name">
            <span class="model-icon" style="background:${meta.color}22;color:${meta.color}">${meta.abbr}</span>
            ${meta.label}
            ${!wasCalled ? '<span class="chip" style="opacity:0.6">skipped</span>' : ''}
          </div>
          <div class="model-group-stats">
            ${items.length} prompts · avg acc ${pct(avgAcc)}
            <span class="chevron">▼</span>
          </div>
        </div>
        <div class="model-group-body">
          ${items.map((row) => renderResponseCard(row, data)).join('')}
        </div>
      </div>`;
    })
    .join('');
}

function renderResponseCard(row, data) {
  const sem = row.metrics?.semantic || {};
  const isWinner = row.model === data.best_model && row.prompt_version === data.best_prompt;

  return `
  <div class="response-card ${isWinner ? 'winner' : ''}" data-response-id="${row.response_id || ''}">
    <div class="response-header">
      <span class="chip ${isWinner ? 'chip-gold' : ''}">${row.prompt_version}</span>
      ${isWinner ? '<span class="chip chip-gold">Winner</span>' : ''}
      <span class="chip chip-muted">${fmt(row.metrics?.latency, 0)} ms</span>
    </div>
    ${row.error ? `<div class="error-text">Error: ${row.error}</div>` : ''}
    <div class="prompt-block"><strong>Prompt sent:</strong>\n${escapeHtml(row.prompt || '')}</div>
    <table class="metrics-table">
      <thead><tr>
        <th>Metric</th><th>Lexical</th><th>Semantic</th>
      </tr></thead>
      <tbody>
        <tr><td>Accuracy</td><td>${fmt(row.metrics?.accuracyScore, 4)}</td><td>${sem.accuracy != null ? fmt(sem.accuracy, 4) : '—'}</td></tr>
        <tr><td>Confidence</td><td>${fmt(row.metrics?.confidenceScore, 4)}</td><td>${sem.confidence != null ? fmt(sem.confidence, 4) : '—'}</td></tr>
        <tr><td>Relevance</td><td>${fmt(row.metrics?.relevanceScore, 4)}</td><td>${sem.relevance != null ? fmt(sem.relevance, 4) : '—'}</td></tr>
        <tr><td>Hallucination</td><td>${fmt(row.metrics?.hallucinationScore, 4)}</td><td>${sem.hallucination != null ? fmt(sem.hallucination, 4) : '—'}</td></tr>
        <tr><td>Cost</td><td colspan="2">$${fmt(row.metrics?.cost, 6)} · ${row.metrics?.tokenUsage ?? 0} tokens</td></tr>
      </tbody>
    </table>
    <div class="response-text">${escapeHtml(row.response || '(empty)')}</div>
    <div class="followup-box">
      <label>Follow-up on this response</label>
      <div class="followup-row">
        <textarea class="followup-input" rows="2" placeholder="Ask a follow-up…"></textarea>
        <button class="btn btn-secondary btn-sm followup-btn">Ask</button>
      </div>
      <div class="followup-answer" hidden></div>
    </div>
  </div>`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function bindFollowups() {
  const rowById = Object.fromEntries((lastResult?.all_responses || []).map((r) => [r.response_id, r]));

  modelGroups.querySelectorAll('.response-card').forEach((card) => {
    const row = rowById[card.dataset.responseId];
    const btn = card.querySelector('.followup-btn');
    const input = card.querySelector('.followup-input');
    const output = card.querySelector('.followup-answer');
    if (!btn || !row) return;

    btn.onclick = async () => {
      const q = input.value.trim();
      if (!q) {
        output.hidden = false;
        output.textContent = 'Enter a follow-up question first.';
        return;
      }
      btn.disabled = true;
      output.hidden = false;
      output.textContent = 'Running follow-up…';

      try {
        const res = await fetch('/followup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            followup_question: q,
            target_reference: `${row.model} ${row.prompt_version} ${row.response || ''}`,
            parent_log_id: currentLogId,
            parent_response_id: row.response_id,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Follow-up failed');
        output.textContent = `[${data.matched_model} · ${data.matched_prompt} · match ${fmt(data.match_score, 2)}]\n\n${data.answer}`;
        await loadReport();
      } catch (err) {
        output.textContent = `Error: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    };
  });
}

function renderClusters(status) {
  if (!status?.clusters) {
    clusterGrid.innerHTML = '<span class="muted">No routing data</span>';
    return;
  }

  const minSamples = status.min_samples_for_direct || 20;
  const entries = Object.entries(status.clusters).sort((a, b) => a[0].localeCompare(b[0]));

  clusterGrid.innerHTML = entries
    .map(([id, c]) => {
      const ready = (c.count || 0) >= minSamples && c.mode === 'direct';
      const modeClass = c.mode === 'direct' ? 'direct' : 'explore';
      return `
      <div class="cluster-chip ${modeClass}">
        <div class="cluster-chip-id">${id}${ready ? ' ✓' : ''}</div>
        <div class="cluster-chip-meta">${c.mode || 'explore'} · ${c.count || 0}/${minSamples}</div>
        <div class="cluster-chip-meta">${c.best_model || '—'}</div>
        <button class="btn btn-ghost btn-sm" style="margin-top:6px;padding:2px 6px;font-size:0.65rem" data-force-cluster="${id}">Re-explore</button>
      </div>`;
    })
    .join('');

  clusterGrid.querySelectorAll('[data-force-cluster]').forEach((btn) => {
    btn.onclick = async () => {
      const clusterId = btn.dataset.forceCluster;
      btn.disabled = true;
      try {
        await fetch('/routing/re-explore', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ cluster_id: clusterId }),
        });
        setStatus(`Forced re-exploration for ${clusterId}`, 'success');
        await loadRoutingStatus();
      } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
      } finally {
        btn.disabled = false;
      }
    };
  });
}

async function loadSeedStatus() {
  try {
    const res = await fetch('/routing/seed/status');
    const data = await res.json();
    const min = data.min_samples_for_direct || 20;
    const readyCount = Object.values(data.clusters || {}).filter((c) => c.ready_for_direct).length;
    const total = Object.keys(data.clusters || {}).length;
    seedStatusEl.innerHTML = `
      <div class="stat-row"><span>Threshold</span><span>${min} samples</span></div>
      <div class="stat-row"><span>Ready</span><span class="${data.all_ready ? 'ready' : 'pending'}">${readyCount}/${total}</span></div>
    `;
  } catch (err) {
    seedStatusEl.textContent = `Failed: ${err.message}`;
  }
}

async function loadRoutingStatus() {
  try {
    const res = await fetch('/routing/status');
    routingStatus = await res.json();
    renderClusters(routingStatus);
  } catch (err) {
    clusterGrid.innerHTML = `<p class="muted">Failed to load routing: ${err.message}</p>`;
  }
}

async function loadReport() {
  try {
    const res = await fetch('/report');
    const data = await res.json();
    const domains = data.best_model_per_domain || {};
    const acc = data.accuracy_comparison || {};
    const rows = Object.keys(domains)
      .map(
        (d) => `
      <div class="stat-row">
        <span>${d}</span>
        <span>${domains[d]} · ${pct(acc[d])}</span>
      </div>`
      )
      .join('');
    reportSummary.innerHTML = `
      <div class="stat-row"><span>Queries</span><span>${data.total_queries ?? 0}</span></div>
      <div class="stat-row"><span>Latency</span><span>${fmt(data.average_latency, 0)} ms</span></div>
      ${rows || ''}
    `;
  } catch (err) {
    reportSummary.textContent = `Failed: ${err.message}`;
  }
}

function getRequirementsPayload() {
  return {
    tone_style: reqTone.value.trim(),
    detail_level: reqDetail.value.trim(),
    must_include: reqInclude.value.trim(),
    must_avoid: reqAvoid.value.trim(),
    format_expectations: reqFormat.value.trim(),
    domain_constraints: reqDomain.value.trim(),
    extra: {},
  };
}

function fillRequirementsForm(profile) {
  reqTone.value = profile.tone_style || '';
  reqDetail.value = profile.detail_level || '';
  reqInclude.value = profile.must_include || '';
  reqAvoid.value = profile.must_avoid || '';
  reqFormat.value = profile.format_expectations || '';
  reqDomain.value = profile.domain_constraints || '';
}

function setRequirementsStatus(msg, type = '') {
  requirementsStatusEl.textContent = msg;
  requirementsStatusEl.className = `req-status ${type}`;
}

async function loadRequirements() {
  const clientId = clientIdInput.value.trim() || 'default';
  try {
    const res = await fetch(`/requirements/${encodeURIComponent(clientId)}`);
    if (res.status === 404) {
      fillRequirementsForm({});
      setRequirementsStatus('No saved profile — fill in fields and click Save.', '');
      return;
    }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Load failed');
    fillRequirementsForm(data);
    setRequirementsStatus(`Loaded profile for "${clientId}"`, 'success');
  } catch (err) {
    setRequirementsStatus(`Load error: ${err.message}`, 'error');
  }
}

async function saveRequirements() {
  const clientId = clientIdInput.value.trim() || 'default';
  saveRequirementsBtn.disabled = true;
  try {
    const res = await fetch(`/requirements/${encodeURIComponent(clientId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(getRequirementsPayload()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Save failed');
    setRequirementsStatus(`Saved requirements for "${clientId}"`, 'success');
  } catch (err) {
    setRequirementsStatus(`Save error: ${err.message}`, 'error');
  } finally {
    saveRequirementsBtn.disabled = false;
  }
}

function renderSelfCorrection(data) {
  const met = data.requirements_met;
  scOutcomeBadge.textContent = met ? 'Requirements met' : 'Best effort — not fully met';
  scOutcomeBadge.className = met ? 'chip chip-gold' : 'chip';

  const profile = data.requirements_profile || {};
  scSummary.innerHTML = `
    <p><strong>${escapeHtml(data.message || '')}</strong></p>
    <p class="muted" style="margin-top:8px">Client: ${escapeHtml(data.client_id || '')} ·
      Passed on attempt ${data.passed_on_attempt ?? '—'} of 4 ·
      Final model: ${escapeHtml(data.final_model || '')} ·
      Judge score: ${fmt(data.final_judge?.score, 2)}</p>
    <details style="margin-top:10px">
      <summary class="muted" style="cursor:pointer">Requirements profile used</summary>
      <ul style="margin-top:8px;font-size:0.82rem;color:var(--text-muted)">
        <li>Tone: ${escapeHtml(profile.tone_style || '—')}</li>
        <li>Detail: ${escapeHtml(profile.detail_level || '—')}</li>
        <li>Must include: ${escapeHtml(profile.must_include || '—')}</li>
        <li>Must avoid: ${escapeHtml(profile.must_avoid || '—')}</li>
        <li>Format: ${escapeHtml(profile.format_expectations || '—')}</li>
        <li>Domain: ${escapeHtml(profile.domain_constraints || '—')}</li>
      </ul>
    </details>
  `;

  scAttempts.innerHTML = (data.attempts || [])
    .map((a) => {
      const j = a.judge || {};
      const pass = j.pass;
      const reasons = (j.reasons || [])
        .map((r) => `<li>${escapeHtml(r)}</li>`)
        .join('');
      return `
      <div class="sc-attempt ${pass ? 'pass' : 'fail'}">
        <div class="sc-attempt-head">
          <span class="sc-attempt-title">Attempt ${a.attempt}</span>
          <span class="chip">${escapeHtml(a.model)}</span>
          <span class="chip ${pass ? 'chip-gold' : ''}">${pass ? 'PASS' : 'FAIL'} · score ${fmt(j.score, 2)}</span>
          <span class="chip chip-muted">${escapeHtml(a.selection_mode || '')}</span>
        </div>
        ${!pass && reasons ? `<ul class="sc-reasons">${reasons}</ul><p class="muted" style="font-size:0.78rem;margin-top:6px">→ Next attempt will target these issues</p>` : ''}
        <div class="sc-prompt-preview">${escapeHtml(a.prompt || '')}</div>
        <div class="response-text" style="margin-top:8px">${escapeHtml(a.response || '')}</div>
      </div>`;
    })
    .join('');
}

function renderSelfCorrectAll(data) {
  lastResult = data;
  lastResultMode = 'self_correct';
  showResults(true, true);
  renderSelfCorrection(data);
  winnerModel.textContent = MODEL_META[data.best_model]?.label || data.best_model;
  winnerPrompt.textContent = `Attempt ${data.passed_on_attempt ?? 'best'}`;
  winnerDomain.textContent = data.cluster_id || '';
  bestResponse.textContent = data.best_answer || '';
  if (data.requirements_not_met) {
    bestResponse.textContent += `\n\n⚠ ${data.message}\nReasons: ${(data.final_judge?.reasons || []).join('; ')}`;
  }
}

function renderAll(data) {
  lastResult = data;
  lastResultMode = 'standard';
  showResults(true, false);
  updateModeBadge(data.routing, data.routing?.routing_enabled !== false);
  renderRoutingFlow(data);
  renderWinner(data);
  renderMetrics(data);
  renderResponses(data);
  bindFollowups();
}

async function runQuery() {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus('Enter a query first.', 'error');
    return;
  }

  const clientId = clientIdInput.value.trim() || 'default';
  const enableRouting = routingToggle.checked;
  const useSelfCorrection = selfCorrectionToggle.checked;

  if (useSelfCorrection) {
    const hasProfile =
      reqTone.value.trim() ||
      reqDetail.value.trim() ||
      reqInclude.value.trim() ||
      reqAvoid.value.trim() ||
      reqFormat.value.trim() ||
      reqDomain.value.trim();
    if (!hasProfile) {
      setStatus('Set answer requirements first (tone, must include, must avoid, etc.) then Save.', 'error');
      return;
    }
  }

  showResults(false);

  setStatus(
    useSelfCorrection
      ? 'Running self-correction loop (all models → judge → retry)…'
      : enableRouting
        ? 'Running with adaptive routing…'
        : 'Running all models (routing off)…'
  );
  runBtn.disabled = true;

  try {
    if (useSelfCorrection) {
      await saveRequirements();
      const res = await fetch('/query/self-correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, client_id: clientId, enable_routing: enableRouting }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Self-correction failed');
      data.query = query;
      renderSelfCorrectAll(data);
      setStatus(
        data.requirements_met
          ? `Requirements met on attempt ${data.passed_on_attempt}`
          : `Best available after 4 attempts — see judge reasons`,
        data.requirements_met ? 'success' : 'error'
      );
    } else {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, enable_routing: enableRouting }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Query failed');

      currentLogId = data.log_id || null;
      data.query = query;
      renderAll(data);

      if ((data.errors || []).length) {
        setStatus(`Completed with errors: ${data.errors[0]}`, 'error');
      } else {
        setStatus(`Done — ${data.routing?.mode === 'direct' && enableRouting ? 'direct route' : 'full exploration'}`, 'success');
      }
    }

    await Promise.all([loadReport(), loadRoutingStatus(), loadSeedStatus()]);
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'error');
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener('click', runQuery);
saveRequirementsBtn.addEventListener('click', saveRequirements);
loadRequirementsBtn.addEventListener('click', loadRequirements);
clientIdInput.addEventListener('change', loadRequirements);
reportBtn.addEventListener('click', showReportModal);
closeReportModalBtn.addEventListener('click', closeReportModal);
reportModalBackdrop.addEventListener('click', closeReportModal);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !reportModal.classList.contains('hidden')) closeReportModal();
});
refreshRoutingBtn.addEventListener('click', () => Promise.all([loadRoutingStatus(), loadSeedStatus()]));
forceReexploreAllBtn.addEventListener('click', async () => {
  try {
    await fetch('/routing/re-explore', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ all_clusters: true }),
    });
    setStatus('Forced re-exploration for all clusters', 'success');
    await loadRoutingStatus();
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'error');
  }
});

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runQuery();
});

routingToggle.addEventListener('change', () => {
  const on = routingToggle.checked;
  setStatus(on ? 'Adaptive routing enabled for next query' : 'Next query will call all models', '');
});

seedBootstrapBtn.addEventListener('click', async () => {
  seedBootstrapBtn.disabled = true;
  setStatus('Bootstrapping routing clusters (22 queries × 4 domains)…');
  try {
    const res = await fetch('/routing/seed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'bootstrap' }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Bootstrap failed');
    setStatus(`Bootstrap complete — all ready: ${data.status?.all_ready}`, 'success');
    await Promise.all([loadSeedStatus(), loadRoutingStatus()]);
  } catch (err) {
    setStatus(`Bootstrap error: ${err.message}`, 'error');
  } finally {
    seedBootstrapBtn.disabled = false;
  }
});

seedVerifyBtn.addEventListener('click', async () => {
  seedVerifyBtn.disabled = true;
  setStatus('Verifying direct routing…');
  try {
    const res = await fetch('/routing/seed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'verify' }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Verify failed');
    const lines = Object.entries(data.results || {}).map(
      ([k, v]) => `${k}: ${v.direct_routing_worked ? 'direct ✓' : `explore (${v.reason})`}`
    );
    setStatus(`Verify: ${lines.join(' · ')}`, lines.every((l) => l.includes('✓')) ? 'success' : 'error');
  } catch (err) {
    setStatus(`Verify error: ${err.message}`, 'error');
  } finally {
    seedVerifyBtn.disabled = false;
  }
});

loadReport();
loadRoutingStatus();
loadSeedStatus();
loadRequirements();
