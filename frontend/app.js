const state = {
  page: "Substation Forecast",
  results: {},
};

const PAGES = [
  "Substation Forecast",
  "Feeder Forecast",
  "Enterprise Forecast",
  "Fingerprint Heatmap",
  "Forecast Explanation",
];

const TARGETS = {
  substation: ["SS_KTM_01"],
  feeder: ["FD_RES_01", "FD_COM_01"],
  enterprise: ["ENT_CEMENT_01", "ENT_HOSPITAL_01"],
};

document.addEventListener("DOMContentLoaded", async () => {
  renderNav();
  bindGlobalActions();
  renderPage();
});

function bindGlobalActions() {
  document.getElementById("refresh-button").addEventListener("click", renderPage);
  document.getElementById("generate-button").addEventListener("click", async () => {
    try {
      setError(null);
      await apiPost("/api/v1/forecast/history/generate?days=30&seed=7");
      await apiPost("/api/v1/forecast/fingerprint/build");
      renderPage();
    } catch (error) {
      setError(error.message);
    }
  });
}

function renderNav() {
  const nav = document.getElementById("page-nav");
  nav.innerHTML = "";
  PAGES.forEach((page) => {
    const button = document.createElement("button");
    button.className = `nav-button${state.page === page ? " active" : ""}`;
    button.textContent = page;
    button.addEventListener("click", () => {
      state.page = page;
      renderNav();
      renderPage();
    });
    nav.appendChild(button);
  });
}

function renderPage() {
  document.getElementById("page-title").textContent = state.page;
  const root = document.getElementById("page-content");
  root.innerHTML = "";

  if (state.page === "Substation Forecast") {
    renderForecastTargetPage(root, "substation", "Substation Forecast");
    return;
  }
  if (state.page === "Feeder Forecast") {
    renderForecastTargetPage(root, "feeder", "Feeder Forecast");
    return;
  }
  if (state.page === "Enterprise Forecast") {
    renderForecastTargetPage(root, "enterprise", "Enterprise Forecast");
    return;
  }
  if (state.page === "Fingerprint Heatmap") {
    renderFingerprintPage(root);
    return;
  }
  renderExplanationPage(root);
}

function renderForecastTargetPage(root, entityType, title) {
  const options = TARGETS[entityType]
    .map((value) => `<option value="${value}">${value}</option>`)
    .join("");
  const cached = state.results[entityType] || {};
  root.innerHTML = `
    <section class="card">
      <div class="controls">
        <div class="control-field">
          <label>${title} ID</label>
          <select id="${entityType}-id">${options}</select>
        </div>
        <div class="control-field">
          <label>Horizon</label>
          <select id="${entityType}-horizon">
            <option value="1h">1h</option>
            <option value="4h">4h</option>
            <option value="24h">24h</option>
          </select>
        </div>
        <div class="control-field">
          <label>Model</label>
          <select id="${entityType}-model">
            <option value="tree">tree</option>
            <option value="fingerprint_baseline">fingerprint_baseline</option>
          </select>
        </div>
        <div class="control-field">
          <label>&nbsp;</label>
          <button id="${entityType}-train" class="primary-button">Train</button>
        </div>
        <div class="control-field">
          <label>&nbsp;</label>
          <button id="${entityType}-run" class="secondary-button">Run Forecast</button>
        </div>
        <div class="control-field">
          <label>&nbsp;</label>
          <button id="${entityType}-evaluate" class="secondary-button">Evaluate</button>
        </div>
      </div>
      <div id="${entityType}-result">
        ${cached.forecast ? renderForecastResult(cached.forecast, cached.evaluation) : "<p class='subtle'>No forecast run yet.</p>"}
      </div>
    </section>
  `;

  document.getElementById(`${entityType}-train`).addEventListener("click", async () => {
    await handleForecastTrain(entityType);
  });
  document.getElementById(`${entityType}-run`).addEventListener("click", async () => {
    await handleForecastRun(entityType);
  });
  document.getElementById(`${entityType}-evaluate`).addEventListener("click", async () => {
    await handleForecastEvaluate(entityType);
  });
}

async function handleForecastTrain(entityType) {
  try {
    setError(null);
    const id = document.getElementById(`${entityType}-id`).value;
    const horizon = document.getElementById(`${entityType}-horizon`).value;
    const model = document.getElementById(`${entityType}-model`).value;
    await apiPost(`/api/v1/forecast/train?${entityType}_id=${encodeURIComponent(id)}&horizon=${horizon}&model=${model}`);
    await handleForecastRun(entityType);
  } catch (error) {
    setError(error.message);
  }
}

async function handleForecastRun(entityType) {
  try {
    setError(null);
    const id = document.getElementById(`${entityType}-id`).value;
    const horizon = document.getElementById(`${entityType}-horizon`).value;
    const model = document.getElementById(`${entityType}-model`).value;
    const response = await apiGet(`/api/v1/forecast/${entityType}?${entityType}_id=${encodeURIComponent(id)}&horizon=${horizon}&model=${model}`);
    state.results[entityType] = {
      ...state.results[entityType],
      forecast: response.forecast,
    };
    renderPage();
  } catch (error) {
    setError(error.message);
  }
}

async function handleForecastEvaluate(entityType) {
  try {
    setError(null);
    const id = document.getElementById(`${entityType}-id`).value;
    const horizon = document.getElementById(`${entityType}-horizon`).value;
    const model = document.getElementById(`${entityType}-model`).value;
    const response = await apiGet(`/api/v1/forecast/evaluate?${entityType}_id=${encodeURIComponent(id)}&horizon=${horizon}&model=${model}`);
    state.results[entityType] = {
      ...state.results[entityType],
      evaluation: response.evaluation,
    };
    renderPage();
  } catch (error) {
    setError(error.message);
  }
}

function renderForecastResult(forecast, evaluation) {
  return `
    <div class="two-column-grid">
      <article class="card">
        <h3>Forecast Summary</h3>
        <div class="snapshot-grid">
          ${metricCard("Peak Load", formatNumber(forecast.summary.peak_load_kw), "kW")}
          ${metricCard("Peak Time", forecast.summary.peak_time, "")}
          ${metricCard("Mean Load", formatNumber(forecast.summary.mean_load_kw), "kW")}
          ${metricCard("Energy", formatNumber(forecast.summary.total_energy_kwh), "kWh")}
        </div>
        ${renderForecastChart(forecast.slot_predictions)}
      </article>
      <article class="card">
        <h3>Evaluation and Aggregates</h3>
        ${evaluation ? renderMetrics(evaluation.metrics) : "<p class='subtle'>Run evaluation to see error metrics.</p>"}
        <h4>Aggregated Summary</h4>
        ${renderTable(forecast.aggregated_summary, ["hour_start", "mean_load_kw", "peak_load_kw", "total_energy_kwh"])}
      </article>
    </div>
    <section class="card">
      <h3>Slot Predictions</h3>
      ${renderTable(forecast.slot_predictions, ["timestamp", "predicted_load_kw", "p10_kw", "p90_kw", "fingerprint_mean_kw"])}
    </section>
  `;
}

function renderFingerprintPage(root) {
  root.innerHTML = `
    <section class="card">
      <div class="controls">
        <div class="control-field">
          <label>Entity Type</label>
          <select id="fingerprint-entity-type">
            <option value="feeder">feeder</option>
            <option value="substation">substation</option>
            <option value="enterprise">enterprise</option>
          </select>
        </div>
        <div class="control-field">
          <label>Entity ID</label>
          <select id="fingerprint-entity-id">${TARGETS.feeder.map((value) => `<option value="${value}">${value}</option>`).join("")}</select>
        </div>
        <div class="control-field">
          <label>&nbsp;</label>
          <button id="fingerprint-build" class="primary-button">Build Fingerprints</button>
        </div>
        <div class="control-field">
          <label>&nbsp;</label>
          <button id="fingerprint-load" class="secondary-button">Load Heatmap</button>
        </div>
      </div>
      <div id="fingerprint-result"><p class="subtle">No fingerprint view loaded yet.</p></div>
    </section>
  `;

  document.getElementById("fingerprint-entity-type").addEventListener("change", (event) => {
    const type = event.target.value;
    document.getElementById("fingerprint-entity-id").innerHTML = TARGETS[type]
      .map((value) => `<option value="${value}">${value}</option>`)
      .join("");
  });

  document.getElementById("fingerprint-build").addEventListener("click", async () => {
    try {
      setError(null);
      await apiPost("/api/v1/forecast/fingerprint/build");
      await loadFingerprintHeatmap();
    } catch (error) {
      setError(error.message);
    }
  });
  document.getElementById("fingerprint-load").addEventListener("click", loadFingerprintHeatmap);
}

async function loadFingerprintHeatmap() {
  try {
    setError(null);
    const type = document.getElementById("fingerprint-entity-type").value;
    const id = document.getElementById("fingerprint-entity-id").value;
    const response = await apiGet(`/api/v1/forecast/fingerprint?entity_type=${type}&entity_id=${encodeURIComponent(id)}`);
    document.getElementById("fingerprint-result").innerHTML = renderFingerprintHeatmap(response.records);
  } catch (error) {
    setError(error.message);
  }
}

function renderFingerprintHeatmap(records) {
  if (!records.length) {
    return "<p class='subtle'>No fingerprint records found for this target.</p>";
  }
  const maxValue = Math.max(...records.map((item) => Number(item.fingerprint_mean_kw)));
  const cells = records
    .map((item) => {
      const intensity = Number(item.fingerprint_mean_kw) / maxValue;
      return `<div class="heatmap-cell" style="background: rgba(30, 107, 77, ${Math.max(0.12, intensity).toFixed(2)})" title="day ${item.day_of_week}, slot ${item.slot_index}, mean ${formatNumber(item.fingerprint_mean_kw)}">${item.slot_index}</div>`;
    })
    .join("");
  return `<div class="heatmap-grid">${cells}</div>`;
}

function renderExplanationPage(root) {
  root.innerHTML = `
    <section class="card">
      <div class="controls">
        <div class="control-field">
          <label>Feeder ID</label>
          <select id="explain-feeder-id">${TARGETS.feeder.map((value) => `<option value="${value}">${value}</option>`).join("")}</select>
        </div>
        <div class="control-field">
          <label>Horizon</label>
          <select id="explain-horizon">
            <option value="1h">1h</option>
            <option value="4h">4h</option>
            <option value="24h">24h</option>
          </select>
        </div>
        <div class="control-field">
          <label>&nbsp;</label>
          <button id="explain-train" class="primary-button">Train Tree Model</button>
        </div>
        <div class="control-field">
          <label>&nbsp;</label>
          <button id="explain-run" class="secondary-button">Explain Forecast</button>
        </div>
      </div>
      <div id="explain-result"><p class="subtle">No explanation generated yet.</p></div>
    </section>
  `;

  document.getElementById("explain-train").addEventListener("click", async () => {
    try {
      setError(null);
      const id = document.getElementById("explain-feeder-id").value;
      const horizon = document.getElementById("explain-horizon").value;
      await apiPost(`/api/v1/forecast/train?feeder_id=${encodeURIComponent(id)}&horizon=${horizon}&model=tree`);
      await loadExplanation();
    } catch (error) {
      setError(error.message);
    }
  });
  document.getElementById("explain-run").addEventListener("click", loadExplanation);
}

async function loadExplanation() {
  try {
    setError(null);
    const id = document.getElementById("explain-feeder-id").value;
    const horizon = document.getElementById("explain-horizon").value;
    const response = await apiGet(`/api/v1/explain/forecast?feeder_id=${encodeURIComponent(id)}&horizon=${horizon}&model=tree`);
    document.getElementById("explain-result").innerHTML = `
      <article class="card">
        <h3>Plain-language Explanation</h3>
        <p>${escapeHtml(response.explanation.plain_language_explanation)}</p>
        <h4>Top Features</h4>
        ${renderTable(response.explanation.top_features, ["feature", "importance", "feature_value"])}
      </article>
    `;
  } catch (error) {
    setError(error.message);
  }
}

function renderForecastChart(slotPredictions) {
  const width = 720;
  const height = 240;
  const padding = 32;
  const values = slotPredictions.flatMap((item) => [item.p10_kw, item.p90_kw, item.predicted_load_kw, item.fingerprint_mean_kw]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const xStep = slotPredictions.length > 1 ? (width - padding * 2) / (slotPredictions.length - 1) : 0;
  const scaleY = (value) => height - padding - ((value - min) / Math.max(max - min, 1e-6)) * (height - padding * 2);
  const points = slotPredictions.map((item, index) => `${padding + index * xStep},${scaleY(item.predicted_load_kw)}`).join(" ");
  const baseline = slotPredictions.map((item, index) => `${padding + index * xStep},${scaleY(item.fingerprint_mean_kw)}`).join(" ");
  const upper = slotPredictions.map((item, index) => `${padding + index * xStep},${scaleY(item.p90_kw)}`).join(" ");
  const lower = [...slotPredictions].reverse().map((item, index) => {
    const originalIndex = slotPredictions.length - 1 - index;
    return `${padding + originalIndex * xStep},${scaleY(item.p10_kw)}`;
  }).join(" ");

  return `
    <div class="chart-shell">
      <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Forecast chart">
        <polygon points="${upper} ${lower}" fill="rgba(30, 107, 77, 0.12)"></polygon>
        <polyline points="${baseline}" fill="none" stroke="#d9822b" stroke-width="2" stroke-dasharray="6 6"></polyline>
        <polyline points="${points}" fill="none" stroke="#1e6b4d" stroke-width="3"></polyline>
      </svg>
      <div class="legend-row">
        <span><i class="legend-swatch main"></i> Forecast</span>
        <span><i class="legend-swatch band"></i> p10/p90 band</span>
        <span><i class="legend-swatch warning"></i> Fingerprint baseline</span>
      </div>
    </div>
  `;
}

function renderMetrics(metrics) {
  return `
    <div class="snapshot-grid">
      ${Object.entries(metrics).map(([key, value]) => metricCard(key, formatNumber(value), "")).join("")}
    </div>
  `;
}

function metricCard(label, value, suffix) {
  return `
    <article class="card">
      <p class="metric-label">${escapeHtml(label)}</p>
      <p class="metric-value compact">${escapeHtml(String(value))} ${suffix}</p>
    </article>
  `;
}

function renderTable(rows, columns) {
  if (!rows || !rows.length) {
    return "<p class='subtle'>No data available.</p>";
  }
  return `
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) => `
                <tr>
                  ${columns.map((column) => `<td>${escapeHtml(String(row[column] ?? ""))}</td>`).join("")}
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function apiGet(path) {
  const response = await fetch(path);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail?.message || payload.detail || "API request failed");
  }
  return payload;
}

async function apiPost(path) {
  const response = await fetch(path, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail?.message || payload.detail || "API request failed");
  }
  return payload;
}

function setError(message) {
  const banner = document.getElementById("error-banner");
  if (!message) {
    banner.classList.add("hidden");
    banner.textContent = "";
    return;
  }
  banner.classList.remove("hidden");
  banner.textContent = message;
}

function formatNumber(value) {
  return Number(value).toFixed(3);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
