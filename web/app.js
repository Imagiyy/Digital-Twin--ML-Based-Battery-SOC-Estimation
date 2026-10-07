// Frontend Application Logic for SOC Digital Twin Web Dashboard

const $ = (id) => document.getElementById(id);

// App State
let ws = null;
let isPlaying = true;
let speed = 10;
let latencyHistory = [];
const maxDataPoints = 150; // Cap stored points for smooth 100x rendering

// Chart Instances
let chartSoc = null;
let chartError = null;
let chartVi = null;

// Initialize when DOM ready
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupThemeToggle();
  setupCharts();
  setupControls();
  setupModelExplorer();
  loadEvaluationTable();
  loadSohData();
  loadAlgosData();
  setupCustomImport();
  refreshCycleList();
  connectWebSocket();
});

// Tab Navigation
function setupTabs() {
  const tabs = document.querySelectorAll("nav.tab-nav button");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.remove("active");
      });
      const targetPane = $(targetId);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// Theme Toggle
function setupThemeToggle() {
  const btn = $("theme-btn");
  btn.addEventListener("click", () => {
    const isDark = document.body.getAttribute("data-theme") === "dark";
    if (isDark) {
      document.body.removeAttribute("data-theme");
    } else {
      document.body.setAttribute("data-theme", "dark");
    }
  });
}

// Setup Chart.js Charts
function setupCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        grid: { color: "rgba(100, 116, 139, 0.15)" },
        ticks: { font: { size: 10 }, color: "#64748b" },
      },
      y: {
        grid: { color: "rgba(100, 116, 139, 0.15)" },
        ticks: { font: { size: 10 }, color: "#64748b" },
      },
    },
  };

  // 1. SOC Tracking Chart
  const ctxSoc = $("chart-soc").getContext("2d");
  chartSoc = new Chart(ctxSoc, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "True SOC", data: [], borderColor: "#1f2937", borderWidth: 2, pointRadius: 0 },
        { label: "ML Estimate", data: [], borderColor: "#16a34a", borderWidth: 2, borderDash: [4, 4], pointRadius: 0 },
        { label: "SPKF Estimate", data: [], borderColor: "#8b5cf6", borderWidth: 1.8, borderDash: [3, 3], pointRadius: 0 },
        { label: "EKF Estimate", data: [], borderColor: "#06b6d4", borderWidth: 1.8, borderDash: [2, 2], pointRadius: 0 },
        { label: "Coulomb Count", data: [], borderColor: "#dc2626", borderWidth: 1.5, borderDash: [2, 2], pointRadius: 0 },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: { min: -5, max: 105, title: { display: true, text: "SOC (%)", font: { size: 11 } } },
      },
    },
  });

  // 2. Estimation Error Chart
  const ctxErr = $("chart-error").getContext("2d");
  chartError = new Chart(ctxErr, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "ML Error", data: [], borderColor: "#16a34a", borderWidth: 1.8, pointRadius: 0 },
        { label: "CC Error", data: [], borderColor: "#dc2626", borderWidth: 1.2, borderDash: [2, 2], pointRadius: 0 },
        { label: "+3% Target", data: [], borderColor: "rgba(22,163,74,0.4)", borderWidth: 1, borderDash: [3, 3], pointRadius: 0 },
        { label: "-3% Target", data: [], borderColor: "rgba(22,163,74,0.4)", borderWidth: 1, borderDash: [3, 3], pointRadius: 0 },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        ...commonOptions.scales,
        y: { min: -8, max: 8, title: { display: true, text: "Error (pts)", font: { size: 11 } } },
      },
    },
  });

  // 3. Voltage and Current Chart
  const ctxVi = $("chart-vi").getContext("2d");
  chartVi = new Chart(ctxVi, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Voltage (V)", data: [], borderColor: "#8c1236", borderWidth: 2, yAxisID: "yV", pointRadius: 0 },
        { label: "Current (A)", data: [], borderColor: "#2563eb", borderWidth: 1.5, yAxisID: "yI", pointRadius: 0 },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        x: commonOptions.scales.x,
        yV: { type: "linear", position: "left", min: 2.8, max: 4.3, title: { display: true, text: "Voltage (V)" } },
        yI: { type: "linear", position: "right", min: -3.5, max: 3.5, title: { display: true, text: "Current (A)" } },
      },
    },
  });
}

// Interactive Controls & Event Handlers
function setupControls() {
  const btnPlay = $("btn-play-pause");
  btnPlay.addEventListener("click", () => {
    isPlaying = !isPlaying;
    btnPlay.textContent = isPlaying ? "Pause" : "Play";
    sendControl(isPlaying ? "play" : "pause");
  });

  $("btn-restart").addEventListener("click", () => {
    resetCharts();
    sendControl("restart");
    isPlaying = true;
    btnPlay.textContent = "Pause";
  });

  $("sel-cycle").addEventListener("change", (e) => {
    resetCharts();
    sendControl("select_cycle", e.target.value);
    $("cycle-badge").textContent = e.target.options[e.target.selectedIndex].text.split(":")[0];
  });

  $("sel-speed").addEventListener("change", (e) => {
    speed = parseFloat(e.target.value);
    sendControl("speed", speed);
  });

  $("rng-noise").addEventListener("input", (e) => {
    $("lbl-noise").textContent = e.target.value;
    sendParams({ noise_mv: parseFloat(e.target.value) });
  });

  $("rng-offset").addEventListener("input", (e) => {
    $("lbl-offset").textContent = e.target.value;
    sendParams({ offset_ma: parseFloat(e.target.value) });
  });

  $("sel-adc-bits").addEventListener("change", (e) => {
    sendParams({ adc_bits: parseInt(e.target.value) });
  });

  $("btn-export-csv").addEventListener("click", () => {
    window.location.href = "/api/export";
  });

  $("btn-copy-payload").addEventListener("click", () => {
    const text = $("val-wifi-json").textContent;
    navigator.clipboard.writeText(text).then(() => {
      $("btn-copy-payload").textContent = "Copied!";
      setTimeout(() => { $("btn-copy-payload").textContent = "Copy"; }, 1500);
    });
  });
}

function resetCharts() {
  [chartSoc, chartError, chartVi].forEach((chart) => {
    if (chart) {
      chart.data.labels = [];
      chart.data.datasets.forEach((ds) => { ds.data = []; });
      chart.update();
    }
  });
}

async function sendControl(action, value = null) {
  try {
    await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, value }),
    });
  } catch (err) {
    console.error("Control API error:", err);
  }
}

async function sendParams(params) {
  try {
    await fetch("/api/params", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
  } catch (err) {
    console.error("Params API error:", err);
  }
}

// WebSocket Stream Connection
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
  
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    $("latency-badge").textContent = "WS: Connected";
  };

  ws.onmessage = (event) => {
    const frame = JSON.parse(event.data);
    updateDashboard(frame);
  };

  ws.onclose = () => {
    $("latency-badge").textContent = "WS: Reconnecting...";
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    ws.close();
  };
}

// Update Dashboard View with Telemetry Frame
function updateDashboard(frame) {
  // 1. Latency badge
  if (frame.end_to_end_latency_ms !== undefined) {
    $("latency-badge").textContent = `Latency: ${frame.end_to_end_latency_ms.toFixed(2)} ms`;
  }

  // 2. SOC Ring and Value
  const socVal = frame.soc_ml;
  $("val-soc-ring").textContent = `${socVal.toFixed(1)}%`;
  
  // SVG Ring calculation: circumference = 2 * PI * 70 = 439.8
  const maxCircumference = 439.8;
  const dashOffset = maxCircumference * (1 - Math.max(0, Math.min(100, socVal)) / 100);
  const arc = $("soc-arc");
  arc.style.strokeDashoffset = dashOffset;
  
  // Dynamic color shift
  if (socVal > 30) {
    arc.style.stroke = "var(--ok)"; // Green
  } else if (socVal > 15) {
    arc.style.stroke = "var(--wn)"; // Amber
  } else {
    arc.style.stroke = "var(--er)"; // Red
  }

  // 3. Voltage, Current, Status
  $("val-cell-v").textContent = `${frame.chain.reconstructed_v.toFixed(3)} V`;
  $("val-cell-i").textContent = `${frame.chain.measured_i.toFixed(2)} A`;
  
  const statusEl = $("val-status");
  statusEl.textContent = frame.status;
  if (frame.status === "CHARGING") {
    statusEl.style.color = "var(--wn)";
  } else if (frame.status === "CHARGED_STANDBY") {
    statusEl.style.color = "var(--ok)";
  } else {
    statusEl.style.color = "var(--bl)";
  }

  // 4. TP4056 LED indicators
  const ledChrg = $("led-chrg");
  const ledStdby = $("led-stdby");
  if (frame.led_chrg) {
    ledChrg.className = "led-indicator active-red";
  } else {
    ledChrg.className = "led-indicator";
  }
  if (frame.led_stdby) {
    ledStdby.className = "led-indicator active-green";
  } else {
    ledStdby.className = "led-indicator";
  }

  // 5. Wi-Fi JSON Payload Box
  $("val-wifi-json").textContent = frame.wifi_json;

  // 6. Virtual Hardware Chain Values
  $("flow-v-cell").textContent = `${frame.chain.cell_v.toFixed(3)} V`;
  $("flow-v-pin").textContent = `${frame.chain.pin_v.toFixed(3)} V`;
  $("flow-adc-code").textContent = frame.chain.adc_code;
  $("flow-soc-ml").textContent = `${frame.soc_ml.toFixed(1)} %`;

  // 7. Accuracy Metrics Panel
  if ($("m-ml-mae") && frame.metrics.running_ml_mae !== undefined) {
    $("m-ml-mae").textContent = `${frame.metrics.running_ml_mae.toFixed(2)} %`;
  }
  if ($("m-spkf-mae") && frame.metrics.running_spkf_mae !== undefined) {
    $("m-spkf-mae").textContent = `${frame.metrics.running_spkf_mae.toFixed(2)} %`;
  }
  if ($("m-ekf-mae") && frame.metrics.running_ekf_mae !== undefined) {
    $("m-ekf-mae").textContent = `${frame.metrics.running_ekf_mae.toFixed(2)} %`;
  }
  if ($("m-cc-mae") && frame.metrics.running_cc_mae !== undefined) {
    $("m-cc-mae").textContent = `${frame.metrics.running_cc_mae.toFixed(2)} %`;
  }
  if ($("m-infer-time") && frame.metrics.infer_time_us !== undefined) {
    $("m-infer-time").textContent = `${frame.metrics.infer_time_us.toFixed(1)} \u03bcs`;
  }

  // 8. SOH (State of Health) Monitoring Widget
  if (frame.soh) {
    if ($("val-soh-combined")) {
      $("val-soh-combined").textContent = `${frame.soh.soh_combined_pct.toFixed(1)}%`;
    }
    if ($("val-soh-status")) {
      $("val-soh-status").textContent = frame.soh.status || "HEALTHY";
      $("val-soh-status").style.color = frame.soh.status_color || "var(--ok)";
    }
    if ($("val-soh-r0-rul")) {
      const r0Str = frame.soh.estimated_r0_mohm ? frame.soh.estimated_r0_mohm.toFixed(1) : "--";
      const rulStr = frame.soh.rul_cycles !== undefined ? frame.soh.rul_cycles : "--";
      $("val-soh-r0-rul").textContent = `${r0Str} m\u03a9 \u00b7 ${rulStr} cyc`;
    }
    if ($("val-soh-cap")) {
      $("val-soh-cap").textContent = `${frame.soh.soh_capacity_pct.toFixed(1)}%`;
    }
    if ($("val-soh-res")) {
      $("val-soh-res").textContent = `${frame.soh.soh_resistance_pct.toFixed(1)}%`;
    }
    if ($("bar-soh-cap")) {
      $("bar-soh-cap").style.width = `${Math.min(100, Math.max(0, frame.soh.soh_capacity_pct))}%`;
    }
    if ($("bar-soh-res")) {
      $("bar-soh-res").style.width = `${Math.min(100, Math.max(0, frame.soh.soh_resistance_pct))}%`;
    }
  }

  // 9. Stream to Charts
  const timeHrs = (frame.time_s / 3600).toFixed(2);
  pushChartData(timeHrs, frame);
}

function pushChartData(timeStr, frame) {
  // Chart 1: SOC (True, ML, SPKF, EKF, CC)
  chartSoc.data.labels.push(timeStr);
  chartSoc.data.datasets[0].data.push(frame.soc_true);
  chartSoc.data.datasets[1].data.push(frame.soc_ml);
  chartSoc.data.datasets[2].data.push(frame.soc_spkf !== undefined ? frame.soc_spkf : null);
  chartSoc.data.datasets[3].data.push(frame.soc_ekf !== undefined ? frame.soc_ekf : null);
  chartSoc.data.datasets[4].data.push(frame.soc_cc_biased);

  // Chart 2: Error
  const errMl = frame.soc_ml - frame.soc_true;
  const errCc = frame.soc_cc_biased - frame.soc_true;
  chartError.data.labels.push(timeStr);
  chartError.data.datasets[0].data.push(errMl);
  chartError.data.datasets[1].data.push(errCc);
  chartError.data.datasets[2].data.push(3.0);
  chartError.data.datasets[3].data.push(-3.0);

  // Chart 3: Voltage & Current
  chartVi.data.labels.push(timeStr);
  chartVi.data.datasets[0].data.push(frame.chain.reconstructed_v);
  chartVi.data.datasets[1].data.push(frame.chain.measured_i);

  // Decimation cap
  [chartSoc, chartError, chartVi].forEach((c) => {
    if (c.data.labels.length > maxDataPoints) {
      c.data.labels.shift();
      c.data.datasets.forEach((ds) => ds.data.shift());
    }
    c.update();
  });
}

// Populate Tab 2: Evaluation Table
async function loadEvaluationTable() {
  try {
    const res = await fetch("/api/results");
    const data = await res.json();
    const tbody = document.querySelector("#table-heldout tbody");
    if (!tbody || !data.held_out_results) return;

    tbody.innerHTML = "";
    data.held_out_results.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${row["Cycle"]}</strong></td>
        <td>${row["Rate (A)"]}</td>
        <td style="color:var(--ok);font-weight:700;">${row["ML MAE (%)"]}%</td>
        <td>${row["ML RMSE (%)"]}%</td>
        <td>${row["ML Max (%)"]}%</td>
        <td>${row["CC MAE (%)"]}%</td>
        <td>${row["CC RMSE (%)"]}%</td>
        <td>${row["CC Max (%)"]}%</td>
        <td><span class="pill" style="color:var(--ok);border-color:var(--ok);padding:2px 8px;">Passed ✓</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading evaluation table:", err);
  }
}

// Populate SOH History Table in Tab 2
async function loadSohData() {
  try {
    const res = await fetch("/api/soh");
    const data = await res.json();
    const tbody = document.querySelector("#table-soh-history tbody");
    if (!tbody || !data.degradation_history) return;

    tbody.innerHTML = "";
    data.degradation_history.forEach((row) => {
      const tr = document.createElement("tr");
      const healthColor = row.soh_combined_pct >= 90 ? "var(--ok)" : (row.soh_combined_pct >= 80 ? "var(--wn)" : "var(--er)");
      tr.innerHTML = `
        <td><strong>Cycle ${row.cycle}</strong></td>
        <td>${row.capacity_ah.toFixed(3)} Ah</td>
        <td>${row.soh_capacity_pct.toFixed(1)}%</td>
        <td>${row.r0_mohm.toFixed(1)} m&Omega;</td>
        <td>${row.soh_resistance_pct.toFixed(1)}%</td>
        <td style="color:${healthColor};font-weight:700;">${row.soh_combined_pct.toFixed(1)}%</td>
        <td>${row.rul_cycles} cycles</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading SOH history:", err);
  }
}

// Populate / Verify Algorithm Comparison Table in Tab 2
async function loadAlgosData() {
  try {
    const res = await fetch("/api/algos");
    const data = await res.json();
    if (data.scorecard) {
      const tbody = document.querySelector("#table-algo-scorecard tbody");
      if (tbody) {
        tbody.innerHTML = "";
        data.scorecard.forEach((s) => {
          const isRec = s.status === "RECOMMENDED";
          const tr = document.createElement("tr");
          if (isRec) tr.className = "algo-row-recommended";
          const rankClass = s.rank === 1 ? "algo-rank-1" : (s.rank === 2 ? "algo-rank-2" : (s.rank === 3 ? "algo-rank-3" : ""));
          const statusBadge = isRec 
            ? `<span class="pill" style="background:var(--ok);color:#fff;border-color:var(--ok);">RECOMMENDED</span>`
            : (s.rank === 2 ? `<span class="pill" style="background:var(--bl);color:#fff;border-color:var(--bl);">${s.status}</span>` : `<span style="font-size:11px;color:var(--mut);">${s.status}</span>`);
          tr.innerHTML = `
            <td><span class="algo-rank-badge ${rankClass}">${s.rank}</span></td>
            <td><strong>${s.name}</strong></td>
            <td>${s.category}</td>
            <td><b style="${isRec ? 'color:var(--ok);' : ''}">${s.avg_mae.toFixed(2)} %</b></td>
            <td>${s.avg_rmse.toFixed(2)} %</td>
            <td><b style="${isRec ? 'color:var(--ok);' : ''}">${s.latency_us.toFixed(2)} &mu;s</b></td>
            <td><b style="${isRec ? 'color:var(--ok);' : ''}">${s.ram_flash_bytes} B</b></td>
            <td>${s.sensor_bias_immunity}</td>
            <td>${s.ecm_calibration_needed}</td>
            <td>${statusBadge}</td>
          `;
          tbody.appendChild(tr);
        });
      }
    }
  } catch (err) {
    console.error("Error loading algorithm comparison:", err);
  }
}

// Populate Tab 3: Model Explorer
async function setupModelExplorer() {
  const grid = $("hidden-neurons-grid");
  if (!grid) return;
  grid.innerHTML = "";
  for (let j = 0; j < 16; j++) {
    const n = document.createElement("div");
    n.className = "neuron";
    n.textContent = `h${j + 1}`;
    grid.appendChild(n);
  }

  try {
    const res = await fetch("/api/model");
    const data = await res.json();
    if (data.footprint) {
      $("fp-ram").textContent = `${data.footprint.memory_ram_flash_bytes_fp32} bytes`;
      $("fp-quant").textContent = `${data.footprint.int8_quantization_mae_pct.toFixed(4)} % points`;
    }
  } catch (err) {
    console.error("Model explorer fetch error:", err);
  }
}

// ============================================================================
// Tab 5: Custom Data Import & SOC Prediction Logic
// ============================================================================
let chartCustomSoc = null;
let chartCustomVi = null;
let chartCustomCurr = null;
let lastUploadedCsvContent = "";
let lastUploadedFilename = "custom_data.csv";

function setupCustomImport() {
  const dropzone = $("custom-dropzone");
  const fileInput = $("file-custom-input");
  const fileNameDisplay = $("dropzone-file-name");
  const btnSample = $("btn-sample-csv");
  const btnLoadProfile = $("btn-load-sample-profile");
  const btnToggleEditor = $("btn-toggle-editor");
  const editorContainer = $("raw-editor-container");
  const txtRaw = $("txt-custom-raw");
  const btnRun = $("btn-run-prediction");
  const alertMsg = $("custom-alert-msg");
  const resultsSec = $("custom-results-section");
  const btnExport = $("btn-export-custom-csv");
  const btnReplayTwin = $("btn-replay-in-twin");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    lastUploadedFilename = file.name;
    if (fileNameDisplay) {
      fileNameDisplay.innerHTML = `<strong>Selected:</strong> ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    }
    const reader = new FileReader();
    reader.onload = (evt) => {
      lastUploadedCsvContent = evt.target.result;
      if (txtRaw) txtRaw.value = lastUploadedCsvContent;
      hideAlert();
    };
    reader.readAsText(file);
  }

  function showAlert(msg, isError = false) {
    if (!alertMsg) return;
    alertMsg.style.display = "block";
    alertMsg.style.background = isError ? "var(--er)" : "var(--ok)";
    alertMsg.style.color = "#fff";
    alertMsg.textContent = msg;
  }

  function hideAlert() {
    if (alertMsg) alertMsg.style.display = "none";
  }

  if (btnSample) {
    btnSample.addEventListener("click", () => {
      window.location.href = "/api/sample-csv";
    });
  }

  if (btnLoadProfile) {
    btnLoadProfile.addEventListener("click", async () => {
      try {
        btnLoadProfile.textContent = "Loading...";
        const res = await fetch("/api/sample-csv");
        const text = await res.text();
        lastUploadedCsvContent = text;
        lastUploadedFilename = "sample_test_profile.csv";
        if (txtRaw) txtRaw.value = text;
        if (fileNameDisplay) {
          fileNameDisplay.innerHTML = `<strong>Loaded:</strong> sample_test_profile.csv (Default Predefined Columns)`;
        }
        btnLoadProfile.textContent = "⚡ Load Sample Test Profile";
        runPrediction();
      } catch (err) {
        btnLoadProfile.textContent = "⚡ Load Sample Test Profile";
        showAlert("Failed to load sample template: " + err.message, true);
      }
    });
  }

  if (btnToggleEditor && editorContainer) {
    btnToggleEditor.addEventListener("click", () => {
      const isHidden = (editorContainer.style.display === "none");
      editorContainer.style.display = isHidden ? "block" : "none";
    });
  }

  if (btnRun) {
    btnRun.addEventListener("click", () => runPrediction());
  }

  async function runPrediction() {
    const content = (txtRaw && txtRaw.value.trim()) ? txtRaw.value : lastUploadedCsvContent;
    if (!content || !content.trim()) {
      showAlert("Please choose a CSV file or enter data into the editor before predicting.", true);
      return;
    }

    const autoLoadTwin = $("chk-auto-load-twin") ? $("chk-auto-load-twin").checked : false;
    btnRun.textContent = "Predicting...";
    btnRun.disabled = true;

    try {
      const res = await fetch("/api/predict-custom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: lastUploadedFilename,
          content: content,
          load_into_twin: autoLoadTwin,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Error predicting SOC on custom data.");
      }

      const stats = data.stats;
      if ($("kpi-sample-count")) $("kpi-sample-count").textContent = `${stats.sample_count} rows`;
      if ($("kpi-duration")) $("kpi-duration").textContent = `${stats.duration_s} s (${stats.duration_min} min)`;
      if ($("kpi-dt")) $("kpi-dt").textContent = `${stats.dt_s} s`;
      if ($("kpi-v-range")) $("kpi-v-range").textContent = `${stats.v_min} - ${stats.v_max} V (avg ${stats.v_mean} V)`;
      if ($("kpi-i-range")) $("kpi-i-range").textContent = `${stats.i_min} - ${stats.i_max} A (avg ${stats.i_mean} A)`;
      if ($("kpi-soc-start")) $("kpi-soc-start").textContent = `${stats.soc_start} %`;
      if ($("kpi-soc-end")) $("kpi-soc-end").textContent = `${stats.soc_end} %`;
      if ($("kpi-soc-delta")) $("kpi-soc-delta").textContent = `${stats.soc_delta} % points`;

      const gtBox = $("gt-accuracy-box");
      const legendTrueSoc = $("legend-item-true-soc");
      if (stats.has_ground_truth) {
        if (gtBox) gtBox.style.display = "block";
        if (legendTrueSoc) legendTrueSoc.style.display = "flex";
        if ($("kpi-ml-mae")) $("kpi-ml-mae").textContent = `${stats.ml_mae} %`;
        if ($("kpi-ml-rmse")) $("kpi-ml-rmse").textContent = `${stats.ml_rmse} %`;
        if ($("kpi-ml-max")) $("kpi-ml-max").textContent = `${stats.ml_max_error} %`;
      } else {
        if (gtBox) gtBox.style.display = "none";
        if (legendTrueSoc) legendTrueSoc.style.display = "none";
      }

      renderCustomCharts(data.series, stats.has_ground_truth);
      renderCustomPreviewTable(data.preview);

      if (resultsSec) resultsSec.style.display = "block";

      refreshCycleList();

      showAlert(`Prediction complete for ${stats.sample_count} samples!` + (autoLoadTwin ? " (Loaded into Live Digital Twin)" : ""), false);
    } catch (err) {
      showAlert(err.message, true);
    } finally {
      btnRun.textContent = "🚀 Predict SOC";
      btnRun.disabled = false;
    }
  }

  if (btnExport) {
    btnExport.addEventListener("click", () => {
      window.location.href = "/api/export-custom";
    });
  }

  if (btnReplayTwin) {
    btnReplayTwin.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/load-custom-twin", { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to load into twin.");

        const tabBtns = document.querySelectorAll("nav.tab-nav button");
        tabBtns.forEach((b) => {
          if (b.getAttribute("data-tab") === "tab-live") b.click();
        });
        resetCharts();
        await refreshCycleList();
        $("cycle-badge").textContent = `Custom: ${lastUploadedFilename}`;
      } catch (err) {
        showAlert(err.message, true);
      }
    });
  }
}

function renderCustomCharts(series, hasGroundTruth) {
  const commonOpts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        grid: { color: "rgba(100, 116, 139, 0.15)" },
        ticks: { font: { size: 10 }, color: "#64748b" },
      },
      y: {
        grid: { color: "rgba(100, 116, 139, 0.15)" },
        ticks: { font: { size: 10 }, color: "#64748b" },
      },
    },
  };

  const ctxSoc = $("chart-custom-soc").getContext("2d");
  const socDatasets = [
    { label: "ML Predicted SOC", data: series.predicted_soc, borderColor: "#16a34a", borderWidth: 2, pointRadius: 0 },
    { label: "Coulomb Counting Baseline", data: series.coulomb_soc, borderColor: "#dc2626", borderWidth: 1.5, borderDash: [3, 3], pointRadius: 0 },
  ];
  if (hasGroundTruth && series.true_soc) {
    socDatasets.unshift({
      label: "True Reference SOC",
      data: series.true_soc,
      borderColor: "#1f2937",
      borderWidth: 2,
      pointRadius: 0,
    });
  }

  if (chartCustomSoc) chartCustomSoc.destroy();
  chartCustomSoc = new Chart(ctxSoc, {
    type: "line",
    data: {
      labels: series.time.map((t) => `${t}s`),
      datasets: socDatasets,
    },
    options: {
      ...commonOpts,
      scales: {
        ...commonOpts.scales,
        y: { min: -2, max: 102, title: { display: true, text: "SOC (%)", font: { size: 11 } } },
      },
    },
  });

  const ctxVi = $("chart-custom-vi").getContext("2d");
  if (chartCustomVi) chartCustomVi.destroy();
  chartCustomVi = new Chart(ctxVi, {
    type: "line",
    data: {
      labels: series.time.map((t) => `${t}s`),
      datasets: [
        { label: "Terminal Voltage (V)", data: series.voltage, borderColor: "#2563eb", borderWidth: 2, pointRadius: 0 },
        { label: "V_smooth (V)", data: series.v_smooth, borderColor: "#9333ea", borderWidth: 1.5, borderDash: [2, 2], pointRadius: 0 },
      ],
    },
    options: {
      ...commonOpts,
      scales: {
        ...commonOpts.scales,
        y: { title: { display: true, text: "Voltage (V)", font: { size: 11 } } },
      },
    },
  });

  const ctxCurr = $("chart-custom-curr").getContext("2d");
  if (chartCustomCurr) chartCustomCurr.destroy();
  chartCustomCurr = new Chart(ctxCurr, {
    type: "line",
    data: {
      labels: series.time.map((t) => `${t}s`),
      datasets: [
        { label: "Current (A)", data: series.current, borderColor: "#d97706", borderWidth: 1.5, pointRadius: 0 },
      ],
    },
    options: {
      ...commonOpts,
      scales: {
        ...commonOpts.scales,
        y: { title: { display: true, text: "Current (A)", font: { size: 11 } } },
      },
    },
  });
}

function renderCustomPreviewTable(previewRows) {
  const tbody = $("table-custom-preview").querySelector("tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  previewRows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>#${row.row}</strong></td>
      <td>${row.time_s} s</td>
      <td>${row.voltage} V</td>
      <td>${row.current} A</td>
      <td>${row.v_smooth} V</td>
      <td style="color:var(--ok);font-weight:700;">${row.predicted_soc} %</td>
      <td>${row.true_soc === "N/A" ? '<span style="color:var(--mut);">N/A</span>' : `<strong>${row.true_soc} %</strong>`}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function refreshCycleList() {
  try {
    const res = await fetch("/api/cycles");
    const data = await res.json();
    const sel = $("sel-cycle");
    if (!sel) return;
    const currentVal = sel.value;
    sel.innerHTML = "";

    if (data.custom && data.custom.length > 0) {
      data.custom.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.id;
        opt.textContent = `★ ${c.name} (${c.rate_a} A)`;
        if (data.current_cycle === c.id || currentVal === c.id) opt.selected = true;
        sel.appendChild(opt);
      });
    }

    if (data.held_out) {
      data.held_out.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.id;
        opt.textContent = c.name;
        if (data.current_cycle === c.id && (!data.custom || data.current_cycle !== "custom")) opt.selected = true;
        sel.appendChild(opt);
      });
    }
  } catch (err) {
    console.error("Error refreshing cycle list:", err);
  }
}
