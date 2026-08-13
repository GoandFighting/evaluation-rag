const state = {
  datasetId: null,
  metrics: [],
  detectedFields: [],
  effectiveFields: [],
  adapterName: "",
  adapters: [],
  selectedFile: null,
  uploadSequence: 0,
  healthcheckSequence: 0,
  results: [],
};
const dimensions = ["end_to_end", "retrieval", "generation"];

const fileInput = document.querySelector("#file-input");
const fileLabel = document.querySelector("#file-label");
const uploadStatus = document.querySelector("#upload-status");
const metricsSection = document.querySelector("#metrics-section");
const resultsSection = document.querySelector("#results-section");
const runButton = document.querySelector("#run-button");
const adapterSelect = document.querySelector("#adapter-select");
const adapterDescription = document.querySelector("#adapter-description");
const dropZone = document.querySelector("#drop-zone");
const metricDashboard = document.querySelector("#metric-dashboard");
const dashboardClose = document.querySelector("#dashboard-close");
let dragDepth = 0;

loadAdapters();

adapterSelect.addEventListener("change", () => {
  state.adapterName = adapterSelect.value;
  const adapter = state.adapters.find((item) => item.name === state.adapterName);
  adapterDescription.textContent = adapter
    ? `${adapter.description}${adapter.available ? "" : "（当前不可用）"}`
    : "离线模式不会调用外部 RAG。";
  state.datasetId = null;
  metricsSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
  if (adapter) void checkAdapterHealth(adapter);
  if (state.selectedFile) void uploadSelectedFile();
});

dashboardClose.addEventListener("click", () => closeDashboard());
metricDashboard.addEventListener("click", (event) => {
  if (event.target === metricDashboard) closeDashboard();
});

fileInput.addEventListener("change", () => {
  selectFile(fileInput.files[0] || null);
});

for (const eventName of ["dragenter", "dragover", "dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
}

dropZone.addEventListener("dragenter", (event) => {
  if (!hasDraggedFiles(event)) return;
  dragDepth += 1;
  dropZone.classList.add("is-dragover");
});

dropZone.addEventListener("dragover", (event) => {
  if (hasDraggedFiles(event)) event.dataTransfer.dropEffect = "copy";
});

dropZone.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropZone.classList.remove("is-dragover");
});

dropZone.addEventListener("drop", (event) => {
  dragDepth = 0;
  dropZone.classList.remove("is-dragover");
  const files = [...(event.dataTransfer?.files || [])];
  if (files.length !== 1) {
    selectFile(null, "每次只能拖入一个 JSON 或 JSONL 文件。");
    return;
  }
  selectFile(files[0]);
});

async function uploadSelectedFile() {
  const file = state.selectedFile;
  if (!file) return;
  const sequence = ++state.uploadSequence;
  state.datasetId = null;
  state.metrics = [];
  state.results = [];
  runButton.disabled = true;
  dropZone.classList.add("is-uploading");
  uploadStatus.textContent = "正在上传并检查数据…";
  const body = new FormData();
  body.append("file", file);
  try {
    const query = state.adapterName
      ? `?adapter_name=${encodeURIComponent(state.adapterName)}`
      : "";
    const response = await fetch(`/api/datasets/upload${query}`, { method: "POST", body });
    const data = await readResponse(response);
    if (sequence !== state.uploadSequence) return;
    state.datasetId = data.dataset_id;
    state.metrics = data.metrics;
    state.detectedFields = data.detected_fields;
    state.effectiveFields = data.effective_fields || data.detected_fields;
    renderEvaluationPreparation(data);
    metricsSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    metricsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    uploadStatus.textContent = `${file.name} 已就绪`;
  } catch (error) {
    if (sequence !== state.uploadSequence) return;
    state.datasetId = null;
    uploadStatus.textContent = error.message;
  } finally {
    if (sequence === state.uploadSequence) {
      dropZone.classList.remove("is-uploading");
      runButton.disabled = !state.metrics.some((metric) => metric.implemented);
    }
  }
}

function selectFile(file, explicitError = "") {
  state.datasetId = null;
  metricsSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
  if (!file && !explicitError) {
    state.uploadSequence += 1;
    state.selectedFile = null;
    fileInput.value = "";
    fileLabel.textContent = "点击选择或拖入数据文件";
    dropZone.classList.remove("is-uploading");
    runButton.disabled = true;
    uploadStatus.textContent = "";
    return;
  }
  const error = explicitError || validateFile(file);
  if (error) {
    state.uploadSequence += 1;
    state.selectedFile = null;
    fileInput.value = "";
    fileLabel.textContent = "点击选择或拖入数据文件";
    dropZone.classList.remove("is-uploading");
    runButton.disabled = true;
    uploadStatus.textContent = error;
    return;
  }
  state.selectedFile = file;
  fileInput.value = "";
  fileLabel.textContent = file.name;
  uploadStatus.textContent = `已选择 ${file.name}，准备上传…`;
  void uploadSelectedFile();
}

function validateFile(file) {
  if (!file) return "";
  const filename = file.name.toLowerCase();
  if (!filename.endsWith(".json") && !filename.endsWith(".jsonl")) {
    return "仅支持 .json 和 .jsonl 文件。";
  }
  if (file.size > 20 * 1024 * 1024) return "文件不能超过 20 MB。";
  return "";
}

function hasDraggedFiles(event) {
  return [...(event.dataTransfer?.types || [])].includes("Files");
}

runButton.addEventListener("click", async () => {
  const selected = state.metrics.filter((metric) => metric.implemented).map((metric) => metric.name);
  if (!selected.length) {
    document.querySelector("#run-status").textContent = "当前没有已实现的指标。";
    return;
  }
  const status = document.querySelector("#run-status");
  setBusy(runButton, status, true, "正在评测…");
  try {
    const response = await fetch("/api/evaluations/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: state.datasetId,
        metric_names: selected,
        adapter_name: state.adapterName || null,
      }),
    });
    const data = await readResponse(response);
    renderResults(data);
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    status.textContent = `完成，共处理 ${data.sample_count} 条样本`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    runButton.disabled = !state.datasetId;
  }
});

function renderEvaluationPreparation(data) {
  const source = data.adapter ? ` · 回答来源：${data.adapter.label}` : " · 离线数据";
  document.querySelector("#dataset-summary").textContent = `${data.filename} · ${data.sample_count} 条样本${source}`;
  document.querySelector("#field-list").innerHTML = data.detected_fields.length
    ? `<span class="field-label">检测到的评测字段</span>${data.detected_fields.map((field) => `<code>${escapeHtml(field)}</code>`).join("")}`
    : `<span class="warning">未检测到可用于现有指标的字段</span>`;
  if (data.adapter) {
    const supplied = data.effective_fields.filter((field) => !data.detected_fields.includes(field));
    document.querySelector("#field-list").insertAdjacentHTML(
      "beforeend",
      `<span class="field-label adapter-fields">Adapter 将补充</span>${supplied.map((field) => `<code>${escapeHtml(field)}</code>`).join("")}`,
    );
  }

  const implemented = data.metrics.filter((metric) => metric.implemented);
  const labels = implemented.map((metric) => `<span>${escapeHtml(metric.label)}</span>`).join("");
  document.querySelector("#automatic-metric-summary").innerHTML = implemented.length
    ? `<div><strong>将自动评测 ${implemented.length} 个已实现指标</strong><small>字段不足的样本会显示为“不适用”</small></div><div class="automatic-metric-list">${labels}</div>`
    : '<div class="dimension-note"><strong>暂无已实现指标</strong><span>请先完成至少一个指标的后端实现。</span></div>';
  runButton.disabled = implemented.length === 0;
}

function renderResults(data) {
  state.results = data.results;
  renderInvocations(data.invocations || []);
  const summaryHost = document.querySelector("#result-summary");
  summaryHost.innerHTML = "";
  for (const dimension of dimensions) {
    const summaries = data.summary.filter((item) => item.dimension === dimension);
    const block = document.createElement("section");
    block.className = "result-dimension";
    const dimensionMetric = state.metrics.find((item) => item.dimension === dimension);
    block.innerHTML = `<h3>${dimensionMetric?.dimension_label || dimension}</h3>${summaries.length ? '<div class="score-grid"></div>' : `<div class="dimension-note">${dimensionEmptyMessage(dimension)}</div>`}`;
    if (!summaries.length) {
      summaryHost.appendChild(block);
      continue;
    }
    const grid = block.querySelector(".score-grid");
    for (const item of summaries) {
      const percent = item.average === null ? 0 : Math.round(item.average * 100);
      const card = document.createElement("button");
      card.type = "button";
      card.className = "score-card";
      card.setAttribute("aria-label", `查看${item.metric_label}样本明细`);
      card.innerHTML = `<div><strong>${escapeHtml(item.metric_label)}</strong><span class="score-value">${item.average === null ? "—" : percent + "%"}</span></div><div class="bar"><i style="width:${percent}%"></i></div><small class="score-card-hint">点击查看样本明细 →</small>`;
      card.addEventListener("click", () => openMetricDashboard(item));
      grid.appendChild(card);
    }
    summaryHost.appendChild(block);
  }
}

function openMetricDashboard(summary) {
  const metric = state.metrics.find((item) => item.name === summary.metric_name);
  const results = state.results.filter((item) => item.metric_name === summary.metric_name);
  const percent = summary.average === null ? 0 : Math.round(summary.average * 100);
  document.querySelector("#dashboard-dimension").textContent = summary.dimension_label;
  document.querySelector("#dashboard-title").textContent = summary.metric_label;
  document.querySelector("#dashboard-description").textContent = metric?.description || "";
  document.querySelector("#dashboard-average").textContent = summary.average === null ? "—" : `${percent}%`;
  document.querySelector("#dashboard-bar").style.width = `${percent}%`;
  document.querySelector("#dashboard-rows").innerHTML = results.map((item) => `<tr><td><code>${escapeHtml(item.sample_id)}</code></td><td><span class="status ${item.status}">${statusLabel(item.status)}</span></td><td>${item.score === null ? "—" : item.score.toFixed(3)}</td><td>${escapeHtml(item.reason)}</td></tr>`).join("");
  if (typeof metricDashboard.showModal === "function") metricDashboard.showModal();
  else metricDashboard.setAttribute("open", "");
}

function closeDashboard() {
  if (typeof metricDashboard.close === "function") metricDashboard.close();
  else metricDashboard.removeAttribute("open");
}

function dimensionEmptyMessage(dimension) {
  if (dimension !== "end_to_end" && !state.effectiveFields.includes("chunks.content")) {
    return "当前数据集没有 chunks，必须提供检索片段后才能进行该维度评测。";
  }
  return "当前维度没有已实现的指标。";
}

function renderInvocations(invocations) {
  const host = document.querySelector("#invocation-summary");
  if (!invocations.length) {
    host.innerHTML = "";
    return;
  }
  const success = invocations.filter((item) => item.status === "success").length;
  const failed = invocations.filter((item) => item.status === "failed").length;
  const skipped = invocations.length - success - failed;
  const rows = invocations.map((item) => {
    const answer = item.answer
      ? `<details><summary>查看回答</summary><pre>${escapeHtml(item.answer)}</pre></details>`
      : escapeHtml(item.reason);
    return `<tr><td><code>${escapeHtml(item.sample_id)}</code></td><td><span class="status ${item.status}">${statusLabel(item.status)}</span></td><td>${item.latency_ms ?? "—"}</td><td>${item.chunk_count}</td><td>${answer}</td></tr>`;
  }).join("");
  host.innerHTML = `<div class="invocation-card"><strong>目标 RAG 调用</strong><span>${success} 成功 · ${failed} 失败 · ${skipped} 跳过</span></div><details class="details invocation-details"><summary>调用明细</summary><div class="table-wrap"><table><thead><tr><th>样本</th><th>状态</th><th>耗时 (ms)</th><th>片段</th><th>回答 / 原因</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
}

async function loadAdapters() {
  try {
    const response = await fetch("/api/rag-adapters");
    const adapters = await readResponse(response);
    state.adapters = adapters;
    for (const adapter of adapters) {
      const option = document.createElement("option");
      option.value = adapter.name;
      option.textContent = `${adapter.label}${adapter.available ? "" : "（不可用）"}`;
      option.disabled = !adapter.available;
      adapterSelect.appendChild(option);
    }
  } catch (error) {
    adapterDescription.textContent = `Adapter 列表加载失败：${error.message}`;
  }
}

async function checkAdapterHealth(adapter) {
  const sequence = ++state.healthcheckSequence;
  adapterDescription.textContent = `${adapter.description} 正在检查连接…`;
  try {
    const response = await fetch(
      `/api/rag-adapters/${encodeURIComponent(adapter.name)}/healthcheck`,
      { method: "POST" },
    );
    const health = await readResponse(response);
    if (sequence !== state.healthcheckSequence || state.adapterName !== adapter.name) return;
    adapterDescription.textContent = `${adapter.description} ${health.ok ? "连接正常" : `连接失败：${health.message}`}`;
  } catch (error) {
    if (sequence !== state.healthcheckSequence || state.adapterName !== adapter.name) return;
    adapterDescription.textContent = `${adapter.description} 连接检查失败：${error.message}`;
  }
}

function setBusy(button, status, busy, message) {
  button.disabled = busy;
  status.textContent = message;
}

async function readResponse(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "请求失败");
  return data;
}

function statusLabel(value) {
  return { success: "成功", not_applicable: "不适用", not_configured: "模型未配置", failed: "失败", not_implemented: "待实现" }[value] || value;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
