const state = { datasetId: null, metrics: [], detectedFields: [] };
const dimensions = ["end_to_end", "retrieval", "generation"];

const fileInput = document.querySelector("#file-input");
const fileLabel = document.querySelector("#file-label");
const uploadButton = document.querySelector("#upload-button");
const uploadStatus = document.querySelector("#upload-status");
const metricsSection = document.querySelector("#metrics-section");
const resultsSection = document.querySelector("#results-section");
const runButton = document.querySelector("#run-button");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileLabel.textContent = file ? file.name : "点击选择或拖入数据文件";
  uploadButton.disabled = !file;
});

uploadButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  setBusy(uploadButton, uploadStatus, true, "正在解析字段…");
  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/api/datasets/upload", { method: "POST", body });
    const data = await readResponse(response);
    state.datasetId = data.dataset_id;
    state.metrics = data.metrics;
    state.detectedFields = data.detected_fields;
    renderMetrics(data);
    metricsSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    metricsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    uploadStatus.textContent = "解析完成";
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    uploadButton.disabled = false;
  }
});

runButton.addEventListener("click", async () => {
  const selected = [...document.querySelectorAll("[data-metric]:checked")].map((item) => item.dataset.metric);
  if (!selected.length) {
    document.querySelector("#run-status").textContent = "请至少选择一个可运行指标。";
    return;
  }
  const status = document.querySelector("#run-status");
  setBusy(runButton, status, true, "正在评测…");
  try {
    const response = await fetch("/api/evaluations/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: state.datasetId, metric_names: selected }),
    });
    const data = await readResponse(response);
    renderResults(data);
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    status.textContent = `完成，共处理 ${data.sample_count} 条样本`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
});

function renderMetrics(data) {
  document.querySelector("#dataset-summary").textContent = `${data.filename} · ${data.sample_count} 条样本`;
  document.querySelector("#field-list").innerHTML = data.detected_fields.length
    ? `<span class="field-label">检测到的评测字段</span>${data.detected_fields.map((field) => `<code>${escapeHtml(field)}</code>`).join("")}`
    : `<span class="warning">未检测到可用于现有指标的字段</span>`;

  const host = document.querySelector("#metric-groups");
  host.innerHTML = "";
  for (const dimension of dimensions) {
    const metrics = data.metrics.filter((metric) => metric.dimension === dimension);
    const section = document.createElement("section");
    section.className = "metric-group";
    section.innerHTML = `<div class="group-heading"><h3>${metrics[0].dimension_label}</h3><span>${metrics.filter((m) => m.runnable).length}/${metrics.length} 可运行</span></div>${dimensionNotice(dimension)}<div class="metric-grid"></div>`;
    const grid = section.querySelector(".metric-grid");
    for (const metric of metrics) grid.appendChild(metricCard(metric));
    host.appendChild(section);
  }
}

function metricCard(metric) {
  const node = document.querySelector("#metric-card-template").content.cloneNode(true);
  const input = node.querySelector("input");
  input.dataset.metric = metric.name;
  input.disabled = !metric.runnable;
  input.checked = metric.runnable;
  node.querySelector("strong").textContent = metric.label;
  node.querySelector(".description").textContent = metric.description;
  node.querySelector(".requirements").textContent = `需要：${metric.required_fields.join(" + ")}`;
  const stateNode = node.querySelector(".metric-state");
  if (!metric.field_ready) {
    stateNode.textContent = "字段不足";
    stateNode.className += " unavailable";
  } else if (!metric.implemented) {
    stateNode.textContent = "待实现";
    stateNode.className += " pending";
  } else {
    stateNode.textContent = `${metric.eligible_samples}/${metric.total_samples} 可评`;
    stateNode.className += " ready";
  }
  return node;
}

function renderResults(data) {
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
      const card = document.createElement("article");
      card.className = "score-card";
      card.innerHTML = `<div><strong>${escapeHtml(item.metric_label)}</strong><span>${item.average === null ? "—" : percent + "%"}</span></div><div class="bar"><i style="width:${percent}%"></i></div><small>${item.success_count} 成功 · ${item.not_applicable_count} 不适用 · ${item.failed_count} 失败</small>`;
      grid.appendChild(card);
    }
    summaryHost.appendChild(block);
  }
  document.querySelector("#result-rows").innerHTML = data.results.map((item) => `<tr><td><code>${escapeHtml(item.sample_id)}</code></td><td>${escapeHtml(item.dimension_label)}</td><td>${escapeHtml(item.metric_label)}</td><td><span class="status ${item.status}">${statusLabel(item.status)}</span></td><td>${item.score === null ? "—" : item.score.toFixed(3)}</td><td>${escapeHtml(item.reason)}</td></tr>`).join("");
}

function dimensionNotice(dimension) {
  if (dimension === "end_to_end" || state.detectedFields.includes("chunks.content")) return "";
  return '<div class="dimension-note"><strong>缺少 chunks</strong><span>检索模块和生成模块必须包含检索片段，当前数据集无法进行该维度评测。</span></div>';
}

function dimensionEmptyMessage(dimension) {
  if (dimension !== "end_to_end" && !state.detectedFields.includes("chunks.content")) {
    return "当前数据集没有 chunks，必须提供检索片段后才能进行该维度评测。";
  }
  return "本次没有选择可运行指标；请检查字段覆盖率或指标实现状态。";
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
  return { success: "成功", not_applicable: "不适用", failed: "失败", not_implemented: "待实现" }[value] || value;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
