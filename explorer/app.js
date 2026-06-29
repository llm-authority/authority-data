const state = {
  manifest: null,
  dataset: "",
  config: "",
  split: "train",
  q: "",
  label: "",
  actionType: "",
  offset: 0,
  limit: 50,
  rows: [],
  selectedRow: null,
  summary: null,
  filteredRows: 0,
  totalRows: 0,
};

const els = {
  datasetSelect: document.querySelector("#datasetSelect"),
  configSelect: document.querySelector("#configSelect"),
  splitSelect: document.querySelector("#splitSelect"),
  splitCount: document.querySelector("#splitCount"),
  searchInput: document.querySelector("#searchInput"),
  labelFilter: document.querySelector("#labelFilter"),
  actionFilter: document.querySelector("#actionFilter"),
  reloadButton: document.querySelector("#reloadButton"),
  stats: document.querySelector("#stats"),
  rowsBody: document.querySelector("#rowsBody"),
  resultCount: document.querySelector("#resultCount"),
  pageInfo: document.querySelector("#pageInfo"),
  prevButton: document.querySelector("#prevButton"),
  nextButton: document.querySelector("#nextButton"),
  detailKicker: document.querySelector("#detailKicker"),
  detailTitle: document.querySelector("#detailTitle"),
  detailLabel: document.querySelector("#detailLabel"),
  detailSections: document.querySelector("#detailSections"),
};

const preferredFields = [
  "sample_id",
  "source_sample_id",
  "benchmark",
  "authority_specification_type",
  "condition",
  "attack_type",
  "attack_placement",
  "decision_kind",
  "action_type",
  "tool",
  "required_tool",
  "adversarial_tool",
  "label",
  "permitted",
  "prohibited",
  "task_goal",
  "external_context",
  "target_action",
  "prompt_v1",
  "prompt_v2",
  "benchmark_prompt",
];

function option(value, label = value) {
  const node = document.createElement("option");
  node.value = value;
  node.textContent = label;
  return node;
}

function selectedDataset() {
  return state.manifest.datasets.find((dataset) => dataset.id === state.dataset);
}

function selectedConfig() {
  const dataset = selectedDataset();
  return dataset?.configs.find((config) => config.id === state.config);
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function compact(value, fallback = "") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return String(value);
}

function rowType(row) {
  return compact(row.action_type || row.decision_kind || row.authority_specification_type, "-");
}

function rowTool(row) {
  return compact(row.tool || row.required_tool || row.adversarial_tool, "-");
}

function setOptions(select, values, selectedValue, emptyLabel = null) {
  select.innerHTML = "";
  if (emptyLabel) {
    select.append(option("", emptyLabel));
  }
  values.forEach((value) => select.append(option(value)));
  select.value = values.includes(selectedValue) || selectedValue === "" ? selectedValue : values[0] || "";
}

function renderDatasetControls() {
  const datasetIds = state.manifest.datasets.map((dataset) => dataset.id);
  setOptions(els.datasetSelect, datasetIds, state.dataset);

  const dataset = selectedDataset();
  const configIds = dataset.configs.map((config) => config.id);
  setOptions(els.configSelect, configIds, state.config);

  const config = selectedConfig();
  const splitIds = Object.keys(config?.splits || {});
  setOptions(els.splitSelect, splitIds, state.split);

  const rows = config?.splits?.[state.split]?.rows || 0;
  els.splitCount.textContent = `${formatNumber(rows)} rows in ${state.split}`;
}

function renderFilters() {
  const labels = state.summary?.labels || [];
  const actions = state.summary?.actionTypes || [];
  setOptions(els.labelFilter, labels, state.label, "All labels");
  setOptions(els.actionFilter, actions, state.actionType, "All action types");
}

function renderStats() {
  const labelCounts = state.summary?.labelCounts || {};
  const yes = labelCounts.yes || 0;
  const no = labelCounts.no || 0;
  const other = Object.entries(labelCounts)
    .filter(([label]) => label !== "yes" && label !== "no")
    .reduce((sum, [, count]) => sum + count, 0);

  const stats = [
    ["Filtered", state.filteredRows],
    ["Total", state.totalRows],
    ["Yes", yes],
    ["No / Other", no + other],
  ];

  els.stats.innerHTML = "";
  stats.forEach(([label, value]) => {
    const node = document.createElement("div");
    node.className = "stat";
    node.innerHTML = `<span class="meta-line">${label}</span><b>${formatNumber(value)}</b>`;
    els.stats.append(node);
  });
}

function renderRows() {
  els.rowsBody.innerHTML = "";
  if (state.rows.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.className = "empty";
    cell.textContent = "No rows";
    row.append(cell);
    els.rowsBody.append(row);
    return;
  }

  state.rows.forEach((dataRow, index) => {
    const tr = document.createElement("tr");
    const isSelected = state.selectedRow && state.selectedRow.sample_id === dataRow.sample_id;
    tr.className = isSelected ? "selected" : "";
    const sampleCell = document.createElement("td");
    sampleCell.title = compact(dataRow.sample_id || dataRow.source_sample_id);
    sampleCell.textContent = compact(dataRow.sample_id || dataRow.source_sample_id, "-");

    const labelCell = document.createElement("td");
    const labelBadge = document.createElement("span");
    labelBadge.className = `badge ${compact(dataRow.label)}`;
    labelBadge.textContent = compact(dataRow.label, "-");
    labelCell.append(labelBadge);

    const typeCell = document.createElement("td");
    typeCell.title = rowType(dataRow);
    typeCell.textContent = rowType(dataRow);

    const toolCell = document.createElement("td");
    toolCell.title = rowTool(dataRow);
    toolCell.textContent = rowTool(dataRow);

    const taskCell = document.createElement("td");
    taskCell.title = compact(dataRow.task_goal);
    taskCell.textContent = compact(dataRow.task_goal, "-");

    tr.append(sampleCell, labelCell, typeCell, toolCell, taskCell);
    tr.addEventListener("click", () => {
      state.selectedRow = dataRow;
      renderRows();
      renderDetail();
    });
    if (!state.selectedRow && index === 0) {
      state.selectedRow = dataRow;
    }
    els.rowsBody.append(tr);
  });
}

function orderedFields(row) {
  const seen = new Set();
  const ordered = [];
  for (const field of preferredFields) {
    if (Object.prototype.hasOwnProperty.call(row, field)) {
      ordered.push(field);
      seen.add(field);
    }
  }
  Object.keys(row)
    .sort()
    .forEach((field) => {
      if (!seen.has(field)) {
        ordered.push(field);
      }
    });
  return ordered;
}

function renderDetail() {
  const row = state.selectedRow;
  els.detailSections.innerHTML = "";
  if (!row) {
    els.detailKicker.textContent = "";
    els.detailTitle.textContent = "Select a row";
    els.detailLabel.textContent = "";
    return;
  }

  els.detailKicker.textContent = compact(row.source_sample_id || row.benchmark || state.dataset);
  els.detailTitle.textContent = compact(row.sample_id, "Row");
  els.detailLabel.className = `badge ${compact(row.label)}`;
  els.detailLabel.textContent = compact(row.label, "-");

  for (const field of orderedFields(row)) {
    const node = document.createElement("section");
    node.className = "field";
    const name = document.createElement("div");
    name.className = "field-name";
    name.textContent = field;
    const value = document.createElement("pre");
    value.className = "field-value";
    value.textContent = compact(row[field], "(empty)");
    node.append(name, value);
    els.detailSections.append(node);
  }
}

function renderPagination() {
  const start = state.filteredRows === 0 ? 0 : state.offset + 1;
  const end = Math.min(state.offset + state.limit, state.filteredRows);
  els.resultCount.textContent = `${formatNumber(start)}-${formatNumber(end)} of ${formatNumber(state.filteredRows)}`;
  els.pageInfo.textContent = `Page ${Math.floor(state.offset / state.limit) + 1}`;
  els.prevButton.disabled = state.offset === 0;
  els.nextButton.disabled = state.offset + state.limit >= state.filteredRows;
}

function queryParams() {
  const params = new URLSearchParams({
    dataset: state.dataset,
    config: state.config,
    split: state.split,
    offset: String(state.offset),
    limit: String(state.limit),
  });
  if (state.q) params.set("q", state.q);
  if (state.label) params.set("label", state.label);
  if (state.actionType) params.set("action_type", state.actionType);
  return params;
}

async function loadRows({ keepSelected = false } = {}) {
  const response = await fetch(`/api/rows?${queryParams()}`);
  if (!response.ok) {
    throw new Error(`Failed to load rows: ${response.status}`);
  }
  const payload = await response.json();
  state.rows = payload.rows;
  state.filteredRows = payload.filteredRows;
  state.totalRows = payload.totalRows;
  state.summary = payload.summary;
  if (!keepSelected) {
    state.selectedRow = state.rows[0] || null;
  }
  renderDatasetControls();
  renderFilters();
  renderStats();
  renderPagination();
  renderRows();
  renderDetail();
}

async function loadManifest() {
  const response = await fetch("/api/manifest");
  if (!response.ok) {
    throw new Error(`Failed to load manifest: ${response.status}`);
  }
  state.manifest = await response.json();
  const firstDataset = state.manifest.datasets[0];
  state.dataset = firstDataset?.id || "";
  state.config = firstDataset?.configs[0]?.id || "";
  state.split = Object.keys(firstDataset?.configs[0]?.splits || {})[0] || "train";
  renderDatasetControls();
  await loadRows();
}

function resetPage() {
  state.offset = 0;
  state.selectedRow = null;
}

function bindEvents() {
  els.datasetSelect.addEventListener("change", async () => {
    state.dataset = els.datasetSelect.value;
    state.config = selectedDataset().configs[0].id;
    state.split = Object.keys(selectedDataset().configs[0].splits)[0];
    resetPage();
    await loadRows();
  });

  els.configSelect.addEventListener("change", async () => {
    state.config = els.configSelect.value;
    state.split = Object.keys(selectedConfig().splits)[0];
    resetPage();
    await loadRows();
  });

  els.splitSelect.addEventListener("change", async () => {
    state.split = els.splitSelect.value;
    resetPage();
    await loadRows();
  });

  els.labelFilter.addEventListener("change", async () => {
    state.label = els.labelFilter.value;
    resetPage();
    await loadRows();
  });

  els.actionFilter.addEventListener("change", async () => {
    state.actionType = els.actionFilter.value;
    resetPage();
    await loadRows();
  });

  els.searchInput.addEventListener("input", () => {
    window.clearTimeout(els.searchInput._timer);
    els.searchInput._timer = window.setTimeout(async () => {
      state.q = els.searchInput.value;
      resetPage();
      await loadRows();
    }, 180);
  });

  els.reloadButton.addEventListener("click", async () => {
    await loadRows();
  });

  els.prevButton.addEventListener("click", async () => {
    state.offset = Math.max(state.offset - state.limit, 0);
    state.selectedRow = null;
    await loadRows();
  });

  els.nextButton.addEventListener("click", async () => {
    state.offset += state.limit;
    state.selectedRow = null;
    await loadRows();
  });
}

bindEvents();
loadManifest().catch((error) => {
  els.rowsBody.innerHTML = "";
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 5;
  cell.className = "empty";
  cell.textContent = error.message;
  row.append(cell);
  els.rowsBody.append(row);
});
