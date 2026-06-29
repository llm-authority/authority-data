const fs = require("fs");
const http = require("http");
const path = require("path");
const url = require("url");

const ROOT = path.resolve(__dirname, "..");
const DATA_ROOT = path.join(ROOT, "authority_data", "data");
const PUBLIC_ROOT = path.join(ROOT, "explorer");
const DEFAULT_PORT = Number(process.env.PORT || 5173);
const HOST = process.env.HOST || "0.0.0.0";
const MAX_LIMIT = 200;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

let manifestCache = null;
const fileCache = new Map();

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function sendText(res, status, text) {
  res.writeHead(status, { "Content-Type": "text/plain; charset=utf-8" });
  res.end(text);
}

function safeSegment(value) {
  return typeof value === "string" && /^[a-zA-Z0-9_-]+$/.test(value);
}

function isWithin(root, candidate) {
  const resolvedRoot = path.resolve(root);
  const resolvedCandidate = path.resolve(candidate);
  return resolvedCandidate === resolvedRoot || resolvedCandidate.startsWith(resolvedRoot + path.sep);
}

function countLines(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trim();
  return text ? text.split("\n").length : 0;
}

function readRows(filePath) {
  const stats = fs.statSync(filePath);
  const cached = fileCache.get(filePath);
  if (cached && cached.mtimeMs === stats.mtimeMs) {
    return cached.rows;
  }

  const text = fs.readFileSync(filePath, "utf8").trim();
  const rows = text
    ? text.split("\n").filter(Boolean).map((line, index) => {
        try {
          return JSON.parse(line);
        } catch (error) {
          throw new Error(`Invalid JSON on line ${index + 1} in ${filePath}: ${error.message}`);
        }
      })
    : [];
  fileCache.set(filePath, { mtimeMs: stats.mtimeMs, rows });
  return rows;
}

function listDirs(dirPath) {
  if (!fs.existsSync(dirPath)) {
    return [];
  }
  return fs
    .readdirSync(dirPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
}

function collectOptions(rows, field) {
  return [...new Set(rows.map((row) => row[field]).filter(Boolean).map(String))].sort();
}

function buildManifest() {
  const datasets = [];
  const roots = [
    { id: "authority", label: "Authority", basePath: path.join(DATA_ROOT, "authority") },
    { id: "agentdojo", label: "AgentDojo", basePath: path.join(DATA_ROOT, "benchmarks", "agentdojo") },
    { id: "injecagent", label: "InjecAgent", basePath: path.join(DATA_ROOT, "benchmarks", "injecagent") },
  ];

  for (const root of roots) {
    const configs = [];
    for (const config of listDirs(root.basePath)) {
      const configPath = path.join(root.basePath, config);
      const splits = {};
      for (const split of ["train", "test"]) {
        const splitPath = path.join(configPath, `${split}.jsonl`);
        if (fs.existsSync(splitPath)) {
          splits[split] = { rows: countLines(splitPath) };
        }
      }
      if (Object.keys(splits).length > 0) {
        configs.push({ id: config, label: config, splits });
      }
    }
    if (configs.length > 0) {
      datasets.push({ id: root.id, label: root.label, configs });
    }
  }

  manifestCache = { datasets };
  return manifestCache;
}

function resolveDataFile(query) {
  const dataset = query.dataset;
  const config = query.config;
  const split = query.split || "train";
  if (!safeSegment(dataset) || !safeSegment(config) || !safeSegment(split)) {
    return null;
  }

  const datasetRoots = {
    authority: path.join(DATA_ROOT, "authority"),
    agentdojo: path.join(DATA_ROOT, "benchmarks", "agentdojo"),
    injecagent: path.join(DATA_ROOT, "benchmarks", "injecagent"),
  };
  const basePath = datasetRoots[dataset];
  if (!basePath) {
    return null;
  }

  const filePath = path.resolve(basePath, config, `${split}.jsonl`);
  if (!isWithin(basePath, filePath)) {
    return null;
  }
  if (!fs.existsSync(filePath)) {
    return null;
  }
  return filePath;
}

function filterRows(rows, query) {
  const search = String(query.q || "").trim().toLowerCase();
  const label = String(query.label || "").trim();
  const actionType = String(query.action_type || "").trim();

  return rows.filter((row) => {
    if (label && String(row.label || "") !== label) {
      return false;
    }
    if (actionType && String(row.action_type || row.decision_kind || "") !== actionType) {
      return false;
    }
    if (!search) {
      return true;
    }
    const haystack = [
      row.sample_id,
      row.source_sample_id,
      row.tool,
      row.required_tool,
      row.adversarial_tool,
      row.task_goal,
      row.target_action,
      row.external_context,
      row.permitted,
      row.prohibited,
    ]
      .filter(Boolean)
      .join("\n")
      .toLowerCase();
    return haystack.includes(search);
  });
}

function summarize(rows) {
  const fields = [...new Set(rows.flatMap((row) => Object.keys(row)))].sort();
  const labels = collectOptions(rows, "label");
  const actionTypes = [
    ...new Set([
      ...collectOptions(rows, "action_type"),
      ...collectOptions(rows, "decision_kind"),
    ]),
  ].sort();
  const labelCounts = rows.reduce((counts, row) => {
    const label = row.label || "(missing)";
    counts[label] = (counts[label] || 0) + 1;
    return counts;
  }, {});

  return { fields, labels, actionTypes, labelCounts };
}

function handleApi(req, res, parsedUrl) {
  if (parsedUrl.pathname === "/api/manifest") {
    sendJson(res, 200, manifestCache || buildManifest());
    return;
  }

  if (parsedUrl.pathname === "/api/rows") {
    const filePath = resolveDataFile(parsedUrl.query);
    if (!filePath) {
      sendJson(res, 404, { error: "Dataset split not found." });
      return;
    }

    const rows = readRows(filePath);
    const filteredRows = filterRows(rows, parsedUrl.query);
    const offset = Math.max(Number(parsedUrl.query.offset || 0), 0);
    const limit = Math.min(Math.max(Number(parsedUrl.query.limit || 50), 1), MAX_LIMIT);
    const pageRows = filteredRows.slice(offset, offset + limit);
    sendJson(res, 200, {
      rows: pageRows,
      totalRows: rows.length,
      filteredRows: filteredRows.length,
      offset,
      limit,
      summary: summarize(rows),
    });
    return;
  }

  sendJson(res, 404, { error: "Unknown API route." });
}

function serveStatic(req, res, parsedUrl) {
  const pathname = parsedUrl.pathname === "/" ? "/index.html" : parsedUrl.pathname;
  const filePath = path.resolve(PUBLIC_ROOT, pathname.replace(/^\/+/, ""));
  if (!isWithin(PUBLIC_ROOT, filePath)) {
    sendText(res, 403, "Forbidden");
    return;
  }
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    sendText(res, 404, "Not found");
    return;
  }

  const ext = path.extname(filePath);
  const content = fs.readFileSync(filePath);
  res.writeHead(200, {
    "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
    "Content-Length": content.length,
  });
  res.end(content);
}

function createServer() {
  return http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    if (parsedUrl.pathname.startsWith("/api/")) {
      try {
        handleApi(req, res, parsedUrl);
      } catch (error) {
        sendJson(res, 500, { error: error.message });
      }
      return;
    }
    serveStatic(req, res, parsedUrl);
  });
}

function listen(port) {
  const server = createServer();
  server.on("error", (error) => {
    if (error.code === "EADDRINUSE" && port < DEFAULT_PORT + 20) {
      listen(port + 1);
      return;
    }
    throw error;
  });
  server.listen(port, HOST, () => {
    buildManifest();
    console.log(`Authority Data Explorer: http://127.0.0.1:${port}`);
  });
}

if (require.main === module) {
  listen(DEFAULT_PORT);
}

module.exports = {
  buildManifest,
  filterRows,
  readRows,
  resolveDataFile,
  summarize,
};
