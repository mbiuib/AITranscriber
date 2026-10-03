/* ============================================================
   Whisper Transcriber — Enterprise UI
   ============================================================ */

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];

// ============================================================
// State
// ============================================================
const ACTIVE_JOB_KEY = "whisper_active_job_id";
const state = {
  file: null,
  jobId: null,
  es: null,
  segments: [],
  text: "",
  metadata: {},
  running: false,
  consoleOpen: true,
  logCount: 0,
  i18n: {},
  lang: "ru",
  theme: "dark",
  schema: {},
  settings: {},
  settingsDirty: {},
  jobs: [],
  monitorTimer: null,
  seenSegmentsCount: 0,
};

// ============================================================
// i18n
// ============================================================
async function loadLocales() {
  const r = await fetch("/api/locales");
  const data = await r.json();
  state.lang = data.current;

  const rr = await fetch(`/api/locales/${state.lang}`);
  state.i18n = await rr.json();

  // Кнопки языков
  const sw = $("#lang-switch");
  sw.innerHTML = "";
  for (const code of data.available) {
    const b = document.createElement("button");
    b.textContent = code.toUpperCase();
    if (code === state.lang) b.classList.add("active");
    b.addEventListener("click", () => switchLang(code));
    sw.appendChild(b);
  }
  applyI18n();
}

async function switchLang(code) {
  state.lang = code;
  await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values: { "ui.language": code } }),
  });
  await loadLocales();
  renderSettingsForm();
}

function t(key, vars = {}) {
  let s = state.i18n[key] ?? key;
  for (const [k, v] of Object.entries(vars)) {
    s = s.replace(new RegExp(`\\{${k}\\}`, "g"), v);
  }
  return s;
}

function applyI18n() {
  $$("[data-i18n]").forEach(el => {
    const key = el.dataset.i18n;
    el.textContent = t(key);
  });
}

// ============================================================
// Theme
// ============================================================
async function setTheme(theme) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  $$("[data-theme-set]").forEach(b =>
    b.classList.toggle("active", b.dataset.themeSet === theme)
  );
  await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values: { "ui.theme": theme } }),
  });
}

// ============================================================
// Toasts
// ============================================================
function toast(kind, icon, text, timeout = 3500) {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.innerHTML = `<span class="toast-icon">${icon}</span><span class="toast-text"></span>`;
  el.querySelector(".toast-text").textContent = text;
  $("#toasts").appendChild(el);
  setTimeout(() => {
    el.classList.add("hide");
    setTimeout(() => el.remove(), 250);
  }, timeout);
}

// ============================================================
// System info / GPU
// ============================================================
async function loadSystem() {
  try {
    const r = await fetch("/api/system/resources");
    const d = await r.json();
    const pill = $("#gpu-pill");
    if (d.gpu) {
      pill.classList.add("ok");
      pill.classList.remove("err");
      pill.querySelector(".text").textContent =
        `${d.gpu.name} · ${d.gpu.memory_used_gb}/${d.gpu.memory_total_gb} GB · ${d.gpu.utilization}%`;
    } else {
      pill.classList.remove("ok");
      pill.classList.add("err");
      pill.querySelector(".text").textContent = t("status.gpu_unavailable");
    }
  } catch {
    const pill = $("#gpu-pill");
    pill.classList.remove("ok");
    pill.classList.add("err");
    pill.querySelector(".text").textContent = t("status.disconnected");
  }
}

// ============================================================
// Settings
// ============================================================
async function loadSettings() {
  const [sr, schr] = await Promise.all([
    fetch("/api/settings").then(r => r.json()),
    fetch("/api/settings/schema").then(r => r.json()),
  ]);
  state.settings = sr.values;
  state.schema = schr.schema;
  state.settingsDirty = {};
  renderSettingsForm();
  applyQuickOptions();
}

function renderSettingsForm() {
  const form = $("#settings-form");
  form.innerHTML = "";

  // Группируем по group
  const groups = {};
  for (const [key, def] of Object.entries(state.schema)) {
    const g = def.group || "misc";
    if (!groups[g]) groups[g] = [];
    groups[g].push([key, def]);
  }
  for (const list of Object.values(groups)) {
    list.sort((a, b) => (a[1].order || 0) - (b[1].order || 0));
  }

  const groupOrder = ["transcription", "advanced", "server", "ui"];
  for (const g of groupOrder) {
    if (!groups[g]) continue;
    const card = document.createElement("div");
    card.className = "settings-group";
    card.innerHTML = `
      <div class="settings-group-header">${t("group." + g)}</div>
      <div class="settings-group-body"></div>`;
    const body = card.querySelector(".settings-group-body");

    for (const [key, def] of groups[g]) {
      body.appendChild(renderField(key, def));
    }
    form.appendChild(card);
  }
}

function renderField(key, def) {
  const wrap = document.createElement("div");
  const isToggle = def.type === "bool";
  wrap.className = "settings-field" + (isToggle ? " toggle" : "");

  const val = state.settingsDirty[key] !== undefined
    ? state.settingsDirty[key]
    : state.settings[key];

  if (isToggle) {
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!val;
    cb.id = `set-${key}`;
    cb.addEventListener("change", () => {
      state.settingsDirty[key] = cb.checked;
    });
    const lbl = document.createElement("label");
    lbl.htmlFor = cb.id;
    lbl.textContent = def.label || key;
    wrap.append(cb, lbl);
  } else {
    const lbl = document.createElement("label");
    lbl.textContent = def.label || key;
    wrap.appendChild(lbl);

    let inp;
    if (def.type === "select") {
      inp = document.createElement("select");
      for (const opt of def.options) {
        const o = document.createElement("option");
        o.value = opt; o.textContent = opt;
        if (opt === val) o.selected = true;
        inp.appendChild(o);
      }
    } else if (def.type === "int") {
      inp = document.createElement("input");
      inp.type = "number";
      if (def.min != null) inp.min = def.min;
      if (def.max != null) inp.max = def.max;
      inp.value = val ?? "";
    } else {
      inp = document.createElement("input");
      inp.type = "text";
      inp.value = val ?? "";
    }
    inp.id = `set-${key}`;
    inp.addEventListener("input", () => {
      state.settingsDirty[key] = def.type === "int"
        ? parseInt(inp.value, 10)
        : inp.value;
    });
    wrap.appendChild(inp);
  }

  if (def.hint) {
    const h = document.createElement("div");
    h.className = "hint";
    h.textContent = def.hint;
    wrap.appendChild(h);
  }

  return wrap;
}

function applyQuickOptions() {
  const q = $("#quick-options");
  q.innerHTML = "";

  const modelSel = document.createElement("select");
  for (const m of state.schema["transcription.model"].options) {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    if (m === state.settings["transcription.model"]) o.selected = true;
    modelSel.appendChild(o);
  }
  const langSel = document.createElement("select");
  for (const l of state.schema["transcription.language"].options) {
    const o = document.createElement("option");
    o.value = l;
    o.textContent = l === "auto" ? "auto" : l;
    if (l === state.settings["transcription.language"]) o.selected = true;
    langSel.appendChild(o);
  }
  const bsInput = document.createElement("input");
  bsInput.type = "number";
  bsInput.min = 1; bsInput.max = 32;
  bsInput.value = state.settings["transcription.batch_size"];

  q.appendChild(wrapField(t("field.model"), modelSel));
  q.appendChild(wrapField(t("field.language"), langSel));
  q.appendChild(wrapField(t("field.batch_size"), bsInput));

  state.quickRefs = { modelSel, langSel, bsInput };
}

function wrapField(label, input) {
  const d = document.createElement("div");
  d.className = "field";
  const l = document.createElement("div");
  l.className = "field-label";
  l.textContent = label;
  d.append(l, input);
  return d;
}

async function saveSettings() {
  const values = { ...state.settingsDirty };
  if (!Object.keys(values).length) {
    toast("info", "ℹ️", "Нет изменений");
    return;
  }
  const r = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    toast("error", "❌", err.detail || "Ошибка сохранения");
    return;
  }
  const d = await r.json();
  state.settings = d.values;
  state.settingsDirty = {};
  renderSettingsForm();
  applyQuickOptions();
  toast("success", "✅", t("toast.settings_saved"));
}

async function resetSettings() {
  if (!confirm(t("settings.reset_confirm"))) return;
  const r = await fetch("/api/settings/reset", { method: "POST" });
  const d = await r.json();
  state.settings = d.values;
  state.settingsDirty = {};
  renderSettingsForm();
  applyQuickOptions();
  toast("success", "🔄", t("toast.settings_reset"));
}

// ============================================================
// Jobs / upload
// ============================================================
function setupDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files[0] && setFile(input.files[0]));

  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.remove("dragover");
  }));
  dz.addEventListener("drop", e => {
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  $("#chip-remove").addEventListener("click", e => {
    e.stopPropagation(); clearFile();
  });
}

function setFile(file) {
  state.file = file;
  $("#chip-name").textContent = file.name;
  $("#chip-size").textContent = formatSize(file.size);
  $("#file-chip").hidden = false;
  $("#chip-remove").style.display = "";
  $("#dropzone").classList.add("has-file");
  $(".dropzone-title").textContent = t("upload.file_ready");
  $(".dropzone-hint").textContent = t("upload.change");
  $("#start-btn").disabled = false;
}

function clearFile() {
  state.file = null;
  $("#file-input").value = "";
  $("#file-chip").hidden = true;
  $("#chip-remove").style.display = "";
  $("#dropzone").classList.remove("has-file");
  $(".dropzone-title").textContent = t("upload.drop");
  $(".dropzone-hint").textContent = t("upload.hint");
  $("#start-btn").disabled = true;
}

function formatSize(bytes) {
  const u = ["B", "KB", "MB", "GB"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + " " + u[i];
}

async function start() {
  if (!state.file || state.running) return;
  resetResult();

  state.running = true;
  $("#start-btn").disabled = true;
  $("#cancel-btn").disabled = false;
  ensureConsoleOpen();

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("model", state.quickRefs.modelSel.value);
  fd.append("language", state.quickRefs.langSel.value);

  try {
    const r = await fetch("/api/jobs", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
    const d = await r.json();
    state.jobId = d.job_id;
    toast("success", "✅", t("toast.job_created"));
    saveActiveJob(d.job_id);
    subscribe(d.job_id);
    refreshHistory();
  } catch (e) {
    toast("error", "❌", e.message);
    state.running = false;
    $("#start-btn").disabled = false;
    $("#cancel-btn").disabled = true;
  }
}

async function cancelJob() {
  if (!state.jobId) return;
  await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
  toast("warn", "⏹", t("toast.cancelled"));
}

function subscribe(jobId) {
  if (state.es) state.es.close();
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  state.es = es;
  es.onmessage = ev => {
    try { handleEvent(JSON.parse(ev.data)); }
    catch (e) { console.error(e); }
  };
  es.onerror = () => { if (!state.running) es.close(); };
}

function saveActiveJob(jobId) {
  try { localStorage.setItem(ACTIVE_JOB_KEY, jobId); } catch {}
}

function clearActiveJob() {
  try { localStorage.removeItem(ACTIVE_JOB_KEY); } catch {}
}

function getSavedJobId() {
  try { return localStorage.getItem(ACTIVE_JOB_KEY); } catch { return null; }
}

async function restoreActiveJob() {
  let jobId = getSavedJobId();

  if (!jobId) {
    try {
      const r = await fetch("/api/jobs");
      const d = await r.json();
      const active = (d.jobs || []).find(j =>
        ["pending", "downloading", "loading", "transcribing"].includes(j.status)
      );
      if (active) jobId = active.id;
    } catch {}
  }

  if (!jobId) return;

  let snap;
  try {
    const r = await fetch(`/api/jobs/${jobId}`);
    if (r.status === 404) {
      clearActiveJob();
      return;
    }
    if (!r.ok) return;
    snap = await r.json();
  } catch {
    return;
  }

  applySnapshot(snap);
  saveActiveJob(snap.id);
  state.jobId = snap.id;

  const isActive = ["pending", "downloading", "loading", "transcribing"]
    .includes(snap.status);

  if (isActive) {
    state.running = true;
    $("#start-btn").disabled = true;
    $("#cancel-btn").disabled = false;
    ensureConsoleOpen();

    $("#chip-name").textContent = snap.filename;
    $("#chip-size").textContent = formatSize(snap.file_size || 0);
    $("#file-chip").hidden = false;
    $("#chip-remove").style.display = "none";

    subscribe(snap.id);

    toast("info", "🔄", "Восстановлено: задача выполняется");
  } else {
    $("#chip-name").textContent = snap.filename;
    $("#chip-size").textContent = formatSize(snap.file_size || 0);
    $("#file-chip").hidden = false;
    $("#chip-remove").style.display = "none";

    if (snap.status === "done") {
      toast("success", "✅", "Восстановлен результат");
    } else if (snap.status === "error") {
      toast("error", "❌", "Задача завершилась с ошибкой");
    } else if (snap.status === "cancelled") {
      toast("warn", "⏹", "Задача была отменена");
    }
    clearActiveJob();
  }
}

function handleEvent(ev) {
  switch (ev.type) {
    case "snapshot": applySnapshot(ev.data); break;
    case "log":      appendLog(ev.entry); break;
    case "progress": updateProgress(ev.value, ev.message, ev.stage); break;
    case "status":   updateStatus(ev.status, ev.message); break;
    case "segment":  appendSegment(ev.segment); break;
    case "done":     onDone(ev.text, ev.metadata); break;
    case "error":    onError(ev.message); break;
    case "cancelled": onCancelled(); break;
  }
}

function applySnapshot(snap) {
  for (const log of snap.logs || []) appendLog(log, true);

  if (snap.segments?.length) {
    state.segments = [];
    state.seenSegmentsCount = 0;
    $("#result-content").innerHTML =
      '<div class="segments" id="segments"></div>';
    for (const s of snap.segments) appendSegment(s, true);
    state.seenSegmentsCount = state.segments.length;
    updateScrollButton();
  }

  if (snap.progress !== undefined) {
    updateProgress(snap.progress, snap.message, snap.stage);
  }
  updateStatus(snap.status, snap.message);

  if (snap.status === "done") {
    onDone(snap.text, snap.metadata);
  }
}

function updateProgress(v, msg, stage) {
  const fill = $("#progress-fill");
  fill.style.width = v + "%";
  fill.classList.toggle("active", v > 0 && v < 100);
  $("#progress-pct").textContent = v + "%";
  if (stage) $("#progress-stage").textContent = t("stage." + stage);
  if (msg) $("#progress-msg").textContent = msg;
}

function updateStatus(status, message) {
  if (["done", "error", "cancelled"].includes(status)) {
    state.running = false;
    $("#start-btn").disabled = !state.file;
    $("#cancel-btn").disabled = true;
  }
}

function appendLog(entry, silent = false) {
  const body = $("#console-body");
  body.querySelector(".console-empty")?.remove();

  const wasAtBottom = isNearBottom(body, 40);

  const line = document.createElement("div");
  line.className = "log-line";
  const d = new Date(entry.time * 1000);
  const time = d.toLocaleTimeString("ru-RU", { hour12: false });
  line.innerHTML =
    `<span class="log-time">${time}</span>` +
    `<span class="log-level ${entry.level}">${entry.level.toUpperCase()}</span>` +
    `<span class="log-message"></span>`;
  line.querySelector(".log-message").textContent = entry.message;
  body.appendChild(line);

  if (wasAtBottom && !silent) {
    body.scrollTop = body.scrollHeight;
  }

  state.logCount = body.querySelectorAll(".log-line").length;
  $("#console-counter").textContent = state.logCount;
}

function appendSegment(seg, silent = false) {
  let wrap = $("#segments");
  if (!wrap) {
    $("#result-content").innerHTML = '<div class="segments" id="segments"></div>';
    wrap = $("#segments");
  }
  $("#result-content .empty-state")?.remove();

  const rc = $("#result-content");
  const wasAtBottom = isNearBottom(rc);

  const line = document.createElement("div");
  line.className = "segment" + (silent ? "" : " new-segment");
  const g = document.createElement("span");
  g.className = "gutter";
  g.textContent = formatTs(seg.start);
  const tx = document.createElement("span");
  tx.className = "text";
  tx.textContent = seg.text;
  line.append(g, tx);
  wrap.appendChild(line);

  state.segments.push(seg);

  if (silent) {
    // Восстановление из snapshot — считаем, что всё просмотрено
    state.seenSegmentsCount = state.segments.length;
    rc.scrollTop = rc.scrollHeight;
  } else if (wasAtBottom) {
    // Пользователь внизу — сразу помечаем новый сегмент как просмотренный
    state.seenSegmentsCount = state.segments.length;
    rc.scrollTop = rc.scrollHeight;
  }
  // Если пользователь не внизу — просто не трогаем скролл
  // и не увеличиваем seenSegmentsCount

  updateScrollButton();

  ["act-copy", "act-txt", "act-srt", "act-json"].forEach(id => {
    $("#" + id).disabled = false;
  });
}

function isNearBottom(el, threshold = 60) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

function updateScrollButton() {
  const btn = $("#scroll-bottom-btn");
  if (!btn) return;

  const unread = state.segments.length - state.seenSegmentsCount;

  if (unread > 0 && state.running) {
    btn.classList.add("visible");
    $("#scroll-bottom-badge").textContent = unread > 99 ? "99+" : unread;
  } else {
    btn.classList.remove("visible");
  }
}

function formatTs(sec) {
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return `${pad(h,2)}:${pad(m,2)}:${pad(ss,2)}`;
}
const pad = (n, l) => String(n).padStart(l, "0");

function onDone(text, metadata) {
  state.text = text || "";
  state.metadata = metadata || {};
  state.running = false;
  state.seenSegmentsCount = state.segments.length;
  updateScrollButton();
  clearActiveJob();

  $("#start-btn").disabled = !state.file;
  $("#cancel-btn").disabled = true;
  updateProgress(100, "Готово", "done");

  const dur = metadata.duration || 0;
  const proc = metadata.processing_time || 0;
  const speed = metadata.speed_factor || (dur / Math.max(proc, .01));

  const stats = $("#result-stats");
  stats.hidden = false;
  stats.textContent = t("result.stats", {
    segments: state.segments.length,
    duration: formatTs(dur),
    speed: speed.toFixed(2),
  });

  toast("success", "🎉", t("toast.done"));
  state.es?.close();
  refreshHistory();
}

function onError(msg) {
  state.running = false;
  clearActiveJob();
  $("#start-btn").disabled = !state.file;
  $("#cancel-btn").disabled = true;
  $("#progress-fill").classList.remove("active");
  toast("error", "❌", msg.slice(0, 140), 6000);
  state.es?.close();
  refreshHistory();
}

function onCancelled() {
  state.running = false;
  clearActiveJob();
  $("#start-btn").disabled = !state.file;
  $("#cancel-btn").disabled = true;
  $("#progress-fill").classList.remove("active");
  toast("warn", "⏹", t("toast.cancelled"));
  state.es?.close();
}

function setupResultScroll() {
  const rc = $("#result-content");
  const btn = $("#scroll-bottom-btn");

  rc.addEventListener("scroll", () => {
    if (isNearBottom(rc)) {
      // Дошёл до низа — всё считается просмотренным
      state.seenSegmentsCount = state.segments.length;
    }
    updateScrollButton();
  }, { passive: true });

  btn.addEventListener("click", () => {
    rc.scrollTo({ top: rc.scrollHeight, behavior: "smooth" });
    state.seenSegmentsCount = state.segments.length;
    setTimeout(updateScrollButton, 350);
  });
}

function resetResult() {
  state.segments = [];
  state.text = "";
  state.metadata = {};
  state.jobId = null;
  state.seenSegmentsCount = 0;
  updateScrollButton();

  $("#result-stats").hidden = true;
  $("#result-content").innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">⏳</div>
      <div class="empty-text">${t("common.loading")}</div>
    </div>`;
  updateProgress(0, "", "pending");
  ["act-copy", "act-txt", "act-srt", "act-json"].forEach(id => {
    $("#" + id).disabled = true;
  });
  state.es?.close();
}

// ============================================================
// Export
// ============================================================
async function copyText() {
  if (!state.segments.length) return;
  const txt = state.segments.map(s => s.text).join("\n");
  await navigator.clipboard.writeText(txt);
  toast("success", "📋", t("toast.copied"));
}

function downloadTxt() {
  const txt = state.segments.map(s => s.text).join("\n\n");
  downloadBlob(txt, baseName() + ".txt", "text/plain");
}

function downloadSrt() {
  const srt = state.segments.map((s, i) =>
    `${i+1}\n${fmtSrt(s.start)} --> ${fmtSrt(s.end)}\n${s.text}\n`
  ).join("\n");
  downloadBlob(srt, baseName() + ".srt", "application/x-subrip");
}

function downloadJson() {
  downloadBlob(
    JSON.stringify({ text: state.text, segments: state.segments, metadata: state.metadata }, null, 2),
    baseName() + ".json", "application/json"
  );
}

function fmtSrt(sec) {
  const s = Math.floor(sec);
  const ms = Math.floor((sec - s) * 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return `${pad(h,2)}:${pad(m,2)}:${pad(ss,2)},${pad(ms,3)}`;
}

function baseName() {
  if (!state.file) return "transcript";
  return state.file.name.replace(/\.[^.]+$/, "");
}

function downloadBlob(content, name, mime) {
  const blob = new Blob([content], { type: mime + ";charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("success", "💾", t("toast.saved", { name }));
}

// ============================================================
// History
// ============================================================
async function refreshHistory() {
  try {
    const r = await fetch("/api/jobs");
    const d = await r.json();
    state.jobs = d.jobs;
    renderHistory();
    const badge = $("#nav-badge-history");
    const active = state.jobs.filter(j =>
      ["pending", "downloading", "loading", "transcribing"].includes(j.status)
    ).length;
    badge.hidden = active === 0;
    badge.textContent = active;
  } catch {}
}

function renderHistory() {
  const list = $("#history-list");
  if (!state.jobs.length) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📚</div>
        <div class="empty-text">${t("history.empty")}</div>
      </div>`;
    return;
  }
  list.innerHTML = "";
  for (const j of state.jobs) {
    const item = document.createElement("div");
    item.className = "history-item";
    const created = new Date(j.created_at * 1000).toLocaleString("ru-RU");
    const dur = j.duration ? formatTs(j.duration) : "—";
    item.innerHTML = `
      <div class="history-icon">🎬</div>
      <div class="history-main">
        <div class="history-name"></div>
        <div class="history-meta">${created} · ${j.model} · ${dur} · ${j.segments_count} сегм.</div>
      </div>
      <div class="history-status ${j.status}">${t("stage." + j.status) || j.status}</div>
      <button class="btn-icon" data-action="delete" title="Удалить">🗑️</button>
    `;
    item.querySelector(".history-name").textContent = j.filename;
    item.querySelector('[data-action="delete"]').addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(t("history.delete_confirm"))) return;
      await fetch(`/api/jobs/${j.id}`, { method: "DELETE" });
      toast("success", "🗑️", t("toast.job_deleted"));
      refreshHistory();
    });
    list.appendChild(item);
  }
}

// ============================================================
// System view
// ============================================================
async function updateMonitor() {
  const r = await fetch("/api/system/resources");
  const d = await r.json();
  const grid = $("#metrics-grid");
  const html = [];

  // CPU
  const cpu = d.cpu;
  html.push(metricCard(
    t("system.cpu"),
    `${cpu.percent.toFixed(0)}%`,
    `${cpu.count} потоков`,
    cpu.percent,
    d.history.cpu
  ));

  // RAM
  const ram = d.ram;
  html.push(metricCard(
    t("system.ram"),
    `${ram.percent.toFixed(0)}%`,
    `${ram.used_gb} / ${ram.total_gb} GB`,
    ram.percent,
    d.history.ram
  ));

  // GPU
  if (d.gpu) {
    html.push(metricCard(
      t("system.gpu"),
      `${d.gpu.utilization}%`,
      d.gpu.name,
      d.gpu.utilization,
      d.history.gpu_util
    ));
    html.push(metricCard(
      t("system.vram"),
      `${d.gpu.memory_percent}%`,
      `${d.gpu.memory_used_gb} / ${d.gpu.memory_total_gb} GB`,
      d.gpu.memory_percent,
      d.history.gpu_mem,
      d.gpu.temperature != null ? `${d.gpu.temperature}°C` : null
    ));
  }

  grid.innerHTML = html.join("");

  // Disks
  const dl = $("#disks-list");
  dl.innerHTML = (d.disks || []).map(disk => `
    <div class="disk-item">
      <div class="disk-label">${disk.label}</div>
      <div class="disk-path">${disk.path}</div>
      <div class="disk-bar"><div class="disk-bar-fill" style="width:${disk.percent}%"></div></div>
      <div class="disk-info">${disk.free_gb} GB free / ${disk.total_gb} GB</div>
    </div>
  `).join("");

  // Update GPU pill
  const pill = $("#gpu-pill");
  if (d.gpu) {
    pill.classList.add("ok"); pill.classList.remove("err");
    pill.querySelector(".text").textContent =
      `${d.gpu.name} · ${d.gpu.memory_used_gb}/${d.gpu.memory_total_gb} GB · ${d.gpu.utilization}%`;
  }
}

function metricCard(title, value, sub, percent, history, extra) {
  const cls = percent > 85 ? "warn" : percent < 30 ? "ok" : "";
  return `
    <div class="metric-card">
      <div class="metric-header">
        <span class="metric-title">${title}</span>
        ${extra ? `<span class="muted">${extra}</span>` : ""}
      </div>
      <div class="metric-value">${value}</div>
      <div class="metric-sub">${sub}</div>
      <div class="metric-bar"><div class="metric-bar-fill ${cls}" style="width:${percent}%"></div></div>
      ${renderSparkline(history)}
    </div>
  `;
}

function renderSparkline(data) {
  if (!data || !data.length) return "";
  const w = 240, h = 28;
  const max = Math.max(...data, 1);
  const step = w / Math.max(data.length - 1, 1);
  const pts = data.map((v, i) => `${(i * step).toFixed(1)},${(h - (v / max) * h).toFixed(1)}`).join(" ");
  return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="currentColor" stroke-width="1.5"
      style="color: var(--accent)"/>
  </svg>`;
}

// ============================================================
// Console
// ============================================================
function setupConsole() {
  document.body.classList.toggle("console-open", state.consoleOpen);
  $("#console-chevron").textContent = state.consoleOpen ? "▼" : "▲";

  $("#console-toggle").addEventListener("click", e => {
    if (e.target.closest("button")) return;
    state.consoleOpen = !state.consoleOpen;
    document.body.classList.toggle("console-open", state.consoleOpen);
    $("#console-chevron").textContent = state.consoleOpen ? "▼" : "▲";
  });

  $("#console-clear").addEventListener("click", e => {
    e.stopPropagation();
    $("#console-body").innerHTML =
      `<div class="console-empty">${t("console.empty")}</div>`;
    state.logCount = 0;
    $("#console-counter").textContent = "0";
  });

  $("#console-copy").addEventListener("click", async e => {
    e.stopPropagation();
    const lines = $$(".log-line", $("#console-body")).map(l =>
      `[${l.querySelector(".log-time").textContent}] ` +
      `${l.querySelector(".log-level").textContent.padEnd(8)} ` +
      `${l.querySelector(".log-message").textContent}`
    ).join("\n");
    await navigator.clipboard.writeText(lines);
    toast("success", "📋", t("toast.copied"));
  });
}

function ensureConsoleOpen() {
  if (!state.consoleOpen) {
    state.consoleOpen = true;
    document.body.classList.add("console-open");
    $("#console-chevron").textContent = "▼";
  }
}

// ============================================================
// Navigation
// ============================================================
function setupNav() {
  $$(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
      const v = btn.dataset.view;
      $$(".nav-item").forEach(b => b.classList.toggle("active", b === btn));
      $$(".view").forEach(s => s.classList.toggle("active", s.dataset.view === v));

      if (v === "settings") loadSettings();
      if (v === "system") startMonitor();
      else stopMonitor();
      if (v === "history") refreshHistory();
    });
  });
}

function startMonitor() {
  stopMonitor();
  updateMonitor();
  state.monitorTimer = setInterval(updateMonitor, 1500);
}
function stopMonitor() {
  if (state.monitorTimer) clearInterval(state.monitorTimer);
  state.monitorTimer = null;
}

// ============================================================
// Init
// ============================================================
document.addEventListener("DOMContentLoaded", async () => {
  // Загружаем настройки UI
  await loadLocales();
  await loadSettings();
  // Тема из настроек
  document.documentElement.dataset.theme = state.settings["ui.theme"] || "dark";
  state.theme = state.settings["ui.theme"] || "dark";
  $$("[data-theme-set]").forEach(b =>
    b.classList.toggle("active", b.dataset.themeSet === state.theme)
  );
  $$("[data-theme-set]").forEach(b =>
    b.addEventListener("click", () => setTheme(b.dataset.themeSet))
  );

  setupNav();
  setupDropzone();
  setupConsole();
  setupResultScroll();
  await loadSystem();

  $("#start-btn").addEventListener("click", start);
  $("#cancel-btn").addEventListener("click", cancelJob);
  $("#act-copy").addEventListener("click", copyText);
  $("#act-txt").addEventListener("click", downloadTxt);
  $("#act-srt").addEventListener("click", downloadSrt);
  $("#act-json").addEventListener("click", downloadJson);
  $("#settings-save").addEventListener("click", saveSettings);
  $("#settings-reset").addEventListener("click", resetSettings);
  $("#history-refresh").addEventListener("click", refreshHistory);

  // Retention info
  const hours = state.settings["server.retention_hours"] || 24;
  $("#retention-info").textContent = t("history.retention", { hours });

  // Периодическое обновление GPU
  setInterval(loadSystem, 5000);
  refreshHistory();
  setInterval(refreshHistory, 10000);
  await restoreActiveJob();
});
