/* ============================================================
   Whisper Transcriber — клиентская логика
   ============================================================ */

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];

const ACTIVE_JOB_KEY = "whisper_active_job_id";

/* ============================================================
   State
   ============================================================ */
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
  detailJob: null,
  detailTags: [],
};

/* ============================================================
   i18n
   ============================================================ */
async function loadLocales() {
  const r = await fetch("/api/locales");
  const data = await r.json();
  state.lang = data.current;

  const rr = await fetch(`/api/locales/${state.lang}`);
  state.i18n = await rr.json();

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

/* ============================================================
   Theme
   ============================================================ */
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

/* ============================================================
   Toasts
   ============================================================ */
function toast(kind, iconName, text, timeout = 3500) {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.innerHTML =
    `<span class="toast-icon material-symbols-rounded">${iconName}</span>` +
    `<span class="toast-text"></span>`;
  el.querySelector(".toast-text").textContent = text;
  $("#toasts").appendChild(el);
  setTimeout(() => {
    el.classList.add("hide");
    setTimeout(() => el.remove(), 250);
  }, timeout);
}

/* ============================================================
   System info / GPU
   ============================================================ */
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

/* ============================================================
   Settings
   ============================================================ */
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
  } else if (def.type === "int") {
    const lbl = document.createElement("label");
    lbl.textContent = def.label || key;
    wrap.appendChild(lbl);

    const numWrap = createNumberInput(val ?? 0, {
      min: def.min,
      max: def.max,
      id: `set-${key}`,
      onChange: (v) => {
        state.settingsDirty[key] = v;
      },
    });
    wrap.appendChild(numWrap);
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
    } else {
      inp = document.createElement("input");
      inp.type = "text";
      inp.value = val ?? "";
    }
    inp.id = `set-${key}`;
    inp.addEventListener("input", () => {
      state.settingsDirty[key] = inp.value;
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

  const bsInput = createNumberInput(
    state.settings["transcription.batch_size"],
    { min: 1, max: 32, step: 1 }
  );

  q.appendChild(wrapField(t("field.model"), modelSel));
  q.appendChild(wrapField(t("field.language"), langSel));
  q.appendChild(wrapField(t("field.batch_size"), bsInput));

  state.quickRefs = {
    modelSel,
    langSel,
    getBatchSize: () => bsInput.getValue(),
  };
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

/**
 * Создаёт <input type="number"> с кастомными стрелками.
 * Возвращает контейнер с публичным API: getValue(), setValue(), input.
 */
function createNumberInput(value, { min, max, step = 1, id, onChange } = {}) {
  const wrap = document.createElement("div");
  wrap.className = "num-input";

  const input = document.createElement("input");
  input.type = "number";
  input.value = value ?? "";
  if (min != null) input.min = min;
  if (max != null) input.max = max;
  input.step = step;
  if (id) input.id = id;

  const steppers = document.createElement("div");
  steppers.className = "steppers";

  const up = document.createElement("button");
  up.type = "button";
  up.className = "stepper up material-symbols-rounded";
  up.textContent = "keyboard_arrow_up";
  up.tabIndex = -1;

  const down = document.createElement("button");
  down.type = "button";
  down.className = "stepper down material-symbols-rounded";
  down.textContent = "keyboard_arrow_down";
  down.tabIndex = -1;

  const clamp = (v) => {
    if (min != null && v < min) v = min;
    if (max != null && v > max) v = max;
    return v;
  };

  up.addEventListener("click", () => {
    const v = clamp((parseInt(input.value, 10) || 0) + step);
    input.value = v;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    onChange?.(v);
  });

  down.addEventListener("click", () => {
    const v = clamp((parseInt(input.value, 10) || 0) - step);
    input.value = v;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    onChange?.(v);
  });

  input.addEventListener("input", () => {
    onChange?.(parseInt(input.value, 10));
  });

  steppers.append(up, down);
  wrap.append(input, steppers);

  wrap.getValue = () => parseInt(input.value, 10);
  wrap.setValue = (v) => { input.value = clamp(v); };
  wrap.input = input;

  return wrap;
}

async function saveSettings() {
  const values = { ...state.settingsDirty };
  if (!Object.keys(values).length) {
    toast("info", "info", "Нет изменений");
    return;
  }
  const r = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    toast("error", "error", err.detail || "Ошибка сохранения");
    return;
  }
  const d = await r.json();
  state.settings = d.values;
  state.settingsDirty = {};
  renderSettingsForm();
  applyQuickOptions();
  toast("success", "check_circle", t("toast.settings_saved"));
}

async function resetSettings() {
  if (!confirm(t("settings.reset_confirm"))) return;
  const r = await fetch("/api/settings/reset", { method: "POST" });
  const d = await r.json();
  state.settings = d.values;
  state.settingsDirty = {};
  renderSettingsForm();
  applyQuickOptions();
  toast("success", "refresh", t("toast.settings_reset"));
}

/* ============================================================
   Active job persistence (переживание F5)
   ============================================================ */
function saveActiveJob(jobId) {
  try { localStorage.setItem(ACTIVE_JOB_KEY, jobId); } catch {}
}
function clearActiveJob() {
  try { localStorage.removeItem(ACTIVE_JOB_KEY); } catch {}
}
function getSavedJobId() {
  try { return localStorage.getItem(ACTIVE_JOB_KEY); } catch { return null; }
}

/**
 * Восстанавливает активную задачу после перезагрузки страницы.
 *
 * Сначала проверяет ID из localStorage. Если его нет — ищет любую
 * активную задачу через /api/jobs. Для активной задачи восстанавливает
 * снимок и подписывается на SSE, для завершённой — показывает финал
 * без подписки.
 */
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

  $("#chip-name").textContent = snap.filename;
  $("#chip-size").textContent = formatSize(snap.file_size || 0);
  $("#file-chip").hidden = false;
  $("#chip-remove").style.display = "none";

  if (isActive) {
    state.running = true;
    $("#start-btn").disabled = true;
    $("#cancel-btn").disabled = false;
    ensureConsoleOpen();
    subscribe(snap.id);
    toast("info", "refresh", "Восстановлено: задача выполняется");
  } else {
    if (snap.status === "done") {
      toast("success", "check_circle", "Восстановлен результат");
    } else if (snap.status === "error") {
      toast("error", "error", "Задача завершилась с ошибкой");
    } else if (snap.status === "cancelled") {
      toast("warn", "stop_circle", "Задача была отменена");
    }
    clearActiveJob();
  }
}

/* ============================================================
   Dropzone
   ============================================================ */
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

/* ============================================================
   Start / cancel
   ============================================================ */
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
    saveActiveJob(d.job_id);
    toast("success", "check_circle", t("toast.job_created"));
    subscribe(d.job_id);
    refreshHistory();
  } catch (e) {
    toast("error", "error", e.message);
    state.running = false;
    $("#start-btn").disabled = false;
    $("#cancel-btn").disabled = true;
  }
}

async function cancelJob() {
  if (!state.jobId) return;
  await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
  toast("warn", "stop_circle", t("toast.cancelled"));
}

/* ============================================================
   SSE
   ============================================================ */
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
    $("#result-content").innerHTML = '<div class="segments" id="segments"></div>';
    for (const s of snap.segments) appendSegment(s, true);
    state.seenSegmentsCount = state.segments.length;
    updateScrollButton();
  }

  if (snap.progress !== undefined) updateProgress(snap.progress, snap.message, snap.stage);
  updateStatus(snap.status, snap.message);
  if (snap.status === "done") onDone(snap.text, snap.metadata);
}

/* ============================================================
   Progress
   ============================================================ */
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

/* ============================================================
   Logs
   ============================================================ */
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

/* ============================================================
   Segments
   ============================================================ */
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
    state.seenSegmentsCount = state.segments.length;
    rc.scrollTop = rc.scrollHeight;
  } else if (wasAtBottom) {
    state.seenSegmentsCount = state.segments.length;
    rc.scrollTop = rc.scrollHeight;
  }

  updateScrollButton();

  ["act-copy", "act-txt", "act-srt", "act-json"].forEach(id => {
    $("#" + id).disabled = false;
  });
}

function formatTs(sec) {
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return `${pad(h,2)}:${pad(m,2)}:${pad(ss,2)}`;
}
const pad = (n, l) => String(n).padStart(l, "0");

function setupResultScroll() {
  const rc = $("#result-content");
  const btn = $("#scroll-bottom-btn");

  rc.addEventListener("scroll", () => {
    if (isNearBottom(rc)) {
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

/* ============================================================
   Finish / error
   ============================================================ */
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

  toast("success", "celebration", t("toast.done"));
  if (navigator.vibrate) navigator.vibrate([50, 30, 50]);
  state.es?.close();
  refreshHistory();
}

function onError(msg) {
  state.running = false;
  clearActiveJob();
  $("#start-btn").disabled = !state.file;
  $("#cancel-btn").disabled = true;
  $("#progress-fill").classList.remove("active");
  toast("error", "error", msg.slice(0, 140), 6000);
  if (navigator.vibrate) navigator.vibrate(100);
  state.es?.close();
  refreshHistory();
}

function onCancelled() {
  state.running = false;
  clearActiveJob();
  $("#start-btn").disabled = !state.file;
  $("#cancel-btn").disabled = true;
  $("#progress-fill").classList.remove("active");
  toast("warn", "stop_circle", t("toast.cancelled"));
  state.es?.close();
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
      <div class="empty-icon material-symbols-rounded">hourglass_top</div>
      <div class="empty-text">${t("common.loading")}</div>
    </div>`;
  updateProgress(0, "", "pending");
  ["act-copy", "act-txt", "act-srt", "act-json"].forEach(id => {
    $("#" + id).disabled = true;
  });
  state.es?.close();
}

/* ============================================================
   Export
   ============================================================ */
async function copyText() {
  if (!state.segments.length) return;
  const txt = state.segments.map(s => s.text).join("\n");
  await navigator.clipboard.writeText(txt);
  toast("success", "content_copy", t("toast.copied"));
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
  toast("success", "save", t("toast.saved", { name }));
}

/* ============================================================
   History
   ============================================================ */
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
        <div class="empty-icon material-symbols-rounded">history_toggle_off</div>
        <div class="empty-text">${t("history.empty")}</div>
      </div>`;
    return;
  }
  list.innerHTML = "";
  for (const j of state.jobs) {
    const item = document.createElement("div");
    item.className = "history-item";
    item.style.cursor = "pointer";

    const created = new Date(j.created_at * 1000).toLocaleString("ru-RU");
    const dur = j.duration ? formatTs(j.duration) : "—";
    const tagsHtml = (j.tags || []).length
      ? `<div class="history-tags">${j.tags.map(tag =>
          `<span class="history-tag">${escapeHtml(tag)}</span>`).join("")}</div>`
      : "";

    item.innerHTML = `
      <div class="history-icon">
        <span class="material-symbols-rounded">movie</span>
      </div>
      <div class="history-main">
        <div class="history-name"></div>
        <div class="history-meta">${created} · ${j.model} · ${dur} · ${j.segments_count} сегм.</div>
        ${tagsHtml}
      </div>
      <div class="history-status ${j.status}">${t("stage." + j.status) || j.status}</div>
      <button class="btn-icon" data-action="delete" title="Удалить">
        <span class="material-symbols-rounded">delete</span>
      </button>
    `;
    item.querySelector(".history-name").textContent = j.filename;

    item.addEventListener("click", (e) => {
      if (e.target.closest('[data-action="delete"]')) return;
      openJobDetail(j.id);
    });

    item.querySelector('[data-action="delete"]').addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(t("history.delete_confirm"))) return;
      await fetch(`/api/jobs/${j.id}`, { method: "DELETE" });
      toast("success", "delete", t("toast.job_deleted"));
      refreshHistory();
    });
    list.appendChild(item);
  }
}

/* ============================================================
   System view
   ============================================================ */
async function updateMonitor() {
  const r = await fetch("/api/system/resources");
  const d = await r.json();
  const grid = $("#metrics-grid");
  const html = [];

  const cpu = d.cpu;
  html.push(metricCard(
    t("system.cpu"),
    `${cpu.percent.toFixed(0)}%`,
    `${cpu.count} потоков`,
    cpu.percent,
    d.history.cpu
  ));

  const ram = d.ram;
  html.push(metricCard(
    t("system.ram"),
    `${ram.percent.toFixed(0)}%`,
    `${ram.used_gb} / ${ram.total_gb} GB`,
    ram.percent,
    d.history.ram
  ));

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

  const dl = $("#disks-list");
  dl.innerHTML = (d.disks || []).map(disk => `
    <div class="disk-item">
      <div class="disk-label">${disk.label}</div>
      <div class="disk-path">${disk.path}</div>
      <div class="disk-bar"><div class="disk-bar-fill" style="width:${disk.percent}%"></div></div>
      <div class="disk-info">${disk.free_gb} GB free / ${disk.total_gb} GB</div>
    </div>
  `).join("");

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
    <polyline points="${pts}" fill="none" stroke="currentColor" stroke-width="1.5"/>
  </svg>`;
}

/* ============================================================
   Console
   ============================================================ */
function updateConsoleChevron() {
  const ch = $("#console-chevron");
  ch.textContent = state.consoleOpen ? "keyboard_arrow_down" : "keyboard_arrow_up";
  ch.classList.toggle("rotated", !state.consoleOpen);
}

function setupConsole() {
  document.body.classList.toggle("console-open", state.consoleOpen);
  updateConsoleChevron();

  $("#console-toggle").addEventListener("click", e => {
    if (e.target.closest("button")) return;
    state.consoleOpen = !state.consoleOpen;
    document.body.classList.toggle("console-open", state.consoleOpen);
    updateConsoleChevron();
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
    toast("success", "content_copy", t("toast.copied"));
  });
}

function ensureConsoleOpen() {
  if (!state.consoleOpen) {
    state.consoleOpen = true;
    document.body.classList.add("console-open");
    updateConsoleChevron();
  }
}

/* ============================================================
   Job detail modal
   ============================================================ */
async function openJobDetail(jobId) {
  try {
    const r = await fetch(`/api/jobs/${jobId}`);
    if (!r.ok) throw new Error("Задача не найдена");
    const snap = await r.json();

    state.detailJob = snap;
    state.detailTags = [...(snap.tags || [])];

    renderDetail(snap);
    $("#detail-modal").classList.add("visible");
    document.body.style.overflow = "hidden";

    // Активируем вкладку «Транскрипт» по умолчанию
    $$(".modal-tab").forEach(t =>
      t.classList.toggle("active", t.dataset.dtab === "transcript")
    );
    $$(".modal-pane").forEach(p =>
      p.classList.toggle("active", p.dataset.dpane === "transcript")
    );
  } catch (e) {
    toast("error", "error", e.message);
  }
}

function closeJobDetail() {
  $("#detail-modal").classList.remove("visible");
  document.body.style.overflow = "";
  state.detailJob = null;
  state.detailTags = [];
}

function renderDetail(snap) {
  $("#detail-filename").textContent = snap.filename;
  $("#detail-filename").title = snap.filename;

  // Транскрипт
  const tr = $("#detail-transcript");
  if (snap.segments && snap.segments.length) {
    tr.innerHTML = "";
    for (const s of snap.segments) {
      const line = document.createElement("div");
      line.className = "segment";
      const g = document.createElement("span");
      g.className = "gutter";
      g.textContent = formatTs(s.start);
      const tx = document.createElement("span");
      tx.className = "text";
      tx.textContent = s.text;
      line.append(g, tx);
      tr.appendChild(line);
    }
  } else if (snap.text) {
    tr.textContent = snap.text;
  } else {
    tr.innerHTML = `<div class="empty-state"><div class="empty-icon material-symbols-rounded">hourglass_empty</div><div class="empty-text">Результат ещё не готов</div></div>`;
  }

  // Метаданные
  renderDetailMeta(snap);

  // Статистика
  renderDetailStats(snap);

  // Теги и заметки
  renderDetailNotes(snap);
}

function renderDetailMeta(snap) {
  const el = $("#detail-meta");
  const src = snap.source_metadata || {};
  const settings = snap.settings_snapshot || {};
  const meta = snap.metadata || {};

  const rows = [];

  rows.push(["section", "Файл"]);
  rows.push(["Имя", snap.filename]);
  rows.push(["Размер", snap.file_size ? formatSize(snap.file_size) : "—"]);
  if (src.format_name) rows.push(["Формат", src.format_name]);
  if (src.duration) rows.push(["Длительность", formatTs(src.duration)]);
  if (src.bitrate_kbps) rows.push(["Битрейт", `${src.bitrate_kbps} kbps`]);
  if (snap.file_hash) rows.push(["SHA-256", snap.file_hash.slice(0, 16) + "…"]);

  if (src.audio) {
    rows.push(["section", "Аудио"]);
    if (src.audio.codec) rows.push(["Кодек", src.audio.codec.toUpperCase()]);
    if (src.audio.sample_rate) rows.push(["Частота", `${src.audio.sample_rate} Hz`]);
    if (src.audio.channels) rows.push(["Каналов", src.audio.channels]);
  }

  if (src.video) {
    rows.push(["section", "Видео"]);
    if (src.video.codec) rows.push(["Кодек", src.video.codec.toUpperCase()]);
    if (src.video.width && src.video.height) {
      rows.push(["Разрешение", `${src.video.width}×${src.video.height}`]);
    }
    if (src.video.fps) rows.push(["FPS", src.video.fps]);
  }

  rows.push(["section", "Параметры распознавания"]);
  rows.push(["Модель", settings.model || snap.model]);
  rows.push(["Язык (задан)", settings.language || "auto"]);
  rows.push(["Язык (определён)", meta.language || "—"]);
  if (meta.language_probability) {
    rows.push(["Уверенность", `${(meta.language_probability * 100).toFixed(0)}%`]);
  }
  rows.push(["VAD", settings.vad ? "включён" : "выключен"]);
  rows.push(["Batch size", settings.batch_size || "—"]);
  rows.push(["Beam size", settings.beam_size || "—"]);
  rows.push(["Устройство", settings.device || "—"]);
  rows.push(["Точность", settings.compute_type || "—"]);

  rows.push(["section", "Даты"]);
  rows.push(["Создана", new Date(snap.created_at * 1000).toLocaleString("ru-RU")]);
  if (snap.finished_at) {
    rows.push(["Завершена", new Date(snap.finished_at * 1000).toLocaleString("ru-RU")]);
  }

  let html = "";
  let inGrid = false;
  for (const row of rows) {
    if (row[0] === "section") {
      if (inGrid) { html += "</dl>"; inGrid = false; }
      html += `<div class="detail-section"><h3>${row[1]}</h3>`;
    } else {
      if (!inGrid) { html += '<dl class="detail-grid">'; inGrid = true; }
      const val = row[1] === undefined || row[1] === null || row[1] === ""
        ? '<span class="empty">—</span>' : escapeHtml(String(row[1]));
      html += `<dt>${escapeHtml(row[0])}</dt><dd>${val}</dd>`;
    }
  }
  if (inGrid) html += "</dl>";
  el.innerHTML = html;
}

function renderDetailStats(snap) {
  const el = $("#detail-stats");
  const stats = snap.processing_stats || {};
  const meta = snap.metadata || {};

  if (!Object.keys(stats).length && !Object.keys(meta).length) {
    el.innerHTML = `<div class="empty-state"><div class="empty-icon material-symbols-rounded">analytics</div><div class="empty-text">Статистика пока недоступна</div></div>`;
    return;
  }

  const rows = [];

  if (stats.stages) {
    rows.push(["section", "Время по этапам"]);
    const labels = {
      extract_audio: "Извлечение аудио",
      download_model: "Скачивание модели",
      load_model: "Загрузка в GPU",
      transcribe: "Транскрибация",
    };
    for (const [key, label] of Object.entries(labels)) {
      if (stats.stages[key]) rows.push([label, `${stats.stages[key]} с`]);
    }
    if (stats.total_seconds) rows.push(["Всего", `${stats.total_seconds} с`]);
  }

  rows.push(["section", "Показатели"]);
  if (meta.duration) rows.push(["Длительность аудио", formatTs(meta.duration)]);
  if (meta.processing_time) rows.push(["Время обработки", `${meta.processing_time.toFixed(1)} с`]);
  if (meta.speed_factor) rows.push(["Скорость", `×${meta.speed_factor}`]);
  if (stats.rtf) rows.push(["RTF", stats.rtf]);
  if (stats.peak_vram_mb) rows.push(["Пик VRAM", `${stats.peak_vram_mb} МБ`]);
  if (meta.segments_count) rows.push(["Сегментов", meta.segments_count]);
  if (snap.text) rows.push(["Символов", snap.text.length.toLocaleString()]);

  let html = "";
  let inGrid = false;
  for (const row of rows) {
    if (row[0] === "section") {
      if (inGrid) { html += "</dl>"; inGrid = false; }
      html += `<div class="detail-section"><h3>${row[1]}</h3>`;
    } else {
      if (!inGrid) { html += '<dl class="detail-grid">'; inGrid = true; }
      html += `<dt>${escapeHtml(row[0])}</dt><dd>${escapeHtml(String(row[1]))}</dd>`;
    }
  }
  if (inGrid) html += "</dl>";
  el.innerHTML = html;
}

function renderDetailNotes(snap) {
  state.detailTags = [...(snap.tags || [])];
  $("#detail-notes").value = snap.notes || "";
  renderTags();
}

function renderTags() {
  const wrap = $("#detail-tags");
  const input = $("#detail-tag-input");
  wrap.querySelectorAll(".tag-chip").forEach(c => c.remove());

  for (const tag of state.detailTags) {
    const chip = document.createElement("span");
    chip.className = "tag-chip";
    chip.innerHTML = `${escapeHtml(tag)}<span class="material-symbols-rounded" data-remove="${escapeHtml(tag)}">close</span>`;
    wrap.insertBefore(chip, input);
  }

  wrap.querySelectorAll("[data-remove]").forEach(el => {
    el.addEventListener("click", () => {
      const tag = el.dataset.remove;
      state.detailTags = state.detailTags.filter(t => t !== tag);
      renderTags();
    });
  });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

async function saveDetailNotes() {
  if (!state.detailJob) return;
  try {
    const r = await fetch(`/api/jobs/${state.detailJob.id}/notes`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tags: state.detailTags,
        notes: $("#detail-notes").value,
      }),
    });
    if (!r.ok) throw new Error("Ошибка сохранения");
    toast("success", "check_circle", "Заметки сохранены");
    refreshHistory();
  } catch (e) {
    toast("error", "error", e.message);
  }
}

function setupDetailModal() {
  $("#detail-close").addEventListener("click", closeJobDetail);
  $("#detail-modal").addEventListener("click", (e) => {
    if (e.target.id === "detail-modal") closeJobDetail();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("#detail-modal").classList.contains("visible")) {
      closeJobDetail();
    }
  });

  $$(".modal-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.dtab;
      $$(".modal-tab").forEach(t => t.classList.toggle("active", t === tab));
      $$(".modal-pane").forEach(p =>
        p.classList.toggle("active", p.dataset.dpane === target)
      );
    });
  });

  // Ввод тега
  $("#detail-tag-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.value.trim()) {
      e.preventDefault();
      const tag = e.target.value.trim().slice(0, 32).toLowerCase();
      if (!state.detailTags.includes(tag) && state.detailTags.length < 16) {
        state.detailTags.push(tag);
        renderTags();
      }
      e.target.value = "";
    }
  });

  $("#detail-save-notes").addEventListener("click", saveDetailNotes);

  // Экспорт из модалки
  $("#d-act-copy").addEventListener("click", () => {
    const snap = state.detailJob;
    if (!snap || !snap.segments) return;
    const txt = snap.segments.map(s => s.text).join("\n");
    navigator.clipboard.writeText(txt);
    toast("success", "content_copy", t("toast.copied"));
  });

  $("#d-act-txt").addEventListener("click", () => {
    const snap = state.detailJob;
    if (!snap) return;
    const txt = (snap.segments || []).map(s => s.text).join("\n\n");
    downloadBlob(txt, snap.filename.replace(/\.[^.]+$/, "") + ".txt", "text/plain");
  });

  $("#d-act-srt").addEventListener("click", () => {
    const snap = state.detailJob;
    if (!snap || !snap.segments) return;
    const srt = snap.segments.map((s, i) =>
      `${i+1}\n${fmtSrt(s.start)} --> ${fmtSrt(s.end)}\n${s.text}\n`
    ).join("\n");
    downloadBlob(srt, snap.filename.replace(/\.[^.]+$/, "") + ".srt", "application/x-subrip");
  });

  $("#d-act-json").addEventListener("click", () => {
    const snap = state.detailJob;
    if (!snap) return;
    downloadBlob(
      JSON.stringify({
        text: snap.text,
        segments: snap.segments,
        metadata: snap.metadata,
        source_metadata: snap.source_metadata,
        processing_stats: snap.processing_stats,
        tags: snap.tags,
        notes: snap.notes,
      }, null, 2),
      snap.filename.replace(/\.[^.]+$/, "") + ".json",
      "application/json"
    );
  });
}

/* ============================================================
   Navigation
   ============================================================ */
function setupNav() {
  $$(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
      const v = btn.dataset.view;
      $$(".nav-item").forEach(b => b.classList.toggle("active", b === btn));
      $$(".view").forEach(s => s.classList.toggle("active", s.dataset.view === v));

      closeSidebar();

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

/* ============================================================
   Sidebar (mobile drawer)
   ============================================================ */
function openSidebar() {
  if (window.innerWidth > 900) return;
  $(".app").classList.add("sidebar-open");
  $("#sidebar-overlay").classList.add("visible");
  document.body.style.overflow = "hidden";
}

function closeSidebar() {
  $(".app").classList.remove("sidebar-open");
  $("#sidebar-overlay").classList.remove("visible");
  document.body.style.overflow = "";
}

function setupSidebar() {
  $("#hamburger")?.addEventListener("click", openSidebar);
  $("#sidebar-overlay")?.addEventListener("click", closeSidebar);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSidebar();
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (window.innerWidth > 900) {
        closeSidebar();
        document.body.style.overflow = "";
      }
    }, 150);
  });
}

/* ============================================================
   Init
   ============================================================ */
document.addEventListener("DOMContentLoaded", async () => {
  await loadLocales();
  await loadSettings();

  document.documentElement.dataset.theme = state.settings["ui.theme"] || "dark";
  state.theme = state.settings["ui.theme"] || "dark";
  $$("[data-theme-set]").forEach(b =>
    b.classList.toggle("active", b.dataset.themeSet === state.theme)
  );
  $$("[data-theme-set]").forEach(b =>
    b.addEventListener("click", () => setTheme(b.dataset.themeSet))
  );

  setupNav();
  setupSidebar();
  setupDropzone();
  setupConsole();
  setupDetailModal();
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

  const hours = state.settings["server.retention_hours"] || 24;
  $("#retention-info").textContent = t("history.retention", { hours });

  setInterval(loadSystem, 5000);
  refreshHistory();
  setInterval(refreshHistory, 10000);

  await restoreActiveJob();
});
