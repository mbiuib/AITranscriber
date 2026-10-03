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
  historyFilter: { q: "", status: "", favorite: false },
  quickEngine: null,
  files: [],
  dashboardTimer: null,
  selectionMode: false,
  selectedJobs: new Set(),
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
  const info = $("#gpu-info");
  if (!info) return;

  try {
    const r = await fetch("/api/system/resources");
    const d = await r.json();

    if (d.gpu) {
      const text = `${d.gpu.name} · ${d.gpu.memory_used_gb}/${d.gpu.memory_total_gb} GB · ${d.gpu.utilization}%`;
      info.classList.add("ok");
      info.classList.remove("err");
      info.querySelector(".text").textContent = text;
      info.title = text;
    } else {
      info.classList.remove("ok");
      info.classList.add("err");
      info.querySelector(".text").textContent = "GPU недоступна";
      info.title = "CUDA не обнаружена или драйвер не отвечает";
    }
  } catch (e) {
    info.classList.remove("ok");
    info.classList.add("err");
    info.querySelector(".text").textContent = "Сервер недоступен";
    info.title = String(e);
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

  const currentEngine =
    state.settingsDirty["transcription.engine"] ??
    state.settings["transcription.engine"] ??
    "whisper";

  const groups = {};
  for (const [key, def] of Object.entries(state.schema)) {
    if (def.engine && def.engine !== currentEngine) continue;
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
      if (key === "transcription.engine") {
        inp.addEventListener("change", () => {
          state.settingsDirty[key] = inp.value;
          renderSettingsForm();
        });
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

  // Движок — первым полем
  const engineSel = document.createElement("select");
  for (const e of ["whisper", "moss"]) {
    const o = document.createElement("option");
    o.value = e;
    o.textContent = e === "whisper" ? "Whisper + Pyannote" : "MOSS (end-to-end)";
    if (e === getQuickEngine()) o.selected = true;
    engineSel.appendChild(o);
  }
  engineSel.addEventListener("change", () => {
    state.quickEngine = engineSel.value;
    applyQuickOptions();
  });
  q.appendChild(wrapField("Движок", engineSel));

  const engine = getQuickEngine();

  // Язык — для обоих движков
  const langSel = document.createElement("select");
  for (const l of state.schema["transcription.language"].options) {
    const o = document.createElement("option");
    o.value = l;
    o.textContent = l === "auto" ? "auto" : l;
    if (l === state.settings["transcription.language"]) o.selected = true;
    langSel.appendChild(o);
  }
  q.appendChild(wrapField("Язык", langSel));

  let modelSel = null;
  let bsInput = null;

  if (engine === "whisper") {
    // Модель
    modelSel = document.createElement("select");
    for (const m of state.schema["transcription.model"].options) {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = m;
      if (m === state.settings["transcription.model"]) o.selected = true;
      modelSel.appendChild(o);
    }
    q.appendChild(wrapField("Модель", modelSel));

    // Batch size
    bsInput = createNumberInput(
      state.settings["transcription.batch_size"],
      { min: 1, max: 32, step: 1 }
    );
    q.appendChild(wrapField("Batch size", bsInput));
  } else {
    // Для MOSS — подсказка вместо полей
    const hint = document.createElement("div");
    hint.className = "quick-hint";
    hint.innerHTML = `
      <span class="material-symbols-rounded">info</span>
      <span>MOSS обрабатывает аудио чанками по 5 минут.
      Модель и batch size настраиваются автоматически.</span>
    `;
    q.appendChild(hint);
  }

  state.quickRefs = {
    getEngine: () => engine,
    modelSel,
    langSel,
    getBatchSize: () => bsInput ? bsInput.getValue() : null,
  };
}

function getQuickEngine() {
  if (state.quickEngine) return state.quickEngine;
  return state.settings["transcription.engine"] || "whisper";
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
  const confirmed = await showConfirm({
    title: "Сбросить настройки?",
    message: "Все параметры вернутся к значениям по умолчанию из <code>.env</code>.",
    confirmText: "Сбросить",
    icon: "restart_alt",
    danger: true,
  });
  if (!confirmed) return;

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
    toast("info", "refresh", `Восстановлено: ${snap.filename}`);
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
  input.addEventListener("change", () => {
    if (input.files.length) addFiles([...input.files]);
  });

  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); dz.classList.remove("dragover");
  }));
  dz.addEventListener("drop", e => {
    const fs = [...e.dataTransfer.files];
    if (fs.length) addFiles(fs);
  });

  $("#file-list-clear").addEventListener("click", clearFiles);
}

function addFiles(newFiles) {
  for (const f of newFiles) {
    if (!state.files.some(x => x.name === f.name && x.size === f.size)) {
      state.files.push(f);
    }
  }
  renderFileList();
}

function removeFile(index) {
  state.files.splice(index, 1);
  renderFileList();
}

function clearFiles() {
  state.files = [];
  $("#file-input").value = "";
  renderFileList();
}

function renderFileList() {
  const list = $("#file-list");
  const items = $("#file-list-items");
  const count = $("#file-list-count");

  if (!state.files.length) {
    list.hidden = true;
    $("#dropzone").classList.remove("has-file");
    $(".dropzone-title").textContent = "Перетащите файлы сюда";
    $(".dropzone-hint").textContent = "или нажмите для выбора · можно несколько сразу";
    $("#start-btn").disabled = true;
    return;
  }

  list.hidden = false;
  $("#dropzone").classList.add("has-file");
  $(".dropzone-title").textContent = `${state.files.length} ${filesLabel(state.files.length)} готово`;
  $(".dropzone-hint").textContent = "Нажмите, чтобы добавить ещё";
  $("#start-btn").disabled = false;

  count.textContent = `${state.files.length} ${filesLabel(state.files.length)}`;
  items.innerHTML = "";

  state.files.forEach((f, i) => {
    const el = document.createElement("div");
    el.className = "file-list-item";
    el.innerHTML = `
      <span class="material-symbols-rounded">draft</span>
      <span class="name"></span>
      <span class="size">${formatSize(f.size)}</span>
      <button class="remove" title="Убрать">
        <span class="material-symbols-rounded">close</span>
      </button>
    `;
    el.querySelector(".name").textContent = f.name;
    el.querySelector(".remove").addEventListener("click", (e) => {
      e.stopPropagation();
      removeFile(i);
    });
    items.appendChild(el);
  });
}

function filesLabel(n) {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return "файл";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return "файла";
  return "файлов";
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
async function start(forceReprocess = false) {
  if (!state.files.length || state.running) return;
  resetResult();

  state.running = true;
  $("#start-btn").disabled = true;
  $("#cancel-btn").disabled = false;
  ensureConsoleOpen();

    const fd = new FormData();
  for (const f of state.files) {
    fd.append("files", f);
  }
  fd.append("engine", state.quickRefs.getEngine());
  if (state.quickRefs.modelSel) {
    fd.append("model", state.quickRefs.modelSel.value);
  }
  fd.append("language", state.quickRefs.langSel.value);
  if (forceReprocess) fd.append("check_duplicate", "false");

  try {
    const r = await fetch("/api/jobs/batch", { method: "POST", body: fd });

    if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
    const data = await r.json();

    const created = data.results.filter(x => x.status === "created");
    const duplicates = data.results.filter(x => x.status === "duplicate");
    const errors = data.results.filter(x => x.status === "error");

    if (created.length) {
      const first = created[0];
      state.jobId = first.job_id;
      saveActiveJob(first.job_id);
      subscribe(first.job_id);
      toast("success", "check_circle",
        `Создано задач: ${created.length}`);
    }

    if (duplicates.length) {
      const dup = duplicates[0].duplicate_of;
      const choice = await showDuplicateModal(dup);
      if (choice === "open") {
        openJobDetail(dup.id);
      } else if (choice === "reprocess") {
        // Обрабатываем повторно только дубликаты
        await reprocessDuplicates(duplicates);
      }
    }

    if (errors.length) {
      toast("error", "error", `Ошибок: ${errors.length}. Первая: ${errors[0].message}`);
    }

    clearFiles();
    refreshHistory();
    updateMiniDash();
  } catch (e) {
    toast("error", "error", e.message);
  } finally {
    state.running = false;
    $("#start-btn").disabled = !state.files.length;
    $("#cancel-btn").disabled = true;
  }
}

async function reprocessDuplicates(duplicates) {
  // Отправляем по одной без проверки дубликатов
  for (const item of duplicates) {
    if (state.running) return;
    try {
      const f = state.files.find(x => x.name === item.filename);
      if (!f) continue;

      const fd = new FormData();
      fd.append("file", f);
      fd.append("model", state.quickRefs.modelSel.value);
      fd.append("language", state.quickRefs.langSel.value);
      fd.append("check_duplicate", "false");

      await fetch("/api/jobs", { method: "POST", body: fd });
    } catch {}
  }
  refreshHistory();
}

async function cancelJob() {
  if (!state.jobId) return;

  const confirmed = await showConfirm({
    title: "Прервать транскрибацию?",
    message: "Уже обработанные сегменты сохранятся, но полный результат будет недоступен.",
    confirmText: "Прервать",
    icon: "stop_circle",
    danger: true,
  });
  if (!confirmed) return;

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
    case "segments_replaced": onSegmentsReplaced(ev.segments); break;
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

function onSegmentsReplaced(segments) {
  if (!segments || !segments.length) return;

  const rc = $("#result-content");
  const wasAtBottom = isNearBottom(rc);
  const prevSeen = state.seenSegmentsCount;

  state.segments = [];
  $("#result-content").innerHTML = '<div class="segments" id="segments"></div>';

  // silent=true — не трогаем seenSegmentsCount и не дёргаем скролл
  for (const s of segments) {
    appendSegment(s, true);
  }

  if (wasAtBottom) {
  // Не сбрасываем seenSegmentsCount — если пользователь читает выше,
  // он должен увидеть кнопку с числом непросмотренных сегментов
  // даже после завершения задачи
  if (isNearBottom($("#result-content"))) {
    state.seenSegmentsCount = state.segments.length;
  }
    rc.scrollTop = rc.scrollHeight;
  } else {
    // Пользователь читает выше — ограничиваем seen предыдущим
    // значением, чтобы новые сегменты считались непросмотренными
    state.seenSegmentsCount = Math.min(prevSeen, state.segments.length);
  }

  updateScrollButton();

  const unread = state.segments.length - state.seenSegmentsCount;
  if (unread > 0) {
    toast("info", "group",
      `Сегменты обновлены: ${segments.length}, новых: ${unread}`);
  } else {
    toast("info", "group",
      `Сегменты обновлены: ${segments.length} фрагментов`);
  }
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
  if (unread > 0) {
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

  if (seg.speaker) {
    const chip = document.createElement("span");
    chip.className = "speaker";
    chip.textContent = speakerName(seg.speaker, null);
    chip.style.setProperty("--speaker-color", speakerColor(seg.speaker));
    tx.appendChild(chip);
  }
  tx.appendChild(document.createTextNode(seg.text));

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

const SPEAKER_COLORS = [
  "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8",
  "#cba6f7", "#94e2d5", "#fab387", "#74c7ec",
];

function speakerColor(speaker) {
  if (!speaker || speaker === "UNKNOWN") return "var(--text-3)";
  let h = 0;
  for (let i = 0; i < speaker.length; i++) {
    h = (h * 31 + speaker.charCodeAt(i)) | 0;
  }
  return SPEAKER_COLORS[Math.abs(h) % SPEAKER_COLORS.length];
}

function speakerName(speaker, mapping) {
  if (mapping && mapping[speaker]) return mapping[speaker];
  if (!speaker || speaker === "UNKNOWN") return "—";
  if (speaker.startsWith("SPEAKER_")) {
    const idx = parseInt(speaker.split("_")[1], 10);
    if (!isNaN(idx)) return `Спикер ${idx + 1}`;
  }
  return speaker;
}

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
   Export (главное окно)
   ============================================================ */
async function copyText() {
  if (!state.segments.length) return;
  const mapping = state.metadata?.speaker_names || {};
  const txt = state.segments.map(s => {
    const name = s.speaker ? speakerName(s.speaker, mapping) + ": " : "";
    return name + s.text;
  }).join("\n");
  await navigator.clipboard.writeText(txt);
  toast("success", "content_copy", t("toast.copied"));
}

function downloadTxt() {
  const txt = state.segments.map(s => s.text).join("\n\n");
  downloadBlob(txt, baseName() + ".txt", "text/plain");
}

function downloadSrt() {
  const mapping = state.metadata?.speaker_names || {};
  const srt = state.segments.map((s, i) => {
    const name = s.speaker ? speakerName(s.speaker, mapping) + ": " : "";
    return `${i+1}\n${fmtSrt(s.start)} --> ${fmtSrt(s.end)}\n${name}${s.text}\n`;
  }).join("\n");
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
    const f = state.historyFilter || {};
    const params = new URLSearchParams();
    if (f.q) params.set("q", f.q);
    if (f.status) params.set("status", f.status);
    if (f.favorite) params.set("favorite", "true");

    const url = "/api/jobs" + (params.toString() ? "?" + params : "");
    const r = await fetch(url);
    const d = await r.json();
    state.jobs = d.jobs;

    // Чистим выделение от исчезнувших задач
    const existingIds = new Set(state.jobs.map(j => j.id));
    for (const id of state.selectedJobs) {
      if (!existingIds.has(id)) state.selectedJobs.delete(id);
    }

    state.queue = {
      size: d.queue_size || 0,
      running: d.queue_running || 0,
      max: d.queue_max_parallel || 1,
    };
    renderHistory();
    updateMiniDash();

    const badge = $("#nav-badge-history");
    const active = d.active_count || 0;
    badge.hidden = active === 0;
    badge.textContent = active;
  } catch {}
}

function updateMiniDash() {
  const q = state.queue || { size: 0, running: 0, max: 1 };
  $("#mini-running").textContent = q.running;
  $("#mini-queued").textContent = q.size;
  $("#mini-max").textContent = q.max;

  const total = q.running + q.size;
  const pct = q.max > 0 ? Math.min(100, Math.round((q.running / q.max) * 100)) : 0;
  $("#mini-fill").style.width = pct + "%";
}

async function loadDashboard() {
  try {
    const r = await fetch("/api/dashboard");
    const d = await r.json();

    renderDashKpi(d);
    renderDashChart(d.by_day || []);
    renderDashTopModels(d.top_models || []);
    renderDashTopLanguages(d.top_languages || []);

    $("#dashboard-updated").textContent =
      "Обновлено " + new Date().toLocaleTimeString("ru-RU", { hour12: false });
  } catch (e) {
    toast("error", "error", "Не удалось загрузить дашборд");
  }
}

function renderDashKpi(d) {
  const el = $("#dash-kpi-grid");
  const status = d.by_status || {};
  const total = d.total || 0;
  const done = status.done || 0;
  const error = status.error || 0;
  const active = (status.queued || 0) + (status.pending || 0)
    + (status.downloading || 0) + (status.loading || 0)
    + (status.transcribing || 0);
  const starred = d.starred_count || 0;

  const audioH = ((d.total_audio_seconds || 0) / 3600).toFixed(1);
  const procH = ((d.total_processing_seconds || 0) / 60).toFixed(1);
  const rtf = d.avg_rtf != null ? `RTF ${d.avg_rtf}` : "—";

  el.innerHTML = `
    ${kpiCard("library_books", "Всего задач", total, `${done} завершено · ${error} ошибок`)}
    ${kpiCard("schedule", "Активные", active, `В очереди и обработке`, active > 0 ? "accent" : "")}
    ${kpiCard("graphic_eq", "Часов аудио", audioH, `${procH} мин обработки`)}
    ${kpiCard("bolt", "Средний RTF", rtf, "Real-time factor")}
    ${kpiCard("star", "Избранных", starred, "", "warn")}
  `;
}

function kpiCard(icon, label, value, sub, variant = "") {
  const iconCls = variant === "warn" ? "warn" : variant === "accent" ? "" : "success";
  return `
    <div class="dash-kpi">
      <div class="dash-kpi-top">
        <div class="dash-kpi-icon ${iconCls}">
          <span class="material-symbols-rounded">${icon}</span>
        </div>
      </div>
      <div class="dash-kpi-value">${value}</div>
      <div class="dash-kpi-label">${label}</div>
      ${sub ? `<div class="dash-kpi-sub">${sub}</div>` : ""}
    </div>
  `;
}

function renderDashChart(byDay) {
  const el = $("#dash-chart");
  if (!byDay.length) {
    el.innerHTML = '<div class="dash-chart-empty">Нет данных за последние 30 дней</div>';
    return;
  }

  const maxVal = Math.max(...byDay.map(d => d.count), 1);
  const w = 800;
  const h = 180;
  const padL = 30;
  const padR = 10;
  const padT = 10;
  const padB = 30;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  const barW = Math.max(4, (chartW / byDay.length) - 4);
  const step = chartW / byDay.length;

  let bars = "";
  let labels = "";
  byDay.forEach((d, i) => {
    const x = padL + i * step + (step - barW) / 2;
    const totalH = (d.count / maxVal) * chartH;
    const doneH = d.done > 0 ? (d.done / maxVal) * chartH : 0;
    const errorH = d.error > 0 ? (d.error / maxVal) * chartH : 0;

    const y = padT + chartH - totalH;
    bars += `<rect x="${x}" y="${padT + chartH - doneH}" width="${barW}" height="${doneH}"
             fill="var(--success)" opacity="0.85" rx="2"/>`;
    if (errorH > 0) {
      bars += `<rect x="${x}" y="${padT + chartH - doneH - errorH}"
               width="${barW}" height="${errorH}"
               fill="var(--error)" opacity="0.85" rx="2"/>`;
    }

    if (i % Math.ceil(byDay.length / 8) === 0 || i === byDay.length - 1) {
      const date = new Date(d.date * 1000);
      const label = `${date.getDate()}.${String(date.getMonth() + 1).padStart(2, "0")}`;
      labels += `<text x="${x + barW / 2}" y="${h - 8}" text-anchor="middle"
                 font-size="10" fill="var(--text-3)">${label}</text>`;
    }
  });

  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH / 4) * i;
    const val = Math.round(maxVal - (maxVal / 4) * i);
    grid += `<line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}"
             stroke="var(--border)" stroke-dasharray="2 4"/>`;
    grid += `<text x="${padL - 6}" y="${y + 3}" text-anchor="end"
             font-size="9" fill="var(--text-3)">${val}</text>`;
  }

  el.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      ${grid}
      ${bars}
      ${labels}
    </svg>
  `;
}

function renderDashTopModels(models) {
  const el = $("#dash-top-models");
  if (!models.length) {
    el.innerHTML = '<div class="dash-list-empty">Нет данных</div>';
    return;
  }
  const max = Math.max(...models.map(m => m.count), 1);
  el.innerHTML = models.map(m => `
    <div class="dash-list-item">
      <span class="material-symbols-rounded" style="font-size:16px;color:var(--accent)">memory</span>
      <span class="dash-list-name">${escapeHtml(m.model)}</span>
      <span class="dash-list-bar">
        <span class="dash-list-bar-fill" style="width:${(m.count / max) * 100}%"></span>
      </span>
      <span class="dash-list-count">${m.count}</span>
    </div>
  `).join("");
}

function renderDashTopLanguages(langs) {
  const el = $("#dash-top-languages");
  if (!langs.length) {
    el.innerHTML = '<div class="dash-list-empty">Нет данных</div>';
    return;
  }
  const max = Math.max(...langs.map(l => l.count), 1);
  el.innerHTML = langs.map(l => `
    <div class="dash-list-item">
      <span class="material-symbols-rounded" style="font-size:16px;color:var(--accent)">translate</span>
      <span class="dash-list-name">${escapeHtml(l.language.toUpperCase())}</span>
      <span class="dash-list-bar">
        <span class="dash-list-bar-fill" style="width:${(l.count / max) * 100}%"></span>
      </span>
      <span class="dash-list-count">${l.count}</span>
    </div>
  `).join("");
}

function startDashboardMonitor() {
  stopDashboardMonitor();
  loadDashboard();
  state.dashboardTimer = setInterval(loadDashboard, 5000);
}
function stopDashboardMonitor() {
  if (state.dashboardTimer) clearInterval(state.dashboardTimer);
  state.dashboardTimer = null;
}

function updateQueueIndicator() {
  const el = $("#queue-indicator");
  if (!el) return;
  const q = state.queue || { size: 0, running: 0, max: 1 };
  if (q.size === 0 && q.running === 0) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.querySelector(".queue-running").textContent = q.running;
  el.querySelector(".queue-size").textContent = q.size;
  el.querySelector(".queue-max").textContent = q.max;
}

function renderHistory() {
  const list = $("#history-list");
  if (!state.jobs.length) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon material-symbols-rounded">history_toggle_off</div>
        <div class="empty-text">${t("history.empty")}</div>
      </div>`;
    updateBulkActions();
    return;
  }
  list.innerHTML = "";
  for (const j of state.jobs) {
    const item = document.createElement("div");
    item.className = "history-item";
    item.style.cursor = "pointer";
    if (state.selectedJobs.has(j.id)) item.classList.add("selected");

    const created = new Date(j.created_at * 1000).toLocaleString("ru-RU");
    const dur = j.duration ? formatTs(j.duration) : "—";
    const tagsHtml = (j.tags || []).length
      ? `<div class="history-tags">${j.tags.map(tag =>
          `<span class="history-tag">${escapeHtml(tag)}</span>`).join("")}</div>`
      : "";

    let statusHtml;
    if (j.status === "queued" && j.queue_position) {
      statusHtml = `<div class="history-status queued">
        <span class="material-symbols-rounded">schedule</span>
        #${j.queue_position}
      </div>`;
    } else {
      statusHtml = `<div class="history-status ${j.status}">${t("stage." + j.status) || j.status}</div>`;
    }

    const checkboxHtml = state.selectionMode
      ? `<button class="history-select ${state.selectedJobs.has(j.id) ? "checked" : ""}">
           <span class="material-symbols-rounded">${state.selectedJobs.has(j.id) ? "check" : "radio_button_unchecked"}</span>
         </button>`
      : "";

    item.innerHTML = `
      ${checkboxHtml}
      <button class="history-star ${j.starred ? "starred" : ""}" title="В избранное">
        <span class="material-symbols-rounded">star</span>
      </button>
      <div class="history-icon">
        <span class="material-symbols-rounded">movie</span>
      </div>
      <div class="history-main">
        <div class="history-name"></div>
        <div class="history-meta">${created} · ${j.model} · ${dur} · ${j.segments_count} сегм.</div>
        ${tagsHtml}
      </div>
      ${statusHtml}
      <button class="btn-icon" data-action="delete" title="Удалить">
        <span class="material-symbols-rounded">delete</span>
      </button>
    `;
    item.querySelector(".history-name").textContent = j.filename;

    const selectBtn = item.querySelector(".history-select");
    if (selectBtn) {
      selectBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleSelect(j.id);
      });
    }

    item.querySelector(".history-star").addEventListener("click", (e) => {
      e.stopPropagation();
      toggleStar(j.id, j.starred);
    });

    item.addEventListener("click", (e) => {
      if (e.target.closest('[data-action="delete"]')) return;
      if (e.target.closest(".history-star")) return;
      if (e.target.closest(".history-select")) return;

      if (state.selectionMode) {
        toggleSelect(j.id);
      } else {
        openJobDetail(j.id);
      }
    });

    item.querySelector('[data-action="delete"]').addEventListener("click", async (e) => {
      e.stopPropagation();

      if (state.selectionMode) {
        toggleSelect(j.id);
        return;
      }

      const confirmed = await showConfirm({
        title: "Удалить задачу?",
        message: `Файл <strong>${escapeHtml(j.filename)}</strong> и все его данные будут удалены безвозвратно.`,
        confirmText: "Удалить",
        icon: "delete",
        danger: true,
      });
      if (!confirmed) return;

      await fetch(`/api/jobs/${j.id}`, { method: "DELETE" });
      toast("success", "delete", t("toast.job_deleted"));
      refreshHistory();
    });

    list.appendChild(item);
  }
  updateBulkActions();
}

function toggleSelect(jobId) {
  if (state.selectedJobs.has(jobId)) {
    state.selectedJobs.delete(jobId);
  } else {
    state.selectedJobs.add(jobId);
  }
  renderHistory();
}

function toggleSelectionMode() {
  state.selectionMode = !state.selectionMode;
  state.selectedJobs.clear();

  const btn = $("#history-select-toggle");
  btn.classList.toggle("active", state.selectionMode);
  btn.querySelector("span:last-child").textContent =
    state.selectionMode ? "Готово" : "Выбрать";

  renderHistory();
}

function updateBulkActions() {
  const el = $("#bulk-actions");
  if (!state.selectionMode) {
    el.hidden = true;
    return;
  }

  el.hidden = false;
  const n = state.selectedJobs.size;
  $("#bulk-count").textContent = n;
}

function selectAll() {
  const allSelected = state.selectedJobs.size === state.jobs.length;
  if (allSelected) {
    state.selectedJobs.clear();
  } else {
    state.jobs.forEach(j => state.selectedJobs.add(j.id));
  }
  renderHistory();
}

async function bulkDelete() {
  const ids = [...state.selectedJobs];
  if (!ids.length) return;

  const confirmed = await showConfirm({
    title: `Удалить ${ids.length} ${jobsLabel(ids.length)}?`,
    message: "Все выбранные задачи и их файлы будут удалены безвозвратно. " +
             "Это действие нельзя отменить.",
    confirmText: "Удалить всё",
    icon: "delete_sweep",
    danger: true,
  });
  if (!confirmed) return;

  try {
    const r = await fetch("/api/jobs/batch/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    if (!r.ok) throw new Error("Ошибка удаления");
    const d = await r.json();

    toast("success", "delete",
      `Удалено: ${d.deleted}${d.not_found.length ? `, не найдено: ${d.not_found.length}` : ""}`);

    state.selectedJobs.clear();
    state.selectionMode = false;
    const btn = $("#history-select-toggle");
    btn.classList.remove("active");
    btn.querySelector("span:last-child").textContent = "Выбрать";

    refreshHistory();
  } catch (e) {
    toast("error", "error", e.message);
  }
}

async function bulkStar(value) {
  const ids = [...state.selectedJobs];
  if (!ids.length) return;

  try {
    await Promise.all(ids.map(id =>
      fetch(`/api/jobs/${id}/star`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ starred: value }),
      })
    ));
    toast("success", "star",
      `${value ? "Добавлено в избранное" : "Убрано из избранного"}: ${ids.length}`);

    state.selectedJobs.clear();
    state.selectionMode = false;
    const btn = $("#history-select-toggle");
    btn.classList.remove("active");
    btn.querySelector("span:last-child").textContent = "Выбрать";

    refreshHistory();
  } catch (e) {
    toast("error", "error", "Ошибка сохранения");
  }
}

function jobsLabel(n) {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return "задачу";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return "задачи";
  return "задач";
}

function setupBulkActions() {
  $("#history-select-toggle").addEventListener("click", toggleSelectionMode);
  $("#bulk-delete").addEventListener("click", bulkDelete);
  $("#bulk-cancel").addEventListener("click", () => {
    state.selectedJobs.clear();
    state.selectionMode = false;
    const btn = $("#history-select-toggle");
    btn.classList.remove("active");
    btn.querySelector("span:last-child").textContent = "Выбрать";
    renderHistory();
  });
  $("#bulk-star").addEventListener("click", () => bulkStar(true));
  $("#bulk-unstar").addEventListener("click", () => bulkStar(false));
}

function setupHistoryFilters() {
  const q = $("#filter-q");
  const status = $("#filter-status");
  const fav = $("#filter-favorite");
  const reset = $("#filter-reset");

  let debounce;
  q.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      state.historyFilter.q = q.value.trim();
      refreshHistory();
    }, 250);
  });

  status.addEventListener("change", () => {
    state.historyFilter.status = status.value;
    refreshHistory();
  });

  fav.addEventListener("click", () => {
    state.historyFilter.favorite = !state.historyFilter.favorite;
    fav.classList.toggle("active", state.historyFilter.favorite);
    refreshHistory();
  });

  reset.addEventListener("click", () => {
    state.historyFilter = { q: "", status: "", favorite: false };
    q.value = "";
    status.value = "";
    fav.classList.remove("active");
    refreshHistory();
  });
}

async function toggleStar(jobId, current) {
  try {
    const r = await fetch(`/api/jobs/${jobId}/star`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ starred: !current }),
    });
    if (!r.ok) throw new Error("Ошибка сохранения");
    const d = await r.json();
    const job = state.jobs.find(j => j.id === jobId);
    if (job) job.starred = d.starred;
    renderHistory();
    return d.starred;
  } catch (e) {
    toast("error", "error", e.message);
    return current;
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
      if (v === "dashboard") startDashboardMonitor();
      else stopDashboardMonitor();
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

  const star = $("#detail-star");
  star.classList.toggle("starred", !!snap.starred);

  const tr = $("#detail-transcript");
  if (snap.segments && snap.segments.length) {
    tr.innerHTML = "";
    for (const s of snap.segments) {
      tr.appendChild(buildSegmentRow(s));
    }
  } else if (snap.text) {
    tr.textContent = snap.text;
  } else {
    tr.innerHTML = `<div class="empty-state"><div class="empty-icon material-symbols-rounded">hourglass_empty</div><div class="empty-text">Результат ещё не готов</div></div>`;
  }

  renderDetailMeta(snap);
  renderDetailStats(snap);
  renderDetailNotes(snap);
  renderDetailSpeakers(snap);
}

/**
 * Создаёт DOM-строку одного сегмента с обработчиком редактирования.
 * Клик по тексту запускает inline-редактор (textarea).
 *
 * @param {object} seg Объект сегмента {index, start, end, text}.
 * @returns {HTMLElement} Готовая строка.
 */
function buildSegmentRow(seg) {
  const line = document.createElement("div");
  line.className = "segment";

  const g = document.createElement("span");
  g.className = "gutter";
  g.textContent = formatTs(seg.start);

  const tx = document.createElement("span");
  tx.className = "text";

  if (seg.speaker) {
    const mapping = state.detailJob?.metadata?.speaker_names || {};
    const name = speakerName(seg.speaker, mapping);
    const chip = document.createElement("span");
    chip.className = "speaker";
    chip.textContent = name;
    chip.style.setProperty("--speaker-color", speakerColor(seg.speaker));
    tx.appendChild(chip);
  }

  const textNode = document.createTextNode(seg.text);
  tx.appendChild(textNode);

  line.append(g, tx);

  line.addEventListener("click", (e) => {
    if (line.querySelector("textarea.segment-edit")) return;
    if (e.target.tagName === "TEXTAREA") return;
    startSegmentEdit(line, seg);
  });

  return line;
}

function renderDetailSpeakers(snap) {
  const el = $("#detail-speakers");
  const meta = snap.metadata || {};
  const speakers = meta.speakers || [];
  const mapping = meta.speaker_names || {};

  if (!meta.diarization || !speakers.length) {
    el.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon material-symbols-rounded">group_off</div>
        <div class="empty-text">Диаризация не выполнялась</div>
        <div class="empty-hint">Включите её в настройках и запустите транскрибацию заново</div>
      </div>`;
    return;
  }

  el.innerHTML = "";
  for (const sp of speakers) {
    const row = document.createElement("div");
    row.className = "speaker-row";

    const dot = document.createElement("span");
    dot.className = "speaker-dot";
    dot.style.background = speakerColor(sp);

    const key = document.createElement("span");
    key.className = "speaker-key";
    key.textContent = sp;

    const input = document.createElement("input");
    input.type = "text";
    input.value = mapping[sp] || speakerName(sp, null);
    input.placeholder = "Имя спикера";

    input.addEventListener("change", async () => {
      const newName = input.value.trim();
      if (!newName) {
        input.value = mapping[sp] || speakerName(sp, null);
        return;
      }
      try {
        const r = await fetch(
          `/api/jobs/${state.detailJob.id}/speakers/${encodeURIComponent(sp)}`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: newName }),
          }
        );
        if (!r.ok) throw new Error((await r.json()).detail || "Ошибка");
        const d = await r.json();
        state.detailJob.metadata.speaker_names = d.speaker_names;
        toast("success", "check_circle", "Имя сохранено");
        // Перерисовываем транскрипт, чтобы обновлённые имена подхватились
        renderDetail(state.detailJob);
      } catch (e) {
        toast("error", "error", e.message);
      }
    });

    row.append(dot, key, input);
    el.appendChild(row);
  }
}

/**
 * Открывает inline-редактор для сегмента.
 *
 * Enter — сохранить, Shift+Enter — перенос строки, Esc — отменить,
 * blur — сохранить. Высота textarea автоматически подстраивается под
 * содержимое: при открытии и после каждого ввода.
 *
 * @param {HTMLElement} line Строка сегмента.
 * @param {object} seg Объект сегмента.
 */
function startSegmentEdit(line, seg) {
  const textEl = line.querySelector(".text");
  const original = seg.text;

  const ta = document.createElement("textarea");
  ta.className = "segment-edit";
  ta.value = original;
  ta.rows = 1;

  textEl.replaceWith(ta);

  const autoGrow = () => {
    ta.style.height = "auto";
    ta.style.height = ta.scrollHeight + "px";
  };

  autoGrow();
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);

  ta.addEventListener("input", autoGrow);

  let cancelled = false;

  const finish = async (save) => {
    const newText = ta.value.trim();

    if (save && newText && newText !== original) {
      try {
        const r = await fetch(
          `/api/jobs/${state.detailJob.id}/segments/${seg.index}`,
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: newText }),
          }
        );
        if (!r.ok) throw new Error((await r.json()).detail || "Ошибка сохранения");
        const d = await r.json();
        seg.text = newText;
        state.detailJob.text = d.text;
        toast("success", "check_circle", "Сегмент обновлён");
      } catch (e) {
        toast("error", "error", e.message);
      }
    }

    const span = document.createElement("span");
    span.className = "text";
    span.textContent = seg.text;
    ta.replaceWith(span);
  };

  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ta.blur();
    } else if (e.key === "Escape") {
      e.preventDefault();
      cancelled = true;
      ta.blur();
    }
  });

  ta.addEventListener("blur", () => finish(!cancelled));
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
  rows.push(["Движок", settings.engine || meta.engine || "whisper"]);
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

  $("#detail-star").addEventListener("click", async () => {
    if (!state.detailJob) return;
    const newVal = await toggleStar(state.detailJob.id, !!state.detailJob.starred);
    state.detailJob.starred = newVal;
    $("#detail-star").classList.toggle("starred", newVal);
  });

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
      const mapping = snap.metadata?.speaker_names || {};
      const srt = snap.segments.map((s, i) => {
        const name = s.speaker ? speakerName(s.speaker, mapping) + ": " : "";
        return `${i+1}\n${fmtSrt(s.start)} --> ${fmtSrt(s.end)}\n${name}${s.text}\n`;
      }).join("\n");
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

/**
 * Показывает модалку о найденном дубликате файла.
 *
 * Возвращает Promise, который резолвится одним из значений:
 *   "open"       — открыть существующий результат
 *   "reprocess"  — обработать заново
 *   null         — закрыть без действия
 *
 * Закрытие через Esc, клик по overlay или по кнопке X резолвит null.
 *
 * @param {object} dup Информация о дубликате: id, filename, created_at,
 *                     duration, segments_count.
 * @returns {Promise<"open"|"reprocess"|null>}
 */
function showDuplicateModal(dup) {
  return new Promise((resolve) => {
    const modal = $("#duplicate-modal");
    const elFilename = $("#dup-filename");
    const elMeta = $("#dup-meta");

    elFilename.textContent = dup.filename;
    elFilename.title = dup.filename;

    const created = new Date(dup.created_at * 1000).toLocaleString("ru-RU", {
      day: "2-digit", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
    const dur = dup.duration ? formatTs(dup.duration) : "—";
    const segs = dup.segments_count != null ? dup.segments_count : "—";
    elMeta.textContent = `${created}  ·  ${dur}  ·  ${segmentsLabel(segs)}`;

    const cleanup = () => {
      modal.classList.remove("visible");
      document.body.style.overflow = "";
      $("#duplicate-close").removeEventListener("click", onClose);
      $("#dup-open").removeEventListener("click", onOpen);
      $("#dup-reprocess").removeEventListener("click", onReprocess);
      modal.removeEventListener("click", onOverlay);
      document.removeEventListener("keydown", onKey);
    };

    const finish = (result) => {
      cleanup();
      resolve(result);
    };

    const onClose = () => finish(null);
    const onOpen = () => finish("open");
    const onReprocess = () => finish("reprocess");

    const onOverlay = (e) => {
      if (e.target === modal) finish(null);
    };

    const onKey = (e) => {
      if (e.key === "Escape") finish(null);
    };

    $("#duplicate-close").addEventListener("click", onClose);
    $("#dup-open").addEventListener("click", onOpen);
    $("#dup-reprocess").addEventListener("click", onReprocess);
    modal.addEventListener("click", onOverlay);
    document.addEventListener("keydown", onKey);

    modal.classList.add("visible");
    document.body.style.overflow = "hidden";

    // Автофокус на основную кнопку — чтобы Enter сработал сразу
    setTimeout(() => $("#dup-open").focus(), 50);
  });
}

/**
 * Универсальная confirm-модалка, заменяющая системный confirm().
 *
 * Возвращает Promise<boolean>: true — пользователь подтвердил,
 * false — отменил или закрыл модалку любым способом (Esc, клик
 * по overlay, крестик).
 *
 * Enter — подтвердить, Esc — отменить. Автофокус на кнопку действия
 * после появления модалки, чтобы можно было сразу нажать Enter.
 *
 * @param {object} opts Параметры диалога.
 * @param {string} opts.title Заголовок.
 * @param {string} [opts.message] Текст. Поддерживает HTML — экранируйте
 *     пользовательские данные через escapeHtml() самостоятельно.
 * @param {string} [opts.confirmText="Подтвердить"] Текст кнопки действия.
 * @param {string} [opts.cancelText="Отмена"] Текст кнопки отмены.
 * @param {string} [opts.icon="help"] Имя Material Symbol для заголовка.
 * @param {boolean} [opts.danger=false] Красная кнопка для деструктивных
 *     действий (удаление, сброс, прерывание).
 * @returns {Promise<boolean>}
 */
function showConfirm({
  title,
  message = "",
  confirmText = "Подтвердить",
  cancelText = "Отмена",
  icon = "help",
  danger = false,
} = {}) {
  return new Promise((resolve) => {
    const modal = $("#confirm-modal");
    const iconEl = $("#confirm-icon");
    const titleEl = $("#confirm-title");
    const msgEl = $("#confirm-message");
    const okBtn = $("#confirm-ok");
    const cancelBtn = $("#confirm-cancel");

    iconEl.textContent = icon;
    titleEl.textContent = title;
    msgEl.innerHTML = message;
    okBtn.textContent = confirmText;
    cancelBtn.textContent = cancelText;
    modal.classList.toggle("danger", danger);

    const cleanup = () => {
      modal.classList.remove("visible");
      document.body.style.overflow = "";
      $("#confirm-close").removeEventListener("click", onCancel);
      okBtn.removeEventListener("click", onOk);
      cancelBtn.removeEventListener("click", onCancel);
      modal.removeEventListener("click", onOverlay);
      document.removeEventListener("keydown", onKey);
    };

    const finish = (result) => {
      cleanup();
      resolve(result);
    };

    const onOk = () => finish(true);
    const onCancel = () => finish(false);

    const onOverlay = (e) => {
      if (e.target === modal) finish(false);
    };

    const onKey = (e) => {
      if (e.key === "Escape") finish(false);
      else if (e.key === "Enter") finish(true);
    };

    $("#confirm-close").addEventListener("click", onCancel);
    okBtn.addEventListener("click", onOk);
    cancelBtn.addEventListener("click", onCancel);
    modal.addEventListener("click", onOverlay);
    document.addEventListener("keydown", onKey);

    modal.classList.add("visible");
    document.body.style.overflow = "hidden";

    setTimeout(() => okBtn.focus(), 50);
  });
}

/**
 * Возвращает правильную форму слова «сегмент».
 *
 * @param {number} n Количество сегментов.
 * @returns {string} Строка вида "12 сегментов".
 */
function segmentsLabel(n) {
  if (typeof n !== "number") return "";
  const mod10 = n % 10;
  const mod100 = n % 100;
  let word = "сегментов";
  if (mod10 === 1 && mod100 !== 11) word = "сегмент";
  else if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) word = "сегмента";
  return `${n} ${word}`;
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
  setupResultScroll();
  setupDetailModal();
  setupHistoryFilters();
  setupBulkActions();
  await loadSystem();

  $("#start-btn").addEventListener("click", () => start(false));
  $("#cancel-btn").addEventListener("click", cancelJob);
  $("#act-copy").addEventListener("click", copyText);
  $("#act-txt").addEventListener("click", downloadTxt);
  $("#act-srt").addEventListener("click", downloadSrt);
  $("#act-json").addEventListener("click", downloadJson);
  $("#settings-save").addEventListener("click", saveSettings);
  $("#settings-reset").addEventListener("click", resetSettings);
  $("#history-refresh").addEventListener("click", refreshHistory);
  $("#dashboard-refresh").addEventListener("click", loadDashboard);

  const hours = state.settings["server.retention_hours"] || 24;
  $("#retention-info").textContent = t("history.retention", { hours });

  setInterval(loadSystem, 5000);
  refreshHistory();
  setInterval(refreshHistory, 10000);

  await restoreActiveJob();
});
