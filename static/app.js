/* ============================================================
   Whisper Transcriber — клиентская логика
   ============================================================ */

const $ = (sel) => document.querySelector(sel);

// ---------- Состояние ----------
const state = {
  file: null,
  jobId: null,
  eventSource: null,
  segments: [],
  text: "",
  metadata: {},
  running: false,
  consoleOpen: true,
  logCount: 0,
};

// ============================================================
// Инициализация
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
  await loadSystemInfo();
  await loadModels();

  setupDropzone();
  setupOptions();
  setupActions();
  setupConsole();
  setupResult();
});

// ============================================================
// Загрузка информации о системе
// ============================================================

async function loadSystemInfo() {
  try {
    const r = await fetch("/api/system");
    const info = await r.json();
    const el = $("#gpu-info");

    if (info.cuda) {
      el.classList.add("ok");
      el.querySelector(".text").textContent = `${info.gpu} · ${info.vram} GB`;
    } else {
      el.classList.add("err");
      el.querySelector(".text").textContent = "CUDA недоступна";
    }
  } catch (e) {
    console.error(e);
  }
}

async function loadModels() {
  try {
    const r = await fetch("/api/models");
    const data = await r.json();

    const modelSel = $("#opt-model");
    modelSel.innerHTML = "";
    for (const m of data.models) {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = m;
      if (m === data.default) o.selected = true;
      modelSel.appendChild(o);
    }

    const langSel = $("#opt-language");
    langSel.innerHTML = "";
    for (const [code, label] of Object.entries(data.languages)) {
      const o = document.createElement("option");
      o.value = code;
      o.textContent = label;
      if (code === data.default_language) o.selected = true;
      langSel.appendChild(o);
    }

    // Batch size и VAD из .env
    if (data.default_batch_size) {
      $("#opt-batch").value = data.default_batch_size;
    }
    if (typeof data.default_vad === "boolean") {
      $("#opt-vad").checked = data.default_vad;
    }
  } catch (e) {
    console.error(e);
  }
}

// ============================================================
// Dropzone
// ============================================================

function setupDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");

  dz.addEventListener("click", () => input.click());

  input.addEventListener("change", () => {
    if (input.files.length) setFile(input.files[0]);
  });

  ["dragenter", "dragover"].forEach(ev =>
    dz.addEventListener(ev, e => {
      e.preventDefault();
      dz.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach(ev =>
    dz.addEventListener(ev, e => {
      e.preventDefault();
      dz.classList.remove("dragover");
    })
  );

  dz.addEventListener("drop", e => {
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  $("#chip-remove").addEventListener("click", e => {
    e.stopPropagation();
    clearFile();
  });
}

function setFile(file) {
  state.file = file;
  $("#chip-name").textContent = file.name;
  $("#chip-size").textContent = formatSize(file.size);
  $("#file-chip").hidden = false;
  $("#dropzone").classList.add("has-file");
  $("#dropzone-title")?.remove;
  $("#dropzone .dropzone-title").textContent = "Файл готов";
  $("#dropzone .dropzone-hint").textContent = "Нажмите, чтобы выбрать другой";
  $("#start-btn").disabled = false;
}

function clearFile() {
  state.file = null;
  $("#file-input").value = "";
  $("#file-chip").hidden = true;
  $("#dropzone").classList.remove("has-file");
  $("#dropzone .dropzone-title").textContent = "Перетащите файл сюда";
  $("#dropzone .dropzone-hint").textContent = "или нажмите для выбора · MP4 · MKV · M4A · MP3 · WAV";
  $("#start-btn").disabled = true;
}

function formatSize(bytes) {
  const units = ["Б", "КБ", "МБ", "ГБ"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(1)} ${units[i]}`;
}

// ============================================================
// Options
// ============================================================

function setupOptions() {
  // ничего особенного — всё читается при старте
}

// ============================================================
// Actions
// ============================================================

function setupActions() {
  $("#start-btn").addEventListener("click", start);
  $("#cancel-btn").addEventListener("click", cancel);
}

async function start() {
  if (!state.file || state.running) return;

  resetUI();
  state.running = true;
  $("#start-btn").disabled = true;
  $("#cancel-btn").disabled = false;

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("model", $("#opt-model").value);
  fd.append("language", $("#opt-language").value);
  fd.append("batch_size", $("#opt-batch").value);
  fd.append("vad_filter", $("#opt-vad").checked);
  fd.append("initial_prompt", $("#opt-prompt").value);
  fd.append("hotwords", $("#opt-hotwords").value);

  toast("info", "📤", "Загрузка файла на сервер…");

  try {
    const r = await fetch("/api/jobs", { method: "POST", body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    state.jobId = data.job_id;
    toast("success", "✅", "Задача создана");
    subscribe(state.jobId);
  } catch (e) {
    toast("error", "❌", `Ошибка: ${e.message}`);
    state.running = false;
    $("#start-btn").disabled = false;
    $("#cancel-btn").disabled = true;
  }

  if (!state.consoleOpen) {
      state.consoleOpen = true;
      document.body.classList.add("console-open");
      $("#console-chevron").textContent = "▼";
    }
}

async function cancel() {
  if (!state.jobId) return;
  try {
    await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
    toast("warn", "⏹", "Запрошена отмена");
  } catch (e) {
    console.error(e);
  }
}

// ============================================================
// SSE подписка
// ============================================================

function subscribe(jobId) {
  if (state.eventSource) state.eventSource.close();

  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  state.eventSource = es;

  es.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      handleEvent(data);
    } catch (e) {
      console.error("Bad SSE event:", e, ev.data);
    }
  };

  es.onerror = () => {
    // EventSource сам переподключается, кроме случая когда поток закрыт
    if (!state.running) es.close();
  };
}

function handleEvent(ev) {
  switch (ev.type) {
    case "snapshot":
      applySnapshot(ev.data);
      break;
    case "log":
      appendLog(ev.entry);
      break;
    case "progress":
      updateProgress(ev.value, ev.message, ev.stage);
      break;
    case "status":
      updateStatus(ev.status, ev.message);
      break;
    case "segment":
      appendSegment(ev.segment);
      break;
    case "done":
      onDone(ev.text, ev.metadata);
      break;
    case "error":
      onError(ev.message);
      break;
    case "cancelled":
      onCancelled();
      break;
  }
}

function applySnapshot(snap) {
  // Логи
  for (const log of snap.logs || []) appendLog(log, true);

  // Сегменты
  if (snap.segments?.length) {
    state.segments = [];
    $("#result-content").innerHTML = '<div class="segments" id="segments"></div>';
    for (const seg of snap.segments) appendSegment(seg, true);
  }

  // Прогресс и статус
  if (snap.progress !== undefined) {
    updateProgress(snap.progress, snap.message, snap.stage);
  }
  updateStatus(snap.status, snap.message);

  if (snap.status === "done") {
    onDone(snap.text, snap.metadata);
  }
}

// ============================================================
// Прогресс
// ============================================================

function updateProgress(value, message, stage) {
  const fill = $("#progress-fill");
  fill.style.width = value + "%";
  fill.classList.toggle("active", value > 0 && value < 100);

  $("#progress-pct").textContent = value + "%";
  if (stage) {
    $("#progress-stage").textContent = stageLabel(stage);
  }
  if (message) {
    $("#progress-message").textContent = message;
  }
}

function stageLabel(stage) {
  const map = {
    pending: "Ожидание",
    downloading: "Загрузка",
    loading: "Подготовка",
    transcribing: "Транскрибация",
    done: "Готово",
    error: "Ошибка",
    cancelled: "Отменено",
  };
  return map[stage] || stage;
}

function updateStatus(status, message) {
  // React на уровне UI:
  if (status === "downloading") {
    $("#progress-message").textContent = message || "Загрузка…";
  }
  if (status === "done" || status === "error" || status === "cancelled") {
    state.running = false;
    $("#start-btn").disabled = !state.file;
    $("#cancel-btn").disabled = true;
  }
}

// ============================================================
// Логи
// ============================================================

function appendLog(entry, silent = false) {
  const body = $("#console-body");
  const empty = body.querySelector(".console-empty");
  if (empty) empty.remove();

  const line = document.createElement("div");
  line.className = "log-line";

  const time = document.createElement("span");
  time.className = "log-time";
  const d = new Date(entry.time * 1000);
  time.textContent = d.toLocaleTimeString("ru-RU", { hour12: false });

  const level = document.createElement("span");
  level.className = "log-level " + (entry.level || "info");
  level.textContent = (entry.level || "info").toUpperCase();

  const msg = document.createElement("span");
  msg.className = "log-message";
  msg.textContent = entry.message;

  line.append(time, level, msg);

  const wasScrolled = isScrolledToBottom(body);
  body.appendChild(line);

  // Автоскролл только если пользователь и так был внизу
  if (wasScrolled || silent === false) {
    body.scrollTop = body.scrollHeight;
  }

  state.logCount = body.querySelectorAll(".log-line").length;
  $("#console-counter").textContent = state.logCount;
}

function isScrolledToBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 30;
}

// ============================================================
// Сегменты (с "гуттером" слева)
// ============================================================

function appendSegment(seg, silent = false) {
  let wrap = $("#segments");
  if (!wrap) {
    $("#result-content").innerHTML = '<div class="segments" id="segments"></div>';
    wrap = $("#segments");
  }

  const empty = $("#result-content .empty-state");
  if (empty) empty.remove();

  const line = document.createElement("div");
  line.className = "segment" + (silent ? "" : " new-segment");

  const gutter = document.createElement("span");
  gutter.className = "gutter";
  gutter.textContent = formatTs(seg.start);

  const text = document.createElement("span");
  text.className = "text";
  text.textContent = seg.text;

  line.append(gutter, text);
  wrap.appendChild(line);

  // Автоскролл вниз
  const rc = $("#result-content");
  rc.scrollTop = rc.scrollHeight;

  state.segments.push(seg);

  // Активируем кнопки экспорта
  $("#act-copy").disabled = false;
  $("#act-txt").disabled = false;
  $("#act-srt").disabled = false;
  $("#act-json").disabled = false;
}

function formatTs(seconds) {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

// ============================================================
// Завершение / ошибки
// ============================================================

function onDone(text, metadata) {
  state.text = text || "";
  state.metadata = metadata || {};
  state.running = false;

  $("#start-btn").disabled = !state.file;
  $("#cancel-btn").disabled = true;
  updateProgress(100, "Готово", "done");

  const dur = state.metadata.duration || 0;
  const proc = state.metadata.processing_time || 0;
  toast("success", "🎉", `Готово! ${state.segments.length} сегментов за ${proc.toFixed(1)} с`);

  $("#progress-message").textContent =
    `Язык: ${state.metadata.language || "?"} · Длительность: ${formatTs(dur)} · Скорость: ×${(dur / Math.max(proc, 0.01)).toFixed(1)}`;

  state.eventSource?.close();
}

function onError(message) {
  state.running = false;
  $("#start-btn").disabled = !state.file;
  $("#cancel-btn").disabled = true;
  $("#progress-fill").classList.remove("active");
  toast("error", "❌", message.slice(0, 120));
  state.eventSource?.close();
}

function onCancelled() {
  state.running = false;
  $("#start-btn").disabled = !state.file;
  $("#cancel-btn").disabled = true;
  $("#progress-fill").classList.remove("active");
  toast("warn", "⏹", "Транскрибация отменена");
  state.eventSource?.close();
}

// ============================================================
// Сброс UI перед новым запуском
// ============================================================

function resetUI() {
  state.segments = [];
  state.text = "";
  state.metadata = {};
  state.jobId = null;

  $("#result-content").innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">✨</div>
      <div class="empty-text">Ожидание начала…</div>
      <div class="empty-hint">Транскрибация запускается</div>
    </div>`;

  updateProgress(0, "", "pending");
  $("#progress-message").textContent = "";

  $("#act-copy").disabled = true;
  $("#act-txt").disabled = true;
  $("#act-srt").disabled = true;
  $("#act-json").disabled = true;

  if (state.eventSource) state.eventSource.close();
}

// ============================================================
// Экспорт
// ============================================================

function setupResult() {
  $("#act-copy").addEventListener("click", copyText);
  $("#act-txt").addEventListener("click", () => downloadTxt());
  $("#act-srt").addEventListener("click", () => downloadSrt());
  $("#act-json").addEventListener("click", () => downloadJson());
}

async function copyText() {
  if (!state.segments.length) return;
  const text = state.segments.map(s => s.text).join("\n");
  try {
    await navigator.clipboard.writeText(text);
    toast("success", "📋", "Текст скопирован (без таймкодов)");
  } catch {
    // fallback
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    toast("success", "📋", "Текст скопирован");
  }
}

function downloadTxt() {
  const text = state.segments.map(s => s.text).join("\n\n");
  downloadBlob(text, baseName() + ".txt", "text/plain");
}

function downloadSrt() {
  const srt = state.segments.map((s, i) => {
    const idx = i + 1;
    const st = fmtSrt(s.start);
    const en = fmtSrt(s.end);
    return `${idx}\n${st} --> ${en}\n${s.text}\n`;
  }).join("\n");
  downloadBlob(srt, baseName() + ".srt", "application/x-subrip");
}

function downloadJson() {
  const data = {
    text: state.text,
    segments: state.segments,
    metadata: state.metadata,
  };
  downloadBlob(JSON.stringify(data, null, 2), baseName() + ".json", "application/json");
}

function fmtSrt(sec) {
  const s = Math.floor(sec);
  const ms = Math.floor((sec - s) * 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return `${pad(h, 2)}:${pad(m, 2)}:${pad(ss, 2)},${pad(ms, 3)}`;
}

function pad(n, len) { return String(n).padStart(len, "0"); }

function baseName() {
  if (!state.file) return "transcript";
  return state.file.name.replace(/\.[^.]+$/, "");
}

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime + ";charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("success", "💾", `Сохранено: ${filename}`);
}

// ============================================================
// Консоль
// ============================================================

function setupConsole() {
  const header = $("#console-toggle");
  const chevron = $("#console-chevron");
  const body = $("#console-body");

  // Синхронизируем начальное состояние
  document.body.classList.toggle("console-open", state.consoleOpen);
  chevron.textContent = state.consoleOpen ? "▼" : "▲";

  header.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    state.consoleOpen = !state.consoleOpen;
    document.body.classList.toggle("console-open", state.consoleOpen);
    chevron.textContent = state.consoleOpen ? "▼" : "▲";
  });

  $("#console-clear").addEventListener("click", (e) => {
    e.stopPropagation();
    body.innerHTML = '<div class="console-empty">Логи появятся здесь…</div>';
    state.logCount = 0;
    $("#console-counter").textContent = "0";
  });

  $("#console-copy").addEventListener("click", async (e) => {
    e.stopPropagation();
    const lines = [...body.querySelectorAll(".log-line")].map(l => {
      const t = l.querySelector(".log-time").textContent;
      const lv = l.querySelector(".log-level").textContent;
      const msg = l.querySelector(".log-message").textContent;
      return `[${t}] ${lv.padEnd(8)} ${msg}`;
    }).join("\n");
    try {
      await navigator.clipboard.writeText(lines);
      toast("success", "📋", "Логи скопированы");
    } catch {}
  });
}

// ============================================================
// Тосты
// ============================================================

function toast(kind, icon, text, duration = 3500) {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.innerHTML = `<span class="toast-icon">${icon}</span><span class="toast-text"></span>`;
  el.querySelector(".toast-text").textContent = text;

  $("#toasts").appendChild(el);

  setTimeout(() => {
    el.classList.add("hide");
    setTimeout(() => el.remove(), 250);
  }, duration);
}
