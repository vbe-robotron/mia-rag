const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  settings: null,
  books: [],
  busy: false,
  abort: null,
  k: 10,
};

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function inlineMd(text) {
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  text = text.replace(/\[(\d+)\]/g, '<span class="cite" data-cite="$1">[$1]</span>');
  return text;
}

function mdToHtml(src) {
  const lines = escapeHtml(src).split("\n");
  const out = [];
  let i = 0;
  let para = [];
  let list = null;
  let quote = [];

  const flushPara = () => {
    if (para.length) {
      out.push("<p>" + inlineMd(para.join("<br>")) + "</p>");
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      out.push(`<${list.tag}>` + list.items.map((x) => "<li>" + inlineMd(x) + "</li>").join("") + `</${list.tag}>`);
      list = null;
    }
  };
  const flushQuote = () => {
    if (quote.length) {
      out.push("<blockquote><p>" + inlineMd(quote.join("<br>")) + "</p></blockquote>");
      quote = [];
    }
  };

  while (i < lines.length) {
    const line = lines[i];
    if (line.trim().startsWith("```")) {
      flushPara(); flushList(); flushQuote();
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        buf.push(lines[i]);
        i++;
      }
      i++;
      out.push("<pre><code>" + buf.join("\n") + "</code></pre>");
      continue;
    }
    if (!line.trim()) {
      flushPara(); flushList(); flushQuote();
      i++;
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      flushPara(); flushList(); flushQuote();
      const n = h[1].length;
      out.push(`<h${n}>` + inlineMd(h[2]) + `</h${n}>`);
      i++;
      continue;
    }
    const q = line.match(/^&gt;\s?(.*)$/);
    if (q) {
      flushPara(); flushList();
      quote.push(q[1]);
      i++;
      continue;
    }
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara(); flushQuote();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) {
        flushList();
        list = { tag, items: [] };
      }
      list.items.push((ul || ol)[1]);
      i++;
      continue;
    }
    flushList(); flushQuote();
    para.push(line);
    i++;
  }
  flushPara(); flushList(); flushQuote();
  return out.join("\n");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = data.error || data.detail || msg;
    } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

async function readSse(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      const dataLines = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      let data = null;
      try {
        data = JSON.parse(dataLines.join("\n"));
      } catch (_) {
        data = dataLines.join("\n");
      }
      onEvent(event, data);
    }
  }
}

function setPill(el, text, kind) {
  el.textContent = text;
  el.className = "pill" + (kind ? " " + kind : "");
}

function refreshSettingsView(s) {
  state.settings = s;
  const idx = s.index;
  const ready = idx.chunks.exists && idx.embeddings.exists && idx.fts.exists;
  setPill($("#pill-index"), "index: " + (ready ? "ready" : "missing"), ready ? "ok" : "warn");
  setPill($("#pill-key"), "api key: " + (s.api_key_set ? "set" : "missing"), s.api_key_set ? "ok" : "warn");
  $("#k").value = s.top_k || 10;
  $("#k-out").textContent = $("#k").value;
  state.k = Number($("#k").value);

  const setup = $("#setup");
  const missing = [];
  if (!s.api_key_set) missing.push("OpenCode Go の API キーが未設定です。");
  if (!ready) missing.push("索引が未構築です（Corpus folder を設定して Build index）。");
  if (missing.length) {
    $("#setup-text").textContent = missing.join(" ");
    setup.classList.remove("hidden");
  } else {
    setup.classList.add("hidden");
  }
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    setPill($("#pill-ollama"), "ollama: " + (h.ollama.ok ? "ready" : "check"),
      h.ollama.ok ? "ok" : "warn");
    setPill($("#pill-key"), "api key: " + (h.api_key ? "set" : "missing"),
      h.api_key ? "ok" : "warn");
  } catch (_) {}
}

async function loadSettings() {
  const s = await api("/api/settings");
  refreshSettingsView(s);
  return s;
}

async function loadBooks() {
  const data = await api("/api/books");
  state.books = data.titles || [];
  const list = $("#book-list");
  list.innerHTML = "";
  state.books.forEach((title) => {
    const label = document.createElement("label");
    label.className = "book";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = title;
    const span = document.createElement("span");
    span.textContent = title;
    label.append(cb, span);
    list.append(label);
  });
}

function selectedTitles() {
  const boxes = $$("#book-list input:checked");
  return boxes.length ? boxes.map((b) => b.value) : null;
}

function ensureEmptyHidden() {
  $("#empty").classList.add("hidden");
}

function scrollIfNearBottom() {
  const chat = $("#chat");
  if (chat.scrollHeight - chat.scrollTop - chat.clientHeight < 220) {
    chat.scrollTop = chat.scrollHeight;
  }
}

function addUserMessage(text) {
  ensureEmptyHidden();
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(bubble);
  $("#chat").append(wrap);
  scrollIfNearBottom();
}

function addAssistantMessage() {
  ensureEmptyHidden();
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `
    <div class="role">mia-rag</div>
    <div class="status-line"><span class="dot"></span><span class="st">準備中…</span></div>
    <div class="answer" hidden></div>
    <details class="sources" hidden><summary>Sources</summary><div class="source-list"></div></details>
    <div class="msg-actions" hidden>
      <button class="ghost copy">Copy</button>
    </div>`;
  $("#chat").append(wrap);
  scrollIfNearBottom();
  return wrap;
}

function setStatus(wrap, text) {
  const line = wrap.querySelector(".status-line");
  line.classList.remove("hidden");
  line.querySelector(".st").textContent = text;
}

function hideStatus(wrap) {
  wrap.querySelector(".status-line").classList.add("hidden");
}

function renderSources(wrap, sources) {
  const box = wrap.querySelector("details.sources");
  const list = wrap.querySelector(".source-list");
  list.innerHTML = "";
  sources.forEach((s) => {
    const row = document.createElement("div");
    row.className = "source-item";
    row.id = `src-${s.n}`;
    row.innerHTML = `<span class="source-num">[${s.n}]</span>
      <span><span class="source-title">${escapeHtml(s.title)}</span><br>
      <a href="${escapeHtml(s.source)}" target="_blank" rel="noopener">${escapeHtml(s.source)}</a></span>`;
    list.append(row);
  });
  const summary = box.querySelector("summary");
  summary.textContent = `Sources (${sources.length})`;
  box.hidden = false;
  box.open = false;
}

async function ask() {
  if (state.busy) return;
  const q = $("#question").value.trim();
  if (!q) return;
  state.busy = true;
  $("#btn-ask").disabled = true;
  $("#btn-stop").classList.remove("hidden");
  state.abort = new AbortController();

  addUserMessage(q);
  const wrap = addAssistantMessage();
  const answerEl = wrap.querySelector(".answer");
  let raw = "";
  let renderTimer = null;
  let sources = [];

  const scheduleRender = () => {
    if (renderTimer) return;
    renderTimer = setTimeout(() => {
      renderTimer = null;
      answerEl.hidden = false;
      answerEl.innerHTML = mdToHtml(raw);
      scrollIfNearBottom();
    }, 80);
  };

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, titles: selectedTitles(), k: state.k }),
      signal: state.abort.signal,
    });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).error || msg; } catch (_) {}
      throw new Error(msg);
    }
    await readSse(res, (event, data) => {
      if (event === "status") setStatus(wrap, data);
      else if (event === "queries") {
        if (data && data.length) setStatus(wrap, "検索: " + data.join(" · "));
      } else if (event === "sources") {
        sources = data || [];
        renderSources(wrap, sources);
      } else if (event === "delta") {
        raw += data.text || "";
        scheduleRender();
      } else if (event === "error") {
        throw new Error(data.message || "error");
      } else if (event === "done") {
        hideStatus(wrap);
      }
    });
    answerEl.hidden = false;
    answerEl.innerHTML = mdToHtml(raw);
    wrap.querySelector(".msg-actions").hidden = false;
    if (sources.length) wrap.querySelector("details.sources").open = true;
    wrap.querySelector(".copy").onclick = () => {
      navigator.clipboard.writeText(raw);
      wrap.querySelector(".copy").textContent = "Copied";
      setTimeout(() => (wrap.querySelector(".copy").textContent = "Copy"), 1200);
    };
    wrap.querySelector(".answer").addEventListener("click", (e) => {
      const cite = e.target.closest(".cite");
      if (!cite) return;
      const row = wrap.querySelector(`#src-${cite.dataset.cite}`);
      if (row) {
        wrap.querySelector("details.sources").open = true;
        row.scrollIntoView({ behavior: "smooth", block: "center" });
        row.style.background = "#2b2230";
        setTimeout(() => (row.style.background = ""), 1200);
      }
    });
  } catch (err) {
    if (err.name === "AbortError") {
      setStatus(wrap, "停止しました");
    } else {
      setStatus(wrap, "エラー: " + err.message);
    }
  } finally {
    state.busy = false;
    state.abort = null;
    $("#btn-ask").disabled = false;
    $("#btn-stop").classList.add("hidden");
    scrollIfNearBottom();
  }
}

function openModal(id) { $(id).classList.remove("hidden"); }
function closeModal(el) { el.classList.add("hidden"); }

async function openSettings() {
  const s = state.settings || (await loadSettings());
  $("#s-corpus").value = s.corpus_dir || "";
  $("#s-data").value = s.data_dir || "";
  $("#s-base").value = s.api_base || "";
  $("#s-model").value = s.model || "";
  $("#s-ollama").value = s.ollama_url || "";
  $("#s-embed").value = s.embed_model || "";
  $("#s-k").value = s.top_k || 10;
  $("#s-key").value = "";
  $("#s-key").placeholder = s.api_key_set ? "保存済み（変更する場合のみ入力）" : "sk-…";
  $("#s-clear-key").checked = false;
  $("#settings-status").textContent = "";
  openModal("#modal-settings");
}

async function saveSettings() {
  const body = {
    corpus_dir: $("#s-corpus").value,
    data_dir: $("#s-data").value,
    api_base: $("#s-base").value,
    model: $("#s-model").value,
    ollama_url: $("#s-ollama").value,
    embed_model: $("#s-embed").value,
    top_k: Number($("#s-k").value) || 10,
    clear_api_key: $("#s-clear-key").checked,
  };
  const key = $("#s-key").value.trim();
  if (key) body.api_key = key;
  try {
    const s = await api("/api/settings", { method: "POST", body: JSON.stringify(body) });
    refreshSettingsView(s);
    refreshHealth();
    closeModal($("#modal-settings"));
  } catch (e) {
    $("#settings-status").textContent = "保存に失敗: " + e.message;
  }
}

function openBuild() {
  const s = state.settings || {};
  $("#b-corpus").value = s.corpus_dir || "";
  $("#b-data").value = s.data_dir || "";
  $("#build-log").textContent = "";
  $("#btn-start-build").disabled = false;
  openModal("#modal-build");
}

async function startBuild() {
  $("#btn-start-build").disabled = true;
  const log = $("#build-log");
  log.textContent = "";
  const append = (t) => { log.textContent += t + "\n"; log.scrollTop = log.scrollHeight; };
  try {
    const res = await fetch("/api/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        corpus_dir: $("#b-corpus").value,
        data_dir: $("#b-data").value,
        skip_embed: $("#b-skip").checked,
      }),
    });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).error || msg; } catch (_) {}
      throw new Error(msg);
    }
    await readSse(res, (event, data) => {
      if (event === "progress") append(data);
      else if (event === "error") append("ERROR: " + data);
      else if (event === "done") append("完了しました");
    });
    await loadSettings();
    await loadBooks();
    refreshHealth();
  } catch (e) {
    append("ERROR: " + e.message);
  } finally {
    $("#btn-start-build").disabled = false;
  }
}

function bind() {
  $("#btn-ask").onclick = ask;
  $("#btn-stop").onclick = () => state.abort && state.abort.abort();
  $("#btn-new").onclick = () => {
    $("#chat").innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.id = "empty";
    empty.innerHTML = `<h3>20冊のマルクス経済学古典に質問する</h3>
      <p>例: 「資本の原始的蓄積における暴力の役割は？」</p>
      <p>例: 「帝国主義と金融資本の関係を、レーニンとヒルファーディングに即して」</p>`;
    $("#chat").append(empty);
  };
  $("#btn-settings").onclick = openSettings;
  $("#setup-settings").onclick = openSettings;
  $("#btn-build").onclick = openBuild;
  $("#setup-build").onclick = openBuild;
  $("#btn-save-settings").onclick = saveSettings;
  $("#btn-start-build").onclick = startBuild;
  $$("[data-close]").forEach((b) => {
    b.onclick = () => closeModal(b.closest(".modal"));
  });
  $$(".modal").forEach((m) => {
    m.addEventListener("click", (e) => { if (e.target === m) closeModal(m); });
  });
  $("#question").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      ask();
    }
  });
  $("#k").addEventListener("input", (e) => {
    state.k = Number(e.target.value);
    $("#k-out").textContent = e.target.value;
  });
  $("#books-all").onclick = () => $$("#book-list input").forEach((b) => (b.checked = true));
  $("#books-none").onclick = () => $$("#book-list input").forEach((b) => (b.checked = false));
}

(async function init() {
  bind();
  try {
    await loadSettings();
    await loadBooks();
    await refreshHealth();
  } catch (e) {
    console.error(e);
  }
})();
