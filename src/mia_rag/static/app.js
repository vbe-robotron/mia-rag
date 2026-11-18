const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const HISTORY_KEY = "mia-rag-history-v1";

const state = {
  settings: null,
  docs: [],
  authors: [],
  busy: false,
  abort: null,
  k: 10,
  deep: true,
  rebuildNeeded: false,
};

/* ---------- markdown ---------- */
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
  let i = 0, para = [], list = null, quote = [];
  const flushPara = () => {
    if (para.length) { out.push("<p>" + inlineMd(para.join("<br>")) + "</p>"); para = []; }
  };
  const flushList = () => {
    if (list) {
      out.push(`<${list.tag}>` + list.items.map((x) => "<li>" + inlineMd(x) + "</li>").join("") + `</${list.tag}>`);
      list = null;
    }
  };
  const flushQuote = () => {
    if (quote.length) { out.push("<blockquote><p>" + inlineMd(quote.join("<br>")) + "</p></blockquote>"); quote = []; }
  };
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim().startsWith("```")) {
      flushPara(); flushList(); flushQuote();
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) { buf.push(lines[i]); i++; }
      i++;
      out.push("<pre><code>" + buf.join("\n") + "</code></pre>");
      continue;
    }
    if (!line.trim()) { flushPara(); flushList(); flushQuote(); i++; continue; }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      flushPara(); flushList(); flushQuote();
      const n = h[1].length;
      out.push(`<h${n}>` + inlineMd(h[2]) + `</h${n}>`);
      i++; continue;
    }
    const q = line.match(/^&gt;\s?(.*)$/);
    if (q) { flushPara(); flushList(); quote.push(q[1]); i++; continue; }
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara(); flushQuote();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((ul || ol)[1]);
      i++; continue;
    }
    flushList(); flushQuote();
    para.push(line); i++;
  }
  flushPara(); flushList(); flushQuote();
  return out.join("\n");
}

/* ---------- http ---------- */
async function api(path, options = {}) {
  const res = await fetch(path, { ...options });
  if (!res.ok) {
    let msg = res.statusText;
    try { const d = await res.json(); msg = d.error || d.detail || msg; } catch (_) {}
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
      let data;
      try { data = JSON.parse(dataLines.join("\n")); } catch (_) { data = dataLines.join("\n"); }
      onEvent(event, data);
    }
  }
}

/* ---------- status ---------- */
function setPill(el, text, kind) {
  el.textContent = text;
  el.className = "badge rounded-pill " + (kind === "ok" ? "text-bg-success"
    : kind === "warn" ? "text-bg-warning" : "text-bg-secondary");
}

function renderSettings(s) {
  state.settings = s;
  const idx = s.index;
  const ready = idx.chunks.exists && idx.embeddings.exists && idx.fts.exists;
  setPill($("#pill-index"), "index: " + (ready ? "ready" : "missing"), ready ? "ok" : "warn");
  setPill($("#pill-key"), "api key: " + (s.api_key_set ? "set" : "missing"), s.api_key_set ? "ok" : "warn");
  $("#k").value = s.top_k || 10;
  $("#k-out").textContent = $("#k").value;
  state.k = Number($("#k").value);

  const missing = [];
  if (!s.api_key_set) missing.push("OpenCode Go の API キーが未設定です。");
  if (!ready) missing.push("索引が未構築です（Library / Rebuild から構築してください）。");
  if (state.rebuildNeeded) missing.push("文献が変更されています。Rebuild を実行してください。");
  const setup = $("#setup");
  if (missing.length) {
    $("#setup-text").textContent = missing.join(" ");
    setup.classList.remove("d-none");
    setup.classList.add("d-flex");
  } else {
    setup.classList.add("d-none");
    setup.classList.remove("d-flex");
  }
  updateHint();
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    setPill($("#pill-ollama"), "ollama: " + (h.ollama.ok ? "ready" : "check"), h.ollama.ok ? "ok" : "warn");
    setPill($("#pill-key"), "api key: " + (h.api_key ? "set" : "missing"), h.api_key ? "ok" : "warn");
  } catch (_) {}
}

/* ---------- books ---------- */
function checkedBooks() {
  return new Set($$("#book-list .book-check:checked").map((c) => c.value));
}

function renderBooks() {
  const checked = checkedBooks();
  const q = ($("#book-search").value || "").toLowerCase();
  const author = $("#author-filter").value;
  const list = $("#book-list");
  list.innerHTML = "";
  const groups = {};
  state.docs.forEach((d) => {
    if (author && d.author !== author) return;
    if (q && !(d.title + " " + d.author + " " + d.book).toLowerCase().includes(q)) return;
    const key = d.author || "その他";
    if (!groups[key]) groups[key] = [];
    groups[key].push(d);
  });
  const names = Object.keys(groups).sort((a, b) => a.localeCompare(b));
  names.forEach((name, gi) => {
    const wrap = document.createElement("div");
    wrap.className = "author-group";
    const btn = document.createElement("button");
    btn.className = "author-toggle";
    btn.type = "button";
    btn.dataset.bsToggle = "collapse";
    btn.dataset.bsTarget = "#author-" + gi;
    btn.setAttribute("aria-expanded", "true");
    btn.innerHTML = `<span>${escapeHtml(name)}</span><span class="badge text-bg-secondary ms-auto">${groups[name].length}</span>`;
    const body = document.createElement("div");
    body.className = "collapse show";
    body.id = "author-" + gi;
    groups[name].forEach((d) => {
      const label = document.createElement("label");
      label.className = "book-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "form-check-input book-check";
      cb.value = d.book;
      cb.checked = checked.has(d.book);
      cb.addEventListener("change", updateHint);
      const span = document.createElement("span");
      span.textContent = d.title;
      span.title = `${d.title}${d.chunks ? " · " + d.chunks + " chunks" : ""}`;
      label.append(cb, span);
      body.append(label);
    });
    wrap.append(btn, body);
    list.append(wrap);
  });
  if (!names.length) {
    list.innerHTML = '<div class="text-secondary small p-2">該当する文献がありません。</div>';
  }
  updateHint();
}

function renderAuthors() {
  const sel = $("#author-filter");
  const cur = sel.value;
  const authors = [...new Set(state.docs.map((d) => d.author).filter(Boolean))].sort();
  state.authors = authors;
  sel.innerHTML = '<option value="">すべての著者</option>' +
    authors.map((a) => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`).join("");
  sel.value = authors.includes(cur) ? cur : "";
}

async function loadBooks() {
  const data = await api("/api/books");
  state.docs = data.docs || [];
  renderAuthors();
  renderBooks();
}

function selectedFiles() {
  return [...checkedBooks()];
}

function updateHint() {
  const n = checkedBooks().size;
  let text = n ? `${n}冊を対象` : "全文献を対象";
  if (state.deep) text += n ? " · 全文モード" : " · 全文モードは文献選択が必要";
  $("#composer-hint").textContent = text;
}

/* ---------- messages ---------- */
function saveHistory() {
  const items = [];
  $$("#chat .msg").forEach((wrap) => {
    if (wrap.classList.contains("user")) {
      items.push({ role: "user", text: $(".bubble", wrap).textContent });
    } else {
      const ans = $(".answer", wrap);
      if (!ans || ans.hidden) return;
      const sources = $$(".source-item", wrap).map((row) => ({
        n: Number($(".source-num", row).textContent.replace(/\D/g, "")),
        title: $(".source-title", row).textContent,
        source: $("a", row).getAttribute("href"),
      }));
      items.push({ role: "assistant", text: ans.dataset.raw || "", sources });
    }
  });
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(-24)));
  } catch (_) {}
}

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
}

function newChat() {
  $("#chat").innerHTML = `
    <div class="empty text-center text-secondary" id="empty">
      <h3 class="h4 text-light mt-5">20冊のマルクス経済学古典に質問する</h3>
      <p class="mb-1">例: 「資本の原始的蓄積における暴力の役割は？」</p>
      <p>例: 「帝国主義と金融資本の関係を、レーニンとヒルファーディングに即して」</p>
    </div>`;
  clearHistory();
}

function hideEmpty() {
  const e = $("#empty");
  if (e) e.remove();
}

function scrollIfNearBottom() {
  const chat = $("#chat");
  if (chat.scrollHeight - chat.scrollTop - chat.clientHeight < 240) {
    chat.scrollTop = chat.scrollHeight;
  }
}

function addUserMessage(text) {
  hideEmpty();
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.append(bubble);
  $("#chat").append(wrap);
  scrollIfNearBottom();
}

function makeSources(sources) {
  if (!sources || !sources.length) return null;
  const box = document.createElement("details");
  box.className = "sources";
  box.open = true;
  box.innerHTML = `<summary>Sources (${sources.length})</summary>`;
  const list = document.createElement("div");
  sources.forEach((s) => {
    const row = document.createElement("div");
    row.className = "source-item";
    row.id = `src-${s.n}-${Math.random().toString(36).slice(2)}`;
    row.innerHTML = `<span class="source-num">[${s.n}]</span>
      <span><span class="source-title">${escapeHtml(s.title)}</span><br>
      <a href="${escapeHtml(s.source)}" target="_blank" rel="noopener">${escapeHtml(s.source)}</a></span>`;
    list.append(row);
  });
  box.append(list);
  return box;
}

function wireAnswer(wrap, raw) {
  const answer = $(".answer", wrap);
  answer.dataset.raw = raw;
  answer.addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (!cite) return;
    const row = $$(".source-item", wrap)[Number(cite.dataset.cite) - 1];
    if (row) {
      $(".sources", wrap).open = true;
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      row.style.background = "rgba(224,75,75,.18)";
      setTimeout(() => (row.style.background = ""), 1200);
    }
  });
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const copy = document.createElement("button");
  copy.className = "btn btn-sm btn-outline-secondary";
  copy.textContent = "Copy";
  copy.onclick = () => {
    navigator.clipboard.writeText(raw);
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy"), 1200);
  };
  actions.append(copy);
  wrap.append(actions);
}

function addAssistantMessage(text, sources) {
  hideEmpty();
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const role = document.createElement("div");
  role.className = "role";
  role.textContent = "mia-rag";
  const answer = document.createElement("div");
  answer.className = "answer";
  answer.innerHTML = mdToHtml(text);
  wrap.append(role, answer);
  const src = makeSources(sources);
  if (src) wrap.append(src);
  wireAnswer(wrap, text);
  $("#chat").append(wrap);
}

function addStreamingMessage() {
  hideEmpty();
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `
    <div class="role">mia-rag</div>
    <div class="status-line"><span class="spinner-border spinner-border-sm"></span><span class="st">準備中…</span></div>
    <div class="answer d-none"></div>`;
  $("#chat").append(wrap);
  scrollIfNearBottom();
  return wrap;
}

function setStatus(wrap, text) {
  const line = $(".status-line", wrap);
  if (!line) return;
  line.classList.remove("d-none");
  line.querySelector(".st").textContent = text;
}

function hideStatus(wrap) {
  const line = $(".status-line", wrap);
  if (line) line.classList.add("d-none");
}

/* ---------- ask ---------- */
async function ask() {
  if (state.busy) return;
  const q = $("#question").value.trim();
  if (!q) return;
  state.busy = true;
  $("#btn-ask").disabled = true;
  $("#btn-stop").classList.remove("d-none");
  state.abort = new AbortController();

  addUserMessage(q);
  const wrap = addStreamingMessage();
  const answerEl = $(".answer", wrap);
  let raw = "";
  let sources = [];
  let renderTimer = null;
  const scheduleRender = () => {
    if (renderTimer) return;
    renderTimer = setTimeout(() => {
      renderTimer = null;
      answerEl.classList.remove("d-none");
      answerEl.innerHTML = mdToHtml(raw);
      scrollIfNearBottom();
    }, 80);
  };

  const files = selectedFiles();
  const mode = state.deep && files.length ? "full" : "retrieval";
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, files, k: state.k, mode }),
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
      } else if (event === "delta") {
        raw += data.text || "";
        scheduleRender();
      } else if (event === "error") {
        throw new Error(data.message || "error");
      } else if (event === "done") {
        hideStatus(wrap);
      }
    });
    hideStatus(wrap);
    answerEl.classList.remove("d-none");
    answerEl.innerHTML = mdToHtml(raw);
    const src = makeSources(sources);
    if (src) wrap.insertBefore(src, wrap.querySelector(".msg-actions"));
    wireAnswer(wrap, raw);
    saveHistory();
  } catch (err) {
    if (err.name === "AbortError") setStatus(wrap, "停止しました");
    else setStatus(wrap, "エラー: " + err.message);
  } finally {
    state.busy = false;
    state.abort = null;
    $("#btn-ask").disabled = false;
    $("#btn-stop").classList.add("d-none");
    scrollIfNearBottom();
  }
}

/* ---------- settings ---------- */
function fillSettings(s) {
  $("#s-corpus").value = s.corpus_dir || "";
  $("#s-data").value = s.data_dir || "";
  $("#s-base").value = s.api_base || "";
  $("#s-model").value = s.model || "";
  $("#s-ollama").value = s.ollama_url || "";
  $("#s-embed").value = s.embed_model || "";
  $("#s-k").value = s.top_k || 10;
  $("#s-reasoning").checked = !!s.reasoning;
  $("#s-key").value = "";
  $("#s-key").placeholder = s.api_key_set ? "保存済み（変更する場合のみ入力）" : "sk-…";
  $("#s-clear-key").checked = false;
  $("#settings-status").textContent = "";
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
    reasoning: $("#s-reasoning").checked,
    clear_api_key: $("#s-clear-key").checked,
  };
  const key = $("#s-key").value.trim();
  if (key) body.api_key = key;
  try {
    const s = await api("/api/settings", { method: "POST", body: JSON.stringify(body) });
    renderSettings(s);
    refreshHealth();
    bootstrap.Modal.getInstance($("#modal-settings")).hide();
    await loadBooks();
  } catch (e) {
    $("#settings-status").textContent = "保存に失敗: " + e.message;
  }
}

/* ---------- library ---------- */
async function renderLibrary() {
  const data = await api("/api/books");
  state.docs = data.docs || [];
  const list = $("#lib-list");
  list.innerHTML = "";
  $("#lib-count").textContent = state.docs.length + " docs";
  state.docs.forEach((d) => {
    const row = document.createElement("div");
    row.className = "lib-item list-group-item";
    row.innerHTML = `<div class="meta">
        <div class="title">${escapeHtml(d.title)}</div>
        <div class="sub">${escapeHtml(d.author || "—")} · ${Math.round((d.bytes || 0) / 1024)} KB · ${escapeHtml(d.file)}${d.chunks ? " · " + d.chunks + " chunks" : ""}</div>
      </div>`;
    const edit = document.createElement("button");
    edit.className = "btn btn-sm btn-outline-secondary";
    edit.textContent = "Edit";
    edit.onclick = async () => {
      const title = prompt("タイトル", d.title);
      if (title === null) return;
      const author = prompt("著者", d.author || "");
      if (author === null) return;
      await api("/api/corpus/" + encodeURIComponent(d.file), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, author }),
      });
      state.rebuildNeeded = true;
      renderSettings(state.settings);
      renderLibrary();
    };
    const del = document.createElement("button");
    del.className = "btn btn-sm btn-outline-danger";
    del.textContent = "Delete";
    del.onclick = async () => {
      if (!confirm(`削除しますか？\n${d.title}\n\n（再構築時に埋め込みを全作り直します）`)) return;
      await api("/api/corpus/" + encodeURIComponent(d.file), { method: "DELETE" });
      state.rebuildNeeded = true;
      renderSettings(state.settings);
      renderLibrary();
    };
    row.append(edit, del);
    list.append(row);
  });
}

async function uploadFile() {
  const input = $("#lib-file");
  if (!input.files.length) return;
  const fd = new FormData();
  fd.append("file", input.files[0]);
  $("#lib-status").textContent = "アップロード中…";
  try {
    await api("/api/corpus/upload", { method: "POST", body: fd });
    $("#lib-status").textContent = "追加しました。Rebuild を実行してください。";
    input.value = "";
    state.rebuildNeeded = true;
    renderSettings(state.settings);
    renderLibrary();
  } catch (e) {
    $("#lib-status").textContent = "失敗: " + e.message;
  }
}

async function addUrl() {
  const url = $("#lib-url").value.trim();
  if (!url) return;
  $("#lib-status").textContent = "取得中…";
  try {
    await api("/api/corpus/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url,
        author: $("#lib-url-author").value.trim(),
        title: $("#lib-url-title").value.trim(),
        crawl: $("#lib-url-crawl").checked,
      }),
    });
    $("#lib-status").textContent = "追加しました。Rebuild を実行してください。";
    state.rebuildNeeded = true;
    renderSettings(state.settings);
    renderLibrary();
  } catch (e) {
    $("#lib-status").textContent = "失敗: " + e.message;
  }
}

/* ---------- build ---------- */
function fillBuild() {
  const s = state.settings || {};
  $("#b-corpus").value = s.corpus_dir || "";
  $("#b-data").value = s.data_dir || "";
  $("#build-log").textContent = "";
  $("#btn-start-build").disabled = false;
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
    state.rebuildNeeded = false;
    const s = await api("/api/settings");
    renderSettings(s);
    await loadBooks();
    refreshHealth();
  } catch (e) {
    append("ERROR: " + e.message);
  } finally {
    $("#btn-start-build").disabled = false;
  }
}

/* ---------- history ---------- */
function restoreHistory() {
  let items = [];
  try { items = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch (_) {}
  if (!items.length) return;
  hideEmpty();
  items.forEach((m) => {
    if (m.role === "user") addUserMessage(m.text);
    else addAssistantMessage(m.text || "", m.sources || []);
  });
}

/* ---------- bind ---------- */
function bind() {
  $("#btn-ask").onclick = ask;
  $("#btn-stop").onclick = () => state.abort && state.abort.abort();
  $("#btn-new").onclick = newChat;
  $("#btn-save-settings").onclick = saveSettings;
  $("#btn-start-build").onclick = startBuild;
  $("#lib-upload").onclick = uploadFile;
  $("#lib-add-url").onclick = addUrl;
  $("#book-search").addEventListener("input", renderBooks);
  $("#author-filter").addEventListener("change", renderBooks);
  $("#books-all").onclick = () => { $$("#book-list .book-check").forEach((b) => (b.checked = true)); updateHint(); };
  $("#books-none").onclick = () => { $$("#book-list .book-check").forEach((b) => (b.checked = false)); updateHint(); };
  $("#deep").addEventListener("change", (e) => { state.deep = e.target.checked; updateHint(); });
  $("#question").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); ask(); }
  });
  $("#k").addEventListener("input", (e) => {
    state.k = Number(e.target.value);
    $("#k-out").textContent = e.target.value;
  });
  $("#modal-settings").addEventListener("show.bs.modal", () => {
    if (state.settings) fillSettings(state.settings);
  });
  $("#modal-build").addEventListener("show.bs.modal", fillBuild);
  $("#modal-library").addEventListener("show.bs.modal", () => {
    $("#lib-status").textContent = "";
    renderLibrary().catch((e) => ($("#lib-status").textContent = e.message));
  });
}

(async function init() {
  bind();
  state.deep = $("#deep").checked;
  try {
    renderSettings(await api("/api/settings"));
    await loadBooks();
    await refreshHealth();
    restoreHistory();
  } catch (e) {
    console.error(e);
  }
})();
