import argparse
import json
import os
import queue
import re
import threading
import urllib.request
import webbrowser

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .api import resolve_api_key
from .config import load_settings, save_settings
from .corpus import (
    TITLES,
    build_chunks,
    crawl_work,
    default_meta,
    file_hash,
    find_text_files,
    html_to_text,
    load_catalog,
    next_prefix,
    safe_filename,
    save_catalog,
)
from .embed import build_embeddings
from .index import Retriever, ensure_fts, load_chunks
from .rag import RagEngine

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="mia-rag")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_cache = {"key": None, "retriever": None}
_cache_lock = threading.Lock()
_build_lock = threading.Lock()


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def index_state(s):
    def info(path):
        return {
            "path": path,
            "exists": os.path.exists(path),
            "bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        }

    return {
        "chunks": info(s.chunks_path),
        "embeddings": info(s.embeddings_path),
        "fts": info(s.fts_path),
    }


def settings_payload(s):
    return {
        "corpus_dir": s.corpus_dir,
        "data_dir": s.data_dir,
        "api_base": s.api_base,
        "model": s.model,
        "ollama_url": s.ollama_url,
        "embed_model": s.embed_model,
        "top_k": s.top_k,
        "reasoning": s.reasoning,
        "answer_style": s.answer_style,
        "full_text_max_chars": s.full_text_max_chars,
        "api_key_set": bool(s.api_key)
        or bool(os.environ.get("OPENCODE_API_KEY"))
        or os.path.exists(os.path.expanduser("~/.local/share/opencode/auth.json")),
        "index": index_state(s),
    }


def get_retriever(s):
    def stamp(path):
        try:
            st = os.stat(path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    key = (s.chunks_path, stamp(s.chunks_path), stamp(s.embeddings_path), s.fts_path)
    with _cache_lock:
        if _cache["key"] != key or _cache["retriever"] is None:
            _cache["retriever"] = Retriever(
                s.chunks_path, s.embeddings_path, s.fts_path
            )
            _cache["key"] = key
        return _cache["retriever"]


def invalidate_cache():
    with _cache_lock:
        _cache["key"] = None
        _cache["retriever"] = None


def corpus_docs(s, retriever=None):
    catalog = load_catalog(s.data_dir)
    docs = {}
    for path in find_text_files(s.corpus_dir):
        base = os.path.basename(path)
        book = os.path.splitext(base)[0]
        meta = catalog.get("docs", {}).get(book) or default_meta(base)
        docs[book] = {
            "book": book,
            "file": base,
            "title": meta.get("title") or book,
            "author": meta.get("author") or "",
            "bytes": os.path.getsize(path),
        }
    if retriever is not None:
        for d in retriever.docs():
            if d["book"] in docs:
                docs[d["book"]]["chunks"] = d["chunks"]
                if not docs[d["book"]]["author"]:
                    docs[d["book"]]["author"] = d.get("author") or ""
    return sorted(docs.values(), key=lambda d: d["file"])


class AskBody(BaseModel):
    question: str
    titles: list[str] | None = None
    files: list[str] | None = None
    k: int | None = None
    mode: str = "retrieval"


class SettingsBody(BaseModel):
    corpus_dir: str | None = None
    data_dir: str | None = None
    api_base: str | None = None
    model: str | None = None
    ollama_url: str | None = None
    embed_model: str | None = None
    top_k: int | None = None
    reasoning: bool | None = None
    answer_style: str | None = None
    full_text_max_chars: int | None = None
    api_key: str | None = None
    clear_api_key: bool = False


class BuildBody(BaseModel):
    corpus_dir: str | None = None
    data_dir: str | None = None
    skip_embed: bool = False


class UrlBody(BaseModel):
    url: str
    title: str | None = None
    author: str | None = None
    crawl: bool = True


class MetaBody(BaseModel):
    title: str | None = None
    author: str | None = None


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/settings")
def get_settings():
    return settings_payload(load_settings())


@app.post("/api/settings")
def post_settings(body: SettingsBody):
    s = load_settings()
    for field in (
        "corpus_dir",
        "data_dir",
        "api_base",
        "model",
        "ollama_url",
        "embed_model",
    ):
        value = getattr(body, field)
        if value is not None and value.strip():
            value = value.strip()
            if field in ("corpus_dir", "data_dir"):
                value = os.path.abspath(os.path.expanduser(value))
            setattr(s, field, value)
    if body.top_k is not None:
        s.top_k = max(1, min(50, body.top_k))
    if body.reasoning is not None:
        s.reasoning = body.reasoning
    if body.answer_style in ("tutor", "strict"):
        s.answer_style = body.answer_style
    if body.full_text_max_chars is not None:
        s.full_text_max_chars = max(100000, min(3000000, body.full_text_max_chars))
    if body.clear_api_key:
        s.api_key = ""
    elif body.api_key:
        s.api_key = body.api_key.strip()
    save_settings(s)
    invalidate_cache()
    return settings_payload(s)


@app.get("/api/books")
def books():
    s = load_settings()
    retriever = None
    try:
        retriever = get_retriever(s)
    except Exception:
        pass
    docs = corpus_docs(s, retriever)
    return {"titles": [d["title"] for d in docs], "docs": docs}


@app.post("/api/corpus/upload")
async def upload(file: UploadFile = File(...)):
    s = load_settings()
    os.makedirs(s.corpus_dir, exist_ok=True)
    raw = await file.read()
    data = raw.decode("utf-8", "replace")
    if not data.strip():
        return JSONResponse({"error": "empty file"}, status_code=400)
    prefix = next_prefix(s.corpus_dir)
    stem = os.path.splitext(safe_filename(file.filename or "document.txt"))[0]
    name = f"{prefix:02d}_{stem}.txt"
    path = os.path.join(s.corpus_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    return {"ok": True, "file": name}


@app.post("/api/corpus/url")
def add_url(body: UrlBody):
    s = load_settings()
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        return JSONResponse({"error": "invalid URL"}, status_code=400)
    title = body.title or ""
    author = body.author or "Web"
    try:
        if body.crawl:
            found_title, pages = crawl_work(url)
            if not pages:
                return JSONResponse({"error": "no pages found"}, status_code=400)
            text = "\n\n".join(
                f"----- SOURCE: {u} -----\n\n{t}" for u, t in pages
            )
            title = title or found_title
        else:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 mia-rag/0.1"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8", "replace")
                ctype = r.headers.get("Content-Type", "")
            if "html" in ctype or "<html" in raw[:500].lower():
                text = html_to_text(raw)
            else:
                text = raw
            if not title:
                m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
                title = html_to_text(m.group(1)).strip() if m else ""
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not text.strip():
        return JSONResponse({"error": "no text extracted"}, status_code=400)
    title = re.sub(r"\s*[-|–]\s*Marxists Internet Archive.*$", "", title, flags=re.I).strip()
    prefix = next_prefix(s.corpus_dir)
    stem = safe_filename((author + "_" + (title or "document")).replace(" ", "_"))
    stem = os.path.splitext(stem)[0][:80]
    name = f"{prefix:02d}_{stem}.txt"
    os.makedirs(s.corpus_dir, exist_ok=True)
    with open(os.path.join(s.corpus_dir, name), "w", encoding="utf-8") as f:
        f.write(f"----- SOURCE: {url} -----\n\n{text}\n")

    catalog = load_catalog(s.data_dir)
    book = os.path.splitext(name)[0]
    catalog.setdefault("docs", {})[book] = {
        "title": title or book,
        "author": author,
    }
    save_catalog(s.data_dir, catalog)
    return {"ok": True, "file": name, "title": title or book, "author": author}


@app.delete("/api/corpus/{name}")
def delete_doc(name: str):
    s = load_settings()
    name = os.path.basename(name)
    path = os.path.join(s.corpus_dir, name)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    os.remove(path)
    catalog = load_catalog(s.data_dir)
    book = os.path.splitext(name)[0]
    catalog.get("docs", {}).pop(book, None)
    save_catalog(s.data_dir, catalog)
    return {"ok": True, "rebuild_required": True}


@app.patch("/api/corpus/{name}")
def edit_doc(name: str, body: MetaBody):
    s = load_settings()
    name = os.path.basename(name)
    book = os.path.splitext(name)[0]
    catalog = load_catalog(s.data_dir)
    entry = catalog.setdefault("docs", {}).setdefault(book, {})
    if body.title is not None:
        entry["title"] = body.title.strip()
    if body.author is not None:
        entry["author"] = body.author.strip()
    save_catalog(s.data_dir, catalog)
    return {"ok": True, "rebuild_required": True}


@app.get("/api/health")
def health():
    s = load_settings()
    ollama = {"ok": False, "models": []}
    try:
        with urllib.request.urlopen(
            s.ollama_url.rstrip("/") + "/api/tags", timeout=2
        ) as r:
            data = json.loads(r.read())
        names = [m.get("name", "") for m in data.get("models", [])]
        ollama = {
            "ok": any(n.split(":")[0] == s.embed_model.split(":")[0] for n in names),
            "models": names,
        }
    except Exception:
        pass
    try:
        resolve_api_key(s)
        key_ok = True
    except Exception:
        key_ok = False
    return {"ollama": ollama, "api_key": key_ok}


@app.post("/api/ask")
def ask(body: AskBody):
    s = load_settings()
    if not body.question.strip():
        return JSONResponse({"error": "empty question"}, status_code=400)
    try:
        engine = RagEngine(s, retriever=get_retriever(s))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    def stream():
        try:
            for kind, payload in engine.ask_stream(
                body.question.strip(),
                body.k,
                body.titles,
                body.files,
                body.mode,
            ):
                if kind == "delta":
                    yield sse("delta", {"text": payload})
                else:
                    yield sse(kind, payload if payload is not None else {})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/build")
def build(body: BuildBody):
    if not _build_lock.acquire(blocking=False):
        return JSONResponse({"error": "a build is already running"}, status_code=409)
    s = load_settings()
    if body.corpus_dir and body.corpus_dir.strip():
        s.corpus_dir = os.path.abspath(os.path.expanduser(body.corpus_dir.strip()))
    if body.data_dir and body.data_dir.strip():
        s.data_dir = os.path.abspath(os.path.expanduser(body.data_dir.strip()))
    save_settings(s)
    q = queue.Queue()

    def progress(msg):
        q.put(("progress", msg))

    def work():
        try:
            os.makedirs(s.data_dir, exist_ok=True)
            catalog = load_catalog(s.data_dir)
            files = find_text_files(s.corpus_dir)
            books = [os.path.splitext(os.path.basename(p))[0] for p in files]
            old = catalog.get("docs", {})
            old_books = list(old.keys())
            stable = books[: len(old_books)] == old_books and all(
                old.get(b, {}).get("hash") in (None, file_hash(p))
                for b, p in zip(old_books, files[: len(old_books)])
            )
            if not stable and os.path.exists(s.embeddings_path):
                progress("Corpus changed; discarding old embeddings (full rebuild).")
                os.remove(s.embeddings_path)

            meta = {b: old.get(b) or default_meta(os.path.basename(p)) for b, p in zip(books, files)}
            progress("Chunking corpus...")
            info = build_chunks(
                s.corpus_dir,
                s.chunks_path,
                s.chunk_size,
                s.overlap,
                progress,
                meta=meta,
            )
            if not body.skip_embed:
                progress("Embedding chunks (this can take a while on CPU)...")
                build_embeddings(
                    s.chunks_path,
                    s.embeddings_path,
                    s.embed_model,
                    s.ollama_url,
                    progress=progress,
                )
            progress("Building full-text index...")
            ensure_fts(load_chunks(s.chunks_path), s.fts_path)
            new_docs = {}
            for book in books:
                ent = dict(meta.get(book) or {})
                ent.update(info.get(book, {}))
                ent["hash"] = file_hash(
                    os.path.join(s.corpus_dir, ent.get("file", ""))
                )
                new_docs[book] = ent
            catalog["docs"] = new_docs
            save_catalog(s.data_dir, catalog)
            invalidate_cache()
            q.put(("done", index_state(s)))
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            _build_lock.release()

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            kind, payload = q.get()
            yield sse(kind, payload)
            if kind in ("done", "error"):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mia-rag-web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"mia-rag web UI: {url}")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
