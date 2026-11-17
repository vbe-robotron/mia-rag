import argparse
import json
import os
import queue
import threading
import webbrowser

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .api import resolve_api_key
from .config import Settings, load_settings, save_settings
from .corpus import build_chunks
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
    return {
        "chunks": {
            "path": s.chunks_path,
            "exists": os.path.exists(s.chunks_path),
            "bytes": os.path.getsize(s.chunks_path) if os.path.exists(s.chunks_path) else 0,
        },
        "embeddings": {
            "path": s.embeddings_path,
            "exists": os.path.exists(s.embeddings_path),
            "bytes": os.path.getsize(s.embeddings_path)
            if os.path.exists(s.embeddings_path)
            else 0,
        },
        "fts": {
            "path": s.fts_path,
            "exists": os.path.exists(s.fts_path),
            "bytes": os.path.getsize(s.fts_path) if os.path.exists(s.fts_path) else 0,
        },
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


class AskBody(BaseModel):
    question: str
    titles: list[str] | None = None
    k: int | None = None


class SettingsBody(BaseModel):
    corpus_dir: str | None = None
    data_dir: str | None = None
    api_base: str | None = None
    model: str | None = None
    ollama_url: str | None = None
    embed_model: str | None = None
    top_k: int | None = None
    api_key: str | None = None
    clear_api_key: bool = False


class BuildBody(BaseModel):
    corpus_dir: str | None = None
    data_dir: str | None = None
    skip_embed: bool = False


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
            setattr(s, field, value.strip())
    if body.top_k is not None:
        s.top_k = max(1, min(50, body.top_k))
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
    try:
        retriever = get_retriever(s)
        return {"titles": retriever.titles()}
    except Exception:
        from .corpus import TITLES

        return {"titles": list(TITLES.values())}


@app.get("/api/health")
def health():
    s = load_settings()
    ollama = {"ok": False, "models": []}
    try:
        import urllib.request

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
                body.question.strip(), body.k, body.titles
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
        s.corpus_dir = body.corpus_dir.strip()
    if body.data_dir and body.data_dir.strip():
        s.data_dir = body.data_dir.strip()
    save_settings(s)
    q = queue.Queue()

    def progress(msg):
        q.put(("progress", msg))

    def work():
        try:
            os.makedirs(s.data_dir, exist_ok=True)
            progress("Chunking corpus...")
            build_chunks(
                s.corpus_dir, s.chunks_path, s.chunk_size, s.overlap, progress
            )
            if not body.skip_embed:
                progress("Embedding chunks (this can take 30-60 minutes on CPU)...")
                build_embeddings(
                    s.chunks_path,
                    s.embeddings_path,
                    s.embed_model,
                    s.ollama_url,
                    progress=progress,
                )
            progress("Building full-text index...")
            ensure_fts(load_chunks(s.chunks_path), s.fts_path)
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
