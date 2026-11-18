import argparse
import os
import sys

from .config import Settings, load_settings, save_settings
from .corpus import TITLES, build_chunks, default_meta, find_text_files
from .embed import build_embeddings
from .index import Retriever, ensure_fts, load_chunks
from .rag import RagEngine


def _apply_args(settings, args):
    if getattr(args, "data_dir", None):
        settings.data_dir = os.path.abspath(args.data_dir)
    if getattr(args, "corpus", None):
        settings.corpus_dir = os.path.abspath(args.corpus)
    if getattr(args, "model", None):
        settings.model = args.model
    if getattr(args, "ollama", None):
        settings.ollama_url = args.ollama
    return settings


def build(settings, skip_embed=False):
    os.makedirs(settings.data_dir, exist_ok=True)
    print(f"corpus: {settings.corpus_dir}")
    print(f"index:  {settings.data_dir}")

    def progress(msg):
        print(msg, flush=True)

    info = build_chunks(
        settings.corpus_dir,
        settings.chunks_path,
        settings.chunk_size,
        settings.overlap,
        progress,
    )
    n = sum(v["chunks"] for v in info.values())
    if not skip_embed:
        build_embeddings(
            settings.chunks_path,
            settings.embeddings_path,
            settings.embed_model,
            settings.ollama_url,
            progress=progress,
        )
    ensure_fts(load_chunks(settings.chunks_path), settings.fts_path)
    print(f"index ready ({n} chunks)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mia-rag", description="RAG over Marxist economics texts")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--build", action="store_true", help="build chunks/embeddings/FTS index")
    ap.add_argument("--no-embed", action="store_true", help="with --build, skip embeddings")
    ap.add_argument("--data-dir", help="index directory")
    ap.add_argument("--corpus", help="directory containing the .txt corpus")
    ap.add_argument("--model", help="chat model")
    ap.add_argument("--ollama", help="Ollama base URL")
    ap.add_argument("--k", type=int, default=None, help="passages to retrieve")
    ap.add_argument("--books", help="comma-separated title/author filters, e.g. 01,Marx")
    ap.add_argument("--full", action="store_true", help="feed selected books in full")
    ap.add_argument("--reasoning", action="store_true", help="enable model reasoning")
    ap.add_argument("--show", action="store_true", help="print retrieved sources")
    args = ap.parse_args(argv)

    settings = _apply_args(load_settings(), args)
    if args.build:
        build(settings, args.no_embed)
        save_settings(settings)
        return
    if not args.question:
        ap.error("a question is required (or use --build)")

    if args.reasoning:
        settings.reasoning = True
    engine = RagEngine(settings)
    titles = None
    files = None
    if args.books:
        needles = [b.strip().lower() for b in args.books.split(",") if b.strip()]
        titles = []
        files = []
        for path in find_text_files(settings.corpus_dir):
            base = os.path.basename(path)
            book = os.path.splitext(base)[0]
            meta = default_meta(base)
            blob = (book + " " + meta["title"] + " " + meta["author"]).lower()
            if any(n in blob or book[:2] == n for n in needles):
                titles.append(meta["title"])
                files.append(book)
        if not titles:
            sys.exit("no titles matched --books")
    mode = "full" if args.full and files else "retrieval"
    result = engine.ask(" ".join(args.question), args.k, titles, files, mode)
    print(result["answer"])
    if args.show:
        print("\n--- sources ---")
        for s in result["sources"]:
            print(f"[{s['n']}] {s['title']}  {s['source']}")
    save_settings(settings)


if __name__ == "__main__":
    main()
