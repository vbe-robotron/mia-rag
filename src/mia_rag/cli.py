import argparse
import os
import sys

from .config import Settings, load_settings, save_settings
from .corpus import TITLES, build_chunks
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

    n = build_chunks(
        settings.corpus_dir,
        settings.chunks_path,
        settings.chunk_size,
        settings.overlap,
        progress,
    )
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
    ap.add_argument("--books", help="comma-separated title filters, e.g. 01,17")
    ap.add_argument("--show", action="store_true", help="print retrieved sources")
    args = ap.parse_args(argv)

    settings = _apply_args(load_settings(), args)
    if args.build:
        build(settings, args.no_embed)
        save_settings(settings)
        return
    if not args.question:
        ap.error("a question is required (or use --build)")

    engine = RagEngine(settings)
    titles = None
    if args.books:
        needles = [b.strip().lower() for b in args.books.split(",") if b.strip()]
        titles = [
            t for t in TITLES.values() if any(n in t.lower() for n in needles)
        ]
        if not titles:
            sys.exit("no titles matched --books")
    result = engine.ask(" ".join(args.question), args.k, titles)
    print(result["answer"])
    if args.show:
        print("\n--- sources ---")
        for s in result["sources"]:
            print(f"[{s['n']}] {s['title']}  {s['source']}")
    save_settings(settings)


if __name__ == "__main__":
    main()
