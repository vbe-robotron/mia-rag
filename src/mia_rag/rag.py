import json
import os
import re

from .api import ApiClient
from .config import Settings
from .corpus import clean, find_text_files, split_pages
from .embed import embed_texts
from .index import Retriever

SYSTEM_PROMPT = (
    "You are a research assistant specializing in Marxist political economy. "
    "Answer the user's question using ONLY the provided excerpts. "
    "Cite supporting excerpts with bracketed numbers like [1], [3]. "
    "Quote key passages when useful. If the excerpts are insufficient, say so explicitly "
    "and distinguish what the texts do and do not establish. "
    "Answer in the language of the user's question."
)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def extract_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except Exception:
                    return None
    return None


class RagEngine:
    def __init__(self, settings=None, retriever=None):
        self.settings = settings or Settings()
        self.api = ApiClient(self.settings)
        self.retriever = retriever

    def load_index(self):
        s = self.settings
        self.retriever = Retriever(s.chunks_path, s.embeddings_path, s.fts_path)
        return self.retriever

    def _doc_map(self):
        if self.retriever is None:
            self.load_index()
        docs = {d["book"]: d for d in self.retriever.docs()}
        paths = {}
        for path in find_text_files(self.settings.corpus_dir):
            paths[os.path.splitext(os.path.basename(path))[0]] = path
        return docs, paths

    def full_context(self, files):
        docs, paths = self._doc_map()
        budget = self.settings.full_text_max_chars
        total = 0
        n = 0
        parts = []
        sources = []
        for book in files:
            path = paths.get(book)
            if not path:
                continue
            meta = docs.get(book, {})
            title = meta.get("title") or book
            with open(path, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            for source, page in split_pages(raw):
                text = clean(page)
                if not text:
                    continue
                n += 1
                block = f"[{n}] {title}\nSOURCE: {source}\n\n{text}"
                total += len(block)
                if total > budget:
                    return None, None
                parts.append(block)
                sources.append({"n": n, "title": title, "source": source})
        if not parts:
            return None, None
        return "\n\n---\n\n".join(parts), sources

    def prepare_queries(self, query):
        if is_ascii(query):
            return [query]
        prompt = (
            "Read the text below and produce search queries for a full-text index of "
            "Marxist economics literature. Return ONLY a JSON object with keys "
            '"translation" (a concise English statement of its central question, at most '
            '80 words) and "keywords" (an array of 4 short English queries, 2-6 words '
            "each, using technical terms from political economy).\n\n" + query
        )
        data = None
        for max_tokens in (2500, 5000):
            try:
                raw = self.api.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                    reasoning=True,
                )
                data = extract_json(raw)
                if data and data.get("translation"):
                    break
            except Exception:
                pass
        if data:
            out = [data.get("translation") or query]
            out += [k for k in data.get("keywords", []) if isinstance(k, str)]
            out = [q.strip() for q in out if q and q.strip() and is_ascii(q)]
            if out:
                return out[:5]
        try:
            one = self.api.chat(
                [
                    {
                        "role": "user",
                        "content": "Translate the central question of the following text "
                        "into one concise English sentence. Output only the sentence.\n\n"
                        + query,
                    }
                ],
                max_tokens=1200,
                temperature=0,
            ).strip()
            if one and is_ascii(one):
                return [one]
        except Exception:
            pass
        return [query]

    def _messages(self, question, context):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Excerpts:\n\n{context}\n\n---\n\nQuestion: {question}",
            },
        ]

    def ask_stream(
        self, question, k=None, titles=None, files=None, mode="retrieval"
    ):
        if self.retriever is None:
            yield ("status", "Loading index...")
            self.load_index()

        if mode == "full" and files:
            yield ("status", "Loading selected books in full...")
            context, sources = self.full_context(files)
            if context:
                yield ("queries", [])
                yield ("sources", sources)
                yield ("status", "Generating answer from full text...")
                for piece in self.api.chat_stream(self._messages(question, context)):
                    yield ("delta", piece)
                yield ("done", None)
                return
            yield ("status", "Selected books exceed the context budget; using retrieval.")

        yield ("status", "Preparing search queries...")
        queries = self.prepare_queries(question)
        yield ("status", "Embedding query...")
        qvecs = embed_texts(
            queries, self.settings.embed_model, self.settings.ollama_url
        )
        yield ("status", "Retrieving passages...")
        picked = self.retriever.search(
            qvecs,
            queries,
            k or self.settings.top_k,
            titles,
            files,
            expand=(mode == "wide"),
        )
        blocks = []
        sources = []
        for i, (rid, _) in enumerate(picked, 1):
            c = self.retriever.chunks[rid]
            blocks.append(f"[{i}] {c['title']}\nSOURCE: {c['source']}\n\n{c['text']}")
            sources.append({"n": i, "title": c["title"], "source": c["source"]})
        yield ("queries", queries)
        yield ("sources", sources)
        yield ("status", "Generating answer...")
        for piece in self.api.chat_stream(self._messages(question, "\n\n---\n\n".join(blocks))):
            yield ("delta", piece)
        yield ("done", None)

    def ask(
        self, question, k=None, titles=None, files=None, mode="retrieval", progress=None
    ):
        if self.retriever is None:
            if progress:
                progress("Loading index...")
            self.load_index()
        if progress:
            progress("Preparing search queries...")
        queries = self.prepare_queries(question)
        if progress:
            progress("Embedding query...")
        qvecs = embed_texts(
            queries, self.settings.embed_model, self.settings.ollama_url
        )
        if progress:
            progress("Retrieving passages...")
        picked = self.retriever.search(
            qvecs,
            queries,
            k or self.settings.top_k,
            titles,
            files,
            expand=(mode == "wide"),
        )
        blocks = []
        sources = []
        for i, (rid, _) in enumerate(picked, 1):
            c = self.retriever.chunks[rid]
            blocks.append(f"[{i}] {c['title']}\nSOURCE: {c['source']}\n\n{c['text']}")
            sources.append({"n": i, "title": c["title"], "source": c["source"]})
        context = "\n\n---\n\n".join(blocks)
        if progress:
            progress("Generating answer...")
        answer = self.api.chat(self._messages(question, context), max_tokens=6000)
        return {"answer": answer, "sources": sources, "queries": queries}
