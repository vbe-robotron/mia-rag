import json
import os
import re

from .api import ApiClient
from .config import Settings
from .corpus import clean, find_text_files, split_pages
from .embed import embed_texts
from .index import Retriever

STRICT_PROMPT = (
    "You are a research assistant specializing in Marxist political economy. "
    "Answer the user's question using ONLY the provided excerpts. "
    "Cite supporting excerpts with bracketed numbers like [1], [3]. "
    "Quote key passages when useful. If the excerpts are insufficient, say so explicitly "
    "and distinguish what the texts do and do not establish. "
    "Answer in the language of the user's question."
)

TUTOR_PROMPT = (
    "You are an expert teacher of Marxist political economy writing for an intelligent "
    "reader who wants to understand the subject, not a literature review. "
    "Write a coherent, well-structured explanation that actually teaches the concept: "
    "define it, explain its internal logic, show how its parts fit together, and note "
    "the main debates. Synthesize the provided sources into your own argument instead "
    "of summarizing excerpt by excerpt. Cite specific claims and quotations with [n]. "
    "You may supply standard background context without citation, but never invent "
    "quotations or attributions. Do not open with disclaimers about the sources and do "
    "not make the answer mainly about what the sources lack; if something important is "
    "unsupported, mention it briefly at the end. "
    "Interpret the material through historical materialism and dialectics: explain ideas "
    "in terms of the mode of production, class relations, and historical conditions; look "
    "for contradictions, their unity and struggle, quantitative change becoming "
    "qualitative change, and the negation of the negation. Treat capitalism as a "
    "historically specific, contradictory totality rather than a set of timeless facts. "
    "Use headings and short paragraphs. Answer in the language of the question."
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
        style = getattr(self.settings, "answer_style", "tutor")
        prompt = STRICT_PROMPT if style == "strict" else TUTOR_PROMPT
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Excerpts:\n\n{context}\n\n---\n\nQuestion: {question}",
            },
        ]

    def _group(self, picked):
        order = []
        by_url = {}
        for rid, _ in picked:
            c = self.retriever.chunks[rid]
            url = c["source"]
            if url not in by_url:
                by_url[url] = {
                    "n": len(order) + 1,
                    "title": c["title"],
                    "source": url,
                    "texts": [],
                }
                order.append(by_url[url])
            by_url[url]["texts"].append(c["text"])
        blocks = []
        sources = []
        for item in order:
            block = (
                f"[{item['n']}] {item['title']}\nSOURCE: {item['source']}\n\n"
                + "\n\n".join(item["texts"])
            )
            blocks.append(block)
            sources.append(
                {"n": item["n"], "title": item["title"], "source": item["source"]}
            )
        return blocks, sources

    def ask_mapreduce(self, question, k=36, titles=None, files=None):
        if self.retriever is None:
            yield ("status", "Loading index...")
            self.load_index()
        yield ("status", "Preparing search queries...")
        queries = self.prepare_queries(question)
        yield ("status", "Embedding query...")
        qvecs = embed_texts(
            queries, self.settings.embed_model, self.settings.ollama_url
        )
        yield ("status", "Collecting a broad candidate set...")
        picked = self.retriever.search(
            qvecs, queries, max(k, 24), titles, files, expand=False
        )
        blocks, sources = self._group(picked)
        yield ("queries", queries)
        yield ("sources", sources)
        summaries = []
        batch = 6
        for i in range(0, len(blocks), batch):
            part = blocks[i : i + batch]
            yield (
                "status",
                f"Summarizing sources {i + 1}-{i + len(part)} of {len(blocks)}...",
            )
            msg = [
                {
                    "role": "system",
                    "content": (
                        "You are a careful research assistant. From the numbered sources, "
                        "extract the points that bear on the question: definitions, causal "
                        "claims, contradictions, historical conditions, and debates. Cite "
                        "the source numbers as [n]. Be concise (at most 220 words). Do not "
                        "invent quotations or claims."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nSources:\n\n"
                    + "\n\n---\n\n".join(part),
                },
            ]
            summaries.append(self.api.chat(msg, max_tokens=1500, temperature=0.2))
        combined = "\n\n".join(
            f"### Partial summary {i + 1}\n{s}" for i, s in enumerate(summaries)
        )
        yield ("status", "Synthesizing the final answer...")
        messages = [
            {
                "role": "system",
                "content": TUTOR_PROMPT
                + "\n\nYou are given partial summaries of many sources, each citing "
                "source numbers [n]. Write one unified explanation; preserve the citation "
                "numbers and do not claim more than the summaries support.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPartial summaries:\n\n{combined}",
            },
        ]
        for piece in self.api.chat_stream(messages, max_tokens=8000):
            yield ("delta", piece)
        yield ("done", None)

    def ask_stream(
        self, question, k=None, titles=None, files=None, mode="retrieval"
    ):
        if self.retriever is None:
            yield ("status", "Loading index...")
            self.load_index()

        if mode == "mapreduce":
            for event in self.ask_mapreduce(question, k or 36, titles, files):
                yield event
            return

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
        blocks, sources = self._group(picked)
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
        blocks, sources = self._group(picked)
        context = "\n\n---\n\n".join(blocks)
        if progress:
            progress("Generating answer...")
        answer = self.api.chat(self._messages(question, context), max_tokens=6000)
        return {"answer": answer, "sources": sources, "queries": queries}
