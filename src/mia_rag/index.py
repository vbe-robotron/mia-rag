import json
import os
import re
import sqlite3

import numpy as np


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def nav_ids(chunks):
    out = set()
    for c in chunks:
        src = c["source"]
        tail = src.rsplit("/", 1)[-1].lower()
        if not src or src.endswith("/") or "index" in tail:
            out.add(c["id"])
            continue
        lines = [l.strip() for l in c["text"].split("\n") if l.strip()]
        toc = sum(1 for l in lines if re.match(r"^(Ch\.|Chapter|Part)\s+[IVXL0-9]", l))
        if toc >= 3:
            out.add(c["id"])
    return out


def ensure_fts(chunks, fts_path):
    if os.path.exists(fts_path) and os.path.getsize(fts_path) > 0:
        try:
            con = sqlite3.connect(fts_path)
            (count,) = con.execute("SELECT count(*) FROM chunks").fetchone()
            con.close()
            if count == len(chunks):
                return
        except Exception:
            pass
        try:
            os.remove(fts_path)
        except OSError:
            pass
    os.makedirs(os.path.dirname(os.path.abspath(fts_path)), exist_ok=True)
    tmp = fts_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5(text, tokenize='porter unicode61 remove_diacritics 2')"
    )
    con.executemany(
        "INSERT INTO chunks(rowid, text) VALUES (?, ?)",
        ((c["id"], c["text"]) for c in chunks),
    )
    con.commit()
    con.close()
    os.replace(tmp, fts_path)


def fts_search(fts_path, query, allowed, k):
    tokens = re.findall(r"[A-Za-z0-9\u00c0-\u024f']+", query)
    tokens = [t for t in tokens if len(t) > 1][:24]
    if not tokens:
        return []
    match = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
    con = sqlite3.connect(fts_path)
    try:
        rows = con.execute(
            "SELECT rowid, bm25(chunks) FROM chunks WHERE chunks MATCH ? "
            "ORDER BY bm25(chunks) LIMIT ?",
            (match, max(k * 10, 100)),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()
    out = []
    for rid, score in rows:
        if allowed is None or rid in allowed:
            out.append((rid, score))
        if len(out) >= k:
            break
    return out


def vector_search(qvecs, emb, allowed, k):
    scores = np.max(np.stack([emb @ v for v in qvecs]), axis=0)
    if allowed is None:
        idx = np.argpartition(-scores, k)[:k]
    else:
        keep = np.fromiter(allowed, dtype=np.int64)
        sub = scores[keep]
        j = np.argpartition(-sub, min(k, len(sub) - 1))[:k]
        idx = keep[j]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def rrf(vector, bm25_lists, k=60):
    scores = {}
    for rank, (rid, _) in enumerate(vector):
        scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank + 1)
    for bm25 in bm25_lists:
        for rank, (rid, _) in enumerate(bm25):
            scores[rid] = scores.get(rid, 0) + 0.5 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, chunks_path, embeddings_path, fts_path):
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"chunks file not found: {chunks_path}")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"embeddings file not found: {embeddings_path}")
        self.chunks = load_chunks(chunks_path)
        self.emb = np.load(embeddings_path)
        if self.emb.shape[0] != len(self.chunks):
            raise ValueError(
                f"embedding count {self.emb.shape[0]} != chunk count {len(self.chunks)}"
            )
        self.fts_path = fts_path
        ensure_fts(self.chunks, fts_path)
        self.nav = nav_ids(self.chunks)

    def titles(self):
        seen = []
        for c in self.chunks:
            if c["title"] not in seen:
                seen.append(c["title"])
        return seen

    def docs(self):
        docs = {}
        for c in self.chunks:
            key = c.get("book") or c["title"]
            if key not in docs:
                docs[key] = {
                    "book": key,
                    "title": c["title"],
                    "author": c.get("author", ""),
                    "chunks": 0,
                }
            docs[key]["chunks"] += 1
        return list(docs.values())

    def allowed_ids(self, titles=None, files=None):
        allowed = set(range(len(self.chunks))) - self.nav
        if files:
            wanted = set(files)
            allowed = {i for i in allowed if self.chunks[i].get("book") in wanted}
        elif titles:
            wanted = set(titles)
            allowed = {i for i in allowed if self.chunks[i]["title"] in wanted}
        return allowed

    def _expand(self, picked, extra=6):
        seen = {rid for rid, _ in picked}
        out = list(picked)
        for rid, score in picked:
            if len(out) >= len(picked) + extra:
                break
            base = self.chunks[rid]
            for nid in (rid - 1, rid + 1):
                if (
                    nid in seen
                    or nid < 0
                    or nid >= len(self.chunks)
                    or nid in self.nav
                ):
                    continue
                c = self.chunks[nid]
                if c.get("book") == base.get("book") and c["source"] == base["source"]:
                    out.append((nid, score * 0.99))
                    seen.add(nid)
        return out

    def search(self, qvecs, queries, k=10, titles=None, files=None, expand=False):
        allowed = self.allowed_ids(titles, files)
        vec = vector_search(qvecs, self.emb, allowed, max(k * 3, 30))
        bm = [fts_search(self.fts_path, q, allowed, max(k * 3, 30)) for q in queries]
        fused = rrf(vec, bm)
        pinned = []
        for hits in bm[1:]:
            for rid, _ in hits[:2]:
                if rid not in [p[0] for p in pinned]:
                    pinned.append((rid, 0))
        pinned = pinned[:6]
        picked = []
        seen = set()
        for rid, s in fused[: max(k - len(pinned), 1)] + pinned:
            if rid not in seen:
                picked.append((rid, s))
                seen.add(rid)
        picked = picked[:k]
        if expand:
            picked = self._expand(picked)
        primary, extra = [], []
        per_source = {}
        for rid, score in picked:
            src = self.chunks[rid]["source"]
            if per_source.get(src, 0) < 1:
                primary.append((rid, score))
                per_source[src] = per_source.get(src, 0) + 1
            else:
                extra.append((rid, score))
        return (primary + extra)[: max(k, len(primary))]
