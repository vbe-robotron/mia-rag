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
        return
    os.makedirs(os.path.dirname(os.path.abspath(fts_path)), exist_ok=True)
    tmp = fts_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5(text, tokenize='unicode61 remove_diacritics 2')"
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

    def allowed_ids(self, titles=None):
        allowed = set(range(len(self.chunks))) - self.nav
        if titles:
            wanted = set(titles)
            allowed = {i for i in allowed if self.chunks[i]["title"] in wanted}
        return allowed

    def search(self, qvecs, queries, k=10, titles=None):
        allowed = self.allowed_ids(titles)
        vec = vector_search(qvecs, self.emb, allowed, max(k * 3, 30))
        bm = [fts_search(self.fts_path, q, allowed, max(k * 3, 30)) for q in queries]
        fused = rrf(vec, bm)
        pinned = [hits[0] for hits in bm[1:] if hits]
        picked = []
        seen = set()
        for rid, s in fused[: max(k - len(pinned), 1)] + pinned:
            if rid not in seen:
                picked.append((rid, s))
                seen.add(rid)
        return picked[:k]
