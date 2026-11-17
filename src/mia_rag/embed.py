import json
import os
import time
import urllib.request

import numpy as np


def ollama_embed(texts, model, url="http://localhost:11434", retries=5):
    endpoint = url.rstrip("/") + "/api/embed"
    payload = json.dumps({"model": model, "input": texts}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                endpoint, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            return np.asarray(data["embeddings"], dtype=np.float32)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_embeddings(
    chunks_path,
    out_path,
    model="nomic-embed-text",
    url="http://localhost:11434",
    batch=32,
    progress=None,
    limit=0,
):
    total = count_lines(chunks_path)
    if limit:
        total = min(total, limit)
    done = 0
    vectors = []
    if os.path.exists(out_path):
        old = np.load(out_path)
        if old.shape[0] <= total:
            vectors = [old]
            done = old.shape[0]

    chunks = load_chunks(chunks_path)
    t0 = time.time()
    for start in range(done, total, batch):
        part = chunks[start : start + batch]
        vectors.append(ollama_embed([c["text"] for c in part], model, url))
        done = start + len(part)
        if progress and done % (batch * 5) < batch:
            elapsed = max(time.time() - t0, 1e-6)
            rate = (done - (vectors[0].shape[0] if len(vectors) > 1 else 0)) / elapsed
            rate = max(rate, 1e-6)
            eta = (total - done) / rate / 60
            progress(f"{done}/{total} embedded  ({rate:.1f}/s, ETA {eta:.1f} min)")
        if done % (batch * 20) < batch:
            np.save(out_path, np.concatenate(vectors, axis=0))

    arr = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 768), np.float32)
    arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
    np.save(out_path, arr)
    if progress:
        progress(f"saved {arr.shape[0]} vectors")
    return arr


def embed_texts(texts, model="nomic-embed-text", url="http://localhost:11434"):
    arr = ollama_embed(texts, model, url)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
