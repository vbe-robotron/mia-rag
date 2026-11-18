import glob
import json
import os
import re
import urllib.request

TITLES = {
    "01": "Marx — Capital, Vol. I",
    "02": "Marx — Capital, Vol. II",
    "03": "Marx — Capital, Vol. III",
    "04": "Marx — Grundrisse",
    "05": "Marx — Theories of Surplus Value",
    "06": "Marx — A Contribution to the Critique of Political Economy",
    "07": "Marx — Wage Labour and Capital",
    "08": "Marx — Value, Price and Profit",
    "09": "Marx — Economic and Philosophic Manuscripts of 1844",
    "10": "Marx — The Poverty of Philosophy",
    "11": "Engels — Anti-Dühring",
    "12": "Engels — Socialism: Utopian and Scientific",
    "13": "Engels — The Origin of the Family, Private Property and the State",
    "14": "Lenin — Imperialism, the Highest Stage of Capitalism",
    "15": "Lenin — The Development of Capitalism in Russia",
    "16": "Hilferding — Finance Capital",
    "17": "Luxemburg — The Accumulation of Capital",
    "18": "Luxemburg — Reform or Revolution",
    "19": "Bukharin — Imperialism and World Economy",
    "20": "Mandel — An Introduction to Marxist Economic Theory",
}


def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_pages(raw):
    parts = re.split(r"^----- SOURCE: (\S+) -----\s*$", raw, flags=re.M)
    if len(parts) == 1:
        return [("", raw)]
    return [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]


def chunk_page(text, chunk=1400, overlap=250):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    buf = ""
    for p in paras:
        while len(p) > chunk:
            if buf:
                out.append(buf)
                buf = ""
            out.append(p[:chunk])
            p = p[chunk - overlap :]
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= chunk:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            tail = buf[-overlap:] if buf else ""
            buf = (tail + "\n\n" + p).strip() if tail else p
    if buf.strip():
        out.append(buf.strip())
    return out


def find_text_files(text_dir):
    return sorted(glob.glob(os.path.join(text_dir, "*.txt")))


def build_chunks(text_dir, out_path, chunk=1400, overlap=250, progress=None, meta=None):
    files = find_text_files(text_dir)
    if not files:
        raise FileNotFoundError(f"no .txt files in {text_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    info = {}
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for path in files:
            base = os.path.basename(path)
            book = os.path.splitext(base)[0]
            m = (meta or {}).get(book) or default_meta(base)
            title = m.get("title") or book
            author = m.get("author") or ""
            raw = open(path, encoding="utf-8", errors="replace").read()
            count = 0
            for source, page in split_pages(raw):
                for piece in chunk_page(clean(page), chunk, overlap):
                    rec = {
                        "id": n,
                        "book": book,
                        "title": title,
                        "author": author,
                        "source": source,
                        "text": piece,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    count += 1
            info[book] = {"file": base, "title": title, "author": author, "chunks": count, "id_start": n - count}
            if progress:
                progress(f"{book}: {count} chunks")
    if progress:
        progress(f"total: {n} chunks")
    return info


AUTHORS = {
    "01": "Marx", "02": "Marx", "03": "Marx", "04": "Marx", "05": "Marx",
    "06": "Marx", "07": "Marx", "08": "Marx", "09": "Marx", "10": "Marx",
    "11": "Engels", "12": "Engels", "13": "Engels",
    "14": "Lenin", "15": "Lenin",
    "16": "Hilferding", "17": "Luxemburg", "18": "Luxemburg",
    "19": "Bukharin", "20": "Mandel",
}


def default_meta(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    num = stem[:2]
    if num in TITLES:
        return {"title": TITLES[num], "author": AUTHORS.get(num, "")}
    if "_" in stem:
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0].isdigit():
            author = parts[1].replace("-", " ")
            title = " ".join(parts[2:]).replace("_", " ")
            return {"title": title, "author": author}
    return {"title": stem, "author": ""}


def load_catalog(data_dir):
    path = os.path.join(data_dir, "catalog.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"docs": {}}


def save_catalog(data_dir, catalog):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "catalog.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def file_hash(path):
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def next_prefix(text_dir):
    nums = []
    for path in glob.glob(os.path.join(text_dir, "*.txt")):
        name = os.path.basename(path)
        if name[:2].isdigit():
            nums.append(int(name[:2]))
    return max(nums + [20]) + 1


def safe_filename(name):
    name = os.path.basename(name)
    name = re.sub(r"[\x00-\x1f/\\:*?\"<>|]", "_", name).strip(" .")
    return name or "document.txt"


def html_to_text(html):
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []
            self.skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript", "svg"):
                self.skip += 1
            elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"):
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript", "svg") and self.skip:
                self.skip -= 1
            elif tag in ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"):
                self.parts.append("\n")

        def handle_data(self, data):
            if not self.skip:
                self.parts.append(data)

    p = P()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _fetch(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 mia-rag/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace"), r.headers.get("Content-Type", "")


def crawl_work(url, max_pages=80, progress=None, max_attempts=120, time_budget=240):
    import time
    import urllib.parse

    base = urllib.parse.urljoin(url, ".")
    seen = set()
    queue = [url]
    pages = []
    title = ""
    attempts = 0
    deadline = time.time() + time_budget
    while (
        queue
        and len(pages) < max_pages
        and attempts < max_attempts
        and time.time() < deadline
    ):
        u = queue.pop(0)
        u = u.split("#")[0]
        if u in seen:
            continue
        seen.add(u)
        attempts += 1
        try:
            data, ctype = _fetch(u)
        except Exception:
            continue
        if "html" not in ctype and "<html" not in data[:500].lower():
            continue
        text = html_to_text(data)
        low = text.lower()
        if "out-of-date link" in low or "page not found" in low or len(text) < 800:
            continue
        if not title:
            m = re.search(r"<title[^>]*>(.*?)</title>", data, re.I | re.S)
            if m:
                title = html_to_text(m.group(1)).strip()
        pages.append((u, text))
        if progress:
            progress(f"fetched {len(pages)} pages: {u.rsplit('/', 1)[-1]}")
        links = re.findall(r'href=["\']([^"\']+)["\']', data, re.I)
        nxt = []
        for href in links:
            full = urllib.parse.urljoin(u, href)
            if not full.startswith(base):
                continue
            path = urllib.parse.urlparse(full).path
            if not re.search(r"\.html?$", path, re.I):
                continue
            if any(x in path.lower() for x in ("/admin/", "/legal/", "contact", "donate")):
                continue
            if full not in seen and full not in queue and full not in nxt:
                nxt.append(full)
        nxt.sort(key=lambda x: (0 if re.search(r"/(ch|chapter|part|preface|intro|appendix)", x, re.I) else 1, x))
        queue.extend(nxt[:500])
        time.sleep(0.1)
    if progress:
        progress(f"crawled {len(pages)} pages")
    return title, pages
