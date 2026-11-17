import glob
import json
import os
import re

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


def build_chunks(text_dir, out_path, chunk=1400, overlap=250, progress=None):
    files = find_text_files(text_dir)
    if not files:
        raise FileNotFoundError(f"no .txt files in {text_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for path in files:
            base = os.path.basename(path)
            num = base[:2]
            book = os.path.splitext(base)[0]
            title = TITLES.get(num, book)
            raw = open(path, encoding="utf-8", errors="replace").read()
            count = 0
            for source, page in split_pages(raw):
                for piece in chunk_page(clean(page), chunk, overlap):
                    rec = {
                        "id": n,
                        "book": book,
                        "title": title,
                        "source": source,
                        "text": piece,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    count += 1
            if progress:
                progress(f"{book}: {count} chunks")
    if progress:
        progress(f"total: {n} chunks")
    return n
