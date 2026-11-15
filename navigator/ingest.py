import json
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter


def strip_boilerplate(pages: list[dict], min_share: float = 0.2, min_pages: int = 5) -> list[dict]:
    """Drop lines that appear on many pages (menus, footers) and repeated short lines within a page."""
    limit = max(min_pages, int(min_share * len(pages)))
    df = Counter()
    for p in pages:
        df.update({ln.strip() for ln in p["text"].split("\n")})
    cleaned = []
    for p in pages:
        seen, lines = set(), []
        for ln in p["text"].split("\n"):
            s = ln.strip()
            if not s or df[s] >= limit:
                continue
            if len(s) < 40:
                if s in seen:
                    continue
                seen.add(s)
            lines.append(s)
        text = "\n".join(lines)
        if len(text) > 200:
            cleaned.append({**p, "text": text})
    return cleaned


def load_chunks(corpus_path: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    with open(corpus_path, encoding="utf-8") as f:
        pages = strip_boilerplate([json.loads(line) for line in f])
    chunks = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "id": f"{page['url']}#{i}",
                "url": page["url"],
                "title": page["title"],
                "text": f"{page['title']}\n{piece}",
            })
    if not chunks:
        raise ValueError("Corpus is empty. Run: python -m navigator.crawl")
    return chunks
