"""Polite crawler for public *.rptu.de pages. Respects robots.txt, rate-limits, stays on-domain."""
import argparse
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from .config import CFG

UA = "rptu-navigator/0.1 (student portfolio project; polite crawler)"
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".docx", ".xlsx", ".pptx", ".mp4", ".svg", ".css", ".js")
SKIP_PATTERNS = (r"/kolloquien/", r"/archive/", r"/rundmail", r"/login", r"/suche")


def is_allowed_url(url: str, suffix: str = CFG.domain_suffix) -> bool:
    p = urlparse(url)
    host = p.hostname or ""
    if p.scheme not in ("http", "https"):
        return False
    if not (host == suffix or host.endswith("." + suffix)):
        return False
    if p.path.lower().endswith(SKIP_EXT):
        return False
    return not any(re.search(pat, p.path.lower()) for pat in SKIP_PATTERNS)


def extract_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    lang = (soup.html.get("lang") if soup.html else None) or ""
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    text = re.sub(r"\n{2,}", "\n\n", main.get_text("\n", strip=True))
    return {"title": title, "lang": lang[:2], "text": text}


def crawl(seeds, max_pages, max_depth, delay, out, include=()):
    robots: dict = {}

    def can_fetch(url: str) -> bool:
        p = urlparse(url)
        if p.netloc not in robots:
            rp = RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None
            robots[p.netloc] = rp
        rp = robots[p.netloc]
        return bool(rp and rp.can_fetch(UA, url))

    session = requests.Session()
    session.headers["User-Agent"] = UA
    queue = deque((s, 0) for s in seeds)
    seen, saved = set(), 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        while queue and saved < max_pages:
            url, depth = queue.popleft()
            url = urldefrag(url)[0].rstrip("/")
            if url in seen or not is_allowed_url(url):
                continue
            if depth > 0 and include and not any(re.search(p, url) for p in include):
                continue  # followed links must match --include (seeds are exempt)
            if not can_fetch(url):
                continue
            seen.add(url)
            try:
                r = session.get(url, timeout=20)
            except requests.RequestException:
                continue
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue
            page = extract_page(r.text)
            if len(page["text"]) > 200:
                f.write(json.dumps({"url": url, **page}, ensure_ascii=False) + "\n")
                saved += 1
                print(f"[{saved}] {url}")
            if depth < max_depth:
                for a in BeautifulSoup(r.text, "html.parser").find_all("a", href=True):
                    queue.append((urljoin(url, a["href"]), depth + 1))
            time.sleep(delay)
    print(f"Saved {saved} pages to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=CFG.max_pages)
    ap.add_argument("--max-depth", type=int, default=CFG.max_depth)
    ap.add_argument("--seed-url", action="append", help="seed URL (repeatable)")
    ap.add_argument("--no-default-seeds", action="store_true", help="ignore the seeds in config.py")
    ap.add_argument("--include", action="append", help="regex; followed links must match one (repeatable)")
    a = ap.parse_args()
    seeds = (a.seed_url or []) + ([] if a.no_default_seeds else list(CFG.seeds))
    crawl(seeds, a.max_pages, a.max_depth, CFG.delay_s, CFG.corpus_path, a.include or [])
