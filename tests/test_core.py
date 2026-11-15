from navigator.crawl import extract_page, is_allowed_url
from navigator.evaluate import score
from navigator.retrievers import BM25Retriever, rrf

CHUNKS = [
    {"id": "a#0", "url": "a", "title": "", "text": "BM25 ranks documents by term frequency and inverse document frequency."},
    {"id": "b#0", "url": "b", "title": "", "text": "Rückmeldung zum nächsten Semester erfolgt über das Studierendensekretariat."},
    {"id": "c#0", "url": "c", "title": "", "text": "Gradient descent minimizes a loss function iteratively."},
]


def test_bm25_keyword_german():
    assert BM25Retriever(CHUNKS).search("Rückmeldung Semester", k=1)[0]["id"] == "b#0"


def test_rrf_prefers_agreement():
    assert rrf([[CHUNKS[0], CHUNKS[1]], [CHUNKS[1], CHUNKS[2]]], top=3)[0]["id"] == "b#0"


def test_score_dedupes_urls():
    s = score(["x", "x", "gold"], "gold")
    assert s["hit@3"] == 1.0 and s["hit@1"] == 0.0 and s["mrr@10"] == 0.5


def test_url_filter():
    assert is_allowed_url("https://cs.rptu.de/en/studium")
    assert not is_allowed_url("https://evil.com/rptu.de")
    assert not is_allowed_url("https://cs.rptu.de/files/a.pdf")
    assert not is_allowed_url("https://cs.rptu.de/forschung/kolloquien/2011")


def test_extract_page_strips_nav():
    html = "<html lang='de'><title>T</title><body><nav>MENU</nav><main>" + "Inhalt " * 50 + "</main></body></html>"
    p = extract_page(html)
    assert "MENU" not in p["text"] and p["lang"] == "de" and p["title"] == "T"
