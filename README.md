# RPTU Navigator

A bilingual (EN/DE) RAG assistant built on **public RPTU web pages**. It does two things:

1. **Ask RPTU**: answers new students' questions (enrollment, exams, modules, departments) with citations to the source page.
2. **HiWi finder**: reads your resume, finds matching research groups from official pages, and drafts a personalized e-mail to the professor.

It also includes a **retrieval benchmark** comparing BM25, multilingual dense embeddings, and hybrid (Reciprocal Rank Fusion), so design choices are backed by numbers.

## Architecture

```
crawl.py (robots.txt-aware, rate-limited)  ->  data/corpus.jsonl
   -> ingest.py (clean text, chunk, keep url+title)
   -> retrievers.py: BM25 | dense (multilingual MiniLM) | hybrid RRF
   -> llm.py: Groq (openai/gpt-oss-20b), answers ONLY from numbered sources, cites [n]
   -> hiwi.py: resume -> profile+queries -> fused retrieval -> group ranking -> e-mail draft
   -> app.py (Streamlit)        evalset.py + evaluate.py (Hit@k, MRR)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your Groq API key (never commit it)

python -m navigator.crawl --max-pages 400      # add --seed-url <your dept/group site> as needed
python -m navigator.evalset --n 60             # synthetic questions, gold = source page
python -m navigator.evaluate                   # writes results/results.md
streamlit run app.py
python -m pytest                               # offline unit tests
```

## Results

Run `python -m navigator.evaluate` and paste your table here (do not guess numbers):

| retriever | hit@1 | hit@3 | hit@5 | mrr@10 |
|---|---|---|---|---|
| bm25 | | | | |
| dense | | | | |
| hybrid_rrf | | | | |

## Responsible use

- The crawler honors `robots.txt`, sends a descriptive User-Agent, waits 1 s between requests, and stays on `*.rptu.de`.
- Only public pages are used. Do not add login-protected or personal data.
- The HiWi e-mail is a **draft**. Verify names/facts and send individually; do not mass-mail professors.
- Answers can be outdated or wrong; always follow the cited link for deadlines and regulations.

## Known limitations

- Synthetic questions are LLM-written (can favor lexical overlap); a small hand-labeled set from real new-student questions would be more convincing.
- Page-level gold labels; near-duplicate EN/DE pages are treated as distinct URLs.
- PDFs (module handbooks, exam regulations) are skipped; adding a PDF loader is the best next step.

## Ideas to extend

Cross-encoder re-ranking, PDF ingestion, language-aware filtering, a freshness check (re-crawl + diff), answer-faithfulness scoring with an LLM judge.

## Disclaimer

This is an independent student project and is not affiliated with or endorsed by RPTU. It only indexes publicly available pages and links back to the official sources. The crawled data is not included in this repository; run the crawler yourself.
