"""Compare retrievers: page-level Hit@k and MRR@10 (a hit = the gold page appears among retrieved pages)."""
import json
from pathlib import Path

import pandas as pd

from .config import CFG
from .ingest import load_chunks
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever

KS, DEPTH = (1, 3, 5), 20


def score(ranked_urls: list[str], gold: str) -> dict:
    urls = list(dict.fromkeys(ranked_urls))  # dedupe, keep order
    out = {f"hit@{k}": float(gold in urls[:k]) for k in KS}
    out["mrr@10"] = next((1 / (i + 1) for i, u in enumerate(urls[:10]) if u == gold), 0.0)
    return out


def main():
    chunks = load_chunks(CFG.corpus_path, CFG.chunk_size, CFG.chunk_overlap)
    qa = [json.loads(line) for line in open(CFG.eval_path, encoding="utf-8")]
    print(f"{len(chunks)} chunks, {len(qa)} questions")
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks, CFG.embed_model)
    rows = []
    for r in (bm25, dense, HybridRetriever(bm25, dense)):
        per_q = [score([c["url"] for c in r.search(x["question"], DEPTH)], x["gold_url"]) for x in qa]
        rows.append({"retriever": r.name, **pd.DataFrame(per_q).mean().round(3).to_dict()})
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/results.csv", index=False)
    Path("results/results.md").write_text(df.to_markdown(index=False))
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
