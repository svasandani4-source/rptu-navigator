"""Synthetic eval set: an LLM writes a question per sampled chunk; that chunk's page is the gold source."""
import argparse
import json
import random

from .config import CFG
from .ingest import load_chunks
from .llm import chat, extract_json

PROMPT = """Write ONE question a new student might ask that is answered by the passage below.
Paraphrase; do not copy phrases. It must make sense on its own (never say "the passage").
Write it in {lang}. Return JSON only: {{"question": "..."}}

Passage:
{text}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    n = ap.parse_args().n
    chunks = [c for c in load_chunks(CFG.corpus_path, CFG.chunk_size, CFG.chunk_overlap) if len(c["text"]) > 300]
    rng = random.Random(CFG.seed)
    rng.shuffle(chunks)
    kept = 0
    with open(CFG.eval_path, "w", encoding="utf-8") as f:
        for c in chunks:
            if kept >= n:
                break
            lang = rng.choice(["English", "German"])
            obj = extract_json(chat(PROMPT.format(lang=lang, text=c["text"]), CFG.llm_model, temperature=0.3))
            if isinstance(obj, dict) and obj.get("question"):
                f.write(json.dumps({"question": obj["question"], "gold_url": c["url"]}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Wrote {kept} questions to {CFG.eval_path}")


if __name__ == "__main__":
    main()
