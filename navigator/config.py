from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seeds: tuple = (
        "https://rptu.de/",
        "https://cs.rptu.de/en/",
        "https://cs.rptu.de/",
        "https://rptu.de/ueber-die-rptu/rptu-als-arbeitgeberin/stellenangebote-und-karriere/hiwi-stellenboerse",
    )  # add your department / research-group sites here
    domain_suffix: str = "rptu.de"
    max_pages: int = 400
    max_depth: int = 3
    delay_s: float = 1.0           # polite crawling: one request per second
    corpus_path: str = "data/corpus.jsonl"
    eval_path: str = "data/eval_set.jsonl"
    chunk_size: int = 800
    chunk_overlap: int = 120
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # German + English
    llm_model: str = "openai/gpt-oss-20b"
    seed: int = 42


CFG = Config()
