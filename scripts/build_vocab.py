"""Generate docs/vocab.json from the sample corpus.

The website's tokenizer uses the exact same vocabulary as the
Python pipeline — this script builds it with the real
``Vocabulary`` class and dumps it as JSON, so the two can never
drift apart.

Run it from the repo root whenever the corpus changes:

    python3 scripts/build_vocab.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import CORPUS_PATH, load_corpus
from src.vocabulary import SPECIAL_TOKENS, Vocabulary

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT_DIR, "docs", "vocab.json")


def main():
    texts = load_corpus(CORPUS_PATH)
    vocab = Vocabulary.build(texts)
    data = {
        "special_tokens": SPECIAL_TOKENS,
        "word_to_id": vocab.word_to_id,
        "counts": dict(vocab.counts),
        "corpus": texts,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {OUT_PATH}: {len(vocab)} tokens from {len(texts)} sentences")


if __name__ == "__main__":
    main()
