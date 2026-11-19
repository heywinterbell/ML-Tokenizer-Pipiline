"""Runs the whole tokenizer pipeline end to end.

    python -m src.pipeline

What happens, in order:

1. Read the sample corpus (``data/sample_corpus.txt``).
2. Train a tokenizer on it — that builds the vocabulary.
3. Tokenize one example sentence and show its tokens, ids, and
   the decoded text, plus one sentence with unknown words.
4. Print the most frequent tokens and save their bar chart to
   ``outputs/token_frequency.png``.
"""

import os

from .tokenizer import Tokenizer
from .visualize import plot_top_tokens, print_top_tokens

# Paths are built from this file's location, so the pipeline works
# no matter which directory you run it from.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(ROOT_DIR, "data", "sample_corpus.txt")
GRAPH_PATH = os.path.join(ROOT_DIR, "outputs", "token_frequency.png")

DEMO_SENTENCE = "the cat sat on the mat"
UNKNOWN_SENTENCE = "the zebra danced under the moon"


def load_corpus(path):
    """Read the corpus file, one sentence per non-empty line."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    texts = load_corpus(CORPUS_PATH)
    tokenizer = Tokenizer.train(texts)

    print(f"Corpus: {len(texts)} sentences")
    print(f"Vocabulary size: {len(tokenizer)} tokens (including 4 special tokens)")
    print()

    print(f"Sentence:  {DEMO_SENTENCE!r}")
    print(f"Tokens:    {tokenizer.tokens(DEMO_SENTENCE)}")
    print(f"Token ids: {tokenizer.encode(DEMO_SENTENCE)}")
    print(f"Decoded:   {tokenizer.decode(tokenizer.encode(DEMO_SENTENCE))!r}")
    print()

    # "zebra", "danced", "under" and "moon" are not in the corpus,
    # so they all become <UNK> (id 1). That is the vocabulary saying
    # "I have never seen this word".
    print(f"Sentence:  {UNKNOWN_SENTENCE!r}")
    print(f"Token ids: {tokenizer.encode(UNKNOWN_SENTENCE)}")
    print(f"Decoded:   {tokenizer.decode(tokenizer.encode(UNKNOWN_SENTENCE))!r}")
    print()

    print_top_tokens(tokenizer.vocab, n=10)
    print()
    graph = plot_top_tokens(tokenizer.vocab, n=15, output_path=GRAPH_PATH)
    print(f"Graph saved to {graph}")


if __name__ == "__main__":
    main()
