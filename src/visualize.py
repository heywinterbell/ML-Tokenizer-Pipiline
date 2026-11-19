"""Step 3 of the pipeline: draw the vocabulary so you can see it.

A vocabulary is a table of numbers, and tables are hard to read.
A bar chart of the most frequent tokens shows in one glance which
words dominate your text — and whether the vocabulary looks sane
(for example, if "the" is not near the top, something is wrong).

Two outputs:
- a text table printed to the terminal (``print_top_tokens``)
- a bar chart saved as a PNG file (``plot_top_tokens``)
"""

import os

import matplotlib

# "Agg" draws to a file instead of opening a window, so this works
# on machines without a screen (servers, CI, this VM).
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def print_top_tokens(vocab, n=10):
    """Print the ``n`` most frequent tokens as a plain text table."""
    top = vocab.most_common(n)
    width = max(len(word) for word, _ in top)
    print(f"Top {len(top)} most frequent tokens:")
    for word, count in top:
        print(f"  {word:<{width}}  {count}")


def plot_top_tokens(vocab, n=15, output_path="outputs/token_frequency.png"):
    """Save a bar chart of the ``n`` most frequent tokens as a PNG.

    Returns the path of the saved file. The chart is horizontal so
    the token labels stay readable.
    """
    top = vocab.most_common(n)
    words = [word for word, _ in top]
    counts = [count for _, count in top]

    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], counts[::-1])  # reverse: biggest bar on top
    plt.xlabel("Count")
    plt.ylabel("Token")
    plt.title(f"Top {len(top)} most frequent tokens")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path
