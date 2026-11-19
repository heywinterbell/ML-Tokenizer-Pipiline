"""Step 3 of the pipeline: draw the vocabulary so you can see it.

A vocabulary is a table of numbers, and tables are hard to read.
A 3D bar chart of the most frequent tokens shows in one glance
which words dominate your text — and whether the vocabulary looks
sane (for example, if "the" is not near the top, something is wrong).

Two outputs:
- a text table printed to the terminal (``print_top_tokens``)
- a 3D bar chart saved as a PNG file (``plot_top_tokens``)

The chart itself is built by ``build_figure``, which returns a
matplotlib Figure. The GUI embeds that same figure in its window
instead of saving it, so there is only one place that decides what
the chart looks like.
"""

import os

from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the "3d" projection


def print_top_tokens(vocab, n=10):
    """Print the ``n`` most frequent tokens as a plain text table."""
    top = vocab.most_common(n)
    width = max(len(word) for word, _ in top)
    print(f"Top {len(top)} most frequent tokens:")
    for word, count in top:
        print(f"  {word:<{width}}  {count}")


def build_figure(vocab, n=15):
    """Build a 3D bar chart of the ``n`` most frequent tokens.

    Returns a matplotlib Figure. The chart is one row of bars: each
    bar stands on a token (x axis) and its height (z axis) is the
    token's count. The depth (y) axis carries no meaning here, so
    its ticks are hidden.
    """
    top = vocab.most_common(n)
    words = [word for word, _ in top]
    counts = [count for _, count in top]

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    x = list(range(len(words)))
    ax.bar3d(x, [0] * len(words), [0] * len(words), 0.5, 0.5, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(words, fontsize=9)
    # Tilt the labels so neighbouring token names do not run together.
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    # A low, side-on view: bars read left to right, tallest first.
    ax.view_init(elev=22, azim=-55)
    ax.set_xlabel("Token")
    ax.set_yticks([])  # the depth axis means nothing for one row of bars
    ax.set_zlabel("Count")
    ax.set_title(f"Top {len(top)} most frequent tokens")
    fig.tight_layout()
    return fig


def plot_top_tokens(vocab, n=15, output_path="outputs/token_frequency.png"):
    """Save the 3D bar chart of the ``n`` most frequent tokens as a PNG.

    Returns the path of the saved file.
    """
    fig = build_figure(vocab, n=n)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    return output_path
