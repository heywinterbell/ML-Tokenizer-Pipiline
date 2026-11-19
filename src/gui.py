"""A live GUI for the tokenizer pipeline.

Run it with:  python3 -m src.gui

A window opens with a text box. Type any sentence and the tokens,
token ids and decoded text update as you type — no need to press
anything. Words the vocabulary has never seen are listed in red as
unknown. The 3D bar chart at the bottom shows the most frequent tokens.

Built with tkinter, which ships with Python, so there is nothing
extra to install. The 3D bar chart is drawn with matplotlib's TkAgg
backend, which draws the figure inside the window instead of
saving it to a file.
"""

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .pipeline import CORPUS_PATH, load_corpus
from .tokenizer import Tokenizer
from .visualize import build_figure
from .vocabulary import UNK_TOKEN, split_words


def describe_result(tokenizer, text):
    """Tokenize ``text`` and return the results as plain data.

    Kept separate from the window code so it can be tested without
    opening a GUI. Returns a dict with:

    - ``tokens``: the word tokens, with <BOS>/<EOS>
    - ``ids``: the token ids
    - ``decoded``: the sentence decoded back from the ids
    - ``unknown``: words the vocabulary has never seen, sorted
    """
    tokens = tokenizer.tokens(text)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    unk_id = tokenizer.vocab.word_to_id[UNK_TOKEN]
    unknown = sorted(
        {word for word in split_words(text) if tokenizer.vocab.id_for(word) == unk_id}
    )
    return {"tokens": tokens, "ids": ids, "decoded": decoded, "unknown": unknown}


class TokenizerApp:
    """The window. Owns the widgets and updates them as you type."""

    def __init__(self, root, tokenizer, corpus_size):
        self.tokenizer = tokenizer
        self.root = root
        self.root.title("ML Tokenizer — Live Demo")

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=f"Trained on {corpus_size} sentences · "
            f"Vocabulary: {len(tokenizer)} tokens",
        ).pack(anchor=tk.W)

        ttk.Label(frame, text="Type a sentence:").pack(anchor=tk.W, pady=(12, 4))

        self.entry = ttk.Entry(frame, width=60)
        self.entry.pack(fill=tk.X)
        self.entry.insert(0, "the cat sat on the mat")
        # Update on every keystroke (live) and on Enter.
        self.entry.bind("<KeyRelease>", lambda event: self.update())
        self.entry.bind("<Return>", lambda event: self.update())

        ttk.Button(frame, text="Tokenize", command=self.update).pack(
            anchor=tk.W, pady=(8, 12)
        )

        self.tokens_label = self._result_label(frame, "Tokens:")
        self.ids_label = self._result_label(frame, "Token ids:")
        self.decoded_label = self._result_label(frame, "Decoded:")
        self.unknown_label = ttk.Label(frame, text="", foreground="red")
        self.unknown_label.pack(anchor=tk.W, pady=(2, 12))

        ttk.Label(frame, text="Most frequent tokens (3D):").pack(anchor=tk.W)
        canvas = FigureCanvasTkAgg(
            build_figure(tokenizer.vocab, n=10), master=frame
        )
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Tip: words the vocabulary has never seen show up as <UNK> (id 1).",
        ).pack(anchor=tk.W, pady=(12, 0))

        self.update()

    @staticmethod
    def _result_label(frame, title):
        """A bold title plus a monospace value label, packed as a pair."""
        ttk.Label(frame, text=title, font=("", 10, "bold")).pack(anchor=tk.W)
        label = ttk.Label(frame, text="", font=("Menlo", 11), wraplength=650)
        label.pack(anchor=tk.W, pady=(0, 8))
        return label

    def update(self):
        """Read the text box, tokenize it, and refresh the labels."""
        result = describe_result(self.tokenizer, self.entry.get())
        self.tokens_label.config(text=str(result["tokens"]))
        self.ids_label.config(text=str(result["ids"]))
        self.decoded_label.config(text=repr(result["decoded"]))
        if result["unknown"]:
            self.unknown_label.config(
                text="Unknown words: " + ", ".join(result["unknown"])
            )
        else:
            self.unknown_label.config(text="Unknown words: none")


def main():
    """Train the tokenizer on the sample corpus and open the window."""
    texts = load_corpus(CORPUS_PATH)
    tokenizer = Tokenizer.train(texts)
    root = tk.Tk()
    TokenizerApp(root, tokenizer, corpus_size=len(texts))
    root.mainloop()


if __name__ == "__main__":
    main()
