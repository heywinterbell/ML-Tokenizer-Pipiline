# ML Tokenizer Pipeline

A beginner-friendly ML pipeline that turns raw text into numbers a machine
learning model can understand — and shows you what happened at every step.

## What is a tokenizer?

Machine learning models do not read words. They read **numbers**. A tokenizer
is the bridge between the two:

```text
"the cat sat on the mat"  -->  [2, 5, 6, 7, 8, 5, 9, 3]
```

It does this in two steps:

1. **Build a vocabulary** — read a bunch of text, count every word, and give
   each unique word its own number (called a *token id*). The most frequent
   word gets the smallest id.
2. **Encode text** — replace every word in a new sentence with its id from
   the vocabulary. Words the vocabulary has never seen become `<UNK>`
   (unknown).

This project does both steps, then draws a **3D graph of the most frequent
tokens** so you can *see* the vocabulary instead of just trusting it.

## What this pipeline does

```text
data/sample_corpus.txt
        │
        ▼
┌──────────────────┐
│ 1. Build vocab   │  Count words, keep the frequent ones,
│   (vocabulary)   │  add special tokens (<PAD>, <UNK>, ...)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. Tokenize      │  Turn sentences into lists of token ids,
│   (tokenizer)    │  and back again (decode).
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. Visualize     │  3D bar chart of the most frequent tokens,
│   (visualize)    │  saved to outputs/token_frequency.png
└──────────────────┘
```

## Project structure

```text
ML-Tokenizer-Pipiline/
├── README.md                  You are here
├── requirements.txt           Only dependency: matplotlib
├── data/
│   └── sample_corpus.txt      Small example text to learn from
├── src/
│   ├── vocabulary.py          Step 1 — count words, build the vocabulary
│   ├── tokenizer.py           Step 2 — encode text to ids, decode ids to text
│   ├── visualize.py           Step 3 — draw the 3D token frequency graph
│   ├── gui.py                 Live window — type a sentence, see tokens instantly
│   └── pipeline.py            Runs all three steps end to end
├── tests/
│   └── test_tokenizer.py      Checks that each step behaves correctly
└── outputs/                   Generated files (the graph) — not committed
```

## Quick start

```bash
# 1. Install the only dependency (for the graph)
pip install -r requirements.txt

# 2. Run the whole pipeline in the terminal
python3 -m src.pipeline

# 3. Or open the live GUI window instead
python3 -m src.gui

# 4. Run the tests
python3 -m unittest discover -s tests -v
```

On macOS and Linux the command is `python3` — plain `python` does not
exist there. (On Windows, `python` is the usual spelling.)

## Try the GUI

`python3 -m src.gui` opens a window with a text box. Type any sentence
and the tokens, token ids and decoded text update as you type. Words
the vocabulary has never seen are listed in red as unknown, and the
3D bar chart of the most frequent tokens is drawn at the bottom of the
window. It uses tkinter, which ships with Python, so there is nothing
extra to install.

## Example output

```text
Vocabulary size: 28 tokens (including 4 special tokens)

Sentence:  "the cat sat on the mat"
Tokens:    ['<BOS>', 'the', 'cat', 'sat', 'on', 'the', 'mat', '<EOS>']
Token ids: [2, 4, 5, 10, 9, 4, 7, 3]
Decoded:   "the cat sat on the mat"

Top 5 most frequent tokens:
  the    19
  cat     7
  dog     6
  mat     5
  and     3

Graph saved to outputs/token_frequency.png
```

## How the vocabulary is built

1. **Split the text into words.** Text is lowercased and split on
   whitespace and punctuation, so `"Cat"` and `"cat"` count as the same word.
2. **Count how often each word appears.**
3. **Drop rare words.** Words appearing fewer than `min_freq` times (default
   1) are left out — a real vocabulary cannot hold every typo.
4. **Give every kept word an id.** Four special tokens always come first:

   | Token | Id | Meaning |
   |-------|----|---------|
   | `<PAD>` | 0 | Padding — fills short sequences to equal length |
   | `<UNK>` | 1 | Unknown — any word not in the vocabulary |
   | `<BOS>` | 2 | Beginning of sentence |
   | `<EOS>` | 3 | End of sentence |

   Then real words follow, most frequent first. So the most common word in
   your text gets id 4, the next gets 5, and so on.

## Ideas to try next

- **Save / load the vocabulary** (`vocab.json`) so you can reuse it without
  rebuilding it every time.
- **Subword tokenization (BPE)** — split rare words into pieces instead of
  marking them `<UNK>`, so nothing is ever fully unknown.
- **Train a tiny model on the token ids** — for example, predict the next
  token. That is the first step toward a language model.
