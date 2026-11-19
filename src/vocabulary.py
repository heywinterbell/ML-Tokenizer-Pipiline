"""Step 1 of the pipeline: build a vocabulary from raw text.

A vocabulary is a dictionary that gives every word its own number
(a "token id"). The model never sees the word "cat" — it sees the
number 5. This file decides which words get a number and what that
number is.

Rules used here (kept deliberately simple):

1. Text is lowercased, so "Cat" and "cat" count as the same word.
2. Words are counted. The most frequent word gets the smallest id.
3. Words rarer than ``min_freq`` are left out — a real vocabulary
   cannot hold every typo.
4. Four special tokens always come first, so their ids never change:

   ``<PAD>`` = 0  padding, fills short sequences to equal length
   ``<UNK>`` = 1  unknown, any word not in the vocabulary
   ``<BOS>`` = 2  beginning of sentence
   ``<EOS>`` = 3  end of sentence
"""

import re
from collections import Counter

# Special tokens. Their order is the order of their ids: PAD=0, UNK=1, ...
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def split_words(text):
    """Turn text into a list of lowercase words.

    Example: ``split_words("The cat, sat!")`` returns
    ``["the", "cat", "sat"]`` — punctuation is dropped.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class Vocabulary:
    """The word <-> id mapping for one corpus of text."""

    def __init__(self, word_to_id, id_to_word, counts):
        self.word_to_id = word_to_id  # {"the": 4, "cat": 5, ...}
        self.id_to_word = id_to_word  # {4: "the", 5: "cat", ...}
        self.counts = counts  # {"the": 12, "cat": 6, ...} (real words only)

    @classmethod
    def build(cls, texts, min_freq=1):
        """Build a vocabulary from an iterable of text strings.

        ``min_freq``: a word must appear at least this many times to
        get an id. Words below it are forgotten and later become <UNK>.
        """
        # 1. Count every word in every text.
        counts = Counter()
        for text in texts:
            counts.update(split_words(text))

        # 2. Keep only words frequent enough, most frequent first.
        #    Ties are broken alphabetically so the result is the same
        #    every time you run it (deterministic).
        kept = sorted(
            (word for word, count in counts.items() if count >= min_freq),
            key=lambda word: (-counts[word], word),
        )

        # 3. Special tokens first, then real words in frequency order.
        word_to_id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        for word in kept:
            word_to_id[word] = len(word_to_id)

        id_to_word = {id_: word for word, id_ in word_to_id.items()}
        return cls(word_to_id, id_to_word, counts)

    def __len__(self):
        """Number of tokens in the vocabulary, special tokens included."""
        return len(self.word_to_id)

    def id_for(self, word):
        """Id of a word, or the <UNK> id if the word was never seen."""
        return self.word_to_id.get(word, self.word_to_id[UNK_TOKEN])

    def word_for(self, token_id):
        """Word for an id, or <UNK> if the id does not exist."""
        return self.id_to_word.get(token_id, UNK_TOKEN)

    def most_common(self, n=10):
        """The ``n`` most frequent real words as (word, count) pairs."""
        return sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
