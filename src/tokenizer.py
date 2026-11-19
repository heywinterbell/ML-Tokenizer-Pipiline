"""Step 2 of the pipeline: turn sentences into token ids and back.

The vocabulary (see vocabulary.py) is the dictionary. This file is the
translator that uses it:

- ``encode("the cat sat")`` -> ``[2, 4, 5, 6, 3]`` (ids, with <BOS>/<EOS>)
- ``decode([2, 4, 5, 6, 3])`` -> ``"the cat sat"`` (words again)

Words the vocabulary never saw become ``<UNK>`` instead of crashing —
that is exactly what the unknown token is for.
"""

from .vocabulary import BOS_TOKEN, EOS_TOKEN, Vocabulary, split_words


class Tokenizer:
    """Encodes text to token ids and decodes ids back to text."""

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def train(cls, texts, min_freq=1):
        """Build a tokenizer by first building its vocabulary.

        ``texts`` is an iterable of strings (the training corpus).
        ``min_freq`` is passed straight to the vocabulary builder.
        """
        return cls(Vocabulary.build(texts, min_freq=min_freq))

    def encode(self, text, add_special=True):
        """Turn a sentence into a list of token ids.

        Example: ``encode("the cat sat")`` ->
        ``[2, 4, 5, 6, 3]`` where 2 is <BOS> and 3 is <EOS>.
        With ``add_special=False`` the <BOS>/<EOS> ids are left out.
        """
        ids = [self.vocab.id_for(word) for word in split_words(text)]
        if add_special:
            ids = [self.vocab.id_for(BOS_TOKEN)] + ids + [self.vocab.id_for(EOS_TOKEN)]
        return ids

    def decode(self, token_ids, skip_special=True):
        """Turn a list of token ids back into a sentence.

        With ``skip_special=True`` (default) the <BOS>/<EOS>/<PAD>
        tokens are left out of the result.
        """
        words = []
        for token_id in token_ids:
            word = self.vocab.word_for(token_id)
            if skip_special and word in (BOS_TOKEN, EOS_TOKEN, "<PAD>"):
                continue
            words.append(word)
        return " ".join(words)

    def tokens(self, text):
        """The word tokens of a sentence, with <BOS>/<EOS> included.

        This is the human-readable version of ``encode()`` — useful
        for printing what the tokenizer actually saw.
        """
        return [BOS_TOKEN] + split_words(text) + [EOS_TOKEN]

    def __len__(self):
        """Vocabulary size (special tokens included)."""
        return len(self.vocab)
