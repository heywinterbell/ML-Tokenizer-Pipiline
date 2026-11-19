"""Tests for the tokenizer pipeline.

Run them with:  python -m unittest discover -s tests -v

Each test checks one small promise the code makes, in plain words.
If a test fails, its name tells you which promise broke.
"""

import os
import sys
import tempfile
import unittest

# Make "src" importable when the tests run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import Tokenizer
from src.gui import describe_result
from src.vocabulary import Vocabulary
from src.visualize import plot_top_tokens

CORPUS = [
    "The cat sat on the mat",
    "the cat slept",
    "a dog barked",
]


class TestVocabulary(unittest.TestCase):
    def test_special_tokens_have_fixed_ids(self):
        """<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3 — always, in this order."""
        vocab = Vocabulary.build(CORPUS)
        self.assertEqual(vocab.word_to_id["<PAD>"], 0)
        self.assertEqual(vocab.word_to_id["<UNK>"], 1)
        self.assertEqual(vocab.word_to_id["<BOS>"], 2)
        self.assertEqual(vocab.word_to_id["<EOS>"], 3)

    def test_most_frequent_word_gets_smallest_real_id(self):
        """'the' appears 3 times, more than any other word, so it
        gets id 4 — the first id after the special tokens."""
        vocab = Vocabulary.build(CORPUS)
        self.assertEqual(vocab.id_for("the"), 4)

    def test_min_freq_drops_rare_words(self):
        """With min_freq=2, words seen only once get no id and
        later become <UNK>."""
        vocab = Vocabulary.build(CORPUS, min_freq=2)
        self.assertEqual(vocab.id_for("sat"), 1)  # "sat" seen once -> UNK
        self.assertEqual(vocab.id_for("the"), 4)  # "the" seen 3x -> kept

    def test_unknown_word_maps_to_unk(self):
        """A word the vocabulary never saw returns the <UNK> id (1)."""
        vocab = Vocabulary.build(CORPUS)
        self.assertEqual(vocab.id_for("zebra"), 1)

    def test_words_are_lowercased(self):
        """'The' and 'the' are the same word — case does not matter."""
        vocab = Vocabulary.build(["The THE the"])
        self.assertEqual(len(vocab), 5)  # 4 special + "the"
        self.assertEqual(vocab.counts["the"], 3)


class TestTokenizer(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        """Encoding then decoding gives back the original sentence."""
        tokenizer = Tokenizer.train(CORPUS)
        ids = tokenizer.encode("the cat sat")
        self.assertEqual(tokenizer.decode(ids), "the cat sat")

    def test_encode_adds_bos_and_eos(self):
        """Encoded ids start with <BOS> (2) and end with <EOS> (3)."""
        tokenizer = Tokenizer.train(CORPUS)
        ids = tokenizer.encode("the cat")
        self.assertEqual(ids[0], 2)
        self.assertEqual(ids[-1], 3)

    def test_unknown_words_become_unk(self):
        """Unknown words turn into 1s instead of crashing."""
        tokenizer = Tokenizer.train(CORPUS)
        self.assertEqual(
            tokenizer.encode("the zebra ran", add_special=False), [4, 1, 1]
        )

    def test_tokens_show_what_was_seen(self):
        """tokens() returns the readable words, with <BOS>/<EOS>."""
        tokenizer = Tokenizer.train(CORPUS)
        self.assertEqual(
            tokenizer.tokens("the cat"),
            ["<BOS>", "the", "cat", "<EOS>"],
        )


class TestVisualize(unittest.TestCase):
    def test_plot_saves_a_non_empty_png(self):
        """plot_top_tokens writes a real PNG file to the given path."""
        vocab = Vocabulary.build(CORPUS)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "freq.png")
            saved = plot_top_tokens(vocab, n=5, output_path=path)
            self.assertEqual(saved, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)


class TestGuiHelper(unittest.TestCase):
    def test_describe_known_sentence(self):
        """A sentence of known words: ids match encode(), nothing
        is unknown, and decoding gives the sentence back."""
        tokenizer = Tokenizer.train(CORPUS)
        result = describe_result(tokenizer, "the cat sat")
        self.assertEqual(result["tokens"], ["<BOS>", "the", "cat", "sat", "<EOS>"])
        self.assertEqual(result["ids"], tokenizer.encode("the cat sat"))
        self.assertEqual(result["decoded"], "the cat sat")
        self.assertEqual(result["unknown"], [])

    def test_describe_lists_unknown_words(self):
        """Unknown words are listed (sorted) so the GUI can show
        them in red."""
        tokenizer = Tokenizer.train(CORPUS)
        result = describe_result(tokenizer, "the zebra danced under the moon")
        self.assertEqual(
            result["unknown"], ["danced", "moon", "under", "zebra"]
        )
        self.assertEqual(result["decoded"], "the <UNK> <UNK> <UNK> the <UNK>")

    def test_describe_empty_text(self):
        """An empty box still gives valid ids: just <BOS> and <EOS>."""
        tokenizer = Tokenizer.train(CORPUS)
        result = describe_result(tokenizer, "")
        self.assertEqual(result["ids"], [2, 3])
        self.assertEqual(result["unknown"], [])


if __name__ == "__main__":
    unittest.main()
