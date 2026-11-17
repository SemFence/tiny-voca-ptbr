import unittest
from unittest.mock import patch

import dizione_ptbr

from tiny_voca import text
from tiny_voca.config import Config
from tiny_voca.text import (BOS, EOS, PHONE_SET, PHONES, SP, SYMBOLS, SYMBOL_TO_ID, UNK, canonical, frontend_fingerprint,
                            phonemize, tokenize)

BOLA = [("b", 0), ("ɔ", 1), ("l", 0), ("ɐ", 0)]
CASA = [("k", 0), ("a", 1), ("z", 0), ("ɐ", 0)]


class FrontEndTests(unittest.TestCase):
    def test_inventory_is_consistent_with_the_model_vocabulary(self):
        self.assertEqual(len(SYMBOLS), len(set(SYMBOLS)))
        self.assertEqual(Config().vocab_size, len(SYMBOLS))
        self.assertEqual(SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual(len(PHONES), 38)
        self.assertEqual(len(SYMBOLS), 55)
        with self.assertRaises(ValueError):
            Config(vocab_size=259)     # the byte vocabulary is gone

    def test_words_become_phones_with_stress_features(self):
        result = tokenize([(True, '"'), (False, [BOLA, CASA]), (True, ","), (True, ","), (False, [CASA]), (True, "?")])
        self.assertEqual(result.symbols, (BOS, '"', "b", "ɔ", "l", "ɐ", SP, "k", "a", "z", "ɐ", ",", "k", "a", "z", "ɐ", "?", EOS))
        self.assertEqual(result.stress, (0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0))   # repeated "," collapses
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertTrue(all(s in PHONE_SET or s in SYMBOLS for s in result.symbols))
        # Stress is never a token, so nothing but phones, pauses and punctuation needs frames.
        self.assertNotIn("ˈ", "".join(result.symbols))

    def test_phones_outside_the_inventory_are_reported(self):
        self.assertEqual(canonical("tʃ"), ("tʃ",))
        self.assertIsNone(canonical("ʘ"))
        result = tokenize([(False, [[("ʘ", 0), ("a", 1)]])])
        self.assertEqual(result.symbols, (BOS, UNK, "a", EOS))
        self.assertEqual(result.stress, (0, 0, 1, 0))
        self.assertEqual(result.unknown, ("ʘ",))
        with self.assertRaises(ValueError):
            tokenize([(True, ".")])

    def test_fingerprint_tracks_inventory_and_version(self):
        baseline = frontend_fingerprint()
        with patch.object(dizione_ptbr, "VERSION", 999):
            self.assertNotEqual(frontend_fingerprint(), baseline)
        with patch.object(text, "SYMBOLS", SYMBOLS + ("new",)):
            with self.assertRaises(RuntimeError):      # the model and the phonemizer must agree on the inventory
                frontend_fingerprint()

    def test_the_phonemizer_gives_inventory_phones_and_keeps_punctuation(self):
        result = phonemize("Oi, tudo bem?")
        self.assertEqual(result.symbols[:5], (BOS, "o", "j", ",", "t"))   # no <sp> after punctuation
        self.assertEqual(result.stress[1], 1)
        self.assertEqual([s for s in result.symbols if s in ",?"], [",", "?"])
        self.assertEqual(result.symbols[-1], EOS)
        self.assertEqual(result.unknown, ())
        self.assertNotEqual(phonemize("Tudo bem.").ids, phonemize("Tudo bem?").ids)
        self.assertEqual(phonemize("Tudo bem.").ids, phonemize(" Tudo  bem. ").ids)

    def test_numbers_money_and_times_are_spoken(self):
        result = phonemize("Paguei R$ 15,50 às 10h30, dia 3 de maio de 2025.")
        self.assertEqual(result.unknown, ())
        self.assertGreater(len(set(result.symbols) & PHONE_SET), 20)
        self.assertEqual(sum(result.stress[i] > 0 and s not in PHONE_SET for i, s in enumerate(result.symbols)), 0)

    def test_empty_or_unreadable_text_is_refused(self):
        for bad in ("", "   ", "...", "😀"):
            with self.assertRaises(ValueError):
                phonemize(bad)


if __name__ == "__main__":
    unittest.main()
