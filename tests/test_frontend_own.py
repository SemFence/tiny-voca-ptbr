"""The front end behind ``tiny_voca.text``: dizione-ptbr, the only one."""

import unittest

import dizione_ptbr

from tiny_voca import text
from tiny_voca.text import SYMBOLS, SYMBOL_TO_ID

# Fingerprint of the pt-BR front end (dizione-ptbr, version 1): the tokens of every annotation row of the
# released model, and the checkpoint records it. Changing the phonemizer or the inventory changes it.
PTBR_FINGERPRINT = "c8b278571f76e11e47cf66612f6d0470ca3a1b892c85d2091a4cf054c3367862"


class FrontEndTests(unittest.TestCase):
    def test_dizione_ptbr_is_the_only_front_end_and_keeps_its_fingerprint(self):
        self.assertEqual(text.frontend(), "own")
        self.assertEqual(text.frontend_fingerprint(), PTBR_FINGERPRINT)
        identity = text.frontend_identity()
        self.assertEqual((identity["frontend_version"], identity["phonemizer"], identity["symbols"]),
                         (dizione_ptbr.VERSION, "dizione-ptbr", list(SYMBOLS)))
        self.assertEqual(tuple(dizione_ptbr.symbols.SYMBOLS), SYMBOLS)
        with self.assertRaises(ValueError):
            text.set_frontend("espeak")

    def test_phonemize_uses_the_pieces_of_dizione_ptbr(self):
        sentence = "Custa R$ 3,50, tá bom?"
        result = text.phonemize(sentence)
        self.assertEqual(result, text.tokenize(dizione_ptbr.pieces(sentence)))
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertEqual(result.unknown, ())
        self.assertEqual(text.frontend_runtime(), {"phonemizer": dizione_ptbr.identity()["fingerprint"]})


if __name__ == "__main__":
    unittest.main()
