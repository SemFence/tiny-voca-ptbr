"""Front end: text -> phoneme tokens with a stress feature (Brazilian Portuguese).

The same code runs during annotation, training-set preparation and inference,
so tokens can never drift between them. The phonemes come from the separate
``dizione-ptbr`` package (dictionary plus rules, no GPL), which also owns the
symbol inventory: 55 symbols, append-only.

* A token is one phone of that inventory, for example ``t``, ``tʃ`` or ``ẽ``.
* Lexical stress is a parallel feature (0 none, 1 primary), not a token, so it
  never has to be given audio frames.
* Punctuation, word boundaries and the utterance edges are explicit tokens that
  may last zero frames. They carry pauses when the recording has them.
"""

from dataclasses import dataclass
import hashlib
import json
import os

FRONTENDS = ("own",)   # dizione-ptbr is the only front end
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "own")
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
# The single source of truth for the inventory is dizione-ptbr (append-only symbol table).
import dizione_ptbr.symbols as _pt
assert tuple(_pt.PUNCTUATION) == PUNCTUATION and tuple(_pt.SPECIALS) == SPECIALS
PHONES = tuple(_pt.PHONES)
VOWELS = tuple(p for p in PHONES if p in _pt.VOWEL_PHONES)
CONSONANTS = tuple(p for p in PHONES if p not in _pt.VOWEL_PHONES)   # includes the glides j w j̃ w̃
# Append-only: changing the order or removing a symbol invalidates checkpoints.
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOICELESS = frozenset(("p", "t", "k", "f", "s", "ʃ", "tʃ"))   # "x" (the Brazilian r) is often voiced: left out
VOWEL_SET = frozenset(VOWELS)


class NothingToPronounce(ValueError):
    """The text has no word, number or letter to say: only marks, emoji and the like."""


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple  # phones that had no place in the inventory (they become <unk>)

    def __len__(self):
        return len(self.ids)


def frontend():
    """Name of the program that produces the phonemes (always ``"own"``: dizione-ptbr)."""
    return _frontend


def set_frontend(name):
    """Select the phonemizer for this process. Annotations and checkpoints record which one made their tokens."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def _dizione():
    try:
        import dizione_ptbr as dizione
    except ImportError as error:
        raise RuntimeError('the front end needs dizione-ptbr: pip install git+https://github.com/SemFence/dizione-ptbr') from error
    if tuple(dizione.symbols.SYMBOLS) != SYMBOLS:
        raise RuntimeError("dizione-ptbr's symbol inventory differs from this model's")
    return dizione


def frontend_identity():
    phonemizer = _dizione()
    return {"frontend_version": phonemizer.VERSION, "phonemizer": "dizione-ptbr", "symbols": list(SYMBOLS)}


def frontend_runtime():
    """What an annotation run records about the exact program behind its
    tokens: the fingerprint of the phonemizer build."""
    return {"phonemizer": _dizione().identity()["fingerprint"]}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def canonical(phone):
    """The inventory phone as a tuple, or ``None`` when the phone is not in the inventory."""
    return (phone,) if phone in PHONE_SET else None


def tokenize(pieces):
    """Build the token sequence from ``(is_punctuation, value)`` pieces where a
    text piece's value is a list of words, each a list of ``(phone, stress)``.
    Pure function (testable without the phonemizer)."""
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:  # collapse repeats such as "!!"
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            for phone, level in word:
                mapped = canonical(phone)
                if mapped is None:
                    unknown.append(phone)
                    mapped = (UNK,)
                symbols.append(mapped[0])
                stress.append(level)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    return tokenize(_dizione().pieces(text))


def is_phone(token_ids):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    return [int(i) in PHONE_IDS for i in token_ids]
