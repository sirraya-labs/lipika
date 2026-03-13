"""
vani/tokenizer/script_adapter.py

Script-family adapter: Unicode-block-aware phonetic conditioning.

This is Vani's unique research contribution. All 22 Indian scheduled languages
share prosodic patterns distinct from Indo-European or East Asian languages:
  - Retroflex consonants (ट, ड, ण, etc.)
  - Pitch accent vs stress accent distinctions
  - Mora-timed vs stress-timed rhythm

A single script-family embedding, injected into the RVQ encoder and both AR models,
conditions the system on these phonetic properties WITHOUT needing separate models
per language. Cross-lingual prosody transfer becomes natural: the model learns that
Devanagari retroflex patterns share acoustic properties with Telugu retroflex ones.

Unicode block ranges used:
  DEVANAGARI:   U+0900–U+097F  (Hindi, Marathi, Sanskrit, Nepali, etc.)
  BENGALI:      U+0980–U+09FF
  GURMUKHI:     U+0A00–U+0A7F  (Punjabi)
  GUJARATI:     U+0A80–U+0AFF
  ORIYA:        U+0B00–U+0B7F
  TAMIL:        U+0B80–U+0BFF
  TELUGU:       U+0C00–U+0C7F
  KANNADA:      U+0C80–U+0CFF
  MALAYALAM:    U+0D00–U+0D7F
  SINHALA:      U+0D80–U+0DFF
  SAURASHTRA:   (Santali, Dogri approximation)
  LATIN_INDIA:  Latin script (English, Hinglish code-switching)
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import unicodedata


class ScriptFamily(IntEnum):
    DEVANAGARI  = 0   # Hindi, Marathi, Sanskrit, Nepali, Maithili, Dogri, Konkani
    BENGALI     = 1   # Bengali, Assamese
    GURMUKHI    = 2   # Punjabi
    GUJARATI    = 3
    ORIYA       = 4
    TAMIL       = 5
    TELUGU      = 6
    KANNADA     = 7
    MALAYALAM   = 8
    PERSO_ARABIC = 9  # Urdu, Kashmiri, Sindhi
    MEITEI      = 10  # Manipuri (Meitei Mayek)
    LATIN_INDIA = 11  # English, Hinglish, code-switching


# Unicode codepoint ranges for script detection
SCRIPT_RANGES = [
    (0x0900, 0x097F, ScriptFamily.DEVANAGARI),
    (0x0980, 0x09FF, ScriptFamily.BENGALI),
    (0x0A00, 0x0A7F, ScriptFamily.GURMUKHI),
    (0x0A80, 0x0AFF, ScriptFamily.GUJARATI),
    (0x0B00, 0x0B7F, ScriptFamily.ORIYA),
    (0x0B80, 0x0BFF, ScriptFamily.TAMIL),
    (0x0C00, 0x0C7F, ScriptFamily.TELUGU),
    (0x0C80, 0x0CFF, ScriptFamily.KANNADA),
    (0x0D00, 0x0D7F, ScriptFamily.MALAYALAM),
    (0x0600, 0x06FF, ScriptFamily.PERSO_ARABIC),  # Arabic block (Urdu/Kashmiri/Sindhi)
    (0xABC0, 0xABFF, ScriptFamily.MEITEI),
    (0x0041, 0x007A, ScriptFamily.LATIN_INDIA),   # Basic Latin A-z
]

# Language → primary script family mapping
LANGUAGE_TO_SCRIPT = {
    "hi":  ScriptFamily.DEVANAGARI,
    "mr":  ScriptFamily.DEVANAGARI,
    "sa":  ScriptFamily.DEVANAGARI,
    "ne":  ScriptFamily.DEVANAGARI,
    "mai": ScriptFamily.DEVANAGARI,
    "kok": ScriptFamily.DEVANAGARI,
    "doi": ScriptFamily.DEVANAGARI,
    "bn":  ScriptFamily.BENGALI,
    "as":  ScriptFamily.BENGALI,
    "pa":  ScriptFamily.GURMUKHI,
    "gu":  ScriptFamily.GUJARATI,
    "or":  ScriptFamily.ORIYA,
    "ta":  ScriptFamily.TAMIL,
    "te":  ScriptFamily.TELUGU,
    "kn":  ScriptFamily.KANNADA,
    "ml":  ScriptFamily.MALAYALAM,
    "ur":  ScriptFamily.PERSO_ARABIC,
    "ks":  ScriptFamily.PERSO_ARABIC,
    "sd":  ScriptFamily.PERSO_ARABIC,
    "mni": ScriptFamily.MEITEI,
    "sat": ScriptFamily.LATIN_INDIA,    # Santali (Ol Chiki not in our range, fallback)
    "en":  ScriptFamily.LATIN_INDIA,
    "en-IN": ScriptFamily.LATIN_INDIA,
}


def detect_script_family(text: str) -> ScriptFamily:
    """
    Detect dominant script family from text by codepoint majority vote.
    Handles code-switching (Hinglish) gracefully.
    """
    counts = {family: 0 for family in ScriptFamily}
    for char in text:
        cp = ord(char)
        for lo, hi, family in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[family] += 1
                break
    dominant = max(counts, key=lambda f: counts[f])
    return dominant if counts[dominant] > 0 else ScriptFamily.LATIN_INDIA


@dataclass
class ScriptAdapterConfig:
    n_script_families: int = 12
    embed_dim: int = 64
    model_dim: int = 512    # matches encoder/AR hidden size

    # Phonetic feature flags per script family
    # These are injected as bias offsets in the adapter
    has_retroflex: dict = None         # languages with retroflex consonants
    is_mora_timed: dict = None         # mora-timing vs stress-timing


class ScriptFamilyAdapter(nn.Module):
    """
    Learnable script-family embeddings injected into:
      1. RVQ encoder (at bottleneck)
      2. Slow AR (at every layer via AdaLN or prefix)
      3. Fast AR (prefix only)

    Enables cross-lingual prosody transfer: the model learns shared
    acoustic properties across related scripts (e.g., all Indic retroflex
    consonants get similar codebook assignments regardless of script).
    """
    def __init__(self, cfg: ScriptAdapterConfig):
        super().__init__()
        self.cfg = cfg

        # Core family embeddings
        self.family_embed = nn.Embedding(cfg.n_script_families, cfg.embed_dim)

        # Phonetic feature embeddings (retroflex, tonal, aspirated, etc.)
        # These are fixed based on linguistic knowledge, not learned
        self.register_buffer(
            "retroflex_bias",
            self._build_retroflex_bias(cfg.n_script_families, cfg.embed_dim)
        )

        # Project to model dimension
        self.proj = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.model_dim),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )

        # AdaLN (Adaptive Layer Norm) scale/shift for AR model injection
        self.adaln_scale = nn.Linear(cfg.model_dim, cfg.model_dim)
        self.adaln_shift = nn.Linear(cfg.model_dim, cfg.model_dim)
        nn.init.zeros_(self.adaln_scale.weight)
        nn.init.zeros_(self.adaln_shift.weight)
        nn.init.ones_(self.adaln_scale.bias)
        nn.init.zeros_(self.adaln_shift.bias)

    def _build_retroflex_bias(self, n_families: int, dim: int) -> Tensor:
        """
        Manually encoded phonetic priors. Families with retroflex consonants
        (Devanagari, Tamil, Telugu, Kannada, Malayalam, Oriya, Bengali, Gurmukhi)
        get a nonzero bias in the first 8 dimensions of the embedding space.
        This gives the model a head start on phonetic structure.
        """
        bias = torch.zeros(n_families, dim)
        retroflex_families = [
            ScriptFamily.DEVANAGARI, ScriptFamily.TAMIL, ScriptFamily.TELUGU,
            ScriptFamily.KANNADA, ScriptFamily.MALAYALAM, ScriptFamily.ORIYA,
            ScriptFamily.BENGALI, ScriptFamily.GURMUKHI,
        ]
        for f in retroflex_families:
            bias[f, :8] = 0.5   # warm-start retroflex dimension
        return bias

    def forward(self, script_ids: Tensor) -> dict:
        """
        script_ids: (B,) integer ScriptFamily values
        Returns dict with:
            embed: (B, model_dim) for prefix injection
            scale: (B, model_dim) for AdaLN
            shift: (B, model_dim) for AdaLN
        """
        raw = self.family_embed(script_ids) + self.retroflex_bias[script_ids]
        projected = self.proj(raw)                      # (B, model_dim)
        return {
            "embed": projected,
            "scale": self.adaln_scale(projected),
            "shift": self.adaln_shift(projected),
        }

    def get_script_id(self, language_code: str) -> int:
        family = LANGUAGE_TO_SCRIPT.get(language_code, ScriptFamily.LATIN_INDIA)
        return int(family)

    def get_script_id_from_text(self, text: str) -> int:
        return int(detect_script_family(text))
