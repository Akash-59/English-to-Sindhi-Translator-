"""
Unit tests for Sindhi Unicode handling, Perso-Arabic alphabet coverage,
and combining vowel marks (Zabar, Zer, Pesh, Sukun, Shadda).
"""

import unicodedata
import pytest
import config
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer


class TestSindhiUnicode:
    """Verifies Unicode specifications for Sindhi (Perso-Arabic script)."""

    # Sindhi unique implosive and aspirated characters
    SINDHI_SPECIAL_CHARS = [
        ("\u067B", "ٻ", "BBE / Implosive B"),
        ("\u0684", "ڄ", "DYE / Implosive J"),
        ("\u0683", "ڃ", "NYA"),
        ("\u068C", "ڌ", "DHAL"),
        ("\u068A", "ڊ", "DDAL"),
        ("\u068D", "ڍ", "DDHAL"),
        ("\u068F", "ڏ", "DDAL implosive"),
        ("\u06A6", "ڦ", "PE aspirated"),
        ("\u067F", "ٿ", "TE aspirated"),
        ("\u067A", "ٺ", "TTEHEH"),
        ("\u067D", "ٽ", "TTE Sindhi retroflex"),
        ("\u06B3", "ڳ", "GGE / Implosive G"),
        ("\u06B1", "ڱ", "NGE"),
        ("\u06BB", "ڻ", "RRE / Retroflex N"),
        ("\u06AA", "ڪ", "SWASH KAF / Sindhi Kaf"),
    ]

    def test_sindhi_alphabet_glyph_integrity(self):
        """Ensures all 15 unique Sindhi characters are recognized and normalized."""
        for code, char, desc in self.SINDHI_SPECIAL_CHARS:
            norm = unicodedata.normalize("NFC", char)
            assert norm == char, f"Normalization mismatch for {desc} ({char})"
            assert ord(norm) == ord(code), f"Codepoint mismatch for {desc}"

    def test_combining_vowel_marks(self):
        """Verifies Zabar, Zer, Pesh, Sukun, and Shadda combining properties."""
        vowels = [
            (config.ZABAR, "Mn", "Zabar (َ)"),
            (config.ZER, "Mn", "Zer (ِ)"),
            (config.PESH, "Mn", "Pesh (ُ)"),
            (config.SUKUN, "Mn", "Sukun (ْ)"),
            (config.SHADDA, "Mn", "Shadda (ّ)"),
        ]
        for mark, category, name in vowels:
            cat = unicodedata.category(mark)
            assert cat == category, f"Expected {category} for {name}, got {cat}"
            assert SindhiDiacriticRestorer.is_combining_mark(mark), f"{name} should be combining"

    def test_nfc_nfd_roundtrip(self):
        """Ensures text with diacritics survives NFC-NFD conversions without glyph corruption."""
        sample_marked = "عِلمُ رَوشنِي آهي۔"
        nfd_form = unicodedata.normalize("NFD", sample_marked)
        nfc_form = unicodedata.normalize("NFC", nfd_form)
        assert nfc_form == sample_marked, "NFC-NFD roundtrip corrupted Sindhi marked string"

    def test_combining_marks_not_stripped_accidentally(self):
        """Verifies that generic text operations do not inadvertently drop diacritics."""
        original = "پُلِ"
        assert config.PESH in original
        assert config.ZER in original
        stripped = SindhiDiacriticRestorer.strip_diacritics(original)
        assert stripped == "پل"
