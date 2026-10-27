"""Tests for text-only Sindhi character and diacritic analysis."""

import config
import pytest

from src.diacritics.restore_diacritics import SindhiDiacriticRestorer
from src.diacritics.word_analysis import VOWEL_GUIDE, analyze_word


@pytest.fixture
def restorer():
    return SindhiDiacriticRestorer()


def test_analysis_preserves_sindhi_unicode_characters(restorer):
    result = analyze_word("ڪتاب", restorer)

    assert result["base_word"] == "ڪتاب"
    assert result["characters"][0]["character"] == "ڪ"
    assert result["restored_word"] == "ڪِتابُ"
    assert result["audio_available"] is False


def test_analysis_identifies_existing_and_restored_vowel_marks(restorer):
    result = analyze_word("ڪِتابُ", restorer)

    first_character = result["characters"][0]
    last_character = result["characters"][-1]

    assert first_character["existing_marks"] == ["Zer"]
    assert last_character["existing_marks"] == ["Pesh"]
    assert first_character["added_marks"] == []
    assert VOWEL_GUIDE[config.ZABAR]["name"] == "Zabar"
    assert "a-like" in VOWEL_GUIDE[config.ZABAR]["effect"]
    assert "i-like" in VOWEL_GUIDE[config.ZER]["effect"]
    assert "u-like" in VOWEL_GUIDE[config.PESH]["effect"]


def test_analysis_reports_alternative_readings_for_ambiguous_word(restorer):
    result = analyze_word("پل", restorer)

    assert result["restored_word"] == "پل"
    assert result["source"] == "ambiguous"
    assert result["uncertain"] is True
    assert result["ambiguous_forms"] == ["پُلِ", "پَلُ"]
    assert len(result["wordbook_entries"]) == 2


@pytest.mark.parametrize("word", ["", "   ", "ڪتاب علم", "book"])
def test_analysis_rejects_empty_or_multiple_words(restorer, word):
    with pytest.raises(ValueError):
        analyze_word(word, restorer)
