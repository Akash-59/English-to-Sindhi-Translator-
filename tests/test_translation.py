"""
Unit tests for translation preprocessing, segmentation, and model parameter configuration.
"""

import pytest
import config
from src.translation.translator import SindhiTranslator
from src.data.validate_dataset import DatasetValidator


@pytest.fixture
def translator():
    return SindhiTranslator(lazy_init=True)


class TestTranslationPreprocessing:
    """Tests English input normalization and segmentation logic."""

    def test_clean_english_input(self, translator):
        raw = "  Hello,   world!  \t\n  "
        cleaned = translator.clean_english_input(raw)
        assert cleaned == "Hello, world!"

    def test_split_into_segments_short(self, translator):
        text = "Hello world. How are you?"
        segments = translator.split_into_segments(text)
        assert len(segments) == 1
        assert segments[0] == text

    def test_split_into_segments_paragraphs(self, translator):
        text = "First paragraph here.\n\nSecond paragraph here."
        segments = translator.split_into_segments(text)
        assert len(segments) == 2
        assert segments[0] == "First paragraph here."
        assert segments[1] == "Second paragraph here."

    def test_empty_input(self, translator):
        res = translator.translate("")
        assert res["sindhi"] == ""
        assert res["latency_ms"] == 0.0

    def test_language_codes_configured(self):
        assert config.SRC_LANG_CODE == "eng_Latn"
        assert config.TGT_LANG_CODE == "snd_Arab"

    def test_sindhi_script_detection(self):
        assert DatasetValidator.contains_arabic_script("سنڌي")
        assert not DatasetValidator.contains_arabic_script("Sindhi")
        assert DatasetValidator.contains_latin_script("Sindhi")
        assert not DatasetValidator.contains_latin_script("سنڌي")
