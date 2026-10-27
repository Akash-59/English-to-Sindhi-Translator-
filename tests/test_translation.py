"""
Unit tests for translation preprocessing, segmentation, and model parameter configuration.
"""

import pytest
import torch
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


class TestTranslationCandidates:
    """Verifies multi-candidate generation and error handling without model downloads."""

    @staticmethod
    def attach_generation_stubs(translator, decoded, error=None):
        class StubTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1, 2]])}

            def convert_tokens_to_ids(self, token):
                return 42

            def batch_decode(self, sequences, skip_special_tokens=True):
                return decoded

        class StubModel:
            def __init__(self):
                self.generate_kwargs = None

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs
                if error:
                    raise error
                return torch.tensor([[1, 2] for _ in decoded])

        translator.tokenizer = StubTokenizer()
        translator.model = StubModel()
        return translator.model

    def test_candidate_generation_filters_duplicates_and_uses_beam_search(self):
        translator = SindhiTranslator(lazy_init=True)
        model = self.attach_generation_stubs(
            translator,
            ["سلام.", "سلام۔", "توهان ڪيئن آهيو؟"],
        )

        result = translator.translate_candidates("Hello.", num_candidates=3, num_beams=4)

        assert result["candidates"] == ["سلام.", "توهان ڪيئن آهيو؟"]
        assert result["candidate_count"] == 2
        assert model.generate_kwargs["num_beams"] == 4
        assert model.generate_kwargs["num_return_sequences"] == 3
        assert model.generate_kwargs["forced_bos_token_id"] == 42

    def test_candidate_results_are_cached_separately_from_single_output(self):
        translator = SindhiTranslator(lazy_init=True)
        self.attach_generation_stubs(translator, ["سلام", "توهان ڪيئن آهيو؟"])

        first = translator.translate_candidates("Hello")
        first["candidates"].clear()
        cached = translator.translate_candidates("Hello")

        assert cached["cache_hit"] is True
        assert cached["candidates"] == ["سلام", "توهان ڪيئن آهيو؟"]
        assert translator.get_cache_stats()["hits"] == 1

    def test_candidate_generation_surfaces_model_errors(self):
        translator = SindhiTranslator(lazy_init=True)
        self.attach_generation_stubs(
            translator,
            [],
            error=RuntimeError("generation failed"),
        )

        with pytest.raises(RuntimeError, match="generation failed"):
            translator.translate_candidates("Hello")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_candidates": 0},
            {"num_candidates": 5},
            {"num_candidates": 3, "num_beams": 2},
            {"max_length": 0},
        ],
    )
    def test_candidate_generation_rejects_invalid_parameters(self, kwargs):
        translator = SindhiTranslator(lazy_init=True)

        with pytest.raises(ValueError):
            translator.translate_candidates("Hello", **kwargs)


class TestBidirectionalTranslation:
    """Verifies language direction and failures using the existing model interface."""

    class StubTokenizer:
        def __init__(self):
            self.src_lang = None
            self.source_lang_at_tokenization = None
            self.target_language_ids = {
                config.SRC_LANG_CODE: 17,
                config.TGT_LANG_CODE: 29,
            }

        def __call__(self, text, **kwargs):
            self.source_lang_at_tokenization = self.src_lang
            return {"input_ids": torch.tensor([[1, 2]])}

        def convert_tokens_to_ids(self, token):
            return self.target_language_ids[token]

        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["translated text"]

    class StubModel:
        def __init__(self, error=None):
            self.generate_kwargs = None
            self.error = error

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            if self.error:
                raise self.error
            return torch.tensor([[1, 2]])

    @classmethod
    def attach_generation_stubs(cls, translator, error=None):
        tokenizer = cls.StubTokenizer()
        model = cls.StubModel(error=error)
        translator.tokenizer = tokenizer
        translator.model = model
        return tokenizer, model

    def test_english_to_sindhi_uses_english_source_and_sindhi_target(self):
        translator = SindhiTranslator(lazy_init=True)
        tokenizer, model = self.attach_generation_stubs(translator)

        result = translator.translate_segment("Hello.")

        assert result["english"] == "Hello."
        assert result["sindhi"] == "translated text۔"
        assert tokenizer.source_lang_at_tokenization == config.SRC_LANG_CODE
        assert model.generate_kwargs["forced_bos_token_id"] == 29

    def test_sindhi_to_english_uses_sindhi_source_and_english_target(self):
        translator = SindhiTranslator(lazy_init=True)
        tokenizer, model = self.attach_generation_stubs(translator)

        result = translator.translate_sindhi_to_english("مان ٺيڪ آهيان۔")

        assert result["sindhi"] == "مان ٺيڪ آهيان۔"
        assert result["english"] == "translated text"
        assert tokenizer.source_lang_at_tokenization == config.TGT_LANG_CODE
        assert model.generate_kwargs["forced_bos_token_id"] == 17

    def test_translation_directions_use_separate_cache_entries(self):
        translator = SindhiTranslator(lazy_init=True)
        self.attach_generation_stubs(translator)

        forward = translator.translate_segment("سنڌي")
        reverse = translator.translate_sindhi_segment("سنڌي")

        assert forward["cache_hit"] is False
        assert reverse["cache_hit"] is False
        assert reverse["english"] == "translated text"

    def test_sindhi_to_english_surfaces_model_errors(self):
        translator = SindhiTranslator(lazy_init=True)
        self.attach_generation_stubs(
            translator,
            error=RuntimeError("generation failed"),
        )

        with pytest.raises(RuntimeError, match="generation failed"):
            translator.translate_sindhi_to_english("سنڌي متن")
