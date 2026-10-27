"""
Unit tests for Sindhi diacritic restoration engine,
including verified lexical restoration, existing mark preservation, and boundary handling.
"""

import pytest
import config
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer


@pytest.fixture
def restorer():
    return SindhiDiacriticRestorer()


class TestDiacriticRestoration:
    """Verifies diacritic restoration logic and invariants."""

    def test_verified_word_restoration(self, restorer):
        """Checks restoration of core educational words."""
        res_ilm = restorer.restore("علم")
        assert res_ilm["marked"] == "عِلمُ"
        assert res_ilm["verified_word_count"] >= 1

        res_kitab = restorer.restore("ڪتاب")
        assert res_kitab["marked"] == "ڪِتابُ"

        res_ustad = restorer.restore("استاد")
        assert res_ustad["marked"] == "اُستادُ"

    def test_existing_mark_preservation(self, restorer):
        """Existing vowel marks are not replaced, duplicated, or dropped."""
        input_text = "پُلِ"
        res = restorer.restore(input_text)
        assert res["marked"] == input_text

    def test_existing_mark_conflict_is_reported_without_overwrite(self, restorer):
        restorer.verified_dict["علم"] = "عَلَمُ"
        restorer.verified_candidates["علم"] = ["عَلَمُ"]

        result = restorer.restore("عِلمُ")

        assert result["marked"] == "عِلَمُ"
        assert result["tokens"][0]["preserved_mark_conflicts"]

    def test_previously_marked_input_is_not_flagged_as_unknown(self, restorer):
        result = restorer.restore("ژَ")

        assert result["marked"] == "ژَ"
        assert result["tokens"][0]["uncertain"] is False

    def test_empty_and_whitespace_input(self, restorer):
        """Empty or whitespace input should return cleanly without raising errors."""
        assert restorer.restore("")["marked"] == ""
        assert restorer.restore("   ")["marked"] == "   "

    def test_punctuation_preservation(self, restorer):
        """Preserves Sindhi full stop (۔) and Arabic comma (،)."""
        input_text = "علم روشني آهي۔"
        res = restorer.restore(input_text)
        assert res["marked"].endswith("۔")
        assert restorer.restore("۔۔۔")["marked"] == "۔۔۔"

    def test_minimal_pair_restoration(self, restorer):
        """A bare spelling with multiple verified forms is not guessed automatically."""
        restorer.verified_candidates["پل"] = ["پُلِ", "پَلُ"]

        result = restorer.restore("پل")

        assert result["marked"] == "پل"
        assert result["tokens"][0]["ambiguous"] is True
        assert result["tokens"][0]["ambiguous_candidates"] == ["پُلِ", "پَلُ"]
        assert result["tokens"][0]["changes"] == []

        disambiguated = restorer.restore("پُل")
        assert disambiguated["marked"] == "پُلِ"
        assert disambiguated["tokens"][0]["ambiguous"] is False

    def test_classifier_probability_is_reported_as_a_score(self, restorer):
        class ProbabilityPipeline:
            classes_ = ["NONE", "PESH", "ZABAR", "ZER"]

            def predict_proba(self, features):
                return [[0.08, 0.07, 0.72, 0.13] for _ in features]

        restorer.pipeline = ProbabilityPipeline()
        restorer.verified_dict = {}
        restorer.verified_candidates = {}

        result = restorer.restore("ژ")
        change = result["tokens"][0]["changes"][0]

        assert result["marked"] == "ژَ"
        assert change["confidence"] == pytest.approx(0.72)
        assert change["confidence_label"] == "moderate"
        assert change["source"] == "ml_classifier"

    def test_users_can_accept_or_reject_individual_suggestions(self, restorer):
        class ProbabilityPipeline:
            classes_ = ["NONE", "PESH", "ZABAR", "ZER"]

            def predict_proba(self, features):
                return [[0.05, 0.05, 0.85, 0.05] for _ in features]

        restorer.pipeline = ProbabilityPipeline()
        restorer.verified_dict = {}
        restorer.verified_candidates = {}
        result = restorer.restore("ژز")
        first_change_id = result["tokens"][0]["changes"][0]["id"]

        accepted_one = restorer.apply_suggestion_decisions(result, [first_change_id])
        rejected_all = restorer.apply_suggestion_decisions(result, [])

        assert accepted_one == "ژَز"
        assert rejected_all == "ژز"
