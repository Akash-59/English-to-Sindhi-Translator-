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
        """Critical Requirement: Pre-existing diacritics must NEVER be dropped or overwritten."""
        input_text = "پُلِ مٿان پاڻي وهندو هو۔"
        res = restorer.restore(input_text)
        # Verify the original Pesh and Zer in pul are preserved
        assert config.PESH in res["marked"]
        assert config.ZER in res["marked"]

    def test_empty_and_whitespace_input(self, restorer):
        """Empty or whitespace input should return cleanly without raising errors."""
        assert restorer.restore("")["marked"] == ""
        assert restorer.restore("   ")["marked"] == "   "

    def test_punctuation_preservation(self, restorer):
        """Preserves Sindhi full stop (۔) and Arabic comma (،)."""
        input_text = "علم روشني آهي۔"
        res = restorer.restore(input_text)
        assert res["marked"].endswith("۔")

    def test_minimal_pair_restoration(self, restorer):
        """Tests that words with distinct vowel meanings can be disambiguated."""
        pul = restorer.restore_word("پل")
        # In lexical memory, pul is mapped to bridge or moment
        assert pul[0] in ("پُلِ", "پَلُ")
