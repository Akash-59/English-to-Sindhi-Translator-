"""
Sindhi Diacritic Restoration inference engine.
Combines Unicode preservation, verified lexical lookup, and a context-aware
machine learning sequence/character classifier to predict short-vowel marks (Zabar, Zer, Pesh).
"""

import os
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import joblib

import config

COMBINING_MARKS = {
    config.ZABAR: "ZABAR",
    config.ZER: "ZER",
    config.PESH: "PESH",
    config.SUKUN: "SUKUN",
    config.SHADDA: "SHADDA",
    config.KHARII_ZABAR: "KHARII_ZABAR",
}


class SindhiDiacriticRestorer:
    """Restores vowel diacritics onto plain Perso-Arabic Sindhi text."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or config.DIACRITIC_MODEL_FILE
        self.pipeline = None
        self.verified_dict: Dict[str, str] = {}
        self._load_resources()

    def _load_resources(self):
        """Loads trained classifier pipeline and verified lexicon."""
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.pipeline = data.get("pipeline")
                self.verified_dict = data.get("lexicon", {})
                print(f"[DiacriticRestorer] Loaded model and {len(self.verified_dict)} lexical entries from {self.model_path}")
            except Exception as e:
                print(f"[DiacriticRestorer] Warning: Could not load trained model: {e}")
        else:
            print(f"[DiacriticRestorer] Note: Trained model file not found at {self.model_path}. Using rule-based fallback.")

    @staticmethod
    def is_combining_mark(ch: str) -> bool:
        """Returns True if character is a Unicode combining mark."""
        return unicodedata.category(ch) in ("Mn", "Me", "Mc")

    @classmethod
    def strip_diacritics(cls, text: str) -> str:
        """Strips short vowel diacritics for normalization while preserving base Sindhi letters."""
        text = unicodedata.normalize("NFD", text)
        filtered = [ch for ch in text if ch not in COMBINING_MARKS and unicodedata.category(ch) != "Mn"]
        return unicodedata.normalize("NFC", "".join(filtered))

    @classmethod
    def extract_char_features(cls, text: str, index: int) -> Dict[str, Any]:
        """Extracts n-gram character context features for position index in text."""
        char = text[index]
        prev_1 = text[index - 1] if index > 0 else "<START>"
        prev_2 = text[index - 2] if index > 1 else "<START>"
        next_1 = text[index + 1] if index < len(text) - 1 else "<END>"
        next_2 = text[index + 2] if index < len(text) - 2 else "<END>"

        is_start_of_word = (index == 0 or text[index - 1].isspace())
        is_end_of_word = (index == len(text) - 1 or text[index + 1].isspace())

        return {
            "char": char,
            "prev_1": prev_1,
            "prev_2": prev_2,
            "next_1": next_1,
            "next_2": next_2,
            "prev_bigram": prev_2 + prev_1,
            "next_bigram": next_1 + next_2,
            "is_start": is_start_of_word,
            "is_end": is_end_of_word,
            "is_space": char.isspace(),
            "category": unicodedata.category(char),
        }

    def restore_word(self, word: str) -> Tuple[str, bool]:
        """Restores a single word using verified lexicon or falls back to classifier."""
        clean_word = word.strip("۔،!؟.,?")
        prefix_punct = word[:len(word) - len(word.lstrip("۔،!؟.,?"))]
        suffix_punct = word[len(word.rstrip("۔،!؟.,?")):]

        stripped = self.strip_diacritics(clean_word)
        if stripped in self.verified_dict:
            return prefix_punct + self.verified_dict[stripped] + suffix_punct, True

        # If not in lexicon and classifier is available
        if self.pipeline is not None:
            restored = self._restore_with_classifier(clean_word)
            return prefix_punct + restored + suffix_punct, False

        return word, False

    def _restore_with_classifier(self, text: str) -> str:
        """Applies character classifier while strictly preserving existing marks."""
        # Deconstruct into base chars and existing marks
        result_chars = []
        i = 0
        n = len(text)

        while i < n:
            curr_ch = text[i]
            result_chars.append(curr_ch)

            # If current character is followed by an existing combining mark, preserve it!
            if i + 1 < n and self.is_combining_mark(text[i + 1]):
                i += 1
                result_chars.append(text[i])
                i += 1
                continue

            # Don't predict marks for spaces, punctuation, or combining marks
            if curr_ch.isspace() or unicodedata.category(curr_ch).startswith("P") or self.is_combining_mark(curr_ch):
                i += 1
                continue

            # Extract features for current character
            features = self.extract_char_features(text, i)
            pred_label = self.pipeline.predict([features])[0]

            if pred_label == "ZABAR":
                result_chars.append(config.ZABAR)
            elif pred_label == "ZER":
                result_chars.append(config.ZER)
            elif pred_label == "PESH":
                result_chars.append(config.PESH)

            i += 1

        return unicodedata.normalize("NFC", "".join(result_chars))

    def restore(self, text: str) -> Dict[str, Any]:
        """
        Restores vowel diacritics across entire Sindhi text.
        Returns restored text, metadata, and per-token verification indicators.
        """
        if not text or not text.strip():
            return {
                "plain": text,
                "marked": text,
                "words_restored": 0,
                "verified_word_count": 0,
                "predicted_word_count": 0,
                "tokens": [],
            }

        # Normalize NFC first
        text = unicodedata.normalize("NFC", text.strip())
        tokens = text.split(" ")
        restored_tokens = []
        token_details = []
        verified_count = 0
        predicted_count = 0

        for tok in tokens:
            if not tok:
                continue
            restored_tok, is_verified = self.restore_word(tok)
            restored_tokens.append(restored_tok)

            if is_verified:
                verified_count += 1
            elif restored_tok != tok:
                predicted_count += 1

            token_details.append({
                "plain": tok,
                "marked": restored_tok,
                "source": "verified_lexicon" if is_verified else ("ml_classifier" if restored_tok != tok else "unmodified"),
                "is_verified": is_verified,
            })

        marked_text = " ".join(restored_tokens)

        return {
            "plain": text,
            "marked": marked_text,
            "words_restored": verified_count + predicted_count,
            "verified_word_count": verified_count,
            "predicted_word_count": predicted_count,
            "tokens": token_details,
        }
