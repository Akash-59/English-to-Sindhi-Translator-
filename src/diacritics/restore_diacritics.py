"""
Sindhi Diacritic Restoration inference engine.
Combines Unicode preservation, verified lexical lookup, and a context-aware
machine learning sequence/character classifier to predict short-vowel marks (Zabar, Zer, Pesh).
"""

import csv
import os
import re
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

    UNCERTAIN_PROBABILITY = 0.65
    HIGH_PROBABILITY = 0.85

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or config.DIACRITIC_MODEL_FILE
        self.pipeline = None
        self.verified_dict: Dict[str, str] = {}
        self.verified_candidates: Dict[str, List[str]] = {}
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

        if config.SAMPLE_DIACRITICS_FILE.exists():
            with config.SAMPLE_DIACRITICS_FILE.open(encoding="utf-8", newline="") as data_file:
                for row in csv.DictReader(data_file):
                    if row.get("review_status") != "native_reviewed":
                        continue
                    plain = self.strip_diacritics(row.get("plain", "").strip())
                    marked = unicodedata.normalize("NFC", row.get("marked", "").strip())
                    if plain and marked:
                        candidates = self.verified_candidates.setdefault(plain, [])
                        if marked not in candidates:
                            candidates.append(marked)

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

    @classmethod
    def _graphemes(cls, text: str) -> List[Dict[str, Any]]:
        """Splits a string into base characters and their attached combining marks."""
        graphemes: List[Dict[str, Any]] = []
        for char in unicodedata.normalize("NFD", text):
            if cls.is_combining_mark(char) and graphemes:
                graphemes[-1]["marks"].append(char)
            else:
                graphemes.append({"base": char, "marks": []})
        return graphemes

    @staticmethod
    def _label_mark(label: str) -> Optional[str]:
        return {
            "ZABAR": config.ZABAR,
            "ZER": config.ZER,
            "PESH": config.PESH,
        }.get(str(label))

    def _restore_word_details(self, word: str, token_index: int = 0) -> Dict[str, Any]:
        punctuation = "۔،!؟.,?"
        prefix_end = len(word) - len(word.lstrip(punctuation))
        suffix_start = max(prefix_end, len(word.rstrip(punctuation)))
        prefix_punct = word[:prefix_end]
        clean_word = word[prefix_end:suffix_start]
        suffix_punct = word[suffix_start:]
        stripped = self.strip_diacritics(clean_word)
        original_graphemes = self._graphemes(clean_word)
        candidates = self.verified_candidates.get(stripped, [])
        chosen_lexicon_form = self.verified_dict.get(stripped)
        ambiguous_candidates = candidates
        has_speech_characters = any(
            not grapheme["base"].isspace()
            and not unicodedata.category(grapheme["base"]).startswith("P")
            and not self.is_combining_mark(grapheme["base"])
            for grapheme in original_graphemes
        )
        existing_marks = [
            next((mark for mark in grapheme["marks"] if mark in config.TARGET_DIACRITICS), None)
            for grapheme in original_graphemes
        ]
        has_existing_vowels = any(existing_marks)

        if len(candidates) > 1:
            compatible_candidates = [
                candidate
                for candidate in candidates
                if all(
                    not existing_mark
                    or (
                        index < len(self._graphemes(candidate))
                        and existing_mark in self._graphemes(candidate)[index]["marks"]
                    )
                    for index, existing_mark in enumerate(existing_marks)
                )
            ]
            if has_existing_vowels and len(compatible_candidates) == 1:
                chosen_lexicon_form = compatible_candidates[0]
                ambiguous_candidates = []
            else:
                segments = [
                    {
                        "original": prefix_punct + clean_word + suffix_punct,
                        "suggested": prefix_punct + clean_word + suffix_punct,
                        "changed": False,
                        "change_id": None,
                    }
                ]
                return {
                    "plain": word,
                    "marked": word,
                    "source": "ambiguous",
                    "is_verified": False,
                    "ambiguous": True,
                    "ambiguous_candidates": candidates,
                    "uncertain": True,
                    "uncertain_reason": "This spelling has multiple verified vowel-marked forms; context or regional usage is needed.",
                    "preserved_mark_conflicts": [],
                    "changes": [],
                    "predictions": [],
                    "segments": segments,
                }
        elif len(candidates) == 1:
            chosen_lexicon_form = candidates[0]

        if chosen_lexicon_form is None and not candidates:
            chosen_lexicon_form = self.verified_dict.get(stripped)

        lexicon_graphemes = self._graphemes(chosen_lexicon_form) if chosen_lexicon_form else []
        feature_text = "".join(grapheme["base"] for grapheme in original_graphemes)
        result_segments: List[Dict[str, Any]] = []
        changes: List[Dict[str, Any]] = []
        predictions: List[Dict[str, Any]] = []
        preserved_mark_conflicts = []
        used_lexicon = bool(chosen_lexicon_form)
        uncertain = False
        no_confident_mark = False

        for char_index, grapheme in enumerate(original_graphemes):
            base = grapheme["base"]
            supplied_marks = grapheme["marks"]
            predicted_marks: List[str] = []
            prediction_source = "unmodified"
            confidence = None
            confidence_label = None

            if supplied_marks:
                prediction_source = "preserved"
                if used_lexicon and char_index < len(lexicon_graphemes):
                    expected = [
                        mark for mark in lexicon_graphemes[char_index]["marks"]
                        if mark in config.TARGET_DIACRITICS
                    ]
                    actual = [mark for mark in supplied_marks if mark in config.TARGET_DIACRITICS]
                    if expected and actual != expected:
                        preserved_mark_conflicts.append({
                            "character": base,
                            "existing_marks": actual,
                            "lexicon_marks": expected,
                        })
            elif used_lexicon and char_index < len(lexicon_graphemes):
                predicted_marks = lexicon_graphemes[char_index]["marks"]
                prediction_source = "verified_lexicon"
            elif (
                self.pipeline is not None
                and not base.isspace()
                and not unicodedata.category(base).startswith("P")
                and not self.is_combining_mark(base)
            ):
                features = self.extract_char_features(feature_text, char_index)
                if hasattr(self.pipeline, "predict_proba"):
                    probabilities = self.pipeline.predict_proba([features])[0]
                    classes = list(self.pipeline.classes_)
                    class_index = max(range(len(classes)), key=lambda index: probabilities[index])
                    label = str(classes[class_index])
                    confidence = float(probabilities[class_index])
                else:
                    label = str(self.pipeline.predict([features])[0])

                confidence_label = (
                    "high" if confidence is not None and confidence >= self.HIGH_PROBABILITY
                    else "moderate" if confidence is not None and confidence >= self.UNCERTAIN_PROBABILITY
                    else "uncertain" if confidence is not None
                    else "unavailable"
                )
                if confidence is not None and confidence < self.UNCERTAIN_PROBABILITY:
                    uncertain = True
                mark = self._label_mark(label)
                predictions.append({
                    "character": base,
                    "label": label,
                    "mark": mark,
                    "confidence": confidence,
                    "confidence_label": confidence_label,
                })
                if mark:
                    predicted_marks = [mark]
                    prediction_source = "ml_classifier"
                elif confidence is not None and confidence < self.UNCERTAIN_PROBABILITY:
                    no_confident_mark = True
            else:
                no_confident_mark = True

            original_cluster = unicodedata.normalize("NFC", base + "".join(supplied_marks))
            suggested_cluster = unicodedata.normalize(
                "NFC", base + "".join(supplied_marks or predicted_marks)
            )
            vowel_additions = [
                mark for mark in predicted_marks
                if mark in config.TARGET_DIACRITICS and mark not in supplied_marks
            ]
            change_id = None
            if vowel_additions:
                change_id = f"{token_index}:{char_index}"
                change = {
                    "id": change_id,
                    "character": base,
                    "mark": vowel_additions[0],
                    "mark_name": config.DIACRITIC_NAMES.get(vowel_additions[0], vowel_additions[0]),
                    "source": prediction_source,
                    "confidence": confidence,
                    "confidence_label": confidence_label,
                }
                changes.append(change)

            result_segments.append({
                "original": original_cluster,
                "suggested": suggested_cluster,
                "changed": bool(vowel_additions),
                "change_id": change_id,
            })

        marked_clean_word = "".join(segment["suggested"] for segment in result_segments)
        marked_word = prefix_punct + marked_clean_word + suffix_punct
        if prefix_punct or suffix_punct:
            result_segments = [
                {
                    "original": prefix_punct,
                    "suggested": prefix_punct,
                    "changed": False,
                    "change_id": None,
                },
                *result_segments,
                {
                    "original": suffix_punct,
                    "suggested": suffix_punct,
                    "changed": False,
                    "change_id": None,
                },
            ]
        uncertain_reason = None
        if uncertain and changes:
            uncertain_reason = "One or more classifier probabilities are low; review these suggestions carefully."
        elif (
            no_confident_mark
            or (
                not changes
                and not used_lexicon
                and not has_existing_vowels
                and has_speech_characters
            )
        ) and not changes:
            uncertain_reason = (
                "No vowel mark was confidently restored; this spelling alone may not determine pronunciation."
            )
        return {
            "plain": word,
            "marked": marked_word,
            "source": "verified_lexicon" if used_lexicon else (
                "ml_classifier" if changes else "unmodified"
            ),
            "is_verified": used_lexicon,
            "ambiguous": False,
            "ambiguous_candidates": ambiguous_candidates if has_existing_vowels else [],
            "uncertain": uncertain or no_confident_mark or uncertain_reason is not None,
            "uncertain_reason": uncertain_reason,
            "preserved_mark_conflicts": preserved_mark_conflicts,
            "changes": changes,
            "predictions": predictions,
            "segments": result_segments,
        }

    def restore_word(self, word: str) -> Tuple[str, bool]:
        """Restores a single word and reports whether a verified lexicon entry was used."""
        details = self._restore_word_details(word)
        return details["marked"], details["is_verified"]

    def _restore_with_classifier(self, text: str) -> str:
        """Compatibility wrapper for classifier-only restoration."""
        details = self._restore_word_details(text)
        return details["marked"]

    @staticmethod
    def apply_suggestion_decisions(
        result: Dict[str, Any],
        accepted_change_ids: Optional[List[str]] = None,
    ) -> str:
        """Returns the restored text with only accepted mark additions applied."""
        accepted = set(accepted_change_ids) if accepted_change_ids is not None else None
        output = []
        for token in result.get("tokens", []):
            output.append(token.get("separator", ""))
            for segment in token.get("segments", []):
                change_id = segment.get("change_id")
                if change_id and accepted is not None and change_id not in accepted:
                    output.append(segment["original"])
                else:
                    output.append(segment["suggested"])
        output.append(result.get("trailing_text", ""))
        return "".join(output)

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
                "trailing_text": text,
            }

        text = unicodedata.normalize("NFC", text)
        token_matches = list(re.finditer(r"\S+", text))
        token_details = []
        verified_count = 0
        predicted_count = 0
        cursor = 0

        for token_index, match in enumerate(token_matches):
            token = match.group()
            details = self._restore_word_details(token, token_index)
            details["separator"] = text[cursor:match.start()]
            token_details.append(details)
            cursor = match.end()
            if details["is_verified"]:
                verified_count += 1
            elif details["changes"]:
                predicted_count += 1

        result = {
            "plain": text,
            "marked": "",
            "words_restored": verified_count + predicted_count,
            "verified_word_count": verified_count,
            "predicted_word_count": predicted_count,
            "tokens": token_details,
            "trailing_text": text[cursor:],
        }
        result["marked"] = self.apply_suggestion_decisions(result)

        return result
