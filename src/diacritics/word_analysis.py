"""Text-only Sindhi word analysis using the existing diacritic restorer."""

import csv
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer


VOWEL_GUIDE = {
    config.ZER: {
        "name": "Zer",
        "glyph": config.ZER,
        "effect": "A mark written below a character; often taught as a short i-like vowel.",
    },
    config.ZABAR: {
        "name": "Zabar",
        "glyph": config.ZABAR,
        "effect": "A mark written above a character; often taught as a short a-like vowel.",
    },
    config.PESH: {
        "name": "Pesh",
        "glyph": config.PESH,
        "effect": "A mark written above a character; often taught as a short u-like vowel.",
    },
}


def _split_graphemes(text: str) -> List[Dict[str, Any]]:
    """Groups combining marks with their preceding Unicode base character."""
    graphemes: List[Dict[str, Any]] = []
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.category(char) in ("Mn", "Me", "Mc") and graphemes:
            graphemes[-1]["marks"].append(char)
        else:
            graphemes.append({"base": char, "marks": []})
    return graphemes


def _base_form(text: str) -> str:
    """Removes only the short-vowel marks analyzed here."""
    base = []
    for grapheme in _split_graphemes(text):
        kept_marks = [
            mark for mark in grapheme["marks"]
            if mark not in config.TARGET_DIACRITICS
        ]
        base.append(grapheme["base"] + "".join(kept_marks))
    return unicodedata.normalize("NFC", "".join(base))


def _contains_sindhi_script(text: str) -> bool:
    return any(
        unicodedata.category(char).startswith("L")
        and (
            0x0600 <= ord(char) <= 0x06FF
            or 0x0750 <= ord(char) <= 0x077F
            or 0x08A0 <= ord(char) <= 0x08FF
            or 0xFB50 <= ord(char) <= 0xFDFF
            or 0xFE70 <= ord(char) <= 0xFEFF
        )
        for char in text
    )


def _reviewed_wordbook_entries(
    word: str,
    restorer: SindhiDiacriticRestorer,
    wordbook_path: Path,
) -> List[Dict[str, str]]:
    entries = []
    normalized_base = restorer.strip_diacritics(word)
    with wordbook_path.open(encoding="utf-8", newline="") as wordbook_file:
        for row in csv.DictReader(wordbook_file):
            if row.get("review_status") != "native_reviewed":
                continue
            if restorer.strip_diacritics(row.get("plain", "").strip()) == normalized_base:
                entries.append({
                    "marked": unicodedata.normalize("NFC", row.get("marked", "").strip()),
                    "notes": row.get("notes", "").strip(),
                    "source": row.get("source", "").strip(),
                })
    return entries


def analyze_word(
    word: str,
    restorer: SindhiDiacriticRestorer,
    wordbook_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyzes one word's written marks and model/lexicon restoration suggestions."""
    normalized_word = unicodedata.normalize("NFC", word.strip())
    if not normalized_word:
        raise ValueError("Enter a Sindhi word to analyze.")
    if len(normalized_word.split()) != 1:
        raise ValueError("Enter one word at a time.")
    if not _contains_sindhi_script(normalized_word):
        raise ValueError("Enter one Sindhi word in Perso-Arabic script.")

    analysis = restorer.restore(normalized_word)
    token = analysis["tokens"][0]
    original_graphemes = _split_graphemes(normalized_word)
    restored_graphemes = _split_graphemes(token["marked"])
    prediction_by_index = {}
    for change in token["changes"]:
        try:
            char_index = int(change["id"].rsplit(":", 1)[1])
        except (KeyError, ValueError):
            continue
        prediction_by_index[char_index] = change

    characters = []
    for index, grapheme in enumerate(original_graphemes):
        restored = restored_graphemes[index] if index < len(restored_graphemes) else grapheme
        existing_vowels = [
            mark for mark in grapheme["marks"]
            if mark in config.TARGET_DIACRITICS
        ]
        restored_vowels = [
            mark for mark in restored["marks"]
            if mark in config.TARGET_DIACRITICS
        ]
        additions = list((Counter(restored_vowels) - Counter(existing_vowels)).elements())
        marks_to_explain = restored_vowels or existing_vowels
        change = prediction_by_index.get(index)
        characters.append({
            "character": grapheme["base"],
            "existing_marks": [
                VOWEL_GUIDE[mark]["name"] for mark in existing_vowels
            ],
            "restored_marks": [
                VOWEL_GUIDE[mark]["name"] for mark in restored_vowels
            ],
            "added_marks": [
                VOWEL_GUIDE[mark]["name"] for mark in additions
            ],
            "mark_effects": [
                VOWEL_GUIDE[mark]["effect"] for mark in marks_to_explain
            ],
            "source": change["source"] if change else (
                "preserved" if existing_vowels else "no mark added"
            ),
            "confidence": change["confidence"] if change else None,
        })

    path = wordbook_path or config.SAMPLE_DIACRITICS_FILE
    wordbook_entries = _reviewed_wordbook_entries(normalized_word, restorer, path)
    ambiguous_forms = token["ambiguous_candidates"]
    return {
        "input": normalized_word,
        "base_word": _base_form(normalized_word),
        "restored_word": token["marked"],
        "source": token["source"],
        "uncertain": token["uncertain"],
        "uncertain_reason": token["uncertain_reason"],
        "ambiguous_forms": ambiguous_forms,
        "wordbook_entries": wordbook_entries,
        "characters": characters,
        "audio_available": False,
    }
