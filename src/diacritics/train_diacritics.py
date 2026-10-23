"""
Training pipeline for the Sindhi Diacritic Restoration Model.
Aligns plain and phonologically marked text, extracts character-context features,
and trains a balanced scikit-learn sequence classifier.
"""

import os
from pathlib import Path
import unicodedata
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

import config
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer, COMBINING_MARKS


def align_and_extract_features(plain_text: str, marked_text: str) -> List[Tuple[Dict[str, Any], str]]:
    """
    Aligns a plain string with its marked version and generates feature-label pairs.
    Handles Unicode NFC/NFD decomposition and combining mark assignments.
    """
    plain_norm = unicodedata.normalize("NFC", plain_text)
    marked_norm = unicodedata.normalize("NFC", marked_text)

    # Decompose marked string to inspect base letters and attached marks
    marked_chars = list(marked_norm)
    samples = []
    
    plain_idx = 0
    m_idx = 0
    m_len = len(marked_chars)

    while plain_idx < len(plain_norm) and m_idx < m_len:
        p_ch = plain_norm[plain_idx]
        m_ch = marked_chars[m_idx]

        # Check if marked char matches plain char
        if p_ch == m_ch:
            # Check if immediately followed by a vowel combining mark
            target_label = "NONE"
            if m_idx + 1 < m_len and SindhiDiacriticRestorer.is_combining_mark(marked_chars[m_idx + 1]):
                mark = marked_chars[m_idx + 1]
                if mark == config.ZABAR:
                    target_label = "ZABAR"
                elif mark == config.ZER:
                    target_label = "ZER"
                elif mark == config.PESH:
                    target_label = "PESH"
                m_idx += 1  # Skip the mark

            feats = SindhiDiacriticRestorer.extract_char_features(plain_norm, plain_idx)
            samples.append((feats, target_label))

            plain_idx += 1
            m_idx += 1
        else:
            # Shift marked if combining mark was detached or out of sync
            if SindhiDiacriticRestorer.is_combining_mark(m_ch):
                m_idx += 1
            else:
                plain_idx += 1
                m_idx += 1

    return samples


def train_diacritic_model(
    train_file: Path = config.DIACRITICS_TRAIN_FILE,
    output_file: Path = config.DIACRITIC_MODEL_FILE,
):
    """Loads training data, builds verified lexicon, extracts samples, and trains classifier."""
    print(f"[TrainDiacritics] Loading training data from: {train_file}")
    df = pd.read_csv(train_file)

    lexicon: Dict[str, str] = {}
    all_features = []
    all_labels = []

    for _, row in df.iterrows():
        plain = str(row["plain"]).strip()
        marked = str(row["marked"]).strip()

        # Build verified lexicon for individual words and short phrases
        words_plain = plain.split()
        words_marked = marked.split()
        if len(words_plain) == len(words_marked):
            for wp, wm in zip(words_plain, words_marked):
                clean_wp = SindhiDiacriticRestorer.strip_diacritics(wp)
                lexicon[clean_wp] = wm

        # Extract character features and labels
        samples = align_and_extract_features(plain, marked)
        for feats, lbl in samples:
            all_features.append(feats)
            all_labels.append(lbl)

    print(f"[TrainDiacritics] Extracted {len(all_features)} character samples.")
    print(f"[TrainDiacritics] Built lexicon with {len(lexicon)} verified entries.")

    label_counts = pd.Series(all_labels).value_counts().to_dict()
    print(f"[TrainDiacritics] Label distribution: {label_counts}")

    # Build scikit-learn pipeline with DictVectorizer and balanced Logistic Regression
    pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=42,
        )),
    ])

    print("[TrainDiacritics] Fitting classification pipeline...")
    pipeline.fit(all_features, all_labels)

    # Calculate train metrics
    train_preds = pipeline.predict(all_features)
    print("\n[TrainDiacritics] Training Set Classification Report:")
    print(classification_report(all_labels, train_preds, zero_division=0))

    # Save artifact
    os.makedirs(output_file.parent, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "lexicon": lexicon}, output_file)
    print(f"[TrainDiacritics] Model successfully saved -> {output_file}")


if __name__ == "__main__":
    train_diacritic_model()
