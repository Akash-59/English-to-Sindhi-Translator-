"""
Evaluation script for Sindhi Diacritic Restoration on held-out test data.
Computes per-vowel precision, recall, and F1 (Zabar, Zer, Pesh),
Macro F1, exact sentence-level match accuracy, and existing-mark preservation accuracy.
"""

import os
from pathlib import Path
import json
import unicodedata
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import classification_report, precision_recall_fscore_support

import config
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer
from src.diacritics.train_diacritics import align_and_extract_features


def evaluate_diacritics(
    test_file: Path = config.DIACRITICS_TEST_FILE,
    report_file: Path = config.REPORTS_DIR / "diacritic_evaluation_report.json",
) -> Dict[str, Any]:
    """Runs rigorous evaluation on held-out test data."""
    print(f"[EvalDiacritics] Evaluating on held-out test set: {test_file}")
    df = pd.read_csv(test_file)
    restorer = SindhiDiacriticRestorer()

    exact_matches = 0
    total_sentences = len(df)
    existing_marks_preserved = 0
    total_existing_marks = 0

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        plain = str(row["plain"]).strip()
        marked_gold = str(row["marked"]).strip()

        # Run system prediction
        prediction_res = restorer.restore(plain)
        marked_pred = prediction_res["marked"]

        # Exact match test (normalized NFC)
        gold_nfc = unicodedata.normalize("NFC", marked_gold)
        pred_nfc = unicodedata.normalize("NFC", marked_pred)
        if pred_nfc == gold_nfc:
            exact_matches += 1

        # Character alignment for token/label metrics
        gold_samples = align_and_extract_features(plain, marked_gold)
        pred_samples = align_and_extract_features(plain, marked_pred)

        min_len = min(len(gold_samples), len(pred_samples))
        for i in range(min_len):
            y_true.append(gold_samples[i][1])
            y_pred.append(pred_samples[i][1])

        # Test existing mark preservation:
        # Feed the already marked gold text through restorer and check if any marks were destroyed
        pass_through_res = restorer.restore(marked_gold)
        gold_marks = [c for c in marked_gold if c in config.TARGET_DIACRITICS]
        passed_marks = [c for c in pass_through_res["marked"] if c in config.TARGET_DIACRITICS]
        total_existing_marks += len(gold_marks)
        existing_marks_preserved += sum(1 for m in gold_marks if m in passed_marks)

    # Compute metrics
    labels = ["ZABAR", "ZER", "PESH", "NONE"]
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for idx, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    macro_f1 = round(float(f1.mean()), 4)
    exact_match_acc = round(exact_matches / total_sentences if total_sentences else 0.0, 4)
    preservation_rate = round(
        existing_marks_preserved / total_existing_marks if total_existing_marks else 1.0, 4
    )

    report = {
        "test_dataset": str(test_file),
        "total_test_sentences": total_sentences,
        "exact_sentence_accuracy": exact_match_acc,
        "existing_mark_preservation_accuracy": preservation_rate,
        "macro_f1": macro_f1,
        "per_diacritic_metrics": per_class,
    }

    print("\n--- Diacritic Restoration Test Results ---")
    print(f"Exact Sentence Accuracy: {exact_match_acc * 100:.2f}% ({exact_matches}/{total_sentences})")
    print(f"Existing Mark Preservation: {preservation_rate * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    for lbl in ["ZABAR", "ZER", "PESH"]:
        stats = per_class[lbl]
        print(f" {lbl:5s} -> P: {stats['precision']:.4f}, R: {stats['recall']:.4f}, F1: {stats['f1']:.4f}")

    os.makedirs(report_file.parent, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[EvalDiacritics] Saved evaluation report -> {report_file}")

    return report


if __name__ == "__main__":
    evaluate_diacritics()
