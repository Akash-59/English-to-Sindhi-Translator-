"""
Dataset validation and cleaning pipeline for English-Sindhi parallel data.
Handles Unicode NFC normalization, script verification, deduplication, length filtering,
leakage-free train/val/test splitting, and structured quality reporting.
"""

import os
import re
import unicodedata
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

import config


# Arabic script Unicode range (including Sindhi specific extensions)
ARABIC_SCRIPT_REGEX = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_SCRIPT_REGEX = re.compile(r"[a-zA-Z]")


class DatasetValidator:
    """Validates, cleans, and splits English-Sindhi parallel text data."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.report: Dict[str, Any] = {}

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes Unicode into canonical NFC form and collapses duplicate whitespace."""
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def contains_arabic_script(text: str) -> bool:
        """Checks if text contains Arabic/Sindhi script characters."""
        return bool(ARABIC_SCRIPT_REGEX.search(text))

    @staticmethod
    def contains_latin_script(text: str) -> bool:
        """Checks if text contains Latin script characters."""
        return bool(LATIN_SCRIPT_REGEX.search(text))

    def clean_parallel_df(
        self,
        df: pd.DataFrame,
        en_col: str = "english",
        sd_col: str = "sindhi",
        min_chars: int = 2,
        max_chars: int = 1000,
        min_word_ratio: float = 0.15,
        max_word_ratio: float = 6.0,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans and audits a parallel dataframe.
        Identifies and flags invalid records rather than silently dropping them.
        """
        initial_count = len(df)
        exclusions = {
            "empty_or_null": 0,
            "too_short": 0,
            "too_long": 0,
            "script_mismatch_english": 0,
            "script_mismatch_sindhi": 0,
            "ratio_anomaly": 0,
            "exact_duplicate_pairs": 0,
            "conflicting_duplicate_sources": 0,
        }

        records = []
        for idx, row in df.iterrows():
            en_raw = row.get(en_col, "")
            sd_raw = row.get(sd_col, "")

            en = self.normalize_text(en_raw)
            sd = self.normalize_text(sd_raw)

            # Check null/empty
            if not en or not sd:
                exclusions["empty_or_null"] += 1
                continue

            # Check length bounds
            if len(en) < min_chars or len(sd) < min_chars:
                exclusions["too_short"] += 1
                continue
            if len(en) > max_chars or len(sd) > max_chars:
                exclusions["too_long"] += 1
                continue

            # Script verification
            # English should contain Latin characters and NO Arabic characters
            if not self.contains_latin_script(en) or self.contains_arabic_script(en):
                exclusions["script_mismatch_english"] += 1
                continue

            # Sindhi must contain Arabic/Sindhi characters
            if not self.contains_arabic_script(sd):
                exclusions["script_mismatch_sindhi"] += 1
                continue

            # Length ratio check (protect against gross misalignment)
            en_words = max(1, len(en.split()))
            sd_words = max(1, len(sd.split()))
            ratio = sd_words / en_words
            if ratio < min_word_ratio or ratio > max_word_ratio:
                exclusions["ratio_anomaly"] += 1
                continue

            records.append({
                "english": en,
                "sindhi": sd,
                "source": row.get("source", "unknown"),
                "license": row.get("license", "unspecified"),
            })

        cleaned_df = pd.DataFrame(records)

        # Deduplicate exact pairs
        before_dedup = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=["english", "sindhi"]).reset_index(drop=True)
        exclusions["exact_duplicate_pairs"] = before_dedup - len(cleaned_df)

        # Deduplicate source side to avoid train/test contamination or contradictory targets
        before_src_dedup = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=["english"]).reset_index(drop=True)
        exclusions["conflicting_duplicate_sources"] = before_src_dedup - len(cleaned_df)

        final_count = len(cleaned_df)

        audit_report = {
            "initial_records": initial_count,
            "final_clean_records": final_count,
            "retention_rate_pct": round((final_count / initial_count * 100) if initial_count else 0, 2),
            "exclusions": exclusions,
        }
        self.report["cleaning"] = audit_report
        return cleaned_df, audit_report

    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits clean dataframe into train/val/test partitions reproducibly."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

        shuffled = df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_df = shuffled.iloc[:n_train].reset_index(drop=True)
        val_df = shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = shuffled.iloc[n_train + n_val:].reset_index(drop=True)

        # Verify zero leakage across splits
        train_en_set = set(train_df["english"])
        val_en_set = set(val_df["english"])
        test_en_set = set(test_df["english"])

        leakage_val = len(train_en_set.intersection(val_en_set))
        leakage_test = len(train_en_set.intersection(test_en_set))

        split_info = {
            "total": n,
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "leakage_train_val": leakage_val,
            "leakage_train_test": leakage_test,
        }
        self.report["splitting"] = split_info
        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
    ):
        """Saves partitions to CSV with UTF-8 encoding."""
        os.makedirs(output_dir / "train", exist_ok=True)
        os.makedirs(output_dir / "validation", exist_ok=True)
        os.makedirs(output_dir / "test", exist_ok=True)

        train_path = output_dir / "train" / "parallel_train.csv"
        val_path = output_dir / "validation" / "parallel_val.csv"
        test_path = output_dir / "test" / "parallel_test.csv"

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")

        print(f"[DatasetValidator] Saved train ({len(train_df)}) -> {train_path}")
        print(f"[DatasetValidator] Saved val   ({len(val_df)})   -> {val_path}")
        print(f"[DatasetValidator] Saved test  ({len(test_df)})  -> {test_path}")

    def save_report(self, report_path: Path):
        """Saves audit report to JSON."""
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print(f"[DatasetValidator] Cleaning report written to: {report_path}")
