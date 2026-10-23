"""
Configuration module for Offline English-to-Sindhi Translation & Diacritic Restoration System.
Designed for 8 GB RAM CPU-only deployment on Windows / Linux.
"""

from pathlib import Path
import os

# Root directory
BASE_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "validation"
TEST_DATA_DIR = DATA_DIR / "test"

PARALLEL_TRAIN_FILE = TRAIN_DATA_DIR / "parallel_train.csv"
PARALLEL_VAL_FILE = VAL_DATA_DIR / "parallel_val.csv"
PARALLEL_TEST_FILE = TEST_DATA_DIR / "parallel_test.csv"

DIACRITICS_TRAIN_FILE = TRAIN_DATA_DIR / "diacritics_train.csv"
DIACRITICS_VAL_FILE = VAL_DATA_DIR / "diacritics_val.csv"
DIACRITICS_TEST_FILE = TEST_DATA_DIR / "diacritics_test.csv"

SAMPLE_PARALLEL_FILE = DATA_DIR / "sample_parallel.csv"
SAMPLE_DIACRITICS_FILE = DATA_DIR / "sample_diacritics.csv"

# Model paths
MODELS_DIR = BASE_DIR / "models"
TRANSLATION_MODEL_DIR = MODELS_DIR / "translation"
DIACRITIC_MODEL_DIR = MODELS_DIR / "diacritics"
DIACRITIC_MODEL_FILE = DIACRITIC_MODEL_DIR / "diacritic_restorer.joblib"

# Assets and fonts
ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

# Docs and reports
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"

# Default Model Selection
# NLLB-200 Distilled 600M is verified for:
# - Source: English (eng_Latn)
# - Target: Sindhi in Arabic script (snd_Arab)
# - Size: ~2.4 GB checkpoint, fits comfortably in 8 GB RAM on CPU
# - License: CC-BY-NC 4.0 / Open Research
DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG_CODE = "eng_Latn"
TGT_LANG_CODE = "snd_Arab"

# Memory and CPU constraints
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
CPU_INFERENCE_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
BATCH_SIZE = 1  # CPU inference default batch size to minimize RAM spike

# Unicode Diacritic Definitions for Sindhi (Perso-Arabic)
ZABAR = "\u064E"       # Fatha / Zabar (َ)
ZER = "\u0650"         # Kasra / Zer (ِ)
PESH = "\u064F"        # Damma / Pesh (ُ)
SUKUN = "\u0652"       # Sukun / Jazm (ْ)
SHADDA = "\u0651"      # Tashdid (ّ)
KHARII_ZABAR = "\u0670" # Dagger Alif (ٰ)

# Target short vowels for restoration
TARGET_DIACRITICS = [ZABAR, ZER, PESH]
DIACRITIC_NAMES = {
    ZABAR: "Zabar (َ)",
    ZER: "Zer (ِ)",
    PESH: "Pesh (ُ)",
    "NONE": "None"
}

# Offline Flags
FORCE_OFFLINE = os.environ.get("FORCE_OFFLINE", "0") == "1"
