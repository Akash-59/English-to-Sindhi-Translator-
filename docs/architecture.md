# System Architecture

## Project: Offline English-to-Sindhi Neural Machine Translation and Diacritic Restoration System

### 1. Architectural Philosophy
The system is built upon a modular, decoupled architecture tailored specifically for resource-constrained environments (standard 8 GB RAM, CPU-only inference, Windows 10/11 and Linux compatibility). The core tenet is strict separation of concerns between **cross-lingual semantic translation** and **phonological vowel diacritic restoration**, ensuring that uncertain vowel predictions do not degrade lexical translation accuracy.

---

### 2. High-Level Modular Flow

```
                      +-----------------------------+
                      |     English Input Text      |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Text Preprocessing & NFC  |
                      |   Segmentation Pipeline     |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   NLLB-200 Distilled 600M   |
                      |   (Seq2Seq NMT Model)       |
                      |   Source: eng_Latn          |
                      |   Target: snd_Arab          |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Standard Sindhi Output   |
                      |    (Perso-Arabic Script)    |
                      +-----------------------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
             [Standard Mode]                 [Learning Mode]
             Clean readable Sindhi                   |
             without extra vowels                    v
                                     +-----------------------------+
                                     |    Diacritic Restoration    |
                                     |    - Unicode Mark Preserver |
                                     |    - Verified Lexicon Map   |
                                     |    - Contextual ML Classifier|
                                     +-----------------------------+
                                                     |
                                                     v
                                     +-----------------------------+
                                     |    Marked Sindhi Output     |
                                     |   (with َ Zabar, ِ Zer, ُ Pesh)  |
                                     |   & Word-by-Word Breakdown  |
                                     +-----------------------------+
```

---

### 3. Core Subsystems

#### A. Translation Subsystem (`src/translation/`)
1. **`model_loader.py`**:
   - Manages lazy loading and unpickling of neural weights only when active translation is requested.
   - Restricts CPU execution threads via `torch.set_num_threads(4)` to prevent processor thrashing.
   - Enforces `local_files_only=True` to guarantee zero outbound telemetry or network dependency.
2. **`translator.py`**:
   - Performs Unicode normalization (NFC), sentence and paragraph segmentation for texts exceeding 300 characters to prevent memory spikes.
   - Tokenizes input using the verified `eng_Latn` source token and forces generation prefix to `snd_Arab` (Sindhi in Perso-Arabic orthography).
   - Generates output deterministically with beam search (`num_beams=2`) and ngram repetition penalties.

#### B. Diacritic Restoration Subsystem (`src/diacritics/`)
1. **`restore_diacritics.py`**:
   - Dual-tier prediction engine:
     - **Tier 1 (Linguistic Lexicon)**: Exact match against linguistically reviewed ground truth for frequent nouns, educational verbs, and minimal pairs.
     - **Tier 2 (Statistical Character Classifier)**: Extracts 5-gram character window contexts, positional boundary flags, and Unicode categories to predict `NONE`, `ZABAR`, `ZER`, or `PESH`.
   - **Inviolable Invariant**: Existing Unicode combining marks (`\u064e`, `\u0650`, `\u064f`, `\u0651`, `\u0652`) in the input are strictly preserved and never overwritten or discarded.
2. **`train_diacritics.py`**:
   - Aligns plain Sindhi text with annotated marked ground truth.
   - Trains a scikit-learn classification pipeline with `DictVectorizer` and `class_weight='balanced'` Logistic Regression to avoid majority-class collapse.

#### C. Data Management & Validation Subsystem (`src/data/`)
1. **`validate_dataset.py`**:
   - Unicode audit and validation for both language columns.
   - Validates that English inputs contain Latin characters and zero Arabic script.
   - Validates that Sindhi inputs contain valid Perso-Arabic codepoints (`\u0600` - `\u06FF`, `\u0750` - `\u077F`, `\u08A0` - `\u08FF`).
   - Checks word-count ratios (0.15 to 6.0) to filter gross misalignment.
   - Performs leakage-free train/validation/test partitioning (80/10/10) with deterministic seeding.

---

### 4. Memory and Hardware Safety Controls
- **RAM Thresholds**: Total active process RSS remains well below 3.5 GB on an 8 GB physical RAM system.
- **Lazy Allocation**: Model weights can be evicted from memory via the UI "Unload Model" control.
- **CPU Threading**: Thread pool clamped to 4 cores to maintain responsive system multitasking on Windows desktop environments.
