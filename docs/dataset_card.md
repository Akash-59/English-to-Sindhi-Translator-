# Dataset Card

## English-to-Sindhi Parallel & Diacritic Annotation Datasets

### 1. Dataset Summary
This repository organizes, audits, and maintains two foundational datasets specifically engineered for offline natural language translation and phonological vowel learning:
1. **Bilingual Parallel Corpus (`data/sample_parallel.csv`, `data/train/parallel_train.csv`, `data/validation/parallel_val.csv`, `data/test/parallel_test.csv`)**: High-accuracy English-to-Sindhi pairs across core socio-linguistic domains.
2. **Sindhi Diacritic Restoration Dataset (`data/sample_diacritics.csv`, `data/train/diacritics_train.csv`, `data/validation/diacritics_val.csv`, `data/test/diacritics_test.csv`)**: Hand-verified Sindhi vocabulary and sentences annotated with short-vowel diacritics (Zabar, Zer, Pesh, Sukun, Shadda).

---

## 2. Bilingual Parallel Corpus

### Schema
```csv
english,sindhi,source,license
"Hello, how are you?","هيلو، توهان ڪيئن آهيو؟","core_curated","CC0"
"Education is the key to success.","تعليم ڪاميابيءَ جي ڪنجي آهي۔","core_curated","CC0"
```

### Domain Coverage
- **Everyday & Greetings**: Salutations, introductory phrases, personal questions, polite formulas.
- **Education & Academic**: Universities, libraries, books, science, knowledge acquisition, examinations.
- **Family & Household**: Kinship terms (father, mother, brother, sister, children), home life.
- **Health & Medicine**: Doctor visits, medications, preventative wellness, hospitals, hydration.
- **Travel & Logistics**: Directions, transportation, railways, regional landmarks across Sindh.
- **Technology & Computing**: Offline applications, artificial intelligence, passwords, digital literacy.
- **Formal & Administrative**: Legal notices, civic rights, official deadlines, compliance.

### Investigated External Corpora
| Resource Name | Source URL | Script | License | Status & Integration Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Mozilla Data Collective Parallel** | `mozilladatacollective.com/datasets/cmr29jici02trmk07l5xlcsak` | Perso-Arabic / Latin | Open Collective License | Evaluated for expanded fine-tuning |
| **Mozilla Data Collective Monolingual** | `mozilladatacollective.com/datasets/cmr54pbgw01omnt07h9vzqrlb` | Perso-Arabic (Sindhi) | Open Collective License | High-volume corpus for language modeling |
| **OPUS Multilingual Parallel Corpora** | `opus.nlpl.eu` (Tatoeba, GNOME, Ubuntu) | Perso-Arabic / Latin | CC-BY / Open Source | Cleaned subset for baseline alignment |
| **IndicTrans2 Parallel Benchmarks** | `github.com/AI4Bharat/IndicTrans2` | Arabic / Devanagari | MIT / Open Research | Evaluated; note HuggingFace repo gating restrictions |

---

## 3. Verified Diacritic Dataset

### Schema
```csv
plain,marked,source,review_status,notes
"علم","عِلمُ","Sindhi_Educational_Corpus","native_reviewed","ilm = knowledge (Zer under Ain, Pesh on Meem)"
"پل","پُلِ","Sindhi_Lexicon_MinimalPair_Bridge","native_reviewed","pul = bridge (Pesh on Pe, Zer on Lam)"
"پل","پَلُ","Sindhi_Lexicon_MinimalPair_Moment","native_reviewed","pal = moment (Zabar on Pe, Pesh on Lam)"
```

### Short Vowel Inventory Handled
- **U+064E ( َ ) Zabar**: Short /a/ vowel.
- **U+0650 ( ِ ) Zer**: Short /i/ vowel.
- **U+064F ( ُ ) Pesh**: Short /u/ vowel.
- **U+0652 ( ْ ) Sukun / Jazm**: Vowelless consonant marker.
- **U+0651 ( ّ ) Shadda / Tashdid**: Consonant gemination.

### Verification Criteria
- All entries were verified by proficient native readers in accordance with standard orthographic rules published by the Sindhi Language Authority (سنڌي ٻوليءَ جو بااختيار ادارو).
- Explicit minimal pairs were curated to evaluate context-sensitive disambiguation (e.g., *pul* vs *pal*, *mulk* vs *malak*).
- Status annotation: Marked explicitly as `native_reviewed`. Placeholder or unreviewed records are strictly excluded from testing sets.

---

## 4. Cleaning & Validation Pipeline (`src/data/validate_dataset.py`)
1. **Unicode NFC Canonicalization**: Strips decomposed variants and standardizes combining mark attachment order.
2. **Script Verification**:
   - English source must contain Latin characters and zero Arabic glyphs.
   - Sindhi target must contain Arabic-script characters (`\u0600`–`\u06FF`, etc.) and zero Latin letters.
3. **Length Bound & Ratio Filtering**: Filters strings shorter than 2 characters or longer than 1000 characters; enforces length ratio bounds ($0.15 \le \frac{\text{target words}}{\text{source words}} \le 6.0$).
4. **Deduplication**: Eliminates identical pairs and duplicate source prompts to prevent contradictory targets.
5. **Partitioning**: Splits data reproducibly (80% train, 10% validation, 10% test) with verified zero sentence overlap across sets.
