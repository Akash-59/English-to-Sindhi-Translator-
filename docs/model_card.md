# Model Card

## 1. Primary Translation Model: NLLB-200 Distilled 600M

### Model Details
- **Developer / Publisher**: Meta AI Research
- **Repository / Checkpoint**: `facebook/nllb-200-distilled-600M`
- **Architecture**: Encoder-Decoder Transformer (`M2M100ForConditionalGeneration`)
- **Parameter Count**: ~600 Million parameters
- **License**: Creative Commons Non-Commercial (CC-BY-NC 4.0)

### Linguistic Specifications
- **Source Language**: English (`eng_Latn`)
- **Target Language**: Sindhi in Perso-Arabic Script (`snd_Arab`)
- **Target Script Orthography**: Perso-Arabic script as officially recognized in Sindh, Pakistan, comprising 52 letters including unique implosives and retroflexes (`ٻ`, `ڄ`, `ڃ`, `ڌ`, `ڊ`, `ڍ`, `ڏ`, `ڦ`, `ٿ`, `ٺ`, `ٽ`, `ڳ`, `ڱ`, `ڻ`, `ڪ`).
- **Tokenizer**: SentencePiece BPE tokenizer with 256,204 vocabulary size.
- **Forced BOS Token ID**: 256158 (`snd_Arab`)

### Physical Footprint & Hardware Feasibility
- **Disk Footprint**: ~2.46 GB (`pytorch_model.bin`)
- **Inference RAM (CPU)**: ~1.8 GB to 2.4 GB peak RSS during active generation.
- **CPU Execution**: Supported natively on AMD64 x86_64 CPU using PyTorch CPU wheel.
- **Threading**: Clamped to 4 threads (`torch.set_num_threads(4)`).

---

## 2. Diacritic Restoration Model: Contextual Sequence Classifier

### Model Details
- **Architecture**: Dual-Stage Hybrid
  - **Stage 1 (Linguistic Ground Truth)**: Exact-match hash table of reviewed educational lexemes.
  - **Stage 2 (Machine Learning Classifier)**: N-gram character-context statistical classifier (`DictVectorizer` + `LogisticRegression` with `class_weight='balanced'`).
- **Target Labels**:
  - `NONE`: No vowel mark attached.
  - `ZABAR` (`\u064e`): Short vowel /a/.
  - `ZER` (`\u0650`): Short vowel /i/.
  - `PESH` (`\u064f`): Short vowel /u/.
- **Features Extracted**:
  - Current character, `prev_1`, `prev_2`, `next_1`, `next_2`.
  - Bigram transitions (`prev_bigram`, `next_bigram`).
  - Positional flags (`is_start`, `is_end`, `is_space`).
  - Unicode character category (`Lo`, `Mn`, `Po`, etc.).
- **Disk Footprint**: ~45 KB (`models/diacritics/diacritic_restorer.joblib`).
- **Inference Latency**: < 15 ms per sentence on CPU.

---

## 3. Empirical Evaluation Results (Real Held-Out Test Data)

### A. Diacritic Restoration Metrics (`data/test/diacritics_test.csv`)
- **Existing Mark Preservation Rate**: 100.00%
- **Macro F1 Score**: 0.6276
- **Per-Vowel Precision & Recall**:
  - `ZABAR (َ)`: Precision = 0.6667, Recall = 0.5455, F1 = 0.6000
  - `ZER (ِ)`: Precision = 0.8333, Recall = 0.7143, F1 = 0.7692
  - `PESH (ُ)`: Precision = 0.2222, Recall = 0.5000, F1 = 0.3077
  - `NONE`: Precision = 0.9412, Recall = 0.8889, F1 = 0.9143

### B. Translation Quality Metrics (`data/test/parallel_test.csv`)
- **Script Validity Rate**: 100% of generated outputs correctly rendered in valid Perso-Arabic Sindhi script.
- **Average CPU Inference Latency**: ~1.8 seconds per sentence.

---

## 4. Known Limitations & Failure Modes
1. **Context-Sensitive Homographs**: Words with identical consonantal skeletons but distinct short vowels (e.g. `پل` = *pul* [bridge] vs *pal* [moment]; `ملڪ` = *mulk* [country] vs *malak* [angel]) require deep semantic contextual modeling and may default to the most frequent prior when sentence context is sparse.
2. **Specialized Terminology**: Highly technical medical or legal jargon not represented in the general Flores/NLLB training set may produce transliterations or generic vocabulary.
3. **Hardware Latency**: On low-power CPU cores, beam search with `num_beams >= 4` can increase latency up to 4–6 seconds for multi-clause sentences.
