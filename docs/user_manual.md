# User Manual & Operator Guide

## English-to-Sindhi Offline AI Translator & Diacritic Restoration System

---

## 1. Web Application Guide (Streamlit Interface)

### Launching the Application
In your terminal, navigate to the project directory and execute:
```bash
streamlit run app.py
```
The browser will automatically open `http://localhost:8501`.

---

### Interface Controls & Sections

#### A. Translation Workspace ("🔄 Translate & Restore" Tab)
1. **English Input Box**: Enter English text (single word, conversation, paragraph, or article). Character and word counters update automatically.
2. **Translate to Sindhi Button**: Invokes the local NLLB-200 neural model on CPU.
3. **Restore Diacritics Only Button**: If you already have unmarked Sindhi text, paste it into the box and click this button to infer short-vowel marks.
4. **Display Mode Selector**:
   - **Standard Sindhi (Readable)**: Outputs natural, standard Perso-Arabic Sindhi text as commonly printed in newspapers and books.
   - **Learning Mode (Phonological Vowels)**: Adds Zabar ( َ ), Zer ( ِ ), and Pesh ( ُ ) marks to assist learners in correct pronunciation.
5. **Copy-to-Clipboard Area**: A dedicated clean text field for easy one-click copying.

#### B. Educational Learning Mode ("🎓 Educational Learning Mode" Tab)
- **Word-by-Word Cards**: Each word in the translated sentence is segmented into an interactive card displaying:
  - Phonologically marked Sindhi text in prominent Arabic Naskh script.
  - Plain unmarked base spelling.
  - Status indicator badge:
    - **Verified (Green)**: Pronunciation confirmed by native lexical ground truth.
    - **ML Prediction (Blue)**: Statistically inferred short vowels.

#### C. Searchable Wordbook ("📖 Verified Wordbook" Tab)
- Type English meanings, unmarked Sindhi, or marked Sindhi into the search bar.
- Filters verified educational vocabulary in real time.

#### D. System Diagnostics & Memory Control (Sidebar)
- **Offline Badge**: Confirms zero network requests.
- **Process RAM Indicator**: Displays live memory consumption in Megabytes.
- **🗑️ Free Model Memory Button**: Evicts model weights from RAM when you need to free system memory.

#### E. User Feedback ("📝 User Feedback" Tab)
- Submit corrections for translations or vowel diacritics.
- Corrections are stored locally in `data/user_feedback.jsonl` for continuous model improvement.

---

## 2. Command-Line Workflows

### Running Batch Translation Evaluation
```bash
python -m src.evaluation.evaluate_translation
```

### Running Diacritic Model Evaluation
```bash
python -m src.diacritics.evaluate_diacritics
```

### Running Hardware & CPU Latency Benchmarks
```bash
python -m src.evaluation.benchmark_inference
```

### Re-Training Diacritic Model
```bash
python -m src.diacritics.train_diacritics
```

---

## 3. Troubleshooting & FAQ

### Q1: Why do some Sindhi letters look disconnected or generic?
**Solution**: Ensure you are using a browser supporting Arabic script shaping (Chrome, Edge, Firefox). The interface automatically applies the `Amiri` and `Lateef` font stack, with fallback to `Segoe UI Historic` and `Tahoma`.

### Q2: How do I verify the system works with no internet?
**Solution**: Disconnect your Wi-Fi or unplug your Ethernet cable. The translator runs 100% locally from the `models/` directory.

### Q3: How do I change the number of CPU threads?
**Solution**: Set the environment variable `OMP_NUM_THREADS` before launching:
```bash
$env:OMP_NUM_THREADS="2"  # Windows PowerShell
export OMP_NUM_THREADS=2  # Linux
```
