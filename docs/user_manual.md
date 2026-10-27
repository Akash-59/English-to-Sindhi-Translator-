# User Manual & Operator Guide

## English-to-Sindhi Translator & Diacritic Restoration System

---

## 1. Web Application Guide (Streamlit Interface)

### Launching the Application
In your terminal, navigate to the project directory and execute:
```bash
streamlit run app.py
```
The browser will automatically open `http://localhost:8501`.

This is the default local mode and retains the offline NLLB-200, Canopy-M, and
Windows SAPI paths. A separate public-cloud setup is documented in
[cloud_deployment.md](./cloud_deployment.md). Cloud mode sends submitted translation
text to Cloudflare, does not collect audio, and does not load the local NLLB or Canopy
weights.

---

### Interface Controls & Sections

#### A. English-to-Sindhi Translation ("🔄 Translate & Restore" Tab)
1. **English Input Box**: Enter English text (single word, conversation, paragraph, or article). Character and word counters update automatically.
2. **Translate to Sindhi Button**: Invokes the local NLLB-200 neural model on CPU.
3. **Compare alternative translations**: Optionally requests up to three beam-search candidates, removes identical outputs, and shows the distinct results side by side. The model may return fewer than three distinct candidates. Alternatives are suggestions ranked by the model, not verified or guaranteed-correct translations.
4. **Preferred translation**: Choose a candidate, edit the preferred Sindhi text, then use the copy button (or select and copy the text if clipboard access is unavailable).
5. **Restore Diacritics Only Button**: If you already have unmarked Sindhi text, paste it into the box and click this button to infer short-vowel marks.
6. **Display Mode Selector**:
   - **Standard Sindhi (Readable)**: Outputs natural, standard Perso-Arabic Sindhi text as commonly printed in newspapers and books.
   - **Learning Mode (Phonological Vowels)**: Adds Zabar ( َ ), Zer ( ِ ), and Pesh ( ُ ) marks to assist learners in correct pronunciation.
7. **Copy-to-Clipboard Area**: A dedicated clean text field for easy copying.

#### B. Sindhi-to-English Translation ("↔️ Sindhi to English" Tab)
- Enter or paste Sindhi text in the editable input field and choose **Translate to English**. The existing local NLLB-200 Distilled 600M model is reused with Sindhi (`snd_Arab`) as the source and English (`eng_Latn`) as the target.
- Edit the Sindhi input and submit again to translate the revised text. The English result is also editable and can be selected and copied.
- This is machine translation: review the result, especially for ambiguous, regional, or context-dependent wording. No translation-accuracy improvement is claimed by adding the reverse direction.
- The separate Sindhi Voice tab provides microphone recording and offline English speech playback.

#### C. Sindhi Voice ("🎙️ Sindhi Voice" Tab)
- Record Sindhi audio using the microphone and choose **Transcribe Sindhi offline**. Canopy-M runs in a separate CPU process, with an offline-only local checkpoint. The output is an unverified suggestion; Arabic-script characters do not prove the language or transcription is correct.
- The ASR worker reports CPU inference time and sampled peak process memory. It runs only when the configured amount of system memory is available, and checks again inside the isolated worker before loading weights. The default threshold is 3 GiB; NLLB is unloaded first.
- Each new recording requires its own explicit consent checkbox before local Canopy-M transcription can start. Recording consent does not send audio outside the Streamlit host.
- The minimum-memory threshold is fixed at 3 GiB and cannot be lowered through configuration. When current available memory is lower, the app disables transcription and does not load or retry Canopy-M.
- Submit the edited Sindhi text to reuse the existing local Sindhi-to-English translation. Edit the English result if needed, then choose **Generate offline English speech** to play it using an installed Windows English SAPI voice.
- Audio recording is handled by the Streamlit host. Run the application locally to keep recording and offline recognition on your own machine.
- The English voice uses an installed local Windows voice; other operating systems may not support this playback path. No Sindhi pronunciation audio is generated.
- The public cloud deployment intentionally disables microphone recording because no hosted provider has been verified for accurate Sindhi transcription in Perso-Arabic script. It also cannot use Windows SAPI.

#### D. Diacritic Intelligence ("✨ Diacritic Intelligence" Tab)
- Enter Sindhi text directly or start with a translation from the translator tab.
- Original and suggested text are displayed side by side; added vowel marks are highlighted.
- Review each suggestion using its verified-lexicon source or classifier probability. Classifier scores are not calibrated guarantees of correctness.
- Accept or reject suggestions individually, then edit the final text manually.
- Existing user-entered marks are preserved. If they conflict with a lexicon entry, the interface explains that the supplied marks were retained.
- Known words with multiple verified vowel-marked forms are left unchanged until the user supplies context or edits the result.
- Word-by-word cards for translated text remain available below the restoration workspace.

#### E. Word Sound & Pronunciation ("🔊 Word Sound & Pronunciation" Tab)
- Enter one Sindhi word or choose an existing wordbook example to inspect its base form, supplied/restored marks, and character-by-character mark breakdown.
- Zer, zabar, and pesh receive simple English explanations. “i-like”, “a-like”, and “u-like” are approximate teaching descriptions, not a phonetic transcription or confirmation of a regional spoken form.
- The existing restorer and reviewed wordbook supply mark suggestions, ambiguity alternatives, and available notes.
- The result is editable for personal use.
- This is text-only analysis. The project has no speech recordings or offline audio resource; no audio pronunciation is generated or verified.

#### F. Searchable Wordbook ("📖 Verified Wordbook" Tab)
- Type English meanings, unmarked Sindhi, or marked Sindhi into the search bar.
- Filters verified educational vocabulary in real time.

#### G. System Diagnostics & Memory Control (Sidebar)
- **Offline Badge**: Confirms zero network requests.
- **Process RAM Indicator**: Displays live memory consumption in Megabytes.
- **🗑️ Free Model Memory Button**: Evicts model weights from RAM when you need to free system memory.

#### H. User Feedback ("📝 User Feedback" Tab)
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

### Q2: How do I verify local mode works with no internet?
**Solution**: Disconnect your Wi-Fi or unplug your Ethernet cable. The translator runs 100% locally from the `models/` directory.

### Q3: How do I change the number of CPU threads?
**Solution**: Set the environment variable `OMP_NUM_THREADS` before launching:
```bash
$env:OMP_NUM_THREADS="2"  # Windows PowerShell
export OMP_NUM_THREADS=2  # Linux
```

For public deployment steps, provider limits, and speech limitations, see
[cloud_deployment.md](./cloud_deployment.md).
