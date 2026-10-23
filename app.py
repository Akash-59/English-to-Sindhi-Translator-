"""
Streamlit Web Application: Offline English-to-Sindhi AI Translation & Diacritic Restoration System
Designed for local, CPU-only 8 GB RAM execution with dual-mode (Standard & Educational Learning Mode).
"""

import os
import time
import json
from pathlib import Path
import streamlit as st
import pandas as pd

import config
from src.translation.translator import SindhiTranslator
from src.translation.model_loader import TranslationModelLoader
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer

# Page configuration
st.set_page_config(
    page_title="Sindhi AI - Offline English to Sindhi Translator",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for RTL, Typography, and Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;600;700&display=swap');

    .sindhi-text {
        font-family: 'Amiri', 'Lateef', 'Scheherazade New', 'Segoe UI Historic', 'Tahoma', serif;
        direction: rtl;
        text-align: right;
        font-size: 1.55rem;
        line-height: 2.2rem;
        padding: 1.2rem;
        background: #fdfdfd;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        color: #1a202c;
    }
    
    .dark .sindhi-text {
        background: #1a202c;
        border: 1px solid #2d3748;
        color: #f7fafc;
    }

    .sindhi-word-chip {
        display: inline-block;
        padding: 4px 10px;
        margin: 4px;
        border-radius: 6px;
        background: #edf2f7;
        font-family: 'Amiri', serif;
        font-size: 1.3rem;
        direction: rtl;
    }

    .verified-chip {
        border-left: 4px solid #38a169;
        background: #f0fff4;
    }

    .predicted-chip {
        border-left: 4px solid #3182ce;
        background: #ebf8ff;
    }

    .badge-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-offline {
        background-color: #c6f6d5;
        color: #22543d;
    }
    .badge-cpu {
        background-color: #feebc8;
        color: #7b341e;
    }
    .diacritic-guide-card {
        padding: 14px;
        border-radius: 8px;
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


@st.cache_resource(show_spinner="Initializing Translation Engine...")
def get_translator():
    return SindhiTranslator(lazy_init=True)


@st.cache_resource(show_spinner="Initializing Diacritic Restoration Model...")
def get_diacritic_engine():
    return SindhiDiacriticRestorer()


translator = get_translator()
diacritic_engine = get_diacritic_engine()


# Sidebar Navigation & Diagnostics
with st.sidebar:
    st.title("⚙️ System Status")
    
    # Offline & CPU Badges
    st.markdown('<span class="badge-status badge-offline">🟢 100% Offline Capable</span> <span class="badge-status badge-cpu">💻 CPU-Optimized</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Model Specifications")
    st.write(f"**NMT Model:** NLLB-200 Distilled (600M)")
    st.write(f"**Target Code:** `snd_Arab` (Perso-Arabic Sindhi)")
    st.write(f"**Diacritic Restorer:** Scikit-Learn Contextual Sequence Classifier + Verified Lexicon")
    
    # Live Diagnostics
    status = TranslationModelLoader.get_status()
    st.write(f"**Process RAM (RSS):** {status['process_rss_mb']:.1f} MB")
    st.write(f"**CPU Threads:** {status['cpu_threads']}")
    st.write(f"**Local Weights Loaded:** {'Yes' if status['loaded'] else 'On-Demand (Lazy)'}")
    
    st.markdown("---")
    st.subheader("📚 Sindhi Vowel Quick Guide")
    st.markdown("""
    - **Zabar ( َ ) [U+064E]:** Short short vowel /a/ (e.g. مَدَدَ - madad)
    - **Zer ( ِ ) [U+0650]:** Short vowel /i/ (e.g. عِلمَ - ilm, ڪِتابَ - kitab)
    - **Pesh ( ُ ) [U+064F]:** Short vowel /u/ (e.g. پُلِ - pul, اُستادَ - ustad)
    - **Jazm / Sukun ( ْ ):** Vowelless consonant marker
    """)

    st.markdown("---")
    if st.button("🗑️ Free Model Memory (Unload)"):
        TranslationModelLoader.unload()
        st.success("Translation model unloaded from RAM!")


# Main Application Interface
st.title("📖 English to Sindhi Neural Translator & Diacritic Restorer")
st.caption("Offline, privacy-preserving AI translation with Perso-Arabic script orthography and phonological vowel learning.")

# Mode Selector Tabs
tab_translate, tab_learning, tab_dictionary, tab_benchmarks, tab_feedback = st.tabs([
    "🔄 Translate & Restore",
    "🎓 Educational Learning Mode",
    "📖 Verified Wordbook",
    "📊 Benchmarks & Research",
    "📝 User Feedback",
])

with tab_translate:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("English Input")
        default_prompt = "I am learning Sindhi. Books are our best friends."
        user_input = st.text_area(
            "Enter English text (sentence, paragraph, or phrase):",
            value=default_prompt,
            height=180,
            placeholder="Type English text here...",
            help="Accepts everyday conversation, education, healthcare, technology, or short articles.",
        )

        char_cnt = len(user_input)
        word_cnt = len(user_input.split())
        st.caption(f"Characters: {char_cnt} | Words: {word_cnt}")

        col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
        with col_b1:
            translate_btn = st.button("🚀 Translate to Sindhi", type="primary", use_container_width=True)
        with col_b2:
            restore_only_btn = st.button("✨ Restore Diacritics", use_container_width=True)
        with col_b3:
            clear_btn = st.button("🧹 Clear", use_container_width=True)

        if clear_btn:
            st.rerun()

    with col_output:
        st.subheader("Sindhi Output (سنڌي)")
        
        mode = st.radio(
            "Display Mode:",
            ["Standard Sindhi (Readable)", "Learning Mode (Phonological Vowels َ ِ ُ)"],
            horizontal=True,
        )

        output_placeholder = st.empty()

        if translate_btn and user_input.strip():
            with st.spinner("Translating on CPU..."):
                start_t = time.time()
                trans_result = translator.translate(user_input)
                raw_sindhi = trans_result["sindhi"]
                
                # Diacritic restoration
                diacritic_result = diacritic_engine.restore(raw_sindhi)
                marked_sindhi = diacritic_result["marked"]
                total_latency = (time.time() - start_t) * 1000.0

                # Cache in session state
                st.session_state["latest_english"] = user_input
                st.session_state["latest_sindhi"] = raw_sindhi
                st.session_state["latest_marked"] = marked_sindhi
                st.session_state["latest_tokens"] = diacritic_result["tokens"]
                st.session_state["latest_latency"] = total_latency

        elif restore_only_btn and user_input.strip():
            # If user entered Sindhi directly in input to restore marks
            with st.spinner("Restoring Sindhi vowel marks..."):
                diacritic_result = diacritic_engine.restore(user_input)
                st.session_state["latest_english"] = "(Direct Sindhi Input)"
                st.session_state["latest_sindhi"] = user_input
                st.session_state["latest_marked"] = diacritic_result["marked"]
                st.session_state["latest_tokens"] = diacritic_result["tokens"]
                st.session_state["latest_latency"] = 10.0

        # Display result
        if "latest_sindhi" in st.session_state:
            active_text = (
                st.session_state["latest_marked"]
                if "Learning" in mode
                else st.session_state["latest_sindhi"]
            )

            output_placeholder.markdown(
                f'<div class="sindhi-text">{active_text}</div>',
                unsafe_allow_html=True,
            )

            st.caption(f"⚡ Translation Latency: {st.session_state.get('latest_latency', 0):.1f} ms | Words: {len(active_text.split())}")

            # Copy button simulation
            st.text_area("Selectable text for copying:", value=active_text, height=70)


with tab_learning:
    st.subheader("🎓 Educational Learning & Pronunciation Guidance")
    st.info("Learning mode displays vowel markings (Zabar َ, Zer ِ, Pesh ُ) and distinguishes linguistically verified pronunciations from statistical predictions.")

    if "latest_tokens" in st.session_state and st.session_state["latest_tokens"]:
        st.markdown("### Word-by-Word Decomposition")
        
        cols = st.columns(min(4, len(st.session_state["latest_tokens"])))
        for idx, tok in enumerate(st.session_state["latest_tokens"]):
            c_idx = idx % 4
            chip_cls = "verified-chip" if tok["is_verified"] else "predicted-chip"
            badge_label = "Verified" if tok["is_verified"] else "ML Prediction"

            with cols[c_idx]:
                st.markdown(f"""
                <div class="diacritic-guide-card {chip_cls}">
                    <div style="direction: rtl; font-size: 1.6rem; font-family: 'Amiri', serif; font-weight: bold;">{tok['marked']}</div>
                    <div style="font-size: 0.85rem; color: #718096; margin-top: 4px;">Plain: <span style="direction: rtl;">{tok['plain']}</span></div>
                    <div style="font-size: 0.75rem; color: {'#276749' if tok['is_verified'] else '#2b6cb0'}; font-weight: 600;">Status: {badge_label}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.write("Translate a sentence above to view interactive word-by-word pronunciation guidance.")


with tab_dictionary:
    st.subheader("📖 Searchable Educational Sindhi Vocabulary")
    st.write("Browse verified Sindhi words with verified vowel markings and phonological annotations.")

    sample_diac_path = config.SAMPLE_DIACRITICS_FILE
    if sample_diac_path.exists():
        vocab_df = pd.read_csv(sample_diac_path)
        search_query = st.text_input("🔍 Search vocabulary (by Sindhi plain text, marked text, or meaning):")
        
        if search_query:
            filtered_df = vocab_df[
                vocab_df["plain"].str.contains(search_query, na=False)
                | vocab_df["marked"].str.contains(search_query, na=False)
                | vocab_df["notes"].str.contains(search_query, na=False, case=False)
            ]
        else:
            filtered_df = vocab_df

        st.dataframe(
            filtered_df[["plain", "marked", "notes", "source", "review_status"]],
            use_container_width=True,
            column_config={
                "plain": "Unmarked Sindhi",
                "marked": "Marked Sindhi (َ ِ ُ)",
                "notes": "Meaning / Pronunciation Note",
                "source": "Linguistic Source",
                "review_status": "Review Status",
            },
        )
    else:
        st.warning("Sample vocabulary file not found.")


with tab_benchmarks:
    st.subheader("📊 Offline Benchmark Reports & Academic Evaluation")
    
    col_rep1, col_rep2 = st.columns(2)

    with col_rep1:
        st.markdown("#### 🎯 Diacritic Restoration Evaluation")
        diac_rep_path = config.REPORTS_DIR / "diacritic_evaluation_report.json"
        if diac_rep_path.exists():
            with open(diac_rep_path, "r", encoding="utf-8") as f:
                d_rep = json.load(f)
            st.metric("Macro F1", f"{d_rep.get('macro_f1', 0.0):.4f}")
            st.metric("Existing Mark Preservation", f"{d_rep.get('existing_mark_preservation_accuracy', 0.0) * 100:.1f}%")
            st.metric("Exact Sentence Accuracy", f"{d_rep.get('exact_sentence_accuracy', 0.0) * 100:.1f}%")
            st.json(d_rep.get("per_diacritic_metrics", {}))
        else:
            st.info("Run `python -m src.diacritics.evaluate_diacritics` to view evaluation.")

    with col_rep2:
        st.markdown("#### ⚡ Hardware & Memory Constraints (8 GB CPU)")
        bench_rep_path = config.REPORTS_DIR / "benchmark_report.json"
        if bench_rep_path.exists():
            with open(bench_rep_path, "r", encoding="utf-8") as f:
                b_rep = json.load(f)
            st.metric("Cold Start Latency", f"{b_rep.get('cold_start_time_sec', 0.0):.2f} s")
            st.metric("Model RAM Footprint", f"{b_rep.get('memory_profile_mb', {}).get('model_memory_footprint_mb', 0.0):.1f} MB")
            st.metric("Average Throughput", f"{b_rep.get('performance', {}).get('average_throughput_tokens_per_sec', 0.0):.1f} tok/s")
        else:
            st.info("Run `python -m src.evaluation.benchmark_inference` to generate hardware benchmarks.")


with tab_feedback:
    st.subheader("📝 User Feedback & Human Evaluation")
    st.write("Submit human corrections for translations or vowel diacritics to help improve future model iterations.")

    with st.form("feedback_form"):
        fb_english = st.text_input("Source English Text:", value=st.session_state.get("latest_english", ""))
        fb_sindhi_predicted = st.text_input("Predicted Sindhi Text:", value=st.session_state.get("latest_marked", ""))
        fb_correction = st.text_input("Corrected Sindhi Text (with vowel marks if known):")
        fb_rating = st.slider("Translation Quality Rating:", min_value=1, max_value=5, value=4)
        fb_notes = st.text_area("Linguistic comments or notes:")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            feedback_file = config.DATA_DIR / "user_feedback.jsonl"
            fb_entry = {
                "timestamp": time.time(),
                "english": fb_english,
                "predicted": fb_sindhi_predicted,
                "corrected": fb_correction,
                "rating": fb_rating,
                "notes": fb_notes,
            }
            with open(feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(fb_entry, ensure_ascii=False) + "\n")
            st.success("Thank you! Feedback saved locally for academic research review.")
