"""
Streamlit Web Application: Offline English-to-Sindhi AI Translation & Diacritic Restoration System
Designed for local, CPU-only 8 GB RAM execution with dual-mode (Standard & Educational Learning Mode).

UI revision: refined visual identity, responsive layout, theme-aware (light/dark) styling.
All application logic and features are unchanged.
"""

import os
import time
import json
import html
from pathlib import Path
import streamlit as st
import pandas as pd
import psutil
from streamlit.errors import StreamlitSecretNotFoundError

import config
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer
from src.diacritics.word_analysis import VOWEL_GUIDE, analyze_word
from src.speech.providers import get_verified_remote_sindhi_provider


def setting_value(name: str):
    if name in os.environ:
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None


st.set_page_config(
    page_title="Sindhi AI - English to Sindhi Translator",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

REMOTE_MODE = str(setting_value("SINDHI_PROVIDER_MODE") or "").lower() == "cloudflare"

if REMOTE_MODE:
    from src.translation.cloudflare_provider import (
        CloudflareWorkersAI,
        SUPPORTED_LANGUAGES,
    )

    translator = CloudflareWorkersAI(
        setting_value("CLOUDFLARE_ACCOUNT_ID"),
        setting_value("CLOUDFLARE_API_TOKEN"),
        free_plan_confirmed=(
            str(setting_value("CLOUDFLARE_FREE_PLAN_CONFIRMED") or "").lower()
            == "true"
        ),
    )
else:
    from src.translation.translator import SindhiTranslator
    from src.translation.model_loader import TranslationModelLoader
    from src.speech.canopy_asr import recording_consent_key, transcribe_sindhi_audio
    from src.speech.memory_guard import available_ram_bytes, minimum_available_ram_bytes
    from src.speech.offline_tts import synthesize_english_speech


# ---------------------------------------------------------------------------
# Design system (theme-aware, responsive)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
    --sa-primary: #0f766e;
    --sa-primary-dark: #115e59;
    --sa-accent: #d97706;
    --sa-verified: #16a34a;
    --sa-predicted: #2563eb;
    --sa-surface: rgba(127, 127, 127, 0.07);
    --sa-surface-strong: rgba(127, 127, 127, 0.12);
    --sa-border: rgba(127, 127, 127, 0.25);
    --sa-muted: rgba(127, 127, 127, 0.95);
    --sa-radius: 16px;
    --sa-shadow: 0 6px 24px rgba(15, 118, 110, 0.10);
    --sa-sindhi-font: 'Amiri', 'Noto Naskh Arabic', 'Lateef', 'Scheherazade New', 'Segoe UI Historic', Tahoma, serif;
}

html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, .stButton {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}

/* Layout */
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ---------- Hero ---------- */
.sa-hero {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.4rem;
    color: #ffffff;
    background:
        radial-gradient(circle at 85% 20%, rgba(251, 191, 36, 0.35), transparent 45%),
        radial-gradient(circle at 10% 90%, rgba(45, 212, 191, 0.30), transparent 50%),
        linear-gradient(135deg, #0b3b3a 0%, #0f766e 55%, #115e59 100%);
    box-shadow: 0 12px 36px rgba(11, 59, 58, 0.30);
}
.sa-hero::after {
    content: "سنڌي";
    position: absolute;
    right: 1.6rem;
    bottom: -1.6rem;
    font-family: var(--sa-sindhi-font);
    font-size: 8.5rem;
    line-height: 1;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.08);
    pointer-events: none;
}
.sa-hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
    margin-bottom: 0.9rem;
}
.sa-hero h1 {
    color: #ffffff !important;
    font-size: clamp(1.5rem, 3.2vw, 2.3rem);
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 0.5rem 0;
    padding: 0;
}
.sa-hero p {
    color: rgba(255, 255, 255, 0.88);
    font-size: clamp(0.92rem, 1.4vw, 1.05rem);
    max-width: 720px;
    margin: 0 0 1.1rem 0;
}
.sa-hero-badges { display: flex; flex-wrap: wrap; gap: 8px; position: relative; z-index: 1; }
.sa-hero-badge {
    font-size: 0.8rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(4px);
}

/* ---------- Section headings ---------- */
.sa-section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
}
.sa-section-title .sa-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: linear-gradient(135deg, var(--sa-primary), var(--sa-accent));
}
.sa-section-sub { color: var(--sa-muted); font-size: 0.88rem; margin-bottom: 0.8rem; }

/* ---------- Sindhi output ---------- */
.sindhi-text {
    font-family: var(--sa-sindhi-font);
    direction: rtl;
    text-align: right;
    font-size: clamp(1.35rem, 2.4vw, 1.75rem);
    line-height: 2.3;
    padding: 1.3rem 1.4rem;
    border-radius: var(--sa-radius);
    border: 1px solid var(--sa-border);
    border-right: 5px solid var(--sa-primary);
    background: var(--sa-surface);
    box-shadow: var(--sa-shadow);
    min-height: 140px;
    word-wrap: break-word;
    overflow-wrap: anywhere;
}
.sa-diacritic-added {
    background: rgba(22, 163, 74, 0.24);
    border-radius: 5px;
    box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.24);
}
.sa-empty {
    text-align: center;
    padding: 2.2rem 1rem;
    border: 2px dashed var(--sa-border);
    border-radius: var(--sa-radius);
    color: var(--sa-muted);
    background: var(--sa-surface);
}
.sa-empty .sa-empty-icon { font-size: 2rem; margin-bottom: 0.3rem; }

/* ---------- Pills / stats ---------- */
.sa-stats { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.6rem 0 0.4rem 0; }
.sa-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.8rem; font-weight: 600;
    padding: 5px 12px; border-radius: 999px;
    background: var(--sa-surface-strong);
    border: 1px solid var(--sa-border);
}
.sa-pill.green  { background: rgba(22, 163, 74, 0.14);  border-color: rgba(22, 163, 74, 0.40);  color: #15803d; }
.sa-pill.amber  { background: rgba(217, 119, 6, 0.14);  border-color: rgba(217, 119, 6, 0.40);  color: #b45309; }
.sa-pill.teal   { background: rgba(15, 118, 110, 0.14); border-color: rgba(15, 118, 110, 0.40); color: #0f766e; }
.sa-pill.blue   { background: rgba(37, 99, 235, 0.14);  border-color: rgba(37, 99, 235, 0.40);  color: #1d4ed8; }

/* Legacy badge classes kept for compatibility */
.badge-status { display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 0.82rem; font-weight: 600; margin: 2px 4px 2px 0; }
.badge-offline { background-color: rgba(22, 163, 74, 0.16); color: #15803d; border: 1px solid rgba(22, 163, 74, 0.4); }
.badge-cpu { background-color: rgba(217, 119, 6, 0.16); color: #b45309; border: 1px solid rgba(217, 119, 6, 0.4); }

/* ---------- Sidebar ---------- */
.sa-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 0.4rem; }
.sa-brand-logo {
    width: 44px; height: 44px; border-radius: 12px;
    display: grid; place-items: center;
    font-family: var(--sa-sindhi-font); font-size: 1.7rem; font-weight: 700; color: #fff;
    background: linear-gradient(135deg, var(--sa-primary), var(--sa-accent));
    box-shadow: 0 4px 14px rgba(15, 118, 110, 0.35);
}
.sa-brand-name { font-weight: 800; font-size: 1.05rem; line-height: 1.1; }
.sa-brand-tag { font-size: 0.75rem; color: var(--sa-muted); }
.sa-spec {
    padding: 10px 12px; border-radius: 12px; margin-bottom: 8px;
    background: var(--sa-surface); border: 1px solid var(--sa-border);
}
.sa-spec-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--sa-muted); }
.sa-spec-value { font-size: 0.88rem; font-weight: 600; margin-top: 2px; }
.sa-metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
.sa-metric {
    padding: 10px 12px; border-radius: 12px;
    background: var(--sa-surface); border: 1px solid var(--sa-border);
}
.sa-metric-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--sa-muted); }
.sa-metric-value { font-size: 1.05rem; font-weight: 800; }

.sa-vowel {
    display: flex; align-items: center; gap: 12px;
    padding: 9px 12px; border-radius: 12px; margin-bottom: 8px;
    background: var(--sa-surface); border: 1px solid var(--sa-border);
}
.sa-vowel-glyph {
    flex: 0 0 auto;
    width: 40px; height: 40px; border-radius: 10px;
    display: grid; place-items: center;
    font-family: var(--sa-sindhi-font); font-size: 1.6rem;
    background: rgba(15, 118, 110, 0.14); color: var(--sa-primary);
}
.sa-vowel-title { font-weight: 700; font-size: 0.88rem; }
.sa-vowel-desc { font-size: 0.76rem; color: var(--sa-muted); }

/* ---------- Learning mode cards ---------- */
.sa-legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 0.4rem 0 1rem 0; }
.sa-word-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 14px;
}
.diacritic-guide-card {
    position: relative;
    padding: 14px 16px;
    border-radius: 14px;
    background: var(--sa-surface);
    border: 1px solid var(--sa-border);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.diacritic-guide-card:hover { transform: translateY(-3px); box-shadow: var(--sa-shadow); }
.diacritic-guide-card.verified-chip  { border-top: 4px solid var(--sa-verified); }
.diacritic-guide-card.predicted-chip { border-top: 4px solid var(--sa-predicted); }
.sa-word-index { position: absolute; top: 8px; left: 12px; font-size: 0.68rem; font-weight: 700; color: var(--sa-muted); }
.sa-word-marked {
    direction: rtl; text-align: center;
    font-family: var(--sa-sindhi-font); font-size: 2rem; font-weight: 700; line-height: 1.8;
    margin-top: 6px;
}
.sa-word-plain { font-size: 0.82rem; color: var(--sa-muted); text-align: center; }
.sa-word-plain span { direction: rtl; font-family: var(--sa-sindhi-font); font-size: 1.05rem; }
.sa-word-status {
    display: block; width: fit-content; margin: 10px auto 0 auto;
    font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;
}
.sa-word-status.verified  { background: rgba(22, 163, 74, 0.16);  color: #15803d; }
.sa-word-status.predicted { background: rgba(37, 99, 235, 0.16);  color: #1d4ed8; }
.sa-word-status.uncertain { background: rgba(217, 119, 6, 0.16); color: #b45309; }

/* Legacy chip classes kept for compatibility */
.sindhi-word-chip {
    display: inline-block; padding: 4px 10px; margin: 4px; border-radius: 6px;
    background: var(--sa-surface-strong); font-family: var(--sa-sindhi-font);
    font-size: 1.3rem; direction: rtl;
}

/* ---------- Native widget polish ---------- */
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600; font-size: 0.95rem; padding: 0.6rem 1rem; border-radius: 10px 10px 0 0;
}
div[data-testid="stTabs"] [role="tablist"] { flex-wrap: wrap; gap: 2px; }
div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
    border-radius: 12px;
}
div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
    border-radius: 12px; font-weight: 600; transition: transform 0.12s ease, box-shadow 0.12s ease;
}
div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px); box-shadow: 0 6px 16px rgba(15, 118, 110, 0.25);
}
div[data-testid="stButton"] button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--sa-primary), var(--sa-primary-dark)); border: none; color: #fff;
}
div[data-testid="stMetric"] {
    padding: 14px 16px; border-radius: 14px;
    background: var(--sa-surface); border: 1px solid var(--sa-border);
}
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--sa-radius); }

.sa-footer {
    margin-top: 2.5rem; padding-top: 1rem; text-align: center;
    border-top: 1px solid var(--sa-border); color: var(--sa-muted); font-size: 0.8rem;
}

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .sa-hero { padding: 1.4rem 1.3rem; border-radius: 18px; }
    .sa-hero::after { font-size: 6rem; }
}
@media (max-width: 640px) {
    .sa-hero::after { display: none; }
    .sa-word-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .sa-word-marked { font-size: 1.6rem; }
    div[data-testid="stTabs"] button[role="tab"] { padding: 0.5rem 0.7rem; font-size: 0.85rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def esc(text) -> str:
    """HTML-escape arbitrary text and keep line breaks."""
    return html.escape(str(text)).replace("\n", "<br>")


def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="sa-section-sub">{esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sa-section-title"><span class="sa-dot"></span>{esc(title)}</div>{sub}',
        unsafe_allow_html=True,
    )


def highlighted_restoration_html(result) -> str:
    """Renders suggested vowel additions with escaped, marked-up text."""
    chunks = []
    for token in result.get("tokens", []):
        chunks.append(esc(token.get("separator", "")).replace("\n", "<br>"))
        for segment in token.get("segments", []):
            safe_text = esc(segment["suggested"]).replace("\n", "<br>")
            if segment.get("changed"):
                safe_text = f'<span class="sa-diacritic-added">{safe_text}</span>'
            chunks.append(safe_text)
    chunks.append(esc(result.get("trailing_text", "")).replace("\n", "<br>"))
    return "".join(chunks)


def activate_translation(text: str, engine: SindhiDiacriticRestorer) -> None:
    """Updates the active translation and its separate learning-mode restoration."""
    result = engine.restore(text)
    st.session_state["latest_sindhi"] = text
    st.session_state["latest_marked"] = result["marked"]
    st.session_state["latest_tokens"] = result["tokens"]


def clear_sindhi_to_english_translation() -> None:
    """Clears only the Sindhi-to-English workspace."""
    st.session_state["sindhi_to_english_input"] = ""
    st.session_state["sindhi_to_english_output"] = ""
    st.session_state.pop("sindhi_to_english_latency", None)


def clear_voice_workspace() -> None:
    """Clears voice workflow text and generated speech without changing other tabs."""
    for key in (
        "voice_sindhi_text",
        "voice_english_output",
        "voice_english_latency",
        "voice_english_audio",
        "voice_asr_latency_ms",
        "voice_asr_peak_rss_mb",
        "voice_asr_script_detected",
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith("voice_offline_consent_"):
            del st.session_state[key]


def reset_diacritic_state() -> None:
    """Clears only state owned by the diacritic intelligence section."""
    for key in list(st.session_state.keys()):
        if key.startswith("diacritic_choice_"):
            del st.session_state[key]
    st.session_state["diacritic_input"] = ""
    for key in ("diacritic_result", "diacritic_editor", "diacritic_elapsed_ms"):
        st.session_state.pop(key, None)


def clear_main_translation_workspace() -> None:
    st.session_state["main_translation_input"] = ""
    for key in (
        "latest_english",
        "latest_sindhi",
        "latest_marked",
        "latest_tokens",
        "latest_latency",
        "latest_target_language",
        "latest_output_editor",
        "latest_translation_candidates",
        "preferred_translation_editor",
    ):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


if not REMOTE_MODE:
    @st.cache_resource(show_spinner="Initializing Translation Engine...")
    def get_translator():
        return SindhiTranslator(lazy_init=True)


@st.cache_resource(show_spinner="Initializing Diacritic Restoration Model...")
def get_diacritic_engine():
    return SindhiDiacriticRestorer()


if not REMOTE_MODE:
    translator = get_translator()
diacritic_engine = get_diacritic_engine()


# ---------------------------------------------------------------------------
# Sidebar: Navigation & Diagnostics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sa-brand">'
        '<div class="sa-brand-logo">س</div>'
        '<div><div class="sa-brand-name">Sindhi AI</div>'
        '<div class="sa-brand-tag">Offline Translator &amp; Vowel Restorer</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if REMOTE_MODE:
        st.markdown(
            '<span class="badge-status badge-cpu">☁️ Cloud translation</span>'
            '<span class="badge-status badge-cpu">🔒 No paid fallback</span>',
            unsafe_allow_html=True,
        )
        st.info(
            "In cloud mode, text is sent to Cloudflare Workers AI only after you select "
            "Translate. Recordings are not collected or sent."
        )
    else:
        st.markdown(
            '<span class="badge-status badge-offline">🟢 100% Offline Capable</span>'
            '<span class="badge-status badge-cpu">💻 CPU-Optimized</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### 🧠 Model Specifications")
    if REMOTE_MODE:
        st.markdown(
            '<div class="sa-spec"><div class="sa-spec-label">Translation model</div>'
            '<div class="sa-spec-value">Cloudflare Workers AI · Meta M2M100 1.2B</div></div>'
            '<div class="sa-spec"><div class="sa-spec-label">Supported languages</div>'
            '<div class="sa-spec-value">M2M100 listed languages, including Sindhi</div></div>'
            '<div class="sa-spec"><div class="sa-spec-label">Diacritic Restorer</div>'
            '<div class="sa-spec-value">Local classifier (when packaged) + reviewed lexicon</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sa-spec"><div class="sa-spec-label">NMT Model</div>'
            '<div class="sa-spec-value">NLLB-200 Distilled (600M)</div></div>'
            '<div class="sa-spec"><div class="sa-spec-label">Target Code</div>'
            '<div class="sa-spec-value"><code>snd_Arab</code> · Perso-Arabic Sindhi</div></div>'
            '<div class="sa-spec"><div class="sa-spec-label">Diacritic Restorer</div>'
            '<div class="sa-spec-value">Scikit-Learn Contextual Sequence Classifier + Verified Lexicon</div></div>',
            unsafe_allow_html=True,
        )

    # Live Diagnostics
    st.markdown("##### 📡 Live Diagnostics")
    if REMOTE_MODE:
        st.caption(
            "Local NLLB and Canopy weights are not imported or loaded in cloud mode."
        )
        if not translator.configured:
            st.error(
                "Cloudflare credentials are missing or the operator has not confirmed "
                "the account is on the Free plan with billing disabled."
            )
    else:
        status = TranslationModelLoader.get_status()
        st.markdown(
            '<div class="sa-metric-grid">'
            f'<div class="sa-metric"><div class="sa-metric-label">Process RAM</div>'
            f'<div class="sa-metric-value">{status["process_rss_mb"]:.1f} MB</div></div>'
            f'<div class="sa-metric"><div class="sa-metric-label">CPU Threads</div>'
            f'<div class="sa-metric-value">{status["cpu_threads"]}</div></div>'
            "</div>"
            f'<div class="sa-spec"><div class="sa-spec-label">Local Weights Loaded</div>'
            f'<div class="sa-spec-value">{"✅ Yes" if status["loaded"] else "⏳ On-Demand (Lazy)"}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### 📚 Sindhi Vowel Quick Guide")
    st.markdown(
        '<div class="sa-vowel"><div class="sa-vowel-glyph">&#1614;</div><div>'
        '<div class="sa-vowel-title">Zabar · U+064E</div>'
        '<div class="sa-vowel-desc">Short vowel /a/ (e.g. مَدَدَ - madad)</div></div></div>'
        '<div class="sa-vowel"><div class="sa-vowel-glyph">&#1616;</div><div>'
        '<div class="sa-vowel-title">Zer · U+0650</div>'
        '<div class="sa-vowel-desc">Short vowel /i/ (e.g. عِلمَ - ilm, ڪِتابَ - kitab)</div></div></div>'
        '<div class="sa-vowel"><div class="sa-vowel-glyph">&#1615;</div><div>'
        '<div class="sa-vowel-title">Pesh · U+064F</div>'
        '<div class="sa-vowel-desc">Short vowel /u/ (e.g. پُلِ - pul, اُستادَ - ustad)</div></div></div>'
        '<div class="sa-vowel"><div class="sa-vowel-glyph">&#1618;</div><div>'
        '<div class="sa-vowel-title">Jazm / Sukun</div>'
        '<div class="sa-vowel-desc">Vowelless consonant marker</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if not REMOTE_MODE:
        if st.button("🗑️ Free Model Memory (Unload)", use_container_width=True):
            TranslationModelLoader.unload()
            st.success("Translation model unloaded from RAM!")


# ---------------------------------------------------------------------------
# Main Application Interface
# ---------------------------------------------------------------------------
hero_description = (
    "Cloud-hosted translation with local Sindhi diacritic tools. Submitted text is sent "
    "to Cloudflare Workers AI; audio is not collected."
    if REMOTE_MODE
    else "Offline, privacy-preserving AI translation with Perso-Arabic script orthography and phonological vowel learning."
)
hero_privacy_badge = (
    '<span class="sa-hero-badge">☁️ Cloud inference · text leaves this app</span>'
    if REMOTE_MODE
    else '<span class="sa-hero-badge">🔒 Private &amp; Offline</span>'
)
st.markdown(
    f"""
<div class="sa-hero">
    <span class="sa-hero-eyebrow">Neural Machine Translation · Sindhi NLP</span>
    <h1>English to Sindhi Neural Translator &amp; Diacritic Restorer</h1>
    <p>{html.escape(hero_description)}</p>
    <div class="sa-hero-badges">
        {hero_privacy_badge}
        <span class="sa-hero-badge">{"☁️ Provider-hosted model" if REMOTE_MODE else "💻 Runs on 8 GB CPU"}</span>
        <span class="sa-hero-badge">🎓 Learning Mode</span>
        <span class="sa-hero-badge">📖 Verified Wordbook</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Mode Selector Tabs
tab_translate, tab_sindhi_to_english, tab_voice, tab_learning, tab_word_analysis, tab_dictionary, tab_benchmarks, tab_feedback = st.tabs([
    "🔄 Translate & Restore",
    "↔️ Sindhi to English",
    "🎙️ Sindhi Voice",
    "✨ Diacritic Intelligence",
    "🔊 Word Sound & Pronunciation",
    "📖 Verified Wordbook",
    "📊 Benchmarks & Research",
    "📝 User Feedback",
])

# ---------------------------------------------------------------------------
# Tab 1: Translate & Restore
# ---------------------------------------------------------------------------
with tab_translate:
    if REMOTE_MODE:
        st.caption(
            "Cloud translation sends the submitted text to Cloudflare Workers AI. "
            "Only M2M100's documented language list is offered here; Sindhi output quality "
            "and orthography still require human review."
        )
        language_columns = st.columns([1, 0.25, 1])
        with language_columns[0]:
            source_language = st.selectbox(
                "Source language",
                list(SUPPORTED_LANGUAGES),
                index=list(SUPPORTED_LANGUAGES).index("English"),
                key="cloud_source_language",
            )
        with language_columns[1]:
            st.button(
                "⇄",
                key="swap_cloud_languages",
                help="Swap source and target languages",
                on_click=lambda: st.session_state.update({
                    "cloud_source_language": st.session_state.get(
                        "cloud_target_language", "Sindhi"
                    ),
                    "cloud_target_language": st.session_state.get(
                        "cloud_source_language", "English"
                    ),
                }),
            )
        with language_columns[2]:
            target_language = st.selectbox(
                "Target language",
                list(SUPPORTED_LANGUAGES),
                index=list(SUPPORTED_LANGUAGES).index("Sindhi"),
                key="cloud_target_language",
            )
    else:
        source_language = "English"
        target_language = "Sindhi"

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        with st.container(border=True):
            section_header(
                f"{source_language} Input",
                "Type or paste the text you want to translate.",
            )
            default_prompt = (
                "I am learning Sindhi. Books are our best friends."
                if source_language == "English"
                else ""
            )
            if "main_translation_input" not in st.session_state:
                st.session_state["main_translation_input"] = default_prompt
            user_input = st.text_area(
                f"Enter {source_language} text (sentence, paragraph, or phrase):",
                key="main_translation_input",
                height=180,
                placeholder=f"Type {source_language} text here...",
                help="Text is sent to the configured translation provider after submission.",
            )

            char_cnt = len(user_input)
            word_cnt = len(user_input.split())
            st.markdown(
                f'<div class="sa-stats">'
                f'<span class="sa-pill">🔤 {char_cnt} characters</span>'
                f'<span class="sa-pill">📝 {word_cnt} words</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

            compare_alternatives = st.checkbox(
                "Compare alternative translations",
                value=False,
                disabled=REMOTE_MODE,
                help="Uses beam search to find up to three distinct candidates. Alternatives are suggestions, not verified translations.",
            )
            if REMOTE_MODE:
                st.caption(
                    "This remote model endpoint returns one result per request; it does not "
                    "provide verified alternative translations."
                )

            col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
            with col_b1:
                translate_btn = st.button(
                    f"🚀 Translate to {target_language}",
                    type="primary",
                    use_container_width=True,
                )
            with col_b2:
                restore_only_btn = st.button(
                    "✨ Restore Diacritics",
                    use_container_width=True,
                    disabled=REMOTE_MODE and source_language != "Sindhi",
                )
            with col_b3:
                st.button(
                    "🧹 Clear",
                    use_container_width=True,
                    on_click=clear_main_translation_workspace,
                )

    with col_output:
        with st.container(border=True):
            section_header(
                f"{target_language} Output",
                "Sindhi remains right-to-left; other target languages use standard text direction.",
            )

            if not REMOTE_MODE or target_language == "Sindhi":
                mode = st.radio(
                    "Display Mode:",
                    ["Standard Sindhi (Readable)", "Learning Mode (Phonological Vowels َ ِ ُ)"],
                    horizontal=True,
                )
            else:
                mode = "Standard Sindhi (Readable)"

            output_placeholder = st.empty()

            if translate_btn and user_input.strip():
                try:
                    with st.spinner(
                        "Translating with Cloudflare Workers AI..."
                        if REMOTE_MODE
                        else "Translating on CPU..."
                    ):
                        start_t = time.time()
                        if REMOTE_MODE:
                            trans_result = translator.translate(
                                user_input,
                                source_language,
                                target_language,
                            )
                            raw_sindhi = trans_result["translation"]
                            candidates = []
                        elif compare_alternatives:
                            candidate_result = translator.translate_candidates(
                                user_input,
                                num_candidates=3,
                                num_beams=4,
                            )
                            candidates = candidate_result["candidates"]
                            if not candidates:
                                st.error("The model did not return a usable Sindhi translation candidate.")
                                raw_sindhi = None
                            else:
                                st.session_state["latest_translation_candidates"] = candidates
                                st.session_state["preferred_translation_editor"] = candidates[0]
                                raw_sindhi = candidates[0]
                        else:
                            trans_result = translator.translate(user_input)
                            raw_sindhi = trans_result["sindhi"]
                            candidates = []
                            st.session_state.pop("latest_translation_candidates", None)
                            st.session_state.pop("preferred_translation_editor", None)

                        if raw_sindhi is not None:
                            if target_language == "Sindhi":
                                diacritic_result = diacritic_engine.restore(raw_sindhi)
                                marked_sindhi = diacritic_result["marked"]
                                tokens = diacritic_result["tokens"]
                            else:
                                marked_sindhi = raw_sindhi
                                tokens = []
                            total_latency = (time.time() - start_t) * 1000.0

                            st.session_state["latest_english"] = user_input
                            st.session_state["latest_sindhi"] = raw_sindhi
                            st.session_state["latest_marked"] = marked_sindhi
                            st.session_state["latest_tokens"] = tokens
                            st.session_state["latest_latency"] = total_latency
                            st.session_state["latest_target_language"] = target_language
                            st.session_state["latest_output_editor"] = raw_sindhi
                            st.session_state.pop("latest_translation_candidates", None)
                            if (
                                REMOTE_MODE
                                and target_language == "Sindhi"
                                and not any(
                                    "\u0600" <= character <= "\u06ff"
                                    and character.isalpha()
                                    for character in raw_sindhi
                                )
                            ):
                                st.warning(
                                    "The output did not contain recognizable Arabic-derived "
                                    "script letters. Review or edit it; script detection does "
                                    "not establish translation correctness."
                                )
                except (RuntimeError, ValueError) as error:
                    st.error(f"Translation failed: {error}")

            elif restore_only_btn and user_input.strip():
                if REMOTE_MODE and source_language != "Sindhi":
                    st.warning("Diacritic restoration is only available for Sindhi text.")
                else:
                    with st.spinner("Restoring Sindhi vowel marks..."):
                        diacritic_result = diacritic_engine.restore(user_input)
                        st.session_state["latest_english"] = "(Direct Sindhi Input)"
                        st.session_state["latest_sindhi"] = user_input
                        st.session_state["latest_marked"] = diacritic_result["marked"]
                        st.session_state["latest_tokens"] = diacritic_result["tokens"]
                        st.session_state["latest_latency"] = 10.0
                        st.session_state["latest_target_language"] = "Sindhi"
                        st.session_state["latest_output_editor"] = user_input

            # Display result
            if "latest_sindhi" in st.session_state:
                active_text = (
                    st.session_state["latest_marked"]
                    if "Learning" in mode
                    and st.session_state.get("latest_target_language", "Sindhi") == "Sindhi"
                    else st.session_state["latest_sindhi"]
                )

                if st.session_state.get("latest_target_language", "Sindhi") == "Sindhi":
                    output_placeholder.markdown(
                        f'<div class="sindhi-text">{esc(active_text)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    output_placeholder.write(active_text)
                if REMOTE_MODE:
                    st.caption(
                        "No calibrated translation confidence score is provided by this endpoint."
                    )

                st.markdown(
                    f'<div class="sa-stats">'
                    f'<span class="sa-pill amber">⚡ {st.session_state.get("latest_latency", 0):.1f} ms latency</span>'
                    f'<span class="sa-pill teal">📝 {len(active_text.split())} words</span>'
                    f'<span class="sa-pill green">{"☁️ Cloud" if REMOTE_MODE else "🟢 Offline"}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.text_area(
                    "Edit the final translation",
                    key="latest_output_editor",
                    height=80,
                )
                st.code(st.session_state["latest_output_editor"], language=None)

                candidates = st.session_state.get("latest_translation_candidates", [])
                if candidates:
                    st.markdown("#### Translation comparison")
                    st.caption(
                        "Candidates are beam-search suggestions ranked by the model, not guaranteed-correct "
                        "translations. Diacritic restoration remains a separate operation."
                    )
                    candidate_columns = st.columns(len(candidates))
                    for index, (candidate, candidate_column) in enumerate(
                        zip(candidates, candidate_columns), start=1
                    ):
                        with candidate_column:
                            st.markdown(f"**Candidate {index}**")
                            st.markdown(
                                f'<div class="sindhi-text">{esc(candidate)}</div>',
                                unsafe_allow_html=True,
                            )
                            if st.button(
                                "Use this translation",
                                key=f"select_translation_candidate_{index}",
                                use_container_width=True,
                            ):
                                st.session_state["preferred_translation_editor"] = candidate
                                activate_translation(candidate, diacritic_engine)
                                st.rerun()

                    if len(candidates) == 1:
                        st.info("The model returned only one distinct candidate; no distinct alternative was available.")
                    st.text_area(
                        "Edit your preferred Sindhi translation",
                        key="preferred_translation_editor",
                        height=100,
                    )
                    if st.button("Use edited translation", key="apply_preferred_translation"):
                        activate_translation(
                            st.session_state["preferred_translation_editor"],
                            diacritic_engine,
                        )
                        st.rerun()
                    st.markdown("Copy-ready preferred translation")
                    st.code(st.session_state["preferred_translation_editor"], language=None)
            else:
                output_placeholder.markdown(
                    '<div class="sa-empty"><div class="sa-empty-icon">✍️</div>'
                    "<div><b>Your Sindhi translation will appear here</b></div>"
                    "<div style='font-size:0.85rem'>Enter English text and press “Translate to Sindhi”.</div></div>",
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Tab 2: Sindhi-to-English Translation
# ---------------------------------------------------------------------------
with tab_sindhi_to_english:
    section_header(
        "Sindhi to English Translation",
        "Translate editable Sindhi text with the configured provider.",
    )
    if REMOTE_MODE:
        st.caption(
            "Submitting sends this text to Cloudflare Workers AI. Results are machine-generated "
            "suggestions and may need review; audio is not used."
        )
    else:
        st.caption(
            "Text translation only. Results are machine-generated suggestions and may need review; "
            "this feature does not use microphone input or audio."
        )

    if "sindhi_to_english_output" not in st.session_state:
        st.session_state["sindhi_to_english_output"] = ""

    with st.form("sindhi_to_english_form"):
        sindhi_to_translate = st.text_area(
            "Sindhi input (سنڌي)",
            key="sindhi_to_english_input",
            height=150,
            placeholder="سنڌي جملو هتي لکو يا پيسٽ ڪريو",
            help="Edit the Sindhi text here, then submit again to translate your changes.",
        )
        translate_to_english = st.form_submit_button(
            "Translate to English",
            type="primary",
            use_container_width=True,
        )

    st.button(
        "Clear Sindhi-to-English text",
        key="clear_sindhi_to_english",
        on_click=clear_sindhi_to_english_translation,
    )

    if translate_to_english:
        if not sindhi_to_translate.strip():
            st.warning("Enter Sindhi text before translating.")
        else:
            try:
                with st.spinner(
                    "Translating with Cloudflare Workers AI..."
                    if REMOTE_MODE
                    else "Translating Sindhi to English on CPU..."
                ):
                    if REMOTE_MODE:
                        reverse_result = translator.translate(
                            sindhi_to_translate,
                            "Sindhi",
                            "English",
                        )
                        english_text = reverse_result["translation"]
                    else:
                        reverse_result = translator.translate_sindhi_to_english(
                            sindhi_to_translate
                        )
                        english_text = reverse_result["english"]
                st.session_state["sindhi_to_english_output"] = english_text
                st.session_state["sindhi_to_english_latency"] = reverse_result["latency_ms"]
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                st.error(f"Sindhi-to-English translation failed: {error}")

    st.text_area(
        "English translation (editable)",
        key="sindhi_to_english_output",
        height=150,
        placeholder="The English translation will appear here.",
    )
    if "sindhi_to_english_latency" in st.session_state:
        st.caption(
            f"Translation time: {st.session_state['sindhi_to_english_latency']:.1f} ms"
        )

# ---------------------------------------------------------------------------
# Tab 3: Sindhi voice workflow
# ---------------------------------------------------------------------------
with tab_voice:
    if REMOTE_MODE:
        section_header(
            "Sindhi Voice Input",
            "Speech transcription is unavailable in the public deployment until a provider is validated for Sindhi and Perso-Arabic output.",
        )
        if get_verified_remote_sindhi_provider() is None:
            st.warning(
                "No audio is recorded or sent in cloud mode. Cloudflare offers a Whisper ASR "
                "endpoint, but available documentation does not establish reliable Sindhi "
                "transcription or Perso-Arabic script quality. It is intentionally not enabled."
            )
            st.info(
                "A provider interface is in place for a future verified Sindhi ASR backend. "
                "Use the text translator to enter or paste Sindhi manually. Canopy-M remains "
                "available only in the local app and is still protected by its RAM guard."
            )
        else:
            st.error("A speech provider is available but has no user interface configured.")
        st.caption(
            "Hosted English speech playback is unavailable: the existing offline SAPI voice "
            "runs only on Windows. The local Windows app retains that feature."
        )
    else:
        section_header(
            "Sindhi Voice Input",
            "Record speech, review the Sindhi text, translate it with the existing NLLB model, and play English speech locally.",
        )
        st.caption(
            "Offline-only recognition: recordings are processed by Canopy-M on the machine running "
            "Streamlit and are never sent to an online speech-recognition service. Transcription is "
            "an unverified suggestion; review and edit it before translation."
        )

        recording = st.audio_input(
            "Record Sindhi speech",
            sample_rate=16000,
            key="voice_audio_input",
            help="The recording is processed locally and requires consent for each recording.",
        )
        if recording is not None:
            st.audio(recording, format="audio/wav")
            audio_bytes = recording.getvalue()
            consent_key = recording_consent_key(audio_bytes)
            offline_consent = st.checkbox(
                "I consent to process this recording locally with Canopy-M.",
                key=consent_key,
            )
        else:
            audio_bytes = b""
            offline_consent = False

        minimum_asr_ram_bytes = minimum_available_ram_bytes()
        available_asr_ram_bytes = available_ram_bytes()
        minimum_asr_ram_gib = minimum_asr_ram_bytes / (1024 ** 3)
        available_asr_ram_gib = available_asr_ram_bytes / (1024 ** 3)

        st.caption(
            f"Canopy-M requires at least {minimum_asr_ram_gib:.1f} GiB currently available. "
            f"Current availability: {available_asr_ram_gib:.2f} GiB. "
            "The safety threshold is fixed and cannot be lowered."
        )
        asr_memory_available = available_asr_ram_bytes >= minimum_asr_ram_bytes
        if not asr_memory_available:
            st.error(
                "Offline Sindhi recognition is unavailable because memory is insufficient. "
                "Canopy-M will not be loaded or retried. Free memory and restart recognition, "
                "or type/paste Sindhi text below."
            )

        if st.button(
            "Transcribe Sindhi offline",
            key="transcribe_sindhi_voice",
            disabled=recording is None or not offline_consent or not asr_memory_available,
            type="primary",
            use_container_width=True,
        ):
            try:
                translator.model = None
                translator.tokenizer = None
                TranslationModelLoader.unload()
                with st.spinner("Recognizing Sindhi locally on CPU..."):
                    asr_result = transcribe_sindhi_audio(audio_bytes)
                st.session_state["voice_asr_latency_ms"] = asr_result["inference_ms"]
                st.session_state["voice_asr_peak_rss_mb"] = asr_result["peak_rss_mb"]

                st.session_state["voice_sindhi_text"] = asr_result["text"]
                st.session_state["voice_asr_script_detected"] = asr_result.get(
                    "script_looks_perso_arabic",
                    any(
                        "\u0600" <= character <= "\u06ff" and character.isalpha()
                        for character in asr_result["text"]
                    ),
                )
                st.session_state.pop("voice_english_output", None)
                st.session_state.pop("voice_english_audio", None)
            except (FileNotFoundError, OSError, PermissionError, RuntimeError, ValueError) as error:
                st.error(f"Sindhi transcription failed: {error}")

        if "voice_asr_latency_ms" in st.session_state:
            st.caption(
                "ASR CPU inference: "
                f"{st.session_state['voice_asr_latency_ms']:.1f} ms · "
                f"ASR peak process memory: {st.session_state['voice_asr_peak_rss_mb']:.1f} MB"
            )
        if "voice_sindhi_text" in st.session_state:
            if st.session_state.get("voice_asr_script_detected", False):
                st.info(
                    "This transcript contains Perso-Arabic script characters; that does not verify "
                    "Sindhi transcription correctness. Review and edit it below."
                )
            elif st.session_state["voice_sindhi_text"]:
                st.warning(
                    "The result does not appear to contain Sindhi Perso-Arabic letters. "
                    "Review or replace it manually before translating."
                )

        with st.form("voice_translation_form"):
            voice_sindhi_text = st.text_area(
                "Sindhi text to translate (editable)",
                key="voice_sindhi_text",
                height=140,
                placeholder="سنڌي متن هتي لکو يا رڪارڊنگ ٻڌي پاڻ ٽائيپ ڪريو",
                help="Review or correct this text, then submit again to translate your edits.",
            )
            translate_voice_text = st.form_submit_button(
                "Translate edited Sindhi to English",
                type="primary",
                use_container_width=True,
            )

        st.button(
            "Clear voice text and result",
            key="clear_voice_workspace",
            on_click=clear_voice_workspace,
        )

        if "voice_english_output" not in st.session_state:
            st.session_state["voice_english_output"] = ""

        if translate_voice_text:
            if not voice_sindhi_text.strip():
                st.warning("Enter or correct Sindhi text before translating.")
            else:
                available_gib = psutil.virtual_memory().available / (1024 ** 3)
                if translator.model is None and available_gib < 3.0:
                    st.warning(
                        "The local NLLB model is not loaded and there is not enough free memory "
                        "to load it safely. Free at least 3 GiB of RAM and try again. "
                        f"Currently available: {available_gib:.2f} GiB."
                    )
                else:
                    try:
                        with st.spinner("Translating Sindhi to English on CPU..."):
                            voice_result = translator.translate_sindhi_to_english(
                                voice_sindhi_text
                            )
                        st.session_state["voice_english_output"] = voice_result["english"]
                        st.session_state["voice_english_latency"] = voice_result["latency_ms"]
                        st.session_state.pop("voice_english_audio", None)
                    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                        st.error(f"Voice workflow translation failed: {error}")

        voice_english_output = st.text_area(
            "English translation (editable)",
            key="voice_english_output",
            height=140,
            placeholder="The English translation will appear here.",
        )
        if "voice_english_latency" in st.session_state:
            st.caption(
                f"Translation time: {st.session_state['voice_english_latency']:.1f} ms"
            )

        if st.button(
            "Generate offline English speech",
            key="generate_voice_english_audio",
            disabled=not voice_english_output.strip(),
        ):
            try:
                st.session_state["voice_english_audio"] = synthesize_english_speech(
                    voice_english_output
                )
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f"Offline English speech generation failed: {error}")

        if st.session_state.get("voice_english_audio"):
            st.audio(st.session_state["voice_english_audio"], format="audio/wav")

# ---------------------------------------------------------------------------
# Tab 4: Educational Learning Mode
# ---------------------------------------------------------------------------
with tab_learning:
    section_header(
        "Sindhi Diacritic Intelligence",
        "Compare original Sindhi with suggested vowel marks, inspect model scores, and keep or edit only the suggestions you want.",
    )
    st.caption(
        "Classifier probabilities are model scores, not calibrated guarantees of correctness. "
        "Sindhi spelling can be ambiguous across context and regional pronunciation."
    )

    if "diacritic_input" not in st.session_state:
        st.session_state["diacritic_input"] = st.session_state.get(
            "latest_output_editor",
            st.session_state.get("latest_sindhi", ""),
        )

    diacritic_source = st.text_area(
        "Sindhi text to analyze",
        key="diacritic_input",
        height=110,
        placeholder="سنڌي متن هتي لکو يا پيسٽ ڪريو",
        help="Existing vowel marks are preserved. Ambiguous lexicon entries are shown without an automatic choice.",
    )
    action_col, reset_col = st.columns([3, 1])
    with action_col:
        analyze_diacritics = st.button(
            "Analyze & Suggest Diacritics",
            type="primary",
            use_container_width=True,
            disabled=not diacritic_source.strip(),
        )
    with reset_col:
        st.button("Clear", use_container_width=True, on_click=reset_diacritic_state)

    if analyze_diacritics:
        start_time = time.perf_counter()
        result = diacritic_engine.restore(diacritic_source)
        st.session_state["diacritic_result"] = result
        st.session_state["diacritic_editor"] = result["marked"]
        st.session_state["diacritic_elapsed_ms"] = (time.perf_counter() - start_time) * 1000
        for key in list(st.session_state.keys()):
            if key.startswith("diacritic_choice_"):
                del st.session_state[key]

    diacritic_result = st.session_state.get("diacritic_result")
    if diacritic_result:
        original_col, restored_col = st.columns(2, gap="large")
        with original_col:
            st.markdown("**Original Sindhi text**")
            st.markdown(
                f'<div class="sindhi-text">{esc(diacritic_result["plain"])}</div>',
                unsafe_allow_html=True,
            )
        with restored_col:
            st.markdown("**Suggested restoration**")
            st.markdown(
                f'<div class="sindhi-text">{highlighted_restoration_html(diacritic_result)}</div>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Analysis: {st.session_state.get('diacritic_elapsed_ms', 0.0):.2f} ms · "
            "Highlighted characters have suggested zer, zabar, or pesh additions."
        )

        changes = [
            (token_index, change_index, token, change)
            for token_index, token in enumerate(diacritic_result["tokens"])
            for change_index, change in enumerate(token["changes"])
        ]
        if changes:
            st.markdown("#### Review suggestions")
            with st.form("diacritic_suggestion_form"):
                decisions = {}
                for token_index, change_index, token, change in changes:
                    decision = st.radio(
                        f"{token['plain']} · {change['character']} → {change['mark_name']}",
                        ["Accept", "Reject"],
                        horizontal=True,
                        key=f"diacritic_choice_{token_index}_{change_index}",
                    )
                    if change["confidence"] is None:
                        st.caption("Verified lexicon suggestion; no classifier probability is available.")
                    else:
                        st.caption(
                            f"Classifier probability: {change['confidence']:.0%} "
                            f"({change['confidence_label']}; not calibrated correctness)."
                        )
                    decisions[change["id"]] = decision

                apply_choices = st.form_submit_button("Apply accept/reject choices")

            if apply_choices:
                accepted_ids = [
                    change_id for change_id, choice in decisions.items()
                    if choice == "Accept"
                ]
                st.session_state["diacritic_editor"] = (
                    diacritic_engine.apply_suggestion_decisions(diacritic_result, accepted_ids)
                )
                st.success("Suggestion choices applied. You can make further manual edits below.")
        else:
            st.info("No vowel-mark additions were suggested for this text.")

        classifier_predictions = [
            (token, prediction)
            for token in diacritic_result["tokens"]
            for prediction in token["predictions"]
        ]
        if classifier_predictions:
            with st.expander("Per-character classifier estimates"):
                for token, prediction in classifier_predictions:
                    predicted_mark = prediction["mark"] or "no vowel mark"
                    if prediction["confidence"] is None:
                        score_text = "probability unavailable"
                    else:
                        score_text = (
                            f"{prediction['confidence']:.0%} model probability "
                            f"({prediction['confidence_label']}; not calibrated correctness)"
                        )
                    st.caption(
                        f"{token['plain']} · {prediction['character']}: "
                        f"{predicted_mark} · {score_text}"
                    )

        for token in diacritic_result["tokens"]:
            if token["ambiguous"]:
                candidates_text = " · ".join(token["ambiguous_candidates"])
                st.warning(
                    f"{token['plain']}: multiple verified readings are available ({candidates_text}). "
                    "No reading was selected automatically."
                )
            if token["preserved_mark_conflicts"]:
                st.warning(
                    f"{token['plain']}: existing user-entered marks were kept even though they differ "
                    "from the lexicon suggestion."
                )
            if token["uncertain_reason"]:
                st.info(f"{token['plain']}: {token['uncertain_reason']}")

        st.text_area(
            "Final Sindhi text — editable",
            key="diacritic_editor",
            height=110,
            help="Edit the suggested output freely; your changes are kept in this field.",
        )

        if any(SindhiDiacriticRestorer.is_combining_mark(char) for char in diacritic_result["plain"]):
            st.caption("Pre-existing Unicode diacritics in the input were preserved.")

    if "latest_tokens" in st.session_state and st.session_state["latest_tokens"]:
        tokens = st.session_state["latest_tokens"]
        verified_n = sum(1 for t in tokens if t["source"] == "verified_lexicon")
        predicted_n = sum(1 for t in tokens if t["source"] == "ml_classifier")
        ambiguous_n = sum(1 for t in tokens if t["source"] == "ambiguous")
        unmarked_n = len(tokens) - verified_n - predicted_n - ambiguous_n

        st.markdown("### Word-by-Word Decomposition")
        st.markdown(
            f'<div class="sa-legend">'
            f'<span class="sa-pill">🔢 {len(tokens)} words</span>'
            f'<span class="sa-pill green">✔ {verified_n} Lexicon entries</span>'
            f'<span class="sa-pill blue">🤖 {predicted_n} ML Prediction</span>'
            f'<span class="sa-pill amber">⚠ {ambiguous_n} Ambiguous</span>'
            f'<span class="sa-pill">— {unmarked_n} No mark added</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        cards = []
        for idx, tok in enumerate(tokens, start=1):
            source = tok["source"]
            chip_cls = (
                "verified-chip" if source == "verified_lexicon"
                else "predicted-chip" if source == "ml_classifier"
                else ""
            )
            badge_cls = (
                "verified" if source == "verified_lexicon"
                else "predicted" if source == "ml_classifier"
                else "uncertain" if source == "ambiguous"
                else ""
            )
            badge_label = (
                "✔ Lexicon entry" if source == "verified_lexicon"
                else "🤖 ML Prediction" if source == "ml_classifier"
                else "⚠ Ambiguous" if source == "ambiguous"
                else "No mark added"
            )
            cards.append(
                f'<div class="diacritic-guide-card {chip_cls}">'
                f'<span class="sa-word-index">#{idx}</span>'
                f'<div class="sa-word-marked">{esc(tok["marked"])}</div>'
                f'<div class="sa-word-plain">Plain: <span>{esc(tok["plain"])}</span></div>'
                f'<span class="sa-word-status {badge_cls}">{badge_label}</span>'
                f"</div>"
            )
        st.markdown(f'<div class="sa-word-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="sa-empty"><div class="sa-empty-icon">🎓</div>'
            "<div><b>No words to explore yet</b></div>"
            "<div style='font-size:0.85rem'>Translate a sentence in the first tab to view interactive "
            "word-by-word pronunciation guidance.</div></div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tab 3: Word Sound & Pronunciation Analysis
# ---------------------------------------------------------------------------
with tab_word_analysis:
    section_header(
        "Sindhi Word Sound & Pronunciation Analysis",
        "Explore written vowel marks and cautious text-based guidance using the existing restorer and reviewed wordbook.",
    )
    st.info(
        "Text-only analysis: this project has no pronunciation recordings or speech resource. "
        "The displayed vowel guidance is not audio and does not verify a single regional pronunciation."
    )
    st.markdown("**Examples of the vowel marks**")
    example_columns = st.columns(3)
    for column, (mark, guide) in zip(
        example_columns,
        (
            (config.ZER, VOWEL_GUIDE[config.ZER]),
            (config.ZABAR, VOWEL_GUIDE[config.ZABAR]),
            (config.PESH, VOWEL_GUIDE[config.PESH]),
        ),
    ):
        with column:
            st.markdown(
                f'<div class="sa-vowel"><div class="sa-vowel-glyph">ڪ{esc(mark)}</div><div>'
                f'<div class="sa-vowel-title">{esc(guide["name"])}</div>'
                f'<div class="sa-vowel-desc">{esc(guide["effect"])}</div></div></div>',
                unsafe_allow_html=True,
            )

    if "word_analysis_input" not in st.session_state:
        st.session_state["word_analysis_input"] = ""
    if "word_analysis_result" not in st.session_state:
        st.session_state["word_analysis_result"] = None

    st.caption("Try an existing wordbook example, or enter one Sindhi word in Perso-Arabic script.")
    example_columns = st.columns(3)
    for column, (label, value) in zip(
        example_columns,
        (("Book · ڪتاب", "ڪتاب"), ("Knowledge · علم", "علم"), ("Ambiguous · پل", "پل")),
    ):
        with column:
            st.button(
                label,
                key=f"word_analysis_example_{value}",
                use_container_width=True,
                on_click=lambda example=value: st.session_state.update({
                    "word_analysis_input": example,
                    "word_analysis_result": None,
                }),
            )

    analysis_input = st.text_input(
        "Sindhi word",
        key="word_analysis_input",
        placeholder="مثال: ڪتاب",
    )
    analyze_col, clear_col = st.columns([3, 1])
    with analyze_col:
        analyze_word_button = st.button(
            "Analyze word",
            type="primary",
            use_container_width=True,
            disabled=not analysis_input.strip(),
        )
    with clear_col:
        st.button(
            "Clear",
            key="clear_word_analysis",
            use_container_width=True,
            on_click=lambda: st.session_state.update({
                "word_analysis_input": "",
                "word_analysis_result": None,
            }),
        )

    if analyze_word_button:
        try:
            st.session_state["word_analysis_result"] = analyze_word(
                analysis_input,
                diacritic_engine,
            )
            st.session_state["word_analysis_editor"] = (
                st.session_state["word_analysis_result"]["restored_word"]
            )
        except ValueError as error:
            st.error(str(error))

    word_result = st.session_state.get("word_analysis_result")
    if word_result:
        base_column, restored_column = st.columns(2, gap="large")
        with base_column:
            st.markdown("**Base written form**")
            st.markdown(
                f'<div class="sindhi-text">{esc(word_result["base_word"])}</div>',
                unsafe_allow_html=True,
            )
        with restored_column:
            st.markdown("**Restored / supplied-mark form**")
            st.markdown(
                f'<div class="sindhi-text">{esc(word_result["restored_word"])}</div>',
                unsafe_allow_html=True,
            )

        if word_result["ambiguous_forms"]:
            st.warning(
                "This spelling has multiple reviewed vowel-marked readings. The system does not "
                "select one without enough context."
            )
            for reading in word_result["ambiguous_forms"]:
                st.markdown(f'<div class="sindhi-word-chip">{esc(reading)}</div>', unsafe_allow_html=True)
        elif word_result["uncertain_reason"]:
            st.info(word_result["uncertain_reason"])
        elif word_result["source"] == "ml_classifier":
            st.caption(
                "The restored form includes classifier suggestions. These are statistical text predictions, "
                "not verified pronunciations."
            )

        st.markdown("#### Character-by-character marks")
        char_columns = st.columns(max(1, min(4, len(word_result["characters"]))))
        for index, character in enumerate(word_result["characters"]):
            with char_columns[index % len(char_columns)]:
                char_html = (
                    f'<div class="diacritic-guide-card">'
                    f'<div class="sa-word-marked">{esc(character["character"])}</div>'
                    f'<div class="sa-word-plain">Existing marks: '
                    f'{esc(", ".join(character["existing_marks"]) or "None")}</div>'
                    f'<div class="sa-word-plain">Restored marks: '
                    f'{esc(", ".join(character["restored_marks"]) or "None")}</div>'
                )
                if character["added_marks"]:
                    char_html += (
                        f'<div class="sa-word-plain">Suggested: '
                        f'{esc(", ".join(character["added_marks"]))}</div>'
                    )
                for effect in character["mark_effects"]:
                    char_html += f'<div class="sa-word-plain">{esc(effect)}</div>'
                if character["confidence"] is not None:
                    char_html += (
                        f'<div class="sa-word-plain">Classifier score: '
                        f'{character["confidence"]:.0%}; not calibrated correctness.</div>'
                    )
                char_html += "</div>"
                st.markdown(char_html, unsafe_allow_html=True)

        if word_result["wordbook_entries"]:
            with st.expander("Reviewed wordbook notes"):
                for entry in word_result["wordbook_entries"]:
                    if entry["notes"]:
                        st.markdown(f"- {esc(entry['notes'])}")
                    if entry["source"]:
                        st.caption(f"Source label: {entry['source']}")
                st.caption(
                    "These are the wordbook's own notes; romanized notes are not a standardized phonetic transcription."
                )

        st.text_area(
            "Editable word analysis result",
            key="word_analysis_editor",
            height=90,
            help="Edit the restored written form for your own notes; this does not create or play audio.",
        )
        st.caption("Audio pronunciation is not available in this project.")

# ---------------------------------------------------------------------------
# Tab 4: Verified Wordbook
# ---------------------------------------------------------------------------
with tab_dictionary:
    section_header(
        "Searchable Educational Sindhi Vocabulary",
        "Browse verified Sindhi words with verified vowel markings and phonological annotations.",
    )

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

        st.markdown(
            f'<div class="sa-stats">'
            f'<span class="sa-pill teal">📚 {len(filtered_df)} of {len(vocab_df)} entries</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

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

# ---------------------------------------------------------------------------
# Tab 4: Benchmarks & Research
# ---------------------------------------------------------------------------
with tab_benchmarks:
    section_header(
        "Benchmark Reports & Academic Evaluation",
        "Existing measurements are from local runs and do not represent cloud-provider performance.",
    )

    col_rep1, col_rep2 = st.columns(2, gap="large")

    with col_rep1:
        with st.container(border=True):
            st.markdown("#### 🎯 Diacritic Restoration Evaluation")
            diac_rep_path = config.REPORTS_DIR / "diacritic_evaluation_report.json"
            if diac_rep_path.exists():
                with open(diac_rep_path, "r", encoding="utf-8") as f:
                    d_rep = json.load(f)
                st.metric("Macro F1", f"{d_rep.get('macro_f1', 0.0):.4f}")
                st.metric("Existing Mark Preservation", f"{d_rep.get('existing_mark_preservation_accuracy', 0.0) * 100:.1f}%")
                st.metric("Exact Sentence Accuracy", f"{d_rep.get('exact_sentence_accuracy', 0.0) * 100:.1f}%")
                with st.expander("Per-diacritic metrics"):
                    st.json(d_rep.get("per_diacritic_metrics", {}))
            else:
                st.info("Run `python -m src.diacritics.evaluate_diacritics` to view evaluation.")

    with col_rep2:
        with st.container(border=True):
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

# ---------------------------------------------------------------------------
# Tab 5: User Feedback
# ---------------------------------------------------------------------------
with tab_feedback:
    if REMOTE_MODE:
        section_header("User Feedback & Human Evaluation", "Feedback storage is disabled in public cloud mode.")
        st.info(
            "The local app can save feedback on the machine running Streamlit. Public deployment "
            "does not collect or persist submitted corrections or notes."
        )
    else:
        section_header(
            "User Feedback & Human Evaluation",
            "Submit human corrections for translations or vowel diacritics to help improve future model iterations.",
        )

        with st.container(border=True):
            with st.form("feedback_form"):
                fb_english = st.text_input("Source English Text:", value=st.session_state.get("latest_english", ""))
                fb_sindhi_predicted = st.text_input("Predicted Sindhi Text:", value=st.session_state.get("latest_marked", ""))
                fb_correction = st.text_input("Corrected Sindhi Text (with vowel marks if known):")
                fb_rating = st.slider("Translation Quality Rating:", min_value=1, max_value=5, value=4)
                fb_notes = st.text_area("Linguistic comments or notes:")
                submitted = st.form_submit_button("Submit Feedback", type="primary")

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

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
footer_description = (
    "Sindhi AI · Cloudflare Workers AI translation · existing local NLLB/Canopy modes remain available on your machine"
    if REMOTE_MODE
    else "Sindhi AI · Offline English-to-Sindhi Translation & Diacritic Restoration · Powered by NLLB-200 · Runs locally on CPU"
)
st.markdown(
    f'<div class="sa-footer">{html.escape(footer_description)}</div>',
    unsafe_allow_html=True,
)