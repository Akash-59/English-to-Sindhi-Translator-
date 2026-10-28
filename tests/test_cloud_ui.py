"""Streamlit cloud UI smoke tests using AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_cloud_app_renders_premium_translator_shell(monkeypatch):
    monkeypatch.setenv("SINDHI_PROVIDER_MODE", "cloudflare")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_FREE_PLAN_CONFIRMED", raising=False)

    app_path = Path(__file__).resolve().parents[1] / "cloud" / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.run()

    assert not app.exception
    markdown = " ".join(item.value for item in app.markdown)
    assert "Sindhi Translator" in markdown
    assert "Enter English text" in [item.label for item in app.text_area]
    radio_labels = " ".join(str(option) for item in app.radio for option in item.options)
    assert "English → Sindhi" in radio_labels
    assert "Sindhi → English" in radio_labels
    button_labels = [getattr(item, "label", "") for item in app.button]
    assert "Translate" in button_labels
    assert "Clear" in button_labels


def test_app_voice_tab_explicitly_separates_cloud_and_offline_code():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "app.py").read_text(encoding="utf-8")

    assert "OFFLINE_VOICE_FEATURES_AVAILABLE = not REMOTE_MODE" in source
    assert "if not OFFLINE_VOICE_FEATURES_AVAILABLE:" in source
    assert "Speech transcription is unavailable in the public deployment" in source
    assert "No audio is recorded or sent in cloud mode" in source

    offline_calls = (
        "minimum_available_ram_bytes()",
        "available_ram_bytes()",
        "recording_consent_key(audio_bytes)",
        "transcribe_sindhi_audio(audio_bytes)",
        "synthesize_english_speech(",
        "TranslationModelLoader.unload()",
    )
    voice_section = source.split("with tab_voice:", 1)[1].split("# Tab 4:", 1)[0]
    cloud_voice_section = voice_section.split("    else:\n", 1)[0]
    for call in offline_calls:
        assert call in voice_section
        assert call not in cloud_voice_section
