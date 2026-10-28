"""Streamlit cloud UI smoke tests using AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_cloud_app_renders_premium_translator_shell(monkeypatch, tmp_path):
    monkeypatch.setenv("SINDHI_PROVIDER_MODE", "cloudflare")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_FREE_PLAN_CONFIRMED", raising=False)

    app_path = Path(__file__).resolve().parents[1] / "cloud" / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30)
    app.run()

    assert not app.exception
    markdown = " ".join(item.value for item in app.markdown)
    assert "English to Sindhi Translator" in markdown
    assert "Translate naturally. Communicate confidently." in markdown
    assert "Start translating" in markdown
    assert "Source language" in [item.label for item in app.selectbox]
    assert "Target language" in [item.label for item in app.selectbox]
