import requests
import pytest

from src.translation.cloudflare_provider import (
    CloudflareWorkersAI,
    MAX_INPUT_CHARACTERS,
    TranslationProviderError,
)


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {
            "success": True,
            "result": {"translated_text": "سنڌي ترجمو"},
        }

    def json(self):
        return self._body


def test_cloudflare_translation_uses_m2m100_and_sindhi_language():
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    provider = CloudflareWorkersAI(
        "account",
        "token",
        free_plan_confirmed=True,
        post=post,
    )

    result = provider.translate("Good morning.")

    assert result["translation"] == "سنڌي ترجمو"
    assert captured["url"].endswith("/@cf/meta/m2m100-1.2b")
    assert captured["kwargs"]["json"] == {
        "text": "Good morning.",
        "source_lang": "english",
        "target_lang": "sindhi",
    }
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer token"


def test_cloudflare_translation_supports_reverse_direction_and_other_model_languages():
    captured = {}
    provider = CloudflareWorkersAI(
        "account",
        "token",
        free_plan_confirmed=True,
        post=lambda url, **kwargs: (
            captured.update(kwargs) or FakeResponse(body={
                "success": True,
                "result": {"translated_text": "Hello."},
            })
        ),
    )

    result = provider.translate("سلام", "Sindhi", "English")

    assert result["translation"] == "Hello."
    assert captured["json"]["source_lang"] == "sindhi"
    assert captured["json"]["target_lang"] == "english"

    provider.translate("Hola", "Spanish", "English")
    assert captured["json"]["source_lang"] == "spanish"


@pytest.mark.parametrize(
    ("text", "source", "target", "error"),
    [
        (" ", "English", "Sindhi", ValueError),
        ("x" * (MAX_INPUT_CHARACTERS + 1), "English", "Sindhi", ValueError),
        ("hello", "Klingon", "Sindhi", ValueError),
        ("hello", "English", "English", ValueError),
    ],
)
def test_cloudflare_translation_validates_local_input(text, source, target, error):
    provider = CloudflareWorkersAI(
        "account",
        "token",
        free_plan_confirmed=True,
        post=lambda *args, **kwargs: pytest.fail(),
    )

    if not text.strip():
        assert provider.translate(text, source, target)["translation"] == ""
    else:
        with pytest.raises(error):
            provider.translate(text, source, target)


def test_cloudflare_translation_reports_missing_credentials_without_request():
    provider = CloudflareWorkersAI(None, None, post=lambda *args, **kwargs: pytest.fail())

    with pytest.raises(TranslationProviderError, match="not configured"):
        provider.translate("hello")


def test_cloudflare_translation_requires_free_plan_confirmation_without_request():
    provider = CloudflareWorkersAI(
        "account",
        "token",
        post=lambda *args, **kwargs: pytest.fail(),
    )

    with pytest.raises(TranslationProviderError, match="Free plan"):
        provider.translate("hello")


def test_cloudflare_translation_maps_auth_and_quota_errors():
    for status, expected in (
        (401, "API token"),
        (429, "free usage or request limit"),
    ):
        provider = CloudflareWorkersAI(
            "account",
            "token",
            free_plan_confirmed=True,
            post=lambda *args, status=status, **kwargs: FakeResponse(status),
        )
        with pytest.raises(TranslationProviderError, match=expected):
            provider.translate("hello")


def test_cloudflare_translation_handles_network_and_bad_payloads():
    provider = CloudflareWorkersAI(
        "account",
        "token",
        free_plan_confirmed=True,
        post=lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout()),
    )
    with pytest.raises(TranslationProviderError, match="timed out"):
        provider.translate("hello")

    provider = CloudflareWorkersAI(
        "account",
        "token",
        free_plan_confirmed=True,
        post=lambda *args, **kwargs: FakeResponse(body={"success": True, "result": {}}),
    )
    with pytest.raises(TranslationProviderError, match="no translation text"):
        provider.translate("hello")


def test_no_remote_sindhi_asr_is_advertised_as_verified():
    from src.speech.providers import get_verified_remote_sindhi_provider

    assert get_verified_remote_sindhi_provider() is None
