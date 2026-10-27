"""Cloudflare Workers AI text translation provider."""

import time
from typing import Any, Dict, Mapping, Optional

import requests


MODEL_ID = "@cf/meta/m2m100-1.2b"
MAX_INPUT_CHARACTERS = 5000

SUPPORTED_LANGUAGES = {
    "Afrikaans": "afrikaans",
    "Amharic": "amharic",
    "Arabic": "arabic",
    "Asturian": "asturian",
    "Azerbaijani": "azerbaijani",
    "Bashkir": "bashkir",
    "Belarusian": "belarusian",
    "Bulgarian": "bulgarian",
    "Bengali": "bengali",
    "Breton": "breton",
    "Bosnian": "bosnian",
    "Catalan": "catalan",
    "Cebuano": "cebuano",
    "Czech": "czech",
    "Welsh": "welsh",
    "Danish": "danish",
    "German": "german",
    "Greek": "greek",
    "English": "english",
    "Spanish": "spanish",
    "Estonian": "estonian",
    "Persian": "persian",
    "Fulah": "fulah",
    "Finnish": "finnish",
    "French": "french",
    "Western Frisian": "western frisian",
    "Irish": "irish",
    "Scottish Gaelic": "scottish gaelic",
    "Galician": "galician",
    "Gujarati": "gujarati",
    "Hausa": "hausa",
    "Hebrew": "hebrew",
    "Hindi": "hindi",
    "Croatian": "croatian",
    "Haitian Creole": "haitian creole",
    "Hungarian": "hungarian",
    "Armenian": "armenian",
    "Indonesian": "indonesian",
    "Igbo": "igbo",
    "Iloko": "iloko",
    "Icelandic": "icelandic",
    "Italian": "italian",
    "Japanese": "japanese",
    "Javanese": "javanese",
    "Georgian": "georgian",
    "Kazakh": "kazakh",
    "Central Khmer": "central khmer",
    "Kannada": "kannada",
    "Korean": "korean",
    "Luxembourgish": "luxembourgish",
    "Ganda": "ganda",
    "Lingala": "lingala",
    "Lao": "lao",
    "Lithuanian": "lithuanian",
    "Latvian": "latvian",
    "Malagasy": "malagasy",
    "Macedonian": "macedonian",
    "Malayalam": "malayalam",
    "Mongolian": "mongolian",
    "Marathi": "marathi",
    "Malay": "malay",
    "Burmese": "burmese",
    "Nepali": "nepali",
    "Dutch": "dutch",
    "Norwegian": "norwegian",
    "Northern Sotho": "northern sotho",
    "Occitan": "occitan",
    "Oriya": "oriya",
    "Punjabi": "punjabi",
    "Pashto": "pashto",
    "Portuguese": "portuguese",
    "Romanian": "romanian",
    "Russian": "russian",
    "Sindhi": "sindhi",
    "Sinhala": "sinhala",
    "Slovak": "slovak",
    "Slovenian": "slovenian",
    "Somali": "somali",
    "Albanian": "albanian",
    "Serbian": "serbian",
    "Swati": "swati",
    "Sundanese": "sundanese",
    "Swedish": "swedish",
    "Swahili": "swahili",
    "Tamil": "tamil",
    "Thai": "thai",
    "Tagalog": "tagalog",
    "Tswana": "tswana",
    "Turkish": "turkish",
    "Ukrainian": "ukrainian",
    "Urdu": "urdu",
    "Uzbek": "uzbek",
    "Vietnamese": "vietnamese",
    "Wolof": "wolof",
    "Xhosa": "xhosa",
    "Yiddish": "yiddish",
    "Yoruba": "yoruba",
    "Chinese": "chinese",
    "Zulu": "zulu",
}


class TranslationProviderError(RuntimeError):
    """A user-actionable translation provider failure."""


class CloudflareWorkersAI:
    """Small REST client; it never falls back to local or paid inference."""

    def __init__(
        self,
        account_id: Optional[str],
        api_token: Optional[str],
        *,
        free_plan_confirmed: bool = False,
        post: Any = requests.post,
        timeout_seconds: float = 30,
    ):
        self._account_id = account_id.strip() if account_id else ""
        self._api_token = api_token.strip() if api_token else ""
        self._free_plan_confirmed = free_plan_confirmed
        self._post = post
        self._timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(
            self._account_id
            and self._api_token
            and self._free_plan_confirmed
        )

    def translate(
        self,
        text: str,
        source_language: str = "English",
        target_language: str = "Sindhi",
    ) -> Dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            return {
                "source": "",
                "translation": "",
                "latency_ms": 0.0,
            }
        if len(cleaned) > MAX_INPUT_CHARACTERS:
            raise ValueError(
                f"Text must be no longer than {MAX_INPUT_CHARACTERS} characters."
            )
        if source_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {source_language}.")
        if target_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported target language: {target_language}.")
        if source_language == target_language:
            raise ValueError("Choose different source and target languages.")
        if not self.configured:
            if self._account_id and self._api_token:
                raise TranslationProviderError(
                    "Cloud translation is disabled until the operator confirms the "
                    "Cloudflare account is on the Free plan with no billing enabled."
                )
            raise TranslationProviderError(
                "Cloud translation is not configured. Add Cloudflare account credentials "
                "to the hosting platform's secret manager."
            )

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{MODEL_ID}"
        )
        payload = {
            "text": cleaned,
            "source_lang": SUPPORTED_LANGUAGES[source_language],
            "target_lang": SUPPORTED_LANGUAGES[target_language],
        }
        start_time = time.perf_counter()
        try:
            response = self._post(
                url,
                headers={"Authorization": f"Bearer {self._api_token}"},
                json=payload,
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as error:
            raise TranslationProviderError(
                "Cloud translation timed out. You can try the request again."
            ) from error
        except requests.RequestException as error:
            raise TranslationProviderError(
                "Cloud translation could not connect. Check the network and retry."
            ) from error

        if response.status_code == 401:
            raise TranslationProviderError(
                "Cloudflare rejected the API token. Check its Workers AI permissions."
            )
        if response.status_code == 429:
            raise TranslationProviderError(
                "The Cloudflare free usage or request limit was reached. Wait for the "
                "limit to reset; this app will not switch to a paid plan."
            )
        if response.status_code in (402, 403):
            raise TranslationProviderError(
                "Cloudflare denied this request or the account is not entitled to this "
                "model. No paid fallback was attempted."
            )
        if response.status_code >= 500:
            raise TranslationProviderError(
                "Cloudflare's translation service is temporarily unavailable. Retry later."
            )
        if response.status_code >= 400:
            raise TranslationProviderError(
                f"Cloud translation request failed (HTTP {response.status_code})."
            )

        try:
            body = response.json()
        except ValueError as error:
            raise TranslationProviderError(
                "Cloudflare returned an invalid response."
            ) from error
        if not isinstance(body, Mapping) or body.get("success") is False:
            raise TranslationProviderError(
                "Cloudflare reported that the translation request failed."
            )
        result = body.get("result")
        translated_text = (
            result.get("translated_text")
            if isinstance(result, Mapping)
            else None
        )
        if not isinstance(translated_text, str) or not translated_text.strip():
            raise TranslationProviderError(
                "Cloudflare returned no translation text."
            )

        return {
            "source": cleaned,
            "translation": translated_text.strip(),
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
