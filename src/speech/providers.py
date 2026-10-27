"""Interfaces for future hosted Sindhi speech-recognition providers."""

from typing import Protocol


class SindhiSpeechProvider(Protocol):
    """A provider must establish Sindhi and output-script suitability before use."""

    name: str
    language: str
    output_script: str
    script_and_accuracy_verified: bool

    def transcribe(self, audio_bytes: bytes, *, consented: bool) -> str:
        """Return a transcription only after the user explicitly approves transmission."""


def get_verified_remote_sindhi_provider() -> None:
    """No hosted provider currently meets the project's Sindhi/script verification bar."""
    return None
