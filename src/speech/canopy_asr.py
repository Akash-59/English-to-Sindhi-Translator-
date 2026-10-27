"""Offline Sindhi transcription through an isolated Canopy worker process."""

import json
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from src.speech.memory_guard import require_available_ram

MODEL_REPOSITORY = "ProximaAI/Canopy-M"
MODEL_FILES = [
    "canopy-m.pt",
    "model_config.json",
    "language_map.json",
    "tokenizer.json",
    "tokenizer_version",
]


def recording_consent_key(audio_bytes: bytes) -> str:
    """Return a recording-specific consent state key without storing the audio."""
    return f"voice_offline_consent_{hashlib.sha256(audio_bytes).hexdigest()}"


def find_local_canopy_checkpoint() -> Path:
    """Resolve the already-cached checkpoint without contacting the Hub."""
    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_REPOSITORY,
            allow_patterns=MODEL_FILES,
            local_files_only=True,
        )
    )
    checkpoint = snapshot_path / "canopy-m.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Canopy-M checkpoint is not cached locally at {checkpoint}."
        )
    return checkpoint


def transcribe_sindhi_audio(audio_bytes: bytes) -> dict[str, Any]:
    """Transcribe one recorded WAV clip locally, with no model loaded in this process."""
    if not audio_bytes:
        raise ValueError("Record Sindhi audio before starting transcription.")
    if not audio_bytes.startswith(b"RIFF"):
        raise ValueError("The microphone recording is not a supported WAV file.")

    require_available_ram()

    checkpoint_path = find_local_canopy_checkpoint()
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HUGGINGFACE_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"

    with tempfile.NamedTemporaryFile(prefix="sindhi_asr_", suffix=".wav", delete=False) as audio_file:
        audio_path = Path(audio_file.name)
        audio_file.write(audio_bytes)

    try:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.speech.canopy_worker",
                    str(checkpoint_path),
                    str(audio_path),
                ],
                cwd=repository_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Offline Sindhi transcription timed out.") from error

        if result.returncode != 0:
            details = result.stderr.strip() or "Canopy-M transcription process failed."
            raise RuntimeError(details)

        output_lines = result.stdout.strip().splitlines()
        if not output_lines:
            raise RuntimeError("Canopy-M returned no transcription result.")
        try:
            response = json.loads(output_lines[-1])
        except json.JSONDecodeError as error:
            raise RuntimeError("Canopy-M returned an invalid worker response.") from error
        if not isinstance(response.get("text"), str):
            raise RuntimeError("Canopy-M worker response did not contain transcription text.")
        return response
    finally:
        if audio_path.exists():
            audio_path.unlink()
