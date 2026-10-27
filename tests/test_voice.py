import base64
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.speech import canopy_asr
from src.speech import memory_guard
from src.speech import canopy_worker
from src.speech import offline_tts


def _decoded_argument(value):
    return base64.b64decode(value).decode("utf-8")


def test_synthesize_english_speech_uses_local_english_voice_and_cleans_temp_file(monkeypatch):
    monkeypatch.setattr(offline_tts.os, "name", "nt")
    monkeypatch.setattr(offline_tts.shutil, "which", lambda name: "powershell.exe")
    captured = {}

    def fake_run(command, **kwargs):
        environment = kwargs["env"]
        output_path = Path(
            _decoded_argument(environment["SINDHI_TTS_OUTPUT_PATH_B64"])
        )
        captured["path"] = output_path
        captured["text"] = _decoded_argument(environment["SINDHI_TTS_TEXT_B64"])
        captured["script"] = command[-1]
        output_path.write_bytes(b"RIFF" + b"\x00" * 48)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(offline_tts.subprocess, "run", fake_run)

    audio = offline_tts.synthesize_english_speech("Hello from Sindhi voice.")

    assert audio.startswith(b"RIFF")
    assert captured["text"] == "Hello from Sindhi voice."
    assert 'Culture.Name -like "en-*"' in captured["script"]
    assert "SetOutputToWaveFile" in captured["script"]
    assert not captured["path"].exists()


def test_synthesize_english_speech_reports_missing_english_voice(monkeypatch):
    monkeypatch.setattr(offline_tts.os, "name", "nt")
    monkeypatch.setattr(offline_tts.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(
        offline_tts.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="No enabled offline English SAPI voice is installed."
        ),
    )

    with pytest.raises(RuntimeError, match="No enabled offline English SAPI voice"):
        offline_tts.synthesize_english_speech("Hello.")


def test_synthesize_english_speech_rejects_empty_text():
    with pytest.raises(ValueError, match="English text is required"):
        offline_tts.synthesize_english_speech("  ")


def test_synthesize_english_speech_rejects_non_windows(monkeypatch):
    monkeypatch.setattr(offline_tts.os, "name", "posix")

    with pytest.raises(RuntimeError, match="available only on Windows"):
        offline_tts.synthesize_english_speech("Hello.")


def test_canopy_transcription_stops_when_memory_is_low(monkeypatch):
    monkeypatch.setattr(
        memory_guard.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1024 ** 3),
    )
    monkeypatch.setattr(
        canopy_asr,
        "find_local_canopy_checkpoint",
        lambda: pytest.fail("checkpoint lookup must not run below the memory limit"),
    )

    with pytest.raises(RuntimeError, match="at least 3.0 GiB"):
        canopy_asr.transcribe_sindhi_audio(b"RIFF" + b"\x00" * 48)


def test_memory_guard_defaults_to_three_gib(monkeypatch):
    assert memory_guard.minimum_available_ram_bytes() == 3 * 1024 ** 3


def test_memory_guard_threshold_cannot_be_lowered_by_environment(monkeypatch):
    monkeypatch.setenv("CANOPY_MIN_AVAILABLE_RAM_GIB", "0.5")

    assert memory_guard.minimum_available_ram_bytes() == 3 * 1024 ** 3


def test_worker_rechecks_memory_before_loading_model(monkeypatch):
    monkeypatch.setattr(canopy_worker.sys, "argv", ["canopy_worker", "checkpoint", "audio.wav"])
    monkeypatch.setattr(
        canopy_worker,
        "require_available_ram",
        lambda: (_ for _ in ()).throw(RuntimeError("insufficient worker memory")),
    )
    monkeypatch.setattr(
        canopy_worker.Canopy,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("ASR weights must not load below the worker threshold"),
    )

    with pytest.raises(RuntimeError, match="insufficient worker memory"):
        canopy_worker.main()


def test_canopy_transcription_uses_offline_worker_and_cleans_audio_file(monkeypatch, tmp_path):
    checkpoint = tmp_path / "canopy-m.pt"
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(
        memory_guard.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4 * 1024 ** 3),
    )
    monkeypatch.setattr(canopy_asr, "find_local_canopy_checkpoint", lambda: checkpoint)
    captured = {}

    def fake_run(command, **kwargs):
        audio_path = Path(command[-1])
        captured["audio"] = audio_path.read_bytes()
        captured["audio_path"] = audio_path
        captured["environment"] = kwargs["env"]
        captured["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "text": "سنڌي جملو",
                    "load_ms": 45.0,
                    "inference_ms": 120.0,
                    "peak_rss_mb": 420.0,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(canopy_asr.subprocess, "run", fake_run)
    result = canopy_asr.transcribe_sindhi_audio(b"RIFF" + b"\x00" * 48)

    assert result["text"] == "سنڌي جملو"
    assert result["inference_ms"] == 120.0
    assert result["peak_rss_mb"] == 420.0
    assert captured["audio"].startswith(b"RIFF")
    assert captured["environment"]["HF_HUB_OFFLINE"] == "1"
    assert captured["command"][1:3] == ["-m", "src.speech.canopy_worker"]
    assert not captured["audio_path"].exists()


def test_canopy_transcription_reports_worker_errors(monkeypatch, tmp_path):
    checkpoint = tmp_path / "canopy-m.pt"
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(
        memory_guard.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4 * 1024 ** 3),
    )
    monkeypatch.setattr(canopy_asr, "find_local_canopy_checkpoint", lambda: checkpoint)
    monkeypatch.setattr(
        canopy_asr.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Canopy worker failed."
        ),
    )

    with pytest.raises(RuntimeError, match="Canopy worker failed"):
        canopy_asr.transcribe_sindhi_audio(b"RIFF" + b"\x00" * 48)


def test_recording_consent_key_is_specific_to_audio():
    first = canopy_asr.recording_consent_key(b"first test recording")
    second = canopy_asr.recording_consent_key(b"second test recording")

    assert first.startswith("voice_offline_consent_")
    assert first != second
