"""Isolated CPU worker for local Canopy-M speech recognition."""

import json
import sys
import threading
import time

import psutil
from canopy import Canopy
from src.speech.memory_guard import require_available_ram


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("Expected a local checkpoint path and WAV path.")

    checkpoint_path, audio_path = sys.argv[1:]
    process = psutil.Process()
    peak_rss_bytes = [process.memory_info().rss]
    stop_sampling = threading.Event()

    def sample_memory() -> None:
        while not stop_sampling.wait(0.05):
            peak_rss_bytes[0] = max(peak_rss_bytes[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    started_at = time.perf_counter()
    try:
        require_available_ram()
        model = Canopy.from_pretrained(checkpoint_path, device="cpu")
        loaded_at = time.perf_counter()
        text = model.transcribe(audio_path, language="snd", decoder="greedy")
        finished_at = time.perf_counter()
        peak_rss_bytes[0] = max(peak_rss_bytes[0], process.memory_info().rss)
    finally:
        stop_sampling.set()
        sampler.join()

    print(
        json.dumps(
            {
                "text": text,
                "load_ms": round((loaded_at - started_at) * 1000, 2),
                "inference_ms": round((finished_at - loaded_at) * 1000, 2),
                "peak_rss_mb": round(peak_rss_bytes[0] / (1024 ** 2), 2),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
