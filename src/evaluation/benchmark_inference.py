"""
Inference performance benchmarking script for CPU-only 8 GB RAM environment.
Measures cold-start latency, memory footprint, throughput, and disk footprint.
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
from typing import Dict, Any, List

import config
from src.translation.model_loader import TranslationModelLoader
from src.translation.translator import SindhiTranslator
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer


def get_dir_size_mb(path: Path) -> float:
    """Calculates directory size in Megabytes."""
    if not path.exists():
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / (1024 * 1024), 2)


def run_benchmark(
    report_file: Path = config.REPORTS_DIR / "benchmark_report.json",
) -> Dict[str, Any]:
    """Runs cold start, latency, throughput, and memory measurements."""
    process = psutil.Process(os.getpid())
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)

    print("[Benchmark] Measuring cold start...")
    start_cold = time.time()
    translator = SindhiTranslator(lazy_init=False)
    diacritic_engine = SindhiDiacriticRestorer()
    cold_start_sec = time.time() - start_cold

    ram_after_load_mb = process.memory_info().rss / (1024 * 1024)
    model_ram_mb = ram_after_load_mb - ram_initial_mb

    test_sentences = [
        ("Short (Everyday)", "Hello, how are you?"),
        ("Medium (Education)", "The student is reading an important book in the university library."),
        ("Long (Technology)", "Artificial intelligence and machine translation systems are transforming cross-lingual communication across low-resource languages."),
    ]

    benchmark_runs: List[Dict[str, Any]] = []
    total_tokens = 0
    total_time_sec = 0.0

    print("[Benchmark] Running inference latency tests on CPU...")
    for category, text in test_sentences:
        t0 = time.time()
        # Translate
        trans_res = translator.translate_segment(text, num_beams=2)
        # Restore diacritics
        diac_res = diacritic_engine.restore(trans_res["sindhi"])
        elapsed = time.time() - t0

        tokens_gen = trans_res.get("target_tokens", len(trans_res["sindhi"].split()))
        total_tokens += tokens_gen
        total_time_sec += elapsed

        benchmark_runs.append({
            "category": category,
            "input_english": text,
            "output_sindhi": trans_res["sindhi"],
            "output_diacritics": diac_res["marked"],
            "tokens_generated": tokens_gen,
            "latency_ms": round(elapsed * 1000.0, 2),
            "throughput_tokens_per_sec": round(tokens_gen / elapsed if elapsed else 0, 2),
        })

    ram_peak_mb = process.memory_info().rss / (1024 * 1024)
    avg_throughput = round(total_tokens / total_time_sec if total_time_sec else 0, 2)

    # Disk sizes
    trans_disk_mb = get_dir_size_mb(config.TRANSLATION_MODEL_DIR)
    diac_disk_mb = get_dir_size_mb(config.DIACRITIC_MODEL_DIR)

    report = {
        "hardware": {
            "platform": sys.platform,
            "total_system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_threads_configured": config.CPU_INFERENCE_NUM_THREADS,
        },
        "cold_start_time_sec": round(cold_start_sec, 2),
        "memory_profile_mb": {
            "initial_rss_mb": round(ram_initial_mb, 2),
            "loaded_model_rss_mb": round(ram_after_load_mb, 2),
            "model_memory_footprint_mb": round(model_ram_mb, 2),
            "peak_rss_mb": round(ram_peak_mb, 2),
        },
        "disk_profile_mb": {
            "translation_model_disk_mb": trans_disk_mb,
            "diacritic_model_disk_mb": diac_disk_mb,
            "total_models_disk_mb": round(trans_disk_mb + diac_disk_mb, 2),
        },
        "performance": {
            "average_throughput_tokens_per_sec": avg_throughput,
            "runs": benchmark_runs,
        },
    }

    print("\n--- Benchmark Summary ---")
    print(f"Cold Start Time: {cold_start_sec:.2f} s")
    print(f"Model RAM Footprint: {model_ram_mb:.1f} MB (Peak RSS: {ram_peak_mb:.1f} MB)")
    print(f"Total Models Disk Size: {trans_disk_mb + diac_disk_mb:.1f} MB")
    print(f"Avg Throughput: {avg_throughput} tokens/sec")

    os.makedirs(report_file.parent, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Benchmark] Saved benchmark report -> {report_file}")

    return report


if __name__ == "__main__":
    run_benchmark()
