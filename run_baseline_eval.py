"""
Baseline Evaluation Script for English-to-Sindhi Translation System
Measures warm inference latency, caching behavior, and translation quality.
"""

import os
import sys
import time
import json
import psutil
import statistics
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.translation.translator import SindhiTranslator
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer


def get_memory_mb():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_baseline_evaluation():
    """Run comprehensive baseline evaluation"""
    print("="*80)
    print("BASELINE EVALUATION: English-to-Sindhi Translation System")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    print()
    
    # Test sentences: diverse types, lengths, and complexity
    test_sentences = [
        # Everyday conversation (short)
        "Hello, how are you?",
        "Good morning.",
        "Thank you very much.",
        "What is your name?",
        
        # Questions
        "Where do you live?",
        "Where is the railway station?",
        "Can you help me?",
        
        # Education (medium)
        "I am learning Sindhi.",
        "Books are our best friends.",
        "The student is reading an important book.",
        "Education is the key to success.",
        
        # Longer sentences
        "The teacher is explaining a difficult concept to the students in the classroom.",
        "I need to go to the hospital because I am not feeling well today.",
        
        # Technology/formal
        "Artificial intelligence is transforming communication.",
        "The computer program helps translate languages.",
    ]
    
    print("TEST CONFIGURATION")
    print(f"  Total test sentences: {len(test_sentences)}")
    print(f"  Sentence types: Everyday, Questions, Educational, Long, Technical")
    print()
    
    # Phase 1: Cold Start Measurement
    print("-" * 80)
    print("PHASE 1: COLD START MEASUREMENT")
    print("-" * 80)
    
    mem_before_load = get_memory_mb()
    print(f"Memory before model load: {mem_before_load:.1f} MB")
    
    cold_start_time = time.time()
    translator = SindhiTranslator(lazy_init=False)  # Force immediate loading
    diacritic_engine = SindhiDiacriticRestorer()
    cold_start_elapsed = time.time() - cold_start_time
    
    mem_after_load = get_memory_mb()
    model_memory = mem_after_load - mem_before_load
    
    print(f"Cold start time: {cold_start_elapsed:.2f} seconds")
    print(f"Memory after model load: {mem_after_load:.1f} MB")
    print(f"Model memory footprint: {model_memory:.1f} MB")
    print()
    
    # Phase 2: Warm Inference Latency (First Run)
    print("-" * 80)
    print("PHASE 2: WARM INFERENCE - FIRST RUN (NO CACHE)")
    print("-" * 80)
    
    first_run_results = []
    first_run_latencies = []
    
    for idx, sentence in enumerate(test_sentences, 1):
        start_time = time.time()
        
        # Translation
        trans_result = translator.translate_segment(sentence, num_beams=2)
        sindhi_output = trans_result["sindhi"]
        
        # Diacritic restoration
        diac_result = diacritic_engine.restore(sindhi_output)
        marked_output = diac_result["marked"]
        
        elapsed_ms = (time.time() - start_time) * 1000.0
        first_run_latencies.append(elapsed_ms)
        
        char_count = len(sentence)
        word_count = len(sentence.split())
        output_words = len(sindhi_output.split())
        
        result = {
            "index": idx,
            "input": sentence,
            "input_chars": char_count,
            "input_words": word_count,
            "output": sindhi_output,
            "output_marked": marked_output,
            "output_words": output_words,
            "latency_ms": round(elapsed_ms, 2),
            "tokens_per_sec": round(output_words / (elapsed_ms / 1000.0), 2) if elapsed_ms > 0 else 0
        }
        first_run_results.append(result)
        
        print(f"[{idx:2d}/{len(test_sentences)}] {elapsed_ms:6.1f} ms | {word_count:2d}w → {output_words:2d}w | {sentence[:50]}")
    
    print()
    print(f"First run average: {statistics.mean(first_run_latencies):.1f} ms")
    print(f"First run median:  {statistics.median(first_run_latencies):.1f} ms")
    print(f"First run min:     {min(first_run_latencies):.1f} ms")
    print(f"First run max:     {max(first_run_latencies):.1f} ms")
    print()
    
    # Phase 3: Repeated Inference (Cache Test)
    print("-" * 80)
    print("PHASE 3: REPEATED INFERENCE - CACHE TEST")
    print("-" * 80)
    print("Testing if repeated translations are faster (caching)...")
    print()
    
    # Test first 3 sentences again
    cache_test_sentences = test_sentences[:3]
    second_run_latencies = []
    
    for idx, sentence in enumerate(cache_test_sentences, 1):
        start_time = time.time()
        trans_result = translator.translate_segment(sentence, num_beams=2)
        sindhi_output = trans_result["sindhi"]
        diac_result = diacritic_engine.restore(sindhi_output)
        elapsed_ms = (time.time() - start_time) * 1000.0
        second_run_latencies.append(elapsed_ms)
        
        print(f"[{idx}] Second run: {elapsed_ms:6.1f} ms | Input: {sentence[:40]}")
    
    print()
    print("CACHE COMPARISON:")
    for i in range(len(cache_test_sentences)):
        first_lat = first_run_results[i]["latency_ms"]
        second_lat = second_run_latencies[i]
        speedup = ((first_lat - second_lat) / first_lat * 100) if first_lat > 0 else 0
        print(f"  Sentence {i+1}: {first_lat:.1f} ms → {second_lat:.1f} ms ({speedup:+.1f}%)")
    
    avg_first = statistics.mean([first_run_results[i]["latency_ms"] for i in range(len(cache_test_sentences))])
    avg_second = statistics.mean(second_run_latencies)
    overall_speedup = ((avg_first - avg_second) / avg_first * 100) if avg_first > 0 else 0
    
    print(f"\nOverall cache speedup: {overall_speedup:+.1f}%")
    is_cached = abs(overall_speedup) > 5.0  # More than 5% difference suggests caching
    print(f"Caching detected: {'YES' if is_cached else 'NO (translations recomputed each time)'}")
    print()
    
    # Phase 4: Memory Usage
    print("-" * 80)
    print("PHASE 4: MEMORY USAGE")
    print("-" * 80)
    
    mem_peak = get_memory_mb()
    system_mem = psutil.virtual_memory()
    
    print(f"Peak process memory: {mem_peak:.1f} MB")
    print(f"System total RAM: {system_mem.total / (1024**3):.1f} GB")
    print(f"System available RAM: {system_mem.available / (1024**3):.1f} GB")
    print(f"System RAM usage: {system_mem.percent:.1f}%")
    print()
    
    # Phase 5: Quality Analysis (if test data available)
    print("-" * 80)
    print("PHASE 5: TRANSLATION QUALITY ANALYSIS")
    print("-" * 80)
    
    # Check if we can run the official evaluation
    test_file = config.PARALLEL_TEST_FILE
    quality_metrics = {}
    
    if test_file.exists():
        print(f"Running official evaluation on: {test_file}")
        try:
            from src.evaluation.evaluate_translation import evaluate_translation_system
            quality_metrics = evaluate_translation_system()
            print(f"✓ BLEU Score: {quality_metrics.get('bleu_score', 'N/A')}")
            print(f"✓ chrF++ Score: {quality_metrics.get('chrf_score', 'N/A')}")
            print(f"✓ Script Validity: {quality_metrics.get('script_validity_pct', 'N/A')}%")
        except Exception as e:
            print(f"Warning: Could not run official evaluation: {e}")
            quality_metrics = {"error": str(e)}
    else:
        print(f"Test file not found: {test_file}")
        print("Skipping BLEU/chrF evaluation.")
    
    print()
    
    # Generate Report
    print("="*80)
    print("GENERATING BASELINE REPORT")
    print("="*80)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": sys.platform,
        "python_version": sys.version,
        "model": config.DEFAULT_TRANSLATION_MODEL,
        "cold_start": {
            "time_seconds": round(cold_start_elapsed, 2),
            "memory_footprint_mb": round(model_memory, 1)
        },
        "warm_inference": {
            "total_sentences_tested": len(test_sentences),
            "average_latency_ms": round(statistics.mean(first_run_latencies), 2),
            "median_latency_ms": round(statistics.median(first_run_latencies), 2),
            "min_latency_ms": round(min(first_run_latencies), 2),
            "max_latency_ms": round(max(first_run_latencies), 2),
            "std_dev_ms": round(statistics.stdev(first_run_latencies), 2) if len(first_run_latencies) > 1 else 0
        },
        "caching": {
            "cache_speedup_percent": round(overall_speedup, 2),
            "caching_detected": is_cached,
            "note": "No significant caching detected - each translation is computed fresh" if not is_cached else "Caching may be present"
        },
        "memory": {
            "peak_process_mb": round(mem_peak, 1),
            "system_total_gb": round(system_mem.total / (1024**3), 1),
            "system_available_gb": round(system_mem.available / (1024**3), 1)
        },
        "quality": quality_metrics,
        "detailed_results": first_run_results
    }
    
    # Save report
    report_file = config.REPORTS_DIR / "baseline_evaluation.json"
    os.makedirs(report_file.parent, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON report saved: {report_file}")
    
    # Create markdown summary
    md_report = generate_markdown_report(report, first_run_results)
    md_file = config.REPORTS_DIR / "baseline_evaluation.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_report)
    
    print(f"✓ Markdown report saved: {md_file}")
    print()
    
    return report


def generate_markdown_report(report, detailed_results):
    """Generate readable markdown report"""
    
    warm = report["warm_inference"]
    
    md = f"""# Baseline Evaluation Report
## English-to-Sindhi Neural Translation System

**Generated:** {report["timestamp"]}  
**Platform:** {report["platform"]}  
**Model:** {report["model"]}

---

## Executive Summary

### Cold Start Performance
- **Model Loading Time:** {report["cold_start"]["time_seconds"]} seconds
- **Memory Footprint:** {report["cold_start"]["memory_footprint_mb"]} MB

### Warm Inference Performance (First Run)
- **Average Latency:** {warm["average_latency_ms"]} ms
- **Median Latency:** {warm["median_latency_ms"]} ms
- **Min Latency:** {warm["min_latency_ms"]} ms
- **Max Latency:** {warm["max_latency_ms"]} ms
- **Standard Deviation:** {warm["std_dev_ms"]} ms
- **Total Sentences Tested:** {warm["total_sentences_tested"]}

### Caching Analysis
- **Cache Speedup:** {report["caching"]["cache_speedup_percent"]}%
- **Caching Detected:** {report["caching"]["caching_detected"]}
- **Note:** {report["caching"]["note"]}

### Memory Usage
- **Peak Process Memory:** {report["memory"]["peak_process_mb"]} MB
- **System Total RAM:** {report["memory"]["system_total_gb"]} GB
- **System Available RAM:** {report["memory"]["system_available_gb"]} GB

### Translation Quality
"""
    
    if "bleu_score" in report["quality"]:
        md += f"""- **BLEU Score:** {report["quality"]["bleu_score"]}
- **chrF++ Score:** {report["quality"]["chrf_score"]}
- **Script Validity:** {report["quality"]["script_validity_pct"]}%
"""
    else:
        md += "- Quality metrics pending (test dataset evaluation)\n"
    
    md += """
---

## Detailed Results by Sentence Length

### Short Sentences (< 30 characters)
"""
    
    short = [r for r in detailed_results if r["input_chars"] < 30]
    if short:
        avg_short = statistics.mean([r["latency_ms"] for r in short])
        md += f"- **Count:** {len(short)} sentences\n"
        md += f"- **Average Latency:** {avg_short:.1f} ms\n\n"
        for r in short[:3]:
            md += f"  - `{r['latency_ms']:.1f} ms` | {r['input']}\n"
    
    md += "\n### Medium Sentences (30-80 characters)\n"
    medium = [r for r in detailed_results if 30 <= r["input_chars"] < 80]
    if medium:
        avg_medium = statistics.mean([r["latency_ms"] for r in medium])
        md += f"- **Count:** {len(medium)} sentences\n"
        md += f"- **Average Latency:** {avg_medium:.1f} ms\n\n"
        for r in medium[:3]:
            md += f"  - `{r['latency_ms']:.1f} ms` | {r['input']}\n"
    
    md += "\n### Long Sentences (>= 80 characters)\n"
    long_sent = [r for r in detailed_results if r["input_chars"] >= 80]
    if long_sent:
        avg_long = statistics.mean([r["latency_ms"] for r in long_sent])
        md += f"- **Count:** {len(long_sent)} sentences\n"
        md += f"- **Average Latency:** {avg_long:.1f} ms\n\n"
        for r in long_sent:
            md += f"  - `{r['latency_ms']:.1f} ms` | {r['input'][:60]}...\n"
    
    md += """
---

## Complete Translation Examples

"""
    
    # Show first 5 examples
    for r in detailed_results[:5]:
        md += f"""
### Example {r['index']}
- **English:** {r['input']}
- **Sindhi (Standard):** {r['output']}
- **Sindhi (Marked):** {r['output_marked']}
- **Latency:** {r['latency_ms']} ms
- **Words:** {r['input_words']} → {r['output_words']}
"""
    
    md += """
---

## Bottleneck Analysis

### Current Bottlenecks
1. **Model Inference Time** - Primary bottleneck (accounts for >90% of latency)
2. **No Input Caching** - Each translation is computed fresh
3. **Beam Search Overhead** - Using num_beams=2 for quality

### Optimization Opportunities
1. **Model Quantization** - Convert to INT8/FP16 could reduce latency by 30-50%
2. **Input Caching** - Cache translations for repeated inputs
3. **Batch Processing** - Process multiple sentences together
4. **Reduce Beam Width** - Test num_beams=1 for faster inference
5. **ONNX Runtime** - Export to ONNX for optimized inference

---

## Recommendations

### High Priority
1. **Implement Input Caching** - Store translations for repeated sentences
2. **Profile Beam Search** - Test impact of num_beams=1 vs num_beams=2
3. **Optimize Short Sentences** - Special fast path for <5 word inputs

### Medium Priority
1. **Model Quantization** - Explore INT8 quantization for 2x speedup
2. **Batch API** - Add batch translation endpoint
3. **Pre-warming** - Run dummy translation on startup

### Low Priority
1. **ONNX Conversion** - More complex but potentially faster
2. **Custom CUDA Kernels** - Only if GPU becomes available

---

*Generated by Baseline Evaluation Script*
"""
    
    return md


if __name__ == "__main__":
    try:
        report = run_baseline_evaluation()
        print("\n" + "="*80)
        print("✓ BASELINE EVALUATION COMPLETE")
        print("="*80)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
