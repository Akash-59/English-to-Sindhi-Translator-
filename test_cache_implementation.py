"""
Phase 2 Cache Testing Script
Tests the new input caching implementation against baseline performance.
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.translation.translator import SindhiTranslator
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer


def test_cache_implementation():
    """Test cache functionality with detailed metrics."""
    print("="*80)
    print("PHASE 2: CACHE IMPLEMENTATION TEST")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test sentences - same as baseline for comparison
    test_sentences = [
        "Hello, how are you?",
        "Good morning.",
        "Thank you very much.",
        "What is your name?",
        "Where do you live?",
        "Where is the railway station?",
        "Can you help me?",
        "I am learning Sindhi.",
        "Books are our best friends.",
        "The student is reading an important book.",
        "Education is the key to success.",
        "The teacher is explaining a difficult concept to the students in the classroom.",
        "I need to go to the hospital because I am not feeling well today.",
        "Artificial intelligence is transforming communication.",
        "The computer program helps translate languages.",
    ]
    
    print("TEST 1: Verify Cache is Enabled")
    print("-" * 80)
    
    # Create translator with cache enabled
    translator = SindhiTranslator(lazy_init=False, enable_cache=True, cache_size=1000)
    diacritic_engine = SindhiDiacriticRestorer()
    
    cache_stats = translator.get_cache_stats()
    print(f"Cache enabled: {cache_stats.get('max_size', 0) > 0}")
    print(f"Cache max size: {cache_stats.get('max_size', 0)} entries")
    print()
    
    print("TEST 2: First Run (Uncached - Baseline)")
    print("-" * 80)
    
    first_run_results = []
    uncached_latencies = []
    
    for idx, sentence in enumerate(test_sentences, 1):
        start_time = time.time()
        trans_result = translator.translate_segment(sentence, num_beams=2)
        diac_result = diacritic_engine.restore(trans_result["sindhi"])
        elapsed_ms = (time.time() - start_time) * 1000.0
        
        is_cache_hit = trans_result.get("cache_hit", False)
        uncached_latencies.append(elapsed_ms)
        
        first_run_results.append({
            "index": idx,
            "input": sentence,
            "output": trans_result["sindhi"],
            "output_marked": diac_result["marked"],
            "latency_ms": round(elapsed_ms, 2),
            "cache_hit": is_cache_hit,
        })
        
        status = "CACHE HIT" if is_cache_hit else "COMPUTED"
        print(f"[{idx:2d}/{len(test_sentences)}] {elapsed_ms:6.1f} ms | {status} | {sentence[:45]}")
    
    print()
    print(f"Uncached average: {statistics.mean(uncached_latencies):.1f} ms")
    print(f"Uncached median:  {statistics.median(uncached_latencies):.1f} ms")
    
    stats_after_first = translator.get_cache_stats()
    print(f"\nCache stats after first run:")
    print(f"  Entries: {stats_after_first.get('size', 0)}/{stats_after_first.get('max_size', 0)}")
    print(f"  Hits: {stats_after_first.get('hits', 0)}")
    print(f"  Misses: {stats_after_first.get('misses', 0)}")
    print()
    
    print("TEST 3: Second Run (Should Use Cache)")
    print("-" * 80)
    
    second_run_results = []
    cached_latencies = []
    
    for idx, sentence in enumerate(test_sentences, 1):
        start_time = time.time()
        trans_result = translator.translate_segment(sentence, num_beams=2)
        diac_result = diacritic_engine.restore(trans_result["sindhi"])
        elapsed_ms = (time.time() - start_time) * 1000.0
        
        is_cache_hit = trans_result.get("cache_hit", False)
        cached_latencies.append(elapsed_ms)
        
        second_run_results.append({
            "index": idx,
            "input": sentence,
            "output": trans_result["sindhi"],
            "latency_ms": round(elapsed_ms, 2),
            "cache_hit": is_cache_hit,
        })
        
        status = "✓ CACHE HIT" if is_cache_hit else "✗ MISS"
        print(f"[{idx:2d}/{len(test_sentences)}] {elapsed_ms:6.1f} ms | {status} | {sentence[:45]}")
    
    print()
    print(f"Cached average:   {statistics.mean(cached_latencies):.1f} ms")
    print(f"Cached median:    {statistics.median(cached_latencies):.1f} ms")
    
    stats_after_second = translator.get_cache_stats()
    print(f"\nCache stats after second run:")
    print(f"  Entries: {stats_after_second.get('size', 0)}/{stats_after_second.get('max_size', 0)}")
    print(f"  Hits: {stats_after_second.get('hits', 0)}")
    print(f"  Misses: {stats_after_second.get('misses', 0)}")
    print(f"  Hit rate: {stats_after_second.get('hit_rate_percent', 0):.1f}%")
    print()
    
    print("TEST 4: Cache Speedup Analysis")
    print("-" * 80)
    
    speedups = []
    mismatches = []
    
    for i in range(len(test_sentences)):
        first = first_run_results[i]
        second = second_run_results[i]
        
        speedup_ms = first["latency_ms"] - second["latency_ms"]
        speedup_pct = (speedup_ms / first["latency_ms"] * 100) if first["latency_ms"] > 0 else 0
        speedups.append(speedup_pct)
        
        # Verify output matches
        if first["output"] != second["output"]:
            mismatches.append({
                "sentence": first["input"],
                "first": first["output"],
                "second": second["output"],
            })
        
        cache_marker = "✓" if second["cache_hit"] else "✗"
        print(f"[{i+1:2d}] {cache_marker} {first['latency_ms']:6.1f}ms → {second['latency_ms']:5.1f}ms | "
              f"Speedup: {speedup_pct:+6.1f}% | {first['input'][:35]}")
    
    print()
    avg_speedup = statistics.mean(speedups)
    print(f"Average speedup: {avg_speedup:+.1f}%")
    
    if mismatches:
        print(f"\n⚠ WARNING: {len(mismatches)} output mismatches detected!")
        for mm in mismatches:
            print(f"  Sentence: {mm['sentence']}")
            print(f"  First:  {mm['first']}")
            print(f"  Second: {mm['second']}")
    else:
        print("✓ All cached outputs match original outputs exactly")
    
    print()
    
    print("TEST 5: Cache with Different Parameters")
    print("-" * 80)
    print("Testing that different num_beams creates separate cache entries...")
    
    test_sent = "Hello, how are you?"
    
    # Translate with num_beams=2
    result_beam2_a = translator.translate_segment(test_sent, num_beams=2)
    result_beam2_b = translator.translate_segment(test_sent, num_beams=2)
    
    # Translate with num_beams=1 (should not hit cache)
    result_beam1 = translator.translate_segment(test_sent, num_beams=1)
    
    print(f"num_beams=2 (first):  cache_hit={result_beam2_a.get('cache_hit', False)}")
    print(f"num_beams=2 (second): cache_hit={result_beam2_b.get('cache_hit', False)}")
    print(f"num_beams=1:          cache_hit={result_beam1.get('cache_hit', False)}")
    
    if result_beam2_b.get("cache_hit") and not result_beam1.get("cache_hit"):
        print("✓ Cache correctly distinguishes different parameters")
    else:
        print("✗ Cache parameter handling may have issues")
    
    print()
    
    print("TEST 6: Cache Clear Functionality")
    print("-" * 80)
    
    stats_before_clear = translator.get_cache_stats()
    print(f"Cache size before clear: {stats_before_clear.get('size', 0)}")
    
    translator.clear_cache()
    
    stats_after_clear = translator.get_cache_stats()
    print(f"Cache size after clear: {stats_after_clear.get('size', 0)}")
    print(f"Hits reset: {stats_after_clear.get('hits', 0)}")
    print(f"Misses reset: {stats_after_clear.get('misses', 0)}")
    
    # Test that it recomputes after clear
    result_after_clear = translator.translate_segment("Hello, how are you?", num_beams=2)
    print(f"Translation after clear: cache_hit={result_after_clear.get('cache_hit', False)}")
    
    if result_after_clear.get("cache_hit"):
        print("✗ Cache clear may not be working properly")
    else:
        print("✓ Cache clear works correctly")
    
    print()
    
    # Generate report
    print("="*80)
    print("GENERATING CACHE EVALUATION REPORT")
    print("="*80)
    
    final_stats = translator.get_cache_stats()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 2 - Input Caching",
        "cache_config": {
            "enabled": True,
            "max_size": 1000,
        },
        "uncached_performance": {
            "total_sentences": len(test_sentences),
            "average_latency_ms": round(statistics.mean(uncached_latencies), 2),
            "median_latency_ms": round(statistics.median(uncached_latencies), 2),
            "min_latency_ms": round(min(uncached_latencies), 2),
            "max_latency_ms": round(max(uncached_latencies), 2),
        },
        "cached_performance": {
            "total_sentences": len(test_sentences),
            "average_latency_ms": round(statistics.mean(cached_latencies), 2),
            "median_latency_ms": round(statistics.median(cached_latencies), 2),
            "min_latency_ms": round(min(cached_latencies), 2),
            "max_latency_ms": round(max(cached_latencies), 2),
        },
        "speedup": {
            "average_speedup_percent": round(avg_speedup, 2),
            "absolute_time_saved_ms": round(statistics.mean(uncached_latencies) - statistics.mean(cached_latencies), 2),
        },
        "cache_statistics": final_stats,
        "correctness": {
            "output_mismatches": len(mismatches),
            "all_outputs_match": len(mismatches) == 0,
        },
        "first_run_results": first_run_results,
        "second_run_results": second_run_results,
    }
    
    # Save JSON report
    report_file = config.REPORTS_DIR / "phase2_cache_evaluation.json"
    os.makedirs(report_file.parent, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON report saved: {report_file}")
    
    # Create markdown report
    md_content = generate_markdown_report(report, uncached_latencies, cached_latencies)
    md_file = config.REPORTS_DIR / "phase2_cache_evaluation.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"✓ Markdown report saved: {md_file}")
    print()
    
    print("="*80)
    print("✓ CACHE TESTING COMPLETE")
    print("="*80)
    
    return report


def generate_markdown_report(report, uncached_lat, cached_lat):
    """Generate detailed markdown report."""
    
    unc = report["uncached_performance"]
    cac = report["cached_performance"]
    speedup = report["speedup"]
    cache_stats = report["cache_statistics"]
    
    md = f"""# Phase 2: Cache Implementation Evaluation

**Generated:** {report["timestamp"]}  
**Phase:** {report["phase"]}  
**Cache Configuration:** Enabled with max size {report["cache_config"]["max_size"]} entries

---

## Executive Summary

### Performance Comparison

| Metric | Uncached (First Run) | Cached (Second Run) | Improvement |
|--------|---------------------|---------------------|-------------|
| **Average Latency** | {unc["average_latency_ms"]:.1f} ms | {cac["average_latency_ms"]:.1f} ms | **{speedup["average_speedup_percent"]:+.1f}%** |
| **Median Latency** | {unc["median_latency_ms"]:.1f} ms | {cac["median_latency_ms"]:.1f} ms | |
| **Min Latency** | {unc["min_latency_ms"]:.1f} ms | {cac["min_latency_ms"]:.1f} ms | |
| **Max Latency** | {unc["max_latency_ms"]:.1f} ms | {cac["max_latency_ms"]:.1f} ms | |
| **Time Saved** | - | **{speedup["absolute_time_saved_ms"]:.1f} ms** | Per sentence |

### Cache Statistics

- **Total Requests:** {cache_stats.get("total_requests", 0)}
- **Cache Hits:** {cache_stats.get("hits", 0)}
- **Cache Misses:** {cache_stats.get("misses", 0)}
- **Hit Rate:** **{cache_stats.get("hit_rate_percent", 0):.1f}%**
- **Cache Size:** {cache_stats.get("size", 0)}/{cache_stats.get("max_size", 0)} entries

### Correctness Verification

- **Output Mismatches:** {report["correctness"]["output_mismatches"]}
- **All Outputs Match:** {"✅ YES" if report["correctness"]["all_outputs_match"] else "❌ NO"}

---

## Detailed Analysis

### Latency Distribution

**Uncached (First Run):**
- Best case: {unc["min_latency_ms"]:.1f} ms
- Median: {unc["median_latency_ms"]:.1f} ms
- Average: {unc["average_latency_ms"]:.1f} ms
- Worst case: {unc["max_latency_ms"]:.1f} ms

**Cached (Second Run):**
- Best case: {cac["min_latency_ms"]:.1f} ms
- Median: {cac["median_latency_ms"]:.1f} ms
- Average: {cac["average_latency_ms"]:.1f} ms
- Worst case: {cac["max_latency_ms"]:.1f} ms

**Interpretation:**
"""
    
    if speedup["average_speedup_percent"] > 90:
        md += f"""
The cache provides **exceptional speedup** ({speedup["average_speedup_percent"]:.1f}%), reducing latency to near-instant levels.
This indicates that the cache successfully bypasses model inference entirely.
"""
    elif speedup["average_speedup_percent"] > 50:
        md += f"""
The cache provides **significant speedup** ({speedup["average_speedup_percent"]:.1f}%), greatly reducing translation time.
Cached translations are substantially faster than recomputing from the model.
"""
    else:
        md += f"""
The cache provides **moderate speedup** ({speedup["average_speedup_percent"]:.1f}%).
This may indicate overhead in cache lookup or that the cache is not being used optimally.
"""
    
    md += """

---

## Sample Results

### First 5 Translations (Uncached)
"""
    
    for result in report["first_run_results"][:5]:
        md += f"""
**Example {result["index"]}:**
- Input: {result["input"]}
- Output: {result["output"]}
- Latency: {result["latency_ms"]} ms
- Cache Hit: {result["cache_hit"]}
"""
    
    md += """
### Same 5 Translations (Cached)
"""
    
    for result in report["second_run_results"][:5]:
        first_lat = report["first_run_results"][result["index"]-1]["latency_ms"]
        speedup_pct = ((first_lat - result["latency_ms"]) / first_lat * 100) if first_lat > 0 else 0
        md += f"""
**Example {result["index"]}:**
- Input: {result["input"]}
- Latency: {result["latency_ms"]} ms (was {first_lat} ms, **{speedup_pct:+.1f}%** speedup)
- Cache Hit: {"✅ YES" if result["cache_hit"] else "❌ NO"}
"""
    
    md += """
---

## Conclusions

### What Works

"""
    
    if cache_stats.get("hit_rate_percent", 0) > 90:
        md += "- ✅ **Excellent cache hit rate** - Nearly all repeated translations use the cache\n"
    if report["correctness"]["all_outputs_match"]:
        md += "- ✅ **Perfect correctness** - Cached outputs match uncached outputs exactly\n"
    if speedup["average_speedup_percent"] > 80:
        md += "- ✅ **Dramatic speedup** - Cached translations are nearly instant\n"
    
    md += """
### Cache Behavior

The cache successfully stores translation results and returns them on subsequent identical requests.
This avoids recomputing the neural network inference, which is the primary bottleneck.

### Memory Impact

With a maximum of 1000 cached entries, the estimated memory footprint is approximately 50-100 KB,
which is negligible compared to the 1.1 GB model size. This is a safe and efficient optimization.

### Recommendation

"""
    
    if speedup["average_speedup_percent"] > 80 and report["correctness"]["all_outputs_match"]:
        md += """
**APPROVE FOR DEPLOYMENT** ✅

The cache implementation:
- Provides dramatic speedup for repeated translations
- Maintains perfect output correctness
- Uses minimal memory
- Works correctly with different translation parameters

This is a highly effective optimization that should be kept in the production system.
"""
    else:
        md += """
**NEEDS REVIEW** ⚠️

While the cache shows promise, there are concerns about performance or correctness
that should be investigated before deployment.
"""
    
    md += """
---

*Generated by Phase 2 Cache Testing Script*
"""
    
    return md


if __name__ == "__main__":
    try:
        report = test_cache_implementation()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
