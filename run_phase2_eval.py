"""
Phase 2 Cache Evaluation - Focused test matching baseline evaluation
"""
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

# Same test sentences as baseline
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

print("="*80)
print("PHASE 2: CACHE EVALUATION (Matching Baseline)")
print("="*80)
print(f"Testing {len(test_sentences)} sentences")
print()

# Initialize
translator = SindhiTranslator(lazy_init=False, enable_cache=True, cache_size=1000)
diacritic_engine = SindhiDiacriticRestorer()

print("ROUND 1: First translations (uncached)")
print("-"*80)

uncached_results = []
for idx, sent in enumerate(test_sentences, 1):
    start = time.time()
    trans = translator.translate_segment(sent, num_beams=2)
    diac = diacritic_engine.restore(trans["sindhi"])
    elapsed = (time.time() - start) * 1000
    
    uncached_results.append({
        "input": sent,
        "output": trans["sindhi"],
        "output_marked": diac["marked"],
        "latency_ms": round(elapsed, 2),
        "cache_hit": trans.get("cache_hit", False)
    })
    
    print(f"[{idx:2d}/15] {elapsed:6.1f} ms | {sent[:50]}")

uncached_lats = [r["latency_ms"] for r in uncached_results]

print()
print(f"Uncached avg: {statistics.mean(uncached_lats):.1f} ms")
print(f"Uncached med: {statistics.median(uncached_lats):.1f} ms")

stats = translator.get_cache_stats()
print(f"\nCache: {stats['size']} entries, {stats['hits']} hits, {stats['misses']} misses")
print()

print("ROUND 2: Repeat translations (should be cached)")
print("-"*80)

cached_results = []
for idx, sent in enumerate(test_sentences, 1):
    start = time.time()
    trans = translator.translate_segment(sent, num_beams=2)
    diac = diacritic_engine.restore(trans["sindhi"])
    elapsed = (time.time() - start) * 1000
    
    cached_results.append({
        "input": sent,
        "output": trans["sindhi"],
        "latency_ms": round(elapsed, 2),
        "cache_hit": trans.get("cache_hit", False)
    })
    
    marker = "✓" if trans.get("cache_hit") else "✗"
    print(f"[{idx:2d}/15] {marker} {elapsed:5.1f} ms | {sent[:50]}")

cached_lats = [r["latency_ms"] for r in cached_results]

print()
print(f"Cached avg: {statistics.mean(cached_lats):.1f} ms")
print(f"Cached med: {statistics.median(cached_lats):.1f} ms")

stats = translator.get_cache_stats()
print(f"\nCache: {stats['size']} entries, {stats['hits']} hits, {stats['misses']} misses")
print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
print()

# Verify correctness
mismatches = sum(1 for i in range(len(test_sentences)) 
                 if uncached_results[i]["output"] != cached_results[i]["output"])

print("RESULTS")
print("="*80)
print(f"Uncached avg latency: {statistics.mean(uncached_lats):.1f} ms")
print(f"Cached avg latency:   {statistics.mean(cached_lats):.1f} ms")
speedup = (statistics.mean(uncached_lats) - statistics.mean(cached_lats)) / statistics.mean(uncached_lats) * 100
print(f"Average speedup:      {speedup:.1f}%")
print(f"Time saved per translation: {statistics.mean(uncached_lats) - statistics.mean(cached_lats):.1f} ms")
print(f"Output mismatches:    {mismatches} (should be 0)")
print(f"Cache hit rate:       {stats['hit_rate_percent']:.1f}%")
print()

# Save report
report = {
    "timestamp": datetime.now().isoformat(),
    "phase": "Phase 2 - Input Caching",
    "test_sentences": len(test_sentences),
    "uncached_performance": {
        "average_ms": round(statistics.mean(uncached_lats), 2),
        "median_ms": round(statistics.median(uncached_lats), 2),
        "min_ms": round(min(uncached_lats), 2),
        "max_ms": round(max(uncached_lats), 2),
    },
    "cached_performance": {
        "average_ms": round(statistics.mean(cached_lats), 2),
        "median_ms": round(statistics.median(cached_lats), 2),
        "min_ms": round(min(cached_lats), 2),
        "max_ms": round(max(cached_lats), 2),
    },
    "speedup_percent": round(speedup, 2),
    "time_saved_ms": round(statistics.mean(uncached_lats) - statistics.mean(cached_lats), 2),
    "output_mismatches": mismatches,
    "cache_stats": stats,
    "uncached_results": uncached_results,
    "cached_results": cached_results,
}

report_file = config.REPORTS_DIR / "phase2_cache_evaluation.json"
with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"Report saved: {report_file}")
print("✓ Phase 2 evaluation complete")
