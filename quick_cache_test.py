"""
Quick cache verification test - simpler version
"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.translation.translator import SindhiTranslator

print("Quick Cache Test")
print("="*60)

# Create translator with cache
print("\n1. Creating translator with cache enabled...")
translator = SindhiTranslator(lazy_init=False, enable_cache=True)
print("   ✓ Translator created")

# Check cache stats
stats = translator.get_cache_stats()
print(f"\n2. Initial cache stats:")
print(f"   Max size: {stats.get('max_size', 0)}")
print(f"   Current size: {stats.get('size', 0)}")
print(f"   Enabled: {stats.get('max_size', 0) > 0}")

# First translation
print("\n3. First translation (uncached)...")
test_text = "Hello, how are you?"
start = time.time()
result1 = translator.translate_segment(test_text, num_beams=2)
elapsed1 = (time.time() - start) * 1000
print(f"   Input: {test_text}")
print(f"   Output: {result1['sindhi']}")
print(f"   Latency: {elapsed1:.1f} ms")
print(f"   Cache hit: {result1.get('cache_hit', False)}")

# Check cache
stats = translator.get_cache_stats()
print(f"\n4. Cache stats after first translation:")
print(f"   Size: {stats.get('size', 0)}")
print(f"   Hits: {stats.get('hits', 0)}")
print(f"   Misses: {stats.get('misses', 0)}")

# Second translation (should hit cache)
print("\n5. Second translation (should be cached)...")
start = time.time()
result2 = translator.translate_segment(test_text, num_beams=2)
elapsed2 = (time.time() - start) * 1000
print(f"   Output: {result2['sindhi']}")
print(f"   Latency: {elapsed2:.1f} ms")
print(f"   Cache hit: {result2.get('cache_hit', False)}")

# Verify outputs match
print(f"\n6. Verification:")
print(f"   Outputs match: {result1['sindhi'] == result2['sindhi']}")
print(f"   Speedup: {((elapsed1 - elapsed2) / elapsed1 * 100):.1f}%")
print(f"   Time saved: {(elapsed1 - elapsed2):.1f} ms")

# Final cache stats
stats = translator.get_cache_stats()
print(f"\n7. Final cache stats:")
print(f"   Size: {stats.get('size', 0)}")
print(f"   Hits: {stats.get('hits', 0)}")
print(f"   Misses: {stats.get('misses', 0)}")
print(f"   Hit rate: {stats.get('hit_rate_percent', 0):.1f}%")

print("\n" + "="*60)
print("✓ Quick test complete")
