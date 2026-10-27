"""
Translator module for bidirectional English-Sindhi translation.
Handles text preprocessing, batching, sentence/paragraph segmentation,
Seq2Seq generation, and right-to-left Perso-Arabic Sindhi output formatting.

Phase 2 Enhancement: Added bounded in-memory LRU cache for repeated translations.
"""

import time
import unicodedata
import re
import copy
from threading import RLock
from typing import List, Dict, Any, Optional
from collections import OrderedDict

import torch

import config
from src.translation.model_loader import TranslationModelLoader


class TranslationCache:
    """
    Bounded LRU (Least Recently Used) cache for translation results.
    Thread-safe for single-process use. Memory-bounded by entry count.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache with maximum size.
        
        Args:
            max_size: Maximum number of cached translations (default: 1000)
                     Estimated memory: ~50-100 bytes per entry = ~50-100 KB total
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _make_key(
        self,
        text: str,
        num_beams: int,
        max_length: int,
        num_return_sequences: int = 1,
        operation: str = "single",
    ) -> str:
        """Create cache key from input parameters."""
        # Use cleaned text and parameters to ensure identical settings
        return f"{operation}|{text}|{num_beams}|{max_length}|{num_return_sequences}"

    def get(
        self,
        text: str,
        num_beams: int,
        max_length: int,
        num_return_sequences: int = 1,
        operation: str = "single",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached translation if available.
        
        Returns:
            Cached result dict or None if not found
        """
        key = self._make_key(text, num_beams, max_length, num_return_sequences, operation)
        
        if key in self._cache:
            self._hits += 1
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            # Return a copy to prevent external modification
            return copy.deepcopy(self._cache[key])
        else:
            self._misses += 1
            return None
    
    def put(
        self,
        text: str,
        num_beams: int,
        max_length: int,
        result: Dict[str, Any],
        num_return_sequences: int = 1,
        operation: str = "single",
    ) -> None:
        """
        Store translation result in cache.
        
        Args:
            text: Cleaned input text
            num_beams: Beam search parameter
            max_length: Max generation length
            result: Translation result dictionary
        """
        key = self._make_key(text, num_beams, max_length, num_return_sequences, operation)
        
        # Add or update
        self._cache[key] = copy.deepcopy(result)
        self._cache.move_to_end(key)
        
        # Evict oldest if over limit
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total,
        }


class SindhiTranslator:
    """English-to-Sindhi translation engine based on verified multilingual Seq2Seq models."""

    def __init__(self, lazy_init: bool = True, enable_cache: bool = True, cache_size: int = 1000):
        """
        Initialize translator with optional caching.
        
        Args:
            lazy_init: If True, load model on first use (default: True)
            enable_cache: If True, cache translation results (default: True)
            cache_size: Maximum number of cached translations (default: 1000)
        """
        self.loader = TranslationModelLoader
        self.model = None
        self.tokenizer = None
        self.enable_cache = enable_cache
        self._generation_lock = RLock()
        
        # Initialize cache if enabled
        if self.enable_cache:
            self._cache = TranslationCache(max_size=cache_size)
        else:
            self._cache = None
        
        if not lazy_init:
            self._ensure_loaded()

    def _ensure_loaded(self):
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = self.loader.load_model_and_tokenizer()

    def clean_english_input(self, text: str) -> str:
        """Preprocesses and normalizes English input text."""
        if not text:
            return ""
        # Normalize Unicode
        text = unicodedata.normalize("NFC", text)
        # Normalize multiple spaces but preserve single newlines
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def split_into_segments(self, text: str) -> List[str]:
        """Splits multi-paragraph or multi-sentence text to prevent GPU/RAM memory spikes."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            return []
        
        segments = []
        for para in paragraphs:
            # If paragraph is long, split on sentence boundaries (. ! ?)
            if len(para) > 300:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                segments.extend([s.strip() for s in sentences if s.strip()])
            else:
                segments.append(para)
        return segments

    def translate_segment(
        self,
        text: str,
        num_beams: int = 2,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        Translates a single English segment into Sindhi.
        Uses cache if enabled and entry exists.
        """
        return self._translate_segment(
            text,
            source_lang=config.SRC_LANG_CODE,
            target_lang=config.TGT_LANG_CODE,
            input_key="english",
            output_key="sindhi",
            operation="single",
            num_beams=num_beams,
            max_length=max_length,
        )

    def _translate_segment(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        input_key: str,
        output_key: str,
        operation: str,
        num_beams: int,
        max_length: int,
    ) -> Dict[str, Any]:
        cleaned = self.clean_english_input(text)
        if not cleaned:
            return {
                input_key: "",
                output_key: "",
                "latency_ms": 0.0,
                "cache_hit": False,
            }

        if self._cache is not None:
            cached_result = self._cache.get(
                cleaned,
                num_beams,
                max_length,
                operation=operation,
            )
            if cached_result is not None:
                cached_result["cache_hit"] = True
                return cached_result

        self._ensure_loaded()
        start_time = time.time()

        with self._generation_lock:
            self.tokenizer.src_lang = source_lang
            inputs = self.tokenizer(
                cleaned,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_INPUT_LENGTH,
            )
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

        translated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        translated_text = unicodedata.normalize("NFC", translated_text.strip())
        if (
            target_lang == config.TGT_LANG_CODE
            and cleaned.endswith(".")
            and not (translated_text.endswith("۔") or translated_text.endswith("."))
        ):
            translated_text += "۔"

        latency_ms = (time.time() - start_time) * 1000.0

        result = {
            input_key: cleaned,
            output_key: translated_text,
            "latency_ms": round(latency_ms, 2),
            "source_tokens": int(inputs["input_ids"].shape[1]),
            "target_tokens": int(generated_tokens.shape[1]),
            "cache_hit": False,
        }

        if self._cache is not None:
            self._cache.put(
                cleaned,
                num_beams,
                max_length,
                result,
                operation=operation,
            )

        return result

    def translate_sindhi_segment(
        self,
        text: str,
        num_beams: int = 2,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """Translates a single Sindhi segment into English."""
        return self._translate_segment(
            text,
            source_lang=config.TGT_LANG_CODE,
            target_lang=config.SRC_LANG_CODE,
            input_key="sindhi",
            output_key="english",
            operation="sindhi_to_english",
            num_beams=num_beams,
            max_length=max_length,
        )

    def translate_sindhi_to_english(
        self,
        text: str,
        num_beams: int = 2,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """Translates Sindhi text to English while preserving paragraph breaks."""
        cleaned = self.clean_english_input(text)
        if not cleaned:
            return {
                "sindhi": "",
                "english": "",
                "segments": [],
                "latency_ms": 0.0,
                "character_count": 0,
                "word_count": 0,
            }

        start_total = time.time()
        translated_lines = []
        segments_info = []

        for line in text.split("\n"):
            if not line.strip():
                translated_lines.append("")
                continue

            segment_result = self.translate_sindhi_segment(
                line.strip(),
                num_beams=num_beams,
                max_length=max_length,
            )
            translated_lines.append(segment_result["english"])
            segments_info.append(segment_result)

        full_english = "\n".join(translated_lines)
        total_latency_ms = (time.time() - start_total) * 1000.0

        return {
            "sindhi": cleaned,
            "english": full_english,
            "segments": segments_info,
            "latency_ms": round(total_latency_ms, 2),
            "character_count": len(full_english),
            "word_count": len(full_english.split()),
        }

    def translate_candidates(
        self,
        text: str,
        num_candidates: int = 3,
        num_beams: int = 4,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """Generates distinct beam-search candidates, ordered by model sequence score."""
        if not 1 <= num_candidates <= 4:
            raise ValueError("num_candidates must be between 1 and 4.")
        if num_beams < num_candidates:
            raise ValueError("num_beams must be at least num_candidates.")
        if max_length < 1:
            raise ValueError("max_length must be positive.")

        cleaned = self.clean_english_input(text)
        if not cleaned:
            return {
                "english": "",
                "candidates": [],
                "latency_ms": 0.0,
                "cache_hit": False,
            }

        if self._cache is not None:
            cached_result = self._cache.get(
                cleaned,
                num_beams,
                max_length,
                num_return_sequences=num_candidates,
                operation="candidates",
            )
            if cached_result is not None:
                cached_result["cache_hit"] = True
                return cached_result

        self._ensure_loaded()
        start_time = time.time()
        with self._generation_lock:
            self.tokenizer.src_lang = config.SRC_LANG_CODE
            inputs = self.tokenizer(
                cleaned,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_INPUT_LENGTH,
            )
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(config.TGT_LANG_CODE)

            with torch.no_grad():
                generated_sequences = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_candidates,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

        decoded = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True,
        )
        candidates = []
        seen = set()
        for candidate in decoded:
            candidate = unicodedata.normalize("NFC", candidate.strip())
            if cleaned.endswith(".") and not candidate.endswith(("۔", ".", "!", "?", "؟")):
                candidate += "۔"
            candidate = re.sub(r"\s+", " ", candidate).strip()
            duplicate_key = candidate.rstrip("۔.!?؟،,").casefold()
            if candidate and (not duplicate_key or duplicate_key not in seen):
                seen.add(duplicate_key)
                candidates.append(candidate)

        latency_ms = (time.time() - start_time) * 1000.0
        result = {
            "english": cleaned,
            "candidates": candidates,
            "latency_ms": round(latency_ms, 2),
            "source_tokens": int(inputs["input_ids"].shape[1]),
            "candidate_count": len(candidates),
            "cache_hit": False,
        }
        if self._cache is not None:
            self._cache.put(
                cleaned,
                num_beams,
                max_length,
                result,
                num_return_sequences=num_candidates,
                operation="candidates",
            )
        return result

    def translate(
        self,
        text: str,
        num_beams: int = 2,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        Translates English text (single word, sentence, or document) to Sindhi.
        Preserves paragraph structure and measures overall latency.
        """
        cleaned = self.clean_english_input(text)
        if not cleaned:
            return {
                "english": "",
                "sindhi": "",
                "segments": [],
                "latency_ms": 0.0,
                "character_count": 0,
                "word_count": 0,
            }

        start_total = time.time()
        
        # Check if text contains multiple paragraphs
        lines = text.split("\n")
        translated_lines = []
        segments_info = []

        for line in lines:
            if not line.strip():
                translated_lines.append("")
                continue

            seg_res = self.translate_segment(line.strip(), num_beams=num_beams, max_length=max_length)
            translated_lines.append(seg_res["sindhi"])
            segments_info.append(seg_res)

        full_sindhi = "\n".join(translated_lines)
        total_latency_ms = (time.time() - start_total) * 1000.0

        return {
            "english": cleaned,
            "sindhi": full_sindhi,
            "segments": segments_info,
            "latency_ms": round(total_latency_ms, 2),
            "character_count": len(full_sindhi),
            "word_count": len(full_sindhi.split()),
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache performance metrics or None if cache disabled
        """
        if self._cache is not None:
            return self._cache.stats()
        return {"enabled": False, "message": "Cache is disabled"}
    
    def clear_cache(self) -> None:
        """Clear all cached translations."""
        if self._cache is not None:
            self._cache.clear()
