"""
Translator module for English-to-Sindhi translation.
Handles text preprocessing, batching, sentence/paragraph segmentation,
Seq2Seq generation, and right-to-left Perso-Arabic Sindhi output formatting.
"""

import time
import unicodedata
import re
from typing import List, Dict, Any, Optional

import torch

import config
from src.translation.model_loader import TranslationModelLoader


class SindhiTranslator:
    """English-to-Sindhi translation engine based on verified multilingual Seq2Seq models."""

    def __init__(self, lazy_init: bool = True):
        self.loader = TranslationModelLoader
        self.model = None
        self.tokenizer = None
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
        """Translates a single English segment into Sindhi."""
        cleaned = self.clean_english_input(text)
        if not cleaned:
            return {"english": "", "sindhi": "", "latency_ms": 0.0}

        self._ensure_loaded()
        start_time = time.time()

        # Tokenize with forced source language
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
        )

        # Get target language token ID for snd_Arab
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(config.TGT_LANG_CODE)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        # Decode output
        translated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        # Post-process Sindhi text
        translated_text = unicodedata.normalize("NFC", translated_text.strip())
        
        # Ensure proper Sindhi full-stop if source had full-stop
        if cleaned.endswith(".") and not (translated_text.endswith("۔") or translated_text.endswith(".")):
            translated_text += "۔"

        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "english": cleaned,
            "sindhi": translated_text,
            "latency_ms": round(latency_ms, 2),
            "source_tokens": int(inputs["input_ids"].shape[1]),
            "target_tokens": int(generated_tokens.shape[1]),
        }

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
