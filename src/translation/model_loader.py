"""
Model loader module for English-to-Sindhi translation.
Provides lazy-loading, memory tracking, and strictly offline local model loading.
"""

import os
import time
import psutil
from pathlib import Path
from typing import Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import config


class TranslationModelLoader:
    """Manages loading and lifecycle of Seq2Seq translation models."""

    _instance = None
    _model = None
    _tokenizer = None
    _load_time_sec = 0.0
    _model_path_used = ""

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TranslationModelLoader()
        return cls._instance

    @staticmethod
    def get_memory_usage_mb() -> float:
        """Returns the current process RSS memory in Megabytes."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @classmethod
    def find_local_model_path(cls) -> Optional[Path]:
        """Checks for existing local model weights."""
        candidate_paths = [
            config.TRANSLATION_MODEL_DIR / "nllb-200-distilled-600M",
            config.TRANSLATION_MODEL_DIR,
        ]
        for path in candidate_paths:
            if path.exists() and (
                (path / "config.json").exists()
                and (
                    (path / "pytorch_model.bin").exists()
                    or (path / "model.safetensors").exists()
                )
            ):
                return path
        return None

    @classmethod
    def load_model_and_tokenizer(
        cls,
        force_reload: bool = False,
        local_only: bool = False,
    ) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """
        Loads the translation model and tokenizer with CPU-optimized settings.
        Supports offline loading via local_files_only.
        """
        if cls._model is not None and cls._tokenizer is not None and not force_reload:
            return cls._model, cls._tokenizer

        start_time = time.time()
        start_mem = cls.get_memory_usage_mb()

        # Set PyTorch CPU thread count to prevent resource starvation
        torch.set_num_threads(config.CPU_INFERENCE_NUM_THREADS)

        local_path = cls.find_local_model_path()
        model_identifier = str(local_path) if local_path else config.DEFAULT_TRANSLATION_MODEL

        # If forcing offline or local_only, require local_path
        if (config.FORCE_OFFLINE or local_only) and not local_path:
            raise FileNotFoundError(
                f"Offline mode is active, but local translation model weights were not found at "
                f"{config.TRANSLATION_MODEL_DIR / 'nllb-200-distilled-600M'}. "
                f"Please download the weights into that directory first."
            )

        load_kwargs = {}
        if local_path or config.FORCE_OFFLINE or local_only:
            load_kwargs["local_files_only"] = True

        print(f"[ModelLoader] Loading tokenizer from: {model_identifier}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_identifier,
            src_lang=config.SRC_LANG_CODE,
            **load_kwargs,
        )

        print(f"[ModelLoader] Loading model weights from: {model_identifier} on CPU...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_identifier,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )

        # Set to eval mode for deterministic inference
        model.eval()

        cls._model = model
        cls._tokenizer = tokenizer
        cls._model_path_used = model_identifier
        cls._load_time_sec = time.time() - start_time
        end_mem = cls.get_memory_usage_mb()

        print(
            f"[ModelLoader] Model loaded in {cls._load_time_sec:.2f}s. "
            f"RAM consumed: {end_mem - start_mem:.1f} MB (Total process RSS: {end_mem:.1f} MB)"
        )

        return cls._model, cls._tokenizer

    @classmethod
    def get_status(cls) -> dict:
        """Returns loaded model status and diagnostics."""
        local_path = cls.find_local_model_path()
        return {
            "loaded": cls._model is not None,
            "model_path": cls._model_path_used,
            "local_weights_available": local_path is not None,
            "local_path": str(local_path) if local_path else None,
            "load_time_sec": cls._load_time_sec,
            "process_rss_mb": cls.get_memory_usage_mb(),
            "cpu_threads": torch.get_num_threads(),
        }

    @classmethod
    def unload(cls):
        """Frees model from RAM to support low-memory environments."""
        cls._model = None
        cls._tokenizer = None
        import gc
        gc.collect()
        print("[ModelLoader] Model unloaded from memory.")
