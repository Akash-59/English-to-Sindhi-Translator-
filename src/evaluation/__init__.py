"""Evaluation and benchmarking package."""
from src.evaluation.evaluate_translation import evaluate_translation_system
from src.evaluation.benchmark_inference import run_benchmark

__all__ = ["evaluate_translation_system", "run_benchmark"]
