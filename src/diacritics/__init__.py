"""Diacritics restoration package."""
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer
from src.diacritics.train_diacritics import train_diacritic_model
from src.diacritics.evaluate_diacritics import evaluate_diacritics

__all__ = ["SindhiDiacriticRestorer", "train_diacritic_model", "evaluate_diacritics"]
