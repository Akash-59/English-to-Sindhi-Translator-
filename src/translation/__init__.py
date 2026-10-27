"""Translation package initialization."""

__all__ = ["TranslationModelLoader", "SindhiTranslator"]


def __getattr__(name):
    if name == "TranslationModelLoader":
        from src.translation.model_loader import TranslationModelLoader

        return TranslationModelLoader
    if name == "SindhiTranslator":
        from src.translation.translator import SindhiTranslator

        return SindhiTranslator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
