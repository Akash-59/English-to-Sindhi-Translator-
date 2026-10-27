"""Memory safety checks shared by the app and isolated ASR worker."""

import psutil


MINIMUM_AVAILABLE_RAM_GIB = 3.0
MINIMUM_AVAILABLE_RAM_BYTES = int(MINIMUM_AVAILABLE_RAM_GIB * (1024 ** 3))


def minimum_available_ram_bytes() -> int:
    """Return the fixed conservative minimum; it cannot be lowered by configuration."""
    return MINIMUM_AVAILABLE_RAM_BYTES


def available_ram_bytes() -> int:
    """Return current system memory available for an ASR process."""
    return psutil.virtual_memory().available


def require_available_ram(available_bytes: int | None = None) -> None:
    """Raise a clear error when free system memory is below the configured threshold."""
    minimum_bytes = minimum_available_ram_bytes()
    if available_bytes is None:
        available_bytes = available_ram_bytes()
    if available_bytes < minimum_bytes:
        available_gib = available_bytes / (1024 ** 3)
        minimum_gib = minimum_bytes / (1024 ** 3)
        raise RuntimeError(
            "Canopy-M transcription requires at least "
            f"{minimum_gib:.1f} GiB of available RAM; "
            f"currently available: {available_gib:.2f} GiB."
        )
