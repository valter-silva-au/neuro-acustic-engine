"""
Binaural beat calculation helpers.

Provides pure-math utilities for computing left/right channel frequencies
from a carrier frequency and desired beat frequency.
"""

from neuroacoustic.core.config import BRAINWAVE_BANDS


def calculate_binaural_frequencies(
    carrier_freq: float, beat_freq: float
) -> tuple[float, float]:
    """
    Calculate the discrete left and right ear frequencies for a binaural beat.

    The brain perceives a phantom beat at |f_right - f_left| Hz when two
    slightly different tones are presented to each ear via stereo isolation.

    Args:
        carrier_freq: The center (perceived pitch) frequency in Hz.
        beat_freq: The desired entrainment beat frequency in Hz.

    Returns:
        Tuple of (left_freq_hz, right_freq_hz).
    """
    left_hz = carrier_freq - (beat_freq / 2.0)
    right_hz = carrier_freq + (beat_freq / 2.0)
    return (left_hz, right_hz)


def classify_beat_band(beat_freq: float) -> str | None:
    """
    Classify a beat frequency into its brainwave band name.

    Args:
        beat_freq: The beat frequency in Hz.

    Returns:
        Band name (e.g. "theta", "alpha") or None if out of range.
    """
    for band_name, band_info in BRAINWAVE_BANDS.items():
        low, high = band_info["range_hz"]
        if low <= beat_freq < high:
            return band_name
    return None


def get_default_carrier_for_band(band_name: str) -> float | None:
    """
    Get the recommended carrier frequency for a brainwave band.

    Args:
        band_name: One of "delta", "theta", "alpha", "beta", "gamma".

    Returns:
        Default carrier frequency in Hz, or None if band unknown.
    """
    band = BRAINWAVE_BANDS.get(band_name)
    if band is None:
        return None
    return band["default_carrier_hz"]
