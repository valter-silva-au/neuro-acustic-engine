"""
Colored noise generators using FFT-based spectral shaping.

Generates white, pink (1/f), and brown (1/f^2) noise profiles for
use as ambient background textures in the cognitive audio engine.
"""

import numpy as np


def generate_white_noise(
    duration_seconds: float,
    amplitude: float = 0.3,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Generate white noise (flat power spectral density).

    Args:
        duration_seconds: Length in seconds.
        amplitude: Peak amplitude (0.0 to 1.0).
        sample_rate: Audio sample rate.

    Returns:
        Stereo NumPy array of shape (num_samples, 2).
    """
    num_samples = int(duration_seconds * sample_rate)
    noise = np.random.randn(num_samples) * amplitude
    # Clip to prevent rare outliers from exceeding amplitude bounds
    noise = np.clip(noise, -amplitude, amplitude)
    return np.column_stack((noise, noise))


def generate_pink_noise(
    duration_seconds: float,
    amplitude: float = 0.3,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Generate pink noise (1/f power spectral density, -3 dB/octave).

    Uses FFT-based spectral shaping: generate white noise in the frequency
    domain, apply 1/sqrt(f) amplitude scaling, then IFFT back to time domain.

    Args:
        duration_seconds: Length in seconds.
        amplitude: Peak amplitude (0.0 to 1.0).
        sample_rate: Audio sample rate.

    Returns:
        Stereo NumPy array of shape (num_samples, 2).
    """
    num_samples = int(duration_seconds * sample_rate)

    # Generate white noise in frequency domain
    white_spectrum = np.fft.rfft(np.random.randn(num_samples))

    # Create 1/sqrt(f) filter (pink = 1/f power, so amplitude scales as 1/sqrt(f))
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    # Avoid division by zero at DC
    freqs[0] = 1.0
    pink_filter = 1.0 / np.sqrt(freqs)

    # Apply filter and convert back to time domain
    pink_spectrum = white_spectrum * pink_filter
    noise = np.fft.irfft(pink_spectrum, n=num_samples)

    # Normalize to target amplitude
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak * amplitude

    return np.column_stack((noise, noise))


def generate_brown_noise(
    duration_seconds: float,
    amplitude: float = 0.3,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Generate brown/Brownian noise (1/f^2 power spectral density, -6 dB/octave).

    Uses FFT-based spectral shaping with 1/f amplitude scaling.

    Args:
        duration_seconds: Length in seconds.
        amplitude: Peak amplitude (0.0 to 1.0).
        sample_rate: Audio sample rate.

    Returns:
        Stereo NumPy array of shape (num_samples, 2).
    """
    num_samples = int(duration_seconds * sample_rate)

    # Generate white noise in frequency domain
    white_spectrum = np.fft.rfft(np.random.randn(num_samples))

    # Create 1/f filter (brown = 1/f^2 power, so amplitude scales as 1/f)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    freqs[0] = 1.0
    brown_filter = 1.0 / freqs

    # Apply filter and convert back to time domain
    brown_spectrum = white_spectrum * brown_filter
    noise = np.fft.irfft(brown_spectrum, n=num_samples)

    # Normalize to target amplitude
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak * amplitude

    return np.column_stack((noise, noise))


def generate_noise(
    color: str,
    duration_seconds: float,
    amplitude: float = 0.3,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Generate colored noise by name.

    Args:
        color: One of "white", "pink", "brown".
        duration_seconds: Length in seconds.
        amplitude: Peak amplitude (0.0 to 1.0).
        sample_rate: Audio sample rate.

    Returns:
        Stereo NumPy array of shape (num_samples, 2).

    Raises:
        ValueError: If color is not recognized.
    """
    generators = {
        "white": generate_white_noise,
        "pink": generate_pink_noise,
        "brown": generate_brown_noise,
    }
    generator = generators.get(color)
    if generator is None:
        raise ValueError(f"Unknown noise color: {color!r}. Use 'white', 'pink', or 'brown'.")
    return generator(duration_seconds, amplitude, sample_rate)
