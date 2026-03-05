"""
Isochronic tone generation.

Isochronic tones use amplitude modulation of a carrier frequency with a
square/trapezoidal pulse at the target entrainment rate. Unlike binaural
beats, they do not require headphones as the pulsing is acoustic, not
perceptual.
"""

import numpy as np


def generate_isochronic_envelope(
    pulse_rate_hz: float,
    duration_seconds: float,
    sample_rate: int = 44100,
    duty_cycle: float = 0.5,
    smoothing_ms: float = 5.0,
) -> np.ndarray:
    """
    Generate an amplitude envelope for isochronic tone modulation.

    Creates a smoothed square wave that pulses at the target entrainment
    frequency. The smoothing prevents harsh clicks at pulse boundaries.

    Args:
        pulse_rate_hz: Rate of on/off pulsing in Hz (entrainment target).
        duration_seconds: Length of envelope in seconds.
        sample_rate: Audio sample rate.
        duty_cycle: Fraction of each period where tone is ON (0.0 to 1.0).
        smoothing_ms: Rise/fall time in milliseconds for pulse edges.

    Returns:
        NumPy array of amplitude values (0.0 to 1.0).
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Generate base square wave at the pulse rate
    phase = (t * pulse_rate_hz) % 1.0
    envelope = (phase < duty_cycle).astype(np.float64)

    # Apply smoothing to edges using a short raised-cosine ramp
    smooth_samples = max(1, int(smoothing_ms * sample_rate / 1000.0))
    if smooth_samples > 1:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, smooth_samples)))
        kernel = np.ones(smooth_samples)
        kernel[:len(ramp)] = ramp
        kernel = kernel / kernel.sum()
        envelope = np.convolve(envelope, kernel, mode="same")

    return envelope


def generate_isochronic_tone(
    carrier_freq_hz: float,
    pulse_rate_hz: float,
    duration_seconds: float,
    amplitude: float = 0.5,
    sample_rate: int = 44100,
    duty_cycle: float = 0.5,
) -> np.ndarray:
    """
    Generate a complete isochronic tone as a stereo audio signal.

    The carrier sine wave is multiplied by a pulsing envelope at the
    target entrainment frequency.

    Args:
        carrier_freq_hz: Frequency of the audible tone in Hz.
        pulse_rate_hz: Rate of amplitude pulsing in Hz.
        duration_seconds: Length in seconds.
        amplitude: Peak amplitude (0.0 to 1.0).
        sample_rate: Audio sample rate.
        duty_cycle: Fraction of pulse period where tone is on.

    Returns:
        NumPy array of shape (num_samples, 2) for stereo output.
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.arange(num_samples) / sample_rate

    carrier = np.sin(2.0 * np.pi * carrier_freq_hz * t)
    envelope = generate_isochronic_envelope(
        pulse_rate_hz, duration_seconds, sample_rate, duty_cycle
    )

    mono = amplitude * carrier * envelope
    return np.column_stack((mono, mono))
