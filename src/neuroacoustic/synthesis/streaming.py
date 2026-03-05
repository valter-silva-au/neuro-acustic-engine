"""
Streaming WAV writer for long-duration audio generation.

Generates audio in small chunks (default 30s) and writes them
sequentially to a WAV file, allowing multi-hour generation without
exceeding memory limits.

8 hours at 22050 Hz stereo 16-bit = ~2.4 GB on disk (within WAV 4GB limit).
In-memory chunk at 30s = ~5 MB — trivially small.
"""

import wave
from dataclasses import dataclass

import numpy as np


DEFAULT_SAMPLE_RATE = 22050  # 22050 Hz keeps 8h files under WAV's 4GB limit
DEFAULT_CHUNK_SECONDS = 30.0


@dataclass
class PhaseState:
    """Track oscillator phase across chunks for seamless continuity."""
    phase: float = 0.0

    def advance(self, freq: float, num_samples: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """
        Generate a sine wave continuing from the current phase.

        This ensures zero discontinuity between chunks — the waveform
        is mathematically continuous across chunk boundaries.
        """
        t = np.arange(num_samples) / sample_rate
        signal = np.sin(2 * np.pi * freq * t + self.phase)
        # Advance phase for next chunk
        self.phase += 2 * np.pi * freq * num_samples / sample_rate
        # Keep phase in [0, 2π) to prevent floating-point drift
        self.phase %= (2 * np.pi)
        return signal


class StreamingWavWriter:
    """
    Write WAV files in streaming chunks to support multi-hour generation.

    Usage:
        with StreamingWavWriter("output.wav") as writer:
            for chunk in generate_chunks():
                writer.write_chunk(chunk)
    """

    def __init__(self, filepath: str, sample_rate: int = DEFAULT_SAMPLE_RATE, channels: int = 2):
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.channels = channels
        self._wf = None
        self._total_frames = 0

    def __enter__(self):
        self._wf = wave.open(self.filepath, "w")
        self._wf.setnchannels(self.channels)
        self._wf.setsampwidth(2)  # 16-bit
        self._wf.setframerate(self.sample_rate)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._wf:
            self._wf.close()

    def write_chunk(self, chunk: np.ndarray) -> None:
        """
        Write a stereo float chunk to the WAV file.

        Args:
            chunk: numpy array of shape (num_samples, 2), values in [-1, 1].
        """
        clipped = np.clip(chunk, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16)
        self._wf.writeframes(int_data.tobytes())
        self._total_frames += len(chunk)

    @property
    def duration_seconds(self) -> float:
        return self._total_frames / self.sample_rate


def raised_cosine_fade(num_samples: int) -> np.ndarray:
    """
    Generate a raised-cosine (Hann) fade curve.

    Unlike a linear fade, a raised cosine has zero derivative at both
    endpoints, producing a perceptually smooth onset with no audible
    "ramp" artifact or stuttering.
    """
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(num_samples) / num_samples))


def generate_binaural_chunk(
    left_phase: PhaseState,
    right_phase: PhaseState,
    carrier_freq: float,
    beat_freq: float,
    num_samples: int,
    amplitude: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Generate one chunk of phase-continuous binaural beat."""
    left_freq = carrier_freq - (beat_freq / 2.0)
    right_freq = carrier_freq + (beat_freq / 2.0)
    left = amplitude * left_phase.advance(left_freq, num_samples, sample_rate)
    right = amplitude * right_phase.advance(right_freq, num_samples, sample_rate)
    return np.column_stack((left, right))


def generate_noise_chunk(
    color: str,
    num_samples: int,
    amplitude: float = 0.3,
    prev_tail: np.ndarray | None = None,
    crossfade_samples: int = 1024,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one chunk of colored noise with inter-chunk crossfading.

    Returns (chunk, tail) where tail should be passed as prev_tail
    to the next call for seamless transitions.
    """
    from neuroacoustic.synthesis.noise_generators import generate_noise

    # Generate slightly longer than needed for crossfade overlap
    extra = crossfade_samples if prev_tail is not None else 0
    total_samples = num_samples + extra
    duration = total_samples / sample_rate

    raw = generate_noise(color, duration, amplitude, sample_rate)

    if prev_tail is not None and extra > 0:
        # Crossfade between previous tail and current head
        fade_out = raised_cosine_fade(crossfade_samples)[::-1].reshape(-1, 1)
        fade_in = raised_cosine_fade(crossfade_samples).reshape(-1, 1)
        blended = prev_tail * fade_out + raw[:crossfade_samples] * fade_in
        remainder = raw[crossfade_samples : crossfade_samples + (num_samples - crossfade_samples)]
        chunk = np.vstack((blended, remainder))
        chunk = chunk[:num_samples]
    else:
        chunk = raw[:num_samples]

    # Save tail for next crossfade
    tail = raw[-crossfade_samples:]
    return chunk, tail


def generate_drone_chunk(
    phases: list[PhaseState],
    mod_phase: PhaseState,
    base_freq: float,
    harmonic_ratios: list[float],
    mod_rate: float,
    mod_depth: float,
    num_samples: int,
    amplitude: float = 0.3,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Generate one chunk of phase-continuous FM drone."""
    modulator = mod_depth * mod_phase.advance(mod_rate, num_samples, sample_rate)

    signal = np.zeros(num_samples)
    for i, ratio in enumerate(harmonic_ratios):
        freq = base_freq * ratio
        harmonic_amp = 1.0 / (i + 1)
        if i >= len(phases):
            phases.append(PhaseState())
        phases[i].advance(freq, num_samples, sample_rate)  # advance phase
        # Apply FM modulation
        signal += harmonic_amp * np.sin(
            np.cumsum(2 * np.pi * freq / sample_rate + modulator * 0.01)
        )

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * amplitude

    return np.column_stack((signal, signal))


def generate_isochronic_chunk(
    carrier_phase: PhaseState,
    pulse_rate: float,
    carrier_freq: float,
    chunk_offset_samples: int,
    num_samples: int,
    amplitude: float = 0.5,
    duty_cycle: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Generate one chunk of phase-continuous isochronic tone."""
    carrier = amplitude * carrier_phase.advance(carrier_freq, num_samples, sample_rate)

    # Pulse envelope must also be continuous across chunks
    t = (np.arange(num_samples) + chunk_offset_samples) / sample_rate
    phase = (t * pulse_rate) % 1.0
    envelope = (phase < duty_cycle).astype(np.float64)

    # Smooth the edges with a short raised-cosine (5ms)
    smooth_samples = max(1, int(0.005 * sample_rate))
    kernel = raised_cosine_fade(smooth_samples)
    kernel = kernel / kernel.sum()
    envelope = np.convolve(envelope, kernel, mode="same")

    mono = carrier * envelope
    return np.column_stack((mono, mono))
