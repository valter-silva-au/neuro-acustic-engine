#!/usr/bin/env python3
"""
Generate all sound types from the NeuroAcoustic Engine as WAV files.

Produces:
  1. Binaural beats (theta, alpha, beta, gamma)
  2. Isochronic tones (theta, alpha, beta, gamma)
  3. Colored noise (white, pink, brown)
  4. FM ambient drones (relaxation, focus, alertness)
  5. Full cognitive soundscapes (layered mixes for each intent)
  6. Semantic-mapped event sounds (calendar events → audio)
"""

import json
import os
import struct
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuroacoustic.core.config import BRAINWAVE_BANDS, INTENT_PROFILES
from neuroacoustic.synthesis.binaural import calculate_binaural_frequencies
from neuroacoustic.synthesis.isochronic import generate_isochronic_tone
from neuroacoustic.synthesis.noise_generators import generate_noise
from neuroacoustic.translation.semantic_mapper import SemanticMapper
from neuroacoustic.translation.timbre_space import (
    compute_harmonic_amplitudes,
    compute_inharmonicity,
)

SAMPLE_RATE = 44100
DURATION = 10.0  # seconds per clip
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def save_wav(filename: str, signal: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Save a stereo float signal (-1 to 1) as a 16-bit WAV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    signal = np.clip(signal, -1.0, 1.0)
    int_signal = (signal * 32767).astype(np.int16)

    with wave.open(filepath, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_signal.tobytes())

    size_kb = os.path.getsize(filepath) / 1024
    print(f"  -> {filepath} ({size_kb:.0f} KB)")


def apply_fade(signal: np.ndarray, fade_seconds: float = 0.5) -> np.ndarray:
    """Apply linear fade-in and fade-out to prevent clicks."""
    fade_samples = int(fade_seconds * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_samples).reshape(-1, 1)
    fade_out = np.linspace(1, 0, fade_samples).reshape(-1, 1)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    return signal


def generate_binaural(
    carrier_freq: float, beat_freq: float, duration: float = DURATION, amplitude: float = 0.5
) -> np.ndarray:
    """Generate a stereo binaural beat using numpy."""
    num_samples = int(duration * SAMPLE_RATE)
    t = np.arange(num_samples) / SAMPLE_RATE
    left_hz, right_hz = calculate_binaural_frequencies(carrier_freq, beat_freq)
    left = amplitude * np.sin(2 * np.pi * left_hz * t)
    right = amplitude * np.sin(2 * np.pi * right_hz * t)
    return apply_fade(np.column_stack((left, right)))


def generate_drone(
    base_freq: float,
    harmonic_ratios: list[float],
    mod_rate: float,
    mod_depth: float = 10.0,
    duration: float = DURATION,
    amplitude: float = 0.4,
) -> np.ndarray:
    """Generate an FM ambient drone with harmonics."""
    num_samples = int(duration * SAMPLE_RATE)
    t = np.arange(num_samples) / SAMPLE_RATE
    modulator = mod_depth * np.sin(2 * np.pi * mod_rate * t)

    signal = np.zeros(num_samples)
    for i, ratio in enumerate(harmonic_ratios):
        freq = base_freq * ratio
        harmonic_amp = 1.0 / (i + 1)
        signal += harmonic_amp * np.sin(2 * np.pi * freq * t + modulator)

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * amplitude

    stereo = np.column_stack((signal, signal))
    return apply_fade(stereo)


def mix_signals(*signals: np.ndarray, levels: list[float] | None = None) -> np.ndarray:
    """Mix multiple stereo signals with optional level control."""
    if levels is None:
        levels = [1.0 / len(signals)] * len(signals)

    # Pad shorter signals to match longest
    max_len = max(s.shape[0] for s in signals)
    mixed = np.zeros((max_len, 2))
    for sig, level in zip(signals, levels):
        mixed[: sig.shape[0]] += sig * level

    # Normalize to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def generate_binaural_beats():
    """Generate binaural beats for each brainwave band."""
    print("\n=== BINAURAL BEATS ===")
    print("(Use headphones for the binaural effect)\n")

    configs = [
        ("delta", 2.0, 100.0, "Deep sleep / healing"),
        ("theta", 6.0, 150.0, "Meditation / creativity"),
        ("alpha", 10.0, 200.0, "Relaxation / calm focus"),
        ("beta", 20.0, 250.0, "Concentration / alertness"),
        ("gamma", 40.0, 400.0, "Peak cognition / insight"),
    ]

    for band, beat_freq, carrier, description in configs:
        print(f"  {band.upper()} ({beat_freq} Hz beat on {carrier} Hz carrier) - {description}")
        signal = generate_binaural(carrier, beat_freq)
        save_wav(f"binaural_{band}_{beat_freq}hz.wav", signal)


def generate_isochronic_tones():
    """Generate isochronic tones for each brainwave band."""
    print("\n=== ISOCHRONIC TONES ===")
    print("(Work without headphones - rhythmic pulsing)\n")

    configs = [
        ("theta", 6.0, 200.0, "Meditation pulse"),
        ("alpha", 10.0, 300.0, "Relaxation pulse"),
        ("beta", 18.0, 400.0, "Focus pulse"),
        ("gamma", 40.0, 500.0, "Cognition pulse"),
    ]

    for band, pulse_rate, carrier, description in configs:
        print(f"  {band.upper()} ({pulse_rate} Hz pulse on {carrier} Hz tone) - {description}")
        signal = generate_isochronic_tone(
            carrier, pulse_rate, DURATION, amplitude=0.5, sample_rate=SAMPLE_RATE
        )
        signal = apply_fade(signal)
        save_wav(f"isochronic_{band}_{pulse_rate}hz.wav", signal)


def generate_colored_noise():
    """Generate all three noise colors."""
    print("\n=== COLORED NOISE ===\n")

    for color in ["white", "pink", "brown"]:
        descriptions = {
            "white": "Flat spectrum - masking / high energy",
            "pink": "1/f spectrum - balanced / natural",
            "brown": "1/f^2 spectrum - deep / warm / relaxing",
        }
        print(f"  {color.upper()} noise - {descriptions[color]}")
        signal = generate_noise(color, DURATION, amplitude=0.4, sample_rate=SAMPLE_RATE)
        signal = apply_fade(signal)
        save_wav(f"noise_{color}.wav", signal)


def generate_ambient_drones():
    """Generate FM ambient drones for different cognitive states."""
    print("\n=== AMBIENT DRONES ===\n")

    configs = [
        (
            "relaxation_drone",
            80.0,
            [1.0, 2.0, 3.0, 5.0],
            0.15,
            5.0,
            "Deep warm drone for theta relaxation",
        ),
        (
            "focus_drone",
            200.0,
            [1.0, 2.0, 3.0],
            0.5,
            8.0,
            "Mid-range drone for focused work",
        ),
        (
            "alertness_drone",
            300.0,
            [1.0, 1.5, 2.0, 3.0, 4.0],
            1.2,
            12.0,
            "Bright drone with complex harmonics",
        ),
        (
            "evening_drone",
            60.0,
            [1.0, 2.0, 4.0],
            0.08,
            3.0,
            "Very slow evolving bass drone for wind-down",
        ),
    ]

    for name, base_freq, harmonics, mod_rate, mod_depth, description in configs:
        print(f"  {name} ({base_freq} Hz base, {len(harmonics)} harmonics) - {description}")
        signal = generate_drone(base_freq, harmonics, mod_rate, mod_depth)
        save_wav(f"drone_{name}.wav", signal)


def generate_cognitive_soundscapes():
    """Generate full layered soundscapes for each cognitive intent."""
    print("\n=== COGNITIVE SOUNDSCAPES ===")
    print("(Layered: binaural + noise + drone)\n")

    soundscapes = [
        {
            "name": "deep_focus",
            "description": "Gamma entrainment for peak concentration",
            "binaural": (432.0, 40.0, 0.4),
            "noise": ("pink", 0.15),
            "drone": (216.0, [1.0, 2.0, 3.0], 0.5, 8.0, 0.25),
            "mix_levels": [0.45, 0.25, 0.30],
        },
        {
            "name": "deep_relaxation",
            "description": "Theta waves for meditation and creativity",
            "binaural": (110.0, 5.0, 0.5),
            "noise": ("brown", 0.35),
            "drone": (55.0, [1.0, 2.0, 4.0], 0.1, 3.0, 0.3),
            "mix_levels": [0.40, 0.35, 0.25],
        },
        {
            "name": "morning_energy",
            "description": "Beta state for alert, productive mornings",
            "binaural": (250.0, 18.0, 0.45),
            "noise": ("pink", 0.2),
            "drone": (125.0, [1.0, 2.0, 3.0, 5.0], 0.8, 6.0, 0.2),
            "mix_levels": [0.45, 0.25, 0.30],
        },
        {
            "name": "evening_wind_down",
            "description": "Alpha transition for decompression",
            "binaural": (180.0, 9.0, 0.4),
            "noise": ("brown", 0.4),
            "drone": (60.0, [1.0, 2.0], 0.05, 2.0, 0.25),
            "mix_levels": [0.35, 0.40, 0.25],
        },
        {
            "name": "learning_session",
            "description": "40Hz gamma burst for memory formation",
            "binaural": (528.0, 40.0, 0.4),
            "noise": ("pink", 0.12),
            "drone": (264.0, [1.0, 2.0, 3.0], 0.6, 10.0, 0.2),
            "mix_levels": [0.50, 0.20, 0.30],
        },
    ]

    for scape in soundscapes:
        print(f"  {scape['name']} - {scape['description']}")

        # Layer 1: Binaural beat
        carrier, beat, amp = scape["binaural"]
        binaural = generate_binaural(carrier, beat, amplitude=amp)

        # Layer 2: Colored noise
        color, noise_amp = scape["noise"]
        noise_sig = generate_noise(color, DURATION, amplitude=noise_amp, sample_rate=SAMPLE_RATE)
        noise_sig = apply_fade(noise_sig)

        # Layer 3: Ambient drone
        d_freq, d_harmonics, d_mod, d_depth, d_amp = scape["drone"]
        drone = generate_drone(d_freq, d_harmonics, d_mod, d_depth, amplitude=d_amp)

        # Mix all layers
        mixed = mix_signals(binaural, noise_sig, drone, levels=scape["mix_levels"])
        mixed = apply_fade(mixed)
        save_wav(f"soundscape_{scape['name']}.wav", mixed)


def generate_semantic_mapped_events():
    """Generate sounds from real-world event scenarios through the SemanticMapper."""
    print("\n=== SEMANTIC-MAPPED EVENTS ===")
    print("(Real-world events translated to audio via SemanticMapper)\n")

    mapper = SemanticMapper()

    events = [
        {
            "intent": "focus",
            "urgency": "high",
            "sentiment": "neutral",
            "data_source": "calendar",
            "content_summary": "URGENT: Production deployment review",
        },
        {
            "intent": "relaxation",
            "urgency": "low",
            "sentiment": "positive",
            "data_source": "calendar",
            "content_summary": "Evening yoga session",
        },
        {
            "intent": "alertness",
            "urgency": "medium",
            "sentiment": "positive",
            "data_source": "rss_feed",
            "content_summary": "Major breakthrough in quantum computing announced",
        },
        {
            "intent": "learning",
            "urgency": "medium",
            "sentiment": "neutral",
            "data_source": "calendar",
            "content_summary": "Study session: Advanced neural network architectures",
        },
        {
            "intent": "alertness",
            "urgency": "critical",
            "sentiment": "negative",
            "data_source": "rss_feed",
            "content_summary": "Critical security vulnerability discovered in major framework",
        },
        {
            "intent": "relaxation",
            "urgency": "low",
            "sentiment": "neutral",
            "data_source": "calendar",
            "content_summary": "Lunch break - time to decompress",
        },
    ]

    for i, event in enumerate(events, 1):
        payload = json.dumps(event)
        params = mapper.translate_payload(payload)

        print(f"  Event {i}: \"{event['content_summary']}\"")
        print(f"    Intent: {event['intent']} | Urgency: {event['urgency']} | Sentiment: {event['sentiment']}")
        print(f"    -> Carrier: {params['carrier_freq']} Hz | Beat: {params['beat_freq']} Hz | Band: {params['target_band']}")
        print(f"    -> Amplitude: {params['target_amplitude']} | Noise: {params['noise_color']} | Brightness: {params['timbre_brightness']}")

        # Build layered audio from mapped parameters
        binaural = generate_binaural(
            params["carrier_freq"], params["beat_freq"], amplitude=params["target_amplitude"]
        )

        noise_sig = generate_noise(
            params["noise_color"], DURATION, amplitude=params["target_amplitude"] * 0.5
        )
        noise_sig = apply_fade(noise_sig)

        # Drone based on carrier / 2 with sentiment-derived harmonics
        inharmonicity = compute_inharmonicity(event.get("sentiment", "neutral"))
        harmonic_list = compute_harmonic_amplitudes(
            params.get("harmonic_ratios", [1.0, 2.0, 3.0]),
            inharmonicity,
            num_harmonics=4,
        )
        drone_harmonics = [h[0] for h in harmonic_list]
        drone = generate_drone(
            params["carrier_freq"] / 2,
            drone_harmonics,
            mod_rate=0.3 + params["timbre_brightness"],
            mod_depth=5.0 + params["timbre_brightness"] * 10,
            amplitude=params["target_amplitude"] * 0.6,
        )

        mixed = mix_signals(binaural, noise_sig, drone, levels=[0.5, 0.25, 0.25])
        mixed = apply_fade(mixed)

        safe_name = event["content_summary"][:40].replace(" ", "_").replace(":", "").replace("/", "").lower()
        save_wav(f"event_{i}_{safe_name}.wav", mixed)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  NEUROACOUSTIC ENGINE - Complete Sound Generation Demo")
    print("=" * 60)

    generate_binaural_beats()
    generate_isochronic_tones()
    generate_colored_noise()
    generate_ambient_drones()
    generate_cognitive_soundscapes()
    generate_semantic_mapped_events()

    # Count output files
    wav_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")]
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in wav_files) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"  Generated {len(wav_files)} WAV files ({total_size:.1f} MB total)")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
