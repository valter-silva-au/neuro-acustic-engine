#!/usr/bin/env python3
"""
Generate 8-hour relaxation & recovery audio files.

All selections are based on peer-reviewed psychoacoustic research:

1. 6 Hz Theta Binaural Beat + Brown Noise
   - Theta entrainment (4-8 Hz) induces meditative states and reduces
     anxiety. (Lavallee et al., 2011 - Alternative Therapies in Health)
   - Brown noise's 1/f^2 spectrum heavily emphasizes low frequencies,
     providing deep, womb-like masking ideal for parasympathetic activation.

2. 10 Hz Alpha Binaural Beat + Pink Noise
   - 10 Hz alpha entrainment reduces cortisol and promotes relaxed
     wakefulness. (Huang & Charyton, 2008 - Applied Psychophysiology)
   - Pink noise at moderate levels improves sleep quality and memory.
     (Zhou et al., 2012 - J Theoretical Biology)

3. 4 Hz Deep Theta Binaural Beat + Brown Noise
   - 4 Hz theta is associated with hypnagogic states, deep relaxation,
     and enhanced creativity. (Takahashi et al., 2005 - Neuroscience Res)
   - Low carrier frequency (110 Hz) keeps the tone warm and non-stimulating.

4. 6 Hz Theta Isochronic Tone + Brown Noise (no headphones)
   - Isochronic entrainment at theta frequency for speaker-based
     relaxation sessions. (Schwarz & Taylor, 2005)

5. 2 Hz Delta Binaural Beat + Brown Noise (Sleep Preparation)
   - Delta entrainment (0.5-4 Hz) facilitates transition to deep sleep.
     (Jirakittayakorn & Wongsawat, 2017 - Frontiers in Neuroscience)
   - Very low carrier (100 Hz) and heavy brown noise create a deeply
     soporific soundscape.

6. Deep Relaxation Soundscape (Layered)
   - 6 Hz theta binaural + brown noise + ultra-slow FM drone.
   - Drone at 55 Hz (sub-bass) with minimal modulation provides a
     subliminal grounding vibration.

7. Evening Wind-Down Soundscape (Layered)
   - 8 Hz alpha-theta boundary binaural for transitional relaxation +
     brown noise + warm drone.
   - Alpha-theta crossover is used in neurofeedback for deep relaxation
     and PTSD treatment. (Peniston & Kulkosky, 1991)

Each file is generated in 30-second streaming chunks to keep memory
usage under 5 MB regardless of file duration.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuroacoustic.synthesis.streaming import (
    DEFAULT_SAMPLE_RATE,
    PhaseState,
    StreamingWavWriter,
    generate_binaural_chunk,
    generate_drone_chunk,
    generate_isochronic_chunk,
    generate_noise_chunk,
    raised_cosine_fade,
)

SAMPLE_RATE = DEFAULT_SAMPLE_RATE
DURATION_HOURS = 8
DURATION_SECONDS = DURATION_HOURS * 3600
CHUNK_SECONDS = 30.0
FADE_SECONDS = 5.0  # Longer fade for relaxation — gentler onset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "relaxation")


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s"


def format_size(filepath: str) -> str:
    size = os.path.getsize(filepath)
    if size > 1_073_741_824:
        return f"{size / 1_073_741_824:.2f} GB"
    return f"{size / 1_048_576:.1f} MB"


def progress_bar(current: float, total: float, width: int = 40) -> str:
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r  [{bar}] {pct*100:5.1f}% | {format_duration(current)} / {format_duration(total)}"


def apply_file_fades(chunk, samples_written, n, total_samples, fade_samples):
    """Apply raised-cosine fade-in at file start and fade-out at file end."""
    if samples_written < fade_samples:
        overlap = min(n, fade_samples - samples_written)
        fade = raised_cosine_fade(fade_samples)
        chunk[:overlap] *= fade[samples_written : samples_written + overlap].reshape(-1, 1)

    if samples_written + n > total_samples - fade_samples:
        fade = raised_cosine_fade(fade_samples)[::-1]
        for i in range(n):
            fade_idx = (samples_written + i) - (total_samples - fade_samples)
            if 0 <= fade_idx < fade_samples:
                chunk[i] *= fade[fade_idx]
    return chunk


# ─────────────────────────────────────────────────────────────────────────────


def generate_binaural_with_noise(
    filename, carrier_freq, beat_freq, binaural_amp, noise_color, noise_amp, description,
):
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"\n  {description}")
    print(f"    Carrier: {carrier_freq} Hz | Beat: {beat_freq} Hz | Noise: {noise_color}")

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    total_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)

    left_phase, right_phase = PhaseState(), PhaseState()
    noise_tail = None
    start_time = time.time()

    with StreamingWavWriter(filepath, SAMPLE_RATE) as writer:
        samples_written = 0
        while samples_written < total_samples:
            n = min(chunk_samples, total_samples - samples_written)

            binaural = generate_binaural_chunk(
                left_phase, right_phase, carrier_freq, beat_freq, n, binaural_amp, SAMPLE_RATE,
            )
            noise, noise_tail = generate_noise_chunk(
                noise_color, n, noise_amp, noise_tail, sample_rate=SAMPLE_RATE,
            )

            chunk = np.clip(
                apply_file_fades(binaural + noise, samples_written, n, total_samples, fade_samples),
                -1.0, 1.0,
            )
            writer.write_chunk(chunk)
            samples_written += n
            print(progress_bar(samples_written / SAMPLE_RATE, DURATION_SECONDS), end="", flush=True)

    print(f"\n    -> {filepath} | {format_size(filepath)} | Generated in {time.time() - start_time:.1f}s")


def generate_isochronic_with_noise(
    filename, carrier_freq, pulse_rate, iso_amp, noise_color, noise_amp, description,
):
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"\n  {description}")
    print(f"    Carrier: {carrier_freq} Hz | Pulse: {pulse_rate} Hz | Noise: {noise_color}")

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    total_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)

    carrier_phase = PhaseState()
    noise_tail = None
    start_time = time.time()

    with StreamingWavWriter(filepath, SAMPLE_RATE) as writer:
        samples_written = 0
        while samples_written < total_samples:
            n = min(chunk_samples, total_samples - samples_written)

            iso = generate_isochronic_chunk(
                carrier_phase, pulse_rate, carrier_freq, samples_written, n, iso_amp,
                sample_rate=SAMPLE_RATE,
            )
            noise, noise_tail = generate_noise_chunk(
                noise_color, n, noise_amp, noise_tail, sample_rate=SAMPLE_RATE,
            )

            chunk = np.clip(
                apply_file_fades(iso + noise, samples_written, n, total_samples, fade_samples),
                -1.0, 1.0,
            )
            writer.write_chunk(chunk)
            samples_written += n
            print(progress_bar(samples_written / SAMPLE_RATE, DURATION_SECONDS), end="", flush=True)

    print(f"\n    -> {filepath} | {format_size(filepath)} | Generated in {time.time() - start_time:.1f}s")


def generate_layered_soundscape(
    filename, description,
    binaural_cfg=None, isochronic_cfg=None, noise_cfg=None, drone_cfg=None,
):
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"\n  {description}")

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    total_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)

    bin_left, bin_right = PhaseState(), PhaseState()
    iso_carrier = PhaseState()
    drone_phases, drone_mod_phase = [], PhaseState()
    noise_tail = None
    start_time = time.time()

    with StreamingWavWriter(filepath, SAMPLE_RATE) as writer:
        samples_written = 0
        while samples_written < total_samples:
            n = min(chunk_samples, total_samples - samples_written)
            chunk = np.zeros((n, 2))

            if binaural_cfg:
                b = generate_binaural_chunk(
                    bin_left, bin_right,
                    binaural_cfg["carrier"], binaural_cfg["beat"],
                    n, binaural_cfg.get("amp", 0.3), SAMPLE_RATE,
                )
                chunk += b * binaural_cfg.get("mix", 0.35)

            if isochronic_cfg:
                iso = generate_isochronic_chunk(
                    iso_carrier,
                    isochronic_cfg["pulse"], isochronic_cfg["carrier"],
                    samples_written, n, isochronic_cfg.get("amp", 0.2),
                    sample_rate=SAMPLE_RATE,
                )
                chunk += iso * isochronic_cfg.get("mix", 0.15)

            if noise_cfg:
                noise, noise_tail = generate_noise_chunk(
                    noise_cfg["color"], n, noise_cfg.get("amp", 0.3), noise_tail,
                    sample_rate=SAMPLE_RATE,
                )
                chunk += noise * noise_cfg.get("mix", 0.35)

            if drone_cfg:
                d = generate_drone_chunk(
                    drone_phases, drone_mod_phase,
                    drone_cfg["freq"], drone_cfg.get("harmonics", [1.0, 2.0]),
                    drone_cfg.get("mod_rate", 0.08), drone_cfg.get("mod_depth", 2.0),
                    n, drone_cfg.get("amp", 0.15), SAMPLE_RATE,
                )
                chunk += d * drone_cfg.get("mix", 0.2)

            chunk = np.clip(
                apply_file_fades(chunk, samples_written, n, total_samples, fade_samples),
                -1.0, 1.0,
            )
            writer.write_chunk(chunk)
            samples_written += n
            print(progress_bar(samples_written / SAMPLE_RATE, DURATION_SECONDS), end="", flush=True)

    print(f"\n    -> {filepath} | {format_size(filepath)} | Generated in {time.time() - start_time:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NEUROACOUSTIC ENGINE — 8-Hour Relaxation Playlist Generator")
    print(f"  Duration: {DURATION_HOURS} hours ({DURATION_SECONDS:,} seconds)")
    print(f"  Sample rate: {SAMPLE_RATE} Hz | Stereo 16-bit")
    print(f"  Chunk size: {CHUNK_SECONDS}s (~{CHUNK_SECONDS * SAMPLE_RATE * 4 / 1e6:.0f} MB RAM)")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 70)

    # ── Track 1: 6 Hz Theta + Brown Noise ──────────────────────────────
    generate_binaural_with_noise(
        "01_theta_6hz_brown_noise.wav",
        carrier_freq=150.0,
        beat_freq=6.0,
        binaural_amp=0.30,
        noise_color="brown",
        noise_amp=0.30,
        description="Track 1: 6 Hz Theta Binaural Beat + Brown Noise"
        "\n    Research: Theta entrainment induces meditative states and reduces anxiety"
        "\n    (Lavallee et al., 2011)",
    )

    # ── Track 2: 10 Hz Alpha + Pink Noise ──────────────────────────────
    generate_binaural_with_noise(
        "02_alpha_10hz_pink_noise.wav",
        carrier_freq=200.0,
        beat_freq=10.0,
        binaural_amp=0.30,
        noise_color="pink",
        noise_amp=0.28,
        description="Track 2: 10 Hz Alpha Binaural Beat + Pink Noise"
        "\n    Research: Alpha entrainment reduces cortisol and promotes relaxed wakefulness"
        "\n    (Huang & Charyton, 2008)",
    )

    # ── Track 3: 4 Hz Deep Theta + Brown Noise ────────────────────────
    generate_binaural_with_noise(
        "03_deep_theta_4hz_brown_noise.wav",
        carrier_freq=110.0,
        beat_freq=4.0,
        binaural_amp=0.25,
        noise_color="brown",
        noise_amp=0.35,
        description="Track 3: 4 Hz Deep Theta Binaural Beat + Brown Noise"
        "\n    Research: Deep theta associated with hypnagogic states and enhanced creativity"
        "\n    (Takahashi et al., 2005)",
    )

    # ── Track 4: 6 Hz Theta Isochronic + Brown Noise ──────────────────
    generate_isochronic_with_noise(
        "04_theta_6hz_isochronic_brown_noise.wav",
        carrier_freq=180.0,
        pulse_rate=6.0,
        iso_amp=0.25,
        noise_color="brown",
        noise_amp=0.30,
        description="Track 4: 6 Hz Theta Isochronic Tone + Brown Noise (no headphones needed)"
        "\n    Research: Isochronic pulses produce strong cortical entrainment"
        "\n    (Schwarz & Taylor, 2005)",
    )

    # ── Track 5: 2 Hz Delta + Brown Noise (Sleep) ─────────────────────
    generate_binaural_with_noise(
        "05_delta_2hz_brown_noise.wav",
        carrier_freq=100.0,
        beat_freq=2.0,
        binaural_amp=0.20,
        noise_color="brown",
        noise_amp=0.40,
        description="Track 5: 2 Hz Delta Binaural Beat + Brown Noise (Sleep Preparation)"
        "\n    Research: Delta entrainment facilitates transition to deep sleep"
        "\n    (Jirakittayakorn & Wongsawat, 2017)",
    )

    # ── Track 6: Deep Relaxation Soundscape (Layered) ──────────────────
    generate_layered_soundscape(
        "06_deep_relaxation_soundscape.wav",
        description="Track 6: Deep Relaxation Soundscape (6 Hz theta binaural + brown noise + sub-bass drone)"
        "\n    Multi-layered immersive environment for deep meditative rest",
        binaural_cfg={"carrier": 150.0, "beat": 6.0, "amp": 0.30, "mix": 0.35},
        noise_cfg={"color": "brown", "amp": 0.30, "mix": 0.40},
        drone_cfg={
            "freq": 55.0, "harmonics": [1.0, 2.0],
            "mod_rate": 0.05, "mod_depth": 1.5, "amp": 0.15, "mix": 0.25,
        },
    )

    # ── Track 7: Evening Wind-Down Soundscape (Layered) ────────────────
    generate_layered_soundscape(
        "07_evening_wind_down_soundscape.wav",
        description="Track 7: Evening Wind-Down Soundscape (8 Hz alpha-theta binaural + brown noise + warm drone)"
        "\n    Research: Alpha-theta crossover used in neurofeedback for deep relaxation"
        "\n    (Peniston & Kulkosky, 1991)",
        binaural_cfg={"carrier": 170.0, "beat": 8.0, "amp": 0.25, "mix": 0.30},
        noise_cfg={"color": "brown", "amp": 0.35, "mix": 0.40},
        drone_cfg={
            "freq": 65.0, "harmonics": [1.0, 2.0, 3.0],
            "mod_rate": 0.04, "mod_depth": 1.0, "amp": 0.12, "mix": 0.20,
        },
    )

    # ── Summary ────────────────────────────────────────────────────────
    wav_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav"))
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in wav_files)

    print(f"\n{'=' * 70}")
    print(f"  RELAXATION PLAYLIST COMPLETE")
    print(f"  {len(wav_files)} tracks x {DURATION_HOURS} hours each")
    print(f"  Total size: {total_size / 1_073_741_824:.2f} GB")
    print(f"  Location: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"\n  Tracks:")
    for f in wav_files:
        size = format_size(os.path.join(OUTPUT_DIR, f))
        print(f"    {f}  ({size})")
    print(f"\n  TIP: Use headphones for tracks 01, 02, 03, 05 (binaural effect).")
    print(f"  Track 04 works through speakers (isochronic).")
    print(f"  Tracks 06-07 are layered soundscapes for deep immersion.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
