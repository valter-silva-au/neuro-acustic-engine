#!/usr/bin/env python3
"""
Generate 8-hour focus & concentration audio files.

All selections are based on peer-reviewed psychoacoustic research:

1. 40 Hz Gamma Binaural Beat + Pink Noise
   - 40 Hz gamma entrainment enhances memory encoding and attentional binding.
     (Jirakittayakorn & Wongsawat, 2017 - Int J Psychophysiology)
   - Pink noise improves sleep stability and memory consolidation.
     (Zhou et al., 2012 - J Theoretical Biology)
   - Carrier at 250 Hz: optimal for sustained listening without fatigue.

2. 16 Hz Beta Binaural Beat + Brown Noise
   - 16 Hz (SMR / low-beta) improves sustained attention and reduces
     mind-wandering. (Egner & Gruzelier, 2004 - Clinical Neurophysiology)
   - Brown noise provides deep masking of environmental distractions.

3. 40 Hz Isochronic Tone + Pink Noise
   - Isochronic tones produce stronger cortical entrainment than binaural
     beats due to sharp amplitude onsets. (Schwarz & Taylor, 2005)
   - No headphones required — works through speakers.

4. 14 Hz Beta Binaural Beat + Pink Noise (Sustained Attention)
   - 14 Hz (low beta / SMR boundary) is associated with calm focus and
     reduced hyperactivity. (Vernon et al., 2003 - Applied Psychophysiology)

5. 18 Hz Beta Binaural Beat + Brown Noise (Active Problem-Solving)
   - Mid-beta range supports analytical thinking and active concentration.
     (Lane et al., 1998 - Physiology & Behavior)

6. Deep Focus Soundscape (Layered)
   - Combines 40 Hz gamma binaural + pink noise + low FM drone.
   - Multi-layered approach for deep immersive work sessions.
   - Drone at half-carrier frequency provides subliminal harmonic anchor.

7. Flow State Soundscape (Layered)
   - 10 Hz alpha binaural for relaxed awareness + 40 Hz gamma isochronic
     for cognitive binding, layered over brown noise.
   - Alpha-gamma coupling is observed during flow states.
     (Katahira et al., 2018 - Frontiers in Psychology)

Each file is generated in 30-second streaming chunks to keep memory
usage under 15 MB regardless of file duration.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuroacoustic.synthesis.streaming import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_SAMPLE_RATE,
    PhaseState,
    StreamingWavWriter,
    generate_binaural_chunk,
    generate_drone_chunk,
    generate_isochronic_chunk,
    generate_noise_chunk,
    raised_cosine_fade,
)

SAMPLE_RATE = DEFAULT_SAMPLE_RATE  # 22050 Hz — keeps 8h files under WAV's 4GB limit

DURATION_HOURS = 8
DURATION_SECONDS = DURATION_HOURS * 3600
CHUNK_SECONDS = 30.0
FADE_SECONDS = 3.0  # Raised-cosine fade at start/end of entire file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "focus")


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


def progress_bar(current: float, total: float, width: int = 40, label: str = "") -> str:
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = format_duration(current)
    return f"\r  [{bar}] {pct*100:5.1f}% | {elapsed} / {format_duration(total)} {label}"


# ─────────────────────────────────────────────────────────────────────────────
# TRACK GENERATORS
# ─────────────────────────────────────────────────────────────────────────────


def generate_binaural_with_noise(
    filename: str,
    carrier_freq: float,
    beat_freq: float,
    binaural_amp: float,
    noise_color: str,
    noise_amp: float,
    description: str,
):
    """Generate a binaural beat layered with colored noise."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"\n  {description}")
    print(f"    Carrier: {carrier_freq} Hz | Beat: {beat_freq} Hz | Noise: {noise_color}")

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    total_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)

    left_phase = PhaseState()
    right_phase = PhaseState()
    noise_tail = None

    start_time = time.time()

    with StreamingWavWriter(filepath) as writer:
        samples_written = 0

        while samples_written < total_samples:
            remaining = total_samples - samples_written
            n = min(chunk_samples, remaining)

            # Phase-continuous binaural beat
            binaural = generate_binaural_chunk(
                left_phase, right_phase, carrier_freq, beat_freq, n, binaural_amp
            )

            # Crossfaded noise
            noise, noise_tail = generate_noise_chunk(
                noise_color, n, noise_amp, noise_tail
            )

            # Mix
            chunk = binaural + noise

            # Apply file-level fade-in / fade-out
            if samples_written < fade_samples:
                overlap = min(n, fade_samples - samples_written)
                fade = raised_cosine_fade(fade_samples)
                start_idx = samples_written
                chunk[:overlap] *= fade[start_idx : start_idx + overlap].reshape(-1, 1)

            if samples_written + n > total_samples - fade_samples:
                fade_out_start = max(0, (total_samples - fade_samples) - samples_written)
                fade = raised_cosine_fade(fade_samples)[::-1]
                for i in range(fade_out_start, n):
                    fade_idx = (samples_written + i) - (total_samples - fade_samples)
                    if 0 <= fade_idx < fade_samples:
                        chunk[i] *= fade[fade_idx]

            # Clip and write
            chunk = np.clip(chunk, -1.0, 1.0)
            writer.write_chunk(chunk)
            samples_written += n

            # Progress
            elapsed = time.time() - start_time
            print(progress_bar(samples_written / SAMPLE_RATE, DURATION_SECONDS), end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n    -> {filepath} | {format_size(filepath)} | Generated in {elapsed:.1f}s")


def generate_isochronic_with_noise(
    filename: str,
    carrier_freq: float,
    pulse_rate: float,
    iso_amp: float,
    noise_color: str,
    noise_amp: float,
    description: str,
):
    """Generate an isochronic tone layered with colored noise."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"\n  {description}")
    print(f"    Carrier: {carrier_freq} Hz | Pulse: {pulse_rate} Hz | Noise: {noise_color}")

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    total_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)

    carrier_phase = PhaseState()
    noise_tail = None

    start_time = time.time()

    with StreamingWavWriter(filepath) as writer:
        samples_written = 0

        while samples_written < total_samples:
            remaining = total_samples - samples_written
            n = min(chunk_samples, remaining)

            # Phase-continuous isochronic tone
            iso = generate_isochronic_chunk(
                carrier_phase, pulse_rate, carrier_freq, samples_written, n, iso_amp
            )

            # Crossfaded noise
            noise, noise_tail = generate_noise_chunk(
                noise_color, n, noise_amp, noise_tail
            )

            chunk = iso + noise

            # File-level fades
            if samples_written < fade_samples:
                overlap = min(n, fade_samples - samples_written)
                fade = raised_cosine_fade(fade_samples)
                start_idx = samples_written
                chunk[:overlap] *= fade[start_idx : start_idx + overlap].reshape(-1, 1)

            if samples_written + n > total_samples - fade_samples:
                fade_out_start = max(0, (total_samples - fade_samples) - samples_written)
                fade = raised_cosine_fade(fade_samples)[::-1]
                for i in range(fade_out_start, n):
                    fade_idx = (samples_written + i) - (total_samples - fade_samples)
                    if 0 <= fade_idx < fade_samples:
                        chunk[i] *= fade[fade_idx]

            chunk = np.clip(chunk, -1.0, 1.0)
            writer.write_chunk(chunk)
            samples_written += n

            print(progress_bar(samples_written / SAMPLE_RATE, DURATION_SECONDS), end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n    -> {filepath} | {format_size(filepath)} | Generated in {elapsed:.1f}s")


def generate_layered_soundscape(
    filename: str,
    description: str,
    binaural_cfg: dict | None = None,
    isochronic_cfg: dict | None = None,
    noise_cfg: dict | None = None,
    drone_cfg: dict | None = None,
):
    """Generate a multi-layered focus soundscape."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"\n  {description}")

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    total_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)

    # Initialize phase states
    bin_left = PhaseState()
    bin_right = PhaseState()
    iso_carrier = PhaseState()
    drone_phases = []
    drone_mod_phase = PhaseState()
    noise_tail = None

    start_time = time.time()

    with StreamingWavWriter(filepath) as writer:
        samples_written = 0

        while samples_written < total_samples:
            remaining = total_samples - samples_written
            n = min(chunk_samples, remaining)

            chunk = np.zeros((n, 2))

            # Binaural layer
            if binaural_cfg:
                b = generate_binaural_chunk(
                    bin_left, bin_right,
                    binaural_cfg["carrier"], binaural_cfg["beat"],
                    n, binaural_cfg.get("amp", 0.35),
                )
                chunk += b * binaural_cfg.get("mix", 0.4)

            # Isochronic layer
            if isochronic_cfg:
                iso = generate_isochronic_chunk(
                    iso_carrier,
                    isochronic_cfg["pulse"],
                    isochronic_cfg["carrier"],
                    samples_written, n,
                    isochronic_cfg.get("amp", 0.3),
                )
                chunk += iso * isochronic_cfg.get("mix", 0.2)

            # Noise layer
            if noise_cfg:
                noise, noise_tail = generate_noise_chunk(
                    noise_cfg["color"], n, noise_cfg.get("amp", 0.25), noise_tail
                )
                chunk += noise * noise_cfg.get("mix", 0.3)

            # Drone layer
            if drone_cfg:
                d = generate_drone_chunk(
                    drone_phases, drone_mod_phase,
                    drone_cfg["freq"], drone_cfg.get("harmonics", [1.0, 2.0]),
                    drone_cfg.get("mod_rate", 0.15),
                    drone_cfg.get("mod_depth", 3.0),
                    n, drone_cfg.get("amp", 0.2),
                )
                chunk += d * drone_cfg.get("mix", 0.2)

            # File-level fades
            if samples_written < fade_samples:
                overlap = min(n, fade_samples - samples_written)
                fade = raised_cosine_fade(fade_samples)
                start_idx = samples_written
                chunk[:overlap] *= fade[start_idx : start_idx + overlap].reshape(-1, 1)

            if samples_written + n > total_samples - fade_samples:
                fade_out_start = max(0, (total_samples - fade_samples) - samples_written)
                fade = raised_cosine_fade(fade_samples)[::-1]
                for i in range(fade_out_start, n):
                    fade_idx = (samples_written + i) - (total_samples - fade_samples)
                    if 0 <= fade_idx < fade_samples:
                        chunk[i] *= fade[fade_idx]

            chunk = np.clip(chunk, -1.0, 1.0)
            writer.write_chunk(chunk)
            samples_written += n

            print(progress_bar(samples_written / SAMPLE_RATE, DURATION_SECONDS), end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n    -> {filepath} | {format_size(filepath)} | Generated in {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  NEUROACOUSTIC ENGINE — 8-Hour Focus Playlist Generator")
    print(f"  Duration: {DURATION_HOURS} hours ({DURATION_SECONDS:,} seconds)")
    print(f"  Sample rate: {SAMPLE_RATE} Hz | Stereo 16-bit")
    print(f"  Chunk size: {CHUNK_SECONDS}s (~{CHUNK_SECONDS * SAMPLE_RATE * 4 / 1e6:.0f} MB RAM)")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 70)

    # ── Track 1: 40 Hz Gamma + Pink Noise ──────────────────────────────
    generate_binaural_with_noise(
        "01_gamma_40hz_pink_noise.wav",
        carrier_freq=250.0,
        beat_freq=40.0,
        binaural_amp=0.35,
        noise_color="pink",
        noise_amp=0.20,
        description="Track 1: 40 Hz Gamma Binaural Beat + Pink Noise"
        "\n    Research: Gamma entrainment enhances attentional binding and memory"
        "\n    (Jirakittayakorn & Wongsawat, 2017)",
    )

    # ── Track 2: 16 Hz Beta (SMR) + Brown Noise ───────────────────────
    generate_binaural_with_noise(
        "02_beta_16hz_brown_noise.wav",
        carrier_freq=200.0,
        beat_freq=16.0,
        binaural_amp=0.35,
        noise_color="brown",
        noise_amp=0.25,
        description="Track 2: 16 Hz Beta (SMR) Binaural Beat + Brown Noise"
        "\n    Research: SMR training improves sustained attention"
        "\n    (Egner & Gruzelier, 2004)",
    )

    # ── Track 3: 40 Hz Isochronic + Pink Noise ────────────────────────
    generate_isochronic_with_noise(
        "03_gamma_40hz_isochronic_pink_noise.wav",
        carrier_freq=300.0,
        pulse_rate=40.0,
        iso_amp=0.30,
        noise_color="pink",
        noise_amp=0.22,
        description="Track 3: 40 Hz Gamma Isochronic Tone + Pink Noise (no headphones needed)"
        "\n    Research: Isochronic pulses produce stronger cortical response than binaural"
        "\n    (Schwarz & Taylor, 2005)",
    )

    # ── Track 4: 14 Hz Low Beta + Pink Noise ──────────────────────────
    generate_binaural_with_noise(
        "04_beta_14hz_pink_noise.wav",
        carrier_freq=220.0,
        beat_freq=14.0,
        binaural_amp=0.30,
        noise_color="pink",
        noise_amp=0.25,
        description="Track 4: 14 Hz Low Beta Binaural Beat + Pink Noise"
        "\n    Research: SMR boundary frequency reduces hyperactivity, promotes calm focus"
        "\n    (Vernon et al., 2003)",
    )

    # ── Track 5: 18 Hz Mid-Beta + Brown Noise ─────────────────────────
    generate_binaural_with_noise(
        "05_beta_18hz_brown_noise.wav",
        carrier_freq=230.0,
        beat_freq=18.0,
        binaural_amp=0.35,
        noise_color="brown",
        noise_amp=0.22,
        description="Track 5: 18 Hz Mid-Beta Binaural Beat + Brown Noise"
        "\n    Research: Mid-beta range supports analytical thinking"
        "\n    (Lane et al., 1998)",
    )

    # ── Track 6: Deep Focus Soundscape (Layered) ──────────────────────
    generate_layered_soundscape(
        "06_deep_focus_soundscape.wav",
        description="Track 6: Deep Focus Soundscape (40 Hz binaural + pink noise + drone)"
        "\n    Multi-layered immersive environment for extended deep work sessions",
        binaural_cfg={"carrier": 250.0, "beat": 40.0, "amp": 0.35, "mix": 0.40},
        noise_cfg={"color": "pink", "amp": 0.22, "mix": 0.35},
        drone_cfg={
            "freq": 125.0, "harmonics": [1.0, 2.0, 3.0],
            "mod_rate": 0.12, "mod_depth": 4.0, "amp": 0.18, "mix": 0.25,
        },
    )

    # ── Track 7: Flow State Soundscape (Alpha-Gamma Coupling) ─────────
    generate_layered_soundscape(
        "07_flow_state_soundscape.wav",
        description="Track 7: Flow State Soundscape (10 Hz alpha binaural + 40 Hz gamma isochronic + brown noise)"
        "\n    Research: Alpha-gamma coupling is observed during flow states"
        "\n    (Katahira et al., 2018)",
        binaural_cfg={"carrier": 200.0, "beat": 10.0, "amp": 0.30, "mix": 0.35},
        isochronic_cfg={"carrier": 400.0, "pulse": 40.0, "amp": 0.15, "mix": 0.15},
        noise_cfg={"color": "brown", "amp": 0.25, "mix": 0.35},
        drone_cfg={
            "freq": 100.0, "harmonics": [1.0, 2.0],
            "mod_rate": 0.08, "mod_depth": 2.0, "amp": 0.15, "mix": 0.15,
        },
    )

    # ── Summary ────────────────────────────────────────────────────────
    wav_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav"))
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in wav_files)

    print(f"\n{'=' * 70}")
    print(f"  FOCUS PLAYLIST COMPLETE")
    print(f"  {len(wav_files)} tracks x {DURATION_HOURS} hours each")
    print(f"  Total size: {total_size / 1_073_741_824:.2f} GB")
    print(f"  Location: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"\n  Tracks:")
    for f in wav_files:
        size = format_size(os.path.join(OUTPUT_DIR, f))
        print(f"    {f}  ({size})")
    print(f"\n  TIP: Use headphones for tracks 01, 02, 04, 05 (binaural effect).")
    print(f"  Track 03 works through speakers (isochronic).")
    print(f"  Tracks 06-07 are layered soundscapes for deep immersion.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
