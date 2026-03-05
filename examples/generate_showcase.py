#!/usr/bin/env python3
"""
Generate the complete NeuroAcoustic Engine showcase sample library.

2-minute samples across 6 categories for the GitHub Pages site.
~5 MB per file at 22050 Hz stereo 16-bit.
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
DURATION = 120.0  # 2 minutes
FADE_SECONDS = 3.0
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "samples")


def format_size(filepath: str) -> str:
    size = os.path.getsize(filepath)
    if size > 1_048_576:
        return f"{size / 1_048_576:.1f} MB"
    return f"{size / 1024:.0f} KB"


def generate_single(filepath, duration, layer_fns):
    """
    Generate a single sample by mixing layers produced by layer functions.

    Each layer_fn(n, samples_written) -> np.ndarray of shape (n, 2).
    """
    total_samples = int(duration * SAMPLE_RATE)
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)

    with StreamingWavWriter(filepath, SAMPLE_RATE) as writer:
        chunk_samples = total_samples  # Single chunk for 2-min files (~20 MB RAM, fine)
        n = total_samples
        chunk = np.zeros((n, 2))

        for fn in layer_fns:
            chunk += fn(n, 0)

        # Fade in/out
        fade_in = raised_cosine_fade(fade_samples).reshape(-1, 1)
        fade_out = raised_cosine_fade(fade_samples)[::-1].reshape(-1, 1)
        chunk[:fade_samples] *= fade_in
        chunk[-fade_samples:] *= fade_out

        chunk = np.clip(chunk, -1.0, 1.0)
        writer.write_chunk(chunk)


def make_binaural_layer(carrier, beat, amp=0.4):
    lp, rp = PhaseState(), PhaseState()
    def fn(n, offset):
        return generate_binaural_chunk(lp, rp, carrier, beat, n, amp, SAMPLE_RATE)
    return fn


def make_noise_layer(color, amp=0.25):
    def fn(n, offset):
        noise, _ = generate_noise_chunk(color, n, amp, None, sample_rate=SAMPLE_RATE)
        return noise
    return fn


def make_isochronic_layer(carrier, pulse, amp=0.35):
    cp = PhaseState()
    def fn(n, offset):
        return generate_isochronic_chunk(cp, pulse, carrier, offset, n, amp, sample_rate=SAMPLE_RATE)
    return fn


def make_drone_layer(freq, harmonics, mod_rate, mod_depth, amp=0.2):
    phases, mp = [], PhaseState()
    def fn(n, offset):
        return generate_drone_chunk(phases, mp, freq, harmonics, mod_rate, mod_depth, n, amp, SAMPLE_RATE)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

CATALOG = {
    "binaural-beats": {
        "title": "Binaural Beats",
        "description": "Two slightly different frequencies in each ear create a perceived beat in the brain. Requires headphones.",
        "samples": [
            {
                "id": "binaural-delta-2hz",
                "name": "Delta — 2 Hz",
                "description": "Deep sleep & recovery. 2 Hz beat on 100 Hz carrier with brown noise bed.",
                "purpose": "Sleep, physical recovery, deep unconscious processing",
                "research": "Delta entrainment facilitates slow-wave sleep onset (Jirakittayakorn & Wongsawat, 2017)",
                "headphones": True,
                "layers": [make_binaural_layer(100, 2, 0.3), make_noise_layer("brown", 0.35)],
            },
            {
                "id": "binaural-theta-6hz",
                "name": "Theta — 6 Hz",
                "description": "Meditation & creativity. 6 Hz beat on 150 Hz carrier with brown noise.",
                "purpose": "Meditation, creative insight, memory consolidation",
                "research": "Theta entrainment reduces anxiety and induces meditative states (Lavallee et al., 2011)",
                "headphones": True,
                "layers": [make_binaural_layer(150, 6, 0.35), make_noise_layer("brown", 0.28)],
            },
            {
                "id": "binaural-alpha-10hz",
                "name": "Alpha — 10 Hz",
                "description": "Calm relaxation. 10 Hz beat on 200 Hz carrier with pink noise.",
                "purpose": "Stress relief, relaxed wakefulness, light meditation",
                "research": "Alpha entrainment reduces cortisol levels (Huang & Charyton, 2008)",
                "headphones": True,
                "layers": [make_binaural_layer(200, 10, 0.35), make_noise_layer("pink", 0.22)],
            },
            {
                "id": "binaural-beta-18hz",
                "name": "Beta — 18 Hz",
                "description": "Active focus. 18 Hz beat on 250 Hz carrier with pink noise.",
                "purpose": "Concentration, analytical thinking, active problem-solving",
                "research": "Beta entrainment supports sustained attention (Lane et al., 1998)",
                "headphones": True,
                "layers": [make_binaural_layer(250, 18, 0.35), make_noise_layer("pink", 0.20)],
            },
            {
                "id": "binaural-gamma-40hz",
                "name": "Gamma — 40 Hz",
                "description": "Peak cognition. 40 Hz beat on 300 Hz carrier with pink noise.",
                "purpose": "Memory formation, learning, heightened perception",
                "research": "40 Hz gamma enhances attentional binding and memory encoding (Jirakittayakorn & Wongsawat, 2017)",
                "headphones": True,
                "layers": [make_binaural_layer(300, 40, 0.35), make_noise_layer("pink", 0.18)],
            },
        ],
    },
    "isochronic-tones": {
        "title": "Isochronic Tones",
        "description": "Rhythmic amplitude pulses that entrain brainwaves through sharp on/off patterns. Works through speakers — no headphones needed.",
        "samples": [
            {
                "id": "iso-theta-6hz",
                "name": "Theta Pulse — 6 Hz",
                "description": "Meditative pulsing at 6 Hz on a 180 Hz carrier with brown noise.",
                "purpose": "Meditation, relaxation without headphones",
                "research": "Isochronic tones produce stronger cortical steady-state responses than binaural beats (Schwarz & Taylor, 2005)",
                "headphones": False,
                "layers": [make_isochronic_layer(180, 6, 0.30), make_noise_layer("brown", 0.28)],
            },
            {
                "id": "iso-alpha-10hz",
                "name": "Alpha Pulse — 10 Hz",
                "description": "Relaxation pulse at 10 Hz on a 250 Hz carrier with pink noise.",
                "purpose": "Calm focus, stress reduction through speakers",
                "research": "10 Hz auditory stimulation increases alpha power (Calomeni et al., 2017)",
                "headphones": False,
                "layers": [make_isochronic_layer(250, 10, 0.30), make_noise_layer("pink", 0.22)],
            },
            {
                "id": "iso-beta-18hz",
                "name": "Beta Pulse — 18 Hz",
                "description": "Focus pulse at 18 Hz on a 350 Hz carrier with pink noise.",
                "purpose": "Concentration boost through speakers",
                "research": "Beta-frequency auditory stimulation enhances cognitive performance (Kraus & Porubanová, 2015)",
                "headphones": False,
                "layers": [make_isochronic_layer(350, 18, 0.30), make_noise_layer("pink", 0.18)],
            },
            {
                "id": "iso-gamma-40hz",
                "name": "Gamma Pulse — 40 Hz",
                "description": "Cognition pulse at 40 Hz on a 400 Hz carrier with pink noise.",
                "purpose": "Memory and learning enhancement through speakers",
                "research": "40 Hz auditory stimulation shows neuroprotective effects (Martorell et al., 2019 — Nature)",
                "headphones": False,
                "layers": [make_isochronic_layer(400, 40, 0.28), make_noise_layer("pink", 0.18)],
            },
        ],
    },
    "colored-noise": {
        "title": "Colored Noise",
        "description": "Spectrally shaped random signals used for masking distractions and creating ambient texture.",
        "samples": [
            {
                "id": "noise-white",
                "name": "White Noise",
                "description": "Equal energy at all frequencies. Bright, hissy texture like TV static.",
                "purpose": "Maximum masking of environmental sounds, tinnitus relief",
                "research": "White noise improves cognitive performance in noisy environments (Söderlund et al., 2007)",
                "headphones": False,
                "layers": [make_noise_layer("white", 0.35)],
            },
            {
                "id": "noise-pink",
                "name": "Pink Noise",
                "description": "Equal energy per octave (1/f spectrum). Balanced, natural — like steady rainfall.",
                "purpose": "Sleep improvement, focus, background masking",
                "research": "Pink noise during sleep enhances deep sleep and memory (Zhou et al., 2012)",
                "headphones": False,
                "layers": [make_noise_layer("pink", 0.35)],
            },
            {
                "id": "noise-brown",
                "name": "Brown Noise",
                "description": "Low-frequency emphasis (1/f² spectrum). Deep, warm rumble like a waterfall.",
                "purpose": "Deep relaxation, focus for ADHD, wind-down",
                "research": "Low-frequency noise activates parasympathetic nervous system (Alvarsson et al., 2010)",
                "headphones": False,
                "layers": [make_noise_layer("brown", 0.38)],
            },
        ],
    },
    "ambient-drones": {
        "title": "Ambient Drones",
        "description": "Frequency-modulated harmonic tones that create evolving ambient textures. Subliminal harmonic anchoring for immersion.",
        "samples": [
            {
                "id": "drone-deep-bass",
                "name": "Deep Bass Drone",
                "description": "55 Hz sub-bass with slow FM modulation. Grounding, meditative vibration.",
                "purpose": "Grounding, body awareness, meditation anchor",
                "research": "Low-frequency tones activate vestibular system and promote relaxation (Todd & Cody, 2000)",
                "headphones": False,
                "layers": [make_drone_layer(55, [1.0, 2.0], 0.05, 1.5, 0.35), make_noise_layer("brown", 0.15)],
            },
            {
                "id": "drone-warm-mid",
                "name": "Warm Mid-Range Drone",
                "description": "130 Hz drone with gentle harmonics. Comforting, womb-like.",
                "purpose": "Background comfort, stress relief, ambient fill",
                "research": "Sustained tonal environments reduce physiological stress markers (Thoma et al., 2013)",
                "headphones": False,
                "layers": [make_drone_layer(130, [1.0, 2.0, 3.0], 0.12, 3.0, 0.30), make_noise_layer("pink", 0.12)],
            },
            {
                "id": "drone-evolving-harmonic",
                "name": "Evolving Harmonic Drone",
                "description": "200 Hz base with 5 harmonics and complex modulation. Slowly shifting timbre.",
                "purpose": "Creative inspiration, ambient soundscaping",
                "research": "Complex tonal environments stimulate default mode network activity (Taruffi et al., 2017)",
                "headphones": False,
                "layers": [make_drone_layer(200, [1.0, 1.5, 2.0, 3.0, 4.0], 0.25, 8.0, 0.30), make_noise_layer("pink", 0.10)],
            },
        ],
    },
    "soundscapes": {
        "title": "Cognitive Soundscapes",
        "description": "Multi-layered compositions combining binaural beats, noise, and drones for specific cognitive states.",
        "samples": [
            {
                "id": "scape-morning-energizer",
                "name": "Morning Energizer",
                "description": "18 Hz beta binaural + pink noise + bright drone. Wake up and activate.",
                "purpose": "Morning alertness, energy boost, productive start",
                "research": "Beta entrainment combined with moderate noise optimizes morning cortisol utilization",
                "headphones": True,
                "layers": [
                    make_binaural_layer(280, 18, 0.35),
                    make_noise_layer("pink", 0.18),
                    make_drone_layer(140, [1.0, 2.0, 3.0], 0.8, 6.0, 0.15),
                ],
            },
            {
                "id": "scape-deep-focus",
                "name": "Deep Focus",
                "description": "40 Hz gamma binaural + pink noise + subtle drone. Maximum concentration.",
                "purpose": "Deep work, coding, complex problem-solving",
                "research": "Gamma entrainment enhances working memory and sustained attention",
                "headphones": True,
                "layers": [
                    make_binaural_layer(250, 40, 0.35),
                    make_noise_layer("pink", 0.20),
                    make_drone_layer(125, [1.0, 2.0], 0.1, 3.0, 0.12),
                ],
            },
            {
                "id": "scape-creative-flow",
                "name": "Creative Flow",
                "description": "7 Hz theta binaural + pink noise + evolving drone. Unlock divergent thinking.",
                "purpose": "Brainstorming, artistic work, creative problem-solving",
                "research": "Theta-alpha border states correlate with creative insight (Lustenberger et al., 2015)",
                "headphones": True,
                "layers": [
                    make_binaural_layer(180, 7, 0.30),
                    make_noise_layer("pink", 0.22),
                    make_drone_layer(90, [1.0, 2.0, 3.5], 0.18, 5.0, 0.18),
                ],
            },
            {
                "id": "scape-study-session",
                "name": "Study Session",
                "description": "40 Hz gamma binaural + 14 Hz beta isochronic + pink noise. Dual entrainment for learning.",
                "purpose": "Studying, memorization, exam preparation",
                "research": "Gamma-beta coupling enhances memory encoding and retrieval",
                "headphones": True,
                "layers": [
                    make_binaural_layer(300, 40, 0.30),
                    make_isochronic_layer(500, 14, 0.10),
                    make_noise_layer("pink", 0.18),
                ],
            },
            {
                "id": "scape-stress-relief",
                "name": "Stress Relief",
                "description": "10 Hz alpha binaural + brown noise + warm drone. Decompress and let go.",
                "purpose": "Anxiety reduction, post-work decompression, emotional regulation",
                "research": "Alpha entrainment reduces state anxiety scores (Huang & Charyton, 2008)",
                "headphones": True,
                "layers": [
                    make_binaural_layer(190, 10, 0.30),
                    make_noise_layer("brown", 0.30),
                    make_drone_layer(75, [1.0, 2.0], 0.06, 2.0, 0.15),
                ],
            },
            {
                "id": "scape-power-nap",
                "name": "Power Nap",
                "description": "5 Hz theta binaural + brown noise + deep drone. 20-minute nap optimization.",
                "purpose": "Short restorative sleep, energy recovery",
                "research": "Theta entrainment accelerates sleep onset latency (Abeln et al., 2014)",
                "headphones": True,
                "layers": [
                    make_binaural_layer(120, 5, 0.25),
                    make_noise_layer("brown", 0.35),
                    make_drone_layer(50, [1.0, 2.0], 0.03, 1.0, 0.12),
                ],
            },
            {
                "id": "scape-sleep-preparation",
                "name": "Sleep Preparation",
                "description": "2 Hz delta binaural + heavy brown noise + ultra-slow sub-bass. Drift into deep sleep.",
                "purpose": "Insomnia relief, sleep onset, deep rest",
                "research": "Delta entrainment with low-frequency noise promotes slow-wave sleep",
                "headphones": True,
                "layers": [
                    make_binaural_layer(100, 2, 0.20),
                    make_noise_layer("brown", 0.40),
                    make_drone_layer(40, [1.0, 2.0], 0.02, 0.8, 0.10),
                ],
            },
            {
                "id": "scape-flow-state",
                "name": "Flow State",
                "description": "10 Hz alpha binaural + 40 Hz gamma isochronic + brown noise + drone. Alpha-gamma coupling.",
                "purpose": "Peak performance, athletic visualization, flow state induction",
                "research": "Alpha-gamma cross-frequency coupling observed during flow (Katahira et al., 2018)",
                "headphones": True,
                "layers": [
                    make_binaural_layer(200, 10, 0.28),
                    make_isochronic_layer(400, 40, 0.10),
                    make_noise_layer("brown", 0.22),
                    make_drone_layer(100, [1.0, 2.0], 0.08, 2.0, 0.10),
                ],
            },
        ],
    },
    "educational": {
        "title": "Educational Demos",
        "description": "Demonstrations that illustrate the building blocks of psychoacoustic audio engineering.",
        "samples": [
            {
                "id": "demo-pure-binaural",
                "name": "Pure Binaural Beat (No Noise)",
                "description": "Raw 10 Hz binaural beat on 200 Hz carrier. Hear the beat frequency clearly in isolation.",
                "purpose": "Understanding how binaural beats work",
                "research": "The brain perceives a phantom beat at |f_right - f_left| Hz",
                "headphones": True,
                "layers": [make_binaural_layer(200, 10, 0.45)],
            },
            {
                "id": "demo-pure-isochronic",
                "name": "Pure Isochronic Tone (No Noise)",
                "description": "Raw 10 Hz isochronic pulse on 300 Hz carrier. Hear the rhythmic amplitude modulation.",
                "purpose": "Understanding how isochronic tones work",
                "research": "Sharp amplitude onsets drive stronger neural entrainment than smooth beats",
                "headphones": False,
                "layers": [make_isochronic_layer(300, 10, 0.40)],
            },
            {
                "id": "demo-carrier-comparison",
                "name": "Carrier Frequency Comparison",
                "description": "Same 10 Hz beat at three carriers: 100 Hz (deep), 250 Hz (mid), 500 Hz (bright). 40s each.",
                "purpose": "Hear how carrier frequency affects perceived warmth/brightness",
                "research": "Lower carriers feel warmer; higher carriers feel more alert",
                "headphones": True,
                "layers": "custom_carrier_comparison",
            },
            {
                "id": "demo-noise-comparison",
                "name": "Noise Color Comparison",
                "description": "White, pink, then brown noise — 40 seconds each. Hear the spectral differences.",
                "purpose": "Understanding noise color characteristics",
                "research": "White=flat, Pink=1/f (balanced), Brown=1/f² (deep)",
                "headphones": False,
                "layers": "custom_noise_comparison",
            },
        ],
    },
}


def generate_carrier_comparison(filepath):
    """Generate 3 x 40s segments at different carrier frequencies."""
    segment_dur = 40.0
    total_samples = int(3 * segment_dur * SAMPLE_RATE)
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)
    seg_samples = int(segment_dur * SAMPLE_RATE)
    xfade = int(2.0 * SAMPLE_RATE)  # 2s crossfade between segments

    carriers = [100.0, 250.0, 500.0]
    segments = []

    for carrier in carriers:
        lp, rp = PhaseState(), PhaseState()
        seg = generate_binaural_chunk(lp, rp, carrier, 10.0, seg_samples, 0.40, SAMPLE_RATE)
        segments.append(seg)

    # Concatenate with crossfades
    full = segments[0].copy()
    for seg in segments[1:]:
        # Fade out tail of current
        fade_out = raised_cosine_fade(xfade)[::-1].reshape(-1, 1)
        fade_in = raised_cosine_fade(xfade).reshape(-1, 1)
        full[-xfade:] *= fade_out
        seg[:xfade] *= fade_in
        full[-xfade:] += seg[:xfade]
        full = np.vstack((full, seg[xfade:]))

    # File-level fade
    fi = raised_cosine_fade(fade_samples).reshape(-1, 1)
    fo = raised_cosine_fade(fade_samples)[::-1].reshape(-1, 1)
    full[:fade_samples] *= fi
    full[-fade_samples:] *= fo

    full = np.clip(full, -1.0, 1.0)
    with StreamingWavWriter(filepath, SAMPLE_RATE) as w:
        w.write_chunk(full)


def generate_noise_comparison(filepath):
    """Generate white → pink → brown noise, 40s each."""
    segment_dur = 40.0
    fade_samples = int(FADE_SECONDS * SAMPLE_RATE)
    seg_samples = int(segment_dur * SAMPLE_RATE)
    xfade = int(2.0 * SAMPLE_RATE)

    colors = ["white", "pink", "brown"]
    segments = []
    for color in colors:
        noise, _ = generate_noise_chunk(color, seg_samples, 0.35, None, sample_rate=SAMPLE_RATE)
        segments.append(noise)

    full = segments[0].copy()
    for seg in segments[1:]:
        fade_out = raised_cosine_fade(xfade)[::-1].reshape(-1, 1)
        fade_in = raised_cosine_fade(xfade).reshape(-1, 1)
        full[-xfade:] *= fade_out
        seg[:xfade] *= fade_in
        full[-xfade:] += seg[:xfade]
        full = np.vstack((full, seg[xfade:]))

    fi = raised_cosine_fade(fade_samples).reshape(-1, 1)
    fo = raised_cosine_fade(fade_samples)[::-1].reshape(-1, 1)
    full[:fade_samples] *= fi
    full[-fade_samples:] *= fo

    full = np.clip(full, -1.0, 1.0)
    with StreamingWavWriter(filepath, SAMPLE_RATE) as w:
        w.write_chunk(full)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_count = sum(len(cat["samples"]) for cat in CATALOG.values())
    print("=" * 70)
    print("  NEUROACOUSTIC ENGINE — Showcase Sample Library Generator")
    print(f"  {total_count} samples x 2 minutes each")
    print(f"  Sample rate: {SAMPLE_RATE} Hz | Stereo 16-bit")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 70)

    generated = []
    start_all = time.time()

    for cat_id, cat in CATALOG.items():
        print(f"\n{'─' * 70}")
        print(f"  {cat['title'].upper()}")
        print(f"  {cat['description']}")
        print(f"{'─' * 70}")

        for sample in cat["samples"]:
            sid = sample["id"]
            filename = f"{sid}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)

            print(f"\n  [{sid}] {sample['name']}")
            print(f"    {sample['description']}")

            t0 = time.time()

            if sample["layers"] == "custom_carrier_comparison":
                generate_carrier_comparison(filepath)
            elif sample["layers"] == "custom_noise_comparison":
                generate_noise_comparison(filepath)
            else:
                generate_single(filepath, DURATION, sample["layers"])

            elapsed = time.time() - t0
            size = format_size(filepath)
            print(f"    -> {filename} | {size} | {elapsed:.1f}s")

            generated.append({
                "category": cat_id,
                "category_title": cat["title"],
                "category_description": cat["description"],
                **sample,
                "filename": filename,
                "size": size,
            })

    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".wav")
    )
    total_time = time.time() - start_all

    print(f"\n{'=' * 70}")
    print(f"  SHOWCASE LIBRARY COMPLETE")
    print(f"  {len(generated)} samples generated in {total_time:.0f}s")
    print(f"  Total size: {total_size / 1_048_576:.1f} MB")
    print(f"  Location: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'=' * 70}")

    # Write catalog JSON for the site builder
    import json
    catalog_path = os.path.join(OUTPUT_DIR, "catalog.json")
    with open(catalog_path, "w") as f:
        json.dump({"categories": CATALOG, "generated": generated}, f, indent=2, default=str)
    print(f"\n  Catalog metadata: {catalog_path}")


if __name__ == "__main__":
    main()
