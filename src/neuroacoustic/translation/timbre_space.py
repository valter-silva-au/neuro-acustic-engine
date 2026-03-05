"""
Timbre space mappings based on multidimensional psychoacoustic scaling.

Maps semantic dimensions (sentiment, urgency, complexity) to physical
acoustic properties: spectral centroid, envelope attack time, and
inharmonicity.
"""


def compute_spectral_centroid_target(timbre_brightness: float, carrier_freq: float) -> float:
    """
    Compute the target spectral centroid based on brightness and carrier.

    Higher brightness shifts energy toward higher harmonics, making the
    sound feel sharper and more alert. Lower brightness produces a darker,
    warmer tone suitable for relaxation.

    Args:
        timbre_brightness: Normalized brightness (0.0=dark, 1.0=bright).
        carrier_freq: The base carrier frequency in Hz.

    Returns:
        Target spectral centroid in Hz.
    """
    # Centroid ranges from 1x to 4x the carrier based on brightness
    multiplier = 1.0 + (timbre_brightness * 3.0)
    return carrier_freq * multiplier


def compute_attack_time_ms(urgency: str) -> float:
    """
    Compute envelope attack time from urgency level.

    High urgency produces short, sharp onsets that demand attention.
    Low urgency produces long, gentle swells that blend into the background.

    Args:
        urgency: One of "low", "medium", "high", "critical".

    Returns:
        Attack time in milliseconds.
    """
    attack_map = {
        "low": 800.0,
        "medium": 400.0,
        "high": 100.0,
        "critical": 30.0,
    }
    return attack_map.get(urgency, 400.0)


def compute_inharmonicity(sentiment: str) -> float:
    """
    Compute inharmonicity factor from sentiment.

    Negative sentiment introduces detuning of overtones from integer
    multiples of the fundamental, creating tension and unease.
    Positive sentiment maintains pure harmonic ratios for consonance.

    Args:
        sentiment: One of "positive", "neutral", "negative".

    Returns:
        Inharmonicity factor (0.0=pure harmonics, 1.0=maximum detuning).
    """
    inharmonicity_map = {
        "positive": 0.0,
        "neutral": 0.05,
        "negative": 0.25,
    }
    return inharmonicity_map.get(sentiment, 0.05)


def compute_harmonic_amplitudes(
    harmonic_ratios: list[float],
    inharmonicity: float,
    num_harmonics: int = 6,
) -> list[tuple[float, float]]:
    """
    Compute harmonic frequencies and amplitudes for a given timbre profile.

    Args:
        harmonic_ratios: Base harmonic multipliers (e.g. [1.0, 2.0, 3.0]).
        inharmonicity: Detuning factor (0.0 to 1.0).
        num_harmonics: Number of harmonics to generate.

    Returns:
        List of (frequency_multiplier, amplitude) tuples. The frequency
        multipliers are slightly detuned from integer ratios when
        inharmonicity > 0.
    """
    harmonics = []
    for i in range(num_harmonics):
        if i < len(harmonic_ratios):
            base_ratio = harmonic_ratios[i]
        else:
            base_ratio = float(i + 1)

        # Apply inharmonicity: detune proportionally to harmonic number
        detune = inharmonicity * (i * 0.02)
        freq_mult = base_ratio + detune

        # Amplitude rolls off with harmonic number
        amp = 1.0 / (i + 1)

        harmonics.append((freq_mult, amp))

    return harmonics
