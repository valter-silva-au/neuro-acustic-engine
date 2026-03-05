"""
Global psychoacoustic configuration constants.

Defines brainwave frequency bands, intent-to-DSP mappings,
urgency/sentiment modifiers, and time-of-day schedule profiles.
"""

# Brainwave frequency bands and their cognitive associations
BRAINWAVE_BANDS = {
    "delta": {
        "range_hz": (0.5, 4.0),
        "cognitive_state": "deep sleep, healing, recovery",
        "default_carrier_hz": 100.0,
    },
    "theta": {
        "range_hz": (4.0, 8.0),
        "cognitive_state": "meditation, creativity, memory consolidation",
        "default_carrier_hz": 150.0,
    },
    "alpha": {
        "range_hz": (8.0, 14.0),
        "cognitive_state": "relaxation, calm focus, flow state",
        "default_carrier_hz": 200.0,
    },
    "beta": {
        "range_hz": (14.0, 30.0),
        "cognitive_state": "active thinking, concentration, alertness",
        "default_carrier_hz": 250.0,
    },
    "gamma": {
        "range_hz": (30.0, 100.0),
        "cognitive_state": "peak cognition, memory formation, insight",
        "default_carrier_hz": 400.0,
    },
}

# Intent profiles: cognitive state -> base DSP parameters
INTENT_PROFILES = {
    "focus": {
        "beat_freq": 40.0,
        "carrier_base": 432.0,
        "noise_color": "pink",
        "target_band": "gamma",
    },
    "relaxation": {
        "beat_freq": 5.0,
        "carrier_base": 110.0,
        "noise_color": "brown",
        "target_band": "theta",
    },
    "alertness": {
        "beat_freq": 20.0,
        "carrier_base": 250.0,
        "noise_color": "white",
        "target_band": "beta",
    },
    "learning": {
        "beat_freq": 40.0,
        "carrier_base": 528.0,
        "noise_color": "pink",
        "target_band": "gamma",
    },
}

# Urgency modifiers applied multiplicatively/additively to base params
URGENCY_MODIFIERS = {
    "low": {
        "amplitude": 0.2,
        "beat_freq_multiplier": 1.0,
        "carrier_shift_hz": 0.0,
    },
    "medium": {
        "amplitude": 0.3,
        "beat_freq_multiplier": 1.0,
        "carrier_shift_hz": 0.0,
    },
    "high": {
        "amplitude": 0.6,
        "beat_freq_multiplier": 1.25,
        "carrier_shift_hz": 0.0,
    },
    "critical": {
        "amplitude": 0.8,
        "beat_freq_multiplier": 1.5,
        "carrier_shift_hz": 30.0,
    },
}

# Sentiment modifiers for tonal quality
SENTIMENT_MODIFIERS = {
    "positive": {
        "carrier_multiplier": 1.5,  # Perfect fifth interval
        "timbre_brightness": 0.8,
        "harmonic_ratios": [1.0, 2.0, 3.0],
    },
    "neutral": {
        "carrier_multiplier": 1.0,
        "timbre_brightness": 0.5,
        "harmonic_ratios": [1.0, 2.0],
    },
    "negative": {
        "carrier_multiplier": 0.8,
        "timbre_brightness": 0.2,
        "harmonic_ratios": [1.0, 1.5],  # Tritone-adjacent for tension
    },
}

# Time-of-day schedule profiles
TIME_SCHEDULES = {
    "early_morning": {
        "hours": (5, 7),
        "target_band": "alpha",
        "beat_freq": 10.0,
        "carrier_base": 200.0,
        "noise_color": "pink",
        "noise_level": 0.25,
        "modulation_rate": 0.3,
    },
    "morning_focus": {
        "hours": (7, 12),
        "target_band": "beta",
        "beat_freq": 18.0,
        "carrier_base": 250.0,
        "noise_color": "pink",
        "noise_level": 0.2,
        "modulation_rate": 0.8,
    },
    "midday_peak": {
        "hours": (12, 14),
        "target_band": "beta",
        "beat_freq": 22.0,
        "carrier_base": 300.0,
        "noise_color": "white",
        "noise_level": 0.15,
        "modulation_rate": 1.2,
    },
    "afternoon": {
        "hours": (14, 17),
        "target_band": "beta",
        "beat_freq": 15.0,
        "carrier_base": 230.0,
        "noise_color": "brown",
        "noise_level": 0.2,
        "modulation_rate": 0.6,
    },
    "evening_wind_down": {
        "hours": (17, 21),
        "target_band": "alpha",
        "beat_freq": 9.0,
        "carrier_base": 180.0,
        "noise_color": "pink",
        "noise_level": 0.3,
        "modulation_rate": 0.2,
    },
    "night": {
        "hours": (21, 5),
        "target_band": "theta",
        "beat_freq": 5.0,
        "carrier_base": 120.0,
        "noise_color": "brown",
        "noise_level": 0.4,
        "modulation_rate": 0.1,
    },
}

# Activity type overrides (take precedence over time-of-day)
ACTIVITY_OVERRIDES = {
    "meeting": {
        "target_band": "beta",
        "beat_freq": 16.0,
        "noise_color": "pink",
        "noise_level": 0.12,
    },
    "deep_work": {
        "target_band": "gamma",
        "beat_freq": 40.0,
        "noise_color": "brown",
        "noise_level": 0.08,
    },
    "creative_work": {
        "target_band": "theta",
        "beat_freq": 6.0,
        "noise_color": "pink",
        "noise_level": 0.25,
    },
    "break": {
        "target_band": "alpha",
        "beat_freq": 10.0,
        "noise_color": "brown",
        "noise_level": 0.35,
    },
    "exercise": {
        "target_band": "beta",
        "beat_freq": 22.0,
        "noise_color": "white",
        "noise_level": 0.15,
    },
}

# Default SigTo interpolation time (seconds) for smooth parameter transitions
DEFAULT_INTERPOLATION_TIME = 0.1

# Default amplitude fade time (seconds) for start/stop
DEFAULT_FADE_TIME = 0.5

# Default base amplitude
DEFAULT_AMPLITUDE = 0.3
