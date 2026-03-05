"""
Semantic-to-Acoustic Translation Layer.

Translates parsed semantic metadata (intent, urgency, sentiment) from
the ingestion agent into specific psychoacoustic parameters for the
audio synthesis engine.
"""

import json

from neuroacoustic.core.config import (
    INTENT_PROFILES,
    SENTIMENT_MODIFIERS,
    URGENCY_MODIFIERS,
)


class SemanticMapper:
    """
    Maps structured event data to DSP parameters using psychoacoustic rules.

    The mapping pipeline:
    1. Resolve intent profile (focus, relaxation, alertness, learning)
    2. Apply urgency modifiers (amplitude, beat frequency scaling)
    3. Apply sentiment modifiers (carrier shift, timbre brightness)
    4. Attach source metadata for debugging and logging
    """

    def __init__(self):
        self.intent_profiles = INTENT_PROFILES
        self.urgency_modifiers = URGENCY_MODIFIERS
        self.sentiment_modifiers = SENTIMENT_MODIFIERS

    def translate_payload(self, json_payload: str) -> dict:
        """
        Parse a JSON payload and calculate final DSP parameters.

        Args:
            json_payload: JSON string with keys like "intent", "urgency",
                         "sentiment", "data_source", "content_summary".

        Returns:
            Dictionary of audio parameters:
                - carrier_freq: Hz
                - beat_freq: Hz
                - target_amplitude: 0.0-1.0
                - noise_color: "white", "pink", or "brown"
                - timbre_brightness: 0.0-1.0
                - original_metadata: the parsed input dict
        """
        try:
            data = json.loads(json_payload)
        except (json.JSONDecodeError, TypeError):
            return self._default_state()

        intent = data.get("intent", "relaxation").lower()
        urgency = data.get("urgency", "low").lower()
        sentiment = data.get("sentiment", "neutral").lower()

        # Resolve base profile from intent
        profile = self.intent_profiles.get(intent, self.intent_profiles["relaxation"])
        carrier_freq = profile["carrier_base"]
        beat_freq = profile["beat_freq"]
        amplitude = 0.3
        timbre_brightness = 0.5

        # Apply urgency modifiers
        urg_mod = self.urgency_modifiers.get(urgency, self.urgency_modifiers["medium"])
        amplitude = urg_mod["amplitude"]
        beat_freq *= urg_mod["beat_freq_multiplier"]
        carrier_freq += urg_mod["carrier_shift_hz"]

        # Apply sentiment modifiers
        sent_mod = self.sentiment_modifiers.get(
            sentiment, self.sentiment_modifiers["neutral"]
        )
        carrier_freq *= sent_mod["carrier_multiplier"]
        timbre_brightness = sent_mod["timbre_brightness"]

        return {
            "carrier_freq": round(carrier_freq, 2),
            "beat_freq": round(beat_freq, 2),
            "target_amplitude": amplitude,
            "noise_color": profile["noise_color"],
            "timbre_brightness": timbre_brightness,
            "target_band": profile["target_band"],
            "original_metadata": data,
        }

    def _default_state(self) -> dict:
        """Provide a safe fallback state for invalid or missing input."""
        return {
            "carrier_freq": 100.0,
            "beat_freq": 4.0,
            "target_amplitude": 0.1,
            "noise_color": "brown",
            "timbre_brightness": 0.1,
            "target_band": "theta",
            "original_metadata": {"intent": "error_fallback"},
        }


if __name__ == "__main__":
    mapper = SemanticMapper()

    # Simulate a payload from the ingestion agent
    tech_news_payload = json.dumps(
        {
            "intent": "alertness",
            "data_source": "rss_feed",
            "urgency": "medium",
            "sentiment": "positive",
            "content_summary": "Major breakthrough in solid-state battery technology.",
        }
    )

    audio_params = mapper.translate_payload(tech_news_payload)

    print("Parsed Text to Audio Parameters:")
    for key, value in audio_params.items():
        print(f"  {key}: {value}")
