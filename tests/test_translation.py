"""Tests for the semantic translation module."""

import json

from neuroacoustic.translation.semantic_mapper import SemanticMapper
from neuroacoustic.translation.timbre_space import (
    compute_attack_time_ms,
    compute_harmonic_amplitudes,
    compute_inharmonicity,
    compute_spectral_centroid_target,
)


class TestSemanticMapper:
    def setup_method(self):
        self.mapper = SemanticMapper()

    def test_focus_intent(self):
        payload = json.dumps({"intent": "focus", "urgency": "medium", "sentiment": "neutral"})
        result = self.mapper.translate_payload(payload)
        assert result["target_band"] == "gamma"
        assert result["beat_freq"] == 40.0
        assert result["noise_color"] == "pink"

    def test_relaxation_intent(self):
        payload = json.dumps({"intent": "relaxation", "urgency": "low", "sentiment": "neutral"})
        result = self.mapper.translate_payload(payload)
        assert result["target_band"] == "theta"
        assert result["beat_freq"] == 5.0
        assert result["noise_color"] == "brown"

    def test_high_urgency_increases_beat_freq(self):
        base = json.dumps({"intent": "alertness", "urgency": "low", "sentiment": "neutral"})
        urgent = json.dumps({"intent": "alertness", "urgency": "high", "sentiment": "neutral"})
        base_result = self.mapper.translate_payload(base)
        urgent_result = self.mapper.translate_payload(urgent)
        assert urgent_result["beat_freq"] > base_result["beat_freq"]

    def test_high_urgency_increases_amplitude(self):
        base = json.dumps({"intent": "focus", "urgency": "low", "sentiment": "neutral"})
        urgent = json.dumps({"intent": "focus", "urgency": "high", "sentiment": "neutral"})
        base_result = self.mapper.translate_payload(base)
        urgent_result = self.mapper.translate_payload(urgent)
        assert urgent_result["target_amplitude"] > base_result["target_amplitude"]

    def test_positive_sentiment_raises_carrier(self):
        neutral = json.dumps({"intent": "alertness", "urgency": "medium", "sentiment": "neutral"})
        positive = json.dumps({"intent": "alertness", "urgency": "medium", "sentiment": "positive"})
        neutral_result = self.mapper.translate_payload(neutral)
        positive_result = self.mapper.translate_payload(positive)
        assert positive_result["carrier_freq"] > neutral_result["carrier_freq"]

    def test_negative_sentiment_lowers_carrier(self):
        neutral = json.dumps({"intent": "alertness", "urgency": "medium", "sentiment": "neutral"})
        negative = json.dumps({"intent": "alertness", "urgency": "medium", "sentiment": "negative"})
        neutral_result = self.mapper.translate_payload(neutral)
        negative_result = self.mapper.translate_payload(negative)
        assert negative_result["carrier_freq"] < neutral_result["carrier_freq"]

    def test_invalid_json_returns_default(self):
        result = self.mapper.translate_payload("not valid json")
        assert result["carrier_freq"] == 100.0
        assert result["beat_freq"] == 4.0
        assert result["target_band"] == "theta"

    def test_missing_fields_use_defaults(self):
        payload = json.dumps({})
        result = self.mapper.translate_payload(payload)
        # Should default to relaxation intent
        assert result["target_band"] == "theta"

    def test_metadata_preserved(self):
        payload = json.dumps({
            "intent": "focus",
            "urgency": "medium",
            "sentiment": "neutral",
            "data_source": "calendar",
        })
        result = self.mapper.translate_payload(payload)
        assert result["original_metadata"]["data_source"] == "calendar"

    def test_timbre_brightness_positive(self):
        positive = json.dumps({"intent": "focus", "urgency": "medium", "sentiment": "positive"})
        result = self.mapper.translate_payload(positive)
        assert result["timbre_brightness"] == 0.8

    def test_timbre_brightness_negative(self):
        negative = json.dumps({"intent": "focus", "urgency": "medium", "sentiment": "negative"})
        result = self.mapper.translate_payload(negative)
        assert result["timbre_brightness"] == 0.2


class TestTimbreSpace:
    def test_spectral_centroid_bright(self):
        centroid = compute_spectral_centroid_target(1.0, 200.0)
        assert centroid == 800.0  # 200 * (1 + 1*3)

    def test_spectral_centroid_dark(self):
        centroid = compute_spectral_centroid_target(0.0, 200.0)
        assert centroid == 200.0

    def test_attack_time_urgency(self):
        assert compute_attack_time_ms("critical") < compute_attack_time_ms("low")
        assert compute_attack_time_ms("high") < compute_attack_time_ms("medium")

    def test_inharmonicity_negative(self):
        assert compute_inharmonicity("negative") > compute_inharmonicity("positive")

    def test_inharmonicity_positive_is_pure(self):
        assert compute_inharmonicity("positive") == 0.0

    def test_harmonic_amplitudes_count(self):
        harmonics = compute_harmonic_amplitudes([1.0, 2.0], 0.0, num_harmonics=4)
        assert len(harmonics) == 4

    def test_harmonic_amplitudes_rolloff(self):
        harmonics = compute_harmonic_amplitudes([1.0, 2.0, 3.0], 0.0)
        amplitudes = [amp for _, amp in harmonics]
        # Each harmonic should be quieter than the previous
        for i in range(1, len(amplitudes)):
            assert amplitudes[i] < amplitudes[i - 1]

    def test_harmonic_detuning_with_inharmonicity(self):
        pure = compute_harmonic_amplitudes([1.0, 2.0], 0.0, num_harmonics=3)
        detuned = compute_harmonic_amplitudes([1.0, 2.0], 0.25, num_harmonics=3)
        # Higher harmonics should be more detuned
        for i in range(1, 3):
            pure_freq = pure[i][0]
            detuned_freq = detuned[i][0]
            assert detuned_freq != pure_freq
