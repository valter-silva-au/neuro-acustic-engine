"""Tests for the synthesis module."""

import numpy as np
import pytest

from neuroacoustic.synthesis.binaural import (
    calculate_binaural_frequencies,
    classify_beat_band,
    get_default_carrier_for_band,
)
from neuroacoustic.synthesis.isochronic import (
    generate_isochronic_envelope,
    generate_isochronic_tone,
)
from neuroacoustic.synthesis.noise_generators import (
    generate_brown_noise,
    generate_noise,
    generate_pink_noise,
    generate_white_noise,
)


class TestBinauralHelpers:
    def test_frequency_calculation(self):
        left, right = calculate_binaural_frequencies(200.0, 10.0)
        assert left == 195.0
        assert right == 205.0

    def test_frequency_calculation_zero_beat(self):
        left, right = calculate_binaural_frequencies(440.0, 0.0)
        assert left == 440.0
        assert right == 440.0

    def test_classify_theta(self):
        assert classify_beat_band(6.0) == "theta"

    def test_classify_alpha(self):
        assert classify_beat_band(10.0) == "alpha"

    def test_classify_beta(self):
        assert classify_beat_band(20.0) == "beta"

    def test_classify_gamma(self):
        assert classify_beat_band(40.0) == "gamma"

    def test_classify_out_of_range(self):
        assert classify_beat_band(200.0) is None

    def test_default_carrier_theta(self):
        assert get_default_carrier_for_band("theta") == 150.0

    def test_default_carrier_unknown(self):
        assert get_default_carrier_for_band("unknown") is None


class TestIsochronicTones:
    def test_envelope_shape(self):
        env = generate_isochronic_envelope(10.0, 1.0, sample_rate=44100)
        assert len(env) == 44100
        assert env.min() >= 0.0
        assert env.max() <= 1.0

    def test_tone_stereo_output(self):
        tone = generate_isochronic_tone(440.0, 10.0, 0.5)
        assert tone.shape == (22050, 2)

    def test_tone_amplitude_within_bounds(self):
        tone = generate_isochronic_tone(440.0, 10.0, 1.0, amplitude=0.5)
        assert np.max(np.abs(tone)) <= 0.6  # Allow small headroom for smoothing


class TestNoiseGenerators:
    def test_white_noise_shape(self):
        noise = generate_white_noise(0.5, amplitude=0.3)
        assert noise.shape == (22050, 2)

    def test_white_noise_amplitude(self):
        noise = generate_white_noise(1.0, amplitude=0.3)
        assert np.max(np.abs(noise)) <= 0.3 + 0.01

    def test_pink_noise_shape(self):
        noise = generate_pink_noise(0.5, amplitude=0.3)
        assert noise.shape == (22050, 2)

    def test_pink_noise_normalized(self):
        noise = generate_pink_noise(1.0, amplitude=0.5)
        assert np.max(np.abs(noise)) <= 0.5 + 0.01

    def test_brown_noise_shape(self):
        noise = generate_brown_noise(0.5, amplitude=0.3)
        assert noise.shape == (22050, 2)

    def test_brown_noise_normalized(self):
        noise = generate_brown_noise(1.0, amplitude=0.4)
        assert np.max(np.abs(noise)) <= 0.4 + 0.01

    def test_generate_noise_dispatcher(self):
        for color in ["white", "pink", "brown"]:
            noise = generate_noise(color, 0.5)
            assert noise.shape[1] == 2

    def test_generate_noise_invalid_color(self):
        with pytest.raises(ValueError, match="Unknown noise color"):
            generate_noise("purple", 0.5)

    def test_pink_noise_spectral_profile(self):
        """Pink noise should have more low-frequency energy than white noise."""
        pink = generate_pink_noise(2.0, amplitude=0.5, sample_rate=44100)
        white = generate_white_noise(2.0, amplitude=0.5, sample_rate=44100)

        # Compare energy in low band (0-500 Hz) vs high band (5000-22050 Hz)
        pink_fft = np.abs(np.fft.rfft(pink[:, 0]))
        white_fft = np.abs(np.fft.rfft(white[:, 0]))
        freqs = np.fft.rfftfreq(len(pink), 1 / 44100)

        low_mask = freqs < 500
        high_mask = freqs > 5000

        pink_ratio = pink_fft[low_mask].mean() / max(pink_fft[high_mask].mean(), 1e-10)
        white_ratio = white_fft[low_mask].mean() / max(white_fft[high_mask].mean(), 1e-10)

        # Pink noise should have a much higher low/high ratio than white
        assert pink_ratio > white_ratio
