"""
Cognitive Audio Synthesizer using the pyo DSP framework.

Generates binaural beats via two hard-panned sine oscillators with
SigTo-based parameter interpolation for click-free transitions.
"""

import time

from pyo import Mix, Pan, Server, Sine, SigTo

from neuroacoustic.core.config import (
    DEFAULT_AMPLITUDE,
    DEFAULT_FADE_TIME,
    DEFAULT_INTERPOLATION_TIME,
)


class CognitiveAudioSynthesizer:
    """
    Real-time local audio generation module using the pyo C-level DSP server.

    Generates binaural beats by driving two independent sine oscillators
    at slightly different frequencies, hard-panned to left and right channels.
    All parameter changes use SigTo interpolation to prevent audible artifacts.
    """

    def __init__(
        self,
        interpolation_time: float = DEFAULT_INTERPOLATION_TIME,
        fade_time: float = DEFAULT_FADE_TIME,
    ):
        self._interpolation_time = interpolation_time
        self._fade_time = fade_time

        # Initialize pyo audio server (output only, no mic input)
        # Use 'default' ALSA device which routes through PipeWire/PulseAudio
        self.server = Server(duplex=0)
        self.server.setOutputDevice(self._find_output_device())
        self.server.boot()

        # SigTo objects provide smooth interpolation for all parameter changes
        self._freq_left = SigTo(value=200.0, time=interpolation_time)
        self._freq_right = SigTo(value=200.0, time=interpolation_time)
        self._amplitude = SigTo(value=0.0, time=fade_time)

        # Two independent sine oscillators for binaural beat generation
        self.osc_left = Sine(freq=self._freq_left, mul=self._amplitude)
        self.osc_right = Sine(freq=self._freq_right, mul=self._amplitude)

        # Hard-pan: left oscillator to left ear, right oscillator to right ear
        self.pan_left = Pan(self.osc_left, outs=2, pan=0.0)
        self.pan_right = Pan(self.osc_right, outs=2, pan=1.0)

        # Combine both panned signals into a stereo mix
        self.mixer = Mix([self.pan_left, self.pan_right], voices=2)

        # Semantic metadata for the current audio state
        self.current_metadata = {
            "intent": "idle",
            "data_source": "system_init",
            "urgency": "none",
            "frequencies": [],
        }

    @staticmethod
    def _find_output_device() -> int:
        """Find the best output device, preferring 'default' or 'pipewire'."""
        from pyo import pa_get_output_devices
        names, ids = pa_get_output_devices()
        # Prefer 'default' (routes through PipeWire/PulseAudio)
        for name, dev_id in zip(names, ids):
            if name == "default":
                return dev_id
        for name, dev_id in zip(names, ids):
            if "pipewire" in name.lower():
                return dev_id
        return ids[0] if ids else 0

    def start(self) -> None:
        """Start the audio server and begin outputting the synthesized mix."""
        self.server.start()
        self.mixer.out()

    def stop(self) -> None:
        """Gracefully fade out and stop the audio server."""
        self._amplitude.value = 0.0
        time.sleep(self._fade_time + 0.1)
        self.server.stop()

    def update_state(
        self,
        carrier_freq: float,
        beat_freq: float,
        amplitude: float = DEFAULT_AMPLITUDE,
        metadata: dict | None = None,
    ) -> None:
        """
        Update the DSP parameters to generate a specific binaural beat.

        The carrier frequency determines the perceived pitch. The beat frequency
        is the difference between left and right channels that the brain
        perceives as the entrainment pulse.

        Left ear:  carrier_freq - (beat_freq / 2)
        Right ear: carrier_freq + (beat_freq / 2)

        Args:
            carrier_freq: Center frequency in Hz (perceived pitch).
            beat_freq: Binaural beat frequency in Hz (entrainment target).
            amplitude: Output amplitude (0.0 to 1.0).
            metadata: Semantic metadata dict to attach to this state.
        """
        left_hz = carrier_freq - (beat_freq / 2.0)
        right_hz = carrier_freq + (beat_freq / 2.0)

        self._freq_left.value = left_hz
        self._freq_right.value = right_hz
        self._amplitude.value = amplitude

        if metadata is not None:
            self.current_metadata = metadata.copy()
        self.current_metadata["frequencies"] = [round(left_hz, 2), round(right_hz, 2)]

    def get_metadata(self) -> dict:
        """Return the metadata representing the current acoustic state."""
        return self.current_metadata.copy()


if __name__ == "__main__":
    synth = CognitiveAudioSynthesizer()
    synth.start()

    # Transition to a Theta relaxation state (5Hz beat on 150Hz carrier)
    test_meta = {
        "intent": "relaxation",
        "data_source": "calendar_event",
        "urgency": "low",
    }
    synth.update_state(carrier_freq=150.0, beat_freq=6.0, amplitude=0.3, metadata=test_meta)

    time.sleep(5)
    print(synth.get_metadata())
    synth.stop()
