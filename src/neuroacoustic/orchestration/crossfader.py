"""
Constant-power crossfade implementation.

Provides mathematically correct crossfading between audio states that
maintains consistent perceived loudness throughout the transition.
A linear crossfade causes a perceived volume dip at the midpoint because
acoustic power is proportional to amplitude squared.
"""

import math


def constant_power_gains(t: float) -> tuple[float, float]:
    """
    Calculate constant-power crossfade gains for a given transition point.

    At the midpoint (t=0), both signals are at sqrt(0.5) ~ 0.707 amplitude,
    ensuring their combined power equals 1.0 (no perceived volume change).

    Args:
        t: Normalized transition position from -1.0 (start, only signal A)
           to +1.0 (end, only signal B). 0.0 is the midpoint.

    Returns:
        Tuple of (gain_outgoing, gain_incoming).
    """
    t_clamped = max(-1.0, min(1.0, t))
    gain_out = math.sqrt(0.5 * (1.0 - t_clamped))
    gain_in = math.sqrt(0.5 * (1.0 + t_clamped))
    return (gain_out, gain_in)


def interpolate_parameter(
    start_value: float,
    end_value: float,
    progress: float,
) -> float:
    """
    Linearly interpolate a DSP parameter between two states.

    Used for smooth frequency glides (e.g., beat frequency transitioning
    from 20Hz beta to 5Hz theta over 30 seconds).

    Args:
        start_value: Parameter value at the beginning of transition.
        end_value: Parameter value at the end of transition.
        progress: Normalized progress from 0.0 (start) to 1.0 (end).

    Returns:
        Interpolated parameter value.
    """
    progress_clamped = max(0.0, min(1.0, progress))
    return start_value + (end_value - start_value) * progress_clamped


def calculate_transition_steps(
    duration_seconds: float,
    sample_rate: int = 44100,
    block_size: int = 2048,
) -> int:
    """
    Calculate the number of audio blocks in a crossfade transition.

    Args:
        duration_seconds: Total transition duration.
        sample_rate: Audio sample rate.
        block_size: Audio buffer block size.

    Returns:
        Number of blocks over which to spread the transition.
    """
    total_samples = int(duration_seconds * sample_rate)
    return max(1, total_samples // block_size)


class CrossfadeManager:
    """
    Manages smooth transitions between audio states using constant-power crossfading.

    Tracks current and target audio states and provides interpolated parameters
    over a configurable transition duration.
    """

    def __init__(self, transition_duration: float = 30.0):
        """
        Initialize the crossfade manager.

        Args:
            transition_duration: Duration in seconds for state transitions.
        """
        self._transition_duration = transition_duration
        self._current_state = {
            "carrier_freq": 150.0,
            "beat_freq": 5.0,
            "target_amplitude": 0.0,
            "noise_color": "brown",
            "timbre_brightness": 0.5,
            "target_band": "theta",
        }
        self._target_state = None
        self._transition_start_time = None

    def set_target_state(self, state: dict) -> None:
        """
        Set a new target state to transition to.

        Args:
            state: Dictionary containing audio parameters (carrier_freq, beat_freq, etc.)
        """
        import time

        self._target_state = state.copy()
        self._transition_start_time = time.time()

    def get_current_params(self) -> dict:
        """
        Get the current interpolated audio parameters.

        If a transition is in progress, returns smoothly interpolated values
        between current and target states. Otherwise returns the current state.

        Returns:
            Dictionary of audio parameters ready for the synthesizer.
        """
        if self._target_state is None or self._transition_start_time is None:
            return self._current_state.copy()

        import time

        elapsed = time.time() - self._transition_start_time
        progress = min(1.0, elapsed / self._transition_duration)

        if progress >= 1.0:
            # Transition complete
            self._current_state = self._target_state.copy()
            self._target_state = None
            self._transition_start_time = None
            return self._current_state.copy()

        # Interpolate numeric parameters
        interpolated = {}
        for key in ["carrier_freq", "beat_freq", "target_amplitude", "timbre_brightness"]:
            if key in self._current_state and key in self._target_state:
                interpolated[key] = interpolate_parameter(
                    self._current_state[key], self._target_state[key], progress
                )
            elif key in self._current_state:
                interpolated[key] = self._current_state[key]

        # Non-interpolated fields (use target if available, else current)
        for key in ["noise_color", "target_band"]:
            if key in self._target_state:
                interpolated[key] = self._target_state[key]
            elif key in self._current_state:
                interpolated[key] = self._current_state[key]

        return interpolated

    def is_transitioning(self) -> bool:
        """Check if a transition is currently in progress."""
        return self._target_state is not None

    def get_transition_progress(self) -> float:
        """
        Get the current transition progress.

        Returns:
            Progress from 0.0 to 1.0, or 0.0 if no transition is active.
        """
        if not self.is_transitioning():
            return 0.0

        import time

        elapsed = time.time() - self._transition_start_time
        return min(1.0, elapsed / self._transition_duration)
