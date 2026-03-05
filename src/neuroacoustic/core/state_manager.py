"""
Thread-safe state management for the NeuroAcoustic Engine.

Tracks the current cognitive audio state and provides safe access
from the orchestration, translation, and synthesis threads.
"""

import threading
from copy import deepcopy


class StateManager:
    """Thread-safe container for the engine's current audio state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "carrier_freq": 150.0,
            "beat_freq": 5.0,
            "amplitude": 0.0,
            "noise_color": "brown",
            "noise_level": 0.0,
            "modulation_rate": 0.1,
            "metadata": {
                "intent": "idle",
                "data_source": "system_init",
                "urgency": "none",
                "frequencies": [],
            },
        }

    def get_state(self) -> dict:
        """Return a deep copy of the current state."""
        with self._lock:
            return deepcopy(self._state)

    def update_state(self, **kwargs) -> None:
        """Update one or more state fields atomically."""
        with self._lock:
            for key, value in kwargs.items():
                if key == "metadata":
                    self._state["metadata"].update(value)
                elif key in self._state:
                    self._state[key] = value

    def get_metadata(self) -> dict:
        """Return a copy of the current metadata."""
        with self._lock:
            return deepcopy(self._state["metadata"])
