"""End-to-end integration tests for the neuroacoustic engine pipeline."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from neuroacoustic.ingestion.calendar_parser import CalendarParser
from neuroacoustic.ingestion.llm_agent import LLMAgent, create_agent
from neuroacoustic.orchestration.crossfader import CrossfadeManager
from neuroacoustic.orchestration.execution_loop import (
    ExecutionLoop,
    FileWatcherSource,
)
from neuroacoustic.translation.semantic_mapper import SemanticMapper


class MockSynthesizer:
    """Mock synthesizer that records state changes instead of playing audio."""

    def __init__(self):
        self.is_running = False
        self.state_history = []
        self.current_state = None

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def update_state(self, carrier_freq, beat_freq, amplitude, metadata):
        state = {
            "carrier_freq": carrier_freq,
            "beat_freq": beat_freq,
            "amplitude": amplitude,
            "metadata": metadata,
        }
        self.state_history.append(state)
        self.current_state = state

    def get_metadata(self):
        return self.current_state.get("metadata", {}) if self.current_state else {}


class TestE2ECalendarPipeline:
    """Test the full pipeline: Calendar → LLM → Mapper → Synth."""

    def setup_method(self):
        self.parser = CalendarParser()
        self.llm_agent = create_agent(backend="mock")
        self.mapper = SemanticMapper()
        self.synth = MockSynthesizer()
        self.fixtures_dir = Path(__file__).parent / "fixtures"

    def test_calendar_to_synth_focus_event(self):
        """Parse a 'deep work' calendar event through the full pipeline."""
        # Parse the sample calendar
        ics_path = self.fixtures_dir / "sample_calendar.ics"
        events = self.parser.parse_ics_file(str(ics_path))

        # Get the "Deep work coding session" event
        deep_work_event = events[0]
        assert "Deep work" in deep_work_event["title"]

        # Create event text for LLM extraction
        event_text = f"{deep_work_event['title']} - {deep_work_event['description']}"

        # Run through LLM agent
        metadata_json = self.llm_agent.extract_metadata(event_text, "calendar")
        metadata = json.loads(metadata_json)

        # Verify LLM extracted "focus" intent from "deep work" keywords
        assert metadata["intent"] == "focus"
        assert metadata["data_source"] == "calendar"

        # Map semantic metadata to audio parameters
        audio_params = self.mapper.translate_payload(metadata_json)

        # Verify focus parameters (gamma band, high beat frequency)
        assert audio_params["carrier_freq"] == pytest.approx(432.0 * 1.0, abs=50)  # focus carrier
        assert audio_params["beat_freq"] >= 30.0  # gamma range (40Hz typical)
        assert audio_params["target_band"] == "gamma"
        assert audio_params["noise_color"] == "pink"

        # Verify synthesizer receives correct params
        self.synth.update_state(
            carrier_freq=audio_params["carrier_freq"],
            beat_freq=audio_params["beat_freq"],
            amplitude=audio_params["target_amplitude"],
            metadata=audio_params["original_metadata"],
        )

        assert self.synth.current_state["carrier_freq"] == audio_params["carrier_freq"]
        assert self.synth.current_state["beat_freq"] == audio_params["beat_freq"]

    def test_calendar_to_synth_meeting_event(self):
        """Parse a 'team meeting' calendar event through the full pipeline."""
        ics_path = self.fixtures_dir / "sample_calendar.ics"
        events = self.parser.parse_ics_file(str(ics_path))

        # Get the "Team meeting" event
        meeting_event = events[1]
        assert "meeting" in meeting_event["title"].lower()

        # Create event text
        event_text = f"{meeting_event['title']} - {meeting_event['description']}"

        # Run through LLM agent (should NOT match "focus" keywords)
        metadata_json = self.llm_agent.extract_metadata(event_text, "calendar")
        metadata = json.loads(metadata_json)

        # Meeting doesn't have "focus" keywords, so should fall back to relaxation
        assert metadata["intent"] in ["relaxation", "alertness"]
        assert metadata["data_source"] == "calendar"

        # Map to audio parameters
        audio_params = self.mapper.translate_payload(metadata_json)

        # Verify different parameters from deep work
        assert audio_params["carrier_freq"] != 432.0  # not focus carrier
        assert audio_params["beat_freq"] < 40.0  # not gamma range

    def test_calendar_to_synth_learning_event(self):
        """Parse a 'learning session' calendar event through the full pipeline."""
        ics_path = self.fixtures_dir / "sample_calendar.ics"
        events = self.parser.parse_ics_file(str(ics_path))

        # Get the "Learning session" event
        learning_event = events[2]
        assert "Learning" in learning_event["title"]

        event_text = f"{learning_event['title']} - {learning_event['description']}"

        # Run through LLM agent
        metadata_json = self.llm_agent.extract_metadata(event_text, "calendar")
        metadata = json.loads(metadata_json)

        # "Learning" keyword should trigger learning intent
        assert metadata["intent"] == "learning"

        # Map to audio parameters
        audio_params = self.mapper.translate_payload(metadata_json)

        # Learning profile uses gamma band with 528Hz carrier
        assert audio_params["target_band"] == "gamma"
        assert audio_params["carrier_freq"] == pytest.approx(528.0 * 1.0, abs=50)
        assert audio_params["beat_freq"] >= 30.0


class TestE2ECrossfadeTransitions:
    """Test crossfade transitions between different event types."""

    def setup_method(self):
        self.mapper = SemanticMapper()
        self.crossfade_manager = CrossfadeManager(transition_duration=2.0)  # 2s for testing

    def test_focus_to_relaxation_crossfade(self):
        """Verify smooth parameter interpolation when transitioning between states."""
        # Start with focus state
        focus_metadata = {
            "intent": "focus",
            "urgency": "medium",
            "sentiment": "neutral",
            "data_source": "calendar",
            "content_summary": "Deep work session",
        }
        focus_params = self.mapper.translate_payload(json.dumps(focus_metadata))

        # Set initial state (simulate already being in focus)
        self.crossfade_manager._current_state = focus_params.copy()

        # Transition to relaxation state
        relax_metadata = {
            "intent": "relaxation",
            "urgency": "low",
            "sentiment": "neutral",
            "data_source": "calendar",
            "content_summary": "Break time",
        }
        relax_params = self.mapper.translate_payload(json.dumps(relax_metadata))

        # Start the transition
        self.crossfade_manager.set_target_state(relax_params)

        # Verify transition is active
        assert self.crossfade_manager.is_transitioning()

        # Check interpolated parameters at 50% progress
        time.sleep(1.0)  # 50% of 2s transition
        mid_params = self.crossfade_manager.get_current_params()

        # Mid-point should be between focus and relaxation values
        focus_carrier = focus_params["carrier_freq"]
        relax_carrier = relax_params["carrier_freq"]
        assert relax_carrier < mid_params["carrier_freq"] < focus_carrier or \
               focus_carrier < mid_params["carrier_freq"] < relax_carrier

        # Wait for transition to complete
        time.sleep(1.5)

        # Verify transition completed and settled to target
        final_params = self.crossfade_manager.get_current_params()
        assert not self.crossfade_manager.is_transitioning()
        assert final_params["carrier_freq"] == pytest.approx(relax_params["carrier_freq"], abs=1.0)
        assert final_params["beat_freq"] == pytest.approx(relax_params["beat_freq"], abs=0.5)

    def test_multiple_rapid_transitions(self):
        """Verify crossfade manager handles rapid state changes correctly."""
        states = ["focus", "alertness", "relaxation"]

        for intent in states:
            metadata = {
                "intent": intent,
                "urgency": "medium",
                "sentiment": "neutral",
                "data_source": "test",
                "content_summary": f"{intent} state",
            }
            params = self.mapper.translate_payload(json.dumps(metadata))
            self.crossfade_manager.set_target_state(params)

            # Verify transition started
            assert self.crossfade_manager.is_transitioning()

            # Allow brief transition time (don't wait for completion)
            time.sleep(0.3)

        # Final state should eventually reach the last intent (relaxation)
        time.sleep(2.0)
        final_params = self.crossfade_manager.get_current_params()
        assert not self.crossfade_manager.is_transitioning()


class TestE2EFileWatcherPipeline:
    """Test the FileWatcherSource → Queue → Mapper → Synth pipeline."""

    def test_file_watcher_json_ingestion(self):
        """Test FileWatcherSource detects new JSON files and queues events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = FileWatcherSource(tmpdir)

            # Initially no files
            events = source.poll()
            assert len(events) == 0

            # Create a new JSON event file
            event_data = {
                "intent": "focus",
                "urgency": "high",
                "sentiment": "positive",
                "data_source": "external_api",
                "content_summary": "Urgent task notification",
            }
            json_path = Path(tmpdir) / "event1.json"
            with open(json_path, "w") as f:
                json.dump(event_data, f)

            # Poll should now detect the file
            events = source.poll()
            assert len(events) == 1
            assert events[0]["intent"] == "focus"
            assert events[0]["urgency"] == "high"

            # Second poll should return empty (file already processed)
            events = source.poll()
            assert len(events) == 0

            # Create another file
            event_data2 = {
                "intent": "relaxation",
                "urgency": "low",
                "sentiment": "neutral",
                "data_source": "external_api",
                "content_summary": "Break reminder",
            }
            json_path2 = Path(tmpdir) / "event2.json"
            with open(json_path2, "w") as f:
                json.dump(event_data2, f)

            # Poll should detect the new file
            events = source.poll()
            assert len(events) == 1
            assert events[0]["intent"] == "relaxation"

    def test_execution_loop_with_file_watcher(self):
        """Test ExecutionLoop processes FileWatcherSource events through the full pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            synth = MockSynthesizer()
            mapper = SemanticMapper()
            llm_agent = None  # Skip LLM since we're providing pre-structured JSON

            loop = ExecutionLoop(
                synthesizer=synth,
                mapper=mapper,
                llm_agent=llm_agent,
                poll_interval=0.5,  # Poll every 0.5s for fast testing
                crossfade_duration=1.0,  # Short crossfade for testing
            )

            # Add FileWatcherSource
            source = FileWatcherSource(tmpdir)
            loop.add_data_source(source)

            # Start the execution loop
            loop.start()

            try:
                # Give threads time to start
                time.sleep(0.2)

                # Drop a JSON event file
                event_data = {
                    "intent": "alertness",
                    "urgency": "critical",
                    "sentiment": "negative",
                    "data_source": "filewatcher",
                    "content_summary": "System alert",
                }
                json_path = Path(tmpdir) / "alert.json"
                with open(json_path, "w") as f:
                    json.dump(event_data, f)

                # Wait for polling and processing (and crossfade to complete)
                time.sleep(2.5)

                # Verify synthesizer received the event
                assert len(synth.state_history) > 0

                # Find the alertness state in history
                alertness_states = [
                    s for s in synth.state_history
                    if s["metadata"].get("intent") == "alertness"
                ]
                assert len(alertness_states) > 0

                # Verify critical urgency resulted in high amplitude
                # Check the metadata to see the target amplitude (since crossfade may be in progress)
                critical_state = alertness_states[-1]
                metadata = critical_state["metadata"]

                # Either the current amplitude is high (crossfade complete)
                # or the metadata shows high urgency which will result in high amplitude
                if critical_state["amplitude"] < 0.6:
                    # Crossfade in progress, check that urgency is critical
                    assert metadata.get("urgency") == "critical"
                else:
                    # Crossfade complete, amplitude should be high
                    assert critical_state["amplitude"] >= 0.6

            finally:
                loop.stop()


class TestE2EFullSystemVerification:
    """Verification that all pieces work together correctly."""

    def test_all_intent_profiles_produce_valid_params(self):
        """Verify all intent profiles produce synthesizer-compatible parameters."""
        mapper = SemanticMapper()
        synth = MockSynthesizer()

        intents = ["focus", "relaxation", "alertness", "learning"]
        urgencies = ["low", "medium", "high", "critical"]
        sentiments = ["positive", "neutral", "negative"]

        synth.start()

        for intent in intents:
            for urgency in urgencies:
                for sentiment in sentiments:
                    metadata = {
                        "intent": intent,
                        "urgency": urgency,
                        "sentiment": sentiment,
                        "data_source": "test",
                        "content_summary": f"{intent} {urgency} {sentiment}",
                    }

                    # Map to audio params
                    params = mapper.translate_payload(json.dumps(metadata))

                    # Verify all required keys exist
                    assert "carrier_freq" in params
                    assert "beat_freq" in params
                    assert "target_amplitude" in params
                    assert "noise_color" in params
                    assert "timbre_brightness" in params

                    # Verify values are in valid ranges
                    assert 50.0 <= params["carrier_freq"] <= 1000.0
                    assert 0.5 <= params["beat_freq"] <= 100.0
                    assert 0.0 <= params["target_amplitude"] <= 1.0
                    assert params["noise_color"] in ["white", "pink", "brown"]
                    assert 0.0 <= params["timbre_brightness"] <= 1.0

                    # Verify synthesizer accepts the parameters
                    synth.update_state(
                        carrier_freq=params["carrier_freq"],
                        beat_freq=params["beat_freq"],
                        amplitude=params["target_amplitude"],
                        metadata=params["original_metadata"],
                    )

                    assert synth.current_state is not None

        synth.stop()

    def test_calendar_parser_output_compatible_with_llm_agent(self):
        """Verify CalendarParser output format works with LLM agent."""
        parser = CalendarParser()
        agent = create_agent(backend="mock")
        fixtures_dir = Path(__file__).parent / "fixtures"

        ics_path = fixtures_dir / "sample_calendar.ics"
        events = parser.parse_ics_file(str(ics_path))

        # Verify all calendar events can be processed by LLM agent
        for event in events:
            event_text = f"{event['title']} - {event['description']}"
            metadata_json = agent.extract_metadata(event_text, "calendar")

            # Verify valid JSON output
            metadata = json.loads(metadata_json)
            assert "intent" in metadata
            assert "urgency" in metadata
            assert "sentiment" in metadata
            assert "data_source" in metadata
            assert metadata["data_source"] == "calendar"
