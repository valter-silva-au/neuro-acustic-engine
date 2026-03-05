"""Tests for the orchestration module."""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from neuroacoustic.orchestration.crossfader import (
    CrossfadeManager,
    constant_power_gains,
    interpolate_parameter,
)
from neuroacoustic.orchestration.execution_loop import (
    CalendarSource,
    ExecutionLoop,
    FileWatcherSource,
    RSSSource,
)


class TestConstantPowerGains:
    def test_start_position(self):
        """At t=-1, only outgoing signal should be audible."""
        gain_out, gain_in = constant_power_gains(-1.0)
        assert gain_out == pytest.approx(1.0, abs=0.01)
        assert gain_in == pytest.approx(0.0, abs=0.01)

    def test_end_position(self):
        """At t=+1, only incoming signal should be audible."""
        gain_out, gain_in = constant_power_gains(1.0)
        assert gain_out == pytest.approx(0.0, abs=0.01)
        assert gain_in == pytest.approx(1.0, abs=0.01)

    def test_midpoint_power(self):
        """At t=0, both signals at sqrt(0.5) for constant power."""
        gain_out, gain_in = constant_power_gains(0.0)
        expected = (0.5**0.5)  # sqrt(0.5) ≈ 0.707
        assert gain_out == pytest.approx(expected, abs=0.01)
        assert gain_in == pytest.approx(expected, abs=0.01)

        # Verify power sum = 1.0 (gain^2 + gain^2 = 1.0)
        power_sum = gain_out**2 + gain_in**2
        assert power_sum == pytest.approx(1.0, abs=0.01)

    def test_clamping(self):
        """Values outside [-1, 1] should be clamped."""
        gain_out, gain_in = constant_power_gains(-5.0)
        assert gain_out == pytest.approx(1.0, abs=0.01)
        assert gain_in == pytest.approx(0.0, abs=0.01)

        gain_out, gain_in = constant_power_gains(5.0)
        assert gain_out == pytest.approx(0.0, abs=0.01)
        assert gain_in == pytest.approx(1.0, abs=0.01)


class TestInterpolateParameter:
    def test_start_value(self):
        result = interpolate_parameter(100.0, 200.0, 0.0)
        assert result == 100.0

    def test_end_value(self):
        result = interpolate_parameter(100.0, 200.0, 1.0)
        assert result == 200.0

    def test_midpoint_value(self):
        result = interpolate_parameter(100.0, 200.0, 0.5)
        assert result == pytest.approx(150.0)

    def test_clamping(self):
        result = interpolate_parameter(100.0, 200.0, -0.5)
        assert result == 100.0

        result = interpolate_parameter(100.0, 200.0, 1.5)
        assert result == 200.0


class TestCrossfadeManager:
    def test_initialization(self):
        manager = CrossfadeManager(transition_duration=10.0)
        params = manager.get_current_params()

        assert "carrier_freq" in params
        assert "beat_freq" in params
        assert "target_amplitude" in params
        assert not manager.is_transitioning()

    def test_set_target_state(self):
        manager = CrossfadeManager(transition_duration=1.0)

        target = {
            "carrier_freq": 200.0,
            "beat_freq": 10.0,
            "target_amplitude": 0.5,
            "noise_color": "pink",
            "timbre_brightness": 0.7,
            "target_band": "beta",
        }

        manager.set_target_state(target)
        assert manager.is_transitioning()
        assert manager.get_transition_progress() >= 0.0

    def test_transition_completion(self):
        manager = CrossfadeManager(transition_duration=0.1)

        target = {
            "carrier_freq": 200.0,
            "beat_freq": 10.0,
            "target_amplitude": 0.5,
            "noise_color": "pink",
        }

        manager.set_target_state(target)
        assert manager.is_transitioning()

        # Wait for transition to complete
        time.sleep(0.15)

        params = manager.get_current_params()
        assert not manager.is_transitioning()
        assert params["carrier_freq"] == 200.0
        assert params["beat_freq"] == 10.0
        assert params["target_amplitude"] == 0.5

    def test_interpolation_progress(self):
        manager = CrossfadeManager(transition_duration=1.0)

        start_state = manager.get_current_params()
        start_freq = start_state["carrier_freq"]

        target = {
            "carrier_freq": 300.0,
            "beat_freq": 15.0,
            "target_amplitude": 0.8,
            "noise_color": "white",
        }

        manager.set_target_state(target)

        # Check at various points during transition
        time.sleep(0.2)
        params_early = manager.get_current_params()

        time.sleep(0.3)
        params_mid = manager.get_current_params()

        # Frequency should be progressing toward target
        assert start_freq < params_early["carrier_freq"] < 300.0
        assert params_early["carrier_freq"] < params_mid["carrier_freq"] <= 300.0


class TestFileWatcherSource:
    def test_initialization(self, tmp_path):
        watcher = FileWatcherSource(str(tmp_path / "watch"))
        assert (tmp_path / "watch").exists()

    def test_poll_empty_directory(self, tmp_path):
        watcher = FileWatcherSource(str(tmp_path / "watch"))
        events = watcher.poll()
        assert events == []

    def test_poll_new_json_file(self, tmp_path):
        watch_dir = tmp_path / "watch"
        watcher = FileWatcherSource(str(watch_dir))

        # Create a JSON file
        test_event = {
            "intent": "focus",
            "urgency": "high",
            "sentiment": "positive",
            "data_source": "test",
            "content_summary": "Test event",
        }

        json_file = watch_dir / "event1.json"
        with open(json_file, "w") as f:
            json.dump(test_event, f)

        # Poll should return the event
        events = watcher.poll()
        assert len(events) == 1
        assert events[0]["intent"] == "focus"
        assert events[0]["urgency"] == "high"

        # Second poll should return nothing (file already processed)
        events = watcher.poll()
        assert events == []

    def test_poll_multiple_files(self, tmp_path):
        watch_dir = tmp_path / "watch"
        watcher = FileWatcherSource(str(watch_dir))

        # Create multiple JSON files
        for i in range(3):
            event = {"id": i, "data": f"event_{i}"}
            json_file = watch_dir / f"event{i}.json"
            with open(json_file, "w") as f:
                json.dump(event, f)

        # Poll should return all events
        events = watcher.poll()
        assert len(events) == 3
        assert {e["id"] for e in events} == {0, 1, 2}

    def test_poll_invalid_json(self, tmp_path):
        watch_dir = tmp_path / "watch"
        watcher = FileWatcherSource(str(watch_dir))

        # Create invalid JSON file
        invalid_file = watch_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json {")

        # Poll should handle error gracefully
        events = watcher.poll()
        assert events == []

    def test_poll_ignores_non_json_files(self, tmp_path):
        watch_dir = tmp_path / "watch"
        watcher = FileWatcherSource(str(watch_dir))

        # Create non-JSON files
        (watch_dir / "event.txt").write_text("some text")
        (watch_dir / "data.xml").write_text("<data></data>")

        # Poll should ignore them
        events = watcher.poll()
        assert events == []


class TestCalendarSource:
    def test_poll_returns_events(self):
        mock_parser = Mock()
        mock_parser.parse_ics_file.return_value = [
            {
                "title": "Meeting",
                "start_time": "2026-03-06T10:00:00",
                "end_time": "2026-03-06T11:00:00",
                "duration_minutes": 60,
                "description": "Team sync",
                "location": "Office",
            }
        ]

        source = CalendarSource(mock_parser, "test.ics")
        events = source.poll()

        assert len(events) == 1
        assert events[0]["title"] == "Meeting"
        mock_parser.parse_ics_file.assert_called_once_with("test.ics")

    def test_poll_returns_only_new_events(self):
        event1 = {"title": "Event 1", "start_time": "2026-03-06T10:00:00"}
        event2 = {"title": "Event 2", "start_time": "2026-03-06T14:00:00"}

        mock_parser = Mock()
        mock_parser.parse_ics_file.return_value = [event1]

        source = CalendarSource(mock_parser, "test.ics")

        # First poll returns event1
        events = source.poll()
        assert len(events) == 1
        assert events[0]["title"] == "Event 1"

        # Second poll with same events returns nothing new
        events = source.poll()
        assert len(events) == 0

        # Third poll with new event returns event2
        mock_parser.parse_ics_file.return_value = [event1, event2]
        events = source.poll()
        assert len(events) == 1
        assert events[0]["title"] == "Event 2"

    def test_poll_handles_parser_error(self):
        mock_parser = Mock()
        mock_parser.parse_ics_file.side_effect = Exception("Parse error")

        source = CalendarSource(mock_parser, "test.ics")
        events = source.poll()

        assert events == []


class TestRSSSource:
    def test_poll_returns_entries(self):
        mock_parser = Mock()
        mock_parser.parse_feed.return_value = [
            {
                "title": "Breaking News",
                "summary": "Important update",
                "source_url": "https://example.com/news/1",
                "published": "2026-03-05T10:30:00",
                "content_type": "rss",
            }
        ]

        source = RSSSource(mock_parser, "https://example.com/feed.xml")
        entries = source.poll()

        assert len(entries) >= 1
        assert entries[0]["title"] == "Breaking News"
        mock_parser.parse_feed.assert_called_once_with("https://example.com/feed.xml")

    def test_poll_rotates_through_entries(self):
        """RSS source rotates through items on each poll, not just new ones."""
        entries_data = [
            {"title": f"Article {i}", "summary": f"Summary {i}"}
            for i in range(10)
        ]

        mock_parser = Mock()
        mock_parser.parse_feed.return_value = entries_data

        source = RSSSource(mock_parser, "https://example.com/feed.xml")

        # First poll returns first window (items 0-4)
        batch1 = source.poll()
        assert len(batch1) == 5
        assert batch1[0]["title"] == "Article 0"

        # Second poll returns next window (items 5-9)
        batch2 = source.poll()
        assert len(batch2) == 5
        assert batch2[0]["title"] == "Article 5"

        # Third poll wraps around back to start
        batch3 = source.poll()
        assert len(batch3) == 5
        assert batch3[0]["title"] == "Article 0"

    def test_poll_handles_parser_error(self):
        mock_parser = Mock()
        mock_parser.parse_feed.side_effect = Exception("Network error")

        source = RSSSource(mock_parser, "https://example.com/feed.xml")
        entries = source.poll()

        assert entries == []


class TestExecutionLoop:
    def test_initialization(self):
        mock_synth = Mock()
        mock_mapper = Mock()

        loop = ExecutionLoop(mock_synth, mock_mapper)

        assert loop._synthesizer == mock_synth
        assert loop._mapper == mock_mapper
        assert loop._running is False
        assert len(loop._data_sources) == 0

    def test_add_data_source(self):
        mock_synth = Mock()
        mock_mapper = Mock()
        mock_source = Mock()

        loop = ExecutionLoop(mock_synth, mock_mapper)
        loop.add_data_source(mock_source)

        assert len(loop._data_sources) == 1
        assert loop._data_sources[0] == mock_source

    def test_start_stop(self):
        mock_synth = Mock()
        mock_mapper = Mock()

        loop = ExecutionLoop(mock_synth, mock_mapper)
        loop.start()

        assert loop._running is True
        mock_synth.start.assert_called_once()

        loop.stop()
        assert loop._running is False
        mock_synth.stop.assert_called_once()

    def test_create_event_text_calendar_event(self):
        mock_synth = Mock()
        mock_mapper = Mock()

        loop = ExecutionLoop(mock_synth, mock_mapper)

        event = {
            "title": "Deep Work",
            "description": "Focus session",
            "location": "Office",
            "start_time": "2026-03-06T10:00:00",
        }

        text = loop._create_event_text(event)
        assert "Deep Work" in text
        assert "Focus session" in text
        assert "Office" in text

    def test_create_event_text_rss_entry(self):
        mock_synth = Mock()
        mock_mapper = Mock()

        loop = ExecutionLoop(mock_synth, mock_mapper)

        event = {
            "title": "Tech News",
            "summary": "AI breakthrough",
            "source_url": "https://example.com",
        }

        text = loop._create_event_text(event)
        assert "Tech News" in text
        assert "AI breakthrough" in text

    def test_orchestration_integration(self):
        """Test the full orchestration pipeline with mocked components."""
        mock_synth = Mock()
        mock_mapper = Mock()
        mock_mapper.translate_payload.return_value = {
            "carrier_freq": 250.0,
            "beat_freq": 20.0,
            "target_amplitude": 0.5,
            "noise_color": "pink",
            "timbre_brightness": 0.6,
            "target_band": "beta",
            "original_metadata": {
                "intent": "focus",
                "urgency": "high",
                "sentiment": "positive",
            },
        }

        mock_llm_agent = Mock()
        mock_llm_agent.extract_metadata.return_value = json.dumps(
            {
                "intent": "focus",
                "urgency": "high",
                "sentiment": "positive",
                "data_source": "test",
                "content_summary": "Test event",
            }
        )

        # Create a mock data source
        mock_source = Mock()
        mock_source.poll.return_value = [
            {
                "title": "Deep Work Session",
                "description": "Focus time",
                "start_time": "2026-03-06T10:00:00",
            }
        ]

        loop = ExecutionLoop(
            mock_synth,
            mock_mapper,
            llm_agent=mock_llm_agent,
            poll_interval=0.1,
            crossfade_duration=0.5,
        )
        loop.add_data_source(mock_source)

        # Start loop and let it process
        loop.start()
        time.sleep(0.3)  # Let context thread poll and queue event

        # Verify LLM agent was called
        assert mock_llm_agent.extract_metadata.call_count >= 1

        # Verify mapper was called
        assert mock_mapper.translate_payload.call_count >= 1

        # Verify synthesizer was updated
        assert mock_synth.update_state.call_count >= 1

        loop.stop()
