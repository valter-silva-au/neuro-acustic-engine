"""Tests for the ingestion module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from neuroacoustic.ingestion.calendar_parser import CalendarParser
from neuroacoustic.ingestion.llm_agent import LLMAgent, OllamaLLMAgent, create_agent
from neuroacoustic.ingestion.rss_parser import RSSParser


class TestCalendarParser:
    def setup_method(self):
        self.parser = CalendarParser()
        self.fixtures_dir = Path(__file__).parent / "fixtures"

    def test_parse_ics_file(self):
        """Test parsing a real ICS file with multiple events."""
        ics_path = self.fixtures_dir / "sample_calendar.ics"
        events = self.parser.parse_ics_file(str(ics_path))

        assert len(events) == 3

        # Check first event
        event1 = events[0]
        assert event1["title"] == "Deep work coding session"
        assert event1["description"] == "Focus time for implementing the neuroacoustic engine parser module"
        assert event1["location"] == "Home Office"
        assert event1["duration_minutes"] == 60
        assert "2026-03-06T14:00:00" in event1["start_time"]
        assert "2026-03-06T15:00:00" in event1["end_time"]

        # Check second event
        event2 = events[1]
        assert event2["title"] == "Team meeting"
        assert event2["duration_minutes"] == 90
        assert event2["location"] == "Conference Room A"

        # Check third event (no location)
        event3 = events[2]
        assert event3["title"] == "Learning session"
        assert event3["location"] == ""

    def test_parse_ics_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_ics_file("nonexistent.ics")

    def test_caldav_not_implemented(self):
        """CalDAV polling remains unimplemented."""
        with pytest.raises(NotImplementedError):
            self.parser.poll_caldav("https://example.com", "user", "pass")


class TestRSSParser:
    def setup_method(self):
        self.parser = RSSParser()

    def test_parse_feed_with_valid_data(self):
        """Test RSS parsing with mocked feedparser response."""
        mock_feed = Mock()
        mock_feed.bozo = False
        mock_feed.entries = [
            {
                "title": "Breaking News",
                "summary": "Important update on technology",
                "link": "https://example.com/news/1",
                "published_parsed": (2026, 3, 5, 10, 30, 0, 0, 0, 0),
            },
            {
                "title": "Tech Article",
                "description": "Deep dive into AI",
                "link": "https://example.com/news/2",
                "updated_parsed": (2026, 3, 4, 15, 0, 0, 0, 0, 0),
            },
        ]

        with patch("neuroacoustic.ingestion.rss_parser.feedparser.parse", return_value=mock_feed):
            entries = self.parser.parse_feed("https://example.com/feed.xml")

        assert len(entries) == 2

        # Check first entry
        entry1 = entries[0]
        assert entry1["title"] == "Breaking News"
        assert entry1["summary"] == "Important update on technology"
        assert entry1["source_url"] == "https://example.com/news/1"
        assert entry1["content_type"] == "rss"
        assert entry1["published"] == "2026-03-05T10:30:00"

        # Check second entry
        entry2 = entries[1]
        assert entry2["title"] == "Tech Article"
        assert entry2["summary"] == "Deep dive into AI"
        assert entry2["published"] == "2026-03-04T15:00:00"

    def test_parse_feed_invalid_url(self):
        """Test that ValueError is raised for invalid feeds."""
        mock_feed = Mock()
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("Invalid feed")
        mock_feed.entries = []

        with patch("neuroacoustic.ingestion.rss_parser.feedparser.parse", return_value=mock_feed):
            with pytest.raises(ValueError):
                self.parser.parse_feed("https://invalid.com/feed.xml")

    def test_parse_feed_with_missing_fields(self):
        """Test parsing entries with missing optional fields."""
        mock_feed = Mock()
        mock_feed.bozo = False
        mock_feed.entries = [
            {
                "title": "Minimal Entry",
                # No summary, link, or date
            }
        ]

        with patch("neuroacoustic.ingestion.rss_parser.feedparser.parse", return_value=mock_feed):
            entries = self.parser.parse_feed("https://example.com/feed.xml")

        assert len(entries) == 1
        assert entries[0]["title"] == "Minimal Entry"
        assert entries[0]["summary"] == ""
        assert entries[0]["source_url"] == ""
        assert entries[0]["published"] == ""


class TestLLMAgent:
    def setup_method(self):
        self.agent = LLMAgent()

    def test_focus_intent_extraction(self):
        result = json.loads(self.agent.extract_metadata("Deep work coding session"))
        assert result["intent"] == "focus"

    def test_learning_intent_extraction(self):
        result = json.loads(self.agent.extract_metadata("Study machine learning course"))
        assert result["intent"] == "learning"

    def test_alertness_intent_extraction(self):
        result = json.loads(self.agent.extract_metadata("Urgent deadline approaching"))
        assert result["intent"] == "alertness"

    def test_relaxation_default(self):
        result = json.loads(self.agent.extract_metadata("Just a regular day"))
        assert result["intent"] == "relaxation"

    def test_positive_sentiment(self):
        result = json.loads(self.agent.extract_metadata("Great breakthrough in research"))
        assert result["sentiment"] == "positive"

    def test_negative_sentiment(self):
        result = json.loads(self.agent.extract_metadata("Critical problem with the system"))
        assert result["sentiment"] == "negative"

    def test_high_urgency(self):
        result = json.loads(self.agent.extract_metadata("Important priority meeting now"))
        assert result["urgency"] == "high"

    def test_data_source_preserved(self):
        result = json.loads(self.agent.extract_metadata("Test event", data_source="calendar"))
        assert result["data_source"] == "calendar"

    def test_content_summary_truncated(self):
        long_text = "A" * 300
        result = json.loads(self.agent.extract_metadata(long_text))
        assert len(result["content_summary"]) == 200

    def test_output_is_valid_json(self):
        result = self.agent.extract_metadata("Some random text here")
        parsed = json.loads(result)
        assert "intent" in parsed
        assert "urgency" in parsed
        assert "sentiment" in parsed
        assert "data_source" in parsed
        assert "content_summary" in parsed


class TestOllamaLLMAgent:
    def setup_method(self):
        self.agent = OllamaLLMAgent()

    def test_ollama_agent_initialization(self):
        agent = OllamaLLMAgent(base_url="http://localhost:11434", model="qwen3.5-4b", timeout=5)
        assert agent.base_url == "http://localhost:11434"
        assert agent.model == "qwen3.5-4b"
        assert agent.timeout == 5
        assert agent.api_url == "http://localhost:11434/api/generate"

    @patch("neuroacoustic.ingestion.llm_agent.requests.post")
    def test_ollama_successful_extraction(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"intent": "focus", "urgency": "high", "sentiment": "positive", "data_source": "calendar", "content_summary": "Deep work session"}'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = json.loads(self.agent.extract_metadata("Deep work coding session", data_source="calendar"))

        assert result["intent"] == "focus"
        assert result["urgency"] == "high"
        assert result["sentiment"] == "positive"
        assert result["data_source"] == "calendar"
        assert "content_summary" in result

        # Verify the API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "qwen3.5-9b"
        assert call_args[1]["json"]["stream"] is False
        assert "Deep work coding session" in call_args[1]["json"]["prompt"]

    @patch("neuroacoustic.ingestion.llm_agent.requests.post")
    def test_ollama_json_with_extra_text(self, mock_post):
        # LLM sometimes adds extra text around the JSON
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": 'Here is the analysis:\n{"intent": "learning", "urgency": "medium", "sentiment": "neutral", "data_source": "rss", "content_summary": "Learning ML"}\nHope this helps!'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = json.loads(self.agent.extract_metadata("Study machine learning", data_source="rss"))

        assert result["intent"] == "learning"
        assert result["urgency"] == "medium"
        assert result["sentiment"] == "neutral"
        assert result["data_source"] == "rss"

    @patch("neuroacoustic.ingestion.llm_agent.requests.post")
    def test_ollama_invalid_field_values(self, mock_post):
        # LLM returns invalid values that should be normalized
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"intent": "invalid_intent", "urgency": "super_high", "sentiment": "very_positive", "data_source": "calendar", "content_summary": "Test"}'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = json.loads(self.agent.extract_metadata("Test event"))

        # Invalid values should be normalized to defaults
        assert result["intent"] == "relaxation"
        assert result["urgency"] == "medium"
        assert result["sentiment"] == "neutral"

    @patch("neuroacoustic.ingestion.llm_agent.requests.post")
    def test_ollama_connection_error_fallback(self, mock_post):
        # Simulate connection error
        mock_post.side_effect = Exception("Connection refused")

        result = json.loads(self.agent.extract_metadata("Test event", data_source="calendar"))

        # Should fall back to default response
        assert result["intent"] == "relaxation"
        assert result["urgency"] == "medium"
        assert result["sentiment"] == "neutral"
        assert result["data_source"] == "calendar"
        assert result["content_summary"] == "Test event"

    @patch("neuroacoustic.ingestion.llm_agent.requests.post")
    def test_ollama_timeout_fallback(self, mock_post):
        # Simulate timeout
        import requests
        mock_post.side_effect = requests.Timeout("Request timed out")

        result = json.loads(self.agent.extract_metadata("Test event", data_source="rss"))

        # Should fall back to default response
        assert result["intent"] == "relaxation"
        assert result["urgency"] == "medium"
        assert result["sentiment"] == "neutral"
        assert result["data_source"] == "rss"

    @patch("neuroacoustic.ingestion.llm_agent.requests.post")
    def test_ollama_invalid_json_response_fallback(self, mock_post):
        # LLM returns invalid JSON
        mock_response = Mock()
        mock_response.json.return_value = {"response": "This is not valid JSON at all"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = json.loads(self.agent.extract_metadata("Test event", data_source="calendar"))

        # Should fall back to default response
        assert result["intent"] == "relaxation"
        assert result["urgency"] == "medium"
        assert result["sentiment"] == "neutral"

    def test_validate_field_with_valid_value(self):
        result = OllamaLLMAgent._validate_field("focus", ["focus", "relaxation", "alertness", "learning"], "relaxation")
        assert result == "focus"

    def test_validate_field_with_invalid_value(self):
        result = OllamaLLMAgent._validate_field("invalid", ["focus", "relaxation", "alertness", "learning"], "relaxation")
        assert result == "relaxation"

    def test_validate_field_with_none(self):
        result = OllamaLLMAgent._validate_field(None, ["focus", "relaxation", "alertness", "learning"], "relaxation")
        assert result == "relaxation"

    def test_validate_field_case_insensitive(self):
        result = OllamaLLMAgent._validate_field("FOCUS", ["focus", "relaxation", "alertness", "learning"], "relaxation")
        assert result == "focus"

    @pytest.mark.integration
    @patch("neuroacoustic.ingestion.llm_agent.requests.post")
    def test_ollama_real_integration(self, mock_post):
        """
        Integration test for Ollama. This is marked as an integration test
        and can be run separately when Ollama is available.
        """
        # This test is mocked by default, but can be configured to hit real Ollama
        # by removing the patch decorator and marking with pytest.mark.integration
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"intent": "focus", "urgency": "high", "sentiment": "positive", "data_source": "test", "content_summary": "Integration test"}'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        agent = OllamaLLMAgent()
        result = json.loads(agent.extract_metadata("Deep work coding session on AI project", data_source="test"))

        assert result["intent"] in ["focus", "relaxation", "alertness", "learning"]
        assert result["urgency"] in ["low", "medium", "high", "critical"]
        assert result["sentiment"] in ["positive", "neutral", "negative"]
        assert result["data_source"] == "test"


class TestAgentFactory:
    def test_create_agent_ollama(self):
        agent = create_agent(backend="ollama")
        assert isinstance(agent, OllamaLLMAgent)

    def test_create_agent_mock(self):
        agent = create_agent(backend="mock")
        assert isinstance(agent, LLMAgent)

    def test_create_agent_default_is_ollama(self):
        agent = create_agent()
        assert isinstance(agent, OllamaLLMAgent)

    def test_create_agent_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_agent(backend="invalid")
