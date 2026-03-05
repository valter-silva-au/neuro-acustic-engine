"""
Main execution loop for the NeuroAcoustic Engine.

Coordinates three threads:
- Context Agent Thread: polls data sources, runs local LLM inference
- Orchestration Thread: receives semantic payloads via queue, drives mapper
- Audio Thread: managed by pyo's C-level server (runs independently)

This module provides the full implementation with LLM extraction,
semantic mapping, and crossfaded audio synthesis.
"""

import json
import queue
import threading
import time
from pathlib import Path
from typing import Protocol

from neuroacoustic.orchestration.crossfader import CrossfadeManager


class DataSource(Protocol):
    """Interface for data source polling."""

    def poll(self) -> list[dict]:
        """Return new events since last poll."""
        ...


class ExecutionLoop:
    """
    Main orchestration loop coordinating data ingestion and audio synthesis.

    Architecture:
        [DataSource] --poll--> [Context Thread] --queue--> [Orchestration Thread]
                                                                    |
                                                             [SemanticMapper]
                                                                    |
                                                        [CognitiveAudioSynthesizer]
    """

    def __init__(
        self,
        synthesizer,
        mapper,
        llm_agent=None,
        poll_interval: float = 60.0,
        crossfade_duration: float = 30.0,
    ):
        self._synthesizer = synthesizer
        self._mapper = mapper
        self._llm_agent = llm_agent
        self._poll_interval = poll_interval
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._data_sources: list[DataSource] = []
        self._crossfade_manager = CrossfadeManager(transition_duration=crossfade_duration)

    def add_data_source(self, source: DataSource) -> None:
        """Register a data source for periodic polling."""
        self._data_sources.append(source)

    def start(self) -> None:
        """Start the engine: boot audio server and begin polling loop."""
        self._running = True
        self._synthesizer.start()

        self._context_thread = threading.Thread(
            target=self._context_loop, daemon=True
        )
        self._orchestration_thread = threading.Thread(
            target=self._orchestration_loop, daemon=True
        )

        self._context_thread.start()
        self._orchestration_thread.start()

    def stop(self) -> None:
        """Gracefully shut down all threads and the audio server."""
        self._running = False
        self._synthesizer.stop()

    def _context_loop(self) -> None:
        """Background thread: polls data sources, aggregates into one dominant state per cycle."""
        while self._running:
            all_metadata = []
            for source in self._data_sources:
                try:
                    events = source.poll()
                    if not events:
                        continue

                    # Sample up to 5 events per source to avoid LLM overload
                    sampled = events[:5]
                    for event in sampled:
                        if self._llm_agent:
                            # For RSS entries, fetch article content for richer analysis
                            if event.get("content_type") == "rss" and event.get("source_url"):
                                article = self._fetch_article(event)
                                if article:
                                    event = {**event, "article_content": article}

                            event_text = self._create_event_text(event)
                            data_source = event.get("content_type", event.get("data_source", "unknown"))
                            metadata_json = self._llm_agent.extract_metadata(event_text, data_source)
                            metadata = json.loads(metadata_json)
                            # Attach source info for display
                            metadata["_source_title"] = event.get("title", event_text[:80])
                            metadata["_source_type"] = data_source
                            metadata["_source_url"] = event.get("source_url", "")
                            metadata["_has_article"] = bool(event.get("article_content"))
                            all_metadata.append(metadata)
                        else:
                            all_metadata.append(event)
                except Exception as e:
                    print(f"Error polling data source: {e}")

            # Aggregate: pick the dominant intent by frequency, highest urgency wins ties
            if all_metadata:
                dominant = self._aggregate_metadata(all_metadata)
                self._event_queue.put(dominant)

            time.sleep(self._poll_interval)

    @staticmethod
    def _aggregate_metadata(items: list[dict]) -> dict:
        """Aggregate multiple metadata dicts into one dominant cognitive state."""
        URGENCY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        # Count intents
        intent_counts = {}
        for item in items:
            intent = item.get("intent", "relaxation")
            if intent not in intent_counts:
                intent_counts[intent] = {"count": 0, "max_urgency": "low", "best_item": item}
            intent_counts[intent]["count"] += 1
            item_urgency = item.get("urgency", "low")
            if URGENCY_RANK.get(item_urgency, 0) > URGENCY_RANK.get(intent_counts[intent]["max_urgency"], 0):
                intent_counts[intent]["max_urgency"] = item_urgency
                intent_counts[intent]["best_item"] = item

        # Collect all source titles for display
        all_sources = [item.get("_source_title", "") for item in items if item.get("_source_title")]
        intent_summary = {k: v["count"] for k, v in intent_counts.items()}

        # Alertness/critical always wins if present
        for intent in ["alertness"]:
            if intent in intent_counts and URGENCY_RANK.get(intent_counts[intent]["max_urgency"], 0) >= 2:
                result = intent_counts[intent]["best_item"].copy()
                result["_all_sources"] = all_sources
                result["_intent_breakdown"] = intent_summary
                result["_total_analyzed"] = len(items)
                return result

        # Otherwise pick most frequent intent
        dominant = max(intent_counts.values(), key=lambda x: x["count"])
        result = dominant["best_item"].copy()
        result["_all_sources"] = all_sources
        result["_intent_breakdown"] = intent_summary
        result["_total_analyzed"] = len(items)
        return result

    def _fetch_article(self, event: dict) -> str:
        """Fetch article content for an RSS entry if available."""
        try:
            from neuroacoustic.ingestion.rss_parser import RSSParser
            parser = RSSParser()
            return parser.fetch_article_content(event, max_chars=1500)
        except Exception:
            return ""

    def _create_event_text(self, event: dict) -> str:
        """Create a text representation of an event for LLM extraction."""
        # Handle calendar events
        if "title" in event and "start_time" in event:
            parts = [event["title"]]
            if event.get("description"):
                parts.append(event["description"])
            if event.get("location"):
                parts.append(f"Location: {event['location']}")
            return " - ".join(parts)

        # Handle RSS entries — prefer article content if fetched
        if "summary" in event:
            parts = [event.get("title", "")]
            if event.get("article_content"):
                # Use article excerpt for much richer LLM context
                parts.append(event["article_content"][:800])
            elif event.get("summary"):
                parts.append(event["summary"])
            return " - ".join(parts)

        # Fallback: JSON representation
        return json.dumps(event)

    def _orchestration_loop(self) -> None:
        """Main thread: processes queued events through the semantic mapper with crossfading."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                # Update synthesizer with current crossfade state even if no new events
                if self._crossfade_manager.is_transitioning():
                    current_params = self._crossfade_manager.get_current_params()
                    self._synthesizer.update_state(
                        carrier_freq=current_params["carrier_freq"],
                        beat_freq=current_params["beat_freq"],
                        amplitude=current_params["target_amplitude"],
                        metadata={},
                    )
                continue

            # Process event through semantic mapper
            payload = json.dumps(event)
            params = self._mapper.translate_payload(payload)

            # Rich terminal display
            meta = params.get("original_metadata", {})
            intent = meta.get("intent", "unknown")
            urgency = meta.get("urgency", "unknown")
            sentiment = meta.get("sentiment", "neutral")
            summary = meta.get("content_summary", "")[:120]
            source_type = meta.get("_source_type", meta.get("data_source", "unknown"))
            source_title = meta.get("_source_title", "")[:80]
            all_sources = meta.get("_all_sources", [])
            intent_breakdown = meta.get("_intent_breakdown", {})
            total = meta.get("_total_analyzed", 1)
            band = params.get("target_band", "")
            noise = params.get("noise_color", "")
            brightness = params.get("timbre_brightness", 0.5)

            # Intent indicators
            INTENT_ICON = {"focus": "◆", "relaxation": "◇", "alertness": "▲", "learning": "■"}
            URGENCY_BAR = {"low": "░", "medium": "▒", "high": "▓", "critical": "█"}
            SENTIMENT_ICON = {"positive": "+", "neutral": "○", "negative": "-"}

            ts = time.strftime("%H:%M:%S")
            print(f"\n{'─' * 64}")
            print(f"  {ts}  {INTENT_ICON.get(intent, '?')} {intent.upper()} / {band}")
            print(f"{'─' * 64}")

            source_url = meta.get("_source_url", "")
            has_article = meta.get("_has_article", False)

            if source_title:
                print(f"  Source:    [{source_type}] {source_title}")
            if source_url:
                print(f"  Link:      {source_url[:80]}")
            if summary and summary != source_title:
                print(f"  Summary:   {summary}")
            if has_article:
                print(f"  Content:   (full article fetched and analyzed)")
            print()

            print(f"  Intent:    {intent:<12}  Urgency:   {URGENCY_BAR.get(urgency, '?')} {urgency}")
            print(f"  Sentiment: {SENTIMENT_ICON.get(sentiment, '?')} {sentiment:<10}  Brightness: {'█' * int(brightness * 10)}{'░' * (10 - int(brightness * 10))} {brightness:.1f}")
            print()

            print(f"  Carrier:   {params['carrier_freq']:>7.1f} Hz    Beat:  {params['beat_freq']:>5.1f} Hz ({band})")
            print(f"  Amplitude: {params['target_amplitude']:>7.2f}        Noise: {noise}")

            if intent_breakdown and total > 1:
                breakdown = " / ".join(f"{k}:{v}" for k, v in sorted(intent_breakdown.items(), key=lambda x: -x[1]))
                print(f"\n  Analyzed {total} items: {breakdown}")

            if all_sources and len(all_sources) > 1:
                print(f"  Headlines:")
                for s in all_sources[:5]:
                    print(f"    • {s[:72]}")
                if len(all_sources) > 5:
                    print(f"    ... and {len(all_sources) - 5} more")

            # Set target state for crossfading
            self._crossfade_manager.set_target_state(params)

            # Immediately update with interpolated params
            current_params = self._crossfade_manager.get_current_params()
            self._synthesizer.update_state(
                carrier_freq=current_params["carrier_freq"],
                beat_freq=current_params["beat_freq"],
                amplitude=current_params["target_amplitude"],
                metadata=params.get("original_metadata", {}),
            )


class FileWatcherSource:
    """
    Data source that watches a directory for new .json event files.

    Useful for testing and for external integrations that want to
    trigger audio state changes by dropping JSON files.
    """

    def __init__(self, watch_dir: str):
        """
        Initialize the file watcher.

        Args:
            watch_dir: Directory path to watch for .json files.
        """
        self._watch_dir = Path(watch_dir)
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        self._processed_files: set[str] = set()

    def poll(self) -> list[dict]:
        """
        Check for new .json files in the watch directory.

        Returns:
            List of event dictionaries parsed from new JSON files.
        """
        events = []

        if not self._watch_dir.exists():
            return events

        for json_file in self._watch_dir.glob("*.json"):
            file_path = str(json_file)
            if file_path not in self._processed_files:
                try:
                    with open(json_file, "r") as f:
                        event = json.load(f)
                    events.append(event)
                    self._processed_files.add(file_path)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Error reading {json_file}: {e}")

        return events


class CalendarSource:
    """Data source wrapper for CalendarParser."""

    def __init__(self, calendar_parser, ics_file_path: str):
        """
        Initialize the calendar source.

        Args:
            calendar_parser: Instance of CalendarParser.
            ics_file_path: Path to the .ics calendar file.
        """
        self._parser = calendar_parser
        self._ics_file_path = ics_file_path
        self._last_events: list[dict] = []

    def poll(self) -> list[dict]:
        """
        Poll the calendar file for events.

        Returns:
            List of new events since last poll.
        """
        try:
            events = self._parser.parse_ics_file(self._ics_file_path)
            # For simplicity, return events that weren't in the last poll
            new_events = [e for e in events if e not in self._last_events]
            self._last_events = events
            return new_events
        except Exception as e:
            print(f"Error polling calendar: {e}")
            return []


class RSSSource:
    """
    Data source wrapper for RSSParser.

    Rotates through feed items on each poll cycle, returning a different
    window of 5 items each time. Refreshes the full feed periodically
    to pick up genuinely new content.
    """

    def __init__(self, rss_parser, feed_url: str, refresh_every: int = 10):
        """
        Initialize the RSS source.

        Args:
            rss_parser: Instance of RSSParser.
            feed_url: URL of the RSS/Atom feed.
            refresh_every: Re-fetch the feed every N poll cycles.
        """
        self._parser = rss_parser
        self._feed_url = feed_url
        self._entries: list[dict] = []
        self._cursor = 0
        self._poll_count = 0
        self._refresh_every = refresh_every
        self._window_size = 5

    def poll(self) -> list[dict]:
        """
        Return the next window of items from the feed, rotating through.

        Re-fetches the feed every `refresh_every` cycles to pick up new content.
        """
        try:
            # Refresh feed periodically or on first poll
            if not self._entries or self._poll_count % self._refresh_every == 0:
                self._entries = self._parser.parse_feed(self._feed_url)
                # Don't reset cursor on refresh — keep rotating

            self._poll_count += 1

            if not self._entries:
                return []

            # Return a sliding window of items, wrapping around
            result = []
            for i in range(self._window_size):
                idx = (self._cursor + i) % len(self._entries)
                result.append(self._entries[idx])

            # Advance cursor for next poll
            self._cursor = (self._cursor + self._window_size) % len(self._entries)

            return result
        except Exception as e:
            print(f"Error polling RSS feed: {e}")
            return []
