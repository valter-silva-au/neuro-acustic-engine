#!/usr/bin/env python3
"""
Live demonstration of the NeuroAcoustic Engine full pipeline.

Demonstrates the complete flow:
1. Calendar events → LLM extraction → Semantic mapping → Audio synthesis
2. Real-time state transitions with crossfading
3. Terminal output showing state changes

This demo uses compressed-time events (seconds instead of hours) to show
multiple state transitions in 60 seconds.

The demo uses the mock LLM agent by default (no network required), but will
attempt to use Ollama if available.
"""

import json
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from neuroacoustic.ingestion.calendar_parser import CalendarParser
from neuroacoustic.ingestion.llm_agent import OllamaLLMAgent, create_agent
from neuroacoustic.orchestration.crossfader import CrossfadeManager
from neuroacoustic.orchestration.execution_loop import (
    CalendarSource,
    ExecutionLoop,
)
from neuroacoustic.synthesis.streaming import (
    DEFAULT_SAMPLE_RATE,
    PhaseState,
    StreamingWavWriter,
    generate_binaural_chunk,
)
from neuroacoustic.translation.semantic_mapper import SemanticMapper


def create_demo_calendar(output_path: str) -> str:
    """
    Create a sample .ics calendar with compressed-time events.

    Events occur at 15-second intervals to demonstrate multiple state
    transitions within 60 seconds.
    """
    now = datetime.now()

    events = [
        {
            "start": now,
            "end": now + timedelta(seconds=15),
            "summary": "Deep work: System design",
            "description": "Focus time for designing the neuroacoustic architecture",
        },
        {
            "start": now + timedelta(seconds=15),
            "end": now + timedelta(seconds=30),
            "summary": "Team standup",
            "description": "Daily sync with the development team",
        },
        {
            "start": now + timedelta(seconds=30),
            "end": now + timedelta(seconds=45),
            "summary": "Coffee break",
            "description": "Relax and recharge",
        },
        {
            "start": now + timedelta(seconds=45),
            "end": now + timedelta(seconds=60),
            "summary": "Creative brainstorming",
            "description": "Learning new synthesis techniques and ideating features",
        },
    ]

    # Generate ICS content
    ics_content = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//NeuroAcoustic Engine Demo//EN
CALSCALE:GREGORIAN
"""

    for i, event in enumerate(events):
        ics_content += f"""BEGIN:VEVENT
UID:demo-event-{i}@neuroacoustic.local
DTSTAMP:{now.strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{event['start'].strftime('%Y%m%dT%H%M%S')}
DTEND:{event['end'].strftime('%Y%m%dT%H%M%S')}
SUMMARY:{event['summary']}
DESCRIPTION:{event['description']}
END:VEVENT
"""

    ics_content += "END:VCALENDAR\n"

    # Write to file
    with open(output_path, "w") as f:
        f.write(ics_content)

    return output_path


class DemoSynthesizer:
    """
    Synthesizer wrapper that handles both real-time audio (pyo) and
    streaming WAV output, with graceful fallback if pyo fails.
    """

    def __init__(self, output_dir: Path, use_pyo: bool = True):
        self.output_dir = output_dir
        self.use_pyo = use_pyo
        self.pyo_synth = None
        self.current_state = {
            "carrier_freq": 150.0,
            "beat_freq": 5.0,
            "amplitude": 0.0,
        }
        self.state_history = []
        self.start_time = None

        # Streaming audio phases for WAV generation
        self.left_phase = PhaseState()
        self.right_phase = PhaseState()

        if use_pyo:
            try:
                from neuroacoustic.synthesis.dsp_engine import (
                    CognitiveAudioSynthesizer,
                )

                self.pyo_synth = CognitiveAudioSynthesizer()
                print("[Audio] Using pyo for real-time audio output")
            except Exception as e:
                print(f"[Audio] pyo not available ({e}), using WAV streaming only")
                self.pyo_synth = None

    def start(self):
        """Start the synthesizer."""
        self.start_time = time.time()
        if self.pyo_synth:
            try:
                self.pyo_synth.start()
            except Exception as e:
                print(f"[Audio] Failed to start pyo audio server ({e})")
                self.pyo_synth = None

    def stop(self):
        """Stop the synthesizer."""
        if self.pyo_synth:
            try:
                self.pyo_synth.stop()
            except Exception:
                pass

    def update_state(self, carrier_freq, beat_freq, amplitude, metadata):
        """Update audio parameters and save WAV segment."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Store state change
        state = {
            "timestamp": elapsed,
            "carrier_freq": carrier_freq,
            "beat_freq": beat_freq,
            "amplitude": amplitude,
            "metadata": metadata,
        }
        self.state_history.append(state)
        self.current_state.update({
            "carrier_freq": carrier_freq,
            "beat_freq": beat_freq,
            "amplitude": amplitude,
        })

        # Update pyo if available
        if self.pyo_synth:
            try:
                self.pyo_synth.update_state(
                    carrier_freq=carrier_freq,
                    beat_freq=beat_freq,
                    amplitude=amplitude,
                    metadata=metadata,
                )
            except Exception as e:
                print(f"[Audio] Error updating pyo state: {e}")

        # Print state change to terminal
        intent = metadata.get("intent", "unknown")
        urgency = metadata.get("urgency", "unknown")
        print(f"\n[{elapsed:05.1f}s] State Change:")
        print(f"  Event: {metadata.get('content_summary', 'N/A')}")
        print(f"  Intent: {intent} | Urgency: {urgency}")
        print(f"  → Carrier: {carrier_freq:.1f} Hz | Beat: {beat_freq:.1f} Hz")
        print(f"  → Amplitude: {amplitude:.2f}")

        # Save short WAV segment for this state (3 seconds)
        self._save_wav_segment(state, duration=3.0)

    def _save_wav_segment(self, state, duration=3.0):
        """Save a short WAV segment for the current state."""
        intent = state["metadata"].get("intent", "unknown")
        timestamp = state["timestamp"]
        wav_filename = f"demo_{timestamp:05.1f}s_{intent}.wav"
        wav_path = self.output_dir / wav_filename

        num_samples = int(duration * DEFAULT_SAMPLE_RATE)

        try:
            with StreamingWavWriter(str(wav_path), sample_rate=DEFAULT_SAMPLE_RATE) as writer:
                chunk = generate_binaural_chunk(
                    left_phase=self.left_phase,
                    right_phase=self.right_phase,
                    carrier_freq=state["carrier_freq"],
                    beat_freq=state["beat_freq"],
                    num_samples=num_samples,
                    amplitude=state["amplitude"],
                    sample_rate=DEFAULT_SAMPLE_RATE,
                )
                writer.write_chunk(chunk)

            print(f"  💾 Saved: {wav_filename}")
        except Exception as e:
            print(f"  ⚠️  Failed to save WAV: {e}")

    def get_metadata(self):
        """Return current metadata."""
        return self.state_history[-1]["metadata"] if self.state_history else {}


def check_ollama_available() -> bool:
    """Check if Ollama is available and responsive."""
    try:
        agent = OllamaLLMAgent(timeout=2)
        # Try a simple extraction to verify it works
        test_result = agent.extract_metadata("test", "test")
        return True
    except Exception:
        return False


def main():
    """Run the live demo."""
    print("=" * 70)
    print("🎵 NeuroAcoustic Engine - Live Demo")
    print("=" * 70)
    print()
    print("This demo runs the full pipeline for 60 seconds:")
    print("  1. Calendar events (compressed time)")
    print("  2. LLM semantic extraction")
    print("  3. Semantic-to-acoustic mapping")
    print("  4. Real-time audio synthesis with crossfading")
    print()

    # Create temporary directory for demo artifacts
    demo_dir = Path(tempfile.mkdtemp(prefix="neuroacoustic_demo_"))
    print(f"📁 Demo artifacts: {demo_dir}")
    print()

    # Create demo calendar
    ics_path = demo_dir / "demo_calendar.ics"
    print("[Setup] Creating demo calendar with compressed-time events...")
    create_demo_calendar(str(ics_path))
    print(f"  ✓ Calendar: {ics_path}")

    # Initialize components
    parser = CalendarParser()
    mapper = SemanticMapper()

    # Check if Ollama is available
    print("\n[Setup] Checking LLM backend availability...")
    ollama_available = check_ollama_available()

    if ollama_available:
        print("  ✓ Ollama is available - using real LLM extraction")
        llm_agent = create_agent(backend="ollama")
    else:
        print("  ⚠️  Ollama not available - using mock LLM (keyword matching)")
        llm_agent = create_agent(backend="mock")

    # Initialize synthesizer
    print("\n[Setup] Initializing synthesizer...")
    synth = DemoSynthesizer(demo_dir, use_pyo=True)

    # Create execution loop
    loop = ExecutionLoop(
        synthesizer=synth,
        mapper=mapper,
        llm_agent=llm_agent,
        poll_interval=5.0,  # Poll every 5 seconds
        crossfade_duration=5.0,  # 5-second crossfades for demo smoothness
    )

    # Add calendar source
    calendar_source = CalendarSource(parser, str(ics_path))
    loop.add_data_source(calendar_source)

    print("  ✓ Pipeline ready")
    print()
    print("=" * 70)
    print("🚀 Starting demo (60 seconds)...")
    print("=" * 70)

    # Start the loop
    loop.start()

    try:
        # Run for 60 seconds
        demo_duration = 60.0
        start = time.time()

        while (time.time() - start) < demo_duration:
            remaining = demo_duration - (time.time() - start)
            if remaining > 0:
                time.sleep(min(1.0, remaining))

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    finally:
        print("\n" + "=" * 70)
        print("🛑 Stopping demo...")
        print("=" * 70)
        loop.stop()
        synth.stop()

    # Print summary
    print("\n" + "=" * 70)
    print("📊 Demo Summary")
    print("=" * 70)
    print(f"\nTotal state changes: {len(synth.state_history)}")
    print(f"Demo artifacts saved to: {demo_dir}")
    print()

    if synth.state_history:
        print("State transition timeline:")
        for state in synth.state_history:
            intent = state["metadata"].get("intent", "unknown")
            timestamp = state["timestamp"]
            carrier = state["carrier_freq"]
            beat = state["beat_freq"]
            print(f"  [{timestamp:05.1f}s] {intent:12s} → {carrier:6.1f} Hz / {beat:5.1f} Hz beat")

    print()
    print("=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  • Listen to WAV files in:", demo_dir)
    print("  • Run full test suite: python -m pytest tests/ -v")
    print("  • View CLI help: python -m neuroacoustic --help")
    print()


if __name__ == "__main__":
    main()
