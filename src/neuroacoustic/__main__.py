"""
CLI entry point for the NeuroAcoustic Engine.

Run with: python -m neuroacoustic [options]
"""

import argparse
import signal
import sys
import time

from neuroacoustic.ingestion.calendar_parser import CalendarParser
from neuroacoustic.ingestion.llm_agent import create_agent
from neuroacoustic.ingestion.rss_parser import RSSParser
from neuroacoustic.orchestration.execution_loop import (
    CalendarSource,
    ExecutionLoop,
    FileWatcherSource,
    RSSSource,
)
from neuroacoustic.synthesis.dsp_engine import CognitiveAudioSynthesizer
from neuroacoustic.translation.semantic_mapper import SemanticMapper


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NeuroAcoustic Engine - Contextual cognitive audio synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--calendar",
        metavar="PATH",
        help="Path to .ics calendar file to monitor",
    )

    parser.add_argument(
        "--rss",
        metavar="URL",
        help="RSS/Atom feed URL to monitor",
    )

    parser.add_argument(
        "--watch-dir",
        metavar="PATH",
        help="Directory to watch for .json event files",
    )

    parser.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="Run for specified duration then exit (default: run indefinitely)",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Data source polling interval in seconds (default: 60)",
    )

    parser.add_argument(
        "--crossfade-duration",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Audio state transition duration in seconds (default: 30)",
    )

    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "mock"],
        default="ollama",
        help="LLM backend for semantic extraction (default: ollama)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate that at least one data source is specified
    if not any([args.calendar, args.rss, args.watch_dir]):
        print("Error: Must specify at least one data source (--calendar, --rss, or --watch-dir)")
        sys.exit(1)

    print("=" * 60)
    print("NeuroAcoustic Engine")
    print("=" * 60)

    # Initialize components
    print("\nInitializing components...")
    synthesizer = CognitiveAudioSynthesizer()
    mapper = SemanticMapper()
    llm_agent = create_agent(backend=args.llm_backend)

    print(f"  LLM Backend: {args.llm_backend}")
    print(f"  Poll Interval: {args.poll_interval}s")
    print(f"  Crossfade Duration: {args.crossfade_duration}s")

    # Create execution loop
    loop = ExecutionLoop(
        synthesizer=synthesizer,
        mapper=mapper,
        llm_agent=llm_agent,
        poll_interval=args.poll_interval,
        crossfade_duration=args.crossfade_duration,
    )

    # Add data sources
    print("\nData sources:")
    if args.calendar:
        print(f"  Calendar: {args.calendar}")
        calendar_parser = CalendarParser()
        calendar_source = CalendarSource(calendar_parser, args.calendar)
        loop.add_data_source(calendar_source)

    if args.rss:
        print(f"  RSS Feed: {args.rss}")
        rss_parser = RSSParser()
        rss_source = RSSSource(rss_parser, args.rss)
        loop.add_data_source(rss_source)

    if args.watch_dir:
        print(f"  Watch Directory: {args.watch_dir}")
        file_watcher = FileWatcherSource(args.watch_dir)
        loop.add_data_source(file_watcher)

    print("\nStarting audio synthesis...")
    print("Press Ctrl+C to stop\n")

    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down gracefully...")
        loop.stop()
        print("Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the engine
    loop.start()

    # Run for specified duration or indefinitely
    if args.duration:
        print(f"Running for {args.duration} seconds...")
        time.sleep(args.duration)
        loop.stop()
        print("\nDuration elapsed. Exiting.")
    else:
        print("Running indefinitely. Press Ctrl+C to stop.")
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            signal_handler(None, None)


if __name__ == "__main__":
    main()
