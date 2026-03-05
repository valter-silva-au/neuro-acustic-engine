"""
RSS/Atom feed ingestion.

Fetches and parses syndicated web feeds, optionally retrieving
full article content from linked URLs for deeper LLM analysis.
"""

import re
from datetime import datetime
from time import struct_time

import feedparser
import requests
from bs4 import BeautifulSoup


class RSSParser:
    """Fetch and parse RSS/Atom feeds into event dictionaries."""

    def parse_feed(self, url: str) -> list[dict]:
        """
        Fetch and parse an RSS or Atom feed.

        Args:
            url: URL of the RSS/Atom feed.

        Returns:
            List of event dicts with keys: title, published, summary,
            source_url, content_type.

        Raises:
            ValueError: If the feed cannot be parsed or is invalid.
        """
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            raise ValueError(f"Failed to parse feed from {url}: {feed.get('bozo_exception', 'Unknown error')}")

        entries = []
        for entry in feed.entries:
            event = self._parse_entry(entry)
            if event:
                entries.append(event)

        return entries

    def _parse_entry(self, entry) -> dict | None:
        """
        Parse a feed entry into an event dict.

        Args:
            entry: feedparser entry object.

        Returns:
            Event dict or None if required fields are missing.
        """
        title = entry.get("title")
        if not title:
            return None

        published = self._extract_published_date(entry)

        summary = (
            entry.get("summary")
            or entry.get("description")
            or entry.get("content", [{}])[0].get("value", "")
        )

        source_url = entry.get("link", "")

        return {
            "title": str(title),
            "published": published,
            "summary": str(summary) if summary else "",
            "source_url": str(source_url),
            "content_type": "rss",
            "article_content": "",  # Populated by fetch_article_content
        }

    def fetch_article_content(self, entry: dict, max_chars: int = 1500) -> str:
        """
        Fetch the full article content from the entry's source URL.

        Downloads the page HTML, extracts readable text from <article>,
        <main>, or <body>, and returns the first max_chars characters.

        Args:
            entry: Event dict with a 'source_url' key.
            max_chars: Maximum characters to return (default 1500).

        Returns:
            Extracted article text, or empty string on failure.
        """
        url = entry.get("source_url", "")
        if not url:
            return ""

        try:
            resp = requests.get(url, timeout=10, headers={
                "User-Agent": "NeuroAcousticEngine/0.1 (RSS content fetcher)"
            })
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script, style, nav, header, footer elements
            for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Try to find article content in order of specificity
            content_el = (
                soup.find("article")
                or soup.find("main")
                or soup.find(attrs={"role": "main"})
                or soup.find("body")
            )

            if not content_el:
                return ""

            # Extract text, collapse whitespace
            text = content_el.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

            return text[:max_chars]
        except Exception:
            return ""

    def _extract_published_date(self, entry) -> str:
        """
        Extract published date from entry in ISO format.

        Args:
            entry: feedparser entry object.

        Returns:
            ISO format date string or empty string if not found.
        """
        # Try published_parsed first, then updated_parsed
        time_struct = entry.get("published_parsed") or entry.get("updated_parsed")

        if time_struct:
            # Handle both struct_time and plain tuples
            if isinstance(time_struct, (struct_time, tuple)):
                dt = datetime(*time_struct[:6])
                return dt.isoformat()

        # Fallback to string versions
        date_str = entry.get("published") or entry.get("updated", "")
        return str(date_str) if date_str else ""
