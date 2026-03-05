"""
Calendar data ingestion (ICS/CalDAV).

Stub module for parsing iCalendar files and CalDAV feeds into
standardized event dictionaries for the semantic mapper.
"""

from datetime import datetime
from pathlib import Path

from icalendar import Calendar


class CalendarParser:
    """Parse ICS calendar files into event dictionaries."""

    def parse_ics_file(self, file_path: str) -> list[dict]:
        """
        Parse a local .ics file into a list of event dicts.

        Args:
            file_path: Path to the .ics calendar file.

        Returns:
            List of event dicts with keys: title, start_time, end_time,
            duration_minutes, description, location.

        Raises:
            FileNotFoundError: If the ICS file does not exist.
            ValueError: If the file is not a valid ICS file.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"ICS file not found: {file_path}")

        with open(file_path, "rb") as f:
            calendar = Calendar.from_ical(f.read())

        events = []
        for component in calendar.walk():
            if component.name == "VEVENT":
                event = self._parse_vevent(component)
                if event:
                    events.append(event)

        return events

    def _parse_vevent(self, vevent) -> dict | None:
        """
        Parse a VEVENT component into an event dict.

        Args:
            vevent: icalendar VEVENT component.

        Returns:
            Event dict or None if required fields are missing.
        """
        summary = vevent.get("SUMMARY")
        dtstart = vevent.get("DTSTART")
        dtend = vevent.get("DTEND")

        if not summary or not dtstart:
            return None

        # Convert datetime objects
        start_time = self._to_datetime(dtstart.dt)
        end_time = self._to_datetime(dtend.dt) if dtend else start_time

        # Calculate duration in minutes
        duration_minutes = int((end_time - start_time).total_seconds() / 60)

        # Extract optional fields
        description = vevent.get("DESCRIPTION", "")
        location = vevent.get("LOCATION", "")

        return {
            "title": str(summary),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_minutes,
            "description": str(description) if description else "",
            "location": str(location) if location else "",
        }

    def _to_datetime(self, dt) -> datetime:
        """
        Convert various datetime types to timezone-naive datetime.

        Args:
            dt: datetime, date, or other datetime-like object.

        Returns:
            Timezone-naive datetime object.
        """
        if isinstance(dt, datetime):
            # Convert timezone-aware datetime to naive (local time)
            if dt.tzinfo is not None:
                return dt.replace(tzinfo=None)
            return dt
        else:
            # Handle date objects (convert to datetime at midnight)
            return datetime.combine(dt, datetime.min.time())

    def poll_caldav(self, url: str, username: str, password: str) -> list[dict]:
        """
        Poll a CalDAV server for upcoming events.

        Args:
            url: CalDAV server URL.
            username: Authentication username.
            password: Authentication password.

        Returns:
            List of event dicts.

        Raises:
            NotImplementedError: CalDAV integration is planned for a future iteration.
        """
        raise NotImplementedError(
            "CalDAV polling requires the caldav library. "
            "Coming in a future iteration."
        )
