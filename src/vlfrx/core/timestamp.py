"""Timestamp handling for vlfrx."""

from __future__ import annotations

import re
import time
from datetime import UTC, datetime


class Timestamp:
    """High-precision timestamp for VLF signal processing.

    Supports nanosecond precision using a compound representation
    (seconds + fractional part) when platform long double is insufficient.

    Attributes:
        secs: Seconds since epoch
        frac: Fractional part (0 <= frac < 1)

    Example:
        >>> t = Timestamp.now()
        >>> t2 = Timestamp(1234567890, 0.5)
        >>> print(t2.format())
        1234567890.5000000
    """

    __slots__ = ("secs", "frac")

    NONE: Timestamp
    ZERO: Timestamp

    def __init__(
        self, secs: int = 0, frac: float = 0.0, _normalize: bool = True
    ) -> None:
        """Create a timestamp from seconds and fractional part.

        Args:
            secs: Seconds since epoch
            frac: Fractional part (will be normalized to [0, 1) unless _normalize=False)
            _normalize: Whether to normalize the timestamp (internal use)
        """
        self.secs = secs
        self.frac = frac
        if _normalize:
            self._normalize()

    @classmethod
    def _create_raw(cls, secs: int, frac: float) -> Timestamp:
        """Create a timestamp without normalization (internal use)."""
        return cls(secs, frac, _normalize=False)

    def _normalize(self) -> None:
        """Normalize the timestamp so frac is in [0, 1)."""
        if self.secs < 0:
            self.secs = 0
            self.frac = 0.0
            return

        if self.secs == 0 and self.frac < 0:
            self.frac = 0.0
            return

        n = int(self.frac)
        self.secs += n
        self.frac -= n

    @classmethod
    def from_seconds(cls, t: float) -> Timestamp:
        """Create timestamp from floating point seconds.

        Args:
            t: Time in seconds since epoch

        Returns:
            Normalized Timestamp
        """
        secs = int(t)
        frac = t - secs
        return cls(secs, frac)

    @classmethod
    def now(cls) -> Timestamp:
        """Get current timestamp.

        Returns:
            Current time as Timestamp
        """
        now = time.time()
        return cls.from_seconds(now)

    @classmethod
    def from_datetime(cls, dt: datetime) -> Timestamp:
        """Create timestamp from datetime object.

        Args:
            dt: Datetime (timezone-aware or naive)

        Returns:
            Timestamp representation
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        secs = int(dt.timestamp())
        frac = (dt.timestamp() - secs) % 1
        return cls._create_raw(secs, frac)

    @classmethod
    def parse(cls, s: str) -> Timestamp:
        """Parse timestamp from string.

        Supports formats:
        - ISO 8601: "2024-01-15T12:30:45.123456"
        - Unix timestamp: "1705326645.123456"
        - Simple: "1234567890.5"

        Args:
            s: String to parse

        Returns:
            Parsed Timestamp

        Raises:
            ValueError: If string format is invalid
        """
        s = s.strip()

        # Try ISO 8601 format
        iso_match = re.match(
            r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?",
            s,
        )
        if iso_match:
            year, month, day, hour, minute, second = map(int, iso_match.groups()[:6])
            frac_str = iso_match.group(7) or ""
            frac = 0.0
            if frac_str:
                frac = int(frac_str.ljust(9, "0")[:9]) / 1e9

            # Compute timestamp directly using calendar.timegm (UTC)
            import calendar

            secs = calendar.timegm((year, month, day, hour, minute, second))
            return cls._create_raw(secs, frac)

        # Try plain number (unix timestamp)
        try:
            return cls.from_seconds(float(s))
        except ValueError:
            pass

        # Try "seconds.decimal" format
        match = re.match(r"(\d+)\.(\d+)", s)
        if match:
            secs = int(match.group(1))
            frac_str = match.group(2)
            frac = int(frac_str.ljust(9, "0")[:9]) / 1e9
            return cls(secs, frac)

        raise ValueError(f"Cannot parse timestamp from: {s}")

    @classmethod
    def from_filename(cls, filename: str) -> Timestamp:
        """Extract timestamp from VT filename.

        Filename format: prefix_YMD_HMS.vt or prefix_YMD_HMS_frac.vt

        Args:
            filename: Filename to parse

        Returns:
            Timestamp from filename, or ZERO if not found
        """
        import os

        basename = os.path.basename(filename)
        match = re.match(r".*_(\d{8})_(\d{6})(?:\.(\d+))?\.vt$", basename)
        if not match:
            return cls.ZERO

        date_str, time_str, frac_str = match.groups()
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])

        dt = datetime(year, month, day, hour, minute, second, tzinfo=UTC)
        ts = cls.from_datetime(dt)

        if frac_str:
            frac = int(frac_str.ljust(9, "0")[:9]) / 1e9
            ts = cls(ts.secs, frac)
            ts._normalize()

        return ts

    def to_seconds(self) -> float:
        """Convert to floating point seconds.

        Returns:
            Seconds as float
        """
        return float(self.secs) + self.frac

    def to_datetime(self) -> datetime:
        """Convert to datetime.

        Returns:
            datetime in UTC
        """
        return datetime.fromtimestamp(self.to_seconds(), tz=UTC)

    def format(self, decimals: int = 7) -> str:
        """Format timestamp as string.

        Args:
            decimals: Number of decimal places (default 7 for nanoseconds)

        Returns:
            Formatted string
        """
        return f"{self.secs}.{int(self.frac * 10**decimals):0{decimals}d}"

    def format_iso(self) -> str:
        """Format as ISO 8601 string (UTC).

        Returns:
            ISO formatted string
        """
        secs = self.secs
        frac_ns = int(self.frac * 1e9)
        total_ns = secs * 1_000_000_000 + frac_ns

        # Convert to UTC datetime
        from datetime import datetime

        dt = datetime.fromtimestamp(total_ns / 1e9, tz=UTC)
        frac_us = frac_ns // 1000 % 1000000
        return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{frac_us:06d}Z"

    def __repr__(self) -> str:
        return f"Timestamp({self.secs}, {self.frac})"

    def __str__(self) -> str:
        return self.format()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self.secs == other.secs and abs(self.frac - other.frac) < 1e-15

    def __lt__(self, other: Timestamp) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        if self.secs != other.secs:
            return self.secs < other.secs
        return self.frac < other.frac

    def __le__(self, other: Timestamp) -> bool:
        return self == other or self < other

    def __gt__(self, other: Timestamp) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        if self.secs != other.secs:
            return self.secs > other.secs
        return self.frac > other.frac

    def __ge__(self, other: Timestamp) -> bool:
        return self == other or self > other

    def __hash__(self) -> int:
        return hash((self.secs, self.frac))

    def __add__(self, other: int | float | Timestamp) -> Timestamp:
        if isinstance(other, Timestamp):
            return Timestamp.from_seconds(self.to_seconds() + other.to_seconds())
        return Timestamp.from_seconds(self.to_seconds() + float(other))

    def __radd__(self, other: int | float | Timestamp) -> Timestamp:
        return self.__add__(other)

    def __sub__(self, other: int | float | Timestamp) -> Timestamp:
        if isinstance(other, Timestamp):
            return Timestamp.from_seconds(self.to_seconds() - other.to_seconds())
        return Timestamp.from_seconds(self.to_seconds() - float(other))

    def __rsub__(self, other: int | float | Timestamp) -> Timestamp:
        if isinstance(other, Timestamp):
            return Timestamp.from_seconds(other.to_seconds() - self.to_seconds())
        return Timestamp.from_seconds(float(other) - self.to_seconds())

    def __mul__(self, other: float) -> Timestamp:
        return Timestamp.from_seconds(self.to_seconds() * other)

    def __rmul__(self, other: float) -> Timestamp:
        return self.__mul__(other)

    def is_zero(self) -> bool:
        """Check if timestamp is zero.

        Returns:
            True if both secs and frac are zero
        """
        return self.secs == 0 and self.frac == 0.0

    def is_none(self) -> bool:
        """Check if timestamp represents NONE.

        Returns:
            True if secs is -1 and frac is 0
        """
        return self.secs == -1 and self.frac == 0.0


# Class-level constants
Timestamp.NONE = Timestamp._create_raw(-1, 0.0)
Timestamp.ZERO = Timestamp._create_raw(0, 0.0)
