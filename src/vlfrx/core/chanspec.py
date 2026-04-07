"""Channel specification parsing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class ChannelSpec:
    """Channel specification for multi-channel signal processing.

    A ChannelSpec defines which channels to process and how to map them.
    Supports various specification formats:
    - Single channel: "0" or "1"
    - Range: "0-3" or "1:4"
    - List: "0,1,2" or "0,3,5"
    - Complex channel pair: "H1,H2,E" (for polarization)

    Attributes:
        map: Array of channel indices to use
        n: Number of channels in specification
    """

    __slots__ = ("map", "n")

    def __init__(self, map: list[int] | None = None) -> None:
        """Create a channel specification.

        Args:
            map: List of channel indices
        """
        if map is None:
            map = []
        self.map = list(map)
        self.n = len(self.map)

    @classmethod
    def parse(cls, spec: str, total_channels: int = 2) -> ChannelSpec:
        """Parse a channel specification string.

        Args:
            spec: Specification string (e.g., "0", "0-3", "0,1,2", "H1,H2,E")
            total_channels: Total number of available channels

        Returns:
            Parsed ChannelSpec
        """
        spec = spec.strip()
        if not spec:
            return cls(list(range(total_channels)))

        # Handle special polarization specs
        if spec in ("H1", "H2", "E"):
            return cls._parse_polarization(spec, total_channels)

        # Try numeric specifications
        indices: list[int] = []

        # Check for range "0-3" or "1:4"
        range_match = re.match(r"(\d+)[-:](-?\d+)", spec)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < 0:
                end = total_channels + end
            indices = list(range(start, end + 1))
            return cls(indices)

        # Check for comma-separated list "0,1,2"
        if "," in spec:
            parts = spec.split(",")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # Check for sub-range in list element
                sub_match = re.match(r"(\d+)[-:](-?\d+)", part)
                if sub_match:
                    start = int(sub_match.group(1))
                    end = int(sub_match.group(2))
                    if end < 0:
                        end = total_channels + end
                    indices.extend(range(start, end + 1))
                else:
                    try:
                        indices.append(int(part))
                    except ValueError:
                        # Could be named channel like "H1"
                        if part in ("H1", "H2", "E"):
                            cls._add_polarization_index(indices, part, total_channels)
            return cls(indices)

        # Single number
        try:
            idx = int(spec)
            return cls([idx])
        except ValueError:
            pass

        # Default: use all channels
        return cls(list(range(total_channels)))

    @staticmethod
    def _parse_polarization(pol: str, total_channels: int) -> ChannelSpec:
        """Parse polarization specification.

        Args:
            pol: Polarization string (H1, H2, or E)
            total_channels: Total available channels

        Returns:
            ChannelSpec with appropriate indices
        """
        # Typical VLF receiver setup: H1=channel 0, H2=channel 1, E=channel 2
        indices: list[int] = []
        ChannelSpec._add_polarization_index(indices, pol, total_channels)
        return ChannelSpec(indices)

    @staticmethod
    def _add_polarization_index(indices: list[int], pol: str, total: int) -> None:
        """Add index for polarization channel."""
        if pol == "H1" and total >= 1:
            indices.append(0)
        elif pol == "H2" and total >= 2:
            indices.append(1)
        elif pol == "E" and total >= 3:
            indices.append(2)

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[int]:
        return iter(self.map)

    def __getitem__(self, i: int) -> int:
        return self.map[i]

    def __repr__(self) -> str:
        return f"ChannelSpec({self.map})"


def parse_chanspec(spec: str, total_channels: int = 2) -> ChannelSpec:
    """Parse a channel specification string.

    Args:
        spec: Specification string
        total_channels: Total available channels

    Returns:
        Parsed ChannelSpec
    """
    return ChannelSpec.parse(spec, total_channels)
