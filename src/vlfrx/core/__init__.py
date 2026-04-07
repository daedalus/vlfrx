"""Core domain logic - no external dependencies."""

from vlfrx.core.buffer import VTBuffer
from vlfrx.core.chanspec import ChannelSpec, parse_chanspec
from vlfrx.core.exceptions import VLFRXError, VTFileError, VTFormatError
from vlfrx.core.timestamp import Timestamp
from vlfrx.core.vtfile import VTBlock, VTFile, open_input, open_output

__all__ = [
    "Timestamp",
    "VTFile",
    "VTBlock",
    "ChannelSpec",
    "VTBuffer",
    "VLFRXError",
    "VTFileError",
    "VTFormatError",
    "open_input",
    "open_output",
    "parse_chanspec",
]
