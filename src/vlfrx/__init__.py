"""vlfrx - VLF Radio Signal Processing Library."""

__version__ = "0.1.0"

from vlfrx.core.chanspec import ChannelSpec, parse_chanspec
from vlfrx.core.timestamp import Timestamp
from vlfrx.core.vtfile import VTBlock, VTFile, open_input, open_output

__all__ = [
    "__version__",
    "Timestamp",
    "VTFile",
    "VTBlock",
    "ChannelSpec",
    "open_input",
    "open_output",
    "parse_chanspec",
]
