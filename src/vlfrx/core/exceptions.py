"""Exception classes for vlfrx."""

from __future__ import annotations


class VLFRXError(Exception):
    """Base exception for vlfrx library."""

    pass


class VTFileError(VLFRXError):
    """Error opening or operating on VT file."""

    pass


class VTFormatError(VLFRXError):
    """Error in VT file format."""

    pass


class VTNetworkError(VLFRXError):
    """Network-related errors."""

    pass


class VTBufferError(VLFRXError):
    """Shared memory buffer errors."""

    pass
