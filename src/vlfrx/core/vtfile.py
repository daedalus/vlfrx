"""VT file format handling."""

from __future__ import annotations

import struct
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    import mmap

import numpy as np

from vlfrx.core.exceptions import VTFileError, VTFormatError
from vlfrx.core.timestamp import Timestamp


class VTFlag(IntEnum):
    """VT file flag values."""

    RELT = 1 << 0  # Timestamps are relative, not absolute
    FLOAT4 = 1 << 1  # 4 byte floats (8 byte default)
    FLOAT8 = 0
    INT1 = 2 << 1  # 1 byte signed integers
    INT2 = 3 << 1  # 2 byte signed integers
    INT4 = 4 << 1  # 4 byte signed integers

    FMTMASK = FLOAT4 | INT1 | INT2 | INT4


class VTType(IntEnum):
    """VT file stream types."""

    BUFFER = 1  # Lock-free buffer (shared memory)
    FILE = 2  # File or FIFO
    NET = 3  # Network connection
    NETP = 4  # Persistent network connection


# Magic numbers
MAGIC_BUF = 26374  # Lock-free buffer header
MAGIC_BLK = 27859  # Data block header

# Block header size
BLOCK_HEADER_SIZE = 48  # Size of VT_BLOCK structure in bytes


class VTBlock:
    """A single data block in a VT file.

    Contains metadata and frame data for a segment of the signal.
    """

    __slots__ = (
        "magic",
        "flags",
        "bsize",
        "chans",
        "sample_rate",
        "secs",
        "nsec",
        "valid",
        "frames",
        "spare",
        "srcal",
        "data",
    )

    def __init__(self) -> None:
        self.magic: int = 0
        self.flags: int = 0
        self.bsize: int = 0
        self.chans: int = 0
        self.sample_rate: int = 0
        self.secs: int = 0
        self.nsec: int = 0
        self.valid: int = 0
        self.frames: int = 0
        self.spare: int = 0
        self.srcal: float = 1.0
        self.data: np.ndarray | None = None

    @property
    def timestamp(self) -> Timestamp:
        """Get block timestamp."""
        return Timestamp(self.secs, self.nsec / 1e9)

    @classmethod
    def from_bytes(cls, data: bytes, chans: int, flags: int) -> VTBlock:
        """Parse block from bytes.

        Args:
            data: Raw block data including header
            chans: Number of channels
            flags: VT flags indicating format

        Returns:
            Parsed VTBlock
        """
        if len(data) < BLOCK_HEADER_SIZE:
            raise VTFormatError(f"Block too small: {len(data)} bytes")

        block = cls()

        # Parse header (packed structure)
        header = struct.unpack("<IIIIIIIIii d", data[:BLOCK_HEADER_SIZE])
        (
            block.magic,
            block.flags,
            block.bsize,
            block.chans,
            block.sample_rate,
            block.secs,
            block.nsec,
            block.valid,
            block.frames,
            block.spare,
            block.srcal,
        ) = header[:11]

        if block.magic != MAGIC_BLK:
            raise VTFormatError(f"Invalid block magic: {block.magic}")

        # Parse frame data
        frame_data = data[BLOCK_HEADER_SIZE:]
        block.data = cls._parse_frame_data(frame_data, block.frames, chans, flags)

        return block

    @staticmethod
    def _parse_frame_data(
        data: bytes, frames: int, chans: int, flags: int
    ) -> np.ndarray:
        """Parse frame data based on format flags."""
        dtype: np.dtype[Any]

        fmt = flags & VTFlag.FMTMASK
        if fmt == VTFlag.FLOAT8 or fmt == 0:
            dtype = np.dtype(np.float64)
            itemsize = 8
        elif fmt == VTFlag.FLOAT4:
            dtype = np.dtype(np.float32)
            itemsize = 4
        elif fmt == VTFlag.INT1:
            dtype = np.dtype(np.int8)
            itemsize = 1
        elif fmt == VTFlag.INT2:
            dtype = np.dtype(np.int16)
            itemsize = 2
        elif fmt == VTFlag.INT4:
            dtype = np.dtype(np.int32)
            itemsize = 4
        else:
            raise VTFormatError(f"Unknown format flags: {flags}")

        itemsize = dtype.itemsize
        expected_size = frames * chans * itemsize
        if len(data) < expected_size:
            raise VTFormatError(
                f"Insufficient data: expected {expected_size}, got {len(data)}"
            )

        arr = np.frombuffer(data[:expected_size], dtype=dtype)
        return arr.reshape((frames, chans))

    def to_bytes(self) -> bytes:
        """Serialize block to bytes."""
        header = struct.pack(
            "<IIIIIIIIii d",
            self.magic,
            self.flags,
            self.bsize,
            self.chans,
            self.sample_rate,
            self.secs,
            self.nsec,
            self.valid,
            self.frames,
            self.spare,
            self.srcal,
        )

        # Convert data to bytes based on format
        data_bytes = self._frame_data_to_bytes()
        return header + data_bytes

    def _frame_data_to_bytes(self) -> bytes:
        """Convert frame data to bytes based on flags."""
        fmt = self.flags & VTFlag.FMTMASK

        if self.data is None:
            return b""

        if fmt == VTFlag.FLOAT8 or fmt == 0:
            return self.data.astype(np.float64).tobytes()
        elif fmt == VTFlag.FLOAT4:
            return self.data.astype(np.float32).tobytes()
        elif fmt == VTFlag.INT1:
            return self.data.astype(np.int8).tobytes()
        elif fmt == VTFlag.INT2:
            return self.data.astype(np.int16).tobytes()
        elif fmt == VTFlag.INT4:
            return self.data.astype(np.int32).tobytes()
        else:
            return self.data.astype(np.float64).tobytes()


class VTFile:
    """File handle for reading/writing VT format files.

    The VT format is a binary format for time-series signal data.
    """

    def __init__(
        self,
        path: str | Path,
        mode: str = "r",
        channels: int = 1,
        sample_rate: int = 48000,
        block_size: int = 8192,
        format: str = "float64",
    ) -> None:
        """Open a VT file.

        Args:
            path: File path
            mode: 'r' for read, 'w' for write
            channels: Number of channels (write mode)
            sample_rate: Sample rate in Hz (write mode)
            block_size: Frames per block (write mode)
            format: Data format ('float64', 'float32', 'int16', 'int32')
        """
        self.path = Path(path)
        self.mode = mode
        self.channels = channels
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.format = format

        self._fh: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None
        self._current_block: VTBlock | None = None
        self._block_index: int = 0
        self._total_blocks: int = 0
        self._block_size_bytes: int = 0
        self._flags: int = 0

        # Write mode state
        self._timebase = Timestamp.ZERO
        self._srcal: float = 1.0
        self._nft: int = 0  # frames since timebase
        self._current_buffer: VTBlock | None = None
        self._buffer_frames: int = 0

        if mode == "r":
            self._open_read()
        elif mode == "w":
            self._open_write()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _open_read(self) -> None:
        """Open file for reading."""
        if not self.path.exists():
            raise VTFileError(f"File not found: {self.path}")

        self._fh = open(self.path, "rb")

        # Get file size
        self._fh.seek(0, 2)
        file_size = self._fh.tell()
        self._fh.seek(0)

        # Read entire file to get first block with actual frame count
        test_data = self._fh.read(file_size)

        if len(test_data) < BLOCK_HEADER_SIZE:
            raise VTFileError("Cannot read first block")

        # First extract just the header to get actual channel count and flags
        header = struct.unpack("<IIIIIIIIii d", test_data[:BLOCK_HEADER_SIZE])
        actual_chans = header[3]
        actual_frames = header[8] if header[8] > 0 else 1
        actual_flags = header[1]  # Get flags from header

        first_block = VTBlock.from_bytes(test_data, actual_chans, actual_flags)
        self.channels = first_block.chans
        self.sample_rate = first_block.sample_rate
        self.block_size = first_block.bsize
        self._flags = first_block.flags

        # Calculate block size based on max capacity (bsize)
        self._block_size_bytes = BLOCK_HEADER_SIZE + (
            self.block_size * self.channels * self._get_itemsize()
        )

        # Total blocks
        self._total_blocks = (
            file_size + self._block_size_bytes - 1
        ) // self._block_size_bytes

        # Cache first block data with correct channels (only actual frames)
        self._first_block_data = test_data[
            : BLOCK_HEADER_SIZE + actual_frames * self.channels * self._get_itemsize()
        ]
        self._first_block_actual_frames = actual_frames

        # Re-open file for normal reading
        self._fh.close()
        self._fh = open(self.path, "rb")

    def _open_write(self) -> None:
        """Open file for writing."""
        self._fh = open(self.path, "wb")
        self._flags = self._get_flags_from_format(self.format)

    def _get_itemsize(self) -> int:
        """Get bytes per sample based on format."""
        fmt = self._flags & VTFlag.FMTMASK
        if fmt == VTFlag.FLOAT8 or fmt == 0:
            return 8
        elif fmt == VTFlag.FLOAT4:
            return 4
        elif fmt == VTFlag.INT1:
            return 1
        elif fmt == VTFlag.INT2:
            return 2
        elif fmt == VTFlag.INT4:
            return 4
        return 8

    def _get_flags_from_format(self, format: str) -> int:
        """Convert format string to VT flags."""
        if format == "float64":
            return VTFlag.FLOAT8
        elif format == "float32":
            return VTFlag.FLOAT4
        elif format == "int16":
            return VTFlag.INT2
        elif format == "int32":
            return VTFlag.INT4
        elif format == "int8":
            return VTFlag.INT1
        else:
            raise ValueError(f"Unknown format: {format}")

    def _read_block_raw(self, index: int) -> bytes | None:
        """Read raw block bytes at given index."""
        if self._fh is None:
            return None

        offset = index * self._block_size_bytes
        self._fh.seek(offset)
        data = self._fh.read(self._block_size_bytes)

        if len(data) < BLOCK_HEADER_SIZE:
            return None

        return data

    def read_block(self) -> VTBlock | None:
        """Read next block from file.

        Returns:
            VTBlock or None if at end of file
        """
        # Handle first block from cached data
        if self._block_index == 0 and hasattr(self, "_first_block_data"):
            block = VTBlock.from_bytes(
                self._first_block_data, self.channels, self._flags
            )
            self._current_block = block
            self._block_index += 1
            del self._first_block_data
            return block

        if self._block_index >= self._total_blocks:
            return None

        data = self._read_block_raw(self._block_index)
        if data is None:
            return None

        block = VTBlock.from_bytes(data, self.channels, self._flags)
        self._current_block = block
        self._block_index += 1

        return block

    def read_frame(self) -> np.ndarray | None:
        """Read a single frame.

        Returns:
            Frame as numpy array (channels,) or None at EOF
        """
        # If no current block or exhausted, read next
        if self._current_block is None or self._current_block.frames == 0:
            block = self.read_block()
            if block is None:
                return None
            if block.data is None or len(block.data) == 0:
                return None

        # Return first frame from current block
        assert self._current_block is not None
        assert self._current_block.data is not None
        frame: np.ndarray = self._current_block.data[0]
        # Remove the used frame
        self._current_block.data = self._current_block.data[1:]
        self._current_block.frames -= 1

        return frame

    def read_frames(self, n: int) -> np.ndarray | None:
        """Read n frames.

        Args:
            n: Number of frames to read

        Returns:
            Array of shape (n, channels) or None if not enough data
        """
        frames: list[np.ndarray] = []

        for _ in range(n):
            frame = self.read_frame()
            if frame is None:
                break
            frames.append(frame)

        if not frames:
            return None

        return np.array(frames)

    def get_timestamp(self) -> Timestamp:
        """Get timestamp of current position.

        Returns:
            Current timestamp
        """
        if self._current_block is None:
            return Timestamp.ZERO

        block_ts = self._current_block.timestamp
        # Add offset for frames already consumed from this block
        offset = (self._block_index - 1 - (self._current_block.frames or 0)) / (
            self.sample_rate * self._srcal
        )
        return Timestamp.from_seconds(block_ts.to_seconds() + offset)

    def write_frame(self, frame: list[float] | np.ndarray) -> None:
        """Write a single frame.

        Args:
            frame: Frame data (list or array of floats)
        """
        if isinstance(frame, list):
            frame = np.array(frame, dtype=np.float64)

        if self._current_buffer is None:
            self._init_write_buffer()

        # Add frame to current buffer
        assert self._current_buffer is not None
        if self._current_buffer.data is None:
            self._current_buffer.data = np.array([frame])
        else:
            self._current_buffer.data = np.vstack([self._current_buffer.data, frame])

        self._buffer_frames += 1

        # If buffer full, write block
        if self._buffer_frames >= self.block_size:
            self._write_block()

    def write_frames(self, frames: np.ndarray) -> None:
        """Write multiple frames.

        Args:
            frames: Array of shape (n, channels)
        """
        for frame in frames:
            self.write_frame(frame)

    def _init_write_buffer(self) -> None:
        """Initialize a new write buffer."""
        self._current_buffer = VTBlock()
        self._current_buffer.magic = MAGIC_BLK
        self._current_buffer.flags = self._flags
        self._current_buffer.bsize = self.block_size
        self._current_buffer.chans = self.channels
        self._current_buffer.sample_rate = self.sample_rate
        self._current_buffer.srcal = self._srcal
        self._current_buffer.valid = 0
        self._current_buffer.frames = 0
        self._buffer_frames = 0

    def _write_block(self) -> None:
        """Write current buffer as a block."""
        if self._current_buffer is None or self._buffer_frames == 0:
            return

        # Set timestamp
        elapsed = self._nft / (self.sample_rate * self._srcal)
        ts = Timestamp.from_seconds(self._timebase.to_seconds() + elapsed)
        self._current_buffer.secs = ts.secs
        self._current_buffer.nsec = int(ts.frac * 1e9)
        self._current_buffer.frames = self._buffer_frames
        self._current_buffer.valid = 1

        # Write to file
        assert self._fh is not None
        data = self._current_buffer.to_bytes()
        self._fh.write(data)

        self._nft += self._buffer_frames
        self._current_buffer = None
        self._buffer_frames = 0

    def set_timebase(self, timestamp: Timestamp, srcal: float = 1.0) -> None:
        """Set the timebase for the file.

        Args:
            timestamp: Base timestamp
            srcal: Sample rate correction factor
        """
        self._timebase = timestamp
        self._srcal = srcal
        self._nft = 0

    def flush(self) -> None:
        """Flush any buffered data to disk."""
        self._write_block()
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        """Close the file."""
        self.flush()
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> VTFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        """Check if file is closed."""
        return self._fh is None or self._fh.closed

    @property
    def sample_rate_corrected(self) -> float:
        """Get actual sample rate with correction factor."""
        return self.sample_rate * self._srcal


def open_input(path: str | Path) -> VTFile:
    """Open a VT file for reading.

    Args:
        path: Path to VT file

    Returns:
        VTFile handle
    """
    return VTFile(path, mode="r")


def open_output(
    path: str | Path,
    channels: int = 1,
    sample_rate: int = 48000,
    block_size: int = 8192,
    format: str = "float64",
) -> VTFile:
    """Open a VT file for writing.

    Args:
        path: Output file path
        channels: Number of channels
        sample_rate: Sample rate in Hz
        block_size: Frames per block
        format: Data format

    Returns:
        VTFile handle
    """
    return VTFile(
        path,
        mode="w",
        channels=channels,
        sample_rate=sample_rate,
        block_size=block_size,
        format=format,
    )
