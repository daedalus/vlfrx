"""Lock-free buffer support for shared memory IPC."""

from __future__ import annotations

import struct
from ctypes import Structure, c_int32, c_uint32

import numpy as np

from vlfrx.core.exceptions import VTBufferError
from vlfrx.core.vtfile import MAGIC_BUF, VTBlock


class VTBufferHeader(Structure):
    """Shared memory buffer header structure."""

    _fields_ = [
        ("magic", c_int32),
        ("flags", c_uint32),
        ("nblocks", c_uint32),
        ("bsize", c_uint32),
        ("chans", c_uint32),
        ("semkey", c_int32),
        ("load", c_uint32),
        ("sample_rate", c_uint32),
    ]


class VTBuffer:
    """Lock-free buffer for inter-process signal data sharing.

    Uses shared memory to allow zero-copy data transfer between
    processes for real-time signal processing.
    """

    def __init__(
        self,
        key: int,
        nblocks: int = 4,
        bsize: int = 8192,
        chans: int = 1,
        sample_rate: int = 48000,
        create: bool = False,
    ) -> None:
        """Create or attach to a shared memory buffer.

        Args:
            key: Shared memory key
            nblocks: Number of blocks in buffer
            bsize: Frames per block
            chans: Number of channels
            sample_rate: Sample rate in Hz
            create: If True, create new buffer; if False, attach to existing
        """
        self.key = key
        self.nblocks = nblocks
        self.bsize = bsize
        self.chans = chans
        self.sample_rate = sample_rate
        self.create = create

        self._shm: np.ndarray | None = None
        self._header: VTBufferHeader | None = None
        self._load_index: int = 0
        self._block_size_bytes: int = 0

        if create:
            self._create_buffer()
        else:
            self._attach_buffer()

    def _create_buffer(self) -> None:
        """Create a new shared memory buffer."""
        try:
            # Calculate total size
            header_size = struct.calcsize("iiIIIIII")
            itemsize = 8  # float64
            data_size = self.bsize * self.chans * itemsize
            block_size = header_size + data_size
            self._block_size_bytes = block_size

            total_size = header_size + self.nblocks * block_size

            # Create anonymous mmap
            self._shm = np.zeros(total_size, dtype=np.uint8)

            # Initialize header
            header = np.frombuffer(self._shm[:header_size], dtype=np.int32)
            header[0] = MAGIC_BUF  # magic
            header[1] = 0  # flags
            header[2] = self.nblocks
            header[3] = self.bsize
            header[4] = self.chans
            header[5] = 0  # semkey
            header[6] = 0  # load
            header[7] = self.sample_rate

        except Exception as e:
            raise VTBufferError(f"Failed to create buffer: {e}")

    def _attach_buffer(self) -> None:
        """Attach to an existing shared memory buffer."""
        # For now, raise not implemented
        raise NotImplementedError("Attaching to existing buffers not yet implemented")

    def get_load_index(self) -> int:
        """Get the current load index (most recent block)."""
        if self._shm is None:
            return 0
        header = np.frombuffer(self._shm[:32], dtype=np.int32)
        return int(header[6])

    def get_block(self, index: int) -> VTBlock | None:
        """Get a block from the buffer.

        Args:
            index: Block index (relative to buffer)

        Returns:
            VTBlock or None if invalid
        """
        if self._shm is None:
            return None

        header_size = 32
        block_size = self._block_size_bytes
        offset = header_size + index * block_size

        if offset + block_size > len(self._shm):
            return None

        block_data = self._shm[offset : offset + block_size]
        return VTBlock.from_bytes(block_data.tobytes(), self.chans, int(self._shm[4]))

    def close(self) -> None:
        """Close and unmap the buffer."""
        self._shm = None
        self._header = None

    def __enter__(self) -> VTBuffer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
