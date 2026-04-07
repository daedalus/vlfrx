"""Tests for VT file module."""

import numpy as np
import pytest

from vlfrx.core.exceptions import VTFileError
from vlfrx.core.vtfile import VTBlock, VTFile, open_input, open_output


class TestVTBlock:
    def test_create_block(self):
        block = VTBlock()
        assert block.magic == 0
        assert block.bsize == 0

    def test_block_to_bytes(self):
        block = VTBlock()
        block.magic = 27859
        block.flags = 0
        block.bsize = 8192
        block.chans = 2
        block.sample_rate = 48000
        block.secs = 1000
        block.nsec = 500000000
        block.valid = 1
        block.frames = 100
        block.spare = 0
        block.srcal = 1.0
        block.data = np.random.randn(100, 2)

        data = block.to_bytes()
        assert len(data) > 0


class TestVTFile:
    def test_create_write_file(self, temp_dir):
        path = temp_dir / "test.vt"
        vt = open_output(path, channels=2, sample_rate=48000)
        vt.close()

        assert path.exists()

    def test_write_read_roundtrip(self, temp_dir):
        path = temp_dir / "roundtrip.vt"

        # Write some frames
        vt_out = open_output(path, channels=2, sample_rate=48000)
        frames = np.random.randn(100, 2)
        vt_out.write_frames(frames)
        vt_out.close()

        # Read back
        vt_in = open_input(path)
        read_frames = []
        while True:
            frame = vt_in.read_frame()
            if frame is None:
                break
            read_frames.append(frame)

        vt_in.close()

        assert len(read_frames) == 100
        np.testing.assert_array_almost_equal(read_frames, frames, decimal=6)

    def test_open_nonexistent_file(self, temp_dir):
        path = temp_dir / "nonexistent.vt"
        with pytest.raises(VTFileError):
            open_input(path)

    def test_write_single_frame(self, temp_dir):
        path = temp_dir / "single.vt"
        vt = open_output(path, channels=1, sample_rate=48000)
        vt.write_frame([0.5])
        vt.close()

        vt = open_input(path)
        frame = vt.read_frame()
        assert frame is not None
        assert abs(frame[0] - 0.5) < 0.01

    def test_set_timebase(self, temp_dir):
        from vlfrx.core.timestamp import Timestamp

        path = temp_dir / "timebase.vt"
        vt = open_output(path, channels=1, sample_rate=48000)
        vt.set_timebase(Timestamp(1000, 0), 1.0)
        vt.write_frame([0.5])
        vt.close()

    def test_context_manager(self, temp_dir):
        path = temp_dir / "context.vt"
        with open_output(path, channels=1, sample_rate=48000) as vt:
            vt.write_frame([1.0])

        with open_input(path) as vt:
            frame = vt.read_frame()
            assert frame is not None

    def test_read_all_frames(self, temp_dir):
        path = temp_dir / "readall.vt"
        n_frames = 500
        n_channels = 2

        vt_out = open_output(path, channels=n_channels, sample_rate=48000)
        frames = np.random.randn(n_frames, n_channels)
        vt_out.write_frames(frames)
        vt_out.close()

        vt_in = open_input(path)
        all_frames = []
        while True:
            frame = vt_in.read_frame()
            if frame is None:
                break
            all_frames.append(frame)

        vt_in.close()

        assert len(all_frames) == n_frames

    def test_file_properties(self, temp_dir):
        path = temp_dir / "props.vt"
        vt = open_output(path, channels=2, sample_rate=44100)
        assert vt.channels == 2
        assert vt.sample_rate == 44100
        assert not vt.closed
        vt.close()
        assert vt.closed

    def test_different_formats(self, temp_dir):
        for fmt in ["float64", "float32", "int16"]:
            path = temp_dir / f"format_{fmt}.vt"
            vt = open_output(path, channels=1, sample_rate=48000, format=fmt)
            vt.write_frame([0.5])
            vt.close()

            vt = open_input(path)
            frame = vt.read_frame()
            assert frame is not None


class TestVTFileErrors:
    def test_invalid_mode(self, temp_dir):
        path = temp_dir / "test.vt"
        with pytest.raises(ValueError):
            VTFile(path, mode="x")


class TestOpenInputOutput:
    def test_open_input_pathlib(self, temp_dir):
        path = temp_dir / "pathlib.vt"
        vt_out = open_output(path, channels=1, sample_rate=48000)
        vt_out.write_frame([0.0])  # Write at least one frame
        vt_out.close()

        vt_in = open_input(path)
        vt_in.close()

    def test_open_output_with_options(self, temp_dir):
        path = temp_dir / "options.vt"
        vt = open_output(
            path,
            channels=4,
            sample_rate=96000,
            block_size=4096,
            format="float32",
        )
        assert vt.channels == 4
        assert vt.sample_rate == 96000
        assert vt.block_size == 4096
        vt.close()
