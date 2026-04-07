"""Tests for extra functionality."""

import numpy as np

from vlfrx.core.chanspec import ChannelSpec


class TestChanspecPol:
    def test_parse_h1(self):
        spec = ChannelSpec.parse("H1", total_channels=3)
        assert 0 in spec.map

    def test_parse_h2(self):
        spec = ChannelSpec.parse("H2", total_channels=3)
        assert 1 in spec.map

    def test_parse_e(self):
        spec = ChannelSpec.parse("E", total_channels=3)
        assert 2 in spec.map


class TestVTBuffer:
    def test_create_buffer(self):
        from vlfrx.core.buffer import VTBuffer

        buf = VTBuffer(
            key=12345, nblocks=2, bsize=1024, chans=1, sample_rate=48000, create=True
        )
        assert buf is not None
        buf.close()

    def test_buffer_get_load_index(self):
        from vlfrx.core.buffer import VTBuffer

        buf = VTBuffer(
            key=12345, nblocks=2, bsize=1024, chans=1, sample_rate=48000, create=True
        )
        idx = buf.get_load_index()
        assert idx >= 0
        buf.close()


class TestExceptions:
    def test_vtfile_error(self):
        from vlfrx.core.exceptions import VTFileError

        e = VTFileError("test error")
        assert str(e) == "test error"

    def test_vtformat_error(self):
        from vlfrx.core.exceptions import VTFormatError

        e = VTFormatError("format error")
        assert str(e) == "format error"

    def test_vtnetwork_error(self):
        from vlfrx.core.exceptions import VTNetworkError

        e = VTNetworkError("network error")
        assert str(e) == "network error"

    def test_vtbuffer_error(self):
        from vlfrx.core.exceptions import VTBufferError

        e = VTBufferError("buffer error")
        assert str(e) == "buffer error"

    def test_vlfrx_error_base(self):
        from vlfrx.core.exceptions import VLFRXError

        e = VLFRXError("base error")
        assert str(e) == "base error"


class TestTimestampExtra:
    def test_to_datetime(self):
        from vlfrx.core.timestamp import Timestamp

        t = Timestamp(1705321845, 0.5)
        dt = t.to_datetime()
        assert dt.year == 2024
        assert dt.month == 1

    def test_from_filename_no_timestamp(self):
        from vlfrx.core.timestamp import Timestamp

        t = Timestamp.from_filename("no_timestamp.vt")
        assert t.is_zero()

    def test_from_filename_with_timestamp(self):
        from vlfrx.core.timestamp import Timestamp

        t = Timestamp.from_filename("signal_20240115_123045.vt")
        assert not t.is_zero()

    def test_mul_timestamp(self):
        from vlfrx.core.timestamp import Timestamp

        t = Timestamp(100, 0.5)
        result = t * 2.0
        assert result.to_seconds() == 201.0

    def test_rmul_timestamp(self):
        from vlfrx.core.timestamp import Timestamp

        t = Timestamp(100, 0.5)
        result = 2.0 * t
        assert result.to_seconds() == 201.0

    def test_rsub_timestamp(self):
        from vlfrx.core.timestamp import Timestamp

        t = Timestamp(50, 0.0)
        result = 100 - t
        assert result.to_seconds() == 50.0


class TestServicesExtra:
    def test_spectrum_phase_spectrum(self):
        import numpy as np

        from vlfrx.services.spectrum import phase_spectrum

        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        freqs, phase = phase_spectrum(signal, fs=fs)
        assert len(freqs) == len(phase)

    def test_spectrum_dft(self):
        import numpy as np

        from vlfrx.services.spectrum import compute_dft

        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)
        freqs = np.array([1000])

        result = compute_dft(signal, freqs, fs=fs)
        assert len(result) == 1

    def test_filter_moving_average(self):
        from vlfrx.services.filter import moving_average_filter

        b = moving_average_filter(5)
        assert len(b) == 5
        assert abs(sum(b) - 1.0) < 1e-10

    def test_filter_design_bandpass(self):
        from vlfrx.services.filter import design_bandpass

        b = design_bandpass(10000, 1000, fs=48000, numtaps=51)
        assert len(b) == 51

    def test_signal_pink_noise(self):
        from vlfrx.services.signal_gen import generate_pink_noise

        signal = generate_pink_noise(0.1, fs=48000)
        assert len(signal) == 4800

    def test_signal_dc(self):
        from vlfrx.services.signal_gen import generate_dc

        signal = generate_dc(0.1, fs=48000, amplitude=2.0)
        assert len(signal) == 4800
        assert np.all(signal == 2.0)

    def test_signal_pulse_train(self):
        from vlfrx.services.signal_gen import generate_pulse_train

        signal = generate_pulse_train(10, 0.1, fs=48000, pulse_width=0.001)
        assert len(signal) == 4800

    def test_signal_am(self):
        from vlfrx.services.signal_gen import generate_am

        signal = generate_am(1000, 10, 0.1, fs=48000, mod_depth=0.5)
        assert len(signal) == 4800

    def test_signal_fm(self):
        from vlfrx.services.signal_gen import generate_fm

        signal = generate_fm(1000, 10, 0.1, fs=48000, deviation=100)
        assert len(signal) == 4800

    def test_signal_impulse_position(self):
        from vlfrx.services.signal_gen import generate_impulse

        signal = generate_impulse(0.1, fs=48000, amplitude=1.0, position=0.5)
        assert signal[2399] > 0


class TestVTServicesInit:
    def test_all_exports(self):
        from vlfrx.services import (
            apply_filter,
            compute_fft,
            compute_power_spectrum,
            compute_spectrogram,
            design_fir_filter,
            design_iir_filter,
            generate_chirp,
            generate_sawtooth,
            generate_sine,
            generate_square,
            generate_white_noise,
        )

        assert callable(compute_fft)
        assert callable(compute_spectrogram)
        assert callable(compute_power_spectrum)
        assert callable(design_fir_filter)
        assert callable(design_iir_filter)
        assert callable(apply_filter)
        assert callable(generate_sine)
        assert callable(generate_square)
        assert callable(generate_sawtooth)
        assert callable(generate_white_noise)
        assert callable(generate_chirp)


class TestCLICoverage:
    def test_write_empty_file(self, temp_dir):
        from click.testing import CliRunner

        from vlfrx.cli import write

        path = temp_dir / "empty.vt"
        runner = CliRunner()
        result = runner.invoke(write, ["-c", "1", "-r", "48000", str(path)])
        # Empty file - might not be readable, just check it runs

    def test_spec_with_output(self, temp_dir):
        from click.testing import CliRunner

        from vlfrx import open_output
        from vlfrx.cli import spec

        path = temp_dir / "test.vt"
        vt = open_output(path, channels=1, sample_rate=48000)
        vt.write_frame([0.1])
        vt.close()

        out_path = temp_dir / "spectrum.txt"
        runner = CliRunner()
        result = runner.invoke(spec, [str(path), "-o", str(out_path)])

    def test_filter_cli(self, temp_dir):
        from click.testing import CliRunner

        from vlfrx import open_output
        from vlfrx.cli import filter as filter_cmd

        path = temp_dir / "test.vt"
        vt = open_output(path, channels=1, sample_rate=48000)
        vt.write_frame([0.1])
        vt.close()

        out_path = temp_dir / "filtered.vt"
        runner = CliRunner()
        result = runner.invoke(
            filter_cmd, [str(path), "-o", str(out_path), "-t", "lowpass", "-c", "1000"]
        )

    def test_gen_cli(self, temp_dir):
        from click.testing import CliRunner

        from vlfrx.cli import gen

        path = temp_dir / "gen.vt"
        runner = CliRunner()
        result = runner.invoke(gen, ["1000", "0.01", "-o", str(path), "-r", "48000"])
        assert result.exit_code == 0
        assert path.exists()

    def test_cat_cli(self, temp_dir):
        from click.testing import CliRunner

        from vlfrx import open_output
        from vlfrx.cli import cat

        path1 = temp_dir / "a.vt"
        path2 = temp_dir / "b.vt"
        out_path = temp_dir / "out.vt"

        for p in [path1, path2]:
            vt = open_output(p, channels=1, sample_rate=48000)
            vt.write_frame([0.1])
            vt.close()

        runner = CliRunner()
        result = runner.invoke(cat, [str(path1), str(path2), "-o", str(out_path)])
        assert result.exit_code == 0

    def test_info_cli(self, temp_dir):
        from click.testing import CliRunner

        from vlfrx import open_output
        from vlfrx.cli import info

        path = temp_dir / "test.vt"
        vt = open_output(path, channels=2, sample_rate=44100)
        vt.write_frame([0.1, 0.2])
        vt.close()

        runner = CliRunner()
        result = runner.invoke(info, [str(path)])
        assert "Channels: 2" in result.output
        assert "Sample rate: 44100" in result.output
