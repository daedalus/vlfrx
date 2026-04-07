"""Tests for CLI commands."""

import numpy as np
from click.testing import CliRunner


class TestCLI:
    def test_main_help(self):
        from vlfrx.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "vlfrx" in result.output

    def test_info_command(self, temp_dir):
        from vlfrx import open_output
        from vlfrx.cli import info

        path = temp_dir / "test.vt"
        vt = open_output(path, channels=2, sample_rate=48000)
        vt.write_frame([0.1, 0.2])
        vt.close()

        runner = CliRunner()
        result = runner.invoke(info, [str(path)])
        assert result.exit_code == 0
        assert "Channels: 2" in result.output
        assert "Sample rate: 48000" in result.output

    def test_gen_command(self, temp_dir):
        from vlfrx.cli import gen

        path = temp_dir / "sine.vt"
        runner = CliRunner()
        result = runner.invoke(gen, ["1000", "0.1", "-o", str(path), "-r", "48000"])

        assert result.exit_code == 0
        assert path.exists()

        # Verify file content
        from vlfrx import open_input

        vt = open_input(path)
        frame = vt.read_frame()
        assert frame is not None
        vt.close()

    def test_read_command(self, temp_dir):
        from vlfrx import open_output
        from vlfrx.cli import read

        path = temp_dir / "test.vt"
        vt = open_output(path, channels=1, sample_rate=48000)
        vt.write_frame([0.1])
        vt.close()

        runner = CliRunner()
        result = runner.invoke(read, [str(path), "-n", "1"])
        assert result.exit_code == 0

    def test_cat_command(self, temp_dir):
        from vlfrx.cli import cat

        path1 = temp_dir / "a.vt"
        path2 = temp_dir / "b.vt"
        output = temp_dir / "c.vt"

        from vlfrx import open_output

        for p in [path1, path2]:
            vt = open_output(p, channels=1, sample_rate=48000)
            vt.write_frame([0.1])
            vt.close()

        runner = CliRunner()
        result = runner.invoke(cat, [str(path1), str(path2), "-o", str(output)])

        assert result.exit_code == 0
        assert output.exists()


class TestChanspec:
    def test_parse_single_channel(self):
        from vlfrx import parse_chanspec

        spec = parse_chanspec("0", total_channels=4)
        assert spec.map == [0]

    def test_parse_range(self):
        from vlfrx import parse_chanspec

        spec = parse_chanspec("0-2", total_channels=4)
        assert spec.map == [0, 1, 2]

    def test_parse_list(self):
        from vlfrx import parse_chanspec

        spec = parse_chanspec("0,2", total_channels=4)
        assert spec.map == [0, 2]

    def test_parse_empty(self):
        from vlfrx import parse_chanspec

        spec = parse_chanspec("", total_channels=4)
        assert spec.map == [0, 1, 2, 3]

    def test_channel_spec_length(self):
        from vlfrx import parse_chanspec

        spec = parse_chanspec("0,1", total_channels=2)
        assert len(spec) == 2

    def test_channel_spec_iterate(self):
        from vlfrx import parse_chanspec

        spec = parse_chanspec("0-1", total_channels=2)
        assert list(spec) == [0, 1]


class TestServices:
    def test_rolling_spectrogram_output_shape(self):
        import numpy as np

        from vlfrx.services.spectrum import compute_rolling_spectrogram

        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        f, t, sxx = compute_rolling_spectrogram(signal, fs=fs, nperseg=256, hop=128)

        assert sxx.shape[0] == 129  # nperseg/2 + 1

    def test_filter_with_zi(self):
        import numpy as np

        from vlfrx.services.filter import apply_filter, design_fir_filter

        b = design_fir_filter("lowpass", 1000, fs=48000)
        signal = np.random.randn(1000)

        # Test with initial state
        zi = np.zeros(len(b) - 1)
        output, zf = apply_filter(signal, b, zi=zi)

        assert len(output) == len(signal)

    def test_apply_iir_filter_directly(self):
        import numpy as np

        from vlfrx.services.filter import apply_iir_filter, design_iir_filter

        b, a = design_iir_filter("butter", 1000, fs=48000, order=4)
        signal = np.random.randn(1000)

        output = apply_iir_filter(signal, b, a)
        assert len(output) == len(signal)

    def test_design_notch(self):
        from vlfrx.services.filter import design_notch

        b, a = design_notch(freq=1000, fs=48000, quality=30)

        assert len(b) > 0
        assert len(a) > 0

    def test_generate_with_seed(self):
        from vlfrx.services.signal_gen import generate_white_noise

        s1 = generate_white_noise(0.1, fs=48000, seed=42)
        s2 = generate_white_noise(0.1, fs=48000, seed=42)

        np.testing.assert_array_equal(s1, s2)

    def test_generate_burst(self):
        from vlfrx.services.signal_gen import generate_burst

        signal = generate_burst(1000, 0.1, fs=48000, burst_duration=0.01, n_bursts=3)

        assert len(signal) == 4800
