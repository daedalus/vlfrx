"""Tests for spectrum services."""

import numpy as np
import pytest
from hypothesis import given, settings, Verbosity
from hypothesis import strategies as st

from vlfrx.services.spectrum import (
    compute_dft,
    compute_fft,
    compute_power_spectrum,
    compute_rolling_spectrogram,
    compute_spectrogram,
    phase_spectrum,
)


class TestComputeFFT:
    def test_basic_fft(self):
        fs = 48000
        t = np.arange(fs) / fs
        signal = np.sin(2 * np.pi * 1000 * t)

        freqs, psd = compute_fft(signal, fs=fs, mode="psd")

        peak_idx = np.argmax(psd)
        assert freqs[peak_idx] > 900 and freqs[peak_idx] < 1100

    def test_fft_complex_mode(self):
        fs = 48000
        signal = np.random.randn(fs)

        freqs, spectrum = compute_fft(signal, fs=fs, mode="complex")

        assert len(freqs) == len(spectrum)
        assert np.iscomplexobj(spectrum)

    def test_fft_with_nperseg(self):
        fs = 48000
        signal = np.random.randn(fs)

        freqs, psd = compute_fft(signal, fs=fs, nperseg=1024)

        assert len(freqs) == 513

    def test_fft_different_windows(self):
        fs = 48000
        signal = np.random.randn(fs)

        for window in ["hann", "hamming", "blackman"]:
            freqs, psd = compute_fft(signal, fs=fs, window=window)
            assert len(freqs) == len(psd)

    def test_fft_invalid_mode(self):
        signal = np.random.randn(1000)
        with pytest.raises(ValueError):
            compute_fft(signal, mode="invalid")

    def test_fft_multidimensional_input(self):
        fs = 48000
        signal = np.random.randn(100, 2)

        freqs, psd = compute_fft(signal, fs=fs)
        assert len(freqs) == len(psd)


class TestComputePowerSpectrum:
    def test_power_spectrum_db(self):
        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        freqs, psd_db = compute_power_spectrum(signal, fs=fs)

        assert np.all(psd_db < 100)

    def test_power_spectrum_empty_signal(self):
        signal = np.array([])
        freqs, psd_db = compute_power_spectrum(signal, fs=48000)
        assert len(freqs) == 0

    def test_power_spectrum_single_frequency(self):
        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(4800) / fs)
        freqs, psd_db = compute_power_spectrum(signal, fs=fs)
        peak_idx = np.argmax(psd_db)
        assert 900 < freqs[peak_idx] < 1100


class TestComputeSpectrogram:
    def test_spectrogram_shape(self):
        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(2 * fs) / fs)

        f, t, sxx = compute_spectrogram(signal, fs=fs, nperseg=256)

        assert f.shape[0] == sxx.shape[0]
        assert t.shape[0] == sxx.shape[1]

    def test_spectrogram_2d_input(self):
        fs = 48000
        signal = np.random.randn(10000, 2)

        f, t, sxx = compute_spectrogram(signal, fs=fs, nperseg=256)

        assert f.shape[0] == sxx.shape[0]
        assert t.shape[0] == sxx.shape[1]

    def test_spectrogram_noverlap(self):
        fs = 48000
        signal = np.random.randn(10000)

        f, t, sxx = compute_spectrogram(signal, fs=fs, nperseg=256, noverlap=128)

        assert f.shape[0] == sxx.shape[0]
        assert t.shape[0] == sxx.shape[1]

    def test_spectrogram_different_modes(self):
        fs = 48000
        signal = np.random.randn(10000)

        for mode in ["psd", "complex", "magnitude"]:
            f, t, sxx = compute_spectrogram(signal, fs=fs, mode=mode)
            assert f.shape[0] == sxx.shape[0]


class TestComputeRollingSpectrogram:
    def test_rolling_spectrogram(self):
        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        f, t, sxx = compute_rolling_spectrogram(signal, fs=fs, nperseg=256, hop=128)

        assert f.shape[0] == 129
        assert sxx.shape[0] == 129

    def test_rolling_spectrogram_small_input(self):
        fs = 48000
        signal = np.random.randn(100)

        f, t, sxx = compute_rolling_spectrogram(signal, fs=fs, nperseg=256, hop=128)

        assert sxx.shape[1] <= 1

    def test_rolling_spectrogram_different_hop(self):
        fs = 48000
        signal = np.random.randn(10000)

        f1, t1, sxx1 = compute_rolling_spectrogram(signal, fs=fs, nperseg=256, hop=64)
        f2, t2, sxx2 = compute_rolling_spectrogram(signal, fs=fs, nperseg=256, hop=256)

        assert sxx1.shape[1] > sxx2.shape[1]


class TestPhaseSpectrum:
    def test_phase_spectrum(self):
        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        freqs, phase = phase_spectrum(signal, fs=fs)

        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

    def test_phase_spectrum_range(self):
        fs = 48000
        signal = np.cos(2 * np.pi * 1000 * np.arange(fs) / fs)

        freqs, phase = phase_spectrum(signal, fs=fs)

        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)


class TestComputeDFT:
    def test_dft_single_frequency(self):
        fs = 48000
        signal = generate_sine(freq=1000, duration=0.1, fs=fs)
        target_freqs = np.array([1000.0])

        result = compute_dft(signal, target_freqs, fs=fs)

        assert len(result) == 1
        assert np.iscomplexobj(result)

    def test_dft_multiple_frequencies(self):
        fs = 48000
        signal = generate_sine(freq=1000, duration=0.1, fs=fs)
        target_freqs = np.array([500.0, 1000.0, 1500.0])

        result = compute_dft(signal, target_freqs, fs=fs)

        assert len(result) == 3
        peak_idx = np.argmax(np.abs(result))
        assert target_freqs[peak_idx] == 1000.0

    def test_dft_at_signal_frequency(self):
        fs = 48000
        freq = 1000
        signal = generate_sine(freq=freq, duration=1.0, fs=fs)
        target_freqs = np.array([float(freq)])

        result = compute_dft(signal, target_freqs, fs=fs)

        assert np.abs(result[0]) > np.abs(result[0]) * 0.9

    def test_dft_empty_frequencies(self):
        signal = np.random.randn(1000)
        target_freqs = np.array([])

        result = compute_dft(signal, target_freqs, fs=48000)

        assert len(result) == 0


def generate_sine(
    freq: float,
    duration: float,
    fs: float = 48000,
    phase: float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Helper function to generate sine wave for tests."""
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    return amplitude * np.sin(2 * np.pi * freq * t + phase)
