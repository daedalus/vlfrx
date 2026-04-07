"""Tests for spectrum services."""

import numpy as np

from vlfrx.services.spectrum import (
    compute_fft,
    compute_power_spectrum,
    compute_rolling_spectrogram,
    compute_spectrogram,
    phase_spectrum,
)


class TestComputeFFT:
    def test_basic_fft(self):
        # Generate a 1 kHz sine wave
        fs = 48000
        t = np.arange(fs) / fs
        signal = np.sin(2 * np.pi * 1000 * t)

        freqs, psd = compute_fft(signal, fs=fs, mode="psd")

        # Check that peak is around 1 kHz
        peak_idx = np.argmax(psd)
        assert freqs[peak_idx] > 900 and freqs[peak_idx] < 1100

    def test_fft_complex_mode(self):
        fs = 48000
        signal = np.random.randn(fs)

        freqs, spectrum = compute_fft(signal, fs=fs, mode="complex")

        assert len(freqs) == len(spectrum)
        assert np.iscomplexobj(spectrum)


class TestComputePowerSpectrum:
    def test_power_spectrum_db(self):
        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        freqs, psd_db = compute_power_spectrum(signal, fs=fs)

        # Should return values in dB (typically negative for noise floor)
        assert np.all(psd_db < 100)  # Reasonable range


class TestComputeSpectrogram:
    def test_spectrogram_shape(self):
        fs = 48000
        # 2 seconds of signal
        signal = np.sin(2 * np.pi * 1000 * np.arange(2 * fs) / fs)

        f, t, sxx = compute_spectrogram(signal, fs=fs, nperseg=256)

        assert f.shape[0] == sxx.shape[0]  # freq bins
        assert t.shape[0] == sxx.shape[1]  # time bins


class TestComputeRollingSpectrogram:
    def test_rolling_spectrogram(self):
        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        f, t, sxx = compute_rolling_spectrogram(signal, fs=fs, nperseg=256, hop=128)

        assert f.shape[0] == 129  # nperseg/2 + 1
        assert sxx.shape[0] == 129


class TestPhaseSpectrum:
    def test_phase_spectrum(self):
        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        freqs, phase = phase_spectrum(signal, fs=fs)

        # Phase should be in range [-pi, pi]
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)
