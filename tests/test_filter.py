"""Tests for filter services."""

import numpy as np
import pytest

from vlfrx.services.filter import (
    apply_filter,
    apply_fir_filter,
    apply_iir_filter,
    design_bandpass,
    design_fir_filter,
    design_iir_filter,
    design_notch,
    moving_average_filter,
)


class TestDesignFIRFilter:
    def test_lowpass_filter(self):
        b = design_fir_filter("lowpass", cutoff=1000, numtaps=101, fs=48000)

        assert len(b) == 101
        # Low frequency components should pass
        # High frequency should be attenuated

    def test_highpass_filter(self):
        b = design_fir_filter("highpass", cutoff=1000, numtaps=101, fs=48000)

        assert len(b) == 101

    def test_bandpass_filter(self):
        b = design_fir_filter("bandpass", cutoff=[1000, 2000], numtaps=101, fs=48000)

        assert len(b) == 101

    def test_invalid_filter_type(self):
        with pytest.raises(ValueError):
            design_fir_filter("invalid", cutoff=1000, fs=48000)


class TestDesignIIRFilter:
    def test_butterworth_lowpass(self):
        b, a = design_iir_filter("butter", cutoff=1000, fs=48000, order=4)

        assert len(b) > 0
        assert len(a) > 0
        assert len(b) == len(a)

    def test_cheby1(self):
        b, a = design_iir_filter("cheby1", cutoff=1000, fs=48000, order=4, rp=0.5)

        assert len(b) == len(a)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            design_iir_filter("invalid", cutoff=1000, fs=48000)


class TestApplyFilter:
    def test_fir_filter(self):
        b = design_fir_filter("lowpass", cutoff=1000, numtaps=101, fs=48000)

        # Generate test signal
        fs = 48000
        t = np.arange(fs) / fs
        signal = np.sin(2 * np.pi * 1000 * t)

        filtered, zf = apply_filter(signal, b)

        assert len(filtered) == len(signal)

    def test_iir_filter(self):
        b, a = design_iir_filter("butter", cutoff=1000, fs=48000, order=4)

        fs = 48000
        signal = np.random.randn(fs)

        filtered, zf = apply_filter(signal, (b, a))

        assert len(filtered) == len(signal)


class TestApplyFIRFilter:
    def test_firfilt(self):
        b = design_fir_filter("lowpass", cutoff=1000, numtaps=101, fs=48000)

        fs = 48000
        signal = np.sin(2 * np.pi * 1000 * np.arange(fs) / fs)

        result = apply_fir_filter(signal, b)

        assert len(result) == len(signal)


class TestApplyIIRFilter:
    def test_iirfilt(self):
        b, a = design_iir_filter("butter", cutoff=1000, fs=48000, order=4)

        fs = 48000
        signal = np.random.randn(fs)

        result = apply_iir_filter(signal, b, a)

        assert len(result) == len(signal)


class TestDesignBandpass:
    def test_bandpass(self):
        b = design_bandpass(center_freq=10000, bandwidth=1000, fs=48000)

        assert len(b) > 0


class TestDesignNotch:
    def test_notch_filter(self):
        b, a = design_notch(freq=1000, fs=48000, quality=30)

        assert len(b) > 0
        assert len(a) > 0


class TestMovingAverage:
    def test_moving_average(self):
        b = moving_average_filter(window_size=10)

        assert len(b) == 10
        assert np.allclose(b, 0.1)
