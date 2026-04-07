"""Tests for signal generation services."""

import numpy as np
import pytest

from vlfrx.services.signal_gen import (
    generate_am,
    generate_burst,
    generate_chirp,
    generate_dc,
    generate_fm,
    generate_impulse,
    generate_pink_noise,
    generate_pulse_train,
    generate_sawtooth,
    generate_sine,
    generate_square,
    generate_white_noise,
)


class TestGenerateSine:
    def test_basic_sine(self):
        signal = generate_sine(freq=1000, duration=0.1, fs=48000)

        assert len(signal) == 4800
        assert signal.max() <= 1.0
        assert signal.min() >= -1.0

    def test_with_phase(self):
        signal = generate_sine(freq=1000, duration=0.1, fs=48000, phase=np.pi / 2)

        assert len(signal) == 4800

    def test_with_amplitude(self):
        signal = generate_sine(freq=1000, duration=0.1, fs=48000, amplitude=0.5)

        assert signal.max() <= 0.5
        assert signal.min() >= -0.5


class TestGenerateSquare:
    def test_square_wave(self):
        signal = generate_square(freq=100, duration=0.1, fs=48000)

        assert len(signal) == 4800
        # Square wave should have only two levels
        unique = np.unique(signal)
        assert len(unique) <= 2

    def test_duty_cycle(self):
        signal = generate_square(freq=100, duration=0.1, fs=48000, duty_cycle=0.25)

        assert len(signal) == 4800


class TestGenerateSawtooth:
    def test_sawtooth_wave(self):
        signal = generate_sawtooth(freq=100, duration=0.1, fs=48000)

        assert len(signal) == 4800


class TestGenerateWhiteNoise:
    def test_white_noise_shape(self):
        signal = generate_white_noise(duration=0.1, fs=48000)

        assert len(signal) == 4800

    def test_white_noise_reproducibility(self):
        signal1 = generate_white_noise(duration=0.1, fs=48000, seed=42)
        signal2 = generate_white_noise(duration=0.1, fs=48000, seed=42)

        np.testing.assert_array_equal(signal1, signal2)


class TestGeneratePinkNoise:
    def test_pink_noise(self):
        signal = generate_pink_noise(duration=0.1, fs=48000)

        assert len(signal) == 4800


class TestGenerateChirp:
    def test_linear_chirp(self):
        signal = generate_chirp(start_freq=100, end_freq=1000, duration=0.1, fs=48000)

        assert len(signal) == 4800

    def test_quadratic_chirp(self):
        signal = generate_chirp(
            start_freq=100, end_freq=1000, duration=0.1, fs=48000, method="quadratic"
        )

        assert len(signal) == 4800

    def test_logarithmic_chirp(self):
        signal = generate_chirp(
            start_freq=100, end_freq=1000, duration=0.1, fs=48000, method="logarithmic"
        )

        assert len(signal) == 4800

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            generate_chirp(
                start_freq=100, end_freq=1000, duration=0.1, fs=48000, method="invalid"
            )


class TestGenerateImpulse:
    def test_impulse(self):
        signal = generate_impulse(duration=0.1, fs=48000)

        assert len(signal) == 4800
        assert np.sum(signal) > 0  # Should have some non-zero value

    def test_impulse_position(self):
        signal = generate_impulse(duration=0.1, fs=48000, position=0.5)

        assert len(signal) == 4800


class TestGenerateDC:
    def test_dc_signal(self):
        signal = generate_dc(duration=0.1, fs=48000, amplitude=2.5)

        assert len(signal) == 4800
        assert np.allclose(signal, 2.5)


class TestGeneratePulseTrain:
    def test_pulse_train(self):
        signal = generate_pulse_train(
            freq=10, duration=0.1, fs=48000, pulse_width=0.001
        )

        assert len(signal) == 4800


class TestGenerateAM:
    def test_am_signal(self):
        signal = generate_am(carrier_freq=1000, mod_freq=10, duration=0.1, fs=48000)

        assert len(signal) == 4800


class TestGenerateFM:
    def test_fm_signal(self):
        signal = generate_fm(
            carrier_freq=1000, mod_freq=10, duration=0.1, fs=48000, deviation=100
        )

        assert len(signal) == 4800


class TestGenerateBurst:
    def test_burst_signal(self):
        signal = generate_burst(
            freq=1000, duration=0.1, fs=48000, burst_duration=0.01, n_bursts=3
        )

        assert len(signal) == 4800
