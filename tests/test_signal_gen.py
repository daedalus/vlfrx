"""Tests for signal generation services."""

import numpy as np
import pytest
from hypothesis import given, settings, Verbosity
from hypothesis import strategies as st

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

    @given(
        freq=st.floats(min_value=1, max_value=20000),
        duration=st.floats(min_value=0.01, max_value=1.0),
        fs=st.sampled_from([8000, 16000, 48000, 96000]),
        phase=st.floats(min_value=0, max_value=2 * np.pi),
        amplitude=st.floats(min_value=0.1, max_value=2.0),
    )
    @settings(verbosity=Verbosity.verbose)
    def test_sine_properties(self, freq, duration, fs, phase, amplitude):
        signal = generate_sine(
            freq=freq, duration=duration, fs=fs, phase=phase, amplitude=amplitude
        )
        n_expected = int(duration * fs)
        assert len(signal) == n_expected
        assert signal.max() <= amplitude * 1.01
        assert signal.min() >= -amplitude * 1.01

    def test_sine_zero_frequency(self):
        signal = generate_sine(freq=0, duration=0.1, fs=48000)
        assert np.allclose(signal, 0)

    def test_sine_single_sample(self):
        fs = 48000
        signal = generate_sine(freq=1000, duration=1 / fs, fs=fs)
        assert len(signal) == 1


class TestGenerateSquare:
    def test_square_wave(self):
        signal = generate_square(freq=100, duration=0.1, fs=48000)

        assert len(signal) == 4800
        unique = np.unique(signal)
        assert len(unique) <= 2

    def test_duty_cycle(self):
        signal = generate_square(freq=100, duration=0.1, fs=48000, duty_cycle=0.25)

        assert len(signal) == 4800

    def test_duty_cycle_75(self):
        signal = generate_square(freq=100, duration=0.1, fs=48000, duty_cycle=0.75)
        assert len(signal) == 4800

    def test_zero_duty_cycle(self):
        signal = generate_square(freq=100, duration=0.1, fs=48000, duty_cycle=0.0)
        assert np.all(signal <= 0)

    def test_full_duty_cycle(self):
        signal = generate_square(freq=100, duration=0.1, fs=48000, duty_cycle=1.0)
        assert np.all(signal >= 0)


class TestGenerateSawtooth:
    def test_sawtooth_wave(self):
        signal = generate_sawtooth(freq=100, duration=0.1, fs=48000)

        assert len(signal) == 4800

    def test_sawtooth_width(self):
        signal = generate_sawtooth(freq=100, duration=0.1, fs=48000, width=0.5)
        assert len(signal) == 4800

    def test_sawtooth_zero_width(self):
        signal = generate_sawtooth(freq=100, duration=0.1, fs=48000, width=0.0)
        assert len(signal) == 4800


class TestGenerateWhiteNoise:
    def test_white_noise_shape(self):
        signal = generate_white_noise(duration=0.1, fs=48000)

        assert len(signal) == 4800

    def test_white_noise_reproducibility(self):
        signal1 = generate_white_noise(duration=0.1, fs=48000, seed=42)
        signal2 = generate_white_noise(duration=0.1, fs=48000, seed=42)

        np.testing.assert_array_equal(signal1, signal2)

    def test_white_noise_different_seeds(self):
        signal1 = generate_white_noise(duration=0.1, fs=48000, seed=42)
        signal2 = generate_white_noise(duration=0.1, fs=48000, seed=43)

        assert not np.array_equal(signal1, signal2)

    def test_white_noise_zero_duration(self):
        signal = generate_white_noise(duration=0, fs=48000)
        assert len(signal) == 0

    def test_white_noise_custom_amplitude(self):
        signal = generate_white_noise(duration=0.1, fs=48000, amplitude=2.0)
        std = np.std(signal)
        assert 1.8 < std < 2.2


class TestGeneratePinkNoise:
    def test_pink_noise(self):
        signal = generate_pink_noise(duration=0.1, fs=48000)

        assert len(signal) == 4800

    def test_pink_noise_reproducibility(self):
        signal1 = generate_pink_noise(duration=0.1, fs=48000, seed=42)
        signal2 = generate_pink_noise(duration=0.1, fs=48000, seed=42)

        np.testing.assert_array_equal(signal1, signal2)


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

    def test_chirp_exponential(self):
        signal = generate_chirp(
            start_freq=100, end_freq=1000, duration=0.1, fs=48000, method="logarithmic"
        )
        assert len(signal) == 4800

    def test_chirp_same_start_end(self):
        signal = generate_chirp(start_freq=1000, end_freq=1000, duration=0.1, fs=48000)
        assert len(signal) == 4800


class TestGenerateImpulse:
    def test_impulse(self):
        signal = generate_impulse(duration=0.1, fs=48000)

        assert len(signal) == 4800
        assert np.sum(signal) > 0

    def test_impulse_position(self):
        signal = generate_impulse(duration=0.1, fs=48000, position=0.5)

        assert len(signal) == 4800

    def test_impulse_position_zero(self):
        signal = generate_impulse(duration=0.1, fs=48000, position=0.0)
        max_idx = np.argmax(signal)
        assert max_idx < 100

    def test_impulse_position_one(self):
        signal = generate_impulse(duration=0.1, fs=48000, position=1.0)
        max_idx = np.argmax(signal)
        assert len(signal) - max_idx < 100


class TestGenerateDC:
    def test_dc_signal(self):
        signal = generate_dc(duration=0.1, fs=48000, amplitude=2.5)

        assert len(signal) == 4800
        assert np.allclose(signal, 2.5)

    def test_dc_zero_amplitude(self):
        signal = generate_dc(duration=0.1, fs=48000, amplitude=0.0)
        assert np.allclose(signal, 0)


class TestGeneratePulseTrain:
    def test_pulse_train(self):
        signal = generate_pulse_train(
            freq=10, duration=0.1, fs=48000, pulse_width=0.001
        )

        assert len(signal) == 4800

    def test_pulse_train_narrow(self):
        signal = generate_pulse_train(
            freq=10, duration=0.1, fs=48000, pulse_width=0.0001
        )
        assert len(signal) == 4800

    def test_pulse_train_wide(self):
        signal = generate_pulse_train(freq=10, duration=0.1, fs=48000, pulse_width=0.05)
        assert len(signal) == 4800


class TestGenerateAM:
    def test_am_signal(self):
        signal = generate_am(carrier_freq=1000, mod_freq=10, duration=0.1, fs=48000)

        assert len(signal) == 4800

    def test_am_zero_mod(self):
        signal = generate_am(carrier_freq=1000, mod_freq=0, duration=0.1, fs=48000)
        assert len(signal) == 4800


class TestGenerateFM:
    def test_fm_signal(self):
        signal = generate_fm(
            carrier_freq=1000, mod_freq=10, duration=0.1, fs=48000, deviation=100
        )

        assert len(signal) == 4800

    def test_fm_zero_deviation(self):
        signal = generate_fm(
            carrier_freq=1000, mod_freq=10, duration=0.1, fs=48000, deviation=0
        )
        expected = generate_sine(freq=1000, duration=0.1, fs=48000)
        np.testing.assert_array_almost_equal(signal, expected)


class TestGenerateBurst:
    def test_burst_signal(self):
        signal = generate_burst(
            freq=1000, duration=0.1, fs=48000, burst_duration=0.01, n_bursts=3
        )

        assert len(signal) == 4800

    def test_burst_single(self):
        signal = generate_burst(
            freq=1000, duration=0.1, fs=48000, burst_duration=0.01, n_bursts=1
        )
        assert len(signal) == 4800

    def test_burst_many(self):
        signal = generate_burst(
            freq=1000, duration=0.1, fs=48000, burst_duration=0.005, n_bursts=10
        )
        assert len(signal) == 4800
