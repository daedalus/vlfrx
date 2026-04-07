"""Signal generation services."""

from __future__ import annotations

import numpy as np


def generate_sine(
    freq: float,
    duration: float,
    fs: float = 48000,
    phase: float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a sine wave.

    Args:
        freq: Frequency in Hz
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        phase: Initial phase in radians
        amplitude: Amplitude

    Returns:
        Sine wave array

    Example:
        >>> signal = generate_sine(1000, 1.0, fs=48000)
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def generate_sawtooth(
    freq: float,
    duration: float,
    fs: float = 48000,
    width: float = 1.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a sawtooth wave.

    Args:
        freq: Frequency in Hz
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        width: Width of the ramp (0 to 1)
        amplitude: Amplitude

    Returns:
        Sawtooth wave array
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    period = 1.0 / freq
    phase = (t % period) / period
    return amplitude * (2 * phase / width - 1)


def generate_square(
    freq: float,
    duration: float,
    fs: float = 48000,
    duty_cycle: float = 0.5,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a square wave.

    Args:
        freq: Frequency in Hz
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        duty_cycle: Duty cycle (0 to 1)
        amplitude: Amplitude

    Returns:
        Square wave array
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    period = 1.0 / freq
    phase = (t % period) / period
    return amplitude * (phase < duty_cycle).astype(float) * 2 - amplitude


def generate_white_noise(
    duration: float,
    fs: float = 48000,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate white noise.

    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        amplitude: Amplitude (standard deviation)
        seed: Random seed for reproducibility

    Returns:
        White noise array
    """
    n_samples = int(duration * fs)
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(n_samples)


def generate_pink_noise(
    duration: float,
    fs: float = 48000,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate pink noise (1/f noise).

    Uses Voss-McCartney algorithm approximation.

    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        amplitude: Amplitude scaling
        seed: Random seed

    Returns:
        Pink noise array
    """
    n_samples = int(duration * fs)
    rng = np.random.default_rng(seed)

    # Simple approximation using filtered white noise
    white = rng.standard_normal(n_samples)

    # Apply 1/f filter approximation
    b = np.array([0.1, 0.2, 0.3, 0.4])
    a = np.array([1, -0.3, -0.2, -0.1])
    from scipy import signal

    pink: np.ndarray = signal.lfilter(b, a, white)

    result = amplitude * pink / np.max(np.abs(pink))
    return np.asarray(result, dtype=np.float64)


def generate_chirp(
    start_freq: float,
    end_freq: float,
    duration: float,
    fs: float = 48000,
    method: str = "linear",
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a frequency sweep (chirp) signal.

    Args:
        start_freq: Starting frequency in Hz
        end_freq: Ending frequency in Hz
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        method: 'linear', 'quadratic', 'logarithmic', 'hyperbolic'
        amplitude: Amplitude

    Returns:
        Chirp signal array

    Example:
        >>> # Linear sweep from 100 Hz to 1000 Hz over 1 second
        >>> signal = generate_chirp(100, 1000, 1.0, fs=48000)
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    if method == "linear":
        freq = start_freq + (end_freq - start_freq) * t / duration
        phase = 2 * np.pi * np.cumsum(freq) / fs

    elif method == "quadratic":
        a = (end_freq - start_freq) / duration**2
        freq = start_freq + a * t**2
        phase = 2 * np.pi * np.cumsum(freq) / fs

    elif method == "logarithmic":
        freq = start_freq * (end_freq / start_freq) ** (t / duration)
        phase = 2 * np.pi * np.cumsum(freq) / fs

    elif method == "hyperbolic":
        freq = (
            start_freq * end_freq / (end_freq - (end_freq - start_freq) * t / duration)
        )
        phase = 2 * np.pi * np.cumsum(freq) / fs

    else:
        raise ValueError(f"Unknown method: {method}")

    return amplitude * np.sin(phase)


def generate_impulse(
    duration: float,
    fs: float = 48000,
    amplitude: float = 1.0,
    position: float = 0.0,
) -> np.ndarray:
    """Generate an impulse signal.

    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        amplitude: Impulse amplitude
        position: Position as fraction of duration (0 to 1)

    Returns:
        Impulse signal array
    """
    n_samples = int(duration * fs)
    signal = np.zeros(n_samples)
    idx = int(position * n_samples)
    if idx < n_samples:
        signal[idx] = amplitude
    return signal


def generate_dc(
    duration: float,
    fs: float = 48000,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a DC (constant) signal.

    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        amplitude: DC value

    Returns:
        DC signal array
    """
    n_samples = int(duration * fs)
    return np.full(n_samples, amplitude)


def generate_pulse_train(
    freq: float,
    duration: float,
    fs: float = 48000,
    pulse_width: float = 0.001,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a pulse train.

    Args:
        freq: Pulse repetition frequency in Hz
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        pulse_width: Width of each pulse in seconds
        amplitude: Pulse amplitude

    Returns:
        Pulse train array
    """
    n_samples = int(duration * fs)
    period = 1.0 / freq

    signal = np.zeros(n_samples)
    for i in range(int(duration * freq) + 1):
        start = int(i * period * fs)
        end = start + int(pulse_width * fs)
        if end < n_samples:
            signal[start:end] = amplitude

    return signal


def generate_am(
    carrier_freq: float,
    mod_freq: float,
    duration: float,
    fs: float = 48000,
    mod_depth: float = 0.5,
    carrier_amplitude: float = 1.0,
) -> np.ndarray:
    """Generate AM (amplitude modulated) signal.

    Args:
        carrier_freq: Carrier frequency in Hz
        mod_freq: Modulation frequency in Hz
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        mod_depth: Modulation depth (0 to 1)
        carrier_amplitude: Carrier amplitude

    Returns:
        AM signal array
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    carrier = carrier_amplitude * np.sin(2 * np.pi * carrier_freq * t)
    mod = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)

    return carrier * mod


def generate_fm(
    carrier_freq: float,
    mod_freq: float,
    duration: float,
    fs: float = 48000,
    deviation: float = 1000.0,
    carrier_amplitude: float = 1.0,
) -> np.ndarray:
    """Generate FM (frequency modulated) signal.

    Args:
        carrier_freq: Carrier frequency in Hz
        mod_freq: Modulation frequency in Hz
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        deviation: Frequency deviation in Hz
        carrier_amplitude: Carrier amplitude

    Returns:
        FM signal array
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    mod = mod_freq * np.sin(2 * np.pi * mod_freq * t)
    phase = 2 * np.pi * np.cumsum(carrier_freq + deviation * mod) / fs

    return carrier_amplitude * np.sin(phase)


def generate_burst(
    freq: float,
    duration: float,
    fs: float = 48000,
    burst_duration: float = 0.01,
    n_bursts: int = 5,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a burst of sine wave pulses.

    Args:
        freq: Frequency in Hz
        duration: Total duration in seconds
        fs: Sampling frequency in Hz
        burst_duration: Duration of each burst
        n_bursts: Number of bursts
        amplitude: Amplitude

    Returns:
        Burst signal array
    """
    n_samples = int(duration * fs)
    signal = np.zeros(n_samples)

    burst_samples = int(burst_duration * fs)
    burst_wave = amplitude * np.sin(2 * np.pi * freq * np.arange(burst_samples) / fs)

    burst_period = duration / n_bursts
    for i in range(n_bursts):
        start = int(i * burst_period * fs)
        end = min(start + burst_samples, n_samples)
        signal[start:end] = burst_wave[: end - start]

    return signal
