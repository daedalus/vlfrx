"""Spectrum analysis services."""

from __future__ import annotations

import numpy as np
from scipy import fft as scipy_fft
from scipy import signal


def compute_fft(
    data: np.ndarray,
    nperseg: int | None = None,
    fs: float = 1.0,
    window: str = "hann",
    noverlap: int = 0,
    mode: str = "psd",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT of a signal.

    Args:
        data: Input signal (1D array)
        nperseg: Length of each segment (for segment-based FFT)
        fs: Sampling frequency in Hz
        window: Window function ('hann', 'hamming', 'blackman', etc.)
        noverlap: Number of points to overlap between segments
        mode: Output mode ('psd' for power spectral density, 'complex' for complex FFT)

    Returns:
        Tuple of (frequencies, spectrum)

    Example:
        >>> fs = 48000
        >>> t = np.arange(fs) / fs
        >>> signal = np.sin(2 * np.pi * 1000 * t)
        >>> freqs, psd = compute_fft(signal, fs=fs)
    """
    if data.ndim > 1:
        data = data.flatten()

    if nperseg is None:
        nperseg = len(data)

    if mode == "psd":
        freqs, psd = signal.welch(
            data,
            fs=fs,
            nperseg=nperseg,
            window=window,
            noverlap=noverlap,
            scaling="density",
        )
        return freqs, psd
    elif mode == "complex":
        # Compute single FFT
        win = signal.get_window(window, nperseg)
        if len(data) < nperseg:
            # Pad with zeros
            padded = np.zeros(nperseg)
            padded[: len(data)] = data
            data = padded
        elif len(data) > nperseg:
            data = data[:nperseg]
        result = scipy_fft.fft(win * data)
        freqs = scipy_fft.fftfreq(nperseg, 1 / fs)
        return freqs[: len(result) // 2], result[: len(result) // 2]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_power_spectrum(
    data: np.ndarray,
    nperseg: int | None = None,
    fs: float = 1.0,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum (power spectral density).

    Args:
        data: Input signal
        nperseg: FFT length
        fs: Sampling frequency
        window: Window function

    Returns:
        Tuple of (frequencies, power in dB)
    """
    freqs, psd = compute_fft(data, nperseg=nperseg, fs=fs, window=window, mode="psd")

    # Convert to dB
    with np.errstate(divide="ignore"):
        psd_db = 10 * np.log10(psd + 1e-12)

    return freqs, psd_db


def compute_spectrogram(
    data: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    mode: str = "psd",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram of a signal.

    Args:
        data: Input signal (1D or 2D, will process each channel)
        fs: Sampling frequency in Hz
        nperseg: Length of each segment
        noverlap: Number of overlapping points (default: nperseg // 2)
        window: Window function
        mode: Output mode ('psd', 'complex', 'magnitude')

    Returns:
        Tuple of (frequencies, times, spectrogram)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    if data.ndim == 1:
        f, t, sxx = signal.spectrogram(
            data,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            mode=mode,
        )
    else:
        # Process first channel for now
        f, t, sxx = signal.spectrogram(
            data[:, 0],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            mode=mode,
        )

    return f, t, sxx


def compute_rolling_spectrogram(
    data: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    hop: int = 128,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling spectrogram (like vtrsgram).

    Similar to spectrogram but more suited for real-time display.

    Args:
        data: Input signal
        fs: Sampling frequency
        nperseg: FFT size
        hop: Hop size between successive spectra
        window: Window function

    Returns:
        Tuple of (frequencies, times, spectrogram)
    """
    if data.ndim > 1:
        data = data[:, 0]  # Use first channel

    win = signal.get_window(window, nperseg)

    # Calculate number of frames
    n_frames = (len(data) - nperseg) // hop + 1
    if n_frames <= 0:
        return np.array([]), np.array([]), np.zeros((nperseg // 2 + 1, 1))

    spectrogram = np.zeros((nperseg // 2 + 1, n_frames))

    for i in range(n_frames):
        start = i * hop
        segment = data[start : start + nperseg]
        if len(segment) < nperseg:
            segment = np.pad(segment, (0, nperseg - len(segment)))
        spectrum = np.abs(scipy_fft.rfft(win * segment)) ** 2
        spectrogram[:, i] = spectrum

    times = np.arange(n_frames) * hop / fs

    freqs = scipy_fft.rfftfreq(nperseg, 1 / fs)

    return freqs, times, spectrogram


def phase_spectrum(data: np.ndarray, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute phase spectrum.

    Args:
        data: Input signal
        fs: Sampling frequency

    Returns:
        Tuple of (frequencies, phase in radians)
    """
    freqs, complex_spectrum = compute_fft(
        data, nperseg=len(data), fs=fs, mode="complex"
    )
    phase = np.angle(complex_spectrum)
    return freqs, phase


def compute_dft(
    signal_data: np.ndarray,
    freqs: np.ndarray,
    fs: float = 1.0,
) -> np.ndarray:
    """Compute DFT at specific frequencies.

    Useful for narrowband analysis at specific frequencies.

    Args:
        signal_data: Input signal
        freqs: Frequencies to compute DFT at
        fs: Sampling frequency

    Returns:
        Complex DFT values at requested frequencies
    """
    n = len(signal_data)
    t = np.arange(n) / fs

    result = np.zeros(len(freqs), dtype=complex)

    for i, f in enumerate(freqs):
        result[i] = np.sum(signal_data * np.exp(-2j * np.pi * f * t))

    return result
