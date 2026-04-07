"""Signal filtering services."""

from __future__ import annotations

import numpy as np
from scipy import signal


def design_fir_filter(
    filter_type: str,
    cutoff: float | tuple[float, float] | list[float],
    numtaps: int = 101,
    fs: float = 1.0,
    width: float | None = None,
    window: str = "hamming",
) -> np.ndarray:
    """Design an FIR filter.

    Args:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        cutoff: Cutoff frequency/ies (Hz or normalized)
        numtaps: Number of filter coefficients
        fs: Sampling frequency (Hz)
        width: Transition width (for windowed sinc)
        window: Window function ('hamming', 'hanning', 'blackman', 'kaiser')

    Returns:
        FIR filter coefficients

    Example:
        >>> # Lowpass filter at 1000 Hz
        >>> b = design_fir_filter('lowpass', 1000, numtaps=101, fs=48000)
    """
    nyquist = fs / 2

    # Handle cutoff - could be float or tuple
    cutoff_val: float
    cutoffs_normalized: list[float]

    if isinstance(cutoff, tuple):
        # Bandpass/stop with two frequencies
        cutoffs_normalized = [c / nyquist if c < nyquist else c for c in cutoff]
        cutoff_val = cutoffs_normalized[0]
    elif isinstance(cutoff, list):
        cutoffs_normalized = [c / nyquist if c < nyquist else c for c in cutoff]
        cutoff_val = cutoffs_normalized[0]
    else:
        # Single cutoff
        cutoff_val = cutoff / nyquist if cutoff < nyquist else cutoff
        cutoffs_normalized = [cutoff_val]

    if filter_type == "lowpass":
        if width is not None:
            width_val = width
            cutoff_norm_cutoff = (float(cutoff_val) - width_val / 2) / nyquist
            if cutoff_norm_cutoff <= 0:
                cutoff_norm_cutoff = 0.01
            b = signal.firwin(numtaps, cutoff_norm_cutoff, window=window)
        else:
            b = signal.firwin(numtaps, cutoff_val, window=window)

    elif filter_type == "highpass":
        b = signal.firwin(numtaps, cutoff_val, window=window, pass_zero=False)

    elif filter_type == "bandpass":
        if isinstance(cutoff, (tuple, list)) and len(cutoff) == 2:
            low, high = cutoff
            b = signal.firwin(
                numtaps, [low / nyquist, high / nyquist], window=window, pass_zero=False
            )
        else:
            raise ValueError("Bandpass requires low and high cutoff frequencies")

    elif filter_type == "bandstop":
        if isinstance(cutoff, (tuple, list)) and len(cutoff) == 2:
            low, high = cutoff
            b = signal.firwin(numtaps, [low / nyquist, high / nyquist], window=window)
        else:
            raise ValueError("Bandstop requires low and high cutoff frequencies")

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return np.asarray(b)


def design_iir_filter(
    filter_type: str,
    cutoff: float | tuple[float, float],
    fs: float = 1.0,
    order: int = 4,
    rp: float = 0.5,
    rs: float = 40.0,
    btype: str = "lowpass",
) -> tuple[np.ndarray, np.ndarray]:
    """Design an IIR filter.

    Args:
        filter_type: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'
        cutoff: Cutoff frequency (Hz) or [low, high] for bandpass/stop
        fs: Sampling frequency (Hz)
        order: Filter order
        rp: Passband ripple (dB) for Chebyshev I and Elliptic
        rs: Stopband attenuation (dB) for Chebyshev II and Elliptic
        btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'

    Returns:
        Tuple of (b, a) filter coefficients

    Example:
        >>> # 4th order Butterworth lowpass at 1000 Hz
        >>> b, a = design_iir_filter('butter', 1000, fs=48000, order=4)
    """
    nyquist = fs / 2
    wn: float | list[float]
    if isinstance(cutoff, tuple):
        wn = [c / nyquist for c in cutoff]
    elif isinstance(cutoff, list):
        wn = [c / nyquist for c in cutoff]
    else:
        wn = cutoff / nyquist

    if filter_type == "butter":
        b, a = signal.butter(order, wn, btype=btype, analog=False)
    elif filter_type == "cheby1":
        b, a = signal.cheby1(order, rp, wn, btype=btype, analog=False)
    elif filter_type == "cheby2":
        b, a = signal.cheby2(order, rs, wn, btype=btype, analog=False)
    elif filter_type == "ellip":
        b, a = signal.ellip(order, rp, rs, wn, btype=btype, analog=False)
    elif filter_type == "bessel":
        b, a = signal.bessel(order, wn, btype=btype, analog=False)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return b, a


def apply_filter(
    data: np.ndarray,
    coefficients: np.ndarray | tuple[np.ndarray, np.ndarray],
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply filter to signal.

    Args:
        data: Input signal
        coefficients: Either FIR (b) or IIR (b, a) coefficients
        zi: Initial filter state (for recursive filtering)

    Returns:
        Tuple of (filtered output, final filter state)

    Example:
        >>> b = design_fir_filter('lowpass', 1000, fs=48000)
        >>> filtered, _ = apply_filter(signal, b)
    """
    if isinstance(coefficients, tuple):
        b, a = coefficients
        if zi is not None:
            output, zf = signal.lfilter(b, a, data, zi=zi)
        else:
            output = signal.lfilter(b, a, data)
            zf = None
    else:
        b = coefficients
        # FIR filter using lfilter (slower but consistent)
        output = signal.lfilter(b, [1.0], data)
        zf = None

    return output, zf


def apply_fir_filter(data: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply FIR filter using efficient convolution.

    Args:
        data: Input signal
        b: FIR coefficients

    Returns:
        Filtered signal
    """
    return np.asarray(signal.filtfilt(b, np.array([1.0]), data))


def apply_iir_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply IIR filter using filtfilt (zero-phase filtering).

    Args:
        data: Input signal
        b, a: IIR filter coefficients

    Returns:
        Filtered signal
    """
    return np.asarray(signal.filtfilt(b, a, data))


def design_bandpass(
    center_freq: float,
    bandwidth: float,
    fs: float = 1.0,
    numtaps: int = 101,
) -> np.ndarray:
    """Design a bandpass filter.

    Args:
        center_freq: Center frequency in Hz
        bandwidth: Bandwidth in Hz
        fs: Sampling frequency in Hz
        numtaps: Number of taps

    Returns:
        FIR filter coefficients
    """
    low = (center_freq - bandwidth / 2) / (fs / 2)
    high = (center_freq + bandwidth / 2) / (fs / 2)
    return np.asarray(signal.firwin(numtaps, [low, high], pass_zero=False))


def design_notch(
    freq: float,
    fs: float = 1.0,
    quality: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a notch filter.

    Args:
        freq: Frequency to notch out (Hz)
        fs: Sampling frequency (Hz)
        quality: Quality factor (higher = narrower notch)

    Returns:
        (b, a) filter coefficients
    """
    w0 = 2 * np.pi * freq / fs
    alpha = np.sin(w0) / (2 * quality)

    b = np.array([1, -2 * np.cos(w0), 1]) / (1 + alpha)
    a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])

    return b, a


def moving_average_filter(window_size: int) -> np.ndarray:
    """Design a simple moving average filter.

    Args:
        window_size: Number of points in moving average

    Returns:
        Filter coefficients (all 1/window_size)
    """
    return np.ones(window_size) / window_size
