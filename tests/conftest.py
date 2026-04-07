"""Test configuration for vlfrx tests."""

import numpy as np
import pytest


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_signal():
    """Generate a test signal (1kHz sine wave)."""
    fs = 48000
    t = np.arange(fs) / fs  # 1 second
    return np.sin(2 * np.pi * 1000 * t)


@pytest.fixture
def sample_frames():
    """Generate test frame data."""
    return np.random.randn(100, 2)  # 100 frames, 2 channels
