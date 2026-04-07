# vlfrx

Python library for VLF (Very Low Frequency) radio signal processing.

This library can be used both as a CLI tool and as a Python library for other projects.

[![PyPI](https://img.shields.io/pypi/v/vlfrx.svg)](https://pypi.org/project/vlfrx/)
[![Python](https://img.shields.io/pypi/pyversions/vlfrx.svg)](https://pypi.org/project/vlfrx/)
[![Coverage](https://codecov.io/gh/daedalus/vlfrx-tools-py/branch/main/graph/badge.svg)](https://codecov.io/gh/daedalus/vlfrx-tools-py)

## Install

```bash
pip install vlfrx
```

## Usage (as Python Library)

```python
from vlfrx import open_input, open_output, Timestamp
import numpy as np
from vlfrx.services.signal_gen import generate_sine
from vlfrx.services.spectrum import compute_fft
from vlfrx.services.filter import design_fir_filter, apply_filter

# Read a VT file
vt = open_input("signal.vt")
frame = vt.read_frame()
print(f"Timestamp: {vt.get_timestamp()}")
print(f"Channels: {vt.channels}, Sample rate: {vt.sample_rate} Hz")
vt.close()

# Write a VT file
signal = generate_sine(1000, 1.0, fs=48000)

vt = open_output("output.vt", channels=1, sample_rate=48000)
vt.write_frames(signal.reshape(-1, 1))
vt.close()

# Compute spectrum
vt = open_input("signal.vt")
frames = []
while True:
    frame = vt.read_frame()
    if frame is None:
        break
    frames.append(frame)
data = np.array(frames)
freqs, psd = compute_fft(data[:, 0], fs=vt.sample_rate)

# Apply filter
b = design_fir_filter('lowpass', 1000, fs=48000)
filtered, _ = apply_filter(data[:, 0], b)
```

### Public API

```python
# Core
from vlfrx import (
    Timestamp,      # High-precision timestamp
    VTFile,         # VT file reader/writer
    VTBlock,        # Data block structure
    ChannelSpec,    # Channel specification
    open_input,     # Open VT file for reading
    open_output,    # Open VT file for writing
    parse_chanspec,
)

# Services
from vlfrx.services.spectrum import (
    compute_fft,
    compute_power_spectrum,
    compute_spectrogram,
    compute_rolling_spectrogram,
    phase_spectrum,
)

from vlfrx.services.filter import (
    design_fir_filter,
    design_iir_filter,
    apply_filter,
    apply_fir_filter,
    apply_iir_filter,
    design_bandpass,
    design_notch,
)

from vlfrx.services.signal_gen import (
    generate_sine,
    generate_square,
    generate_sawtooth,
    generate_white_noise,
    generate_pink_noise,
    generate_chirp,
    generate_am,
    generate_fm,
)
```

## CLI

```bash
# Read and display VT file
vlfrx read input.vt

# Show file info
vlfrx info input.vt

# Compute spectrum
vlfrx spec input.vt -o spectrum.txt

# Generate a sine wave
vlfrx gen 1000 1.0 -o signal.vt

# Concatenate files
vlfrx cat file1.vt file2.vt -o combined.vt

# Apply filter
vlfrx filter input.vt -o output.vt -t lowpass -c 1000
```

## Development

```bash
git clone https://github.com/daedalus/vlfrx-tools-py.git
cd vlfrx-tools-py
pip install -e ".[test]"

# run tests
pytest

# format
ruff format src/ tests/

# lint
ruff check src/ tests/

# type check
mypy src/
```

## License

MIT License - Copyright (c) 2026 Darío Clavijo