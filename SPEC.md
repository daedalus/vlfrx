# SPEC.md — vlfrx-tools-py

## Purpose

Python translation of vlfrx-tools, a C library for processing VLF (Very Low Frequency) radio signals. The library provides a file format (`.vt`) for time-series signal data with timestamps, and a collection of command-line tools for signal processing operations like spectrum analysis, filtering, mixing, and visualization.

## Scope

### In Scope

- Core library (vtlib equivalent):
  - Timestamp handling (timestamp type, operations, string formatting)
  - VT file format reading/writing (binary format with headers and frames)
  - Lock-free buffer support (shared memory IPC)
  - Network streaming support (TCP client/server)
  - Channel specifications and parsing

- Signal processing tools:
  - `vtread` - Read and display VT files
  - `vtwrite` - Write VT files from various inputs
  - `vtspec` - Spectrum analyzer (FFT-based)
  - `vtscope` - Oscilloscope/display tool
  - `vtfilter` - FIR/IIR filtering
  - `vtmix` - Mix/multiply signals
  - `vtmult` - Multiplication operations
  - `vtgen` - Signal generation
  - `vtcat` - Concatenate VT files
  - `vtcmp` - Compare VT files
  - `vtsid` - SID (Sudden Ionospheric Disturbance) detection
  - `vtdate` - Timestamp utilities
  - `vtstat` - Statistics
  - `vtplot` - Plotting tool
  - `vtresample` - Resampling
  - `vtrect` - Rectification
  - `vtraw` - Raw data operations
  - `vtblank` - Blanking operations
  - `vtevent` - Event detection
  - `vttime` - Time operations
  - `vtjoin` - Join files
  - `vtnspec` - Narrowband spectrogram
  - `vtflac` - FLAC audio support
  - `vtpcal` - Phase calibration
  - `vtfm` - FM modulation
  - `vtmod` - Modulation
  - `vtpolar` - Polar coordinate operations
  - `vtrsgram` - Rolling spectrogram
  - `vtsdriq` - SDR IQ processing
  - `vtsoapy` - SoapySDR support
  - `vtvr2` - VR2 format support
  - `vtwspec` - Waterfall spectrum
  - `vtping` - Network ping utility
  - `vtcard` - Sound card interface
  - `vttoga` - TOGA (Time of Group Arrival) analysis

### Not in Scope

- GUI tools (X11/forms-based tools like vtspec, vtscope - these would require separate Python GUI framework)
- Hardware drivers (sound card, SDR hardware)
- Network daemon mode utilities

## Public API / Interface

### Core Module (`vlfrx`)

```python
from vlfrx import VTFile, Timestamp, open_input, open_output

# Open VT file for reading
vt = open_input("file.vt")
frame = vt.read_frame()
timestamp = vt.get_timestamp()
vt.close()

# Open VT file for writing
vt = open_output("output.vt", channels=2, sample_rate=48000)
vt.write_frame([0.1, 0.2])
vt.close()

# Timestamp operations
t = Timestamp.now()
t2 = Timestamp.parse("2024-01-15T12:30:45.123456")
print(t.format())
```

### CLI Tools

Each tool will be exposed as CLI commands via the `vlfrx` entry point:

```bash
vlfrx read input.vt
vlfrx write -o output.vt -r 48000 -c 2
vlfrx spec input.vt
vlfrx filter -f lowpass -c 1000 input.vt
```

### Data Structures

- **Timestamp**: High-precision timestamp (seconds + fractional part)
- **VTFile**: File handle for reading/writing VT format
- **VTBlock**: Data block containing frames with timestamp
- **ChannelSpec**: Channel specification string parser

## Data Formats

### VT File Format (Binary)

Header structure:
- Magic number for validation
- Flags (sample format: float32/float64/int8/int16/int32)
- Block size (frames per block)
- Number of channels
- Sample rate

Block structure:
- Magic number
- Flags
- Block size
- Channels
- Sample rate
- Timestamp (seconds + nanoseconds)
- Valid flag
- Frame count
- Sample rate correction factor
- Frame data (variable size based on format)

### Timestamp Format

- Can be stored as compound (int32 + double) or long double depending on platform
- Precision: nanosecond resolution
- String formats: ISO 8601, various human-readable formats

## Edge Cases

1. **Empty VT file** - Handle gracefully, return empty results
2. **Corrupted file** - Validate magic numbers, report clear errors
3. **Mismatched sample rates** - Warning and auto-resample or fail
4. **Network disconnection** - Reconnect with backoff
5. **Large files** - Stream processing, don't load entire file into memory
6. **Channel count mismatch** - Error or pad/truncate as configured
7. **Invalid timestamp in filename** - Fallback to file mtime
8. **End of file** - Return None/eof marker, allow seeking back
9. **Float format conversion** - Proper clipping for integer formats
10. **Shared memory buffer** - Handle permission issues, clean up on exit

## Performance & Constraints

- Use numpy for signal processing operations
- Lazy loading of file data (block-by-block)
- Memory-mapped I/O for large files where supported
- FFT via numpy/ scipy.fftpack or pyfftw for performance
- Target Python 3.11+
- No native code extensions (pure Python/numpy)

## Dependencies

- numpy >= 1.24
- scipy >= 1.10
- click (CLI framework)

## Testing

- Test VT file round-trip (write then read)
- Test timestamp parsing/formatting
- Test FFT calculations against known signals
- Test filter responses
- Test file concatenation and splitting