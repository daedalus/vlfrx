"""vlfrx - VLF Radio Signal Processing Library CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from pathlib import Path


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """vlfrx - VLF radio signal processing tools."""
    pass


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-t", "--timestamps", is_flag=True, help="Show timestamps")
@click.option("-n", "--frames", type=int, help="Number of frames to display")
def read(input: Path, timestamps: bool, frames: int) -> None:
    """Read and display VT file contents.

    Args:
        input: Input VT file path
    """
    from vlfrx import open_input

    vt = open_input(input)

    click.echo(f"File: {input}")
    click.echo(f"Channels: {vt.channels}")
    click.echo(f"Sample rate: {vt.sample_rate}")

    count = 0
    max_frames = frames if frames else 10

    while count < max_frames:
        frame = vt.read_frame()
        if frame is None:
            break

        if timestamps:
            ts = vt.get_timestamp()
            click.echo(f"[{ts}] ", nl=False)

        click.echo(f"  {frame[:4]}..." if len(frame) > 4 else f"  {frame}")
        count += 1

    vt.close()
    click.echo(f"Displayed {count} frames")


@main.command()
@click.argument("output", type=click.Path())
@click.option("-c", "--channels", type=int, default=1, help="Number of channels")
@click.option("-r", "--sample-rate", type=int, default=48000, help="Sample rate in Hz")
@click.option(
    "-f",
    "--format",
    type=click.Choice(["float64", "float32", "int16", "int32"]),
    default="float64",
    help="Data format",
)
@click.option("-b", "--block-size", type=int, default=8192, help="Block size")
def write(
    output: Path, channels: int, sample_rate: int, format: str, block_size: int
) -> None:
    """Create empty VT file for writing."""
    from vlfrx import open_output

    vt = open_output(
        output,
        channels=channels,
        sample_rate=sample_rate,
        block_size=block_size,
        format=format,
    )
    vt.close()
    click.echo(f"Created {output}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file (default: stdout)")
def spec(input: Path, output: Path | None) -> None:
    """Compute spectrum of VT file."""
    import numpy as np

    from vlfrx import open_input
    from vlfrx.services.spectrum import compute_power_spectrum

    vt = open_input(input)

    # Read all data
    frames = []
    while True:
        frame = vt.read_frame()
        if frame is None:
            break
        frames.append(frame)

    if not frames:
        click.echo("No data in file")
        return

    data = np.array(frames)

    # Compute spectrum
    freqs, psd = compute_power_spectrum(data[:, 0], fs=vt.sample_rate)

    if output:
        np.savetxt(output, np.column_stack([freqs, psd]))
        click.echo(f"Spectrum saved to {output}")
    else:
        for f, p in zip(freqs[:100], psd[:100]):
            click.echo(f"{f:.1f} {p:.2f}")

    vt.close()


@main.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output VT file")
@click.option(
    "-t",
    "--filter-type",
    type=click.Choice(["lowpass", "highpass", "bandpass", "bandstop"]),
    default="lowpass",
    help="Filter type",
)
@click.option("-c", "--cutoff", type=float, multiple=True, help="Cutoff frequency (Hz)")
@click.option("-n", "--numtaps", type=int, default=101, help="FIR filter taps")
def filter(
    input: Path, output: Path, filter_type: str, cutoff: tuple[float, ...], numtaps: int
) -> None:
    """Apply filter to VT file."""
    import numpy as np

    from vlfrx import open_input, open_output
    from vlfrx.services.filter import apply_filter, design_fir_filter

    vt_in = open_input(input)

    # Read data
    frames = []
    while True:
        frame = vt_in.read_frame()
        if frame is None:
            break
        frames.append(frame)

    vt_in.close()

    if not frames:
        click.echo("No data in file")
        return

    data = np.array(frames)

    # Design filter
    if cutoff:
        if len(cutoff) == 1:
            cutoffs: float | tuple[float, float] = cutoff[0]
        elif len(cutoff) == 2:
            cutoffs = (cutoff[0], cutoff[1])
        else:
            raise ValueError("Cutoff requires 1 or 2 frequencies")
        b = design_fir_filter(
            filter_type, cutoffs, numtaps=numtaps, fs=vt_in.sample_rate
        )

        # Apply to each channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch], _ = apply_filter(data[:, ch], b)

        # Write output
        if output:
            vt_out = open_output(
                output, channels=vt_in.channels, sample_rate=vt_in.sample_rate
            )
            vt_out.write_frames(filtered)
            vt_out.close()
            click.echo(f"Filtered data saved to {output}")

    click.echo("Done")


@main.command()
@click.argument("freq", type=float)
@click.argument("duration", type=float)
@click.option("-r", "--sample-rate", type=int, default=48000, help="Sample rate")
@click.option("-o", "--output", type=click.Path(), help="Output VT file")
@click.option("-a", "--amplitude", type=float, default=1.0, help="Amplitude")
def gen(
    freq: float, duration: float, sample_rate: int, output: Path, amplitude: float
) -> None:
    """Generate a sine wave and optionally save to VT file.

    Args:
        freq: Frequency in Hz
        duration: Duration in seconds
    """
    from vlfrx.services.signal_gen import generate_sine

    signal = generate_sine(freq, duration, fs=sample_rate, amplitude=amplitude)

    if output:
        from vlfrx import open_output

        vt = open_output(output, channels=1, sample_rate=sample_rate)
        vt.write_frames(signal.reshape(-1, 1))
        vt.close()
        click.echo(f"Generated signal saved to {output}")
    else:
        click.echo(f"Generated {len(signal)} samples ({duration:.2f}s)")


@main.command()
@click.argument("inputs", type=click.Path(exists=True), nargs=-1)
@click.option("-o", "--output", type=click.Path(), required=True, help="Output VT file")
def cat(inputs: tuple[str, ...], output: Path) -> None:
    """Concatenate multiple VT files."""
    from vlfrx import open_input, open_output

    first_vt = None
    all_frames = []

    for inp in inputs:
        vt = open_input(inp)

        if first_vt is None:
            first_vt = vt

        while True:
            frame = vt.read_frame()
            if frame is None:
                break
            all_frames.append(frame)

        vt.close()

    if first_vt is None:
        click.echo("No input files")
        return

    import numpy as np

    data = np.array(all_frames)

    vt_out = open_output(
        output,
        channels=first_vt.channels,
        sample_rate=first_vt.sample_rate,
    )
    vt_out.write_frames(data)
    vt_out.close()

    click.echo(f"Concatenated {len(inputs)} files, {len(data)} frames -> {output}")


@main.command()
@click.argument("input", type=click.Path(exists=True))
def info(input: Path) -> None:
    """Show VT file information."""
    from vlfrx import open_input

    vt = open_input(input)

    click.echo(f"File: {input}")
    click.echo(f"Channels: {vt.channels}")
    click.echo(f"Sample rate: {vt.sample_rate}")
    click.echo(f"Block size: {vt.block_size}")
    click.echo(f"Format: {vt.format}")

    vt.close()


if __name__ == "__main__":
    main()
