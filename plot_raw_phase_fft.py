"""Interactive raw radar phase and FFT viewer.

The time-range slider recomputes the FFT from the selected raw-phase segment.
The frequency-range slider zooms the spectrum and rescales its amplitude axis.

Example:
    python plot_raw_phase_fft.py --subject VS17 --scene Resting --start 0 --duration 120
    python plot_raw_phase_fft.py --subject VS17 --scene Resting --frequency-range 8 90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider

from plot_radar_ecg_sync import (
    DATASET_DIR,
    ECG_LEAD,
    SCENE,
    START_SECONDS,
    SUBJECT,
    fill_missing_values,
    load_pair,
    time_window,
)


DURATION_SECONDS = 120.0
DEFAULT_FFT_MAX_HZ = 20.0
FFT_DB_FLOOR = np.finfo(np.float64).eps


def one_sided_windowed_fft(
    phase: np.ndarray,
    fs_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Hann-windowed, one-sided amplitude spectrum of raw phase."""
    values = fill_missing_values(np.asarray(phase, dtype=np.float64))
    if values.size < 8:
        raise ValueError("Select at least eight radar samples to calculate the FFT.")

    centered = values - np.mean(values)
    window = np.hanning(centered.size)
    coherent_gain = float(np.sum(window))
    spectrum = np.abs(np.fft.rfft(centered * window)) / coherent_gain
    if spectrum.size > 2:
        spectrum[1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / fs_hz)
    return frequencies, spectrum


def amplitude_to_db(amplitudes: np.ndarray) -> np.ndarray:
    """Express FFT amplitude in dB relative to one phase unit."""
    return 20.0 * np.log10(np.maximum(amplitudes, FFT_DB_FLOOR))


def plot_raw_phase_and_fft(
    data: dict[str, np.ndarray | float | Path],
    subject: str,
    scene: str,
    start: float,
    duration: float | None,
    frequency_range: tuple[float, float] | None,
    output: Path | None,
    enable_slider: bool,
) -> None:
    radar_time = data["radar_time"]
    radar_signal = data["radar_signal"]
    assert isinstance(radar_time, np.ndarray)
    assert isinstance(radar_signal, np.ndarray)

    radar_fs = 1.0 / float(data["radar_dt"])
    if duration is None:
        duration = float(radar_time[-1] - start)
    time_values, phase_values = time_window(radar_time, radar_signal, start, duration)
    if time_values.size < 8:
        raise ValueError("Selected time range is too short for an FFT.")

    nyquist = radar_fs / 2.0
    if frequency_range is None:
        frequency_range = (0.0, min(DEFAULT_FFT_MAX_HZ, nyquist))
    left_freq, right_freq = frequency_range
    if not 0.0 <= left_freq < right_freq <= nyquist:
        raise ValueError(f"Frequency range must satisfy 0 <= low < high <= {nyquist:.3f} Hz.")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1.0, 1.1]})
    phase_line, = axes[0].plot(time_values, phase_values, color="#1f77b4", linewidth=1.0)
    axes[0].set_title(f"{subject.upper()} {scene.capitalize()} raw radar phase and FFT")
    axes[0].set_ylabel("Raw phase")
    axes[0].set_xlabel("Time (s)")

    spectrum_freqs, spectrum_amplitudes = one_sided_windowed_fft(phase_values, radar_fs)
    spectrum_db = amplitude_to_db(spectrum_amplitudes)
    spectrum_line, = axes[1].plot(spectrum_freqs, spectrum_db, color="#2ca02c", linewidth=0.9)
    axes[1].set_title("Raw phase FFT (mean removed, Hann window)", fontsize=10)
    axes[1].set_ylabel("Amplitude (dB re 1 phase unit)")
    axes[1].set_xlabel("Frequency (Hz)")

    def update_spectrum_limits(low_hz: float, high_hz: float) -> None:
        axes[1].set_xlim(low_hz, high_hz)
        visible = spectrum_db[(spectrum_freqs >= low_hz) & (spectrum_freqs <= high_hz)]
        if visible.size == 0:
            return
        upper = float(np.max(visible)) + 3.0
        lower = max(float(np.percentile(visible, 1.0)) - 3.0, upper - 100.0)
        axes[1].set_ylim(lower, upper)

    axes[0].set_xlim(float(time_values[0]), float(time_values[-1]))
    update_spectrum_limits(left_freq, right_freq)
    for axis in axes:
        axis.margins(x=0)
        axis.grid(True, alpha=0.25)

    fig.text(
        0.01,
        0.01,
        f"Radar fs ~= {radar_fs:.2f} Hz | FFT resolution ~= {radar_fs / time_values.size:.4f} Hz | "
        "FFT uses the currently selected time range",
        fontsize=9,
        color="#555555",
    )

    if enable_slider:
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        time_slider_axis = fig.add_axes((0.14, 0.075, 0.72, 0.025))
        frequency_slider_axis = fig.add_axes((0.14, 0.035, 0.72, 0.025))
        time_slider = RangeSlider(
            time_slider_axis,
            "Time range (s)",
            valmin=float(time_values[0]),
            valmax=float(time_values[-1]),
            valinit=(float(time_values[0]), float(time_values[-1])),
            facecolor="#4c78a8",
        )
        frequency_slider = RangeSlider(
            frequency_slider_axis,
            "Frequency range (Hz)",
            valmin=0.0,
            valmax=nyquist,
            valinit=(left_freq, right_freq),
            facecolor="#59a14f",
        )

        def update_time_range(value: tuple[float, float]) -> None:
            nonlocal spectrum_freqs, spectrum_amplitudes, spectrum_db
            lower_time, upper_time = value
            selected = (time_values >= lower_time) & (time_values <= upper_time)
            selected_time = time_values[selected]
            selected_phase = phase_values[selected]
            if selected_time.size < 8:
                return
            spectrum_freqs, spectrum_amplitudes = one_sided_windowed_fft(selected_phase, radar_fs)
            spectrum_db = amplitude_to_db(spectrum_amplitudes)
            phase_line.set_data(selected_time, selected_phase)
            spectrum_line.set_data(spectrum_freqs, spectrum_db)
            axes[0].set_xlim(lower_time, upper_time)
            update_spectrum_limits(*frequency_slider.val)
            fig.canvas.draw_idle()

        def update_frequency_range(value: tuple[float, float]) -> None:
            low_hz, high_hz = value
            if high_hz <= low_hz:
                return
            update_spectrum_limits(low_hz, high_hz)
            fig.canvas.draw_idle()

        time_slider.on_changed(update_time_range)
        frequency_slider.on_changed(update_frequency_range)
        fig._time_slider = time_slider
        fig._frequency_slider = frequency_slider
    else:
        fig.tight_layout(rect=(0, 0.04, 1, 1))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved figure: {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw radar phase with an interactive FFT.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--scene", choices=["Resting", "Apnea", "resting", "apnea"], default=SCENE)
    parser.add_argument("--ecg-lead", choices=["ecg_lead2", "ecg_lead3", "ecg_leadv1"], default=ECG_LEAD)
    parser.add_argument("--start", type=float, default=START_SECONDS)
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS, help="Use 0 for all remaining data.")
    parser.add_argument("--frequency-range", type=float, nargs=2, metavar=("LOW_HZ", "HIGH_HZ"))
    parser.add_argument("--output", type=Path, help="Optional PNG output path.")
    parser.add_argument("--no-slider", action="store_true", help="Disable interactive time and frequency sliders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_pair(args.dataset_dir, args.subject, args.scene, args.ecg_lead)
    plot_raw_phase_and_fft(
        data=data,
        subject=args.subject,
        scene=args.scene,
        start=args.start,
        duration=None if args.duration == 0 else args.duration,
        frequency_range=None if args.frequency_range is None else tuple(args.frequency_range),
        output=args.output,
        enable_slider=not args.no_slider,
    )


if __name__ == "__main__":
    main()
