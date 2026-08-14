"""
Plot raw radar phase, denoised radar heart sound, heartbeat RCG, and ECG.

This script is designed for the processed VITALSENSE_120_DATASET .mat files used by
plot_radar_ecg_sync.py. The available radar signal is a single 1-D VitalSig phase
trace, so the separation uses a respiration fundamental plus harmonic dictionary:

1. Estimate the respiration fundamental from the low-frequency phase spectrum.
2. Fit baseline drift plus respiration harmonics to the raw phase.
3. Subtract that model to obtain a respiration-removed heartbeat phase.
4. Independently apply a 0.5-20 Hz bandpass and the paper's seven-point
   second-difference formula to the raw phase, matching the reference code.
5. Independently extract radar heart sound with a 10-80 Hz bandpass and
   db10 level-5 rigrsure soft-threshold wavelet denoising.

Example:
    python plot_heartbeat_phase_separation.py
    python plot_heartbeat_phase_separation.py --subject VS03 --scene Resting --start 0 --duration 120
    python plot_heartbeat_phase_separation.py --output VS03_heartbeat_phase_separation.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider
from scipy.signal import butter, daub, filtfilt, qmf, upfirdn, welch

from plot_radar_ecg_sync import (
    DATASET_DIR,
    ECG_LEAD,
    SCENE,
    START_SECONDS,
    SUBJECT,
    bandpass_filter,
    fill_missing_values,
    load_pair,
    minmax_normalize_signed,
    second_difference_paper_formula,
    time_window,
)


# ===================== User settings =====================
RESPIRATION_SEARCH_LOW_HZ = 0.08
RESPIRATION_SEARCH_HIGH_HZ = 0.70
RESPIRATION_HARMONIC_HIGH_HZ = 3.00
MAX_RESPIRATION_HARMONICS = 10

HEART_PHASE_LOW_HZ = 0.70
HEART_PHASE_HIGH_HZ = 20.00
HEART_PHASE_BANDPASS_ORDER = 4

RCG_PHASE_BANDPASS_LOW_HZ = 0.5
RCG_PHASE_BANDPASS_HIGH_HZ = 20.0
RCG_PHASE_BANDPASS_ORDER = 4

HEARTSOUND_LOW_HZ = 10.0
HEARTSOUND_HIGH_HZ = 80.0
HEARTSOUND_FILTER_ORDER = 4
HEARTSOUND_WAVELET = "db10"
HEARTSOUND_WAVELET_LEVEL = 5
RAW_PHASE_FFT_MAX_HZ = 100.0

_DB10_FILTER_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}

DURATION_SECONDS = 120.0
ENABLE_TIME_RANGE_SLIDER = True
OUTPUT_IMAGE = None
# OUTPUT_IMAGE = "VS03_heartbeat_phase_separation.png"
# =========================================================


def estimate_respiration_frequency(
    phase: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
) -> float:
    """Estimate respiration fundamental by Welch PSD in the respiration band."""
    phase = fill_missing_values(phase)
    centered = phase - np.nanmedian(phase)
    resp_band = bandpass_filter(centered, fs_hz, low_hz, high_hz, order=4)

    nperseg = min(resp_band.size, max(256, int(round(fs_hz * 30.0))))
    if nperseg < 16:
        raise ValueError("Selected radar window is too short to estimate respiration frequency.")

    freqs, power = welch(
        resp_band,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        raise ValueError("No Welch frequency bins inside the respiration search band.")

    band_freqs = freqs[mask]
    band_power = power[mask]
    return float(band_freqs[int(np.argmax(band_power))])


def build_respiration_dictionary(
    time_s: np.ndarray,
    respiration_hz: float,
    harmonic_high_hz: float,
    max_harmonics: int,
) -> tuple[np.ndarray, list[float]]:
    """Build baseline plus sinusoidal respiration-harmonic dictionary."""
    rel_t = time_s - time_s[0]
    span = max(float(rel_t[-1] - rel_t[0]), 1.0)
    trend = (rel_t - 0.5 * span) / span

    columns = [np.ones_like(rel_t), trend]
    harmonic_freqs: list[float] = []
    for harmonic in range(1, max_harmonics + 1):
        freq = harmonic * respiration_hz
        if freq > harmonic_high_hz:
            break
        harmonic_freqs.append(freq)
        angle = 2.0 * np.pi * freq * rel_t
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))

    return np.column_stack(columns), harmonic_freqs


def remove_respiration_harmonics(
    time_s: np.ndarray,
    phase: np.ndarray,
    fs_hz: float,
    respiration_search_low_hz: float,
    respiration_search_high_hz: float,
    respiration_harmonic_high_hz: float,
    max_respiration_harmonics: int,
    heart_low_hz: float,
    heart_high_hz: float,
    heart_order: int,
) -> dict[str, np.ndarray | float | list[float]]:
    """Return respiration-removed heartbeat phase and diagnostics."""
    phase = fill_missing_values(phase)
    centered = phase - np.nanmedian(phase)

    respiration_hz = estimate_respiration_frequency(
        centered,
        fs_hz,
        respiration_search_low_hz,
        respiration_search_high_hz,
    )
    dictionary, harmonic_freqs = build_respiration_dictionary(
        time_s,
        respiration_hz,
        respiration_harmonic_high_hz,
        max_respiration_harmonics,
    )

    coeffs, *_ = np.linalg.lstsq(dictionary, centered, rcond=None)
    baseline_and_respiration = dictionary @ coeffs
    respiration_removed = centered - baseline_and_respiration
    heartbeat_phase = bandpass_filter(
        respiration_removed,
        fs_hz,
        heart_low_hz,
        heart_high_hz,
        order=heart_order,
    )

    return {
        "respiration_hz": respiration_hz,
        "harmonic_freqs": harmonic_freqs,
        "baseline_and_respiration": baseline_and_respiration,
        "respiration_removed": respiration_removed,
        "heartbeat_phase": heartbeat_phase,
    }


def _heart_sound_bandpass(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    """Apply the reference 10-80 Hz zero-phase Butterworth filter."""
    nyquist = fs_hz / 2.0
    effective_high_hz = min(HEARTSOUND_HIGH_HZ, nyquist * (1.0 - 1e-6))
    if HEARTSOUND_LOW_HZ >= effective_high_hz:
        raise ValueError(
            f"Radar sampling rate {fs_hz:g} Hz is too low for the "
            f"{HEARTSOUND_LOW_HZ:g}-{HEARTSOUND_HIGH_HZ:g} Hz heart-sound band."
        )
    b, a = butter(
        HEARTSOUND_FILTER_ORDER,
        [HEARTSOUND_LOW_HZ / nyquist, effective_high_hz / nyquist],
        btype="bandpass",
    )
    return filtfilt(b, a, fill_missing_values(signal))


def _reference_butter_bandpass_filter(
    signal: np.ndarray,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass copied from the supplied reference."""
    nyquist = fs_hz / 2.0
    effective_low_hz = max(float(low_hz), 1e-9)
    effective_high_hz = min(float(high_hz), nyquist * (1.0 - 1e-6))
    if effective_low_hz >= effective_high_hz:
        raise ValueError(
            "Invalid bandpass range after Nyquist clipping: "
            f"{(low_hz, high_hz)} Hz at fs={fs_hz} Hz"
        )
    low = effective_low_hz / nyquist
    high = effective_high_hz / nyquist
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def extract_reference_rcg(phase: np.ndarray, fs_hz: float) -> np.ndarray:
    """Run the supplied 0.5-20 Hz bandpass and second-difference RCG branch."""
    phase_bandpassed = _reference_butter_bandpass_filter(
        phase,
        fs_hz,
        RCG_PHASE_BANDPASS_LOW_HZ,
        RCG_PHASE_BANDPASS_HIGH_HZ,
        RCG_PHASE_BANDPASS_ORDER,
    )
    return second_difference_paper_formula(phase_bandpassed, fs_hz)


def _get_db10_filters() -> tuple[np.ndarray, np.ndarray]:
    """Return the db10 analysis and synthesis filters used by the reference."""
    if "db10" not in _DB10_FILTER_CACHE:
        lowpass = np.asarray(daub(10), dtype=np.float64)
        highpass = np.asarray(qmf(lowpass), dtype=np.float64)
        _DB10_FILTER_CACHE["db10"] = (lowpass, highpass)
    return _DB10_FILTER_CACHE["db10"]


def _symmetric_dwt_level(
    signal: np.ndarray,
    lowpass: np.ndarray,
    highpass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float64)
    pad = len(lowpass) - 1
    extended = np.pad(values, (pad, pad), mode="symmetric")
    approximation = np.convolve(extended, lowpass[::-1], mode="valid")[::2]
    detail = np.convolve(extended, highpass[::-1], mode="valid")[::2]
    keep_length = min(len(approximation), len(detail))
    return approximation[:keep_length], detail[:keep_length]


def _symmetric_idwt_level(
    approximation: np.ndarray,
    detail: np.ndarray,
    target_length: int,
    lowpass: np.ndarray,
    highpass: np.ndarray,
) -> np.ndarray:
    reconstructed_a = upfirdn(lowpass, approximation, up=2)
    reconstructed_d = upfirdn(highpass, detail, up=2)
    output_length = max(reconstructed_a.size, reconstructed_d.size)
    reconstructed_a = np.pad(
        reconstructed_a, (0, output_length - reconstructed_a.size)
    )
    reconstructed_d = np.pad(
        reconstructed_d, (0, output_length - reconstructed_d.size)
    )
    reconstructed = reconstructed_a + reconstructed_d
    start = len(lowpass) - 1
    stop = start + int(target_length)
    if stop > reconstructed.size:
        reconstructed = np.pad(reconstructed, (0, stop - reconstructed.size))
    return reconstructed[start:stop]


def _wavedec_db10(
    signal: np.ndarray,
    level: int,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    lowpass, highpass = _get_db10_filters()
    approximation = np.asarray(signal, dtype=np.float64)
    details: list[np.ndarray] = []
    lengths = [len(approximation)]
    for _ in range(int(level)):
        approximation, detail = _symmetric_dwt_level(
            approximation, lowpass, highpass
        )
        details.append(detail)
        lengths.append(len(approximation))
    return approximation, details, lengths


def _waverec_db10(
    approximation: np.ndarray,
    details: list[np.ndarray],
    lengths: list[int],
) -> np.ndarray:
    lowpass, highpass = _get_db10_filters()
    reconstructed = np.asarray(approximation, dtype=np.float64)
    for level in range(len(details) - 1, -1, -1):
        reconstructed = _symmetric_idwt_level(
            reconstructed,
            details[level],
            lengths[level],
            lowpass,
            highpass,
        )
    return reconstructed


def _rigrsure_threshold(detail: np.ndarray, sigma: float) -> float:
    if detail.size == 0 or sigma <= 0.0:
        return 0.0
    normalized = np.sort(np.abs(detail) / sigma)
    squared = normalized**2
    cumulative = np.cumsum(squared)
    k = np.arange(1, normalized.size + 1, dtype=np.float64)
    risks = normalized.size - 2.0 * k + cumulative + (normalized.size - k) * squared
    return float(sigma * normalized[int(np.argmin(risks))])


def wavelet_denoise_like_matlab(
    signal: np.ndarray,
    wavelet_name: str = HEARTSOUND_WAVELET,
    level: int = HEARTSOUND_WAVELET_LEVEL,
) -> np.ndarray:
    """Approximate MATLAB wden(signal, 'rigrsure', 's', 'mln', 5, 'db10')."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if values.size < 8:
        return values.copy()
    if wavelet_name.lower() != "db10":
        raise ValueError("Only the reference db10 wavelet is supported.")

    approximation, details, lengths = _wavedec_db10(values, level)
    denoised_details = []
    for detail in details:
        sigma = float(np.median(np.abs(detail)) / 0.6744897501960817)
        threshold = _rigrsure_threshold(detail, sigma)
        denoised_details.append(
            np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        )
    denoised = _waverec_db10(approximation, denoised_details, lengths)
    return denoised[: values.size]


def extract_radar_heart_sound(phase: np.ndarray, fs_hz: float) -> np.ndarray:
    """Return 10-80 Hz bandpassed, db10 level-5 wavelet-denoised phase."""
    bandpassed = _heart_sound_bandpass(phase, fs_hz)
    return wavelet_denoise_like_matlab(bandpassed)


def raw_phase_fft_amplitude(phase: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the one-sided FFT amplitude spectrum of the mean-removed raw phase."""
    values = fill_missing_values(np.asarray(phase, dtype=np.float64))
    if values.size < 2:
        raise ValueError("At least two raw phase samples are required for an FFT.")
    centered = values - np.mean(values)
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / fs_hz)
    amplitudes = np.abs(np.fft.rfft(centered)) / centered.size
    if amplitudes.size > 2:
        amplitudes[1:-1] *= 2.0
    return frequencies, amplitudes


def plot_heartbeat_separation(
    data: dict[str, np.ndarray | float | Path],
    subject: str,
    scene: str,
    ecg_lead: str,
    start: float,
    duration: float | None,
    output: Path | None,
    enable_slider: bool,
) -> None:
    radar_time = data["radar_time"]
    radar_signal = data["radar_signal"]
    ecg_time = data["ecg_time"]
    ecg_signal = data["ecg_signal"]

    assert isinstance(radar_time, np.ndarray)
    assert isinstance(radar_signal, np.ndarray)
    assert isinstance(ecg_time, np.ndarray)
    assert isinstance(ecg_signal, np.ndarray)

    radar_fs = 1.0 / float(data["radar_dt"])
    common_end = min(float(radar_time[-1]), float(ecg_time[-1]))
    if duration is None:
        duration = max(0.0, common_end - start)

    radar_t, raw_phase = time_window(radar_time, radar_signal, start, duration)
    ecg_t, ecg_y = time_window(ecg_time, ecg_signal, start, duration)
    if radar_t.size == 0 or ecg_t.size == 0:
        raise ValueError("Selected time window is empty. Check --start and --duration.")

    separated = remove_respiration_harmonics(
        radar_t,
        raw_phase,
        radar_fs,
        RESPIRATION_SEARCH_LOW_HZ,
        RESPIRATION_SEARCH_HIGH_HZ,
        RESPIRATION_HARMONIC_HIGH_HZ,
        MAX_RESPIRATION_HARMONICS,
        HEART_PHASE_LOW_HZ,
        HEART_PHASE_HIGH_HZ,
        HEART_PHASE_BANDPASS_ORDER,
    )

    radar_heart_sound = extract_radar_heart_sound(raw_phase, radar_fs)
    rcg_signal = extract_reference_rcg(raw_phase, radar_fs)
    fft_freqs, fft_amplitudes = raw_phase_fft_amplitude(raw_phase, radar_fs)
    fft_mask = fft_freqs <= min(RAW_PHASE_FFT_MAX_HZ, radar_fs / 2.0)
    ecg_norm = minmax_normalize_signed(ecg_y)

    respiration_hz = float(separated["respiration_hz"])
    harmonic_freqs = [float(freq) for freq in separated["harmonic_freqs"]]
    harmonic_text = ", ".join(f"{freq:.2f}" for freq in harmonic_freqs)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 11.5))
    grid = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.0, 1.0, 1.0, 1.15, 1.0],
    )
    axes = [fig.add_subplot(grid[0])]
    for index in range(1, 4):
        axes.append(fig.add_subplot(grid[index], sharex=axes[0]))
    axes.append(fig.add_subplot(grid[4]))

    axes[0].plot(radar_t, raw_phase, color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("Raw phase")
    axes[0].set_title(f"{subject.upper()} {scene.capitalize()} heartbeat phase separation and ECG")

    axes[1].plot(radar_t, radar_heart_sound, color="#ff7f0e", linewidth=1.0)
    axes[1].set_ylabel("Heart sound amplitude")
    axes[1].set_title(
        "Radar heart sound: 10-80 Hz bandpass + db10 level-5 wavelet denoising",
        fontsize=10,
    )

    axes[2].plot(radar_t, rcg_signal, color="#9467bd", linewidth=1.0)
    axes[2].set_ylabel("RCG amplitude")
    axes[2].set_title(
        "RCG: 0.5-20 Hz zero-phase bandpass + 7-point second difference",
        fontsize=10,
    )
    axes[3].plot(ecg_t, ecg_norm, color="#d62728", linewidth=0.7)
    axes[3].set_ylabel(f"ECG {ecg_lead} norm")
    axes[3].set_xlabel("Time (s)")

    fft_plot_mask = fft_mask & (fft_amplitudes > 0.0)
    axes[4].semilogy(
        fft_freqs[fft_plot_mask],
        fft_amplitudes[fft_plot_mask],
        color="#2ca02c",
        linewidth=0.9,
    )
    axes[4].set_ylabel("FFT amplitude (log)")
    axes[4].set_xlabel("Frequency (Hz)")
    axes[4].set_title("Raw radar phase FFT (mean removed)", fontsize=10)
    axes[4].set_xlim(0.0, min(RAW_PHASE_FFT_MAX_HZ, radar_fs / 2.0))

    for ax in axes:
        ax.margins(x=0)
        ax.grid(True, alpha=0.25)

    window_start = start
    window_end = start + duration
    axes[3].set_xlim(window_start, window_end)

    fig.text(
        0.01,
        0.01,
        f"Radar fs ~= {radar_fs:.2f} Hz | Respiration f0 ~= {respiration_hz:.3f} Hz "
        f"({respiration_hz * 60.0:.1f} bpm) | Removed harmonics Hz: {harmonic_text} | "
        f"RCG: {RCG_PHASE_BANDPASS_LOW_HZ:g}-{RCG_PHASE_BANDPASS_HIGH_HZ:g} Hz + "
        "7-point second difference | "
        f"Radar heart sound: {HEARTSOUND_LOW_HZ:g}-{HEARTSOUND_HIGH_HZ:g} Hz + "
        f"{HEARTSOUND_WAVELET} L{HEARTSOUND_WAVELET_LEVEL} denoising | "
        f"ECG fs = {float(data['fs_ecg']):.2f} Hz",
        fontsize=9,
        color="#555555",
    )

    if enable_slider:
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        slider_ax = fig.add_axes((0.14, 0.035, 0.72, 0.025))
        time_slider = RangeSlider(
            ax=slider_ax,
            label="Time range (s)",
            valmin=window_start,
            valmax=window_end,
            valinit=(window_start, window_end),
            facecolor="#4c78a8",
        )

        def update_time_range(value: tuple[float, float]) -> None:
            left, right = value
            if right <= left:
                return
            for ax in axes[:4]:
                ax.set_xlim(left, right)
            fig.canvas.draw_idle()

        time_slider.on_changed(update_time_range)
        fig._time_range_slider = time_slider
    else:
        fig.tight_layout(rect=(0, 0.03, 1, 1))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved figure: {output}")
    print(
        f"Estimated respiration: {respiration_hz:.4f} Hz ({respiration_hz * 60.0:.2f} bpm); "
        f"removed harmonics: {harmonic_text}"
    )
    print(
        f"Radar heart sound: {HEARTSOUND_LOW_HZ:g}-{HEARTSOUND_HIGH_HZ:g} Hz "
        f"bandpass + {HEARTSOUND_WAVELET} level-{HEARTSOUND_WAVELET_LEVEL} "
        "rigrsure soft-threshold denoising"
    )
    print(
        f"RCG: {RCG_PHASE_BANDPASS_LOW_HZ:g}-{RCG_PHASE_BANDPASS_HIGH_HZ:g} Hz "
        "zero-phase bandpass + reference 7-point second difference"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw phase, radar heart sound, heartbeat second difference, and ECG."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Path to VITALSENSE_120_DATASET.")
    parser.add_argument("--subject", default=SUBJECT, help="Subject id, for example VS01.")
    parser.add_argument("--scene", choices=["Resting", "Apnea", "resting", "apnea"], default=SCENE)
    parser.add_argument("--ecg-lead", choices=["ecg_lead2", "ecg_lead3", "ecg_leadv1"], default=ECG_LEAD)
    parser.add_argument("--start", type=float, default=START_SECONDS, help="Start time in seconds.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DURATION_SECONDS,
        help="Duration in seconds. Use 0 for full overlap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None if OUTPUT_IMAGE is None else Path(OUTPUT_IMAGE),
        help="Optional PNG path. If omitted, an interactive window is shown.",
    )
    parser.add_argument(
        "--no-slider",
        action="store_true",
        help="Disable the interactive time range slider.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    duration = None if args.duration == 0 else args.duration
    data = load_pair(args.dataset_dir, args.subject, args.scene, args.ecg_lead)
    plot_heartbeat_separation(
        data,
        args.subject,
        args.scene,
        args.ecg_lead,
        args.start,
        duration,
        args.output,
        ENABLE_TIME_RANGE_SLIDER and not args.no_slider,
    )
    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
