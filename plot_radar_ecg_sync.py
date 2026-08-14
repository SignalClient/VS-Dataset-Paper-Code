"""
Plot synchronized 120 GHz radar vital signal and Mindray ECG from VITALSENSE_120_DATASET.

Example:
    python plot_radar_ecg_sync.py
    python plot_radar_ecg_sync.py --subject VS01 --scene Resting --start 0 --duration 120
    python plot_radar_ecg_sync.py --subject VS01 --scene Apnea --output VS01_Apnea_sync.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from matplotlib.widgets import RangeSlider


# ===================== User settings =====================
# Change these values, then run:
#     python plot_radar_ecg_sync.py
#
# SUBJECT: VS01 to VS24
# SCENE: "Resting" or "Apnea"
# ECG_LEAD: "ecg_lead2", "ecg_lead3", or "ecg_leadv1"
# DURATION_SECONDS: use 0 to show the full overlapping signal.
DATASET_DIR = Path(__file__).resolve().parent / "VITALSENSE_120_DATASET"
SUBJECT = "VS07"
SCENE = "Resting"
ECG_LEAD = "ecg_lead2"
START_SECONDS = 0.0
DURATION_SECONDS =120.0
REMOVE_ECG_BASELINE = True
ECG_BASELINE_CUTOFF_HZ = 0.5

# First figure: original synchronized radar and ECG.
PLOT_ORIGINAL_SYNC = True
OUTPUT_IMAGE = None
# OUTPUT_IMAGE = "VS11_Resting_sync_120s.png"

# Second figure: raw phase, 0.5-20 Hz bandpass + second difference, and ECG.
PLOT_BANDPASS_DIFF_ECG = True
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 20.0
BANDPASS_ORDER = 4
ENABLE_TIME_RANGE_SLIDER = True
FILTERED_OUTPUT_IMAGE = None
# FILTERED_OUTPUT_IMAGE = "VS10_Resting_bandpass_diff_ecg.png"

# Third figure: raw phase, 10-80 Hz bandpass phase, and ECG.
PLOT_PHASE_10_80_ECG = True
PHASE_BANDPASS_LOW_HZ = 10.0
PHASE_BANDPASS_HIGH_HZ = 80.0
PHASE_BANDPASS_ORDER = 4
PHASE_BANDPASS_OUTPUT_IMAGE = None
# PHASE_BANDPASS_OUTPUT_IMAGE = "VS10_Resting_phase_10_80_ecg.png"
# =========================================================


def scalar(value: np.ndarray) -> float:
    """Return a MATLAB scalar loaded by scipy.io.loadmat as a Python float."""
    return float(np.asarray(value).squeeze())


def vector(value: np.ndarray) -> np.ndarray:
    """Return a MATLAB vector as a 1-D float array."""
    return np.asarray(value, dtype=float).squeeze()


def robust_normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize with median/IQR so ECG spikes and radar drift fit on one axis."""
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal)
    if not finite.any():
        return np.zeros_like(signal)

    center = np.nanmedian(signal[finite])
    q25, q75 = np.nanpercentile(signal[finite], [25, 75])
    scale = q75 - q25
    if scale <= 0:
        scale = np.nanstd(signal[finite])
    if scale <= 0:
        scale = 1.0

    normalized = (signal - center) / scale
    return np.clip(normalized, -8, 8)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal)
    if not finite.any():
        return np.zeros_like(signal)

    mean = np.nanmean(signal[finite])
    std = np.nanstd(signal[finite])
    if std <= 0:
        std = 1.0
    return (signal - mean) / std


def minmax_normalize_signed(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal)
    if not finite.any():
        return np.zeros_like(signal)

    min_value = np.nanmin(signal[finite])
    max_value = np.nanmax(signal[finite])
    value_range = max_value - min_value
    if value_range <= 0:
        return np.zeros_like(signal)

    normalized = 2.0 * (signal - min_value) / value_range - 1.0
    return np.clip(normalized, -1.0, 1.0)


def fill_missing_values(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal)
    if finite.all():
        return signal
    if not finite.any():
        return np.zeros_like(signal)

    sample_index = np.arange(signal.size)
    filled = signal.copy()
    filled[~finite] = np.interp(sample_index[~finite], sample_index[finite], signal[finite])
    return filled


def bandpass_filter(signal: np.ndarray, fs_hz: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs_hz
    if low_hz <= 0:
        raise ValueError("Bandpass low cutoff must be greater than 0 Hz.")
    if high_hz >= nyquist:
        raise ValueError(f"Bandpass high cutoff must be lower than Nyquist frequency ({nyquist:.3f} Hz).")
    if low_hz >= high_hz:
        raise ValueError("Bandpass low cutoff must be lower than high cutoff.")

    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def highpass_filter(signal: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs_hz
    if cutoff_hz <= 0:
        raise ValueError("Highpass cutoff must be greater than 0 Hz.")
    if cutoff_hz >= nyquist:
        raise ValueError(f"Highpass cutoff must be lower than Nyquist frequency ({nyquist:.3f} Hz).")

    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return sosfiltfilt(sos, signal)


def remove_ecg_baseline(signal: np.ndarray, fs_hz: float, cutoff_hz: float) -> np.ndarray:
    signal = fill_missing_values(signal)
    baseline_removed = highpass_filter(signal, fs_hz, cutoff_hz)
    return baseline_removed - np.nanmedian(baseline_removed)


def second_difference_paper_formula(x: np.ndarray, fs_hz: float) -> np.ndarray:
    dt = 1.0 / fs_hz
    kernel = np.array([1.0, 2.0, -1.0, -4.0, -1.0, 2.0, 1.0], dtype=float)
    kernel /= 16.0 * dt * dt
    padded = np.pad(x, (3, 3), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def load_pair(dataset_dir: Path, subject: str, scene: str, ecg_lead: str) -> dict[str, np.ndarray | float | Path]:
    subject = subject.upper()
    scene = scene.capitalize()
    subject_dir = dataset_dir / subject
    radar_path = subject_dir / f"{subject}_{scene}.mat"
    mindray_path = subject_dir / f"{subject}_{scene}_Mindray.mat"

    if not radar_path.exists():
        raise FileNotFoundError(f"Radar file not found: {radar_path}")
    if not mindray_path.exists():
        raise FileNotFoundError(f"Mindray file not found: {mindray_path}")

    radar_mat = loadmat(radar_path)
    mindray_mat = loadmat(mindray_path)

    if "VitalSig" not in radar_mat:
        raise KeyError(f"'VitalSig' not found in {radar_path}")
    if "T_frame" not in radar_mat:
        raise KeyError(f"'T_frame' not found in {radar_path}")
    if ecg_lead not in mindray_mat:
        available = ", ".join(k for k in mindray_mat if k.startswith("ecg_"))
        raise KeyError(f"'{ecg_lead}' not found in {mindray_path}. Available ECG leads: {available}")
    if "Fs_ecg" not in mindray_mat:
        raise KeyError(f"'Fs_ecg' not found in {mindray_path}")

    radar_signal = vector(radar_mat["VitalSig"])
    radar_dt = scalar(radar_mat["T_frame"])
    ecg_signal = vector(mindray_mat[ecg_lead])
    fs_ecg = scalar(mindray_mat["Fs_ecg"])
    if REMOVE_ECG_BASELINE:
        ecg_signal = remove_ecg_baseline(ecg_signal, fs_ecg, ECG_BASELINE_CUTOFF_HZ)

    return {
        "radar_signal": radar_signal,
        "radar_time": np.arange(radar_signal.size) * radar_dt,
        "ecg_signal": ecg_signal,
        "ecg_time": np.arange(ecg_signal.size) / fs_ecg,
        "radar_dt": radar_dt,
        "fs_ecg": fs_ecg,
        "radar_path": radar_path,
        "mindray_path": mindray_path,
    }


def time_window(time: np.ndarray, signal: np.ndarray, start: float, duration: float | None) -> tuple[np.ndarray, np.ndarray]:
    if duration is None:
        mask = time >= start
    else:
        mask = (time >= start) & (time <= start + duration)
    return time[mask], signal[mask]


def plot_sync(
    data: dict[str, np.ndarray | float | Path],
    subject: str,
    scene: str,
    ecg_lead: str,
    start: float,
    duration: float | None,
    output: Path | None,
) -> None:
    radar_time = data["radar_time"]
    radar_signal = data["radar_signal"]
    ecg_time = data["ecg_time"]
    ecg_signal = data["ecg_signal"]

    assert isinstance(radar_time, np.ndarray)
    assert isinstance(radar_signal, np.ndarray)
    assert isinstance(ecg_time, np.ndarray)
    assert isinstance(ecg_signal, np.ndarray)

    common_end = min(float(radar_time[-1]), float(ecg_time[-1]))
    if duration is None:
        duration = max(0.0, common_end - start)

    radar_t, radar_y = time_window(radar_time, radar_signal, start, duration)
    ecg_t, ecg_y = time_window(ecg_time, ecg_signal, start, duration)

    if radar_t.size == 0 or ecg_t.size == 0:
        raise ValueError("Selected time window is empty. Check --start and --duration.")

    radar_norm = robust_normalize(radar_y)
    ecg_norm = robust_normalize(ecg_y)
    ecg_y_norm = minmax_normalize_signed(ecg_y)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.0]},
    )

    axes[0].plot(radar_t, radar_norm, color="#1f77b4", linewidth=1.2, label="Radar VitalSig (normalized)")
    axes[0].plot(ecg_t, ecg_norm + 4.0, color="#d62728", linewidth=0.7, label=f"{ecg_lead} ECG (normalized + offset)")
    axes[0].set_ylabel("Normalized")
    axes[0].set_title(f"{subject.upper()} {scene.capitalize()} synchronized radar and ECG")
    axes[0].legend(loc="upper right")

    axes[1].plot(radar_t, radar_y, color="#1f77b4", linewidth=1.0)
    axes[1].set_ylabel("Radar VitalSig")

    axes[2].plot(ecg_t, ecg_y_norm, color="#d62728", linewidth=0.7)
    axes[2].set_ylabel(f"ECG {ecg_lead} norm")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.margins(x=0)
        ax.grid(True, alpha=0.25)

    axes[2].set_xlim(start, start + duration)

    radar_fs = 1.0 / float(data["radar_dt"])
    fs_ecg = float(data["fs_ecg"])
    fig.text(
        0.01,
        0.01,
        f"Radar: {Path(data['radar_path']).name}, fs ~= {radar_fs:.2f} Hz | "
        f"ECG: {Path(data['mindray_path']).name}, fs = {fs_ecg:.2f} Hz"
        f"{' | ECG baseline removed' if REMOVE_ECG_BASELINE else ''}",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved figure: {output}")


def plot_bandpass_diff_ecg(
    data: dict[str, np.ndarray | float | Path],
    subject: str,
    scene: str,
    ecg_lead: str,
    start: float,
    duration: float | None,
    low_hz: float,
    high_hz: float,
    order: int,
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
        raise ValueError("Selected time window is empty. Check START_SECONDS and DURATION_SECONDS.")

    filtered_phase = bandpass_filter(raw_phase, radar_fs, low_hz, high_hz, order)
    filtered_second_diff = second_difference_paper_formula(filtered_phase, radar_fs)
    rcg_norm = minmax_normalize_signed(filtered_second_diff)
    ecg_norm = minmax_normalize_signed(ecg_y)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.2]},
    )

    axes[0].plot(radar_t, raw_phase, color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("Raw phase")
    axes[0].set_title(
        f"{subject.upper()} {scene.capitalize()} radar phase filtering and ECG"
    )

    axes[1].plot(radar_t, filtered_phase, color="#2ca02c", linewidth=1.0)
    axes[1].set_ylabel(f"{low_hz:g}-{high_hz:g} Hz")

    axes[2].plot(radar_t, rcg_norm, color="#9467bd", linewidth=1.0)
    axes[2].set_ylabel("RCG norm")

    axes[3].plot(ecg_t, ecg_norm, color="#d62728", linewidth=0.7)
    axes[3].set_ylabel(f"ECG {ecg_lead} norm")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.margins(x=0)
        ax.grid(True, alpha=0.25)

    window_start = start
    window_end = start + duration
    axes[3].set_xlim(window_start, window_end)

    fig.text(
        0.01,
        0.01,
        f"Radar fs ~= {radar_fs:.2f} Hz | Bandpass: {low_hz:g}-{high_hz:g} Hz, order {order} | "
        f"ECG fs = {float(data['fs_ecg']):.2f} Hz"
        f"{' | ECG baseline removed' if REMOVE_ECG_BASELINE else ''}",
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
            for ax in axes:
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


def plot_phase_bandpass_ecg(
    data: dict[str, np.ndarray | float | Path],
    subject: str,
    scene: str,
    ecg_lead: str,
    start: float,
    duration: float | None,
    low_hz: float,
    high_hz: float,
    order: int,
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
        raise ValueError("Selected time window is empty. Check START_SECONDS and DURATION_SECONDS.")

    bandpass_phase = bandpass_filter(raw_phase, radar_fs, low_hz, high_hz, order)
    ecg_norm = minmax_normalize_signed(ecg_y)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.1]},
    )

    axes[0].plot(radar_t, raw_phase, color="#1f77b4", linewidth=1.0)
    axes[0].set_ylabel("Raw phase")
    axes[0].set_title(
        f"{subject.upper()} {scene.capitalize()} raw phase, {low_hz:g}-{high_hz:g} Hz phase, and ECG"
    )

    axes[1].plot(radar_t, bandpass_phase, color="#ff7f0e", linewidth=1.0)
    axes[1].set_ylabel(f"{low_hz:g}-{high_hz:g} Hz")

    axes[2].plot(ecg_t, ecg_norm, color="#d62728", linewidth=0.7)
    axes[2].set_ylabel(f"ECG {ecg_lead} norm")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.margins(x=0)
        ax.grid(True, alpha=0.25)

    window_start = start
    window_end = start + duration
    axes[2].set_xlim(window_start, window_end)

    fig.text(
        0.01,
        0.01,
        f"Radar fs ~= {radar_fs:.2f} Hz | Bandpass: {low_hz:g}-{high_hz:g} Hz, order {order} | "
        f"ECG fs = {float(data['fs_ecg']):.2f} Hz"
        f"{' | ECG baseline removed' if REMOVE_ECG_BASELINE else ''}",
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
            for ax in axes:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display synchronized radar VitalSig and ECG from VITALSENSE_120_DATASET."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Path to VITALSENSE_120_DATASET.")
    parser.add_argument("--subject", default=SUBJECT, help="Subject id, for example VS01.")
    parser.add_argument("--scene", choices=["Resting", "Apnea", "resting", "apnea"], default=SCENE)
    parser.add_argument("--ecg-lead", choices=["ecg_lead2", "ecg_lead3", "ecg_leadv1"], default=ECG_LEAD)
    parser.add_argument("--start", type=float, default=START_SECONDS, help="Start time in seconds.")
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS, help="Duration in seconds. Use 0 for full overlap.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None if OUTPUT_IMAGE is None else Path(OUTPUT_IMAGE),
        help="Optional PNG path for the original synchronized figure.",
    )
    parser.add_argument(
        "--filtered-output",
        type=Path,
        default=None if FILTERED_OUTPUT_IMAGE is None else Path(FILTERED_OUTPUT_IMAGE),
        help="Optional PNG path for the bandpass and second-difference figure.",
    )
    parser.add_argument(
        "--phase-bandpass-output",
        type=Path,
        default=None if PHASE_BANDPASS_OUTPUT_IMAGE is None else Path(PHASE_BANDPASS_OUTPUT_IMAGE),
        help="Optional PNG path for the raw phase, 10-80 Hz phase, and ECG figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    duration = None if args.duration == 0 else args.duration
    data = load_pair(args.dataset_dir, args.subject, args.scene, args.ecg_lead)

    if PLOT_ORIGINAL_SYNC:
        plot_sync(data, args.subject, args.scene, args.ecg_lead, args.start, duration, args.output)

    if PLOT_BANDPASS_DIFF_ECG:
        plot_bandpass_diff_ecg(
            data,
            args.subject,
            args.scene,
            args.ecg_lead,
            args.start,
            duration,
            BANDPASS_LOW_HZ,
            BANDPASS_HIGH_HZ,
            BANDPASS_ORDER,
            args.filtered_output,
            ENABLE_TIME_RANGE_SLIDER,
        )

    if PLOT_PHASE_10_80_ECG:
        plot_phase_bandpass_ecg(
            data,
            args.subject,
            args.scene,
            args.ecg_lead,
            args.start,
            duration,
            PHASE_BANDPASS_LOW_HZ,
            PHASE_BANDPASS_HIGH_HZ,
            PHASE_BANDPASS_ORDER,
            args.phase_bandpass_output,
            ENABLE_TIME_RANGE_SLIDER,
        )

    should_show_original = PLOT_ORIGINAL_SYNC and args.output is None
    should_show_filtered = PLOT_BANDPASS_DIFF_ECG and args.filtered_output is None
    should_show_phase_bandpass = PLOT_PHASE_10_80_ECG and args.phase_bandpass_output is None
    if should_show_original or should_show_filtered or should_show_phase_bandpass:
        plt.show()


if __name__ == "__main__":
    main()
