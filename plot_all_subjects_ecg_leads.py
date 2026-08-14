"""Batch plot all ECG leads for the 24 VITALSENSE subjects.

Each output PNG contains the Resting and Apnea recordings for one subject.
Rows are ECG Lead II, Lead III, and Lead V1; columns are the two scenes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, detrend, sosfiltfilt


LEADS = (
    ("ecg_lead2", "Lead II"),
    ("ecg_lead3", "Lead III"),
    ("ecg_leadv1", "Lead V1"),
)
SCENES = ("Resting", "Apnea")
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 40.0
FILTER_ORDER = 4


def load_ecg(path: Path) -> tuple[float, dict[str, np.ndarray]]:
    """Load the sampling rate and the three ECG leads from one MAT file."""
    required = ["Fs_ecg", *(name for name, _ in LEADS)]
    data = loadmat(path, variable_names=required, squeeze_me=True)

    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing variables in {path}: {', '.join(missing)}")

    fs_hz = float(np.asarray(data["Fs_ecg"]).squeeze())
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError(f"Invalid Fs_ecg in {path}: {fs_hz}")

    signals = {
        name: np.asarray(data[name], dtype=float).reshape(-1)
        for name, _ in LEADS
    }
    return fs_hz, signals


def select_window(
    signal: np.ndarray,
    fs_hz: float,
    start_s: float,
    duration_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the requested signal window and its absolute time axis."""
    start_index = max(0, int(round(start_s * fs_hz)))
    if duration_s is None:
        stop_index = signal.size
    else:
        stop_index = min(signal.size, start_index + int(round(duration_s * fs_hz)))

    if start_index >= stop_index:
        raise ValueError(
            f"Selected window is empty (start={start_s:g} s, "
            f"signal duration={signal.size / fs_hz:g} s)"
        )

    indices = np.arange(start_index, stop_index)
    return indices / fs_hz, signal[start_index:stop_index]


def filter_and_remove_drift(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    """Interpolate missing samples, remove linear drift, and bandpass the ECG."""
    values = np.asarray(signal, dtype=float).copy()
    finite = np.isfinite(values)
    if finite.sum() < 2:
        raise ValueError("ECG window has fewer than two finite samples")
    if not finite.all():
        sample_indices = np.arange(values.size)
        values[~finite] = np.interp(
            sample_indices[~finite], sample_indices[finite], values[finite]
        )

    nyquist_hz = fs_hz / 2.0
    if FILTER_HIGH_HZ >= nyquist_hz:
        raise ValueError(
            f"Filter upper cutoff {FILTER_HIGH_HZ:g} Hz must be below "
            f"Nyquist frequency {nyquist_hz:g} Hz"
        )

    drift_removed = detrend(values, type="linear")
    sos = butter(
        FILTER_ORDER,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
        fs=fs_hz,
        output="sos",
    )
    return sosfiltfilt(sos, drift_removed)


def plot_subject(
    dataset_dir: Path,
    output_dir: Path,
    subject: str,
    start_s: float,
    duration_s: float | None,
    dpi: int,
) -> Path:
    """Create one six-panel ECG overview PNG for a subject."""
    recordings: dict[str, tuple[float, dict[str, np.ndarray], Path]] = {}
    for scene in SCENES:
        path = dataset_dir / subject / f"{subject}_{scene}_Mindray.mat"
        if not path.is_file():
            raise FileNotFoundError(f"Mindray file not found: {path}")
        fs_hz, signals = load_ecg(path)
        recordings[scene] = (fs_hz, signals, path)

    fig, axes = plt.subplots(
        nrows=len(LEADS),
        ncols=len(SCENES),
        figsize=(18, 11),
        sharex="col",
        constrained_layout=True,
    )
    colors = ("#1864ab", "#c92a2a", "#2b8a3e")

    for column, scene in enumerate(SCENES):
        fs_hz, signals, _ = recordings[scene]
        for row, ((variable, lead_label), color) in enumerate(zip(LEADS, colors)):
            axis = axes[row, column]
            filtered_signal = filter_and_remove_drift(signals[variable], fs_hz)
            time_s, filtered_values = select_window(
                filtered_signal, fs_hz, start_s, duration_s
            )
            axis.plot(time_s, filtered_values, color=color, linewidth=0.75)
            axis.set_ylabel(f"{lead_label}\nFiltered amplitude")
            axis.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.75)
            if duration_s is None:
                axis.set_xlim(time_s[0], time_s[-1])
            else:
                axis.set_xlim(start_s, start_s + duration_s)
            if row == 0:
                axis.set_title(f"{scene} (Fs = {fs_hz:g} Hz)", fontsize=13)
            if row == len(LEADS) - 1:
                axis.set_xlabel("Time (s)")

    shown_duration = "full recording" if duration_s is None else f"{duration_s:g} s"
    fig.suptitle(
        f"{subject} ECG leads - 0.5-40 Hz filtered and detrended ({shown_duration})",
        fontsize=16,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject}_ECG_all_leads.png"
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output_path


def discover_subjects(dataset_dir: Path) -> list[str]:
    """Find subject folders named VS followed by two digits."""
    return sorted(
        path.name
        for path in dataset_dir.glob("VS[0-9][0-9]")
        if path.is_dir()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Resting and Apnea ECG Lead II, III, and V1 for all subjects."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("VITALSENSE_120_DATASET"),
        help="Dataset root containing VS01, VS02, ... folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ECG_24_subjects_png"),
        help="Directory in which PNG files are saved.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional subject IDs, for example: --subjects VS01 VS02",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration in seconds (default: 5).",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    subjects = args.subjects or discover_subjects(dataset_dir)

    if not subjects:
        raise FileNotFoundError(f"No VSxx subject folders found in {dataset_dir}")

    print(f"Found {len(subjects)} subjects. Output: {output_dir}")
    failures: list[str] = []
    for subject in subjects:
        subject = subject.upper()
        try:
            output_path = plot_subject(
                dataset_dir,
                output_dir,
                subject,
                args.start,
                args.duration,
                args.dpi,
            )
            print(f"[OK] {subject}: {output_path.name}")
        except (FileNotFoundError, KeyError, ValueError) as error:
            failures.append(f"{subject}: {error}")
            print(f"[ERROR] {subject}: {error}")

    print(f"Completed: {len(subjects) - len(failures)}/{len(subjects)} subjects")
    if failures:
        raise RuntimeError("Some subjects failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
