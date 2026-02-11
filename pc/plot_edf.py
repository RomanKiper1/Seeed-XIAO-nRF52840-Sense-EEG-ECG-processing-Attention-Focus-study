import argparse

import mne
import numpy as np
import matplotlib.pyplot as plt


def robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def winsorize_masked_median(x: np.ndarray, clip_factor: float, kernel_size: int) -> np.ndarray:
    sigma = robust_sigma(x)
    limit = clip_factor * sigma
    clipped = np.clip(x, -limit, limit) if limit > 0 else x.copy()
    mask = np.abs(x) > limit if limit > 0 else np.zeros_like(x, dtype=bool)

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    half = k // 2
    out = clipped.copy()

    for i in range(len(x)):
        if not mask[i]:
            continue
        start = max(0, i - half)
        end = min(len(x), i + half + 1)
        out[i] = np.median(clipped[start:end])
    return out


def plot_overlay(t: np.ndarray, raw: np.ndarray, filtered: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(t, raw, label="RAW", alpha=0.7)
    plt.plot(t, filtered, label="FIL", alpha=0.9)
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", required=True, help="Input EDF file")
    parser.add_argument("--channel", default=None, help="Channel name or index")
    parser.add_argument("--clip", type=float, default=8.0, help="Clip factor c")
    parser.add_argument("--kernel", type=int, default=7, help="Median kernel size (odd)")
    args = parser.parse_args()

    raw = mne.io.read_raw_edf(args.edf, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    times = raw.times

    if args.channel is None:
        picks = [0]
    else:
        try:
            picks = [int(args.channel)]
        except ValueError:
            picks = [raw.ch_names.index(args.channel)]

    data = raw.get_data(picks=picks)[0]
    filtered = winsorize_masked_median(data, args.clip, args.kernel)

    plot_overlay(times, data, filtered, f"EDF RAW vs FIL (sfreq={sfreq} Hz)")


if __name__ == "__main__":
    main()
