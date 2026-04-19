import argparse
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pywt
except ImportError:
    pywt = None

try:
    from scipy.signal import butter, filtfilt, welch, iirnotch, spectrogram
except ImportError:
    butter = filtfilt = welch = None
    iirnotch = None
    spectrogram = None


GANGLION_UV_PER_COUNT = 0.001869917138805

# Match blank_motor_stop.ino: SAMPLING_FREQ / FFT_SIZE
MCU_FS = 200.0
MCU_NFFT = 256
MCU_DF = MCU_FS / MCU_NFFT  # ~0.78125 Hz / bin


@dataclass
class PreprocessConfig:
    scale_mode: str = "auto"
    detrend: str = "median"


@dataclass
class BandpassConfig:
    low_hz: float = 1.0
    high_hz: float = 45.0
    order: int = 4


@dataclass
class NotchConfig:
    freq_hz: float = 50.0
    quality: float = 30.0


@dataclass
class NlmsConfig:
    taps: int = 16
    mu: float = 0.06
    eps: float = 1e-6
    delay_samples: int = 2


@dataclass
class WaveletConfig:
    wavelet: str = "sym8"
    threshold_scale: float = 1.0
    level: int = 0


@dataclass
class WinsorConfig:
    clip_factor: float = 8.0
    kernel_size: int = 7


@dataclass
class WelchConfig:
    nfft: int = 512


def robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def apply_scale(x: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if config.scale_mode == "none":
        return x.copy()
    if config.scale_mode == "ganglion":
        return x * GANGLION_UV_PER_COUNT
    median_abs = np.median(np.abs(x))
    if median_abs > 500.0:
        return x * GANGLION_UV_PER_COUNT
    return x.copy()


def remove_dc(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.copy()
    if mode == "mean":
        return x - np.mean(x)
    return x - np.median(x)


def apply_bandpass(x: np.ndarray, config: BandpassConfig, fs: float) -> np.ndarray:
    if butter is None or filtfilt is None:
        raise ImportError("scipy is required for bandpass. Install with: pip install scipy")
    if len(x) < 2:
        return x.copy()
    nyq = 0.5 * fs
    low = max(0.01, config.low_hz / nyq)
    high = min(0.99, config.high_hz / nyq)
    if low >= high:
        return x.copy()
    b, a = butter(config.order, [low, high], btype="band")
    return filtfilt(b, a, x.astype(np.float64, copy=False))


def apply_notch(x: np.ndarray, config: NotchConfig, fs: float) -> np.ndarray:
    if iirnotch is None or filtfilt is None:
        raise ImportError("scipy is required for notch filter. Install with: pip install scipy")
    if len(x) < 2:
        return x.copy()
    nyq = 0.5 * fs
    if config.freq_hz <= 0 or config.freq_hz >= nyq:
        return x.copy()
    b, a = iirnotch(config.freq_hz, config.quality, fs)
    return filtfilt(b, a, x.astype(np.float64, copy=False))


def winsorize_masked_median(x: np.ndarray, config: WinsorConfig) -> np.ndarray:
    sigma = robust_sigma(x)
    limit = config.clip_factor * sigma
    clipped = np.clip(x, -limit, limit) if limit > 0 else x.copy()
    mask = np.abs(x) > limit if limit > 0 else np.zeros_like(x, dtype=bool)
    k = config.kernel_size if config.kernel_size % 2 == 1 else config.kernel_size + 1
    half = k // 2
    out = clipped.copy()
    for i in range(len(x)):
        if not mask[i]:
            continue
        start = max(0, i - half)
        end = min(len(x), i + half + 1)
        out[i] = np.median(clipped[start:end])
    return out


def compute_psd_welch(signal: np.ndarray, fs: float, config: WelchConfig) -> tuple[np.ndarray, np.ndarray]:
    if welch is None:
        raise ImportError("scipy is required for PSD. Install with: pip install scipy")
    nfft = min(config.nfft, len(signal))
    if nfft < 2:
        return np.array([0.0]), np.array([0.0])
    freqs, psd = welch(signal.astype(np.float64), fs=fs, nperseg=nfft)
    return freqs, psd


def nlms_adaptive_cancel(primary: np.ndarray, reference: np.ndarray, config: NlmsConfig) -> np.ndarray:
    n = len(primary)
    out = primary.copy()
    if n == 0:
        return out

    taps = max(1, int(config.taps))
    delay = max(0, int(config.delay_samples))
    mu = float(config.mu)
    eps = float(config.eps)

    ref = reference.astype(np.float64, copy=False)
    if delay > 0:
        ref_delayed = np.empty_like(ref)
        ref_delayed[:delay] = 0.0
        ref_delayed[delay:] = ref[:-delay]
    else:
        ref_delayed = ref

    w = np.zeros(taps, dtype=np.float64)
    xbuf = np.zeros(taps, dtype=np.float64)

    for i in range(n):
        xbuf[1:] = xbuf[:-1]
        xbuf[0] = ref_delayed[i]
        y_hat = float(np.dot(w, xbuf))
        err = float(primary[i] - y_hat)
        out[i] = err
        norm = float(np.dot(xbuf, xbuf)) + eps
        w += (mu * err / norm) * xbuf

    return out


def wavelet_denoise(x: np.ndarray, config: WaveletConfig) -> np.ndarray:
    if pywt is None:
        raise ImportError("PyWavelets is required for wavelet denoising. Install with: pip install PyWavelets")

    wavelet = pywt.Wavelet(config.wavelet)
    max_level = pywt.dwt_max_level(len(x), wavelet.dec_len)
    level = max_level if config.level <= 0 else min(config.level, max_level)
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    if len(coeffs) <= 1:
        return x.copy()

    sigma = robust_sigma(coeffs[-1])
    threshold = config.threshold_scale * sigma * np.sqrt(2.0 * np.log(max(2, len(x))))
    coeffs_filt = [coeffs[0]]
    for detail in coeffs[1:]:
        coeffs_filt.append(pywt.threshold(detail, threshold, mode="soft"))
    den = pywt.waverec(coeffs_filt, wavelet=wavelet)
    return den[: len(x)]


def detect_separator(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                if "\t" in line:
                    return "\t"
                if "," in line:
                    return ","
                if ";" in line:
                    return ";"
                return ","
    return ","


def detect_header(path: str, sep: str) -> bool:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                tokens = [t.strip().lower() for t in line.split(sep)]
                return "tag" in tokens and "ts_ms" in tokens
    return False


def detect_streamlit_header(path: str, sep: str) -> bool:
    """Detect Streamlit-generated single-channel dumps (e.g. ',index -- streamlit-generated,value')."""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                tokens = [t.strip().lower() for t in line.split(sep)]
                return "value" in tokens and any("streamlit" in tok for tok in tokens)
    return False


def load_table(path: str) -> tuple[pd.DataFrame, str]:
    sep = detect_separator(path)
    has_header = detect_header(path, sep)
    if has_header:
        df = pd.read_csv(path, sep=sep, header=0)
        return df, "stream"

    if detect_streamlit_header(path, sep):
        df = pd.read_csv(path, sep=sep, header=0)
        return df, "streamlit"

    df = pd.read_csv(path, sep=sep, header=None)
    if df.shape[1] < 15:
        raise ValueError("Unknown format: expected at least 15 columns for BrainFlow RAW.")

    base_cols = [
        "packet_id",
        "eeg1",
        "eeg2",
        "eeg3",
        "eeg4",
        "accel_x",
        "accel_y",
        "accel_z",
        "aux1",
        "aux2",
        "aux3",
        "aux4",
        "aux5",
        "timestamp",
        "marker",
    ]
    extra_cols = [f"extra_{i}" for i in range(df.shape[1] - len(base_cols))]
    df.columns = base_cols + extra_cols
    return df, "brainflow"


def select_channels(args: argparse.Namespace) -> list[int]:
    if args.channels:
        return [int(item) for item in args.channels.split(",") if item.strip()]
    if args.preset == "eeg":
        return [1, 2]
    if args.preset == "ecg":
        return [3]
    return [1, 2, 3, 4]


def extract_raw_stream(df: pd.DataFrame, channels: list[int]) -> tuple[np.ndarray, pd.DataFrame]:
    required = {"tag", "ts_ms", "ch1", "ch2", "ch3", "ch4"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    raw_df = df[df["tag"] == "RAW"].copy()
    time_s = raw_df["ts_ms"].values / 1000.0
    channel_cols = [f"ch{ch}" for ch in channels]
    data = raw_df[channel_cols].reset_index(drop=True)
    return time_s, data


def extract_raw_brainflow(df: pd.DataFrame, channels: list[int]) -> tuple[np.ndarray, pd.DataFrame]:
    time_s = df["timestamp"].values - df["timestamp"].values[0]
    channel_cols = [f"eeg{ch}" for ch in channels]
    data = df[channel_cols].reset_index(drop=True)
    return time_s, data


def extract_raw_streamlit(
    df: pd.DataFrame, channels: list[int], fs: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Streamlit 3-column dumps: a single 'value' column is replicated to ch1..ch4."""
    value_col = next((c for c in df.columns if str(c).strip().lower() == "value"), None)
    if value_col is None:
        raise ValueError("Streamlit format requires a 'value' column.")
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)
    time_s = np.arange(n, dtype=np.float64) / float(fs) if fs > 0 else np.arange(n, dtype=np.float64)
    data = pd.DataFrame({f"ch{ch}": values.copy() for ch in channels})
    return time_s, data


def save_filtered_csv(time_s: np.ndarray, output: pd.DataFrame, output_path: str) -> None:
    out = pd.DataFrame({"time_s": time_s})
    for col in output.columns:
        out[col] = output[col].values
    out.to_csv(output_path, index=False)


def save_plot(fig: plt.Figure, path: str, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def mcu_band_bin_edges() -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Theta / Alpha / Beta bin ranges (inclusive), same rounding as MCU (round(f/DF))."""
    df = MCU_DF
    theta = (int(round(4.0 / df)), int(round(8.0 / df)))
    alpha = (int(round(8.0 / df)), int(round(13.0 / df)))
    beta = (int(round(13.0 / df)), int(round(30.0 / df)))
    return theta, alpha, beta


def band_power_rfft_sum(psd_line: np.ndarray, lo: int, hi: int) -> float:
    hi = min(hi, len(psd_line) - 1)
    lo = max(0, lo)
    if lo > hi:
        return 0.0
    return float(np.sum(psd_line[lo : hi + 1]))


def mcu_sliding_band_powers(
    x: np.ndarray,
    fs: float,
    nfft: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hamming window, rFFT magnitude-squared per block; sum bins for theta/alpha/beta (MCU bins).
    Time axis = center sample of each window / fs.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < nfft:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    w = np.hamming(nfft)
    theta_b, alpha_b, beta_b = mcu_band_bin_edges()
    t_centers: list[float] = []
    th: list[float] = []
    al: list[float] = []
    be: list[float] = []
    for start in range(0, len(x) - nfft + 1, hop):
        seg = x[start : start + nfft] * w
        spec = np.fft.rfft(seg)
        p = np.abs(spec) ** 2
        t_centers.append((start + 0.5 * nfft) / fs)
        th.append(band_power_rfft_sum(p, theta_b[0], theta_b[1]))
        al.append(band_power_rfft_sum(p, alpha_b[0], alpha_b[1]))
        be.append(band_power_rfft_sum(p, beta_b[0], beta_b[1]))
    return (
        np.array(t_centers),
        np.array(th),
        np.array(al),
        np.array(be),
    )


def plot_mcu_band_timeseries(
    time_s: np.ndarray,
    theta_p: np.ndarray,
    alpha_p: np.ndarray,
    beta_p: np.ndarray,
    channel_name: str,
    plot_dir: str,
    show: bool,
    hop_samples: int,
) -> None:
    theta_rng, alpha_rng, beta_rng = mcu_band_bin_edges()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    titles = [
        f"Theta (~4–8 Hz), bins {theta_rng[0]}–{theta_rng[1]}",
        f"Alpha (~8–13 Hz), bins {alpha_rng[0]}–{alpha_rng[1]}",
        f"Beta (~13–30 Hz), bins {beta_rng[0]}–{beta_rng[1]}",
    ]
    series = [theta_p, alpha_p, beta_p]
    for ax, y, title in zip(axes, series, titles):
        ax.plot(time_s, y, color="C0", lw=0.8)
        ax.set_ylabel("Σ|FFT|² (arb.)")
        ax.set_title(title)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"{channel_name}: band power vs time (MCU-like: Fs={MCU_FS} Hz, NFFT={MCU_NFFT}, "
        f"hop={hop_samples} samples, Hamming)",
        fontsize=11,
    )
    save_plot(fig, os.path.join(plot_dir, f"{channel_name}_bands_timeseries_mcu.png"), show)


def plot_welch_psd_with_mcu_bands(
    freqs: np.ndarray,
    psd: np.ndarray,
    channel_name: str,
    plot_dir: str,
    show: bool,
    title_suffix: str,
) -> None:
    # --nfft controls Welch resolution; shading is by Hz (theta/alpha/beta), not MCU bins.
    fig = plt.figure(figsize=(11, 5))
    n = min(len(freqs), len(psd))
    db = 10 * np.log10(np.maximum(psd[:n], 1e-20))
    plt.plot(freqs[:n], db, color="k", lw=0.9)
    plt.axvspan(4, 8, alpha=0.2, color="#8888cc", label="Theta 4–8 Hz")
    plt.axvspan(8, 13, alpha=0.2, color="#88cc88", label="Alpha 8–13 Hz")
    plt.axvspan(13, 30, alpha=0.2, color="#cc8888", label="Beta 13–30 Hz")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("PSD (dB)")
    plt.title(f"{channel_name}: Welch PSD + EEG bands {title_suffix}")
    plt.legend(loc="upper right", fontsize=8)
    if len(freqs):
        plt.xlim(0, min(60.0, float(np.max(freqs))))
    save_plot(fig, os.path.join(plot_dir, f"{channel_name}_psd_welch_bands_shaded.png"), show)


def plot_spectrogram_mcu(
    x: np.ndarray,
    fs: float,
    channel_name: str,
    plot_dir: str,
    show: bool,
) -> None:
    """STFT spectrogram (Hamming, nperseg=256); log power — Bishop-style EEG display."""
    if spectrogram is None:
        raise ImportError("scipy.signal.spectrogram required. pip install scipy")
    x = np.asarray(x, dtype=np.float64).ravel()
    nperseg = min(MCU_NFFT, len(x))
    if nperseg < 8:
        return
    noverlap = max(0, (nperseg * 3) // 4)
    f, t, Sxx = spectrogram(
        x,
        fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        scaling="density",
    )
    fig = plt.figure(figsize=(12, 4))
    log_sxx = 10 * np.log10(Sxx + 1e-20)
    vmax = np.percentile(log_sxx, 99)
    vmin = vmax - 40
    plt.pcolormesh(t, f, log_sxx, shading="gouraud", vmin=vmin, vmax=vmax, cmap="magma")
    plt.ylabel("frequency (Hz)")
    plt.xlabel("time (s)")
    plt.title(
        f"{channel_name}: spectrogram (Hamming, nperseg={nperseg}, overlap={noverlap}; "
        f"MCU DF={MCU_DF:.5f} Hz/bin)",
    )
    plt.ylim(0, 45)
    plt.colorbar(label="10·log10 PSD (dB re min)")
    save_plot(fig, os.path.join(plot_dir, f"{channel_name}_spectrogram_mcu.png"), show)


def plot_time_overlay(
    time_s: np.ndarray,
    signals: dict[str, np.ndarray],
    path: str,
    title: str,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(12, 4))
    for label, sig in signals.items():
        plt.plot(time_s, sig, label=label, alpha=0.8)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.title(title)
    plt.legend()
    save_plot(fig, path, show)


def plot_psd_overlay(
    freqs: np.ndarray,
    psd_dict: dict[str, np.ndarray],
    path: str,
    title: str,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(10, 5))
    for label, psd in psd_dict.items():
        db = 10 * np.log10(np.maximum(psd, 1e-20))
        plt.plot(freqs, db, label=label, alpha=0.8)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.title(title)
    plt.legend()
    save_plot(fig, path, show)


def plot_dual_comparison(time_s: np.ndarray,
                         baseline: np.ndarray,
                         nlms: np.ndarray,
                         wavelet: np.ndarray,
                         channel_name: str,
                         plot_dir: str,
                         show: bool,
                         baseline_label: str = "RAW") -> None:
    fig1 = plt.figure(figsize=(12, 4))
    plt.plot(time_s, baseline, label=baseline_label, alpha=0.7)
    plt.plot(time_s, nlms, label="NLMS", alpha=0.9)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.title(f"{channel_name}: {baseline_label} vs NLMS")
    plt.legend()
    save_plot(fig1, os.path.join(plot_dir, f"{channel_name}_raw_vs_nlms.png"), show)

    fig2 = plt.figure(figsize=(12, 4))
    plt.plot(time_s, baseline, label=baseline_label, alpha=0.7)
    plt.plot(time_s, wavelet, label="Wavelet(sym8)", alpha=0.9)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.title(f"{channel_name}: {baseline_label} vs Wavelet(sym8)")
    plt.legend()
    save_plot(fig2, os.path.join(plot_dir, f"{channel_name}_raw_vs_wavelet.png"), show)

    fig3 = plt.figure(figsize=(12, 4))
    plt.plot(time_s, baseline, label=baseline_label, alpha=0.6)
    plt.plot(time_s, nlms, label="NLMS", alpha=0.9)
    plt.plot(time_s, wavelet, label="Wavelet(sym8)", alpha=0.9)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.title(f"{channel_name}: {baseline_label} vs NLMS vs Wavelet(sym8)")
    plt.legend()
    save_plot(fig3, os.path.join(plot_dir, f"{channel_name}_raw_nlms_wavelet.png"), show)

    fig4 = plt.figure(figsize=(12, 4))
    plt.plot(time_s, nlms, label="NLMS", alpha=0.9)
    plt.plot(time_s, wavelet, label="Wavelet(sym8)", alpha=0.9)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.title(f"{channel_name}: NLMS vs Wavelet(sym8)")
    plt.legend()
    save_plot(fig4, os.path.join(plot_dir, f"{channel_name}_nlms_vs_wavelet.png"), show)


def plot_pairwise_vs_winsor(
    time_s: np.ndarray,
    signals: dict[str, np.ndarray],
    winsor: np.ndarray,
    channel_name: str,
    plot_dir: str,
    show: bool,
) -> None:
    """Plot each filter vs Winsor in time domain."""
    for label, sig in signals.items():
        if label == "Winsor":
            continue
        fig = plt.figure(figsize=(12, 4))
        plt.plot(time_s, sig, label=label, alpha=0.8)
        plt.plot(time_s, winsor, label="Winsor", alpha=0.8)
        plt.xlabel("time (s)")
        plt.ylabel("uV")
        plt.title(f"{channel_name}: {label} vs Winsor")
        plt.legend()
        safe_label = label.replace("+", "_").replace(" ", "_").lower()
        save_plot(fig, os.path.join(plot_dir, f"{channel_name}_{safe_label}_vs_winsor.png"), show)


def plot_pairwise_psd_vs_winsor(
    freqs: np.ndarray,
    psd_dict: dict[str, np.ndarray],
    channel_name: str,
    plot_dir: str,
    show: bool,
) -> None:
    """Plot each filter vs Winsor PSD (pairwise PSD comparison)."""
    if "Winsor" not in psd_dict:
        return
    psd_winsor = psd_dict["Winsor"]
    for label, psd in psd_dict.items():
        if label == "Winsor":
            continue
        min_len = min(len(freqs), len(psd), len(psd_winsor))
        if min_len < 2:
            continue
        fig = plt.figure(figsize=(10, 5))
        db_other = 10 * np.log10(np.maximum(psd[:min_len], 1e-20))
        db_winsor = 10 * np.log10(np.maximum(psd_winsor[:min_len], 1e-20))
        plt.plot(freqs[:min_len], db_other, label=label, alpha=0.9)
        plt.plot(freqs[:min_len], db_winsor, label="Winsor", alpha=0.9)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.title(f"{channel_name}: PSD {label} vs Winsor")
        plt.legend()
        safe_label = label.replace("+", "_").replace(" ", "_").lower()
        save_plot(fig, os.path.join(plot_dir, f"{channel_name}_psd_{safe_label}_vs_winsor.png"), show)


def ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, 0.0
    half = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(len(arr))
    return mean, half


def benchmark_filter(run_fn: Callable[[], np.ndarray], runs: int) -> dict[str, float]:
    wall_ms: list[float] = []
    cpu_pct: list[float] = []
    peak_mem_kib: list[float] = []

    for _ in range(max(1, runs)):
        tracemalloc.start()
        t0 = time.perf_counter()
        c0 = time.process_time()
        _ = run_fn()
        c1 = time.process_time()
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        wall = max(1e-12, t1 - t0)
        cpu = (c1 - c0) / wall * 100.0
        wall_ms.append(wall * 1000.0)
        cpu_pct.append(cpu)
        peak_mem_kib.append(peak / 1024.0)

    lat_mean, lat_ci = ci95(wall_ms)
    cpu_mean, cpu_ci = ci95(cpu_pct)
    mem_mean, mem_ci = ci95(peak_mem_kib)
    return {
        "latency_ms_mean": lat_mean,
        "latency_ms_ci95": lat_ci,
        "cpu_pct_mean": cpu_mean,
        "cpu_pct_ci95": cpu_ci,
        "mem_kib_mean": mem_mean,
        "mem_kib_ci95": mem_ci,
    }


def plot_metrics(metrics_df: pd.DataFrame, plot_dir: str, show: bool) -> None:
    labels = [f"{row['channel']}:{row['method']}" for _, row in metrics_df.iterrows()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].bar(x, metrics_df["latency_ms_mean"], yerr=metrics_df["latency_ms_ci95"], capsize=4)
    axes[0].set_ylabel("ms")
    axes[0].set_title("Latency per channel/method (95% CI)")

    axes[1].bar(x, metrics_df["cpu_pct_mean"], yerr=metrics_df["cpu_pct_ci95"], capsize=4)
    axes[1].set_ylabel("%")
    axes[1].set_title("CPU usage estimate (process_time / wall_time, 95% CI)")

    axes[2].bar(x, metrics_df["mem_kib_mean"], yerr=metrics_df["mem_kib_ci95"], capsize=4)
    axes[2].set_ylabel("KiB")
    axes[2].set_title("Peak Python memory per run (tracemalloc, 95% CI)")
    axes[2].set_xticks(x, labels, rotation=30, ha="right")

    complexity = "Big O: NLMS time O(N*L), memory O(L). Wavelet DWT time O(N), memory O(N)."
    fig.text(0.02, 0.01, complexity, fontsize=10)
    save_plot(fig, os.path.join(plot_dir, "performance_metrics_ci95.png"), show)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV/TSV file")
    parser.add_argument("--preset",
                        choices=["eeg", "ecg", "all"],
                        default="eeg",
                        help="Auto-select channels")
    parser.add_argument("--channels", help="Comma-separated channels (1..4) override preset")
    parser.add_argument("--reference-channel", type=int, default=3,
                        help="Reference ECG channel for NLMS on EEG (1..4)")
    parser.add_argument("--ecg-nlms-reference", type=int, default=1,
                        help="Reference channel for NLMS when processing ECG (default: eeg1)")
    parser.add_argument("--mode",
                        choices=["compare", "nlms", "wavelet"],
                        default="compare",
                        help="Filter mode")
    parser.add_argument("--scale",
                        choices=["auto", "ganglion", "none"],
                        default="auto",
                        help="Apply scaling to uV")
    parser.add_argument("--detrend",
                        choices=["none", "mean", "median"],
                        default="median",
                        help="Remove DC offset before filtering")
    parser.add_argument("--bandpass-low", type=float, default=1.0, help="Bandpass lower cutoff (Hz)")
    parser.add_argument("--bandpass-high", type=float, default=45.0, help="Bandpass upper cutoff (Hz)")
    parser.add_argument("--bandpass-order", type=int, default=4, help="Bandpass Butterworth order")
    parser.add_argument("--sampling-rate", type=float, default=200.0, help="Sampling rate (Hz)")
    parser.add_argument("--no-bandpass", action="store_true", help="Disable bandpass filter")
    parser.add_argument("--no-notch", action="store_true", help="Disable 50 Hz notch filter")
    parser.add_argument("--notch-freq", type=float, default=50.0, help="Notch center frequency (Hz)")
    parser.add_argument("--notch-Q", type=float, default=30.0, help="Notch quality factor (narrower=higher)")
    parser.add_argument("--nfft", type=int, default=512, help="Welch PSD window size")
    parser.add_argument("--winsor-clip", type=float, default=8.0, help="Winsor clip factor")
    parser.add_argument("--winsor-kernel", type=int, default=7, help="Winsor median kernel size")
    parser.add_argument("--nlms-taps", type=int, default=16, help="NLMS taps")
    parser.add_argument("--nlms-mu", type=float, default=0.06, help="NLMS step size")
    parser.add_argument("--nlms-eps", type=float, default=1e-6, help="NLMS epsilon")
    parser.add_argument("--nlms-delay", type=int, default=2, help="NLMS reference delay (samples)")
    parser.add_argument("--wavelet", default="sym8", help="Wavelet name (PyWavelets), e.g. sym8")
    parser.add_argument("--wavelet-level", type=int, default=0, help="Wavelet decomposition level, 0=auto")
    parser.add_argument("--wavelet-threshold-scale", type=float, default=1.0, help="Threshold scale multiplier")
    parser.add_argument("--benchmark-runs", type=int, default=20, help="Benchmark repetitions per channel/method")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Start time (s) for slicing data; omit for full range")
    parser.add_argument("--t-end", type=float, default=None,
                        help="End time (s) for slicing data; omit for full range")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--plot-dir", help="Directory for figures")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows")
    parser.add_argument(
        "--bands-mcu",
        action="store_true",
        help="MCU-matched theta/alpha/beta: sliding band power, Welch PSD+shaded bands, spectrogram",
    )
    parser.add_argument(
        "--bands-hop",
        type=int,
        default=64,
        help="Hop (samples) for sliding FFT band time series (default 64)",
    )
    args = parser.parse_args()

    if args.bands_mcu and args.bands_hop < 1:
        raise ValueError("--bands-hop must be >= 1")
    if args.bands_mcu and spectrogram is None:
        print("Warning: scipy.signal.spectrogram unavailable; spectrogram plots will be skipped.")

    channels = select_channels(args)
    if any(ch < 1 or ch > 4 for ch in channels):
        raise ValueError("Channels must be in range 1..4")
    if args.reference_channel < 1 or args.reference_channel > 4:
        raise ValueError("Reference channel must be in range 1..4")
    if args.ecg_nlms_reference < 1 or args.ecg_nlms_reference > 4:
        raise ValueError("ECG NLMS reference must be in range 1..4")
    if args.reference_channel not in channels and args.mode == "compare":
        print(f"Note: Reference (ECG) channel eeg{args.reference_channel} not in channels. Add it for ECG visualization.")

    pre_cfg = PreprocessConfig(scale_mode=args.scale, detrend=args.detrend)
    bp_cfg = BandpassConfig(low_hz=args.bandpass_low, high_hz=args.bandpass_high, order=args.bandpass_order)
    notch_cfg = NotchConfig(freq_hz=args.notch_freq, quality=args.notch_Q)
    nlms_cfg = NlmsConfig(taps=args.nlms_taps, mu=args.nlms_mu, eps=args.nlms_eps, delay_samples=args.nlms_delay)
    wav_cfg = WaveletConfig(wavelet=args.wavelet,
                            threshold_scale=args.wavelet_threshold_scale,
                            level=args.wavelet_level)
    winsor_cfg = WinsorConfig(clip_factor=args.winsor_clip, kernel_size=args.winsor_kernel)
    welch_cfg = WelchConfig(nfft=args.nfft)
    show_plots = not args.no_show
    fs = args.sampling_rate
    use_bandpass = not args.no_bandpass
    use_notch = not args.no_notch

    df, fmt = load_table(args.csv)
    all_chs = sorted(set(channels) | {args.reference_channel} | {args.ecg_nlms_reference})
    if fmt == "stream":
        time_s, selected_raw = extract_raw_stream(df, all_chs)
    elif fmt == "streamlit":
        time_s, selected_raw = extract_raw_streamlit(df, all_chs, fs)
        print(
            f"Note: Streamlit single-channel dump detected ({len(time_s)} samples); "
            f"replicated 'value' column to {', '.join(selected_raw.columns)}.",
        )
    else:
        time_s, selected_raw = extract_raw_brainflow(df, all_chs)

    if args.t_start is not None or args.t_end is not None:
        t_min = args.t_start if args.t_start is not None else time_s.min()
        t_max = args.t_end if args.t_end is not None else time_s.max()
        mask = (time_s >= t_min) & (time_s <= t_max)
        time_s = time_s[mask].copy()
        selected_raw = selected_raw.loc[mask].reset_index(drop=True)
        print(f"Time window: {t_min:.1f}-{t_max:.1f} s ({mask.sum()} samples)")

    channel_names = list(selected_raw.columns)

    ref_col = f"eeg{args.reference_channel}" if fmt == "brainflow" else f"ch{args.reference_channel}"
    ecg_ref_col = f"eeg{args.ecg_nlms_reference}" if fmt == "brainflow" else f"ch{args.ecg_nlms_reference}"
    if ref_col not in selected_raw.columns or ecg_ref_col not in selected_raw.columns:
        raise ValueError(f"Reference columns {ref_col}, {ecg_ref_col} must exist")

    print("=== Filter comparison pipeline started ===")
    print(f"Input file: {args.csv}")
    print(f"Reference (ECG) channel: {ref_col}")
    print(f"Channels: {channel_names}")
    print(f"Mode: {args.mode}, benchmark runs: {args.benchmark_runs}")
    print(f"Bandpass: {'enabled' if use_bandpass else 'disabled'} ({bp_cfg.low_hz}-{bp_cfg.high_hz} Hz)")
    print(f"Notch 50 Hz: {'enabled' if use_notch else 'disabled'}")

    for col in channel_names:
        selected_raw[col] = remove_dc(apply_scale(selected_raw[col].values, pre_cfg), pre_cfg.detrend)

    raw_signals = {col: selected_raw[col].values.astype(np.float64, copy=False) for col in channel_names}

    if use_bandpass:
        bandpass_signals = {col: apply_bandpass(raw_signals[col], bp_cfg, fs) for col in channel_names}
    else:
        bandpass_signals = {col: raw_signals[col].copy() for col in channel_names}
    if use_notch:
        bandpass_signals = {col: apply_notch(bandpass_signals[col], notch_cfg, fs) for col in channel_names}

    base_name, _ = os.path.splitext(args.csv)
    output_path = args.output or f"{base_name}_compare_filters.csv"
    plot_dir = args.plot_dir or f"{base_name}_plots"
    os.makedirs(plot_dir, exist_ok=True)

    out = pd.DataFrame({"time_s": time_s})
    metrics_rows: list[dict[str, float | str]] = []

    for col in channel_names:
        raw_ch = raw_signals[col]
        bp_ch = bandpass_signals[col]
        is_ecg = col == ref_col
        nlms_ref = bandpass_signals[ecg_ref_col] if is_ecg else bandpass_signals[ref_col]

        run_nlms = lambda r=bp_ch, ref=nlms_ref: nlms_adaptive_cancel(r.copy(), ref, nlms_cfg)
        run_wav = lambda r=bp_ch: wavelet_denoise(r.copy(), wav_cfg)
        run_winsor = lambda r=bp_ch: winsorize_masked_median(r.copy(), winsor_cfg)

        nlms_signal = run_nlms() if args.mode in ("compare", "nlms") else None
        wav_signal = run_wav() if args.mode in ("compare", "wavelet") else None
        winsor_signal = run_winsor() if args.mode == "compare" else None

        out[f"raw_{col}"] = raw_ch
        out[f"bandpass_{col}"] = bp_ch
        if nlms_signal is not None:
            out[f"nlms_{col}"] = nlms_signal
        if wav_signal is not None:
            out[f"wavelet_{col}"] = wav_signal
        if winsor_signal is not None:
            out[f"winsor_{col}"] = winsor_signal

        run_bp = lambda: apply_bandpass(raw_ch.copy(), bp_cfg, fs)
        for name, run_fn, sig in [
            ("Bandpass", run_bp, bp_ch if use_bandpass else None),
            ("NLMS", run_nlms, nlms_signal),
            ("Wavelet", run_wav, wav_signal),
            ("WinsorizedMedian", run_winsor, winsor_signal),
        ]:
            if sig is not None:
                m = benchmark_filter(run_fn, args.benchmark_runs)
                m["method"] = name
                m["channel"] = col
                metrics_rows.append(m)
                print(f"[{col}] {name} latency={m['latency_ms_mean']:.3f}±{m['latency_ms_ci95']:.3f} ms")

        if args.mode == "compare" and nlms_signal is not None and wav_signal is not None:
            bp_label = "Bandpass+Notch50" if (use_bandpass and use_notch) else ("Bandpass" if use_bandpass else "RAW")
            plot_dual_comparison(
                time_s, bp_ch, nlms_signal, wav_signal, col, plot_dir, show_plots,
                baseline_label=bp_label,
            )
            plot_time_overlay(
                time_s,
                {"RAW": raw_ch, bp_label: bp_ch},
                os.path.join(plot_dir, f"{col}_raw_vs_bandpass.png"),
                f"{col}: RAW vs {bp_label} ({bp_cfg.low_hz}-{bp_cfg.high_hz} Hz)",
                show_plots,
            )
            plot_time_overlay(
                time_s,
                {"RAW": raw_ch, bp_label: bp_ch, "NLMS": nlms_signal, "Wavelet": wav_signal, "Winsor": winsor_signal},
                os.path.join(plot_dir, f"{col}_pipeline_all.png"),
                f"{col}: Pipeline (RAW, {bp_label}, NLMS, Wavelet, Winsor)",
                show_plots,
            )
            freqs, _ = compute_psd_welch(raw_ch, fs, welch_cfg)
            psd_raw = compute_psd_welch(raw_ch, fs, welch_cfg)[1]
            psd_bp = compute_psd_welch(bp_ch, fs, welch_cfg)[1]
            psd_nlms = compute_psd_welch(nlms_signal, fs, welch_cfg)[1]
            psd_wav = compute_psd_welch(wav_signal, fs, welch_cfg)[1]
            psd_winsor = compute_psd_welch(winsor_signal, fs, welch_cfg)[1]
            min_len = min(len(freqs), len(psd_raw), len(psd_bp), len(psd_nlms), len(psd_wav), len(psd_winsor))
            plot_psd_overlay(
                freqs[:min_len],
                {
                    "RAW": psd_raw[:min_len],
                    bp_label: psd_bp[:min_len],
                    "NLMS": psd_nlms[:min_len],
                    "Wavelet": psd_wav[:min_len],
                    "Winsor": psd_winsor[:min_len],
                },
                os.path.join(plot_dir, f"{col}_psd_pipeline.png"),
                f"{col}: Welch PSD (RAW, {bp_label}, NLMS, Wavelet, Winsor)",
                show_plots,
            )
            signals_for_winsor = {
                "RAW": raw_ch,
                bp_label: bp_ch,
                "NLMS": nlms_signal,
                "Wavelet": wav_signal,
            }
            plot_pairwise_vs_winsor(
                time_s, signals_for_winsor, winsor_signal, col, plot_dir, show_plots
            )
            psd_for_winsor = {
                "RAW": psd_raw[:min_len],
                bp_label: psd_bp[:min_len],
                "NLMS": psd_nlms[:min_len],
                "Wavelet": psd_wav[:min_len],
                "Winsor": psd_winsor[:min_len],
            }
            plot_pairwise_psd_vs_winsor(
                freqs[:min_len], psd_for_winsor, col, plot_dir, show_plots
            )
        elif args.mode == "nlms" and nlms_signal is not None:
            plot_time_overlay(
                time_s,
                {"RAW": raw_ch, "Bandpass": bp_ch, "NLMS": nlms_signal},
                os.path.join(plot_dir, f"{col}_raw_vs_nlms.png"),
                f"{col}: RAW vs Bandpass vs NLMS",
                show_plots,
            )
        elif args.mode == "wavelet" and wav_signal is not None:
            plot_time_overlay(
                time_s,
                {"RAW": raw_ch, "Bandpass": bp_ch, "Wavelet": wav_signal},
                os.path.join(plot_dir, f"{col}_raw_vs_wavelet.png"),
                f"{col}: RAW vs Bandpass vs Wavelet",
                show_plots,
            )

        if args.bands_mcu:
            # Uses bp_ch (bandpass+notch when enabled) to align with MCU pipeline before FFT.
            bp_suffix = (
                "(bandpass+notch pipeline)"
                if (use_bandpass and use_notch)
                else ("(bandpass)" if use_bandpass else "(unfiltered)")
            )
            if welch is not None:
                fw, pw = compute_psd_welch(bp_ch, fs, welch_cfg)
                plot_welch_psd_with_mcu_bands(fw, pw, col, plot_dir, show_plots, bp_suffix)
            else:
                print(f"[{col}] --bands-mcu: Welch PSD skipped (scipy welch missing)")
            t_b, th, al, be = mcu_sliding_band_powers(bp_ch, fs, MCU_NFFT, args.bands_hop)
            if len(t_b) > 0:
                plot_mcu_band_timeseries(t_b, th, al, be, col, plot_dir, show_plots, args.bands_hop)
            else:
                print(
                    f"[{col}] --bands-mcu: band time series skipped "
                    f"(need >= {MCU_NFFT} samples, got {len(bp_ch)})",
                )
            if spectrogram is not None:
                try:
                    plot_spectrogram_mcu(bp_ch, fs, col, plot_dir, show_plots)
                except Exception as exc:
                    print(f"[{col}] spectrogram failed: {exc}")
            else:
                print(f"[{col}] --bands-mcu: spectrogram skipped (scipy spectrogram missing)")

    out.to_csv(output_path, index=False)
    print(f"Saved filtered comparison CSV: {output_path}")

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = os.path.join(plot_dir, "performance_metrics_ci95.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics CSV: {metrics_csv}")
        plot_metrics(metrics_df, plot_dir, show_plots)

    print("=== Done ===")


if __name__ == "__main__":
    main()
