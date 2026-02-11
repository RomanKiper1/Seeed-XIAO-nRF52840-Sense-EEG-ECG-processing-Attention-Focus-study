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


GANGLION_UV_PER_COUNT = 0.001869917138805


@dataclass
class PreprocessConfig:
    scale_mode: str = "auto"
    detrend: str = "median"


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


def load_table(path: str) -> tuple[pd.DataFrame, str]:
    sep = detect_separator(path)
    has_header = detect_header(path, sep)
    df = pd.read_csv(path, sep=sep, header=0 if has_header else None)
    if has_header:
        return df, "stream"

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


def plot_dual_comparison(time_s: np.ndarray,
                         raw: np.ndarray,
                         nlms: np.ndarray,
                         wavelet: np.ndarray,
                         channel_name: str,
                         plot_dir: str,
                         show: bool) -> None:
    fig1 = plt.figure(figsize=(12, 4))
    plt.plot(time_s, raw, label="RAW", alpha=0.7)
    plt.plot(time_s, nlms, label="NLMS", alpha=0.9)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.title(f"{channel_name}: RAW vs NLMS")
    plt.legend()
    save_plot(fig1, os.path.join(plot_dir, f"{channel_name}_raw_vs_nlms.png"), show)

    fig2 = plt.figure(figsize=(12, 4))
    plt.plot(time_s, raw, label="RAW", alpha=0.7)
    plt.plot(time_s, wavelet, label="Wavelet(sym8)", alpha=0.9)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.title(f"{channel_name}: RAW vs Wavelet(sym8)")
    plt.legend()
    save_plot(fig2, os.path.join(plot_dir, f"{channel_name}_raw_vs_wavelet.png"), show)

    fig3 = plt.figure(figsize=(12, 4))
    plt.plot(time_s, raw, label="RAW", alpha=0.6)
    plt.plot(time_s, nlms, label="NLMS", alpha=0.9)
    plt.plot(time_s, wavelet, label="Wavelet(sym8)", alpha=0.9)
    plt.xlabel("time (s)")
    plt.ylabel("uV")
    plt.title(f"{channel_name}: RAW vs NLMS vs Wavelet(sym8)")
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
    parser.add_argument("--reference-channel", type=int, default=3, help="Reference ECG-like channel (1..4)")
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
    parser.add_argument("--nlms-taps", type=int, default=16, help="NLMS taps")
    parser.add_argument("--nlms-mu", type=float, default=0.06, help="NLMS step size")
    parser.add_argument("--nlms-eps", type=float, default=1e-6, help="NLMS epsilon")
    parser.add_argument("--nlms-delay", type=int, default=2, help="NLMS reference delay (samples)")
    parser.add_argument("--wavelet", default="sym8", help="Wavelet name (PyWavelets), e.g. sym8")
    parser.add_argument("--wavelet-level", type=int, default=0, help="Wavelet decomposition level, 0=auto")
    parser.add_argument("--wavelet-threshold-scale", type=float, default=1.0, help="Threshold scale multiplier")
    parser.add_argument("--benchmark-runs", type=int, default=20, help="Benchmark repetitions per channel/method")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--plot-dir", help="Directory for figures")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows")
    args = parser.parse_args()

    channels = select_channels(args)
    if any(ch < 1 or ch > 4 for ch in channels):
        raise ValueError("Channels must be in range 1..4")
    if args.reference_channel < 1 or args.reference_channel > 4:
        raise ValueError("Reference channel must be in range 1..4")

    pre_cfg = PreprocessConfig(scale_mode=args.scale, detrend=args.detrend)
    nlms_cfg = NlmsConfig(taps=args.nlms_taps, mu=args.nlms_mu, eps=args.nlms_eps, delay_samples=args.nlms_delay)
    wav_cfg = WaveletConfig(wavelet=args.wavelet,
                            threshold_scale=args.wavelet_threshold_scale,
                            level=args.wavelet_level)
    show_plots = not args.no_show

    df, fmt = load_table(args.csv)
    if fmt == "stream":
        time_s, selected_raw = extract_raw_stream(df, channels)
        _, ref_raw_df = extract_raw_stream(df, [args.reference_channel])
    else:
        time_s, selected_raw = extract_raw_brainflow(df, channels)
        _, ref_raw_df = extract_raw_brainflow(df, [args.reference_channel])

    channel_names = list(selected_raw.columns)
    ref_col = ref_raw_df.columns[0]

    for col in channel_names:
        selected_raw[col] = remove_dc(apply_scale(selected_raw[col].values, pre_cfg), pre_cfg.detrend)
    reference = remove_dc(apply_scale(ref_raw_df[ref_col].values, pre_cfg), pre_cfg.detrend)

    base_name, _ = os.path.splitext(args.csv)
    output_path = args.output or f"{base_name}_compare_filters.csv"
    plot_dir = args.plot_dir or f"{base_name}_plots"
    os.makedirs(plot_dir, exist_ok=True)

    out = pd.DataFrame()
    metrics_rows: list[dict[str, float | str]] = []

    print("=== Filter comparison pipeline started ===")
    print(f"Input file: {args.csv}")
    print(f"Channels: {channel_names}, reference: {ref_col}")
    print(f"Mode: {args.mode}, benchmark runs: {args.benchmark_runs}")

    for col in channel_names:
        raw_ch = selected_raw[col].values.astype(np.float64, copy=False)
        out[f"raw_{col}"] = raw_ch

        run_nlms = lambda: nlms_adaptive_cancel(raw_ch, reference, nlms_cfg)
        run_wav = lambda: wavelet_denoise(raw_ch, wav_cfg)

        nlms_signal = run_nlms() if args.mode in ("compare", "nlms") else None
        wav_signal = run_wav() if args.mode in ("compare", "wavelet") else None

        if nlms_signal is not None:
            out[f"nlms_{col}"] = nlms_signal
            m = benchmark_filter(run_nlms, args.benchmark_runs)
            m["method"] = "NLMS"
            m["channel"] = col
            metrics_rows.append(m)
            print(f"[{col}] NLMS latency={m['latency_ms_mean']:.3f}±{m['latency_ms_ci95']:.3f} ms, "
                  f"cpu={m['cpu_pct_mean']:.1f}±{m['cpu_pct_ci95']:.1f} %, "
                  f"mem={m['mem_kib_mean']:.1f}±{m['mem_kib_ci95']:.1f} KiB")

        if wav_signal is not None:
            out[f"wavelet_{col}"] = wav_signal
            m = benchmark_filter(run_wav, args.benchmark_runs)
            m["method"] = "Wavelet"
            m["channel"] = col
            metrics_rows.append(m)
            print(f"[{col}] Wavelet latency={m['latency_ms_mean']:.3f}±{m['latency_ms_ci95']:.3f} ms, "
                  f"cpu={m['cpu_pct_mean']:.1f}±{m['cpu_pct_ci95']:.1f} %, "
                  f"mem={m['mem_kib_mean']:.1f}±{m['mem_kib_ci95']:.1f} KiB")

        if args.mode == "compare" and nlms_signal is not None and wav_signal is not None:
            plot_dual_comparison(time_s, raw_ch, nlms_signal, wav_signal, col, plot_dir, show_plots)
        elif args.mode == "nlms" and nlms_signal is not None:
            fig = plt.figure(figsize=(12, 4))
            plt.plot(time_s, raw_ch, label="RAW", alpha=0.7)
            plt.plot(time_s, nlms_signal, label="NLMS", alpha=0.9)
            plt.title(f"{col}: RAW vs NLMS")
            plt.xlabel("time (s)")
            plt.ylabel("uV")
            plt.legend()
            save_plot(fig, os.path.join(plot_dir, f"{col}_raw_vs_nlms.png"), show_plots)
        elif args.mode == "wavelet" and wav_signal is not None:
            fig = plt.figure(figsize=(12, 4))
            plt.plot(time_s, raw_ch, label="RAW", alpha=0.7)
            plt.plot(time_s, wav_signal, label="Wavelet(sym8)", alpha=0.9)
            plt.title(f"{col}: RAW vs Wavelet(sym8)")
            plt.xlabel("time (s)")
            plt.ylabel("uV")
            plt.legend()
            save_plot(fig, os.path.join(plot_dir, f"{col}_raw_vs_wavelet.png"), show_plots)

    save_filtered_csv(time_s, out, output_path)
    print(f"Saved filtered comparison CSV: {output_path}")

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv = os.path.join(plot_dir, "performance_metrics_ci95.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics CSV: {metrics_csv}")
        plot_metrics(metrics_df, plot_dir, show_plots)
        print(f"Saved metrics chart: {os.path.join(plot_dir, 'performance_metrics_ci95.png')}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
