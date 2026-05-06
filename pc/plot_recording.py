"""Plot timeseries + rolling SNR from a biofeedback recording CSV."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("biofeedback_recording_20260506_222754.csv")
WINDOW = 10  # samples (~14 s at 1 packet / 1.4 s)

df = pd.read_csv(CSV)
t = df["elapsed_time"].values

signals = {
    "Attention": df["attention"].values,
    "Theta":     df["theta"].values,
    "Alpha":     df["alpha"].values,
    "Beta":      df["beta"].values,
    "BPM":       df["bpm"].values,
}
motor = df["motor_state"].values

# --- Figure 1: timeseries ---
fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
log_signals = {"Theta", "Alpha", "Beta"}
for ax, (name, y) in zip(axes[:5], signals.items()):
    ax.plot(t, y, lw=0.8)
    ax.set_ylabel(name)
    ax.grid(alpha=0.3)
    if name in log_signals:
        ax.set_yscale("log")
axes[5].step(t, motor, where="post", color="red")
axes[5].set_ylabel("Motor")
axes[5].set_yticks([0, 1])
axes[5].set_xlabel("elapsed_time [s]")
axes[5].grid(alpha=0.3)
fig.suptitle(f"Biofeedback timeseries — {CSV.name}")
fig.tight_layout()
out1 = CSV.with_name(CSV.stem + "_timeseries.png")
fig.savefig(out1, dpi=120)
print(f"saved {out1}")

# --- Figure 2: rolling SNR (dB) ---
# SNR_dB = 20 * log10( mean / std ) over a moving window.
# For bandpowers we compute in log domain to tame heavy-tailed outliers.
def rolling_snr_db(x, win, log=False):
    s = pd.Series(np.log10(np.clip(x, 1e-12, None)) if log else x)
    m = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std()
    snr = m / sd.replace(0, np.nan)
    return 20 * np.log10(np.abs(snr).replace(0, np.nan))

fig2, axes2 = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
for ax, (name, y) in zip(axes2, signals.items()):
    use_log = name in log_signals
    snr = rolling_snr_db(y, WINDOW, log=use_log)
    ax.plot(t, snr, lw=0.9)
    label = f"{name} (log-domain)" if use_log else name
    ax.set_ylabel(f"SNR [dB]\n{label}")
    ax.grid(alpha=0.3)
axes2[-1].set_xlabel("elapsed_time [s]")
fig2.suptitle(f"Rolling SNR (window={WINDOW}) — {CSV.name}")
fig2.tight_layout()
out2 = CSV.with_name(CSV.stem + "_snr.png")
fig2.savefig(out2, dpi=120)
print(f"saved {out2}")
