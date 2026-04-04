"""
Render MCU block processing pipeline in same style as PC pipeline diagram.
Output: docs/mcu_block_diagram.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_W, FIG_H = 14, 6
BOX_W, BOX_H = 1.6, 0.8
GAP_X, GAP_Y = 0.4, 0.5

# Same palette as pipeline_diagram.py
COLOR_PREPROC = "#E8F5E9"
COLOR_FILTER = "#FFF3E0"
COLOR_FFT = "#E3F2FD"
COLOR_ATTENTION = "#F3E5F5"
COLOR_ECG = "#FFEBEE"
COLOR_OUTPUT = "#E0F7FA"
EDGE = "#37474F"
TEXT = "#263238"

def add_box(ax, x, y, label, color, fontsize=9):
    box = FancyBboxPatch((x, y), BOX_W, BOX_H,
                         boxstyle="round,pad=0.03,rounding_size=0.1",
                         facecolor=color, edgecolor=EDGE, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + BOX_W/2, y + BOX_H/2, label, ha="center", va="center",
            fontsize=fontsize, color=TEXT)

def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=EDGE, lw=2))

def main():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    y_top = FIG_H - 1.8
    y_mid = FIG_H / 2
    y_bot = 1.0
    x = 0.6

    # Block Processing header
    ax.text(FIG_W/2, FIG_H - 0.4, "Block processing (every 1.28 s)", fontsize=12,
            ha="center", va="top", color="#546E7A", weight="bold")

    # 1. DC removal
    add_box(ax, x, y_top - BOX_H/2, "Remove DC offset\n(subtract mean per channel)", COLOR_PREPROC, fontsize=8)
    add_arrow(ax, x + BOX_W, y_top, x + BOX_W + GAP_X, y_top)
    x += BOX_W + GAP_X

    # 2. Re-filter
    add_box(ax, x, y_top - BOX_H/2, "Bandpass + Notch 50 Hz\n→ filtered buffer", COLOR_FILTER, fontsize=8)
    ref_right = x + BOX_W
    ref_cy = y_top
    x += BOX_W + GAP_X

    # Branch: FFT ch1, FFT ch2, ECG (parallel)
    fft1_y, fft2_y, ecg_y = y_top + 0.6, y_mid, y_bot + 0.4
    fft_x = x + 0.4

    for dy in [fft1_y, fft2_y, ecg_y]:
        add_arrow(ax, ref_right, ref_cy, fft_x, dy)

    add_box(ax, fft_x, fft1_y - BOX_H/2, "FFT on Ch1 (EEG)\n→ band powers", COLOR_FFT, fontsize=8)
    add_box(ax, fft_x, fft2_y - BOX_H/2, "FFT on Ch2 (EEG)\n→ band powers", COLOR_FFT, fontsize=8)
    add_box(ax, fft_x, ecg_y - BOX_H/2, "R-peak detection\non Ch3 (ECG) → BPM", COLOR_ECG, fontsize=8)

    # Attention = avg(ch1, ch2) — FFT1 and FFT2 merge here
    att_x = fft_x + BOX_W + GAP_X
    add_arrow(ax, fft_x + BOX_W, fft1_y, att_x, y_mid + 0.15)
    add_arrow(ax, fft_x + BOX_W, fft2_y, att_x, y_mid)
    add_box(ax, att_x, y_mid - BOX_H/2, "Attention Index\n= avg(Ch1, Ch2)", COLOR_ATTENTION, fontsize=8)

    # Send packet — Attention and ECG merge here
    send_x = att_x + BOX_W + GAP_X
    add_arrow(ax, att_x + BOX_W, y_mid, send_x, y_mid)
    add_arrow(ax, fft_x + BOX_W, ecg_y, send_x, y_mid)
    add_box(ax, send_x, y_mid - BOX_H/2, "Send packet\n(Serial / BLE)", COLOR_OUTPUT, fontsize=8)

    # MCU badge
    ax.text(FIG_W - 0.4, FIG_H - 0.25, "MCU", fontsize=12, color="#78909C",
            ha="right", va="top", style="italic", weight="bold")

    plt.tight_layout()
    out_path = "docs/mcu_block_diagram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
