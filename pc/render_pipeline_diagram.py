"""
Render PC signal processing pipeline as a clean flowchart (no red line).
Output: docs/pipeline_diagram.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_W, FIG_H = 16, 5
BOX_W, BOX_H = 1.5, 0.75
GAP_X = 0.35

# Soft, professional palette
COLOR_ACQUIRE = "#E3F2FD"
COLOR_PREPROC = "#E8F5E9"
COLOR_FILTER = "#FFF3E0"
COLOR_DENOISE = "#F3E5F5"
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
                arrowprops=dict(arrowstyle="->", color=EDGE, lw=2,
                               connectionstyle="arc3,rad=0"))

def main():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    y = FIG_H / 2
    x = 0.5

    # Linear chain: BleReceiver -> Parser -> RawBuffer -> Preprocess -> Bandpass
    chain = [
        ("BleReceiver", COLOR_ACQUIRE),
        ("GanglionPacketParser", COLOR_ACQUIRE),
        ("SignalBufferRaw", COLOR_ACQUIRE),
        ("Preprocess\n(scale+detrend)", COLOR_PREPROC),
        ("Bandpass 1–45 Hz\n+ Notch 50 Hz", COLOR_FILTER),
    ]
    for i, (label, color) in enumerate(chain):
        add_box(ax, x, y - BOX_H/2, label, color, fontsize=8)
        if i < len(chain) - 1:
            add_arrow(ax, x + BOX_W, y, x + BOX_W + GAP_X, y)
        x += BOX_W + GAP_X

    bp_cx = x - GAP_X  # right edge of bandpass box

    # Three denoisers in parallel
    denoise = [
        ("NLMS", y + 0.7),
        ("Wavelet (Sym8)", y),
        ("Winsorized Median", y - 0.7),
    ]
    dn_x = bp_cx + GAP_X + 0.3
    for label, dy in denoise:
        add_arrow(ax, bp_cx, y, dn_x, dy)
        add_box(ax, dn_x, dy - BOX_H/2, label, COLOR_DENOISE, fontsize=8)

    # Output: all converge
    out_x = dn_x + BOX_W + GAP_X + 0.5
    add_box(ax, out_x, y - BOX_H/2, "Plots + PSD\nWelch Metrics", COLOR_OUTPUT, fontsize=8)
    for _, dy in denoise:
        add_arrow(ax, dn_x + BOX_W, dy, out_x, y)

    # Subtle "PC" badge (no red line)
    ax.text(FIG_W - 0.4, FIG_H - 0.25, "PC", fontsize=12, color="#78909C",
            ha="right", va="top", style="italic", weight="bold")

    plt.tight_layout()
    out_path = "docs/pipeline_diagram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
