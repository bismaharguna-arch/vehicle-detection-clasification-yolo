"""Render diagram arsitektur sistem deteksi kendaraan ke PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")

# ---- warna ----
C_SECT = "#f5f6fa"
C_INPUT = "#3498db"
C_YOLO = "#9b59b6"
C_FUEL = "#1abc9c"
C_OCR = "#e67e22"
C_TRACK = "#e74c3c"
C_PAY = "#34495e"
C_OUT = "#2c3e50"
WHITE = "white"

def box(x, y, w, h, color, title, sub="", fc_text=WHITE, fontsize=12, sub_fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                       linewidth=2, edgecolor=color, facecolor=color, alpha=0.95, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(x + w/2, y + h*0.64, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=fc_text, zorder=4)
        ax.text(x + w/2, y + h*0.30, sub, ha="center", va="center",
                fontsize=sub_fs, color=fc_text, zorder=4)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=fc_text, zorder=4)

def section(x, y, w, h, label):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                       linewidth=2, edgecolor="#bdc3c7", facecolor=C_SECT, zorder=1)
    ax.add_patch(p)
    ax.text(x + w/2, y + h - 0.35, label, ha="center", va="center",
            fontsize=15, fontweight="bold", color="#7f8c8d", zorder=2)

def arrow(x1, y1, x2, y2, color="#2c3e50", style="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=22, linewidth=2.2, color=color, zorder=5,
                 linestyle="--" if dashed else "-"))

# ===== SECTIONS =====
section(0.3, 0.5, 2.6, 8.0, "INPUT")
section(3.2, 0.5, 8.4, 8.0, "PROSES")
section(11.9, 0.5, 2.8, 8.0, "OUTPUT")

# ===== INPUT =====
box(0.7, 4.0, 1.8, 1.2, C_INPUT, "Kamera /", "Video", fontsize=12, sub_fs=11)

# ===== PROSES (alur vertikal) =====
bx, bw = 4.3, 6.2
box(bx, 7.0, bw, 1.0, C_YOLO, "YOLOv8  -  2 Model  (di atas PyTorch)",
    "yolov8n.pt -> kendaraan   |   license_plate -> plat", sub_fs=9.5)
box(bx, 5.6, bw, 1.0, C_FUEL, "FuelClassifier",
    "HSV strip biru (OpenCV)  ->  bensin / listrik", sub_fs=9.5)
box(bx, 4.2, bw, 1.0, C_OCR, "PaddleOCR",
    "+ NumPy + OpenCV  ->  teks plat", sub_fs=9.5)
box(bx, 2.8, bw, 1.0, C_TRACK, "VehicleTracker",
    "anti-spam, kirim 1x (dedup IoU)", sub_fs=9.5)
box(bx, 1.2, bw, 1.1, C_PAY, "Payload",
    "foto JPG + data form\n(plate_number, vehicle_type, is_electric,\nconfidence_score, detected_at, photo)", sub_fs=8)

# ===== OUTPUT =====
box(12.2, 3.9, 2.2, 1.4, C_OUT, "HTTP POST",
    "multipart/form-data\n/api/detections", sub_fs=9.5)

# ===== ARROWS =====
arrow(2.5, 4.6, bx, 7.5)            # input -> yolo
arrow(bx + bw/2, 7.0, bx + bw/2, 6.6)   # yolo -> fuel
arrow(bx + bw/2, 5.6, bx + bw/2, 5.2)   # fuel -> ocr
arrow(bx + bw/2, 4.2, bx + bw/2, 3.8)   # ocr -> track
arrow(bx + bw/2, 2.8, bx + bw/2, 2.3)   # track -> payload
arrow(bx + bw, 1.75, 12.2, 4.3)     # payload -> output

ax.text(7.5, 8.75, "Sistem Deteksi Kendaraan  -  Alur Frame -> Web",
        ha="center", fontsize=16, fontweight="bold", color="#2c3e50")

plt.tight_layout()
out = "output/arsitektur_sistem.png"
import os
os.makedirs("output", exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("saved:", out)
