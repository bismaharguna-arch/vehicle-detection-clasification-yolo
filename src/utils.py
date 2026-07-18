"""
utils.py
Fungsi helper untuk visualisasi dan operasi umum.
"""

import cv2
import numpy as np

# Warna BGR untuk setiap class (BGR, bukan RGB!)
CLASS_COLORS = {
    'mobil': (255, 100, 0),    # Biru terang
    'motor': (0, 200, 255),    # Kuning-orange
    'plat':  (0, 255, 100),    # Hijau terang
}

# Warna BGR untuk jenis bahan bakar (dipakai untuk label plat)
FUEL_COLORS = {
    'listrik': (255, 150, 0),  # Biru
    'bensin':  (0, 140, 255),  # Oranye
}

# Label tampilan untuk jenis bahan bakar. HANYA untuk overlay -- nilai internal
# & payload web tetap 'bensin' (kontrak v1 is_electric = 'bensin'|'listrik'|
# 'unknown'; nilai lain di-coerce server jadi 'unknown', jangan ubah).
FUEL_DISPLAY = {
    'bensin': 'bbm',
}

DEFAULT_COLOR = (200, 200, 200)  # Abu-abu untuk class yang tidak dikenal


def draw_detections(frame, detections):
    """
    Gambar bounding box + label untuk setiap deteksi.
    
    Args:
        frame: numpy array (gambar BGR)
        detections: list of dict dari VehicleDetector.detect()
    
    Returns:
        frame yang sudah digambar (modified in-place dan di-return)
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']

        # Pilih warna kotak sesuai class
        box_color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        # Gambar kotak
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Buat label. Untuk plat, tampilkan jenis bahan bakar + teks OCR + warnai label.
        label = f'{class_name} {confidence:.2f}'
        label_color = box_color
        if class_name == 'plat':
            fuel = det.get('fuel_type', '')
            text = det.get('text', '')
            if fuel:
                label = f'{label} | {FUEL_DISPLAY.get(fuel, fuel)}'
                label_color = FUEL_COLORS.get(fuel, box_color)
            if text:
                label = f'{label} | {text}'
        color = label_color
        
        # Hitung ukuran teks supaya background pas
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Background untuk teks (biar ke-baca di latar apapun)
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w + 5, y1),
            color,
            -1  # Filled
        )
        
        # Tulis teks
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # Teks hitam
            2
        )
    
    return frame


def draw_fps(frame, fps):
    """
    Tulis FPS di pojok kiri atas frame.
    
    Args:
        frame: numpy array (gambar BGR)
        fps: float, frame per second
    
    Returns:
        frame yang sudah digambar
    """
    text = f'FPS: {fps:.1f}'
    
    # Background hitam transparan
    cv2.rectangle(frame, (10, 10), (140, 50), (0, 0, 0), -1)
    
    # Teks putih
    cv2.putText(
        frame, text, (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    
    return frame


def draw_detection_count(frame, detections):
    """
    Tulis jumlah deteksi per class di pojok kanan atas.
    
    Args:
        frame: numpy array
        detections: list of dict
    
    Returns:
        frame yang sudah digambar
    """
    # Hitung per class
    counts = {'mobil': 0, 'motor': 0, 'plat': 0}
    for det in detections:
        if det['class_name'] in counts:
            counts[det['class_name']] += 1
    
    # Posisi mulai (kanan atas)
    h, w = frame.shape[:2]
    x_start = w - 180
    y_start = 30
    
    # Background
    cv2.rectangle(frame, (x_start - 10, 10), (w - 10, 130), (0, 0, 0), -1)
    
    # Tulis per class
    for i, (cls_name, count) in enumerate(counts.items()):
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        text = f'{cls_name}: {count}'
        cv2.putText(
            frame, text, (x_start, y_start + i * 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    
    return frame