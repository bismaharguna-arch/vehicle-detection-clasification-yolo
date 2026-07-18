"""
fuel_classifier.py
Klasifikasi bahan bakar kendaraan berdasarkan plat:
- 'listrik' kalau ada garis biru di bagian bawah plat (plat khusus EV).
- 'bensin'  kalau tidak.
Pakai deteksi warna HSV.
"""

import cv2
import numpy as np


class FuelClassifier:
    def __init__(self,
                 strip_coverage_threshold=0.5,
                 focus_bottom_ratio=0.4,
                 extend_below_ratio=0.5,
                 hsv_lower=(95, 35, 60),
                 hsv_upper=(135, 255, 255),
                 strip_sat_ratio=1.1):
        """
        Args:
            strip_coverage_threshold: Minimum lebar plat yang harus tertutup garis
                                  biru di SATU baris (0-1). >= threshold -> listrik.
                                  Default 0.5 = ada baris yang biru >=50% lebar plat.
            focus_bottom_ratio:   Bagian bawah plat yang dicek (0-1).
                                  Default 0.4 = bawah 40% (lokasi garis biru EV).
            extend_below_ratio:   Perpanjang bbox ke bawah sebesar rasio tinggi box,
                                  karena detektor plat ngebox ketat di baris nomor &
                                  strip EV ada di bawahnya. Default 0.5.
            hsv_lower / hsv_upper: Range warna biru di HSV (OpenCV).
                                  Dikalibrasi dari plat EV asli (screenshot webcam,
                                  Jul 2026): strip asli S cuma 40-90 (biru pudar di
                                  kamera), sedangkan bodi putih/latar S<25. Ambang
                                  S=35 duduk di celah itu. Kalau ganti kamera/lokasi
                                  dan mulai salah, ukur ulang S strip vs latar.
        """
        # ponytail: ganti dari "total rasio piksel biru" ke "lebar baris paling
        # biru". Rasio total ke-trigger band biru tepi-kiri plat UK & noise huruf
        # gelap (false-positive ~55% di video tes). Strip EV Indonesia itu garis
        # HORIZONTAL selebar plat -> diukur lewat baris yg biru-nya rentang lebar,
        # bukan jumlah piksel. BELUM divalidasi di plat EV Indonesia asli; kalau
        # strip aslinya tipis/redup & ga ke-detect, turunkan threshold ini.
        self.strip_coverage_threshold = strip_coverage_threshold
        self.focus_bottom_ratio = focus_bottom_ratio
        self.extend_below_ratio = extend_below_ratio
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        # Anti-cast: strip bawah harus LEBIH pekat (saturasi) dari area teks
        # plat sebesar rasio ini. Kamera dengan white-balance biru bikin SELURUH
        # plat kehitung biru (diukur: plat UK ber-cast, top S>=bot S); strip EV
        # asli selalu lebih pekat dari permukaan teksnya (63 vs 53).
        self.strip_sat_ratio = strip_sat_ratio

    def classify(self, frame, bbox):
        """
        Args:
            frame: numpy array BGR (full frame).
            bbox:  (x1, y1, x2, y2) area plat.

        Returns:
            (fuel_type, blue_ratio)
            - fuel_type: 'listrik' atau 'bensin'
            - blue_ratio: float, rasio piksel biru di area fokus (0-1)
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        if x2 <= x1 or y2 <= y1:
            return 'bensin', 0.0

        # Detektor plat ngebox ketat di baris nomor doang -> strip biru EV ada di
        # tepi BAWAH plat, di luar box. Perpanjang ke bawah biar strip ke-cover.
        box_h = y2 - y1
        y2 = min(h_frame, y2 + int(box_h * self.extend_below_ratio))

        crop = frame[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return 'bensin', 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        focus_y = int(h * (1 - self.focus_bottom_ratio))
        bot = mask[focus_y:, :]
        if bot.size == 0:
            return 'bensin', 0.0

        # Skor = baris paling biru di area strip: fraksi LEBAR plat yang biru.
        # Garis EV horizontal -> ada baris dgn coverage tinggi. Band tepi-kiri
        # UK / noise -> tiap baris cuma biru di sebagian kecil lebar -> skor rendah.
        strip_score = float((bot > 0).mean(axis=1).max())
        if strip_score < self.strip_coverage_threshold:
            return 'bensin', strip_score

        # Anti-cast biru: kalau AREA TEKS plat juga banyak biru dan strip bawah
        # TIDAK lebih pekat darinya, itu cast white-balance / plat berdasar biru
        # (mis. UK), bukan strip EV -> bensin.
        S = hsv[:, :, 1]
        top_mask = mask[:int(h * 0.5), :]
        top_s = S[:int(h * 0.5), :][top_mask > 0]
        bot_s = S[focus_y:, :][bot > 0]
        if top_s.size >= 10 and bot_s.size >= 10:
            if float(np.median(bot_s)) < float(np.median(top_s)) * self.strip_sat_ratio:
                return 'bensin', strip_score

        return 'listrik', strip_score


if __name__ == '__main__':
    # ponytail self-check: strip biru penuh-lebar di bawah = listrik;
    # band biru di tepi kiri (ala plat UK) = bensin.
    fc = FuelClassifier()
    blue = (255, 50, 0)  # BGR biru
    plate = np.full((40, 200, 3), 255, np.uint8)  # plat putih
    ev = plate.copy(); ev[34:, :] = blue           # garis penuh-lebar di bawah
    uk = plate.copy(); uk[:, :12] = blue           # band tepi kiri
    cast = np.full((40, 200, 3), (200, 165, 140), np.uint8)  # seluruh plat biru pudar
    assert fc.classify(ev, (0, 0, 200, 40))[0] == 'listrik', 'strip EV harus listrik'
    assert fc.classify(uk, (0, 0, 200, 40))[0] == 'bensin', 'band kiri harus bensin'
    assert fc.classify(cast, (0, 0, 200, 40))[0] == 'bensin', 'cast biru merata harus bensin'
    print('OK: strip-coverage + anti-cast fuel classifier')
