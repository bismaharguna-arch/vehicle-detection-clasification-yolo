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
                 strip_coverage_threshold=0.85,
                 focus_bottom_ratio=0.4,
                 extend_below_ratio=0.5,
                 hsv_lower=(95, 35, 60),
                 hsv_upper=(135, 255, 255),
                 strip_sat_ratio=1.1,
                 strip_s_min=75,
                 strip_v_min=95):
        """
        Args:
            strip_coverage_threshold: Minimum lebar plat yang harus tertutup garis
                                  biru di SATU baris (0-1). >= threshold -> listrik.
                                  Default 0.85 (dinaikkan dari 0.5 Jul 2026). BATAS
                                  ATAP: 11 EV asli berlabel ber-strip_score >=0.89,
                                  jadi 0.85 cuma bermargin 0.04; di 0.90 EV asli mulai
                                  hilang. JANGAN naikkan lagi. Turunkan (mis. 0.6)
                                  kalau EV miring/terpotong di lapangan terbaca bensin.
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
        # Guard "biru asli" (Jul 2026, dikalibrasi dari 84 capture push berlabel:
        # 11 EV asli vs 73 FP): MEDIAN S/V piksel biru di zona strip. EV asli
        # S 71-233 (med 148) V 71-209 (med 181); FP cast/plat-gelap med S 60.
        # PENTING: ini beda dari kalibrasi S>=70 yang pernah di-revert -- range
        # inRange TETAP longgar (S>=35) jadi coverage strip tidak runtuh; yang
        # dicek cuma "piksel tipikal strip itu biru pekat & cukup terang".
        # S>=75 V>=95 -> 10/11 EV selamat, 17/20 FP tertolak (single-frame).
        # Limitasi jujur: strip EV di malam gelap (S~71 V~71) ikut tertolak.
        self.strip_s_min = strip_s_min
        self.strip_v_min = strip_v_min

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
        bot_sel = bot > 0
        bot_s = S[focus_y:, :][bot_sel]
        if top_s.size >= 10 and bot_s.size >= 10:
            if float(np.median(bot_s)) < float(np.median(top_s)) * self.strip_sat_ratio:
                return 'bensin', strip_score

        # Guard "biru asli": strip EV = pigmen biru pekat & terang. FP dominan
        # (plat hitam ternaungi, plat putih silau ber-cast) lolos coverage tapi
        # median S/V-nya rendah -> tolak. Lihat catatan kalibrasi di __init__.
        if bot_s.size >= 10:
            V = hsv[:, :, 2]
            bot_v = V[focus_y:, :][bot_sel]
            if (float(np.median(bot_s)) < self.strip_s_min
                    or float(np.median(bot_v)) < self.strip_v_min):
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
    dark = plate.copy(); dark[34:, :] = (80, 40, 20)  # "strip" biru gelap (naungan)
    assert fc.classify(ev, (0, 0, 200, 40))[0] == 'listrik', 'strip EV harus listrik'
    assert fc.classify(uk, (0, 0, 200, 40))[0] == 'bensin', 'band kiri harus bensin'
    assert fc.classify(cast, (0, 0, 200, 40))[0] == 'bensin', 'cast biru merata harus bensin'
    assert fc.classify(dark, (0, 0, 200, 40))[0] == 'bensin', 'strip gelap harus bensin (guard V)'
    print('OK: strip-coverage + anti-cast + guard biru-asli fuel classifier')
