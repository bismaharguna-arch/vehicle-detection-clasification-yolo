"""
vehicle_tracker.py
Tracker berbasis assignment global (Hungarian) untuk dedup push ke web monitoring.

Ide:
- Tiap kendaraan di-track antar frame. Pencocokan detection->track memakai
  penugasan GLOBAL (scipy.optimize.linear_sum_assignment) atas cost gabungan
  IoU + jarak titik pusat + kemiripan ukuran/aspek bbox, dengan prediksi posisi
  (velocity 2 frame terakhir) supaya track yang bergerak tetap cocok saat sesaat
  tertutup. Greedy IoU lama gampang menukar ID saat dua kendaraan bertumpuk --
  dan ID tertukar berarti FOTO yang di-push milik kendaraan yang salah.
- 1 track = 1 kendaraan = 1 push (one-shot), plus koreksi plat susulan:
  setiap kali bacaan plat yang sudah settle BERBEDA dari yang terakhir
  terkirim ke web -- baik kosong->terisi (plat telat terbaca) maupun
  salah->benar (mis. 'D 1 S' terkoreksi jadi 'D 15 NW' saat kendaraan
  mendekat) -- koreksi masuk ANTREAN PENDING (lihat di bawah), bukan langsung
  ditandai terkirim.
- Push baru dilakukan kalau:
    * track sudah terlihat >= min_frames_stable frame (anti-flicker)
    * fuel_type sudah berhasil diasosiasikan dari plat di dalam bbox kendaraan
    * bbox kendaraan tinggi >= min_vehicle_height (quality filter)
    * teks plat TIDAK kosong (require_plate=True, default) -- web tidak pernah
      menerima baris tanpa nomor plat; track tanpa plat menunggu sampai
      platnya terbaca (atau ikut terbuang saat timeout tanpa pernah di-push)
- Foto yang dikirim = crop kendaraan dari "best frame": bbox terluas selama
  track hidup, TAPI hanya dari frame yang "bersih" -- bbox kendaraan ini tidak
  bertumpuk berat (IoU > best_frame_max_overlap) dengan kendaraan lain di frame
  itu. Frame kotor cuma dipakai kalau track tidak pernah punya frame bersih.
  Crop di-resize ke max max_image_dim & encode JPEG dengan jpeg_quality.

Antrean koreksi plat (pending updates), hidup INDEPENDEN dari track:
- Kunci antrean = track id, isinya server_id (boleh belum ada) + fields PATCH.
- Koreksi ditandai TERKIRIM hanya setelah PATCH sukses (callback on_result dari
  uploader). Gagal -> retry dengan exponential backoff sampai
  plate_update_retries kali. Dulu pushed_plate di-set sebelum PATCH jalan, jadi
  PATCH yang gagal berarti plat itu hilang selamanya.
- Entri bertahan walau track sudah dibuang (timeout), dan tetap menunggu kalau
  server_id belum datang dari uploader (set_server_id mengisi entri pending juga).
- Saat track hendak mati/timeout, bacaan plat TERBAIK yang ada (paling sering
  muncul selama track hidup, min plate_final_min_votes frame) tetap dikirim walau
  belum lolos gate plate_settle_frames -- lebih baik terkirim daripada hilang.

Kontrak payload (sesuai web monitoring v1, Mei 2026) TIDAK berubah:
    plate_number     : str (kosong selama OCR off)
    vehicle_type     : 'mobil' | 'motor' | 'unknown'
    is_electric      : 'bensin' | 'listrik' | 'unknown'
    confidence_score : float 0-1 (vehicle YOLO detection score)
    detected_at      : ISO 8601 dengan timezone (waktu first-seen, bukan push)
    photo            : file JPG (multipart)

Field internal prefix '_' (track_id, frames_seen, dll) disimpan ke sidecar
JSON saja, tidak dikirim ke server.
"""

import os
import inspect
import json
import math
import re
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy ada di requirements, ini jaring saja
    _HAS_SCIPY = False


# Template plat Indonesia sesuai output normalisasi PlateReader ('B 1234 XYZ'):
# 1-2 huruf kode wilayah + 1-4 angka + 0-3 huruf seri belakang (opsional --
# plat lama/dinas bisa tanpa seri). Bacaan OCR yang TIDAK cocok (mis. 'HY',
# potongan bacaan) diperlakukan sebagai belum terbaca: tidak pernah ikut ke
# web, menunggu bacaan sah (atau menyusul lewat koreksi plat-telat).
PLATE_TEMPLATE_RE = re.compile(r'^[A-Z]{1,2} \d{1,4}(?: [A-Z]{1,3})?$')

# Cost "tidak boleh dicocokkan" pada matriks assignment.
INF_COST = 1e6

# Kalau satu plat jatuh di dalam bbox >1 kendaraan, plat itu milik kendaraan
# terkecil HANYA kalau ia jelas bersarang (area <= rasio ini x kandidat
# berikutnya, mis. motor di depan mobil). Kalau ukurannya mirip (dua mobil
# bertumpuk), plat dianggap ambigu dan tidak diasosiasikan ke siapa pun.
PLATE_NEST_RATIO = 0.7


class VehicleTracker:
    def __init__(self,
                 min_frames_stable=5,
                 iou_match_thresh=0.3,
                 track_timeout=30,
                 fuel_required=True,
                 fuel_wait_frames=15,
                 min_vehicle_height=100,
                 max_image_dim=800,
                 jpeg_quality=85,
                 output_dir='output/captures',
                 padding=0.1,
                 plate_settle_frames=12,
                 plate_wait_frames=20,
                 plate_dedup_seconds=60,
                 require_plate=True,
                 match_max_dist=1.2,
                 match_max_cost=0.8,
                 match_predict_frames=10,
                 match_weights=(0.6, 0.25, 0.15),
                 best_frame_max_overlap=0.5,
                 plate_final_min_votes=2,
                 plate_update_retries=5,
                 plate_update_backoff=2.0,
                 plate_update_ttl=120.0,
                 on_push=None,
                 on_update=None):
        """
        Args:
            min_frames_stable:    minimum frame terlihat sebelum push.
            iou_match_thresh:     IoU minimum (atas bbox PREDIKSI track) untuk
                                  dianggap kandidat kendaraan yang sama. Kandidat
                                  ber-IoU di bawah ini masih boleh lolos lewat
                                  jalur jarak (match_max_dist) -- itu yang bikin
                                  track tetap nyambung saat sesaat tertutup.
            track_timeout:        drop track kalau tidak terlihat selama N frame.
            fuel_required:        True = wajib ada plat+fuel sebelum push.
            fuel_wait_frames:     batas frame menunggu fuel sebelum push 'unknown'.
                                  0 = jangan pernah push tanpa fuel.
            min_vehicle_height:   bbox kendaraan minimum (piksel) untuk di-push.
                                  Kalau best bbox masih < ini saat track timeout,
                                  track dibuang tanpa push (kendaraan terlalu jauh).
            max_image_dim:        crop di-resize sehingga max(w,h) <= ini.
            jpeg_quality:         0-100 untuk encode JPEG. 85 = balance ukuran/kualitas.
            output_dir:           folder simpan crop + JSON sidecar.
            padding:              padding crop kendaraan (rasio bbox, default 0.1).
            plate_settle_frames:  jangan push sampai teks plat BERTAHAN sekian frame
                                  tanpa berubah (bukti OCR sudah re-baca & setuju).
                                  Set >= ocr_interval detector supaya dijamin
                                  melewati minimal 1 siklus re-OCR. Ini yang
                                  mencegah bacaan pertama yang salah ikut terkirim.
                                  Gate ini di-BYPASS saat track hendak mati (lihat
                                  plate_final_min_votes).
            plate_wait_frames:    batas frame menunggu plat "settle" sejak first-seen.
                                  Lewat ini -> push apa adanya (plat mungkin belum
                                  terkonfirmasi) supaya kendaraan cepat-lewat tetap
                                  terkirim. 0 = jangan pernah paksa (tunggu terus).
            plate_dedup_seconds:  jendela anti-duplikat antar-track, dalam DETIK
                                  (wall clock, bukan frame -- supaya tidak
                                  bergantung FPS). Plat yang sudah di-push tidak
                                  di-push ulang sebagai baris baru dalam jendela
                                  ini; menangkap kendaraan yang deteksinya putus
                                  lama (mobil mendekat/parkir, occlusion) lalu
                                  lahir ulang sebagai track baru. <=0 = matikan.
            require_plate:        True (default) = JANGAN pernah push kendaraan
                                  yang teks platnya masih kosong.
            match_max_dist:       jarak pusat maksimum (dinormalisasi ke diagonal
                                  bbox track) supaya sepasang track/deteksi masih
                                  boleh dicocokkan walau IoU-nya 0 -- ini yang
                                  menjembatani occlusion sesaat. Default 1.2.
            match_max_cost:       cost gabungan maksimum (0-1) yang masih diterima
                                  sebagai match. Naikkan = lebih longgar. Default 0.8.
            match_predict_frames: berapa frame sebuah track boleh "diselamatkan" lewat
                                  prediksi posisi saja (IoU 0, cuma dekat). Lewat itu,
                                  match wajib bertumpuk (IoU) -- supaya track lama yang
                                  disimpan sebagai memori dedup tidak menyambar
                                  kendaraan baru di lajur yang sama. Default 10.
            match_weights:        bobot (w_iou, w_dist, w_shape) untuk cost
                                  gabungan; dinormalisasi ke total 1.
            best_frame_max_overlap: kalau bbox kendaraan ini bertumpuk dengan bbox
                                  kendaraan LAIN di frame yang sama dengan IoU >
                                  nilai ini, frame tsb TIDAK dipakai sebagai foto
                                  push (risiko foto salah kendaraan). Frame kotor
                                  hanya dipakai kalau tidak pernah ada frame bersih.
            plate_final_min_votes: saat track mati, bacaan plat terbaik (yang PALING
                                  BARU, karena OCR makin benar saat kendaraan makin
                                  dekat) dikirim walau belum settle, asal bacaan itu
                                  muncul minimal sekian frame. 1 = kirim bacaan apa pun.
            plate_update_retries: berapa kali koreksi plat (PATCH) dicoba ulang
                                  sebelum menyerah. Backoff eksponensial.
            plate_update_backoff: detik dasar backoff antar percobaan PATCH
                                  (delay = backoff * 2^(percobaan-1), cap 60s).
            plate_update_ttl:     detik maksimum sebuah koreksi menunggu server_id
                                  dari uploader sebelum dibuang.
            on_push:              callback(payload: dict, image_path: str). Optional.
            on_update:            callback(server_id, fields: dict, on_result=None).
                                  on_result(ok: bool) WAJIB dipanggil balik supaya
                                  tracker tahu PATCH sukses/gagal (kalau callback
                                  cuma menerima 2 argumen, hasilnya dianggap sukses
                                  -- mode kompatibel lama).
        """
        self.min_frames_stable = min_frames_stable
        self.iou_match_thresh = iou_match_thresh
        self.track_timeout = track_timeout
        self.fuel_required = fuel_required
        self.fuel_wait_frames = fuel_wait_frames
        self.min_vehicle_height = min_vehicle_height
        self.max_image_dim = max_image_dim
        self.jpeg_quality = int(jpeg_quality)
        self.output_dir = output_dir
        self.padding = padding
        self.plate_settle_frames = plate_settle_frames
        self.plate_wait_frames = plate_wait_frames
        self.plate_dedup_seconds = plate_dedup_seconds
        self.require_plate = require_plate

        self.match_max_dist = max(1e-3, float(match_max_dist))
        self.match_max_cost = float(match_max_cost)
        self.match_predict_frames = int(match_predict_frames)
        wsum = float(sum(match_weights)) or 1.0
        self.w_iou, self.w_dist, self.w_shape = [w / wsum for w in match_weights]
        self.best_frame_max_overlap = float(best_frame_max_overlap)

        self.plate_final_min_votes = max(1, int(plate_final_min_votes))
        self.plate_update_retries = max(1, int(plate_update_retries))
        self.plate_update_backoff = float(plate_update_backoff)
        self.plate_update_ttl = float(plate_update_ttl)

        self.on_push = on_push
        self.on_update = on_update
        # Apakah on_update sanggup melapor hasil PATCH? Kalau ya, plat baru
        # ditandai terkirim setelah callback bilang sukses.
        self._update_wants_result = False
        if on_update is not None:
            try:
                self._update_wants_result = len(
                    inspect.signature(on_update).parameters) >= 3
            except (TypeError, ValueError):
                self._update_wants_result = False

        # Memori plat yang baru di-push: text -> {'ts', 'server_id'}.
        # Dipangkas tiap frame ke jendela plate_dedup_seconds (wall clock).
        self._recent_plates = {}
        # track_id -> {'plate', 'ts'} untuk track yang sudah dibuang; dipakai
        # set_server_id (yang datang dari worker uploader, bisa telat) untuk
        # tetap menyinkronkan server_id ke memori dedup.
        self._track_plates = {}

        # Antrean koreksi plat, hidup lepas dari track. Disentuh juga oleh worker
        # thread uploader (lewat on_result / set_server_id) -> dikunci.
        self._pending = {}
        self._pending_lock = threading.RLock()
        self._pending_seq = 0

        self._tracks = []
        self._frame_idx = 0
        os.makedirs(output_dir, exist_ok=True)
        matcher = 'hungarian' if _HAS_SCIPY else 'greedy-fallback (scipy absen)'
        print(f'[Tracker] One-shot push per kendaraan. match={matcher}, '
              f'stable>={min_frames_stable}f, timeout={track_timeout}f, '
              f'min_h={min_vehicle_height}px, max_dim={max_image_dim}px, '
              f'jpeg_q={jpeg_quality}, out={output_dir}')

    # ---- geometri ----------------------------------------------------------

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    @staticmethod
    def _area(b):
        x1, y1, x2, y2 = b
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _height(b):
        return max(0, b[3] - b[1])

    @staticmethod
    def _center(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @staticmethod
    def _wh(b):
        return (max(1.0, float(b[2] - b[0])), max(1.0, float(b[3] - b[1])))

    @staticmethod
    def _center_inside(inner_bbox, container_bbox):
        cx = (inner_bbox[0] + inner_bbox[2]) / 2
        cy = (inner_bbox[1] + inner_bbox[3]) / 2
        return (container_bbox[0] <= cx <= container_bbox[2]
                and container_bbox[1] <= cy <= container_bbox[3])

    @staticmethod
    def _iso_local(ts):
        """Format epoch -> ISO 8601 dengan offset timezone lokal (sesuai
        permintaan web team: TZ-aware, bukan naive)."""
        return datetime.fromtimestamp(ts).astimezone().isoformat(timespec='seconds')

    # ---- perbandingan bacaan plat (guard koreksi & dedup varian) ------------

    @staticmethod
    def _plate_parts(text):
        """'B 1234 XYZ' -> ('B', '1234', 'XYZ'); seri boleh kosong."""
        parts = text.split()
        return (parts[0] if parts else '',
                parts[1] if len(parts) > 1 else '',
                parts[2] if len(parts) > 2 else '')

    @staticmethod
    def _digit_count(text):
        return sum(c.isdigit() for c in text)

    @classmethod
    def _plates_similar(cls, a, b):
        """Apakah dua bacaan template-valid kemungkinan besar PLAT YANG SAMA
        (varian salah-baca OCR)? Temuan uji vid1.mp4 (Jul 2026): dedup
        teks-persis kecolongan varian ('B 5805 TBZ' vs 'S 805 TBZ',
        'D 1371 ALS' vs 'D 1371 ALB') -> satu kendaraan jadi 2 baris di web.
          (1) seri sama + angka yang satu akhiran angka yang lain (digit depan
              hilang saat plat baru sebagian terbaca);
          (2) panjang sama + beda TEPAT 1 HURUF (L/F/E, B/S/Z tertukar).
        Beda 1 ANGKA sengaja TIDAK dianggap sama: 'D 1234 AB' vs 'D 1235 AB'
        bisa dua kendaraan sungguhan -- salah gabung = deteksi hilang."""
        if a == b:
            return True
        _, na, sa = cls._plate_parts(a)
        _, nb, sb = cls._plate_parts(b)
        # Angka DAN seri sama persis -> varian (kode wilayah salah baca,
        # 'DD 1446 YCS' vs 'D 1446 YCS' -- temuan run imgsz 640).
        if sa == sb and na and na == nb:
            return True
        if sa == sb and na and nb and na != nb and (
                na.endswith(nb) or nb.endswith(na)):
            return True
        if len(a) == len(b):
            diffs = [(x, y) for x, y in zip(a, b) if x != y]
            if len(diffs) == 1 and not (diffs[0][0].isdigit()
                                        or diffs[0][1].isdigit()):
                return True
        return False

    def _find_recent_plate(self, text):
        """Cari text di memori dedup: cocok persis dulu, lalu varian.
        Return (key_di_memori, entry) atau (None, None)."""
        e = self._recent_plates.get(text)
        if e is not None:
            return text, e
        for k, v in self._recent_plates.items():
            if self._plates_similar(text, k):
                return k, v
        return None, None

    def _correction_allowed(self, t, cand):
        """Boleh nggak bacaan cand menggantikan pushed_plate track ini?
        Guard hasil uji vid1.mp4 (Jul 2026):
          - anti flip-flop: teks yang PERNAH terkirim untuk baris ini tidak
            dikirim lagi ('L 1601 CF' <-> 'F 1601 CF' dulu PATCH bolak-balik);
          - anti degradasi: jumlah ANGKA tidak boleh berkurang. Plat asli tidak
            pernah kehilangan angka saat kendaraan mendekat; bacaan yang
            angkanya berkurang hampir pasti OCR memburuk karena kendaraan
            menjauh ('Z 1662 AV' -> 'Z 166 Z') -- dan bacaan buruk yang
            konsisten tetap lolos gate settle, jadi settle saja tidak cukup."""
        sent = t.get('pushed_plate', '')
        if not cand or cand == sent:
            return False
        if cand in t.get('sent_plates', ()):
            return False
        if sent and self._digit_count(cand) < self._digit_count(sent):
            return False
        return True

    def _resize_for_upload(self, crop):
        h, w = crop.shape[:2]
        m = max(h, w)
        if m <= self.max_image_dim:
            return crop
        scale = self.max_image_dim / m
        return cv2.resize(crop,
                          (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    # ---- matching: prediksi + cost gabungan + assignment global -------------

    def _predict_bbox(self, t):
        """Posisi bbox track di frame ini, diekstrapolasi dari velocity 2 frame
        terakhir. Untuk track yang barusan terlihat, age=1 (prediksi 1 frame ke
        depan); untuk track yang sedang hilang (occlusion), age = berapa frame
        sejak terakhir terlihat -- inilah yang menjaga track bergerak tetap
        cocok saat muncul lagi."""
        age = max(1, self._frame_idx - t['last_seen'])
        dx = t['vel'][0] * age
        dy = t['vel'][1] * age
        x1, y1, x2, y2 = t['bbox']
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _pair_cost(self, t, det):
        """Cost gabungan track<->deteksi, 0 (identik) .. 1 (nyaris tak mirip).
        INF_COST = terlarang. Kelas berbeda SELALU terlarang."""
        if t['class_name'] != det['class_name']:
            return INF_COST

        pb = self._predict_bbox(t)
        db = det['bbox']
        iou = self._iou(pb, db)

        tw, th = self._wh(pb)
        dw, dh = self._wh(db)
        tcx, tcy = self._center(pb)
        dcx, dcy = self._center(db)
        diag = math.hypot(tw, th)
        dist_norm = math.hypot(dcx - tcx, dcy - tcy) / max(1.0, diag)

        # Gate: cocok kalau bertumpuk cukup (IoU) ATAU pusatnya masih dekat.
        # Jalur jarak (IoU boleh 0) itu yang menyelamatkan track yang sesaat
        # tertutup -- TAPI cuma selama track baru hilang <= match_predict_frames.
        # Track pushed dipertahankan 3x track_timeout sebagai memori dedup;
        # tanpa batas ini, track basi bisa "menyedot" kendaraan BARU yang lewat
        # di posisi lajur yang sama, lalu mem-PATCH baris lama dengan plat mobil
        # lain. Setelah lewat batas, track harus benar-benar bertumpuk (IoU).
        lost_for = self._frame_idx - t['last_seen']
        if iou < self.iou_match_thresh:
            if lost_for > self.match_predict_frames or dist_norm > self.match_max_dist:
                return INF_COST

        # Veto plat untuk track "purnabakti" (sudah push, hilang > timeout,
        # cuma ditahan sebagai memori dedup): di antrean macet, mobil berikutnya
        # maju PERSIS ke posisi mobil sebelumnya, jadi IoU-nya tinggi dan track
        # lama "menyedot" kendaraan baru -- lalu PATCH baris lamanya dengan plat
        # mobil lain (temuan vid1.mp4). Kalau deteksi ini membawa bacaan plat
        # sah yang BUKAN varian plat track ini, itu kendaraan lain: tolak.
        # Kendaraan yang sama muncul lagi platnya pasti mirip -> tetap nyambung.
        if (t['pushed'] and lost_for > self.track_timeout
                and t['plate_text'] and det.get('_plate_text')
                and not self._plates_similar(t['plate_text'],
                                             det['_plate_text'])):
            return INF_COST

        size_sim = (min(tw, dw) / max(tw, dw)) * (min(th, dh) / max(th, dh))
        t_ar = tw / th
        d_ar = dw / dh
        ar_sim = min(t_ar, d_ar) / max(t_ar, d_ar)
        shape_sim = 0.5 * size_sim + 0.5 * ar_sim

        cost = (self.w_iou * (1.0 - iou)
                + self.w_dist * min(1.0, dist_norm / self.match_max_dist)
                + self.w_shape * (1.0 - shape_sim))
        return cost if cost <= self.match_max_cost else INF_COST

    def _match(self, vehicles):
        """Return dict {index deteksi -> index track}. Penugasan GLOBAL: total
        cost minimum atas semua pasangan sekaligus (Hungarian), bukan greedy
        per-deteksi -- itu yang bikin ID dua kendaraan berdekatan tidak tertukar."""
        n, m = len(self._tracks), len(vehicles)
        if n == 0 or m == 0:
            return {}

        cost = np.full((n, m), INF_COST, dtype=np.float64)
        for i, t in enumerate(self._tracks):
            for j, v in enumerate(vehicles):
                cost[i, j] = self._pair_cost(t, v)

        pairs = {}
        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < INF_COST:
                    pairs[int(j)] = int(i)
        else:
            # Fallback (scipy tak terpasang): greedy, tapi atas cost gabungan
            # yang sama dan diurut global -- masih jauh lebih baik dari greedy
            # IoU per-deteksi yang lama.
            cand = sorted((cost[i, j], i, j)
                          for i in range(n) for j in range(m)
                          if cost[i, j] < INF_COST)
            used_t, used_d = set(), set()
            for _, i, j in cand:
                if i in used_t or j in used_d:
                    continue
                used_t.add(i); used_d.add(j)
                pairs[int(j)] = int(i)
        return pairs

    # ---- update per frame ---------------------------------------------------

    def update(self, frame, detections):
        """Panggil tiap frame dengan hasil detector.detect()."""
        self._frame_idx += 1

        vehicles = [d for d in detections if d['class_name'] in ('mobil', 'motor')]
        plates = [d for d in detections if d['class_name'] == 'plat']

        self._associate_plates(vehicles, plates)

        # Seberapa berat tiap deteksi kendaraan bertumpuk dengan kendaraan LAIN
        # di frame ini -> penentu frame "bersih" untuk foto push.
        overlaps = []
        for i, v in enumerate(vehicles):
            mx = 0.0
            for k, o in enumerate(vehicles):
                if k != i:
                    mx = max(mx, self._iou(v['bbox'], o['bbox']))
            overlaps.append(mx)

        # Penugasan global (Hungarian) atas cost IoU+jarak+bentuk.
        pairs = self._match(vehicles)

        for j, v in enumerate(vehicles):
            clean = overlaps[j] <= self.best_frame_max_overlap
            i = pairs.get(j)

            if i is None:
                self._spawn_track(frame, v, clean)
                continue

            t = self._tracks[i]
            new_center = self._center(v['bbox'])
            age = max(1, self._frame_idx - t['last_seen'])
            vx = (new_center[0] - t['center'][0]) / age
            vy = (new_center[1] - t['center'][1]) / age
            # EMA supaya satu frame jitter tidak melempar prediksi.
            t['vel'] = (0.5 * t['vel'][0] + 0.5 * vx,
                        0.5 * t['vel'][1] + 0.5 * vy)
            t['center'] = new_center
            t['bbox'] = v['bbox']
            t['frames_seen'] += 1
            t['last_seen'] = self._frame_idx
            t['conf'] = v['confidence']

            # Fuel: bukti POSITIF (strip biru terlihat -> listrik) menimpa
            # bukti lemah (strip tak terlihat -> bensin). Kendaraan jauh
            # hampir selalu kebaca 'bensin' duluan karena stripnya belum
            # kelihatan; jangan kunci bacaan pertama. Butuh >=2 frame
            # listrik supaya kilatan biru sekali-lewat tidak salah flip.
            if v['_fuel']:
                if v['_fuel'] == 'listrik':
                    t['listrik_seen'] += 1
                if not t['fuel']:
                    t['fuel'] = v['_fuel']
                elif t['fuel'] == 'bensin' and t['listrik_seen'] >= 2:
                    t['fuel'] = 'listrik'

            # Plate text: pakai bacaan non-empty terbaru, TAPI lacak berapa
            # lama teks bertahan tak berubah. Bacaan pertama sering salah dan
            # baru dikoreksi saat re-OCR (tiap ocr_interval frame); kalau teks
            # berubah -> reset counter. plate_settle dipakai sebagai gate push.
            # plate_votes = rekap semua bacaan sah selama track hidup; dipakai
            # sebagai "bacaan terbaik" saat track keburu mati sebelum settle.
            if v['_plate_text']:
                txt = v['_plate_text']
                t['plate_votes'][txt] = t['plate_votes'].get(txt, 0) + 1
                t['plate_last'][txt] = self._frame_idx
                if txt == t['plate_text']:
                    t['plate_settle'] += 1
                else:
                    t['plate_text'] = txt
                    t['plate_settle'] = 1

            self._maybe_best_frame(t, frame, v['bbox'], clean)

        # Putuskan track yang siap di-push
        for t in self._tracks:
            if t['pushed']:
                continue
            if t['frames_seen'] < self.min_frames_stable:
                continue
            # Quality filter: bbox terlalu kecil = kendaraan jauh, foto tidak
            # berguna untuk human review. Jangan mark pushed -- nunggu kalau
            # kendaraannya makin dekat (best_bbox akan grow).
            if self._height(t['best_bbox']) < self.min_vehicle_height:
                continue

            # Gate plat: tunggu bacaan terkonfirmasi (bertahan >= plate_settle_frames
            # frame, artinya sudah lolos minimal 1 re-OCR) supaya bacaan pertama
            # yang salah tidak ikut terkirim. Timeout supaya kendaraan cepat-lewat
            # tetap terkirim walau plat belum sempat dikonfirmasi.
            plate_confirmed = t['plate_text'] and t['plate_settle'] >= self.plate_settle_frames
            plate_timed_out = (self.plate_wait_frames > 0
                               and (self._frame_idx - t['first_seen']) >= self.plate_wait_frames)
            if not plate_confirmed and not plate_timed_out:
                continue

            # Wajib ada nomor plat: web tidak boleh menerima baris tanpa plat.
            if self.require_plate and not t['plate_text']:
                continue

            fuel = self._resolve_fuel(t)
            if fuel is None:
                continue

            if self._dedup_adopt(t, fuel):
                continue

            self._do_push(t, fuel)

        # Koreksi plat susulan (track masih hidup): bacaan yang sudah settle
        # BERBEDA dari yang terakhir SUKSES terkirim ke web -> masuk antrean
        # pending. pushed_plate baru berubah setelah PATCH-nya benar-benar
        # sukses, jadi PATCH yang gagal akan dicoba lagi (backoff), bukan
        # hilang diam-diam.
        if self.on_update is not None:
            for t in self._tracks:
                if not t['pushed'] or not t['plate_text']:
                    continue
                if t['plate_text'] == t['pushed_plate']:
                    continue  # web sudah punya teks ini
                # Koreksi "selevel" (jumlah angka sama, cuma huruf yang beda)
                # rawan flip-flop varian OCR -> wajib settle 2x lebih lama.
                # Koreksi yang MENAMBAH angka (plat makin lengkap) cukup
                # settle normal -- itu justru koreksi yang paling diharapkan.
                need = self.plate_settle_frames
                if (t['pushed_plate'] and self._digit_count(t['plate_text'])
                        == self._digit_count(t['pushed_plate'])):
                    need *= 2
                if t['plate_settle'] < need:
                    continue  # tunggu bacaan terkonfirmasi dulu
                self._queue_plate_update(t, t['plate_text'])

        # Cleanup track lewat timeout. Track yang SUDAH pushed dipertahankan
        # 3x lebih lama: dia ringan (best_frame sudah dilepas) dan berperan
        # sebagai memori dedup -- kendaraan yang deteksinya sempat putus
        # (skor rendah/occlusion) match balik ke track lamanya (plat susulan
        # tetap PATCH ke baris lama), bukan lahir sebagai track baru yang
        # bikin baris duplikat di web.
        alive = []
        for t in self._tracks:
            limit = self.track_timeout * 3 if t['pushed'] else self.track_timeout
            if self._frame_idx - t['last_seen'] <= limit:
                alive.append(t)
            else:
                self._on_track_expire(t)
        self._tracks = alive

        # Kirim / retry koreksi plat yang tertunda (termasuk milik track yang
        # sudah dibuang).
        self._pump_pending()
        self._prune_memory()

    def _associate_plates(self, vehicles, plates):
        """Set v['_fuel'] & v['_plate_text'] dari plat yang centroid-nya ada di
        dalam bbox kendaraan -- TAPI satu plat hanya boleh dimiliki satu kendaraan.

        Saat dua kendaraan bertumpuk, centroid plat kendaraan A juga jatuh di dalam
        bbox kendaraan B; versi lama ambil plat pertama yang cocok, jadi B bisa
        mewarisi plat (dan fuel) milik A -- sumber baris/koreksi salah-kendaraan
        yang sama menyakitkannya dengan ID switch. Aturan sekarang:
          - 1 kandidat kendaraan  -> dimiliki dia.
          - >1 kandidat, yang terkecil jelas BERSARANG di yang lain (mis. motor
            di depan mobil, area <= PLATE_NEST_RATIO x kandidat berikutnya) ->
            milik yang terkecil.
          - >1 kandidat berukuran mirip (dua mobil bertumpuk) -> AMBIGU, plat ini
            tidak diasosiasikan ke siapa pun di frame ini. Menebak = salah data;
            menunggu = cuma telat beberapa frame."""
        for v in vehicles:
            v['_fuel'] = ''
            v['_plate_text'] = ''
        if not plates:
            return

        owned = {}
        for p in plates:
            px1, py1, px2, py2 = p['bbox']
            cands = []
            for v in vehicles:
                if not self._center_inside(p['bbox'], v['bbox']):
                    continue
                # Zona vertikal: plat selalu di bagian bawah kendaraan. Plat
                # yang jatuh di 35% TERATAS bbox kendaraan hampir pasti milik
                # kendaraan lain di belakangnya yang bboxnya lebih tinggi
                # (sumber kontaminasi di antrean macet, temuan vid1.mp4).
                vy1, vy2 = v['bbox'][1], v['bbox'][3]
                pcy = (py1 + py2) / 2
                if pcy < vy1 + 0.35 * (vy2 - vy1):
                    continue
                cands.append(v)
            if not cands:
                continue
            # Kandidat yang memuat SELURUH kotak plat lebih meyakinkan daripada
            # yang cuma memuat titik pusatnya (plat tetangga biasanya cuma
            # nyerempet masuk sebagian).
            full = [v for v in cands
                    if (v['bbox'][0] <= px1 and v['bbox'][1] <= py1
                        and v['bbox'][2] >= px2 and v['bbox'][3] >= py2)]
            if full:
                cands = full
            if len(cands) > 1:
                cands.sort(key=lambda v: self._area(v['bbox']))
                a0 = self._area(cands[0]['bbox'])
                a1 = self._area(cands[1]['bbox'])
                if a1 <= 0 or a0 > PLATE_NEST_RATIO * a1:
                    continue  # ukuran mirip -> ambigu, jangan tebak
            owned.setdefault(id(cands[0]), []).append(p)

        for v in vehicles:
            ps = owned.get(id(v))
            if not ps:
                continue
            # Kalau satu kendaraan kebagian >1 plat, pakai yang bbox-nya terbesar
            # (paling dekat/paling jelas).
            p = max(ps, key=lambda p: self._area(p['bbox']))
            v['_fuel'] = p.get('fuel_type', '')
            # Saring di pintu masuk: teks di luar template plat tidak pernah
            # masuk track, jadi tidak bisa ter-push / ter-PATCH.
            raw = p.get('text', '')
            v['_plate_text'] = raw if PLATE_TEMPLATE_RE.match(raw) else ''

    def _spawn_track(self, frame, v, clean):
        txt = v['_plate_text']
        self._tracks.append({
            'id': uuid.uuid4().hex[:8],
            'class_name': v['class_name'],
            'bbox': v['bbox'],
            'center': self._center(v['bbox']),
            'vel': (0.0, 0.0),
            'best_bbox': v['bbox'],
            'best_area': self._area(v['bbox']),
            'best_frame': self._crop_padded(frame, v['bbox']),
            'best_clean': clean,
            'frames_seen': 1,
            'first_seen': self._frame_idx,
            'last_seen': self._frame_idx,
            'fuel': v['_fuel'],
            'listrik_seen': 1 if v['_fuel'] == 'listrik' else 0,
            'plate_text': txt,
            'plate_settle': 1 if txt else 0,
            'plate_votes': {txt: 1} if txt else {},
            'plate_last': {txt: self._frame_idx} if txt else {},
            'conf': v['confidence'],
            'pushed': False,
            'pushed_plate': '',   # plat yang SUKSES terkirim ke web
            'sent_plates': set(),  # semua teks yang pernah terkirim (anti flip-flop)
            'pushed_fuel': '',
            'server_id': None,    # id record web (diisi set_server_id)
            'created_ts': time.time(),
        })

    def _crop_padded(self, frame, bbox):
        """Crop kendaraan + padding dari frame -- dipanggil saat CAPTURE, bukan
        saat push. Menyimpan crop kecil (puluhan-ratusan KB) menggantikan
        frame.copy() utuh (~6 MB @1080p) yang dulu bikin spike beberapa ms
        tiap best-frame berganti; di kerumunan padat spike itu nyata di FPS."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pw = int((x2 - x1) * self.padding)
        ph = int((y2 - y1) * self.padding)
        cx1 = max(0, x1 - pw); cy1 = max(0, y1 - ph)
        cx2 = min(w, x2 + pw); cy2 = min(h, y2 + ph)
        return frame[cy1:cy2, cx1:cx2].copy()

    def _maybe_best_frame(self, t, frame, bbox, clean):
        """Pilih foto push. Frame BERSIH (bbox tidak bertumpuk berat dengan
        kendaraan lain) selalu menang atas frame kotor -- foto dari frame kotor
        gampang jadi crop kendaraan yang salah saat ID sempat rancu. Di antara
        sesama frame bersih, bbox terluas (paling dekat) yang menang. Frame
        kotor hanya dipakai kalau track tidak pernah punya frame bersih."""
        if t['pushed']:
            return
        area = self._area(bbox)
        if clean:
            if not t['best_clean'] or area > t['best_area']:
                t['best_area'] = area
                t['best_bbox'] = bbox
                t['best_frame'] = self._crop_padded(frame, bbox)
                t['best_clean'] = True
        elif not t['best_clean'] and area > t['best_area']:
            t['best_area'] = area
            t['best_bbox'] = bbox
            t['best_frame'] = self._crop_padded(frame, bbox)

    def _resolve_fuel(self, t):
        """Return fuel string yang boleh dipush, atau None kalau harus menunggu."""
        fuel = t['fuel']
        if fuel:
            return fuel
        if not self.fuel_required:
            return 'unknown'
        if self.fuel_wait_frames > 0:
            age = self._frame_idx - t['first_seen']
            return 'unknown' if age >= self.fuel_wait_frames else None
        return None

    def _dedup_adopt(self, t, fuel):
        """Anti-duplikat antar-track: plat ini baru saja di-push (kendaraan yang
        sama, deteksinya sempat putus lalu lahir ulang sebagai track baru) ->
        jangan bikin baris baru di web. Adopsi server_id record lama supaya
        koreksi berikutnya tetap ke baris itu. Pencocokan TOLERAN VARIAN
        (_plates_similar): 'S 805 TBZ' mengadopsi baris 'B 5805 TBZ' -- dan
        karena pushed_plate di-set ke teks BARIS (bukan bacaan sendiri), bacaan
        kita yang lebih lengkap otomatis jadi kandidat koreksi ke baris itu.
        Return True kalau diadopsi."""
        if not t['plate_text'] or self.plate_dedup_seconds <= 0:
            return False
        key, dup = self._find_recent_plate(t['plate_text'])
        if dup is None:
            return False
        t['pushed'] = True
        t['server_id'] = dup['server_id']
        t['pushed_plate'] = key           # teks yang benar-benar ada di baris web
        t['sent_plates'] = t.get('sent_plates') or set()
        t['sent_plates'].add(key)
        t['pushed_fuel'] = fuel
        t['best_frame'] = None
        # Petakan juga bacaan varian kita ke baris yang sama, supaya track
        # berikutnya dengan bacaan ini langsung cocok persis.
        self._recent_plates[t['plate_text']] = {
            'ts': time.time(), 'server_id': dup['server_id']}
        self._track_plates[t['id']] = {'plate': key, 'ts': time.time()}
        print(f'[Tracker] DEDUP #{t["id"]} plat "{t["plate_text"]}" '
              f'~ "{key}" baru saja di-push (server_id={dup["server_id"]}); '
              f'tidak bikin baris baru.')
        return True

    # ---- antrean koreksi plat (pending updates) -----------------------------

    def _best_plate(self, t):
        """Bacaan plat TERBAIK selama track hidup, dipakai saat track hendak mati
        (gate settle terlalu ketat untuk plat yang cuma sempat terbaca beberapa
        frame saat kendaraan paling dekat).

        Pilihannya: bacaan PALING BARU yang muncul >= plate_final_min_votes frame,
        bukan yang paling sering muncul. Arah koreksi OCR selalu sama -- kendaraan
        makin dekat, bacaan makin benar ('B 1 C' -> 'B 15 CD') -- jadi bacaan lama
        yang salah justru hampir selalu unggul jumlah frame-nya. Ambang votes-lah
        yang menyaring kilatan OCR sekali-lewat, bukan mayoritas."""
        votes = t.get('plate_votes') or {}
        cand = [txt for txt, n in votes.items() if n >= self.plate_final_min_votes]
        if not cand:
            return ''
        return max(cand, key=lambda txt: (t['plate_last'].get(txt, 0), votes[txt]))

    def _queue_plate_update(self, t, plate):
        """Antre koreksi plat untuk track ini. Entri hidup lepas dari track:
        tetap dikirim walau track keburu dibuang, dan tetap menunggu kalau
        server_id belum datang dari uploader."""
        if self.on_update is None or not plate:
            return
        if not self._correction_allowed(t, plate):
            return
        # Anti-kontaminasi (temuan vid1.mp4): kalau teks ini sudah tercatat
        # milik BARIS LAIN, hampir pasti track ini mewarisi plat kendaraan
        # tetangganya (asosiasi bocor di kerumunan rapat). Satu plat tidak
        # mungkin milik dua baris -- jangan PATCH.
        owner = self._recent_plates.get(plate)
        if owner is not None and owner['server_id'] != t['server_id']:
            # Log sekali per (track, teks) -- kandidat yang diblokir terus
            # settle tiap frame, tanpa ini log banjir baris yang sama.
            seen = t.setdefault('_skip_logged', set())
            if plate not in seen:
                seen.add(plate)
                print(f'[Tracker] SKIP UPDATE #{t["id"]} plat "{plate}": sudah '
                      f'milik baris lain (server_id={owner["server_id"]}).')
            return
        fields = {'plate_number': plate}
        if t.get('pushed_fuel') == 'unknown' and t['fuel']:
            fields['is_electric'] = t['fuel']

        with self._pending_lock:
            e = self._pending.get(t['id'])
            if e is not None and e['plate'] == plate:
                return  # sudah antre / sudah menyerah untuk teks yang sama
            # Dua track bisa memegang server_id yang sama (adopsi dedup).
            # Jangan biarkan mereka balapan PATCH ke baris yang sama --
            # entri lama untuk server_id itu digantikan (yang terbaru menang).
            if t['server_id'] is not None:
                for k, o in list(self._pending.items()):
                    if (k != t['id'] and o['server_id'] == t['server_id']
                            and not o['inflight']):
                        del self._pending[k]
            self._pending_seq += 1
            self._pending[t['id']] = {
                'track_id': t['id'],
                'server_id': t['server_id'],
                'plate': plate,
                'fields': fields,
                'attempts': 0,
                'next_ts': 0.0,
                'inflight': False,
                'dead': False,
                'seq': self._pending_seq,
                'created_ts': time.time(),
            }
        old = t.get('pushed_plate') or '(kosong)'
        print(f'[Tracker] QUEUE UPDATE #{t["id"]} plat {old} -> "{plate}" '
              f'(server_id={t["server_id"] if t["server_id"] is not None else "menunggu"})')

    def _pump_pending(self):
        """Kirim koreksi yang siap (punya server_id, sudah lewat backoff).
        Dipanggil tiap frame + saat flush."""
        if self.on_update is None:
            return
        now = time.time()
        ready = []
        with self._pending_lock:
            for e in self._pending.values():
                if e['dead'] or e['inflight'] or e['server_id'] is None:
                    continue
                if now < e['next_ts']:
                    continue
                e['inflight'] = True
                e['attempts'] += 1
                ready.append((e['seq'], e['track_id'], e['server_id'],
                              dict(e['fields']), e['attempts']))

        for seq, tid, sid, fields, attempt in ready:
            print(f'[Tracker] UPDATE #{tid} plat "{fields["plate_number"]}" '
                  f'(PATCH server_id={sid}, percobaan {attempt}/{self.plate_update_retries})')

            def _result(ok, _tid=tid, _seq=seq):
                self._on_update_result(_tid, _seq, ok)

            try:
                if self._update_wants_result:
                    self.on_update(sid, fields, _result)
                else:
                    # Callback lama (2 argumen) tidak bisa melapor -> anggap sukses.
                    self.on_update(sid, fields)
                    _result(True)
            except Exception as e:
                print(f'[Tracker] on_update error: {e}')
                _result(False)

    def _on_update_result(self, track_id, seq, ok):
        """Hasil PATCH dari uploader. Bisa dipanggil dari worker thread uploader.
        Sukses -> tandai plat terkirim (baru di sini!). Gagal -> backoff & ulangi."""
        with self._pending_lock:
            e = self._pending.get(track_id)
            if e is None or e['seq'] != seq:
                return  # sudah digantikan bacaan yang lebih baru / dibuang
            if not ok:
                e['inflight'] = False
                if e['attempts'] >= self.plate_update_retries:
                    e['dead'] = True
                    e['ts_dead'] = time.time()
                    print(f'[Tracker] UPDATE GAGAL PERMANEN #{track_id} '
                          f'plat "{e["plate"]}" setelah {e["attempts"]} percobaan.')
                    return
                delay = min(60.0, self.plate_update_backoff * (2 ** (e['attempts'] - 1)))
                e['next_ts'] = time.time() + delay
                print(f'[Tracker] UPDATE gagal #{track_id} plat "{e["plate"]}"; '
                      f'coba lagi dalam {delay:.0f}s.')
                return
            plate = e['plate']
            fields = e['fields']
            server_id = e['server_id']
            del self._pending[track_id]

        # Sukses: baru SEKARANG plat dianggap sudah ada di web. Perbarui SEMUA
        # track yang memegang server_id ini (adopsi dedup bisa bikin >1 track
        # menunjuk baris yang sama) -- termasuk riwayat sent_plates mereka,
        # supaya tidak ada yang mengirim ulang teks yang sama (anti flip-flop).
        self._recent_plates[plate] = {'ts': time.time(), 'server_id': server_id}
        self._track_plates[track_id] = {'plate': plate, 'ts': time.time()}
        for t in self._tracks:
            if t['id'] == track_id or t['server_id'] == server_id:
                t['pushed_plate'] = plate
                t.setdefault('sent_plates', set()).add(plate)
                if 'is_electric' in fields:
                    t['pushed_fuel'] = fields['is_electric']
        print(f'[Tracker] UPDATE OK #{track_id} plat "{plate}" (server_id={server_id})')

    def _final_correction_ok(self, t, best):
        """Layak nggak bacaan best jadi koreksi terakhir saat track mati?
        Gate settle di-bypass di jalur ini, jadi koreksi SELEVEL (jumlah angka
        sama, cuma huruf beda) butuh bukti lebih: bacaan barunya harus lebih
        sering terlihat daripada yang sudah terkirim. Koreksi yang MENAMBAH
        angka (plat makin lengkap) selalu layak -- itu kasus utamanya."""
        if not best or best == t['pushed_plate']:
            return False
        sent = t['pushed_plate']
        if sent and self._digit_count(best) == self._digit_count(sent):
            votes = t.get('plate_votes', {})
            if votes.get(best, 0) <= votes.get(sent, 0):
                return False
        return True

    def _on_track_expire(self, t):
        """Track mau dibuang (timeout). Jangan biarkan bacaan plat yang belum
        settle ikut hilang."""
        best = self._best_plate(t)
        if t['pushed']:
            if self._final_correction_ok(t, best):
                self._queue_plate_update(t, best)
            return
        # Belum pernah di-push: kirim sekarang dengan bacaan terbaik yang ada
        # (gate settle di-bypass; gate kualitas TETAP berlaku).
        if not best and self.require_plate:
            return
        if t['frames_seen'] < self.min_frames_stable:
            return
        if self._height(t['best_bbox']) < self.min_vehicle_height:
            return
        fuel = t['fuel']
        if not fuel:
            if self.fuel_required and self.fuel_wait_frames == 0:
                return
            fuel = 'unknown'
        if best:
            t['plate_text'] = best
        if self._dedup_adopt(t, fuel):
            return
        self._do_push(t, fuel)

    def _prune_memory(self):
        """Pangkas memori plat & entri pending yang sudah basi."""
        now = time.time()
        if self._recent_plates and self.plate_dedup_seconds > 0:
            self._recent_plates = {
                k: v for k, v in self._recent_plates.items()
                if now - v['ts'] <= self.plate_dedup_seconds
            }
        if self._track_plates:
            keep = max(300.0, self.plate_dedup_seconds)
            self._track_plates = {
                k: v for k, v in self._track_plates.items()
                if now - v['ts'] <= keep
            }
        with self._pending_lock:
            for tid, e in list(self._pending.items()):
                if e['dead']:
                    if now - e.get('ts_dead', e['created_ts']) > 300.0:
                        del self._pending[tid]
                elif (e['server_id'] is None and not e['inflight']
                      and now - e['created_ts'] > self.plate_update_ttl):
                    print(f'[Tracker] UPDATE dibuang #{tid} plat "{e["plate"]}": '
                          f'server_id tidak pernah datang ({self.plate_update_ttl:.0f}s).')
                    del self._pending[tid]

    def set_server_id(self, track_id, server_id):
        """Terima id record server dari uploader (respons 201 POST).
        Dipanggil dari worker thread uploader. Selain track yang masih hidup,
        entri koreksi yang TERTUNDA juga diisi -- koreksi yang settle sebelum
        server_id datang (bahkan setelah track-nya mati) tetap terkirim."""
        plate = ''
        for t in self._tracks:
            if t['id'] == track_id:
                t['server_id'] = server_id
                plate = t.get('pushed_plate', '')
                break
        else:
            meta = self._track_plates.get(track_id)
            plate = meta['plate'] if meta else ''

        with self._pending_lock:
            e = self._pending.get(track_id)
            if e is not None and e['server_id'] is None:
                e['server_id'] = server_id

        # Sinkronkan memori dedup, supaya track lahir-ulang yang mengadopsi
        # plat ini dapat server_id yang benar.
        if plate and plate in self._recent_plates:
            self._recent_plates[plate]['server_id'] = server_id

    # ---- push ---------------------------------------------------------------

    def _do_push(self, t, fuel):
        # best_frame sudah berupa crop kendaraan ber-padding (di-crop saat
        # capture oleh _crop_padded) -- tinggal resize + encode.
        crop = self._resize_for_upload(t['best_frame'])

        ts_push = time.time()
        ts_detected = t['created_ts']  # waktu first-seen track = waktu deteksi
        ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts_detected))
        fname = f'{ts_str}_{t["class_name"]}_{fuel}_{t["id"]}.jpg'
        img_path = os.path.join(self.output_dir, fname)
        cv2.imwrite(img_path, crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        # Skema web monitoring v1 (Indonesia, drop English mapping).
        # Field tambahan (camera_id/client_event_id) sengaja TIDAK dikirim
        # karena server belum punya kolomnya (akan diabaikan diam-diam).
        payload = {
            'plate_number': t.get('plate_text', ''),  # hasil PaddleOCR, '' kalau gagal
            'vehicle_type': t['class_name'],         # 'mobil' / 'motor'
            'is_electric': fuel,                     # 'bensin' / 'listrik' / 'unknown'
            'confidence_score': round(float(t['conf']), 3),
            'detected_at': self._iso_local(ts_detected),
            # --- internal / sidecar only (uploader filter prefix '_') ---
            '_track_id': t['id'],
            '_pushed_at': self._iso_local(ts_push),
            '_frames_seen': t['frames_seen'],
            '_image_file': fname,
            '_image_size_bytes': os.path.getsize(img_path) if os.path.exists(img_path) else 0,
        }
        # Rekam apa yang terkirim saat push -- pembanding untuk koreksi plat
        # susulan, plus memori dedup antar-track (server_id menyusul lewat
        # set_server_id begitu respons 201 datang).
        t['pushed'] = True
        t['pushed_plate'] = payload['plate_number']
        t['sent_plates'] = t.get('sent_plates') or set()
        t['sent_plates'].add(payload['plate_number'])
        t['pushed_fuel'] = fuel
        t['best_frame'] = None  # bebaskan memori
        t['best_clean'] = True
        if payload['plate_number']:
            self._recent_plates[payload['plate_number']] = {
                'ts': time.time(), 'server_id': t['server_id']}
            self._track_plates[t['id']] = {
                'plate': payload['plate_number'], 'ts': time.time()}

        sidecar = os.path.join(self.output_dir, fname.replace('.jpg', '.json'))
        with open(sidecar, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        plate_str = payload['plate_number'] or '(no plate)'
        print(f'[Tracker] PUSH #{t["id"]} {t["class_name"]}/{fuel} '
              f'plate="{plate_str}" '
              f'(seen {t["frames_seen"]}f, {payload["_image_size_bytes"]/1024:.0f}KB) '
              f'-> {fname}')

        if self.on_push is not None:
            try:
                self.on_push(payload, img_path)
            except Exception as e:
                print(f'[Tracker] on_push error: {e}')

    def flush(self, update_wait=5.0):
        """Panggil saat shutdown. (1) Paksa push semua track yang belum di-push
        -- tetap hormati min_vehicle_height, gate fuel, dan require_plate, tapi
        gate settle plat di-bypass (pakai bacaan terbaik). (2) Antre koreksi plat
        terakhir untuk track yang sudah di-push. (3) Tunggu sampai update_wait
        detik supaya koreksi yang masih menunggu server_id/retry sempat terkirim
        sebelum uploader ditutup."""
        for t in list(self._tracks):
            if t['pushed']:
                best = self._best_plate(t)
                if self._final_correction_ok(t, best):
                    self._queue_plate_update(t, best)
                continue
            best = self._best_plate(t) or t['plate_text']
            if t['frames_seen'] < self.min_frames_stable:
                continue
            if self._height(t['best_bbox']) < self.min_vehicle_height:
                continue
            if self.require_plate and not best:
                continue  # wajib ada nomor plat: jangan kirim baris kosong
            if (not t['fuel'] and self.fuel_required
                    and self.fuel_wait_frames == 0):
                continue
            if best:
                t['plate_text'] = best
            fuel = t['fuel'] or 'unknown'
            if self._dedup_adopt(t, fuel):
                continue
            self._do_push(t, fuel)

        # Drain antrean koreksi: server_id untuk push barusan masih dalam
        # perjalanan (uploader worker), dan retry yang ter-backoff butuh waktu.
        deadline = time.time() + max(0.0, update_wait)
        while time.time() < deadline:
            self._pump_pending()
            with self._pending_lock:
                sisa = [e for e in self._pending.values() if not e['dead']]
            if not sisa:
                break
            time.sleep(0.1)

        with self._pending_lock:
            sisa = [e for e in self._pending.values() if not e['dead']]
        if sisa:
            print(f'[Tracker] {len(sisa)} koreksi plat belum sempat terkirim '
                  f'saat shutdown: {[e["plate"] for e in sisa]}')
