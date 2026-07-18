"""
detector.py
Modul deteksi kendaraan + klasifikasi bahan bakar + OCR plat.
v8: 2 model deteksi + FuelClassifier + PlateReader (PaddleOCR).
    - vehicle_model:   deteksi mobil & motor (yolov8n.pt - COCO pretrained)
    - plate_model:     deteksi plat (license_plate_detector.pt)
    - fuel_classifier: klasifikasi listrik/bensin via garis biru di plat
    - plate_reader:    baca teks plat (cache + throttle supaya gak nge-lag)
"""

import threading
from collections import deque
from queue import Queue, Full

from ultralytics import YOLO
import torch


# COCO class id -> nama lokal yang dipakai program
# Cuma car & motorcycle. Bus & truck sengaja TIDAK dideteksi.
COCO_VEHICLE_MAP = {
    2: 'mobil',   # car
    3: 'motor',   # motorcycle
}

VIRTUAL_CLASS_ID = {
    'mobil': 0,
    'motor': 1,
    'plat':  2,
}


class VehicleDetector:
    def __init__(self,
                 vehicle_model_path='models/yolov8n.pt',
                 plate_model_path='models/plate_best (1).pt',
                 unified_model_path=None,
                 conf_threshold=0.4,
                 device=None,
                 smooth_window=5,
                 lower_threshold=0.25,
                 imgsz=960,
                 vehicle_imgsz=640,
                 half=None,
                 iou_threshold=0.45,
                 plate_conf_threshold=None,
                 plate_lower_threshold=None,
                 fuel_classifier=None,
                 plate_reader=None,
                 ocr_interval=10,
                 ocr_iou_match=0.3,
                 ocr_min_height=22,
                 bbox_smooth_weight=2.0):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.conf_threshold = conf_threshold
        self.lower_threshold = lower_threshold
        self.plate_conf_threshold = (
            plate_conf_threshold if plate_conf_threshold is not None else conf_threshold
        )
        self.plate_lower_threshold = (
            plate_lower_threshold if plate_lower_threshold is not None else lower_threshold
        )
        self.smooth_window = smooth_window
        self.imgsz = imgsz
        # Model kendaraan boleh jalan di resolusi lebih rendah: mobil/motor
        # objek BESAR (yolov8n memang native 640), sementara plat kecil butuh
        # imgsz penuh. Memotong 1 dari 2 predict per frame ~2x lebih cepat
        # tanpa menyentuh akurasi plat. 0/None = samakan dengan imgsz.
        self.vehicle_imgsz = int(vehicle_imgsz) if vehicle_imgsz else imgsz
        # FP16: otomatis ON di CUDA (inference ~1.5-2x lebih cepat, penurunan
        # akurasi praktis nol), selalu OFF di CPU (tidak didukung).
        # half=False memaksa FP32 (escape hatch --no-half).
        if half is None:
            self.half = device.startswith('cuda')
        else:
            self.half = bool(half) and device.startswith('cuda')
        self.iou_threshold = iou_threshold

        self.detection_history = deque(maxlen=smooth_window)
        self.fuel_classifier = fuel_classifier
        self.plate_reader = plate_reader
        self.ocr_interval = max(1, ocr_interval)
        self.ocr_iou_match = ocr_iou_match
        # Plat lebih pendek dari ini (px) = terlalu jauh/kecil -> jangan OCR,
        # bacaannya pasti jelek. Ini yang mematikan "salah pas awal" saat plat
        # baru nongol jauh. Teks dibiarkan kosong sampai plat cukup besar.
        self.ocr_min_height = ocr_min_height
        # Bobot frame sekarang saat merata-ratakan bbox lintas history. Lebih
        # besar = lebih responsif (kurang lag), lebih kecil = lebih halus.
        # 1.0 = rata-rata biasa; naikkan kalau kotak terasa telat mengikuti gerak.
        self.bbox_smooth_weight = max(1.0, float(bbox_smooth_weight))

        # Cache OCR per "plat" (di-IoU-match antar frame).
        # Format: list of dict {'bbox', 'text', 'last_frame', '_pending'}
        self._plate_cache = []
        self._frame_idx = 0

        # OCR ASYNC: PaddleOCR jalan di worker thread sendiri supaya loop
        # deteksi tidak pernah menunggu OCR (menghilangkan FPS-spike periodik
        # tiap ocr_interval). Akurasi per-bacaan TIDAK berubah: crop &
        # preprocessing sama persis, cuma dikerjakan di thread lain; hasilnya
        # masuk ke cache begitu selesai (telat beberapa frame, settle-gate
        # tracker sudah memperhitungkan itu).
        self._ocr_queue = None
        if plate_reader is not None:
            self._ocr_queue = Queue(maxsize=4)
            threading.Thread(target=self._ocr_worker, daemon=True,
                             name='ocr-worker').start()

        # ===== Mode pemilihan model =====
        # unified_model_path != None  -> 1 model 3-kelas (mobil/motor/plat) jalan 1x predict/frame.
        # else                         -> dual-model legacy (vehicle + plate terpisah).
        self.unified = bool(unified_model_path)

        if self.unified:
            print(f'[Detector] Loading UNIFIED model: {unified_model_path}')
            self.model = YOLO(unified_model_path)
            self.model.to(device)
            self.model_class_names = self.model.names
            print(f'[Detector] Unified classes: {self.model_class_names}')
            # Konf terendah dari keduanya supaya tidak ada kelas yang ke-filter prematur;
            # threshold final per-kelas tetap diterapkan di tahap FILTER FINAL.
            self._unified_lower = min(self.lower_threshold, self.plate_lower_threshold)
            # Map class_id model -> nama lokal (mobil/motor/plat). 'plat nomor' -> 'plat'.
            self._unified_name_map = {
                cid: ('plat' if str(name).lower().startswith('plat') else str(name).lower())
                for cid, name in self.model_class_names.items()
            }
        else:
            # ===== Vehicle model =====
            print(f'[Detector] Loading VEHICLE model: {vehicle_model_path}')
            self.vehicle_model = YOLO(vehicle_model_path)
            self.vehicle_model.to(device)
            self.vehicle_class_names = self.vehicle_model.names
            print(f'[Detector] Vehicle classes (COCO, akan difilter): '
                  f'{ {k: v for k, v in self.vehicle_class_names.items() if k in COCO_VEHICLE_MAP} }')

            # ===== Plate model =====
            print(f'[Detector] Loading PLATE model:   {plate_model_path}')
            self.plate_model = YOLO(plate_model_path)
            self.plate_model.to(device)
            self.plate_class_names = self.plate_model.names
            print(f'[Detector] Plate classes: {self.plate_class_names}')

        print(f'[Detector] Device: {device.upper()} '
              f'(FP16 {"ON" if self.half else "OFF"})')
        print(f'[Detector] Inference size: plat={imgsz}px, '
              f'kendaraan={self.vehicle_imgsz}px')
        print(f'[Detector] Smoothing: {smooth_window} frames')
        print(f'[Detector] Fuel classifier: {"ON" if fuel_classifier else "OFF"}')
        print(f'[Detector] Plate OCR: {"ON" if plate_reader else "OFF"} '
              f'(interval={self.ocr_interval} frame)')

    def detect(self, frame):
        """Deteksi 1 frame, return list of detections (+ fuel_type & text untuk plat)."""
        self._frame_idx += 1
        current_detections = []

        if self.unified:
            current_detections, plate_dets = self._detect_unified(frame)
        else:
            current_detections, plate_dets = self._detect_dual(frame)

        # ----- 2b. OCR (cache + throttle) -----
        if self.plate_reader is not None and plate_dets:
            self._run_ocr(frame, plate_dets)

        current_detections.extend(plate_dets)

        # ----- 3. SMOOTHING -----
        self.detection_history.append(current_detections)
        smoothed = self._apply_smoothing(current_detections)

        # ----- 4. FILTER FINAL -----
        final = []
        for d in smoothed:
            thr = self.plate_conf_threshold if d['class_name'] == 'plat' else self.conf_threshold
            if d['confidence'] >= thr:
                final.append(d)

        return final

    def _detect_unified(self, frame):
        """1 model 3-kelas: split jadi (vehicle_dets, plate_dets)."""
        results = self.model.predict(
            frame,
            conf=self._unified_lower,
            iou=self.iou_threshold,
            imgsz=self.imgsz,   # unified deteksi plat juga -> butuh imgsz penuh
            half=self.half,
            device=self.device,
            verbose=False,
            agnostic_nms=False,
            max_det=40,
        )

        vehicle_dets = []
        plate_dets = []
        if len(results) == 0:
            return vehicle_dets, plate_dets

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            local_name = self._unified_name_map.get(cls_id)
            if local_name not in VIRTUAL_CLASS_ID:
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox = (x1, y1, x2, y2)

            if local_name == 'plat':
                fuel_type = ''
                blue_ratio = 0.0
                if self.fuel_classifier is not None:
                    fuel_type, blue_ratio = self.fuel_classifier.classify(frame, bbox)
                plate_dets.append({
                    'class_id': VIRTUAL_CLASS_ID['plat'],
                    'class_name': 'plat',
                    'confidence': confidence,
                    'bbox': bbox,
                    'fuel_type': fuel_type,
                    'blue_ratio': blue_ratio,
                    'text': '',
                })
            else:
                vehicle_dets.append({
                    'class_id': VIRTUAL_CLASS_ID[local_name],
                    'class_name': local_name,
                    'confidence': confidence,
                    'bbox': bbox,
                })

        return vehicle_dets, plate_dets

    def _detect_dual(self, frame):
        """Dual-model legacy: vehicle model + plate model terpisah."""
        vehicle_dets = []

        # ----- 1. VEHICLE -----
        veh_results = self.vehicle_model.predict(
            frame,
            conf=self.lower_threshold,
            iou=self.iou_threshold,
            imgsz=self.vehicle_imgsz,   # kendaraan = objek besar, 640 cukup
            half=self.half,
            device=self.device,
            verbose=False,
            agnostic_nms=False,
            max_det=20,
            classes=list(COCO_VEHICLE_MAP.keys()),
        )

        if len(veh_results) > 0:
            for box in veh_results[0].boxes:
                coco_id = int(box.cls[0])
                if coco_id not in COCO_VEHICLE_MAP:
                    continue
                local_name = COCO_VEHICLE_MAP[coco_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                vehicle_dets.append({
                    'class_id': VIRTUAL_CLASS_ID[local_name],
                    'class_name': local_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                })

        # ----- 2. PLATE -----
        plate_results = self.plate_model.predict(
            frame,
            conf=self.plate_lower_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,   # plat = objek kecil, JANGAN turunkan dari sini
            half=self.half,
            device=self.device,
            verbose=False,
            agnostic_nms=False,
            max_det=20,
        )

        plate_dets = []
        if len(plate_results) > 0:
            for box in plate_results[0].boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)

                fuel_type = ''
                blue_ratio = 0.0
                if self.fuel_classifier is not None:
                    fuel_type, blue_ratio = self.fuel_classifier.classify(frame, bbox)

                plate_dets.append({
                    'class_id': VIRTUAL_CLASS_ID['plat'],
                    'class_name': 'plat',
                    'confidence': confidence,
                    'bbox': bbox,
                    'fuel_type': fuel_type,
                    'blue_ratio': blue_ratio,
                    'text': '',
                })

        return vehicle_dets, plate_dets

    def _run_ocr(self, frame, plate_dets):
        """Untuk tiap plat: cari di cache via IoU; jadwalkan (re-)OCR di worker
        thread kalau interval lewat. OCR tidak dikerjakan di sini (async) --
        det['text'] diisi dari cache; hasil bacaan baru masuk ke cache begitu
        worker selesai (telat beberapa frame, tidak masalah untuk settle-gate)."""
        new_cache = []
        for det in plate_dets:
            bbox = det['bbox']
            too_small = (bbox[3] - bbox[1]) < self.ocr_min_height
            matched = None
            best_iou = 0.0
            for entry in self._plate_cache:
                iou = self._compute_iou(bbox, entry['bbox'])
                if iou > best_iou and iou >= self.ocr_iou_match:
                    best_iou = iou
                    matched = entry

            if matched is None:
                # Plat baru. last_frame mundur supaya lolos gate interval:
                # kalau sudah cukup besar langsung dijadwalkan, kalau masih
                # kekecilan otomatis ke-OCR begitu cukup besar.
                entry = {'bbox': bbox, 'text': '', '_pending': False,
                         'last_frame': self._frame_idx - self.ocr_interval}
                if not too_small:
                    self._submit_ocr(frame, bbox, entry)
            else:
                # Plat lama -> reuse; jadwalkan re-OCR kalau interval lewat,
                # cukup besar, dan tidak sedang diproses worker.
                if (not too_small and not matched['_pending']
                        and self._frame_idx - matched['last_frame'] >= self.ocr_interval):
                    self._submit_ocr(frame, bbox, matched)
                matched['bbox'] = bbox
                entry = matched

            det['text'] = entry['text']
            new_cache.append(entry)

        # Simpan cache fresh (drop plat yg gak muncul lagi)
        self._plate_cache = new_cache

    def _submit_ocr(self, frame, bbox, entry):
        """Antre 1 job OCR: salin crop plat (worker tidak boleh pegang frame
        utuh yang bisa berubah) lalu serahkan ke thread OCR. Kalau antrean
        penuh (OCR sedang sibuk), mundurkan last_frame supaya dicoba lagi
        frame berikutnya -- job tidak hilang."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return
        crop = frame[y1:y2, x1:x2].copy()
        try:
            self._ocr_queue.put_nowait((crop, entry))
        except Full:
            entry['last_frame'] = self._frame_idx - self.ocr_interval
            return
        entry['_pending'] = True
        entry['last_frame'] = self._frame_idx

    def _ocr_worker(self):
        """Loop worker OCR (daemon thread). Satu-satunya pemanggil
        plate_reader.read() -- PaddleOCR tidak pernah dipanggil paralel."""
        while True:
            crop, entry = self._ocr_queue.get()
            try:
                ch, cw = crop.shape[:2]
                text = self.plate_reader.read(crop, (0, 0, cw, ch))
                # Bacaan kosong JANGAN menimpa bacaan lama yang sudah ada
                # (parity dengan perilaku sync: re-OCR gagal -> teks bertahan).
                if text:
                    entry['text'] = text
            except Exception as e:
                print(f'[Detector] OCR worker error: {e}')
            finally:
                entry['_pending'] = False
                self._ocr_queue.task_done()

    def _apply_smoothing(self, current_detections):
        """Smoothing confidence DAN posisi bbox dengan rata-rata IoU-match dari
        frame sebelumnya -> kotak bergerak halus, tidak patah-patah. Frame
        sekarang diberi bobot lebih besar (bbox_smooth_weight) supaya tetap
        responsif. OCR/fuel sudah dihitung dari bbox mentah SEBELUM ini."""
        if len(self.detection_history) < 2:
            return current_detections

        smoothed = []
        for det in current_detections:
            confidences = [det['confidence']]

            # Akumulasi bbox berbobot: mulai dari box frame sekarang.
            w0 = self.bbox_smooth_weight
            cx1, cy1, cx2, cy2 = det['bbox']
            sx1, sy1, sx2, sy2 = cx1 * w0, cy1 * w0, cx2 * w0, cy2 * w0
            wsum = w0

            for past_frame in list(self.detection_history)[:-1]:
                best_iou = 0
                best_det = None
                for past_det in past_frame:
                    if past_det['class_id'] != det['class_id']:
                        continue
                    iou = self._compute_iou(det['bbox'], past_det['bbox'])
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_det = past_det

                if best_det is not None:
                    confidences.append(best_det['confidence'])
                    bx1, by1, bx2, by2 = best_det['bbox']
                    sx1 += bx1; sy1 += by1; sx2 += bx2; sy2 += by2
                    wsum += 1.0

            avg_conf = sum(confidences) / len(confidences)
            sbbox = (int(round(sx1 / wsum)), int(round(sy1 / wsum)),
                     int(round(sx2 / wsum)), int(round(sy2 / wsum)))

            new_det = {
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': avg_conf,
                'bbox': sbbox,
            }
            for k in ('fuel_type', 'blue_ratio', 'text'):
                if k in det:
                    new_det[k] = det[k]
            smoothed.append(new_det)

        return smoothed

    @staticmethod
    def _compute_iou(box1, box2):
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        x1_inter = max(x1_a, x1_b)
        y1_inter = max(y1_a, y1_b)
        x2_inter = min(x2_a, x2_b)
        y2_inter = min(y2_a, y2_b)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
