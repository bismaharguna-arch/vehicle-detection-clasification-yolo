"""
pipeline.py
Inti pemrosesan 1 frame, diekstrak dari loop `while True` di main.py supaya bisa
dipakai ulang oleh CLI (main.py) maupun GUI (gui.py) tanpa menduplikasi logika.

DetectionPipeline merakit komponen yang sama persis seperti main.py dulu
merakitnya (fuel -> plate_reader -> detector -> [uploader] -> tracker -> logger)
dan menyediakan:

    process_frame(frame, frame_idx, log_frame=True) -> (annotated, detections)
        detect -> tracker.update (yang otomatis antre push via on_push) ->
        (opsional) log JSONL -> gambar bbox + count. FPS TIDAK digambar di sini
        (caller yang urus, supaya timing FPS-nya identik dengan versi CLI lama).

    push_enabled : bool  -- gate kirim ke server (runtime). OFF = deteksi tetap
                            jalan & capture lokal tetap ditulis tracker, cuma
                            tidak di-POST ke web.
    push_url     : str   -- ganti endpoint uploader saat jalan.

    release()            -- flush tracker + tutup uploader + tutup logger
                            (urutan sama dengan blok finally main.py lama).

Catatan desain (beda dari sketsa Prompt 1, demi menjaga perilaku CLI identik):
- Pipeline TIDAK memiliki kamera. Stepping/loop/pause di CLI itu operasi level
  cap (POS_FRAMES dll); kalau cap dipindah ke sini, semua itu harus dibongkar
  dan berisiko mengubah perilaku. Jadi process_frame menerima 1 frame; caller
  yang buka/tutup cap.
- FPS digambar oleh caller, bukan di process_frame (lihat di atas).

Logika detector/fuel/plate/tracker/uploader TIDAK diubah -- pipeline cuma
merakit & mengorkestrasi mereka.
"""

import os

from detector import VehicleDetector
from fuel_classifier import FuelClassifier
from plate_reader import PlateReader
from vehicle_tracker import VehicleTracker
from uploader import HttpUploader
from logger import DetectionLogger
from utils import draw_detections, draw_detection_count


class DetectionPipeline:
    def __init__(self, args, source_label=''):
        """
        Args:
            args: namespace CLI (argparse.Namespace atau objek dengan atribut yang
                  sama). Field yang dipakai identik dengan main.py.
            source_label: string sumber untuk baris logger (mis. 'webcam:0').
        """
        self.args = args

        # ---- Komponen (urutan & argumen sama persis dengan main.py lama) ----
        self.fuel = FuelClassifier()

        self.plate_reader = None
        if not args.no_ocr:
            try:
                self.plate_reader = PlateReader(
                    gpu=not args.ocr_cpu,
                    crop_bottom_ratio=getattr(args, 'ocr_crop_bottom', 0.0),
                    upscale_width=getattr(args, 'ocr_upscale', 0),
                    min_height_ratio=getattr(args, 'ocr_min_height_ratio', 0.6),
                    strict_region=getattr(args, 'ocr_strict_region', True),
                )
            except Exception as e:
                print(f'[WARN] PaddleOCR gagal di-load ({e}). Lanjut tanpa OCR.')
                self.plate_reader = None

        self.detector = VehicleDetector(
            vehicle_model_path=args.vehicle_model,
            plate_model_path=args.plate_model,
            unified_model_path=(args.model or None),
            conf_threshold=args.conf,
            plate_conf_threshold=args.plate_conf,
            device=args.device,
            imgsz=args.imgsz,
            vehicle_imgsz=getattr(args, 'vehicle_imgsz', 640),
            half=(False if getattr(args, 'no_half', False) else None),
            fuel_classifier=self.fuel,
            plate_reader=self.plate_reader,
            ocr_interval=args.ocr_interval,
            ocr_min_height=getattr(args, 'ocr_min_height', 22),
        )

        # ---- Uploader: dibuat kalau --push (parity CLI). GUI bisa bikin
        # belakangan lewat connect_web(). push_enabled = gate kirim runtime. ----
        self._uploader = None
        self.push_enabled = bool(getattr(args, 'push', False))
        if self.push_enabled:
            self._uploader = HttpUploader(
                url=args.push_url,
                mode=args.push_mode,
                timeout=args.push_timeout,
                retries=args.push_retries,
                api_key=getattr(args, 'api_key', ''),
                on_registered=self._on_registered,
            )

        captures_dir = os.path.join(args.output_dir, 'captures')
        self.tracker = VehicleTracker(
            min_frames_stable=args.push_stable,
            track_timeout=args.push_timeout_frames,
            fuel_required=not args.push_fuel_optional,
            fuel_wait_frames=args.push_fuel_wait,
            min_vehicle_height=args.push_min_height,
            max_image_dim=args.push_max_dim,
            jpeg_quality=args.push_jpeg_quality,
            output_dir=captures_dir,
            # Konfirmasi plat harus melewati minimal 1 siklus re-OCR -> settle
            # >= ocr_interval + margin. Cegah bacaan pertama yang salah terkirim.
            plate_settle_frames=args.ocr_interval + 2,
            plate_wait_frames=getattr(args, 'push_plate_wait', 20),
            plate_dedup_seconds=getattr(args, 'push_dedup_seconds', 60.0),
            require_plate=getattr(args, 'push_require_plate', True),
            # Matching global (Hungarian) + prediksi posisi: anti ID-switch.
            match_max_dist=getattr(args, 'track_max_dist', 1.2),
            match_max_cost=getattr(args, 'track_max_cost', 0.8),
            match_predict_frames=getattr(args, 'track_predict_frames', 10),
            best_frame_max_overlap=getattr(args, 'track_max_overlap', 0.5),
            # Koreksi plat susulan: antrean pending + retry.
            plate_final_min_votes=getattr(args, 'push_plate_final_votes', 2),
            plate_update_retries=getattr(args, 'push_update_retries', 5),
            plate_update_backoff=getattr(args, 'push_update_backoff', 2.0),
            on_push=self._push_callback,  # gate lewat pipeline, bukan ubah tracker
            on_update=self._update_callback,  # koreksi plat-telat, gate sama
        )

        self._logger = None
        if getattr(args, 'log_jsonl', None):
            self._logger = DetectionLogger(args.log_jsonl, source=source_label)

    # ---- Push gate & endpoint (runtime) ------------------------------------

    def _push_callback(self, payload, image_path):
        """Dipanggil tracker.on_push. Cuma teruskan ke uploader kalau push aktif.
        Capture lokal (crop + sidecar) sudah ditulis tracker sebelum ini,
        terlepas dari push_enabled -- sama seperti perilaku CLI lama."""
        if self.push_enabled and self._uploader is not None:
            self._uploader(payload, image_path)

    def _update_callback(self, server_id, fields, on_result=None):
        """Dipanggil tracker.on_update: koreksi plat-telat ke record yang sudah
        ter-push. Gate push_enabled sama dengan _push_callback (tanpa push,
        server_id memang tak pernah ada -- ini sekadar pengaman ganda).

        on_result(ok) diteruskan ke uploader: tracker baru menandai plat
        "terkirim" setelah PATCH-nya sukses, dan mengulang dengan backoff
        kalau gagal."""
        if self.push_enabled and self._uploader is not None:
            self._uploader.push_update(server_id, fields, on_result)
        elif on_result is not None:
            on_result(False)  # tidak terkirim -> tracker simpan & coba lagi

    def _on_registered(self, track_id, server_id):
        """Dipanggil uploader (dari worker thread-nya) saat POST dibalas 201:
        teruskan id record server ke track asalnya, bekal koreksi plat-telat."""
        self.tracker.set_server_id(track_id, server_id)

    @property
    def push_url(self):
        return self._uploader.url if self._uploader is not None else None

    @push_url.setter
    def push_url(self, url):
        if self._uploader is not None:
            self._uploader.url = url

    def connect_web(self, url, mode='multipart', timeout=10.0, retries=2,
                    api_key=None):
        """Buat uploader kalau belum ada (dipakai GUI: 'Sambung' sebelum 'Kirim').
        Kalau sudah ada, cukup ganti URL (dan api_key bila diberikan). Tidak
        menyalakan push_enabled. api_key None = pakai args.api_key / biarkan."""
        if self._uploader is None:
            key = api_key if api_key is not None else getattr(self.args, 'api_key', '')
            self._uploader = HttpUploader(url=url, mode=mode,
                                          timeout=timeout, retries=retries,
                                          api_key=key,
                                          on_registered=self._on_registered)
        else:
            self._uploader.url = url
            if api_key is not None:
                self._uploader.set_api_key(api_key)
        return self._uploader

    @property
    def uploader(self):
        """Untuk baca statistik (sent/failed/dropped) dari GUI."""
        return self._uploader

    # ---- Pemrosesan 1 frame -------------------------------------------------

    def process_frame(self, frame, frame_idx, log_frame=True, draw_overlay=True):
        """detect -> tracker.update -> (log) -> gambar bbox+count.
        Return (annotated, detections). FPS digambar oleh caller.

        draw_overlay=False: deteksi/tracking/push/log TETAP jalan, tapi kotak &
        label TIDAK digambar -> caller dapat frame mentah. Dipakai toggle
        'Tampilkan Hasil Deteksi' di GUI (sembunyikan anotasi tanpa mematikan
        deteksi)."""
        detections = self.detector.detect(frame)
        self.tracker.update(frame, detections)
        if log_frame and self._logger is not None:
            self._logger.log(frame_idx, detections)
        annotated = frame.copy()
        if draw_overlay:
            draw_detections(annotated, detections)
            draw_detection_count(annotated, detections)
        return annotated, detections

    # ---- Shutdown -----------------------------------------------------------

    def release(self):
        """Urutan sama dengan blok finally main.py lama (minus cap/window yang
        dimiliki caller)."""
        try:
            self.tracker.flush()
        except Exception as e:
            print(f'[Pipeline] tracker.flush error: {e}')
        if self._uploader is not None:
            self._uploader.close(drain=True)
        if self._logger is not None:
            self._logger.close()
