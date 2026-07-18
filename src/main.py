"""
main.py
Entry point pipeline deteksi kendaraan.

Loop per frame:
    frame -> VehicleDetector (vehicle YOLO + plate YOLO + FuelClassifier + PaddleOCR)
          -> VehicleTracker  (IoU dedup, 1-shot push)
          -> HttpUploader    (optional, multipart POST ke web monitoring)
          -> draw_detections + preview_server (MJPEG) + cv2.imshow

Kontrak push (Indonesia, v1) di-handle di VehicleTracker / HttpUploader.
File ini cuma orkestrasi + CLI + interactive controls + FPS.

Controls (saat window aktif):
    Q       : quit
    Space   : pause / resume
    S       : screenshot annotated frame ke output/
    L       : toggle loop (untuk source video)
    Left    : step 1 frame mundur (saat paused, source video)
    Right   : step 1 frame maju (saat paused)
"""

import argparse
import os
import sys
import time

import cv2

from pipeline import DetectionPipeline
from utils import draw_detections, draw_fps, draw_detection_count
import preview_server
import registrar


# Kunci X-API-Key web monitoring dibaca dari file lokal api_key.txt di root
# proyek -- file itu masuk .gitignore, jadi kunci TIDAK pernah ikut ke source
# code / repo. CLI maupun GUI otomatis pakai ini lewat default_config();
# override sekali jalan tetap bisa lewat --api-key.
_API_KEY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'api_key.txt')


def _load_api_key():
    """Baca kunci dari api_key.txt (baris pertama, whitespace di-strip).
    File tidak ada / kosong -> '' (header X-API-Key tidak dikirim)."""
    try:
        with open(_API_KEY_FILE, 'r', encoding='utf-8') as f:
            return f.readline().strip()
    except OSError:
        return ''


DEFAULT_API_KEY = _load_api_key()


def build_parser():
    p = argparse.ArgumentParser(
        description='Deteksi kendaraan + klasifikasi BBM + OCR plat + push web monitoring.'
    )

    # ---- Source ----
    p.add_argument('--source', default='0',
                   help='Webcam id (angka) atau path video. Default: 0.')
    p.add_argument('--loop', action='store_true',
                   help='Loop video saat sampai akhir (toggleable dengan tombol L).')
    p.add_argument('--cam-fps', type=int, default=30,
                   help='Paksa FPS webcam. Default 30 (auto-exposure dipaksa ON saat buka, '
                        'jadi terang @30fps kalau cahaya cukup). Di ruangan redup turunkan '
                        '(mis. --cam-fps 10) atau tambah cahaya: 30fps = exposure pendek = gelap.')

    # ---- Models ----
    # Unified 3-kelas (mobil/motor/plat) -> 1 model, 1x predict/frame. Ini default sekarang.
    # Set --model "" untuk balik ke dual-model (--vehicle-model + --plate-model).
    p.add_argument('--model', default='',
                   help='Model unified 3-kelas. Kosongkan ("") untuk pakai dual-model (default).')
    p.add_argument('--vehicle-model', default='models/yolov8n.pt')
    p.add_argument('--plate-model', default='models/plate_best (1).pt')
    p.add_argument('--device', default=None,
                   help='cuda / cpu. Default: auto-detect.')
    p.add_argument('--imgsz', type=int, default=960,
                   help='Resolusi inference YOLO untuk PLAT (dan model unified). '
                        'Default 960 (turun dari 1280 demi FPS; pakai 640 utk FPS max, 1280 utk plat jauh).')
    p.add_argument('--vehicle-imgsz', type=int, default=640,
                   help='Resolusi inference model KENDARAAN (dual-model saja). '
                        'Mobil/motor objek besar, 640 (native yolov8n) cukup -- '
                        'lebih cepat tanpa mengorbankan akurasi plat. '
                        '0 = samakan dengan --imgsz.')
    p.add_argument('--no-half', action='store_true',
                   help='Matikan FP16. Default: FP16 otomatis ON di CUDA '
                        '(inference lebih cepat, akurasi praktis sama); '
                        'di CPU selalu FP32.')
    p.add_argument('--conf', type=float, default=0.4,
                   help='Confidence threshold vehicle.')
    p.add_argument('--plate-conf', type=float, default=None,
                   help='Confidence threshold plat (default = --conf).')

    # ---- OCR ----
    p.add_argument('--no-ocr', action='store_true',
                   help='Matikan PaddleOCR (smoke test / hemat resource).')
    p.add_argument('--ocr-interval', type=int, default=10,
                   help='Re-OCR plat sama tiap N frame. Default 10.')
    p.add_argument('--ocr-cpu', action='store_true',
                   help='Paksa PaddleOCR pakai CPU walau GPU tersedia.')
    p.add_argument('--ocr-min-height', type=int, default=22,
                   help='Jangan OCR plat lebih pendek dari ini (px) -- terlalu jauh, '
                        'bacaan pasti jelek. Cegah "salah pas awal". Default 22.')
    p.add_argument('--ocr-upscale', type=int, default=0,
                   help='Lebar target upscale crop plat (px). 0 = native/OFF (default; '
                        'uji A/B: plat sudah >100px jadi lebih konsisten tanpa upscale). '
                        'Set mis. 200 hanya kalau plat sering kekecilan.')
    p.add_argument('--ocr-min-height-ratio', type=float, default=0.6,
                   help='Ambil hanya teks yang tingginya >= rasio ini dari teks '
                        'tertinggi dalam box plat (buang tanggal pajak yg fontnya '
                        'kecil, tahan plat miring). Default 0.6. Turunkan kalau '
                        'huruf plat ikut kebuang, naikkan kalau pajak masih bocor.')
    p.add_argument('--ocr-crop-bottom', type=float, default=0.0,
                   help='Potong rasio bawah plat (strip pajak "04-30") setelah deskew. '
                        'Default 0 (OFF -- merusak plat kecil). Nyalakan ~0.16 hanya '
                        'kalau plat besar & strip pajak bocor ke teks.')
    p.add_argument('--ocr-no-strict-region', dest='ocr_strict_region',
                   action='store_false',
                   help='Jangan tolak bacaan yang huruf depannya bukan kode wilayah '
                        'Indonesia sah. Default: strict (tolak "I" dsb) supaya bacaan '
                        'ngawur tak ikut terkirim.')

    # ---- Tracker / push ----
    p.add_argument('--push', action='store_true',
                   help='Aktifkan HTTP push ke web monitoring.')
    p.add_argument('--push-url', default=None,
                   help='Endpoint web monitoring, mis. http://192.168.1.10:5000/api/detections')
    p.add_argument('--push-mode', default='multipart', choices=['multipart', 'json'])
    p.add_argument('--api-key', default=DEFAULT_API_KEY,
                   help='Kunci X-API-Key untuk endpoint web (push deteksi & '
                        'heartbeat registrasi). Default: isi file api_key.txt '
                        'di root proyek (gitignored). Kosong = header tidak '
                        'dikirim (server tanpa proteksi).')
    p.add_argument('--push-timeout', type=float, default=10.0)
    p.add_argument('--push-retries', type=int, default=2)
    p.add_argument('--push-stable', type=int, default=5,
                   help='Min frame stabil sebelum push. Default 5.')
    p.add_argument('--push-timeout-frames', type=int, default=30,
                   help='Drop track kalau tidak terlihat selama N frame. Default 30.')
    p.add_argument('--push-min-height', type=int, default=100,
                   help='Min tinggi bbox kendaraan (px) untuk di-push.')
    p.add_argument('--push-max-dim', type=int, default=800,
                   help='Resize foto sehingga max(w,h) <= ini.')
    p.add_argument('--push-jpeg-quality', type=int, default=85)
    p.add_argument('--push-fuel-optional', action='store_true',
                   help='Push tanpa nunggu fuel terdeteksi (fuel="unknown").')
    p.add_argument('--push-fuel-wait', type=int, default=0,
                   help='Frame menunggu fuel sebelum push "unknown". 0 (default) = '
                        'jangan pernah push kendaraan yang platnya belum terdeteksi '
                        '(fuel unknown). Isi >0 untuk perilaku lama (push "unknown" '
                        'setelah N frame).')
    p.add_argument('--push-dedup-seconds', type=float, default=60.0,
                   help='Jendela anti-duplikat dalam DETIK (wall clock, bukan '
                        'frame -> tidak bergantung FPS): plat yang sama tidak '
                        'di-push ulang sebagai baris baru dalam jendela ini '
                        '(menangkap kendaraan yang deteksinya putus lama lalu '
                        'lahir ulang). <=0 = matikan. Default 60.')
    p.add_argument('--push-plate-wait', type=int, default=20,
                   help='Batas frame menunggu teks plat terkonfirmasi (lolos 1 siklus '
                        're-OCR) sebelum push. Lewat ini -> push apa adanya (asal '
                        'nomor plat sudah tidak kosong; lihat --push-allow-empty-plate). '
                        '0 = tunggu terus. Cegah bacaan OCR pertama yang salah ikut terkirim.')
    p.add_argument('--push-allow-empty-plate', dest='push_require_plate',
                   action='store_false',
                   help='Izinkan push kendaraan tanpa nomor plat (plate_number=""). '
                        'Default: JANGAN kirim baris tanpa plat ke web -- track tanpa '
                        'plat menunggu sampai platnya terbaca, kalau tidak ikut terbuang '
                        'saat timeout. Nyalakan flag ini hanya untuk smoke test.')
    p.add_argument('--push-plate-final-votes', type=int, default=2,
                   help='Saat track mati sebelum platnya settle, kirim bacaan plat '
                        'terakhir yang muncul minimal N frame (bukan yang terbanyak: '
                        'bacaan makin baru = kendaraan makin dekat = makin benar). '
                        '1 = kirim bacaan apa pun. Default 2.')
    p.add_argument('--push-update-retries', type=int, default=5,
                   help='Berapa kali koreksi plat (PATCH) diulang kalau gagal, dengan '
                        'backoff eksponensial. Koreksi baru ditandai terkirim setelah '
                        'PATCH sukses. Default 5.')
    p.add_argument('--push-update-backoff', type=float, default=2.0,
                   help='Detik dasar backoff antar percobaan PATCH '
                        '(delay = backoff * 2^(n-1), cap 60s). Default 2.0.')

    # ---- Tracking / asosiasi (anti ID-switch) ----
    p.add_argument('--track-max-dist', type=float, default=1.2,
                   help='Jarak pusat maksimum (dinormalisasi ke diagonal bbox track) '
                        'supaya deteksi masih boleh dicocokkan walau IoU-nya 0 -- '
                        'menjembatani occlusion sesaat. Default 1.2.')
    p.add_argument('--track-predict-frames', type=int, default=10,
                   help='Berapa frame track boleh disambung lewat prediksi posisi saja '
                        '(IoU 0, cuma dekat) saat tertutup. Lewat itu, match wajib '
                        'bertumpuk -- cegah track lama menyambar kendaraan baru di '
                        'lajur yang sama. Default 10.')
    p.add_argument('--track-max-cost', type=float, default=0.8,
                   help='Cost gabungan (IoU + jarak + bentuk, 0-1) maksimum yang masih '
                        'diterima sebagai match. Besar = longgar. Default 0.8.')
    p.add_argument('--track-max-overlap', type=float, default=0.5,
                   help='Kalau bbox kendaraan bertumpuk dengan kendaraan lain (IoU) di '
                        'atas nilai ini, frame tsb tidak dipakai sebagai foto push '
                        '(cegah foto salah kendaraan). Default 0.5.')

    # ---- Preview server (MJPEG) ----
    p.add_argument('--no-preview', action='store_true',
                   help='Matikan MJPEG preview server.')
    p.add_argument('--preview-host', default='0.0.0.0')
    p.add_argument('--preview-port', type=int, default=5001)
    p.add_argument('--preview-quality', type=int, default=75)

    # ---- Window ----
    p.add_argument('--no-window', action='store_true',
                   help='Jangan tampilkan window cv2.imshow (headless).')
    p.add_argument('--window-name', default='Deteksi Kendaraan')

    # ---- Logging / output ----
    p.add_argument('--log-jsonl', default=None,
                   help='Path file JSONL audit log per deteksi.')
    p.add_argument('--output-dir', default='output',
                   help='Folder untuk screenshot dan capture push.')

    return p


def default_config():
    """Namespace default (semua flag di nilai default-nya). Dipakai GUI supaya
    tidak menduplikasi daftar default CLI."""
    return build_parser().parse_args([])


def parse_args():
    p = build_parser()
    args = p.parse_args()

    if args.push and not args.push_url:
        p.error('--push butuh --push-url juga.')

    return args


def open_source(source_arg, cam_fps=0):
    """Buka source video atau webcam. Return (VideoCapture, is_video_file, source_str)."""
    # Coba integer dulu (webcam id)
    try:
        cam_id = int(source_arg)
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW if os.name == 'nt' else 0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)
        # ponytail: cuma set FPS. Prop exposure/gain OpenCV di webcam UVC ini
        # ga reliable (di-tes: hasil bright loncat 20<->134 ga reproducible,
        # set AUTO_EXPOSURE malah bikin gelap+lambat). Atur kecerahan lewat
        # Windows Camera settings / cahaya ruangan, bukan dari sini.
        if cam_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, cam_fps)
        return cap, False, f'webcam:{cam_id}'
    except ValueError:
        pass

    # File path
    if not os.path.exists(source_arg):
        print(f'[ERROR] Source tidak ditemukan: {source_arg}')
        sys.exit(1)
    cap = cv2.VideoCapture(source_arg)
    return cap, True, source_arg


def main():
    args = parse_args()

    # ---- Output dirs (screenshot 'S' butuh ini; captures dikelola pipeline) ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Source ----
    cap, is_video_file, source_label = open_source(args.source, args.cam_fps)
    if not cap.isOpened():
        print(f'[ERROR] Gagal membuka source: {args.source}')
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video_file else 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f'[Main] Source: {source_label} '
          f'({total_frames} frames @ {src_fps:.1f} fps)' if is_video_file
          else f'[Main] Source: {source_label}')

    # ---- Pipeline (rakit detector/fuel/plate/tracker/uploader/logger) ----
    pipeline = DetectionPipeline(args, source_label=source_label)

    if not args.no_preview:
        preview_server.start(
            host=args.preview_host,
            port=args.preview_port,
            jpeg_quality=args.preview_quality,
        )

    # ---- Heartbeat auto-registrasi ke web (kapan pun preview aktif + web dikenal) ----
    # Heartbeat mengiklankan URL PREVIEW ke web, jadi cukup bergantung pada preview:
    # nyala kapan pun preview aktif DAN base URL web tersedia (diturunkan dari
    # --push-url), TANPA mensyaratkan --push. Jadi user bisa kasih --push-url tanpa
    # --push untuk stream-only. Web ambil IP detector dari sumber request; port dari
    # --preview-port. Preview mati atau --push-url kosong/invalid -> jangan kirim.
    web_base = registrar.base_from_push_url(args.push_url)
    if not args.no_preview and web_base:
        registrar.start_heartbeat(web_base, args.preview_port,
                                  api_key=args.api_key)
    elif not args.no_preview and args.push_url:
        print(f'[Heartbeat] --push-url tak valid ({args.push_url}); '
              f'auto-registrasi dilewati.')

    # ---- State ----
    paused = False
    loop_video = args.loop
    frame_idx = 0
    last_frame_for_pause = None
    last_dets_for_pause = []

    fps_window = []
    fps_last_print = time.time()

    print('[Main] Mulai loop. Controls: Q quit, Space pause, S screenshot, L loop, <-/-> step.')

    try:
        while True:
            t0 = time.time()

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_video_file and loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print('[Main] Sumber habis. Keluar.')
                    break
                frame_idx += 1

                annotated, detections = pipeline.process_frame(frame, frame_idx)

                last_frame_for_pause = frame
                last_dets_for_pause = detections
            else:
                # Saat paused: re-render frame terakhir (tanpa re-detect/update).
                if last_frame_for_pause is None:
                    time.sleep(0.05)
                    continue
                detections = last_dets_for_pause
                annotated = draw_detections(last_frame_for_pause.copy(), detections)
                draw_detection_count(annotated, detections)

            # FPS rolling window
            dt = time.time() - t0
            if dt > 0:
                fps_window.append(1.0 / dt)
                if len(fps_window) > 30:
                    fps_window.pop(0)
            cur_fps = sum(fps_window) / len(fps_window) if fps_window else 0.0
            draw_fps(annotated, cur_fps)

            if paused:
                cv2.putText(annotated, 'PAUSED', (15, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Push ke MJPEG preview
            if not args.no_preview:
                preview_server.set_latest(annotated)

            # Periodic FPS log ke stdout
            if time.time() - fps_last_print > 5.0:
                fps_last_print = time.time()
                pos = ''
                if is_video_file and total_frames:
                    pos = f' frame {frame_idx}/{total_frames}'
                print(f'[Main] {cur_fps:.1f} FPS{pos} '
                      f'(tracks active, last detections: {len(detections)})')

            # Window + key handling
            if not args.no_window:
                cv2.imshow(args.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            if key == ord('q') or key == 27:  # Q / ESC
                break
            elif key == ord(' '):
                paused = not paused
                print(f'[Main] {"PAUSED" if paused else "RESUMED"}')
            elif key == ord('s'):
                ts = time.strftime('%Y%m%d_%H%M%S')
                path = os.path.join(args.output_dir, f'screenshot_{ts}_f{frame_idx}.jpg')
                cv2.imwrite(path, annotated)
                print(f'[Main] Screenshot -> {path}')
            elif key == ord('l'):
                loop_video = not loop_video
                print(f'[Main] Loop video: {"ON" if loop_video else "OFF"}')
            elif key == 81 or key == 2424832:  # Left arrow (linux / win)
                if paused and is_video_file:
                    target = max(0, frame_idx - 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    ret, frame = cap.read()
                    if ret:
                        frame_idx = target + 1
                        _, detections = pipeline.process_frame(frame, frame_idx, log_frame=False)
                        last_frame_for_pause = frame
                        last_dets_for_pause = detections
            elif key == 83 or key == 2555904:  # Right arrow
                if paused and is_video_file:
                    ret, frame = cap.read()
                    if ret:
                        frame_idx += 1
                        _, detections = pipeline.process_frame(frame, frame_idx, log_frame=False)
                        last_frame_for_pause = frame
                        last_dets_for_pause = detections

    except KeyboardInterrupt:
        print('\n[Main] Ctrl-C, keluar...')
    finally:
        print('[Main] Cleanup...')
        pipeline.release()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        print('[Main] Selesai.')


if __name__ == '__main__':
    main()
