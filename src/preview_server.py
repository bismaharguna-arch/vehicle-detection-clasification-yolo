"""HTTP MJPEG server untuk live preview ke web monitoring.

Detector main loop panggil set_latest(annotated_frame) tiap frame
setelah render bbox. start() jalankan Flask di thread daemon supaya
tidak block deteksi.
"""
import threading
import logging

import cv2
from flask import Flask, Response, jsonify

_latest_frame_bytes = None
_lock = threading.Lock()
_frame_event = threading.Event()
_jpeg_quality = 75
_started = False


def set_latest(frame_bgr):
    """Update frame terbaru. Aman dipanggil dari thread deteksi."""
    global _latest_frame_bytes
    ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, _jpeg_quality])
    if not ok:
        return
    data = buf.tobytes()
    with _lock:
        _latest_frame_bytes = data
    _frame_event.set()


def _mjpeg_stream():
    """Generator MJPEG. Throttled ke ~setiap frame baru, max ~15 fps."""
    last_sent = None
    while True:
        # Tunggu sampai ada frame baru (timeout supaya client bisa disconnect bersih)
        _frame_event.wait(timeout=1.0)
        _frame_event.clear()
        with _lock:
            frame = _latest_frame_bytes
        if frame is None or frame is last_sent:
            continue
        last_sent = frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
               + frame + b'\r\n')


_app = Flask(__name__)
# Quiet Flask request logging (biar tidak banjir terminal detector)
logging.getLogger('werkzeug').setLevel(logging.ERROR)


@_app.route('/preview')
def preview():
    return Response(
        _mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@_app.route('/health')
def health():
    with _lock:
        has_frame = _latest_frame_bytes is not None
    return jsonify(status='ok', has_frame=has_frame)


def start(host='0.0.0.0', port=5001, jpeg_quality=75):
    """Jalankan Flask di thread daemon. Non-blocking. Idempotent."""
    global _started, _jpeg_quality
    if _started:
        return
    _jpeg_quality = max(1, min(100, int(jpeg_quality)))

    def _run():
        try:
            _app.run(host=host, port=port, threaded=True,
                     debug=False, use_reloader=False)
        except OSError as e:
            print(f'[WARN] Preview server gagal start di {host}:{port} ({e}). '
                  f'Detector tetap jalan tanpa preview.')

    t = threading.Thread(target=_run, daemon=True, name='preview-server')
    t.start()
    _started = True
    print(f'[INFO] Preview server jalan di http://{host}:{port}/preview')
