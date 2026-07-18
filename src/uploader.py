"""
uploader.py
HTTP uploader untuk push deteksi kendaraan ke web monitoring.

Kontrak endpoint POST /api/detections (v1, Mei 2026):
    Content-Type: multipart/form-data
        plate_number     : string (boleh kosong)
        vehicle_type     : "mobil" | "motor" | "unknown"
        is_electric      : "bensin" | "listrik" | "unknown"
        confidence_score : float 0.0-1.0
        detected_at      : ISO 8601 dengan offset timezone (forward-compat;
                           server saat ini ignore, akan dipakai setelah
                           dukungan ditambahkan tanpa redeploy detector)
        photo            : file JPG

    Response sukses:  201 -> {"status":"SUCCESS","id":...}
    Response gagal :  500 -> {"status":"ERROR","message":...}

Mode 'json' (alternatif tanpa foto) kirim payload JSON langsung.

Koreksi plat susulan (Juli 2026): kalau bacaan plat yang sudah settle berbeda
dari yang terkirim ke web (telat terbaca ATAU terkoreksi jadi bacaan yang
lebih benar), tracker mengantre koreksi lewat push_update():
    PATCH <push_url>/<id>/plate
    Body JSON: {"plate_number": str, "is_electric": str}  (minimal satu)
    Response : 200 SUCCESS/SKIPPED, 404 id tak dikenal, 401 tanpa API key.
<id> berasal dari response 201 POST -- diteruskan ke tracker via callback
on_registered(track_id, server_id).

Field internal prefix '_' (track_id, frames_seen, dll) tidak ikut dikirim
ke server -- itu metadata sidecar lokal saja.
"""

import json
import time
import threading
from queue import Queue, Empty

import requests


WEB_FIELDS = ('plate_number', 'vehicle_type', 'is_electric',
              'confidence_score', 'detected_at')


def _strip_internal(payload):
    """Buang key prefix '_' (sidecar-only) dari payload."""
    return {k: v for k, v in payload.items() if not k.startswith('_')}


class HttpUploader:
    def __init__(self, url, mode='multipart', timeout=10.0, retries=2,
                 headers=None, queue_size=64, api_key='', on_registered=None):
        """
        Args:
            url:        endpoint web monitoring (mis. http://192.168.1.10:5000/api/detections).
            mode:       'multipart' (default, kirim foto) atau 'json' (tanpa foto).
            timeout:    detik per request.
            retries:    jumlah retry kalau bukan 201/2xx.
            headers:    header tambahan opsional.
            queue_size: buffer push (drop kalau penuh).
            api_key:    kunci X-API-Key (proteksi sisi web). Kosong = header
                        TIDAK dikirim (kompatibel server lama tanpa proteksi).
            on_registered: callback(track_id, server_id), dipanggil saat POST
                        dibalas 201 ber-id. Dipakai tracker untuk menandai
                        record web mana milik track mana (bekal koreksi
                        plat-telat). Dipanggil dari worker thread.
        """
        self.url = url
        self.mode = mode
        self.timeout = timeout
        self.retries = retries
        self.headers = headers or {}
        self.on_registered = on_registered
        self.set_api_key(api_key)

        self._queue = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        self.sent = 0
        self.updated = 0
        self.failed = 0
        self.dropped = 0

        print(f'[Uploader] HTTP {mode.upper()} -> {url}')

    def set_api_key(self, api_key):
        """Pasang/lepas header X-API-Key saat runtime (dipakai juga GUI).
        Kosong/None = lepas header (mode lama tanpa proteksi)."""
        if api_key:
            self.headers['X-API-Key'] = api_key
        else:
            self.headers.pop('X-API-Key', None)

    def __call__(self, payload, image_path):
        """Dipanggil sebagai on_push callback dari VehicleTracker."""
        try:
            self._queue.put_nowait(('create', payload, image_path, None))
        except Exception:
            self.dropped += 1
            tid = payload.get('_track_id', '?')
            print(f'[Uploader] queue full, drop track={tid}')

    def push_update(self, server_id, fields, on_result=None):
        """Antre koreksi plat: PATCH <push_url>/<server_id>/plate.
        Dipanggil tracker (via pipeline) saat bacaan plat yang sudah settle
        berbeda dari yang terkirim ke web (telat terbaca / terkoreksi).

        on_result(ok: bool) dipanggil dari worker thread begitu PATCH selesai
        (sukses/gagal permanen). Tracker memakainya untuk menandai plat
        "sudah ada di web" HANYA setelah PATCH benar-benar sukses -- kalau gagal
        ia akan mengantre ulang dengan backoff. Queue penuh = gagal juga."""
        try:
            self._queue.put_nowait(('update', server_id, fields, on_result))
        except Exception:
            self.dropped += 1
            print(f'[Uploader] queue full, drop update id={server_id}')
            self._notify(on_result, False)

    @staticmethod
    def _notify(on_result, ok):
        if on_result is None:
            return
        try:
            on_result(ok)
        except Exception as e:
            print(f'[Uploader] on_result error: {e}')

    def _worker(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                continue
            kind, a, b, cb = item
            if kind == 'create':
                self._send(a, b)
            else:  # 'update'
                self._send_update(a, b, cb)
            self._queue.task_done()

    def _send(self, payload, image_path):
        clean = _strip_internal(payload)
        # Hanya field yang ada di kontrak web v1. Field tambahan (camera_id,
        # client_event_id) sengaja TIDAK dikirim -- server belum punya kolom
        # untuk itu, akan diabaikan diam-diam.
        form_data = {
            'plate_number':     clean.get('plate_number', ''),
            'vehicle_type':     clean.get('vehicle_type', 'unknown'),
            'is_electric':      clean.get('is_electric', 'unknown'),
            'confidence_score': str(clean.get('confidence_score', 0.0)),
            'detected_at':      clean.get('detected_at', ''),
        }
        tid = payload.get('_track_id', '?')

        attempt = 0
        while attempt <= self.retries:
            try:
                if self.mode == 'multipart':
                    with open(image_path, 'rb') as f:
                        files = {
                            'photo': (
                                payload.get('_image_file', 'frame.jpg'),
                                f,
                                'image/jpeg',
                            ),
                        }
                        r = requests.post(
                            self.url, data=form_data, files=files,
                            headers=self.headers, timeout=self.timeout,
                        )
                else:  # json
                    r = requests.post(
                        self.url, json=form_data,
                        headers=self.headers, timeout=self.timeout,
                    )

                if r.status_code == 201:
                    self.sent += 1
                    body = self._safe_json(r)
                    sid = body.get('id') if isinstance(body, dict) else None
                    print(f'[Uploader] OK 201 track={tid} '
                          f'server_id={sid if sid is not None else "?"}')
                    # Kabari tracker id record server -> bekal PATCH koreksi
                    # plat-telat. Berjalan di worker thread; set_server_id
                    # cuma assignment dict (aman di bawah GIL).
                    if self.on_registered is not None and sid is not None:
                        try:
                            self.on_registered(payload.get('_track_id'), sid)
                        except Exception as e:
                            print(f'[Uploader] on_registered error: {e}')
                    return
                if 200 <= r.status_code < 300:
                    # Bukan 201, tapi 2xx -- log warning, tetap dianggap sukses.
                    self.sent += 1
                    print(f'[Uploader] WARN status={r.status_code} '
                          f'(expected 201) track={tid} body={r.text[:200]}')
                    return
                # 4xx/5xx -> log + retry
                print(f'[Uploader] HTTP {r.status_code} track={tid} '
                      f'body={r.text[:200]}')
            except Exception as e:
                print(f'[Uploader] error track={tid}: {e}')
            attempt += 1
            time.sleep(0.5 * attempt)

        self.failed += 1

    def _send_update(self, server_id, fields, on_result=None):
        """PATCH koreksi ke record yang sudah ada di server.
        URL diturunkan dari push url: .../api/detections -> .../<id>/plate.
        Hasil akhir dilaporkan lewat on_result(ok) -- tracker yang memutuskan
        mau retry (backoff) atau menyerah."""
        url = f'{self.url.rstrip("/")}/{server_id}/plate'
        attempt = 0
        while attempt <= self.retries:
            try:
                r = requests.patch(url, json=fields, headers=self.headers,
                                   timeout=self.timeout)
                if 200 <= r.status_code < 300:
                    # 200 SUCCESS, atau SKIPPED kalau record sudah dikoreksi
                    # admin (koreksi manusia menang; jangan retry).
                    self.updated += 1
                    body = self._safe_json(r)
                    status = (body.get('status', 'SUCCESS')
                              if isinstance(body, dict) else 'SUCCESS')
                    print(f'[Uploader] UPDATE {status} id={server_id} '
                          f'fields={list(fields.keys())}')
                    self._notify(on_result, True)
                    return
                if r.status_code == 404:
                    # Record hilang di server -- percuma retry.
                    print(f'[Uploader] UPDATE 404 id={server_id}; skip.')
                    self.failed += 1
                    self._notify(on_result, True)  # jangan disuruh coba lagi
                    return
                print(f'[Uploader] UPDATE HTTP {r.status_code} id={server_id} '
                      f'body={r.text[:200]}')
            except Exception as e:
                print(f'[Uploader] UPDATE error id={server_id}: {e}')
            attempt += 1
            time.sleep(0.5 * attempt)
        self.failed += 1
        self._notify(on_result, False)

    @staticmethod
    def _safe_json(r):
        try:
            return r.json()
        except Exception:
            return {}

    def close(self, drain=True, drain_timeout=5.0):
        if drain:
            try:
                deadline = time.time() + drain_timeout
                while not self._queue.empty() and time.time() < deadline:
                    time.sleep(0.1)
            except Exception:
                pass
        self._stop.set()
        self._thread.join(timeout=2.0)
        print(f'[Uploader] sent={self.sent} updated={self.updated} '
              f'failed={self.failed} dropped={self.dropped}')
