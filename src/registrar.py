"""
registrar.py
Heartbeat auto-registrasi detector -> web monitoring.

Sisi WEB punya fitur auto-registrasi: detector cukup mengirim heartbeat berkala
berisi PORT preview-nya, dan web mengambil IP detector dari alamat sumber
request (jadi tidak perlu hardcode IP detector di sisi web lagi).

Kontrak web (fix, jangan diubah):
    POST  http://<IP_WEB>:5000/api/detector/register
    Body  JSON: {"port": <port_preview>}         (opsional {"path": "/preview"})
    OK    200 JSON: {"status":"SUCCESS","preview_url":"...","stale_after_s":60}
    Web menganggap detector OFFLINE bila tidak ada heartbeat > 60 detik.

Pola implementasi mengikuti preview_server.py (thread daemon, non-blocking) dan
uploader.py (pakai requests, timeout pendek, tangkap exception dengan diam).
Heartbeat dikirim ~tiap 15 detik -- aman di bawah ambang stale 60 detik.

Base URL web diturunkan dari --push-url yang sudah dipakai untuk POST deteksi,
mis. "http://10.30.71.12:5000/api/detections" -> "http://10.30.71.12:5000",
lalu ditempel "/api/detector/register". Tidak ada config baru.
"""

import threading
import time

import requests

from urllib.parse import urlsplit


# Path default preview_server (lihat preview_server.py: @_app.route('/preview')).
# Hanya dikirim ke web kalau path detector BEDA dari ini.
_DEFAULT_PREVIEW_PATH = '/preview'


def base_from_push_url(push_url):
    """Turunkan base URL web dari --push-url.

    "http://10.30.71.12:5000/api/detections" -> "http://10.30.71.12:5000".
    Return None kalau URL tidak valid (biar caller bisa skip heartbeat).
    """
    if not push_url:
        return None
    parts = urlsplit(push_url)
    if not parts.scheme or not parts.netloc:
        return None
    return f'{parts.scheme}://{parts.netloc}'


class Heartbeat:
    """Pengirim heartbeat auto-registrasi di thread daemon.

    base_url / enabled bisa diubah saat jalan lewat update() -- dipakai GUI yang
    menyalakan push & live-preview lewat toggle runtime. Untuk CLI cukup buat
    dengan enabled=True dan biarkan.
    """

    def __init__(self, base_url, preview_port, preview_path=_DEFAULT_PREVIEW_PATH,
                 interval=15, timeout=3.0, enabled=True, api_key=''):
        self._base_url = base_url
        self._register_url = (base_url.rstrip('/') + '/api/detector/register'
                              if base_url else None)
        # Kunci X-API-Key (proteksi sisi web). Kosong = header TIDAK dikirim
        # (kompatibel server lama tanpa proteksi).
        self._api_key = api_key or ''

        self.preview_port = int(preview_port)
        self.preview_path = preview_path or _DEFAULT_PREVIEW_PATH
        self.interval = max(1, int(interval))
        self.timeout = float(timeout)

        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()   # bangunkan sleep saat setelan berubah
        self._logged_ok = False          # log sukses cukup sekali (tak banjir)
        self._thread = None

        # ---- status (dibaca lintas-thread lewat status(), dijaga _lock) ----
        # state: 'off' (heartbeat tak aktif) | 'registered' (200 terakhir sukses)
        #        | 'unreachable' (error/non-200 terakhir). GUI merender dari sini.
        self._state = 'off'
        self._last_ok_ts = None          # wall clock (time.time()) sukses terakhir
        self._preview_url = None         # dari respons 200 web (info['preview_url'])

    # ---- kontrol setelan (aman dari thread mana pun) -----------------------

    def update(self, base_url=None, enabled=None, api_key=None):
        """Ubah base_url/enabled/api_key saat jalan. Argumen None = biarkan.
        api_key string kosong = lepas header (beda dari None yang berarti biarkan)."""
        with self._lock:
            if base_url is not None:
                self._base_url = base_url
                self._register_url = base_url.rstrip('/') + '/api/detector/register'
            if enabled is not None:
                self._enabled = bool(enabled)
                self._logged_ok = False  # reset supaya sukses berikutnya kelihatan
                if not self._enabled:
                    self._state = 'off'  # segera tampak mati di GUI (tanpa nunggu loop)
            if api_key is not None:
                self._api_key = api_key
        self._wake.set()  # jangan tunggu penuh 1 interval untuk menerapkan

    def status(self):
        """Snapshot status heartbeat (thread-safe) untuk indikator GUI.

        Return dict: state ('off'|'registered'|'unreachable'), last_ok_ts (epoch
        detik sukses terakhir / None), preview_url (dari respons web / None),
        interval (detik antar-heartbeat -- caller pakai untuk cek 'masih segar')."""
        with self._lock:
            return {
                'state': self._state,
                'last_ok_ts': self._last_ok_ts,
                'preview_url': self._preview_url,
                'interval': self.interval,
            }

    # ---- lifecycle ---------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name='heartbeat')
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._wake.set()

    # ---- loop --------------------------------------------------------------

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                enabled = self._enabled
                url = self._register_url
                api_key = self._api_key
            if enabled and url:
                self._send_once(url, api_key)
            # tidur interval, tapi bisa dibangunkan lebih awal oleh update()/stop()
            self._wake.wait(timeout=self.interval)
            self._wake.clear()

    def _send_once(self, url, api_key=''):
        body = {'port': self.preview_port}
        if self.preview_path and self.preview_path != _DEFAULT_PREVIEW_PATH:
            body['path'] = self.preview_path
        # Header X-API-Key hanya kalau kunci diisi (kosong = mode lama).
        headers = {'X-API-Key': api_key} if api_key else None
        try:
            r = requests.post(url, json=body, timeout=self.timeout,
                              headers=headers)
            if r.status_code == 200:
                info = self._safe_json(r)
                preview_url = info.get('preview_url', '?')
                # Catat status sukses (dibaca GUI). preview_url disimpan tiap
                # sukses; kalau respons tak memuatnya, pertahankan yang lama.
                with self._lock:
                    self._state = 'registered'
                    self._last_ok_ts = time.time()
                    if info.get('preview_url'):
                        self._preview_url = info['preview_url']
                if not self._logged_ok:
                    # Log detail sekali saja saat registrasi pertama berhasil.
                    stale = info.get('stale_after_s', '?')
                    print(f'[Heartbeat] Terdaftar di web -> preview_url={preview_url} '
                          f'stale_after_s={stale}')
                    self._logged_ok = True
                return
            # Non-200: log ringan, coba lagi interval berikutnya.
            self._logged_ok = False
            with self._lock:
                self._state = 'unreachable'
            print(f'[Heartbeat] WARN status={r.status_code} dari {url} '
                  f'body={r.text[:120]}')
        except Exception as e:
            # Web mati / timeout: diam-diam (warning ringan), retry nanti.
            self._logged_ok = False
            with self._lock:
                self._state = 'unreachable'
            print(f'[Heartbeat] web tak terjangkau ({e}); coba lagi '
                  f'{self.interval}s lagi.')

    @staticmethod
    def _safe_json(r):
        try:
            return r.json()
        except Exception:
            return {}


def start_heartbeat(web_base_url, preview_port, preview_path=_DEFAULT_PREVIEW_PATH,
                    interval=15, timeout=3.0, api_key=''):
    """Buat & jalankan Heartbeat di thread daemon. Non-blocking.

    Return objek Heartbeat (punya .update()/.stop()); boleh diabaikan untuk CLI.
    """
    hb = Heartbeat(web_base_url, preview_port, preview_path=preview_path,
                   interval=interval, timeout=timeout, enabled=True,
                   api_key=api_key)
    hb.start()
    print(f'[Heartbeat] Auto-registrasi aktif -> {hb._register_url} '
          f'(port preview {preview_port}, tiap {interval}s)')
    return hb
