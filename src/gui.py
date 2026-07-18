"""
gui.py
GUI desktop (Tkinter) untuk pipeline deteksi kendaraan.

TAMBAHAN, bukan pengganti: main.py (CLI) tetap jalan seperti biasa. GUI ini
memakai DetectionPipeline yang sama (tidak menduplikasi logika deteksi) dan
menampilkan frame beranotasi lewat jalur ImageTk sendiri (bukan cv2.imshow).

Model threading (Tkinter TIDAK thread-safe):
- Thread background (_worker): buka kamera -> loop baca frame ->
  pipeline.process_frame(frame) -> simpan hasil ke self._latest (dilindungi
  lock). TIDAK menyentuh widget apa pun.
- Main thread (Tkinter): _refresh() dipanggil ulang tiap ~30ms via root.after,
  ambil self._latest, konversi BGR->RGB->PIL->ImageTk, tampilkan ke Label.

Menutup jendela = berhenti rapi: stop worker -> release kamera -> pipeline.release().

Prompt 2: sumber default webcam id 0, tombol Start/Stop, status + FPS.
(Pilih sumber / web monitoring menyusul di Prompt 3-4.)
"""

import os
import threading
import time

import glob
import socket
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import requests
from PIL import Image, ImageTk

from main import default_config, DEFAULT_API_KEY
from pipeline import DetectionPipeline
import preview_server
import registrar

PREVIEW_MAX_W = 960  # frame lebih lebar dari ini diperkecil supaya window wajar
PREVIEW_PORT = 5001  # MJPEG live stream (tonton di browser)
WEBCAM_IDS = ('0', '1', '2', '3')  # ponytail: daftar tetap; enumerasi webcam di
                                   # Windows ribet & tak reliabel. User pilih id.
VIDEO_EXTS = ('*.mp4', '*.avi', '*.mov')

# Tujuan koneksi web monitoring (label -> (ip, port)). Dropdown mengisi kotak
# IP & Port. GANTI IP 'Web monitoring tim' ke alamat server tim yang sebenarnya.
WEB_TARGETS = [
    ('Komputer ini (tes lokal)', ('127.0.0.1', '5000')),
    ('Tambah IP sendiri…',       ('', '5000')),           # ip kosong -> bisa diketik
]


class DetectorGUI:
    def __init__(self, root):
        self.root = root
        root.title('Deteksi Kendaraan')

        # ---- state ----
        self.pipeline = None            # dibangun lazily saat Start pertama
        self.cap = None
        self.worker = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None             # frame BGR beranotasi terbaru
        self._fps = 0.0
        self._running = False
        self._frame_idx = 0
        self._status = 'Berhenti'
        self._imgtk = None              # WAJIB simpan ref, kalau tidak ke-GC -> blank
        self._last_btn_running = False  # cache supaya tak reconfigure tiap 30ms

        self.source = 0                 # webcam id default (Prompt 3: bisa diganti)

        # ---- state web monitoring (Prompt 4) ----
        self._web_base = ''             # base url tervalidasi, mis. http://ip:5000
        self._web_key = DEFAULT_API_KEY  # X-API-Key tertanam (lihat main.py);
                                         # sengaja tanpa input di GUI -- user
                                         # tidak perlu tahu/isi kuncinya
        self._connected = False
        self._conn_status = 'Belum tersambung'
        self._want_push = False         # keinginan toggle Kirim (diterapkan ke pipeline)
        self._last_conn_render = None   # cache render indikator

        # ---- state live-ke-web (MJPEG stream ke browser) ----
        self._web_live = False          # umpankan frame ke preview_server?
        self._lan_ip = self._get_lan_ip()
        self._last_hb_render = None     # cache render indikator heartbeat

        # ---- heartbeat auto-registrasi (aktif bila streaming live aktif) ----
        # Heartbeat mengiklankan URL PREVIEW ke web, jadi cukup bergantung pada
        # streaming -- tidak perlu 'Kirim Data' (push) menyala.
        self._heartbeat = None          # dibuat lazily saat pertama kali aktif

        # ---- state tampilan hasil deteksi (on/off anotasi kotak & label) ----
        # Deteksi/tracking/push TETAP jalan saat OFF; cuma anotasi disembunyikan.
        self._show_overlay = True

        # ---- widget: bar kontrol atas ----
        top = ttk.Frame(root, padding=6)
        top.pack(fill='x')
        self.btn_start = ttk.Button(top, text='▶ Start', command=self.start)
        self.btn_start.pack(side='left')
        self.btn_stop = ttk.Button(top, text='■ Stop', command=self.stop,
                                   state='disabled')
        self.btn_stop.pack(side='left', padx=(6, 0))
        self.lbl_status = ttk.Label(top, text='Berhenti')
        self.lbl_status.pack(side='left', padx=12)

        # ---- widget: pemilih sumber (aktif hanya saat Stop) ----
        src = ttk.LabelFrame(root, text='Sumber', padding=6)
        src.pack(fill='x', padx=6, pady=(0, 6))
        self._src_widgets = []  # widget yang di-disable saat jalan

        ttk.Label(src, text='Webcam:').pack(side='left')
        self.cmb_webcam = ttk.Combobox(src, width=4, state='readonly',
                                       values=WEBCAM_IDS)
        self.cmb_webcam.set('0')
        self.cmb_webcam.bind('<<ComboboxSelected>>', self._on_pick_webcam)
        self.cmb_webcam.pack(side='left', padx=(4, 12))
        self._src_widgets.append(self.cmb_webcam)

        self.btn_openvid = ttk.Button(src, text='Buka Video...',
                                      command=self._on_open_video)
        self.btn_openvid.pack(side='left')
        self._src_widgets.append(self.btn_openvid)

        for name, path in self._scan_default_videos():
            b = ttk.Button(src, text=name, command=lambda p=path: self._set_source(p))
            b.pack(side='left', padx=(6, 0))
            self._src_widgets.append(b)

        self.lbl_source = ttk.Label(src, text='→ webcam:0')
        self.lbl_source.pack(side='right')

        # ---- widget: web monitoring (Prompt 4, dropdown tujuan) ----
        web = ttk.LabelFrame(root, text='Web Monitoring', padding=6)
        web.pack(fill='x', padx=6, pady=(0, 6))

        # Baris 1: pilih tujuan -> isi URL -> Sambung -> indikator
        r1 = ttk.Frame(web)
        r1.pack(fill='x')
        ttk.Label(r1, text='Tujuan:').pack(side='left')
        self.cmb_target = ttk.Combobox(r1, width=24, state='readonly',
                                       values=[label for label, _ in WEB_TARGETS])
        self.cmb_target.bind('<<ComboboxSelected>>', self._on_pick_target)
        self.cmb_target.pack(side='left', padx=(4, 8))

        ttk.Label(r1, text='IP:').pack(side='left')
        self.ent_ip = ttk.Entry(r1, width=14)
        self.ent_ip.pack(side='left', padx=(2, 6))
        ttk.Label(r1, text='Port:').pack(side='left')
        self.ent_port = ttk.Entry(r1, width=6)
        self.ent_port.pack(side='left', padx=(2, 8))
        self.btn_connect = ttk.Button(r1, text='Sambung', command=self._on_connect)
        self.btn_connect.pack(side='left')
        self.lbl_conn = ttk.Label(r1, text='● Belum tersambung', foreground='gray')
        self.lbl_conn.pack(side='left', padx=10)

        # Baris 2: toggle Kirim + keterangan + statistik
        r2 = ttk.Frame(web)
        r2.pack(fill='x', pady=(6, 0))
        self._push_var = tk.BooleanVar(value=False)
        self.chk_push = ttk.Checkbutton(r2, text='Kirim Data', variable=self._push_var,
                                        command=self._on_toggle_push, state='disabled')
        self.chk_push.pack(side='left')
        ttk.Label(r2, foreground='gray',
                  text="'Sambung' hanya cek server hidup (belum kirim). "
                       "Nyalakan 'Kirim Data' untuk mulai mengirim hasil deteksi."
                  ).pack(side='left', padx=8)
        self.lbl_stats = ttk.Label(r2, text='Terkirim: 0  Gagal: 0')
        self.lbl_stats.pack(side='right')

        # Pilih tujuan default (lokal) & isi URL-nya.
        self.cmb_target.current(0)
        self._on_pick_target()

        # ---- widget: live ke web (MJPEG stream ke browser) ----
        live = ttk.LabelFrame(root, text='Live ke Web (tonton di browser)', padding=6)
        live.pack(fill='x', padx=6, pady=(0, 6))
        self._live_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(live, text='Aktifkan streaming', variable=self._live_var,
                        command=self._on_toggle_live).pack(side='left')
        self.lbl_live = ttk.Label(live, text='(mati)', foreground='gray')
        self.lbl_live.pack(side='left', padx=10)
        self.btn_openlive = ttk.Button(live, text='Buka di browser',
                                       command=self._open_live, state='disabled')
        self.btn_openlive.pack(side='left')
        # Indikator registrasi heartbeat (apakah web sudah tahu URL preview kita).
        # Diperbarui oleh _refresh (main thread) dari _heartbeat.status().
        self.lbl_hb = ttk.Label(live, text='(preview belum aktif)', foreground='gray')
        self.lbl_hb.pack(side='left', padx=10)

        # ---- widget: tampilan (on/off hasil deteksi) ----
        disp = ttk.LabelFrame(root, text='Tampilan', padding=6)
        disp.pack(fill='x', padx=6, pady=(0, 6))
        self._overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(disp, text='Tampilkan Hasil Deteksi (kotak & label)',
                        variable=self._overlay_var,
                        command=self._on_toggle_overlay).pack(side='left')
        self.lbl_overlay = ttk.Label(disp, text='(aktif)', foreground='green')
        self.lbl_overlay.pack(side='left', padx=10)
        ttk.Label(disp, foreground='gray',
                  text='Mematikan ini hanya menyembunyikan kotak/label; '
                       'deteksi & pengiriman tetap berjalan.'
                  ).pack(side='left', padx=8)

        # ---- widget: panel preview ----
        self.preview = ttk.Label(root, background='black', anchor='center')
        self.preview.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self._refresh()  # mulai loop render main-thread

    # ---- kontrol ------------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._stop.clear()
        self._running = True  # _refresh (main thread) yang sync tombol dari flag ini
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def stop(self):
        # Cukup sinyalkan; worker keluar sendiri, set _running=False & status.
        self._stop.set()
        self._set_status('Menghentikan...')

    # ---- pemilih sumber -----------------------------------------------------

    def _scan_default_videos(self):
        """Cari file video di folder project (cwd). Return [(nama, path_relatif)].
        Kalau tidak ada, list kosong -> tak ada tombol default."""
        found = []
        for pat in VIDEO_EXTS:
            found.extend(glob.glob(pat))
        # dedup + urut stabil
        return [(os.path.basename(p), p) for p in sorted(set(found))]

    def _on_pick_webcam(self, _event=None):
        self._set_source(int(self.cmb_webcam.get()))

    def _on_open_video(self):
        path = filedialog.askopenfilename(
            title='Pilih video',
            filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('Semua', '*.*')])
        if path:
            self._set_source(path)

    def _set_source(self, src):
        """Ganti sumber (hanya berlaku saat berikutnya Start). src = int webcam
        id atau path video."""
        if self._running:
            return  # kontrol harusnya sudah disabled; jaga-jaga saja
        self.source = src
        label = f'webcam:{src}' if isinstance(src, int) else os.path.basename(str(src))
        self.lbl_source.configure(text=f'→ {label}')

    # ---- web monitoring (Prompt 4) -----------------------------------------

    def _on_pick_target(self, _event=None):
        """Isi kotak IP & Port dari tujuan yang dipilih. Kotak selalu bisa
        diedit -- dropdown cuma pengisi cepat, bukan penyetel yang mengunci."""
        ip, port = dict(WEB_TARGETS).get(self.cmb_target.get(), ('', ''))
        for ent, val in ((self.ent_ip, ip), (self.ent_port, port)):
            ent.delete(0, 'end')
            if val:
                ent.insert(0, val)
        self.ent_ip.focus_set()

    def _on_connect(self):
        """Validasi koneksi ke http://<ip>:<port>/api/stats (belum mengirim apa pun)."""
        ip = self.ent_ip.get().strip()
        port = self.ent_port.get().strip()
        if not ip or not port:
            self._conn_status = 'IP/Port kosong'
            self._connected = False
            return
        url = f'http://{ip}:{port}'
        self._conn_status = 'Menyambung...'
        self._connected = False
        # requests.get bisa nge-block; jalankan di thread biar GUI tetap responsif.
        threading.Thread(target=self._test_connection, args=(url,), daemon=True).start()

    def _test_connection(self, url):
        try:
            r = requests.get(url + '/api/stats', timeout=3)
            if r.status_code == 200:
                self._web_base = url
                self._connected = True
                self._conn_status = 'Terhubung'
            else:
                self._connected = False
                self._conn_status = f'Gagal ({r.status_code})'
        except Exception:
            self._connected = False
            self._conn_status = 'Gagal (tak terjangkau)'
        self._apply_web()

    def _on_toggle_push(self):
        # Toggle hanya bisa diklik kalau sudah Terhubung (chk di-disable kalau tidak).
        self._want_push = bool(self._push_var.get())
        self._apply_web()

    def _apply_web(self):
        """Terapkan url + status push ke pipeline. Aman dipanggil dari thread mana
        pun (tidak menyentuh widget). Kalau pipeline belum ada (belum Start),
        setelan tersimpan & diterapkan saat pipeline dibangun di _worker."""
        if self.pipeline is None or not self._connected or not self._web_base:
            return
        self.pipeline.connect_web(self._web_base + '/api/detections',
                                  api_key=self._web_key)
        self.pipeline.push_enabled = self._want_push
        self._apply_heartbeat()

    def _apply_heartbeat(self):
        """Nyalakan/matikan heartbeat auto-registrasi. Karena heartbeat
        mengiklankan URL PREVIEW ke web, syaratnya cukup: streaming live-ke-web
        menyala DAN base URL web tersedia -- TIDAK perlu 'Kirim Data' (push).
        Base URL diturunkan dari _web_base (server yang sama dengan tujuan push),
        port preview = PREVIEW_PORT. Aman dipanggil dari thread mana pun
        (registrar pakai thread daemon sendiri; tidak menyentuh widget)."""
        enabled = bool(self._web_live and self._web_base)
        if self._heartbeat is None:
            if not enabled:
                return
            self._heartbeat = registrar.start_heartbeat(self._web_base, PREVIEW_PORT,
                                                        api_key=self._web_key)
        else:
            self._heartbeat.update(base_url=self._web_base or None, enabled=enabled,
                                   api_key=self._web_key)

    # ---- live ke web (MJPEG stream) ----------------------------------------

    def _get_lan_ip(self):
        """IP LAN mesin ini (untuk URL yang dibuka dari komputer lain).
        Cari IP dari gateway default (tidak bergantung reachability 8.8.8.8)."""
        try:
            route = [l for l in os.popen('route print 0.0.0.0') if '0.0.0.0' in l]
            if route:
                parts = route[0].split()
                for p in parts:
                    if p.count('.') == 3 and not p.startswith('0.0.0.0') and p != '255.255.255.255':
                        return p
            return '127.0.0.1'
        except Exception:
            return '127.0.0.1'

    @property
    def _live_url(self):
        return f'http://{self._lan_ip}:{PREVIEW_PORT}/preview'

    def _on_toggle_live(self):
        self._web_live = bool(self._live_var.get())
        if self._web_live:
            preview_server.start(port=PREVIEW_PORT)  # idempotent, daemon
            self.lbl_live.configure(text=f'buka: {self._live_url}', foreground='blue')
            self.btn_openlive.configure(state='normal')
        else:
            self.lbl_live.configure(text='(mati)', foreground='gray')
            self.btn_openlive.configure(state='disabled')
        self._apply_heartbeat()  # live berubah -> perbarui syarat heartbeat

    def _open_live(self):
        webbrowser.open(self._live_url)

    # ---- tampilan hasil deteksi (on/off anotasi) ---------------------------

    def _on_toggle_overlay(self):
        """Nyalakan/matikan gambar kotak & label di preview. Membaca bool ini
        di worker aman (atomic via GIL), sama pola dengan _web_live."""
        self._show_overlay = bool(self._overlay_var.get())
        if self._show_overlay:
            self.lbl_overlay.configure(text='(aktif)', foreground='green')
        else:
            self.lbl_overlay.configure(text='(mati - kamera mentah)',
                                       foreground='gray')

    # ---- worker (thread background, TANPA akses widget) --------------------

    def _worker(self):
        try:
            if self.pipeline is None:
                self._set_status('Memuat model...')
                cfg = default_config()
                self.pipeline = DetectionPipeline(
                    cfg, source_label=f'gui:{self.source}')
                self._apply_web()  # terapkan setelan web yang mungkin sudah diisi

            self._set_status('Membuka kamera...')
            cap = self._open_source(self.source)
            if cap is None or not cap.isOpened():
                self._set_status('Gagal membuka sumber')
                return
            self.cap = cap

            fps_win = []
            self._set_status('Jalan')
            while not self._stop.is_set():
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                self._frame_idx += 1
                annotated, _ = self.pipeline.process_frame(
                    frame, self._frame_idx, draw_overlay=self._show_overlay)

                dt = time.time() - t0
                if dt > 0:
                    fps_win.append(1.0 / dt)
                    if len(fps_win) > 30:
                        fps_win.pop(0)
                with self._lock:
                    self._latest = annotated
                    self._fps = sum(fps_win) / len(fps_win) if fps_win else 0.0
                if self._web_live:
                    preview_server.set_latest(annotated)  # thread-safe
        except Exception as e:
            self._set_status(f'Error: {e}')
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._running = False
            if not self._status.startswith('Error') \
               and not self._status.startswith('Gagal'):
                self._set_status('Berhenti')

    def _open_source(self, source):
        """Buka webcam (int) atau path video. Mirror logika open_source() CLI."""
        try:
            cam_id = int(source)
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW if os.name == 'nt' else 0)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_id)
            return cap
        except (ValueError, TypeError):
            if not os.path.exists(str(source)):
                return None
            return cv2.VideoCapture(str(source))

    # ---- render (main thread) ----------------------------------------------

    def _refresh(self):
        with self._lock:
            frame = self._latest
            fps = self._fps
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if w > PREVIEW_MAX_W:
                scale = PREVIEW_MAX_W / w
                rgb = cv2.resize(rgb, (PREVIEW_MAX_W, int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            self._imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=self._imgtk)

        # Sync tombol + kontrol sumber dari _running (sumber kebenaran, main thread).
        if self._running != self._last_btn_running:
            self._last_btn_running = self._running
            self.btn_start.configure(state='disabled' if self._running else 'normal')
            self.btn_stop.configure(state='normal' if self._running else 'disabled')
            for wdg in self._src_widgets:
                # Combobox pakai 'readonly' saat aktif; Button pakai 'normal'.
                on_state = 'readonly' if isinstance(wdg, ttk.Combobox) else 'normal'
                wdg.configure(state='disabled' if self._running else on_state)

        self.lbl_status.configure(text=f'{self._status}   |   FPS: {fps:.1f}')

        # Indikator koneksi web (hijau/merah/abu) + enable toggle Kirim.
        render_key = (self._conn_status, self._connected)
        if render_key != self._last_conn_render:
            self._last_conn_render = render_key
            color = ('green' if self._connected
                     else 'red' if self._conn_status.startswith('Gagal')
                     else 'gray')
            self.lbl_conn.configure(text=f'● {self._conn_status}', foreground=color)
            self.chk_push.configure(state='normal' if self._connected else 'disabled')
            if not self._connected and self._push_var.get():
                # putus koneksi -> matikan toggle biar konsisten
                self._push_var.set(False)
                self._want_push = False
                self._apply_heartbeat()  # putus -> stop heartbeat

        # Indikator heartbeat: apakah web sudah tahu URL preview kita?
        # (hijau=terdaftar & segar, oranye=web tak terjangkau, abu=belum aktif)
        if self._heartbeat is not None:
            st = self._heartbeat.status()
        else:
            st = {'state': 'off', 'last_ok_ts': None, 'interval': 15}
        # 'segar' = sukses terakhir belum melewati ~3 interval (di bawah ambang
        # stale 60s web) -> lindungi dari thread yang macet tapi state masih lama.
        fresh = (st['last_ok_ts'] is not None
                 and (time.time() - st['last_ok_ts']) <= st['interval'] * 3)
        if st['state'] == 'registered' and fresh:
            hb_render = ('✓ Preview terdaftar di web', 'green')
        elif st['state'] == 'unreachable' or (st['state'] == 'registered' and not fresh):
            hb_render = ('⚠ Web tak terjangkau', 'orange')
        else:
            hb_render = ('(preview belum aktif)', 'gray')
        if hb_render != self._last_hb_render:
            self._last_hb_render = hb_render
            self.lbl_hb.configure(text=hb_render[0], foreground=hb_render[1])

        # Statistik pengiriman dari uploader (kalau ada).
        up = self.pipeline.uploader if self.pipeline is not None else None
        if up is not None:
            self.lbl_stats.configure(
                text=f'Terkirim: {up.sent}  Gagal: {up.failed}  Drop: {up.dropped}')

        self.root.after(30, self._refresh)

    def _set_status(self, text):
        # Dipanggil dari worker: cuma tulis string (atomic karena GIL); widget
        # di-update oleh _refresh di main thread.
        self._status = text

    # ---- shutdown -----------------------------------------------------------

    def on_close(self):
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.stop()
        if self.worker is not None:
            self.worker.join(timeout=2.0)
        if self.pipeline is not None:
            try:
                self.pipeline.release()
            except Exception as e:
                print(f'[GUI] pipeline.release error: {e}')
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry('1000x700')
    DetectorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
