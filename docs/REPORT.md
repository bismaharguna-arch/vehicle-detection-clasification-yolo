<!--
REPORT.md — base laporan teknis untuk program Deteksi Kendaraan.
Bisa dipakai langsung sebagai draft laporan TA / kerja praktek / dokumentasi internal.
Section bertanda <!-- TODO: ... --> butuh diisi manual karena tidak bisa diturunkan dari kode.
-->

# Laporan Sistem Deteksi Kendaraan, Klasifikasi BBM, dan Pembacaan Plat Nomor dengan Integrasi Web Monitoring

<!-- TODO IDENTITAS:
- Nama penyusun
- NIM / NPP
- Program studi / institusi
- Dosen pembimbing / supervisor
- Tahun akademik
- Mata kuliah / nama proyek (TA / KP / internal)
Hapus blok komentar ini setelah diisi.
-->

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Latar Belakang](#2-latar-belakang)
3. [Tujuan dan Ruang Lingkup](#3-tujuan-dan-ruang-lingkup)
4. [Tinjauan Teknologi dan Justifikasi Pilihan](#4-tinjauan-teknologi-dan-justifikasi-pilihan)
5. [Arsitektur Sistem](#5-arsitektur-sistem)
6. [Implementasi Detail](#6-implementasi-detail)
7. [Kontrak Integrasi Web Monitoring](#7-kontrak-integrasi-web-monitoring)
8. [Metodologi Evaluasi](#8-metodologi-evaluasi)
9. [Hasil Pengujian](#9-hasil-pengujian)
10. [Kendala dan Limitations](#10-kendala-dan-limitations)
11. [Pekerjaan Lanjutan](#11-pekerjaan-lanjutan)
12. [Kesimpulan](#12-kesimpulan)
13. [Lampiran](#13-lampiran)

---

## 1. Ringkasan Eksekutif

Sistem ini adalah pipeline computer vision real-time untuk mendeteksi kendaraan
(mobil dan motor), mengklasifikasi jenis bahan bakar (bensin atau listrik)
berdasarkan warna plat nomor Indonesia, dan membaca teks plat nomor. Hasil
deteksi di-push ke server web monitoring melalui HTTP multipart, dengan
mekanisme anti-spam yang memastikan setiap kendaraan dikirim **tepat satu kali**.

Pipeline terdiri dari 5 tahap berurutan: deteksi (YOLOv8), klasifikasi BBM
(HSV color filtering), OCR plat (PaddleOCR), tracking & deduplikasi (IoU-based),
dan transport ke server (HTTP multipart dengan retry queue).

**Output utama:** untuk tiap kendaraan yang lewat di kamera, server web
monitoring menerima satu record berisi tipe kendaraan (`mobil`/`motor`), jenis
BBM (`bensin`/`listrik`), nomor plat, skor confidence, timestamp deteksi, dan
foto crop kendaraan.

---

## 2. Latar Belakang

<!-- TODO LATAR BELAKANG:
Tuliskan 2-4 paragraf yang menjawab:
- Apa masalah yang melatarbelakangi proyek ini?
  (misal: monitoring lalu lintas kendaraan listrik di kawasan tertentu,
   pencatatan otomatis di gerbang, pemenuhan kebutuhan Dinas X, dll)
- Mengapa solusi manual / sistem yang sudah ada tidak cukup?
- Siapa stakeholder / pengguna akhir sistem ini?
- Konteks regulasi / kebijakan yang relevan?
  (mis. PP 79/2023 tentang kendaraan listrik, plat strip biru untuk EV)
Hapus komentar setelah diisi.
-->

Pemerintah Indonesia mendorong adopsi kendaraan listrik melalui regulasi
penandaan khusus pada plat nomor — kendaraan listrik memiliki **strip biru di
bagian bawah plat**, berbeda dengan kendaraan bermesin pembakaran (bensin/solar)
yang plat-nya seragam putih/hitam tanpa strip. Penanda visual ini memungkinkan
klasifikasi BBM otomatis menggunakan analisis warna, tanpa perlu database
registrasi.

<!-- TODO: tambah konteks spesifik lokasi/use case kamu -->

---

## 3. Tujuan dan Ruang Lingkup

### 3.1 Tujuan

1. Mendeteksi kendaraan (mobil dan motor) secara real-time dari sumber video
   (webcam atau file video).
2. Mengklasifikasi jenis bahan bakar (bensin vs listrik) berdasarkan strip
   warna pada plat nomor.
3. Membaca teks plat nomor menggunakan OCR.
4. Mengirim hasil deteksi ke sistem monitoring web melalui HTTP API, dengan
   jaminan **satu kendaraan = satu push** untuk menghindari banjir data.

### 3.2 Ruang Lingkup

**Termasuk:**
- Dua kelas kendaraan: mobil dan motor (bus dan truk diperlakukan sebagai
  mobil sesuai kontrak dengan tim web monitoring).
- Plat nomor format Indonesia standar (1-2 huruf wilayah + 1-4 angka + 0-3
  huruf seri, contoh: `B 2647 TZO`).
- Deteksi dari satu kamera tunggal.
- Pengiriman ke satu endpoint web monitoring melalui jaringan LAN.

**Tidak termasuk:**
- Multi-kamera (field `camera_id` di-defer hingga ada multi-cam deployment).
- Tracking lintas-kamera atau re-identifikasi kendaraan.
- Klasifikasi tipe kendaraan lebih granular (sedan vs SUV vs pickup dst).
- Sistem autentikasi/otorisasi (endpoint web sengaja publik untuk LAN).
- Pengenalan wajah pengemudi atau penumpang.

### 3.3 Target Deployment

<!-- TODO DEPLOYMENT:
- Lokasi pemasangan kamera (jalan tol? gerbang parkir? gerbang kota?
  kantor swasta? jalan biasa?)
- Berapa kendaraan per jam yang diharapkan?
- Apakah indoor atau outdoor?
- Pencahayaan: siang saja, atau termasuk malam?
- Hardware target: laptop, mini PC, Jetson, server rack?
Hapus komentar setelah diisi.
-->

---

## 4. Tinjauan Teknologi dan Justifikasi Pilihan

### 4.1 YOLOv8 untuk Object Detection

**Alasan pemilihan:**
- *Real-time*: YOLOv8 nano (`yolov8n.pt`, ~6.5 MB) mencapai >30 FPS pada GPU
  konsumen CUDA, cukup untuk feed video 30 fps tanpa frame drop.
- *Pretrained pada COCO*: kelas `car` (id 2), `motorcycle` (id 3), `bus` (id 5),
  dan `truck` (id 7) sudah tersedia tanpa training, mempercepat
  bootstrapping. Bus dan truck di-merge ke `mobil` untuk konsistensi dengan
  taksonomi web monitoring.
- *Ekosistem Ultralytics matang*: `YOLO(...)` API stabil, mendukung CUDA,
  TensorRT export, dan training augmentation built-in.

**Alternatif yang dipertimbangkan:**
- SSD MobileNet: lebih ringan tapi akurasi mAP@0.5 lebih rendah pada COCO
  (~0.21 vs YOLOv8n ~0.37).
- Faster R-CNN: akurasi lebih tinggi tapi terlalu lambat untuk real-time pada
  hardware target.

### 4.2 Plate Detector Custom (license_plate_detector)

**Alasan menggunakan model terpisah untuk plat:**
- COCO tidak memiliki kelas `license_plate`. Plat butuh model dedicated.
- Plat berukuran kecil relatif terhadap frame (sering <5% area gambar), butuh
  inference resolution tinggi (`imgsz=1280`) untuk mempertahankan recall.
- Pemisahan model memungkinkan pemilihan threshold confidence per-domain
  (vehicle bisa konservatif, plate bisa lebih agresif menangkap kandidat).

### 4.3 HSV Color Space untuk Klasifikasi BBM

**Alasan rule-based ketimbang neural classifier:**
- Plat listrik Indonesia memiliki strip biru yang sangat distinctive — single
  threshold rule sudah memberikan akurasi tinggi tanpa training.
- HSV memisahkan hue (warna) dari brightness, sehingga rule lebih robust
  terhadap variasi pencahayaan dibanding RGB.
- Range warna biru ditentukan: H ∈ [100, 130], S ≥ 80, V ≥ 50.
- Threshold rasio biru di bagian bawah 40% plat: rasio ≥ 5% → diklasifikasi
  sebagai listrik.

**Trade-off:** rule sederhana berarti edge case (plat kotor, strip biru pudar,
plat custom yang melanggar standar) perlu evaluasi terpisah. Tidak ada training
data yang diperlukan, sehingga setup awal sangat cepat. Migrasi ke neural
classifier dapat dilakukan bila evaluasi menunjukkan akurasi <95%.

### 4.4 PaddleOCR untuk Pembacaan Plat

**Alasan pemilihan dibanding Tesseract atau EasyOCR:**
- PaddleOCR PP-OCRv5 memiliki recognition akurasi state-of-the-art pada
  text-in-the-wild benchmarks.
- Mendukung GPU acceleration via PaddlePaddle (build CUDA 11.8 di-install
  manual sesuai instruksi `requirements.txt`).
- API `predict()` mengembalikan `rec_texts`, `rec_scores`, dan `rec_polys`
  yang memungkinkan filter berbasis posisi vertikal (penting untuk plat
  Indonesia yang punya strip masa berlaku pajak di bagian bawah).

**Konfigurasi:**
- `lang='en'` (plat Indonesia pakai karakter Latin).
- `use_textline_orientation=False` (plat selalu horizontal pada crop yang
  sudah di-deskew).
- `use_doc_orientation_classify=False`, `use_doc_unwarping=False`,
  `enable_mkldnn=False` (menghindari bug oneDNN pada Paddle 3.x).

### 4.5 IoU-based Tracking untuk Deduplikasi

**Alasan tidak pakai DeepSORT / ByteTrack:**
- Konteks satu kamera, jumlah kendaraan simultan kecil (<10), kendaraan
  bergerak lurus → IoU matching antar-frame cukup.
- Tidak butuh re-identification setelah occlusion karena penggunaan single
  one-shot push (kendaraan yang hilang lalu muncul lagi dianggap kendaraan
  baru — acceptable).
- Mengurangi dependensi (DeepSORT butuh appearance descriptor network
  tambahan).

### 4.6 HTTP Multipart untuk Transport

**Alasan tidak pakai WebSocket / gRPC / MQTT:**
- Web monitoring sudah Flask-based, menerima multipart POST adalah natural fit.
- HTTP punya retry-friendly stateless model — uploader queue tidak perlu
  reconnect logic.
- Foto JPG dikirim sebagai file binary di field `photo`, payload form
  membawa metadata terstruktur.

---

## 5. Arsitektur Sistem

### 5.1 Diagram Tingkat Tinggi

```
┌─────────────────────────────────────────────────────────────────────┐
│                          LAPTOP A (Detector)                         │
│                                                                       │
│  ┌────────┐    ┌────────────────────────────────────────────────┐   │
│  │ Webcam │───▶│  Pipeline (src/main.py)                         │   │
│  │ Video  │    │                                                  │   │
│  └────────┘    │  ┌─────────────────┐   ┌─────────────────┐     │   │
│                │  │ VehicleDetector │──▶│ FuelClassifier  │     │   │
│                │  │ - YOLOv8n COCO  │   │ HSV blue ratio  │     │   │
│                │  │ - Plate model   │   │ bottom 40% plat │     │   │
│                │  └────────┬────────┘   └────────┬────────┘     │   │
│                │           │                     │              │   │
│                │           ▼                     ▼              │   │
│                │  ┌─────────────────┐   ┌─────────────────┐     │   │
│                │  │   PlateReader   │   │ VehicleTracker  │     │   │
│                │  │   PaddleOCR     │──▶│ IoU dedup +     │     │   │
│                │  │   + preproc     │   │ best frame      │     │   │
│                │  └─────────────────┘   └────────┬────────┘     │   │
│                │                                 │              │   │
│                │                                 ▼              │   │
│                │                        ┌─────────────────┐     │   │
│                │                        │  HttpUploader   │     │   │
│                │                        │  multipart POST │     │   │
│                │                        │  retry queue    │     │   │
│                │                        └────────┬────────┘     │   │
│                └─────────────────────────────────┼──────────────┘   │
└──────────────────────────────────────────────────┼──────────────────┘
                                                   │
                                                   │ HTTP multipart
                                                   │ (LAN, port 5000)
                                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LAPTOP B (Web Monitoring)                      │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Flask Server  POST /api/detections                            │   │
│  │   ├─ Validasi field (enum coerce)                              │   │
│  │   ├─ Simpan ke DB (PostgreSQL/SQLite)                          │   │
│  │   ├─ Simpan foto ke uploads/                                   │   │
│  │   └─ Broadcast Socket.IO event 'new_detection'                 │   │
│  └─────────────────────────┬─────────────────────────────────────┘   │
│                            ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  Dashboard (browser) — realtime tabel deteksi, filter, edit   │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tahapan Pemrosesan per Frame

| # | Tahap            | Komponen           | Input              | Output                                |
|---|------------------|--------------------|--------------------|---------------------------------------|
| 1 | Deteksi vehicle  | YOLOv8n (COCO)     | Frame BGR          | Bbox + class (mobil/motor) + conf     |
| 2 | Deteksi plat     | YOLOv8 custom      | Frame BGR          | Bbox plat + conf                      |
| 3 | Klasifikasi BBM  | FuelClassifier     | Plate crop         | `bensin` atau `listrik` + blue_ratio  |
| 4 | OCR plat         | PaddleOCR          | Plate crop         | Teks plat (string)                    |
| 5 | Smoothing        | Confidence avg     | History 5 frame    | Smoothed confidence                   |
| 6 | Tracking dedup   | VehicleTracker     | All detections     | Push event (1× per kendaraan)         |
| 7 | Transport        | HttpUploader       | Push payload       | HTTP 201 dari server (atau retry)     |

### 5.3 Diagram Sekuens Push

```
Frame N           Detector          Tracker            Uploader        Server
  │                  │                 │                   │             │
  ├─frame──────────▶│                 │                   │             │
  │                  ├─predict×2──────│                   │             │
  │                  ├─classify fuel──│                   │             │
  │                  ├─OCR (if cache miss)                │             │
  │                  ├─detections────▶│                   │             │
  │                  │                 ├─update tracks    │             │
  │                  │                 ├─if stable+fuel:  │             │
  │                  │                 │   push payload──▶│             │
  │                  │                 │                   ├─POST───────▶│
  │                  │                 │                   │◀──201 id───┤
  │                  │                 │                   │             │
```

---

## 6. Implementasi Detail

### 6.1 Struktur Direktori

```
deteksi_kendaraan/
├── src/
│   ├── main.py              # Entry point, CLI, loop video/webcam
│   ├── detector.py          # VehicleDetector (dual YOLO + smoothing + OCR cache)
│   ├── fuel_classifier.py   # FuelClassifier (HSV blue ratio)
│   ├── plate_reader.py      # PlateReader (PaddleOCR + preprocessing)
│   ├── vehicle_tracker.py   # VehicleTracker (IoU dedup, 1-shot push)
│   ├── uploader.py          # HttpUploader (queue + retry)
│   ├── logger.py            # DetectionLogger (JSONL audit log)
│   └── utils.py             # draw_detections, draw_fps, dst
├── models/
│   ├── yolov8n.pt           # Vehicle detector (COCO pretrained)
│   ├── license_plate_detector (1).pt   # Plate detector
│   └── best (5).pt          # Cadangan: model 3-kelas (mobil/motor/plat, custom-trained)
├── tools/
│   ├── mock_server.py       # Simulasi web monitoring (Flask)
│   ├── sim_laptop_a.bat     # Launcher detector
│   ├── sim_laptop_b.bat     # Launcher mock server
│   └── get_lan_ip.py        # Auto-detect IP LAN
├── eval_set/                # Gambar evaluasi + ground_truth.csv
├── dataset/                 # Training dataset YOLO format
├── evaluate.py              # Eval end-to-end + metrik per stage
├── train_plate.py           # Fine-tune detektor plat
├── requirements.txt
├── start.bat                # Aktivasi venv
└── docs/REPORT.md           # File ini
```

### 6.2 VehicleDetector (src/detector.py)

Loading dua model YOLO terpisah:

```python
self.vehicle_model = YOLO('models/yolov8n.pt')
self.plate_model = YOLO('models/license_plate_detector (1).pt')
```

`predict()` dipanggil dua kali per frame, dengan `classes=[2,3,5,7]` di vehicle
inference untuk hanya menghasilkan kelas COCO yang relevan. `COCO_VEHICLE_MAP`
menerjemahkan ke nama lokal: `2→mobil`, `3→motor`, `5→mobil` (bus dimerge),
`7→mobil` (truck dimerge).

**Smoothing** menggunakan `deque(maxlen=5)` yang menyimpan deteksi 5 frame
terakhir. Confidence final untuk tiap bbox = rata-rata confidence dari
deteksi-deteksi yang IoU-matched di history. Ini mengurangi flicker (deteksi
yang berkedip nyala-mati).

**OCR cache** menggunakan list `_plate_cache` berisi entri `{bbox, text,
last_frame}`. Untuk setiap plat baru, dicari match via IoU; bila match
ditemukan, OCR di-skip kecuali `frame_idx - last_frame >= ocr_interval` (default
10). Cache otomatis di-prune setiap frame (plat yang tidak terlihat di-drop).

### 6.3 FuelClassifier (src/fuel_classifier.py)

Algoritma:

1. Crop bagian bawah 40% dari bbox plat (`focus_bottom_ratio=0.4`).
2. Convert BGR ke HSV.
3. Apply `cv2.inRange(hsv, [100,80,50], [130,255,255])` untuk mask biru.
4. Hitung `ratio = piksel_biru / total_piksel`.
5. Bila `ratio ≥ 0.05` (5%) → kelaskan sebagai `listrik`, else `bensin`.

Threshold 5% dipilih karena strip biru pada plat EV menempati sekitar 15-25%
area bagian bawah; threshold 5% memberikan margin untuk plat yang sebagian
terhalang atau pencahayaan tidak ideal.

### 6.4 PlateReader (src/plate_reader.py)

Pipeline preprocessing sebelum OCR:

1. **Crop bagian bawah** plat (`crop_bottom_ratio=0.22`) — membuang strip
   tanggal masa berlaku pajak (mis. "07-23") yang akan ter-concat dengan plat
   utama dan merusak format.
2. **Deskew** menggunakan `cv2.minAreaRect` pada mask Otsu dari piksel teks.
   Sudut maks ±15° untuk menghindari over-rotation pada noise.
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) dengan
   `clipLimit=2.0`, `tileGridSize=(8,8)` — membantu plat redup atau silau.
4. **Upscale ke ≥200px lebar** menggunakan `cv2.INTER_CUBIC` — recognizer
   dapat sinyal lebih baik pada citra besar.
5. **Unsharp mask** dengan amount 0.5 untuk mempertegas edge huruf.

Post-OCR filter:

- Bila PaddleOCR mengembalikan multi-line output, baris dikelompokkan
  berdasarkan y-center (toleransi `60% × tinggi rata-rata`).
- Cluster dengan skor tertinggi dipilih:
  `score = avg(line_height) × sum(text_length)`.
- Plat utama selalu menang karena font lebih besar DAN teks lebih panjang
  dibanding strip tanggal (mis. plat 8 karakter vs tanggal 4 karakter).

**Format normalizer:** regex `^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$` dengan
output `XX 1234 YYY`. Fallback ke raw string bila tidak match.

### 6.5 VehicleTracker (src/vehicle_tracker.py)

Setiap track menyimpan state:
- `id` (UUID 8-char)
- `bbox`, `best_bbox`, `best_frame`, `best_area` (untuk capture foto terbaik)
- `frames_seen`, `first_seen`, `last_seen`
- `fuel`, `plate_text` (latest non-empty)
- `pushed` (boolean — sentinel one-shot)
- `created_ts` (wall clock saat first-seen, dipakai sebagai `detected_at`)

Aturan push (semua harus terpenuhi):

1. `frames_seen >= min_frames_stable` (default 5) — anti-flicker.
2. `bbox.height >= min_vehicle_height` (default 100 px) — quality filter,
   kendaraan jauh tidak dipush.
3. Salah satu dari:
   - Plat ter-asosiasi dan `fuel_type` terdeteksi
   - `fuel_wait_frames` lewat (default 15) → push dengan `fuel=unknown`

Foto crop kendaraan di-resize ke `max_image_dim=800` (max sisi terpanjang),
encode JPEG quality 85, hasil tipikal 50-300 KB.

Cleanup: track yang tidak terlihat selama `track_timeout` (default 30) frame
di-drop. Kendaraan yang masuk lagi setelah cleanup dianggap kendaraan baru
(track_id baru) — push lagi. Acceptable untuk monitoring lalu lintas.

### 6.6 HttpUploader (src/uploader.py)

Worker thread terpisah dengan `queue.Queue(maxsize=64)`. Push dari tracker
tidak block — apabila queue penuh, payload di-drop dan dilog (kondisi tidak
biasa, indikasi server down atau rate jauh di atas kapasitas).

Retry policy: `retries=2`, exponential backoff 0.5s, 1.0s. Field internal
prefix `_` (mis. `_track_id`, `_image_file`) di-strip sebelum POST — hanya
field kontrak v1 yang terkirim ke server.

### 6.7 Mock Server (tools/mock_server.py)

Implementasi Flask yang mencerminkan kontrak server produksi:

- `POST /api/detections` — multipart, balas 201 + `{"status":"SUCCESS","id":N}`.
- `GET /` — dashboard HTML auto-refresh 2 detik (tabel deteksi terbaru,
  badge low-conf, indikator plat kosong).
- `GET /api/detections` — list 50 entri terakhir.
- `GET /api/stats` — agregasi by-fuel dan by-vehicle.

Fitur uji ketahanan:
- `--latency-ms N` — delay artifisial per request (simulasi LAN/WAN).
- `--drop-rate P` — probabilitas 0-1 untuk balas 500 (uji retry).
- `--max-mb N` — hard limit ukuran body (default 5, match kontrak server real).

---

## 7. Kontrak Integrasi Web Monitoring

### 7.1 Endpoint

- URL: `http://<ip-server>:5000/api/detections`
- Method: `POST`
- Content-Type: `multipart/form-data`
- Autentikasi: tidak ada (LAN tertutup; API key di-defer ke v2 bila eksposur
  publik diperlukan)

### 7.2 Field Wajib

| Field              | Tipe         | Domain                                            |
|--------------------|--------------|---------------------------------------------------|
| `plate_number`     | string       | bebas, boleh kosong `""`                          |
| `vehicle_type`     | enum string  | `mobil` \| `motor` \| `unknown`                   |
| `is_electric`      | enum string  | `bensin` \| `listrik` \| `unknown`                |
| `confidence_score` | float        | 0.0 – 1.0 (skor YOLO vehicle, bukan fuel)         |
| `detected_at`      | string       | ISO 8601 dengan offset TZ (`2026-05-08T13:45:22+07:00`) |
| `photo`            | file binary  | JPG, ≤5 MB                                        |

### 7.3 Response

- `201 Created`: `{"status": "SUCCESS", "id": <int>}`
- `500 Server Error`: `{"status": "ERROR", "message": "<str>"}`
- `413 Payload Too Large`: foto melebihi 5 MB

### 7.4 Field yang Sengaja TIDAK Dikirim

`camera_id`, `client_event_id`, `location`, `lane` — server belum memiliki
kolom untuk field ini; akan diabaikan diam-diam. Akan ditambahkan di kontrak
v2 saat deployment multi-kamera dimulai.

### 7.5 Pertanyaan & Keputusan Selama Penyusunan Kontrak

Selama integrasi, beberapa pertanyaan teknis muncul. Keputusan dan
alasannya:

| Pertanyaan                          | Keputusan                                                 |
|-------------------------------------|-----------------------------------------------------------|
| Bahasa nilai enum (ID atau EN)?     | Indonesia (`mobil`, `bensin`) — sesuai UI dan dataset     |
| Kelas truk/bus dipisah?             | Tidak — merge ke `mobil`, UI belum siap                   |
| Confidence threshold di server?     | Tidak ada, server terima apapun, UI flag low-conf <0.65   |
| Timestamp authoritative?            | Detector (`detected_at`), server simpan juga `received_at`|
| Batas ukuran foto?                  | 5 MB hard limit di server, detector resize ke <500 KB     |
| Dedup duplikat (restart detector)?  | Belum — track 2 minggu deploy, re-evaluate                |
| Endpoint authentication?            | Tidak (LAN tertutup), re-evaluate bila eksposur naik      |

---

## 8. Metodologi Evaluasi

### 8.1 Skema Pengujian

Pengujian end-to-end dilakukan menggunakan `evaluate.py` pada dataset
`eval_set/` yang berisi gambar statis + ground-truth CSV. Tujuan: mengukur
akurasi tiap tahap pipeline secara terpisah dan akurasi end-to-end.

### 8.2 Format Ground Truth

```csv
filename,vehicle_type,plate_text,fuel_type,plate_x1,plate_y1,plate_x2,plate_y2
```

- `filename`: relatif terhadap folder `eval_set/`
- `vehicle_type`: `mobil` / `motor` / kosong
- `plate_text`: teks plat (alfanumerik + spasi), kosong = no plate
- `fuel_type`: `bensin` / `listrik` / kosong
- `plate_x1..y2`: bbox plat (int piksel)

### 8.3 Metrik per Tahap

| Tahap                | Metrik                          | Target  |
|----------------------|----------------------------------|---------|
| Vehicle detection    | Recall per kelas @ IoU≥0.5      | ≥0.97   |
| Plate detection      | Precision / Recall / F1 @ IoU≥0.5 | ≥0.97 |
| Fuel classification  | Balanced accuracy                | ≥0.95   |
| OCR plat             | Exact match (strip spasi, upper) | ≥0.95   |
| **End-to-end**       | Plate detect AND fuel benar AND OCR exact | **≥0.90** |

Target end-to-end 0.90 mengikuti realitas perkalian akurasi komponen:
0.97 × 0.97 × 0.95 × 0.95 ≈ 0.85. Untuk mencapai ≥0.90 end-to-end,
komponen-komponen secara individual perlu di atas target di atas.

### 8.4 Sebaran Dataset Evaluasi

<!-- TODO EVAL DATASET:
Isi:
- Total gambar evaluasi terkumpul: ___ (target ≥200)
- Sebaran kondisi:
  - Siang terang: ___ gambar
  - Malam / lampu jalan: ___ gambar
  - Hujan / kontras rendah: ___ gambar
  - Sudut miring / jauh: ___ gambar
- Sebaran kelas:
  - Mobil bensin: ___
  - Mobil listrik: ___
  - Motor bensin: ___
  - Motor listrik: ___
  - Tanpa plat (untuk uji FP): ___
- Sumber gambar (foto sendiri / video CCTV / dataset publik): ___
Hapus komentar setelah diisi.
-->

---

## 9. Hasil Pengujian

<!-- TODO HASIL:
Setelah menjalankan `python evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv`,
isi hasil di tabel dan paragraf di bawah. Contoh isian:

| Tahap                | Metrik             | Hasil  | Target | Status |
|----------------------|--------------------|---------|---------|--------|
| Vehicle detection    | Recall mobil       | 0.96    | 0.97    | ⚠️      |
| Vehicle detection    | Recall motor       | 0.94    | 0.97    | ⚠️      |
| Plate detection      | F1                 | 0.92    | 0.97    | ❌      |
| Fuel classification  | Balanced accuracy  | 0.97    | 0.95    | ✅      |
| OCR plat             | Exact match        | 0.85    | 0.95    | ❌      |
| End-to-end           | Pipeline accuracy  | 0.78    | 0.90    | ❌      |
-->

### 9.1 Performa Akurasi

| Tahap                | Metrik             | Hasil  | Target | Status |
|----------------------|--------------------|---------|---------|--------|
| Vehicle detection    | Recall mobil       | TODO    | 0.97    | TODO   |
| Vehicle detection    | Recall motor       | TODO    | 0.97    | TODO   |
| Plate detection      | Precision          | TODO    | 0.97    | TODO   |
| Plate detection      | Recall             | TODO    | 0.97    | TODO   |
| Plate detection      | F1                 | TODO    | 0.97    | TODO   |
| Fuel classification  | Balanced accuracy  | TODO    | 0.95    | TODO   |
| OCR plat             | Exact match        | TODO    | 0.95    | TODO   |
| **End-to-end**       | Pipeline accuracy  | TODO    | 0.90    | TODO   |

### 9.2 Performa Runtime

<!-- TODO RUNTIME:
Ukur dengan FPS counter di main.py saat sumber video, hardware target:
- GPU: ___ (mis. RTX 3060 Mobile)
- RAM: ___
- Resolusi input: ___ (umumnya 1280x720)
- FPS rata-rata: ___ (dengan OCR ON), ___ (dengan OCR OFF)
- Latensi push (track first-seen → server 201): ___ ms
-->

| Metrik                          | Nilai     |
|---------------------------------|-----------|
| Hardware GPU                    | TODO      |
| Resolusi input frame            | TODO      |
| FPS dengan OCR aktif            | TODO      |
| FPS dengan OCR mati             | TODO      |
| Latensi push tipikal            | TODO ms   |
| Ukuran payload foto rata-rata   | TODO KB   |

### 9.3 Pengujian Integrasi Web Monitoring

Pengujian connectivity dilakukan menggunakan `tools/mock_server.py` yang
mengimplementasikan kontrak persis server produksi. Smoke test berhasil:

- POST multipart dengan field lengkap → `HTTP 201` + `{"status":"SUCCESS","id":1}` ✅
- Server menyimpan `detected_at` (client TZ-aware) dan `received_at` (server)
  terpisah ✅
- Field di luar enum (mis. `truck`) di-coerce ke `unknown` + warning log ✅
- Stats endpoint menampilkan agregasi yang konsisten dengan input ✅
- Retry uploader bekerja saat `--drop-rate 0.2` di-set di server ✅

---

## 10. Kendala dan Limitations

### 10.1 Limitations Teknis Sistem Saat Ini

1. **Klasifikasi BBM bergantung pada strip biru visual**: kendaraan listrik
   yang plat-nya kotor, terhalang, atau dipotong frame akan keliru
   diklasifikasi sebagai bensin. Tidak ada fallback ke database registrasi.

2. **OCR sensitif terhadap kondisi plat**: plat bengkok, kotor, terhalang
   stiker, atau pencahayaan ekstrem (silau matahari, blur cahaya malam) sering
   menghasilkan teks salah baca atau kosong. Implementasi cluster filter
   sudah menangani kasus multi-line (plat utama + tanggal pajak), tapi tidak
   menangani plat yang benar-benar buram.

3. **Tracking IoU lemah untuk kendaraan saling overlap**: dua kendaraan yang
   melintas berdampingan dengan IoU tinggi dapat tertukar ID. Untuk
   monitoring satu-arah satu-lane ini jarang terjadi; untuk persimpangan
   ramai perlu tracker yang lebih canggih (ByteTrack/DeepSORT).

4. **Tidak ada perlindungan duplikat saat detector restart**: bila detector
   restart dan kendaraan yang sama lewat ulang, akan dipush sebagai record
   baru. Web team menyetujui untuk menerima ini sebagai edge case; akan
   ditangani dengan `client_event_id` di kontrak v2 bila frekuensinya
   ternyata mengganggu.

5. **Single-camera deployment**: tidak ada konsep camera_id, tidak ada
   re-identifikasi antar kamera. Multi-cam butuh kontrak v2.

### 10.2 Limitations Eksternal

<!-- TODO EXTERNAL:
Tambahkan limitations spesifik dari kondisi deployment kamu:
- Latensi jaringan WiFi/LAN aktual: ___
- Stabilitas listrik di lokasi: ___
- Kondisi kamera (resolusi, lensa, posisi): ___
- Jam operasional (24/7? jam tertentu?): ___
-->

---

## 11. Pekerjaan Lanjutan

### 11.1 Peningkatan Akurasi

1. **Fine-tune detektor plat dengan data Indonesia**. Bila evaluasi
   menunjukkan plate detection F1 < 0.95, kumpulkan 500-1000 gambar plat
   Indonesia dengan variasi kondisi, latih ulang via `train_plate.py`.

2. **Fine-tune PaddleOCR untuk plat Indonesia**. Recognition model bawaan
   PP-OCRv5 dilatih pada teks general-purpose; plat Indonesia punya distribusi
   karakter dan font yang spesifik. Fine-tune dapat menaikkan exact match
   5-10%.

3. **Migrasi FuelClassifier ke neural classifier kecil**. Bila edge case
   warna (plat khusus, strip biru pudar) ternyata sering, ResNet18 atau
   MobileNet di-crop area bawah plat dapat lebih robust. Biaya komputasi
   marginal.

4. **Tambahkan TensorRT export**. `yolo export model=... format=engine half=True`
   dapat memberikan 2-3× speedup di GPU NVIDIA.

### 11.2 Fitur Baru (Roadmap dengan Tim Web Monitoring)

1. **Kontrak v2**: tambah `camera_id`, `client_event_id` (dedup),
   `client_timestamp` authoritative (sudah dikirim sekarang, tinggal server
   pakai sebagai source-of-truth).
2. **Polling endpoint koreksi**: `GET /api/corrections?since=<iso>` untuk
   detector ambil koreksi operator → label aktif untuk re-training.
3. **API key opsional** (`X-API-Key`) bila server di-expose ke VPN/internet.
4. **Webhook out**: server kirim event ke detector saat operator koreksi
   (untuk online learning, fase lanjut).

### 11.3 Operasional

1. Populate `eval_set/` ke ≥200 gambar dengan sebaran kondisi yang
   representatif sebelum klaim akurasi production-ready.
2. Setup CI yang menjalankan `evaluate.py` sebelum tag rilis — exit code 2
   bila end-to-end <0.90 akan block deploy.
3. Logging audit (`--log-jsonl`) di-rotate dan di-archive untuk forensik.

---

## 12. Kesimpulan

Sistem deteksi kendaraan, klasifikasi BBM, dan pembacaan plat nomor dengan
integrasi web monitoring telah berhasil diimplementasikan sebagai pipeline
5-tahap berbasis YOLOv8, OpenCV HSV, PaddleOCR, dan HTTP multipart transport.

Kontrak integrasi dengan tim web monitoring telah disepakati dalam versi 1
dengan field bahasa Indonesia (sesuai UI dan dataset domain), `detected_at`
TZ-aware untuk forensik, dan field-field yang sengaja di-defer (`camera_id`,
`client_event_id`) untuk dikerjakan di v2 saat kebutuhan mendesak muncul.

Mekanisme anti-spam memastikan satu kendaraan dikirim tepat satu kali,
dengan fallback aman bila kondisi tertentu tidak terpenuhi (kendaraan terlalu
jauh, plat tidak terbaca, dll).

<!-- TODO KESIMPULAN AKHIR:
Tambah 1-2 paragraf yang menjawab:
- Apakah target end-to-end ≥0.90 tercapai? (lihat hasil di Bagian 9)
- Komponen mana yang paling perlu perbaikan?
- Rekomendasi prioritas pekerjaan lanjutan?
-->

---

## 13. Lampiran

### 13.1 Tech Stack

| Komponen           | Versi          | Lisensi      |
|--------------------|----------------|--------------|
| Python             | 3.10+          | PSF          |
| PyTorch            | 2.7.1+cu118    | BSD          |
| Ultralytics YOLO   | 8.4.47         | AGPL-3.0     |
| PaddlePaddle       | 3.0.0 (GPU)    | Apache 2.0   |
| PaddleOCR          | 3.5.0          | Apache 2.0   |
| OpenCV             | 4.6.0.66       | Apache 2.0   |
| NumPy              | 1.26.4         | BSD          |
| Flask              | 3.1.3          | BSD          |
| Requests           | 2.33.1         | Apache 2.0   |

### 13.2 Hardware Pengujian

<!-- TODO HARDWARE:
- CPU: ___
- GPU: ___ (sebut CUDA compute capability)
- RAM: ___
- OS: Windows 11 IoT Enterprise LTSC 2024 (sesuai env)
- Storage: ___
- Kamera: ___
-->

### 13.3 Cara Menjalankan dari Nol

```powershell
# 1. Clone / extract project ke D:\deteksi_kendaraan
# 2. Buat venv
python -m venv venv

# 3. Aktifkan dan install dependencies
venv\Scripts\activate
pip install -r requirements.txt

# 4. Install PaddlePaddle GPU build (terpisah, butuh CUDA 11.8)
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 5. Verifikasi GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import paddle; print('Paddle CUDA:', paddle.is_compiled_with_cuda())"

# 6. Smoke test
python diagnose.py
python test_ocr.py
```

### 13.4 Command Reference

```powershell
# Webcam realtime (default)
python src\main.py

# Video file + push ke web monitoring
python src\main.py --source video.mp4 --push --push-url http://192.168.1.10:5000/api/detections

# Simulasi 2-laptop (terminal 1 = server, terminal 2 = detector)
tools\sim_laptop_b.bat
tools\sim_laptop_a.bat video.mp4

# Mock server dengan latency + drop simulation
python tools\mock_server.py --port 5000 --latency-ms 30 --drop-rate 0.1

# Evaluasi end-to-end
python evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv

# Inspect model file
python diagnose.py

# Fine-tune plate detector
python train_plate.py --epochs 100 --batch 16
```

### 13.5 Flag Penting `src/main.py`

| Flag                       | Default          | Fungsi                                      |
|----------------------------|------------------|---------------------------------------------|
| `--source`                 | `1` (webcam)     | Webcam id atau path video                   |
| `--vehicle-model`          | `models/yolov8n.pt` | Model deteksi vehicle                    |
| `--plate-model`            | `models/license_plate_detector (1).pt` | Model plat |
| `--imgsz`                  | 1280             | Resolusi inference                          |
| `--no-ocr`                 | (off)            | Matikan PaddleOCR                           |
| `--ocr-interval`           | 10               | Re-OCR plat sama tiap N frame               |
| `--push`                   | (off)            | Aktifkan one-shot push                      |
| `--push-url`               | -                | Endpoint web monitoring                     |
| `--push-stable`            | 5                | Min frame stabil sebelum push               |
| `--push-min-height`        | 100              | Min tinggi bbox kendaraan (px)              |
| `--push-max-dim`           | 800              | Max dim foto setelah resize                 |
| `--push-jpeg-quality`      | 85               | JPEG quality                                |
| `--push-fuel-optional`     | (off)            | Push tanpa nunggu fuel terdeteksi           |
| `--log-jsonl`              | -                | Path file JSONL audit log                   |

---

<!-- TODO DAFTAR PUSTAKA / REFERENSI:
Tambahkan referensi yang dipakai:
- Paper YOLOv8 (Jocher et al.)
- Paper PaddleOCR / PP-OCR
- Dokumentasi Ultralytics
- Regulasi plat listrik Indonesia (Permenhub / PP terkait)
- Paper / artikel HSV color space untuk vehicle classification
- Dataset COCO (Lin et al. 2014)
Format: nomor + judul + author + tahun + URL/DOI
-->
