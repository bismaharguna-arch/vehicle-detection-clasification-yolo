# Penjelasan Lengkap Program — Deteksi Kendaraan, Klasifikasi Bahan Bakar & Pembacaan Plat

> Dokumen persiapan sidang. Disusun dari pembacaan langsung source code
> (bukan dari dokumentasi lama), Juli 2026.
>
> Semua rujukan baris kode mengacu ke kondisi repo saat dokumen ini ditulis.

---

## Daftar Isi

0. [Koreksi: dokumen vs kode nyata](#0-koreksi-dokumen-vs-kode-nyata)
1. [Apa yang sistem ini kerjakan](#1-apa-yang-sistem-ini-kerjakan)
2. [Peta file](#2-peta-file)
3. [Alur satu frame](#3-alur-satu-frame)
   - [Tahap 1 — Deteksi](#tahap-1--deteksi-srcdetectorpy)
   - [Tahap 2 — Klasifikasi bahan bakar](#tahap-2--klasifikasi-bahan-bakar-srcfuel_classifierpy)
   - [Tahap 3 — OCR plat](#tahap-3--ocr-plat-srcplate_readerpy)
   - [Tahap 4 — Tracking & anti-duplikat](#tahap-4--tracking--anti-duplikat-srcvehicle_trackerpy--inti-skripsi)
   - [Tahap 5 — Pengiriman](#tahap-5--pengiriman-srcuploaderpy)
4. [Komponen pendukung](#4-komponen-pendukung)
5. [Tabel parameter](#5-tabel-parameter-untuk-dihafal)
6. [Evaluasi](#6-evaluasi-evaluatepy)
7. [Keterbatasan](#7-keterbatasan--sampaikan-duluan)
8. [Bank pertanyaan penguji](#8-bank-pertanyaan-penguji)
9. [Prioritas sebelum sidang](#9-prioritas-sebelum-sidang)

---

## 0. Koreksi: dokumen vs kode nyata

Empat hal di dokumentasi/skill proyek **tidak cocok dengan kode**. Jangan
menjelaskan hal-hal di kolom kiri saat sidang.

| Klaim di dokumen/skill | Kenyataan di kode |
|---|---|
| `CLAUDE.md`: "bus dan truck di-merge jadi `mobil`" | **SALAH.** [`src/detector.py:21-24`](../src/detector.py#L21-L24) hanya memetakan COCO `car`→`mobil` dan `motorcycle`→`motor`. Komentar kodenya eksplisit: *"Bus & truck sengaja TIDAK dideteksi."* |
| `CLAUDE.md`: "OCR preprocessing: deskew → CLAHE → upscale → unsharp, lalu crop bawah 22%" | **Semua default OFF.** [`src/plate_reader.py:54-57`](../src/plate_reader.py#L54-L57): `preprocess=False`, `crop_bottom_ratio=0.0`, `upscale_width=0`. Ablasi membuktikan crop 22% **memotong huruf plat** (2/8 benar → 8/8 setelah dimatikan). Strip pajak dibuang oleh **filter tinggi teks** (`min_height_ratio=0.6`). |
| Skill `cv-pipeline` menyebut ByteTrack, `config.yaml`, `plate.pt`, `pytest tests/` | **Tidak satupun ada di proyek ini.** Tracking pakai **Hungarian** buatan sendiri, konfigurasi lewat **argparse**, tidak ada folder `tests/`. **Jangan pernah sebut ByteTrack di sidang.** |
| `evaluate.py` siap dipakai | **RUSAK.** [`evaluate.py:179`](../evaluate.py#L179) memanggil `FuelClassifier(blue_ratio_threshold=...)` padahal parameter itu sudah diganti jadi `strip_coverage_threshold`. Terverifikasi: `TypeError: unexpected keyword argument 'blue_ratio_threshold'`. Skrip evaluasi **crash sebelum memproses 1 gambar pun**. |

Yang terakhir paling genting — kalau penguji minta demo angka akurasi,
programnya mati. Perbaikannya satu baris.

---

## 1. Apa yang sistem ini kerjakan

Sistem mendeteksi kendaraan dari video/webcam, lalu untuk setiap kendaraan yang
lewat mengirimkan **satu baris data** ke server web monitoring:

> "Mobil, plat B 1234 XYZ, bahan bakar konvensional, confidence 0.87,
> terdeteksi 14:32:07+07:00, + foto"

Masalah inti yang diselesaikan **bukan** deteksi objek (YOLO sudah selesai untuk
itu). Masalah intinya:

1. **1 kendaraan = 1 baris.** Kamera melihat mobil yang sama di 60 frame
   berturut-turut. Implementasi naif = 60 baris duplikat.
2. **Data tidak boleh nyasar.** Foto mobil A tidak boleh mendarat di baris mobil
   B. Kegagalan ini **senyap** — tidak ada error, datanya cuma salah.
3. **OCR membaik seiring waktu.** Plat terbaca `D 1 S` saat jauh, `D 15 NW` saat
   dekat. Dikirim terlalu cepat → salah. Terlalu lambat → kendaraan sudah lewat.

Sekitar **70% kode** ([`src/vehicle_tracker.py`](../src/vehicle_tracker.py),
1068 baris) khusus menangani tiga masalah ini. **Ini kontribusi teknis skripsi**,
bukan pemakaian YOLO-nya.

---

## 2. Peta file

```
src/
├── main.py            (382) CLI: argparse, buka kamera, loop, kontrol keyboard, FPS
├── gui.py             (472) Desktop Tkinter (shell tipis di atas pipeline)
├── pipeline.py        (203) ★ Merakit 5 tahap; process_frame() = 1 frame
├── detector.py        (420) Tahap 1: YOLOv8 (2 model) + smoothing + scheduler OCR
├── fuel_classifier.py (140) Tahap 2: HSV strip biru → listrik/konvensional
├── plate_reader.py    (349) Tahap 3: PaddleOCR + normalisasi plat Indonesia
├── vehicle_tracker.py (1068) ★★ Tahap 4: Hungarian tracking + anti-duplikat + koreksi
├── uploader.py        (237) Tahap 5: POST/PATCH ke web (thread terpisah)
├── preview_server.py   (89) Flask MJPEG live stream
├── registrar.py       (174) Heartbeat auto-registrasi ke web
├── logger.py           (57) JSONL audit log
└── utils.py           (140) Gambar bbox/label/FPS
```

**Kunci arsitektur:** [`DetectionPipeline`](../src/pipeline.py) adalah
satu-satunya tempat logika per-frame. CLI dan GUI dua-duanya memanggil
`process_frame()` yang sama. Kalau ditanya *"kenapa tidak ada duplikasi kode
antara CLI dan GUI?"* — ini jawabannya.

---

## 3. Alur satu frame

```
frame ─► VehicleDetector ─► FuelClassifier ─► PlateReader ─► VehicleTracker ─► HttpUploader
        (2× YOLO)          (HSV strip biru)   (PaddleOCR)    (Hungarian)      (POST/PATCH)
                                                              │
                              preview MJPEG ◄─────────────────┴──► JSONL log
```

Dua *side sink* (preview + log) tidak mempengaruhi jalur pengiriman.

---

### Tahap 1 — Deteksi ([`src/detector.py`](../src/detector.py))

#### Dua model, bukan satu

| Model | Tugas | Resolusi | Alasan |
|---|---|---|---|
| `yolov8n.pt` (COCO pretrained) | mobil, motor | **640 px** | Kendaraan = objek besar; 640 itu resolusi native yolov8n |
| `plate_best (1).pt` (custom) | plat nomor | **960 px** | Plat = objek kecil, butuh resolusi tinggi |

Ini keputusan optimasi yang penting untuk sidang. Dulu keduanya jalan di 960.
Menurunkan **hanya model kendaraan** ke 640 memotong beban ~setengah dari salah
satu dari dua `predict()` per frame, **tanpa menyentuh akurasi plat sama sekali**.
Menurunkan `--imgsz` (plat) justru merusak — plat objek kecil.

Ada juga mode **unified** (1 model 3-kelas, 1× predict/frame) di
[`src/detector.py:192`](../src/detector.py#L192), aktif lewat `--model <bobot>`.
Sudah diimplementasi penuh tapi **off by default** karena bobot 3-kelasnya belum
dilatih. Kalau ditanya *"kenapa 2× inference, tidak bisa 1×?"* — jawab: bisa,
jalurnya sudah ada, tinggal butuh dataset gabungan.

#### Trik dua ambang confidence

```python
predict(conf=0.25)          # ambang RENDAH saat inference
...
if d['confidence'] >= 0.4:  # ambang FINAL setelah smoothing
```

YOLO diminta mengembalikan deteksi lemah (0.25), tapi filter final di 0.4 baru
dijalankan **setelah smoothing**. Efeknya: kendaraan yang confidence-nya
sekali-sekali anjlok ke 0.35 (tertutup sesaat, motion blur) tetap selamat karena
rata-rata 5 frame-nya masih di atas 0.4. Tanpa ini, deteksi berkedip → track
pecah → baris duplikat.

#### Smoothing temporal ([`src/detector.py:404`](../src/detector.py#L404))

Menyimpan 5 frame terakhir. Tiap deteksi dicocokkan IoU > 0.3 ke frame-frame
lalu; confidence dirata-rata, bbox dirata-rata **berbobot** (frame sekarang
bobot 2.0, frame lalu 1.0). Hasilnya kotak bergerak halus, bukan patah-patah.

> ⚠️ **Fuel dan OCR dihitung dari bbox MENTAH, sebelum smoothing** — karena crop
> untuk analisis warna/teks harus persis di posisi plat frame ini, bukan
> rata-rata posisi.

#### Optimasi lain yang layak disebut

- **FP16 otomatis** di CUDA (inference ~1.5–2× lebih cepat, akurasi praktis
  sama), otomatis FP32 di CPU.
- **`_extract_boxes()`** ([`src/detector.py:307`](../src/detector.py#L307)):
  menarik `cls/conf/xyxy` sekali sebagai numpy, bukan per-box. Versi lama =
  puluhan sinkronisasi GPU→CPU kecil per frame.
- **`cudnn.benchmark=True` sengaja TIDAK dipakai** — sudah diuji, gain ~nol untuk
  yolov8n tapi mengorbankan output byte-identik antar run. Contoh bagus
  "optimasi yang ditolak dengan data".

---

### Tahap 2 — Klasifikasi bahan bakar ([`src/fuel_classifier.py`](../src/fuel_classifier.py))

**Dasar:** plat kendaraan listrik Indonesia punya **strip biru horizontal di
bagian bawah**. Tidak perlu database registrasi — cukup analisis warna.

#### Algoritma

```
1. Ambil crop plat, PERPANJANG 50% ke bawah
   → detektor mengotaki baris nomor saja; strip EV ada di BAWAH kotak itu
2. BGR → HSV, inRange(H: 95-135, S: ≥35, V: ≥60) → mask biru
3. Fokus ke 40% bagian bawah crop
4. strip_score = fraksi LEBAR terbesar di antara semua baris
```

**Kenapa "baris paling biru" dan bukan "rasio total piksel biru"?**

Ini inti kontribusinya. Versi awal memakai rasio total piksel biru — false
positive ~55%: band biru tepi-kiri plat gaya Eropa, noise huruf gelap, semuanya
ikut kehitung. Strip EV Indonesia itu **garis horizontal selebar plat**, jadi
yang benar diukur adalah *"adakah satu baris yang birunya membentang hampir
selebar plat"*. Band vertikal tepi kiri → tiap baris cuma biru di sebagian kecil
lebar → skor rendah → ditolak.

#### Tiga penjaga berlapis

| Penjaga | Aturan | Menangkal |
|---|---|---|
| **Coverage** | `strip_score ≥ 0.85` | Noise, band tepi |
| **Anti-cast** | median-S bawah ≥ 1.1 × median-S atas | Kamera white-balance biru bikin *seluruh* plat kehitung biru |
| **Biru-asli** | median-S ≥ 75 **dan** median-V ≥ 95 | Plat gelap ternaungi & plat silau ber-cast |

**Angka-angka ini dikalibrasi dari data, bukan dikarang** — 84 capture berlabel
(11 EV asli vs 73 false positive). EV asli: S 71–233 (median 148), V 71–209
(median 181). FP: median S 60. Hasil: **10/11 EV selamat, 17/20 FP tertolak.**

> ⚠️ **Ambang 0.85 adalah batas atap.** EV asli berlabel punya `strip_score`
> minimum 0.89 — margin hanya 0.04. Di 0.90 EV asli mulai hilang. **Jangan
> naikkan.** Kalau di lapangan EV miring/terpotong terbaca konvensional,
> **turunkan** ke ~0.6 lewat `--fuel-coverage`.

#### Kejujuran yang harus disampaikan

Guard V ≥ 95 **ikut menolak strip EV di malam gelap** (S~71, V~71). Ini trade-off
sadar: sistem memilih *melewatkan* EV di kondisi gelap daripada *salah menandai*
puluhan mobil konvensional sebagai listrik. **Sebut ini duluan sebelum penguji
menemukannya.**

Ada **self-test** di
[`src/fuel_classifier.py:142-156`](../src/fuel_classifier.py#L142-L156) — 4 kasus
sintetis (strip EV, band tepi, cast merata, strip gelap):

```powershell
venv\Scripts\python.exe src\fuel_classifier.py
```

---

### Tahap 3 — OCR plat ([`src/plate_reader.py`](../src/plate_reader.py))

#### OCR berjalan ASYNC

[`src/detector.py:107-117`](../src/detector.py#L107-L117): PaddleOCR **tidak
pernah** dipanggil dari loop deteksi. Loop hanya *menjadwalkan* job ke
`Queue(maxsize=4)`; satu daemon thread bernama `ocr-worker` yang mengerjakannya.

**Kenapa?** Sebelumnya OCR sinkron tiap 10 frame → ada *spike* FPS periodik (loop
berhenti menunggu PaddleOCR). Sekarang loop tidak pernah menunggu. Hasil OCR
mendarat di cache telat beberapa frame — dan itu tidak masalah karena tracker
punya *settle gate* yang memang menunggu beberapa frame.

> Aturan yang tidak boleh dilanggar: **jangan pernah panggil `read()` dari lebih
> dari satu thread** (PaddleOCR tidak thread-safe).

#### Tiga penghemat sebelum OCR jalan

1. **Cache per-plat via IoU** — plat dicocokkan antar frame (IoU ≥ 0.3); yang
   sudah pernah dibaca pakai teks lama.
2. **Interval** — re-OCR plat yang sama hanya tiap **10 frame**
   (`--ocr-interval`).
3. **Ambang tinggi** — plat < **22 px** tidak di-OCR sama sekali. Ini yang
   mematikan masalah *"salah pas awal"*: plat jauh pasti terbaca ngawur, lebih
   baik teks dibiarkan kosong sampai kendaraan cukup dekat.

Kalau antrean penuh, `last_frame` dimundurkan → job dicoba lagi frame berikutnya,
**tidak hilang** ([`src/detector.py:366`](../src/detector.py#L366)). Bacaan kosong
**tidak menimpa** bacaan lama yang sudah ada.

#### Pipeline preprocessing (kondisi sekarang)

```
crop plat → grayscale → PaddleOCR
```

Itu saja. Deskew, CLAHE, unsharp, upscale, crop-bawah — **semua default OFF**.
Kodenya masih ada dan bisa dinyalakan lewat flag, tapi ablasi menunjukkan
semuanya netral-sampai-merugikan pada crop plat yang sudah > 100 px.
`crop_bottom_ratio=0.22` bahkan **memotong huruf plat** (2/8 benar → 8/8 saat
dimatikan).

> Kalau penguji tanya *"mana preprocessing-nya?"*, jawaban yang benar:
> **"Sudah diuji A/B dan dimatikan karena merugikan. Fungsinya tetap ada di
> belakang flag."** Itu jawaban yang lebih kuat daripada punya preprocessing yang
> tidak terbukti.

#### Membuang strip pajak

Plat Indonesia punya strip masa-berlaku pajak (`07-23`) di bawah nomor. Kalau
ikut terbaca: `D 1886 AP` + `04-30` → `...CL30`, hancur.

Solusi sekarang — **"teks besar menang"**
([`src/plate_reader.py:184-189`](../src/plate_reader.py#L184-L189)): dari semua
potongan teks yang PaddleOCR kembalikan, ambil hanya yang **tingginya ≥ 0.6 ×
potongan tertinggi**. Font tanggal pajak jauh lebih kecil → otomatis terbuang.

Kelebihannya dibanding clustering berdasarkan posisi-y: **tahan plat miring**.
Clustering by-y gampang gagal saat plat mereng; rasio tinggi tidak peduli
kemiringan.

#### Normalisasi ke template Indonesia

Target: `[1-2 huruf] [1-4 angka] [0-3 huruf]`

**Jalur A — split by gaps**
([`src/plate_reader.py:315`](../src/plate_reader.py#L315)): PaddleOCR sering
sudah memisah plat di batas zona. Token dengan digit terbanyak = zona angka;
sebelumnya = kode wilayah, sesudahnya = seri.

**Jalur B — tebak kombinatorik**
([`src/plate_reader.py:261`](../src/plate_reader.py#L261)): coba semua pembagian
batas yang valid, skor = berapa karakter yang sudah cocok tipe zonanya, ambil
skor tertinggi.

**Koreksi confusion per-zona** — ini kuncinya:

```python
_DIGIT_TO_ALPHA = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '8':'B'}
_ALPHA_TO_DIGIT = {'O':'0','Q':'0','D':'0','I':'1','L':'1','Z':'2',
                   'A':'4','S':'5','B':'8','G':'6','T':'7'}
```

Di zona huruf, karakter yang terbaca angka tapi bentuknya mirip huruf →
dikembalikan ke huruf. Di zona angka, sebaliknya. **Konteks posisi memperbaiki
OCR** — inilah yang "memaksa" hasil ikut template.

**Validasi kode wilayah**
([`src/plate_reader.py:26-36`](../src/plate_reader.py#L26-L36)): daftar 60+ kode
wilayah Indonesia yang sah. Perhatikan huruf tunggal `C I J O Q U V X Y`
**bukan** kode sah. Kalau head tidak sah, sistem mencoba semua kombinasi huruf
yang mungkin (`_HEAD_DIGIT_CAND`) — misalnya `B0` → `BD` (sah). Kalau tetap gagal
dan mode strict (default): **kembalikan string kosong**, tunggu frame yang lebih
baik. **Lebih baik tidak membaca daripada salah baca.**

Self-check tanpa memuat PaddleOCR:

```powershell
venv\Scripts\python.exe src\plate_reader.py
```

#### Penyaring terakhir

Di pintu masuk tracker,
[`src/vehicle_tracker.py:80`](../src/vehicle_tracker.py#L80):

```python
PLATE_TEMPLATE_RE = re.compile(r'^[A-Z]{1,2} \d{1,4}(?: [A-Z]{1,3})?$')
```

Teks yang tidak cocok template **tidak pernah masuk track** — jadi tidak mungkin
ter-push atau ter-PATCH. OCR sampah tidak bisa mencemari web.

---

### Tahap 4 — Tracking & anti-duplikat ([`src/vehicle_tracker.py`](../src/vehicle_tracker.py)) ★ Inti skripsi

Modul terbesar (1068 baris) dan tempat kontribusi utama. **Jelaskan bagian ini
paling lama.**

#### 4.1 Pencocokan: Hungarian, bukan greedy

**Masalah yang dipecahkan — ID switch.** Greedy per-deteksi (ambil IoU tertinggi
satu per satu) menukar ID saat dua kendaraan bertumpuk. **Konsekuensinya: foto
yang dikirim ke web milik mobil yang salah.** Gagal senyap, tidak ada error,
tidak bisa dipulihkan.

**Solusi:** penugasan **global** — `scipy.optimize.linear_sum_assignment`
([`src/vehicle_tracker.py:449`](../src/vehicle_tracker.py#L449)) meminimumkan
**total cost semua pasangan sekaligus**, bukan memilih terbaik satu per satu.

**Fungsi cost** ([`src/vehicle_tracker.py:396`](../src/vehicle_tracker.py#L396)):

```
cost = 0.60 × (1 − IoU)
     + 0.25 × min(1, jarak_pusat_ternormalisasi / 1.2)
     + 0.15 × (1 − kemiripan_bentuk)

kemiripan_bentuk = 0.5 × kemiripan_ukuran + 0.5 × kemiripan_aspek
```

Ditolak (`INF_COST`) kalau: **kelas berbeda** (mobil tidak akan pernah dicocokkan
ke motor), cost > 0.8, atau gagal gate di bawah.

**Prediksi posisi**
([`src/vehicle_tracker.py:384`](../src/vehicle_tracker.py#L384)): bbox track
diekstrapolasi dulu dengan velocity (EMA 2 frame terakhir) × umur. Kendaraan yang
tertutup sesaat tetap cocok saat muncul lagi.

> ⚠️ Penyelamatan lewat prediksi (IoU = 0, cuma dekat) **hanya berlaku selama
> track hilang ≤ 10 frame** (`--track-predict-frames`). Lewat itu wajib ada IoU
> nyata. Alasannya: track yang sudah di-push dipertahankan 3× lebih lama sebagai
> memori dedup — tanpa batas ini, **track basi menyedot kendaraan BARU di posisi
> lajur yang sama**, lalu mem-PATCH baris lama dengan plat mobil lain.

**Veto plat untuk track "purnabakti"**
([`src/vehicle_tracker.py:432`](../src/vehicle_tracker.py#L432)): di antrean
macet, mobil berikutnya maju **persis** ke posisi mobil sebelumnya → IoU tinggi.
Kalau deteksi itu membawa bacaan plat sah yang **bukan varian** plat track lama,
itu kendaraan lain → tolak mutlak.

#### 4.2 Asosiasi plat → kendaraan: eksklusif

Satu plat hanya boleh dimiliki **satu** kendaraan
([`src/vehicle_tracker.py:645`](../src/vehicle_tracker.py#L645)):

| Situasi | Keputusan |
|---|---|
| Centroid plat di **zona 35% teratas** bbox kendaraan | **Tolak.** Plat ada di badan bawah; hit di zona atas = milik kendaraan di belakangnya |
| Kandidat yang memuat **seluruh kotak plat** | Menang atas yang cuma memuat titik pusat |
| 1 kandidat | Miliknya |
| >1 kandidat, terkecil jelas **bersarang** (area ≤ 0.7× berikutnya, mis. motor di depan mobil) | Milik yang terkecil |
| >1 kandidat **berukuran mirip** (dua mobil bertumpuk) | **AMBIGU → tidak diasosiasikan ke siapa pun** |

Baris terakhir itu filosofi desain sistem ini:
**menebak = data salah permanen; menunggu = telat beberapa frame.**

#### 4.3 Gerbang push — kapan kendaraan dikirim

Berurutan di
[`src/vehicle_tracker.py:568-600`](../src/vehicle_tracker.py#L568-L600):

| # | Gerbang | Default | Alasan |
|---|---|---|---|
| 1 | Belum pernah di-push | — | One-shot |
| 2 | `frames_seen ≥ 5` | `--push-stable 5` | Anti-flicker |
| 3 | Tinggi bbox ≥ 100 px | `--push-min-height` | Kendaraan jauh → foto tak berguna |
| 4 | Plat settle ≥ 12 frame **ATAU** umur ≥ 20 frame | `ocr_interval+2` / `--push-plate-wait` | Wajib lolos ≥ 1 siklus re-OCR |
| 5 | Plat tidak kosong | `require_plate=True` | Web tidak pernah menerima baris tanpa plat |
| 6 | Fuel sudah ditentukan | `--push-fuel-wait 0` | Tanpa plat, fuel tak bisa disimpulkan |
| 7 | Bukan duplikat | `--push-dedup-seconds 60` | Lihat 4.5 |

**Gerbang #4 yang paling perlu dijelaskan.** `plate_settle_frames =
ocr_interval + 2 = 12` bukan angka sembarangan — dijamin melewati **minimal satu
siklus re-OCR**. Artinya sistem baru percaya pada bacaan yang sudah dikonfirmasi
ulang oleh OCR kedua. Ini yang mencegah bacaan pertama yang salah ikut terkirim.

#### 4.4 Foto yang dikirim: "best frame" harus BERSIH

[`src/vehicle_tracker.py:760`](../src/vehicle_tracker.py#L760): frame di mana bbox
kendaraan ini bertumpuk dengan kendaraan lain (IoU > 0.5) **tidak boleh** jadi
foto push — terlalu berisiko crop-nya kendaraan yang salah.

Aturan: **frame bersih selalu menang atas frame kotor**; di antara sesama bersih,
bbox terluas (paling dekat) menang. Frame kotor hanya dipakai kalau track tidak
pernah punya frame bersih sama sekali.

Optimasi: yang disimpan adalah **crop kecil ber-padding**
([`src/vehicle_tracker.py:747`](../src/vehicle_tracker.py#L747)), bukan
`frame.copy()` utuh (~6 MB @1080p). Di kerumunan padat, salinan frame utuh tiap
ganti best-frame terasa nyata di FPS.

Foto akhir: resize ke max 800 px, JPEG quality 85.

#### 4.5 Anti-duplikat: dua penjaga, dua skala waktu

Sengaja dua-duanya, **bukan redundan**:

| Penjaga | Skala | Menangkal |
|---|---|---|
| **Retensi track** — track ter-push disimpan **3 × track_timeout = 90 frame** | Frame (sub-detik) | Deteksi berkedip; kendaraan re-match ke track lamanya, PATCH baris lama alih-alih POST baris baru |
| **Memori plat** `_recent_plates` — dipangkas per **60 DETIK wall-clock** | Detik | Kendaraan yang deteksinya putus lama (mendekat/parkir/occlusion) lalu lahir ulang sebagai track baru |

**Kenapa yang kedua berbasis detik, bukan frame?** Ini bug asli yang ditemukan:
jendela berbasis frame **menyusut dalam waktu nyata saat FPS naik**. Di 5 FPS,
300 frame = 60 detik. Di 30 FPS, 300 frame = 10 detik — kendaraan yang terdeteksi
lagi 15 detik kemudian lolos dan bikin baris duplikat. Berbasis wall-clock
membuatnya **independen dari FPS**.

**Dedup toleran-varian**
([`src/vehicle_tracker.py:313`](../src/vehicle_tracker.py#L313)) — dedup
teks-persis kecolongan varian OCR. Dianggap plat yang sama kalau:

- Angka & seri sama persis, kode wilayah beda (`DD 1446 YCS` ≈ `D 1446 YCS`)
- Seri sama + angka yang satu akhiran angka yang lain (`S 805 TBZ` ≈ `B 5805 TBZ`)
- Panjang sama + beda **tepat 1 HURUF** (`D 1371 ALS` ≈ `D 1371 ALB`)

> ⚠️ Beda 1 **ANGKA** sengaja **TIDAK** dianggap sama. `D 1234 AB` vs
> `D 1235 AB` bisa dua kendaraan sungguhan. Trade-off eksplisit:
> **salah gabung = deteksi hilang (fatal); baris duplikat = cuma jelek (kosmetik).**

#### 4.6 Koreksi plat susulan (PATCH) — fitur unggulan

**Skenario:** mobil di-push saat plat masih `D 1 S`. Mobil mendekat, OCR membaca
`D 15 NW`. Baris di web sudah terlanjur salah.

**Solusi:** `PATCH <push_url>/<id>/plate` dengan `plate_number` (plus
`is_electric` kalau tadinya `unknown`).

**Alur ID server:**

```
POST → 201 {"id": 42}
  → uploader.on_registered(track_id, 42)
  → pipeline._on_registered
  → tracker.set_server_id
```

**Yang membuat ini benar — antrean `_pending` yang hidup LEPAS dari track.**
Tiga bug pengiriman lahir dari menggandengkan keduanya:

1. **`pushed_plate` bergerak HANYA setelah PATCH sukses.** Versi lama menandai
   terkirim *sebelum* PATCH jalan → setiap PATCH gagal = plat hilang selamanya.
   Sekarang: retry exponential backoff (5×, dasar 2 detik, cap 60 detik). HTTP
   404 dianggap terminal (record hilang, percuma retry).
2. **Koreksi bertahan walau track sudah mati dan `server_id` belum datang.**
   Plat yang settle sebelum respons 201 tiba tetap ter-PATCH. TTL 120 detik.
3. **Saat track mati, gerbang settle di-BYPASS** — pakai bacaan **PALING BARU**
   yang muncul ≥ 2 frame, **bukan yang paling sering**. Alasannya elegan:
   *OCR mengoreksi ke satu arah saja* (kendaraan makin dekat → bacaan makin
   benar), jadi **suara terbanyak justru andal memilih bacaan awal yang SALAH**.

#### 4.7 Lima penjaga koreksi

Ditemukan dari A/B run penuh `vid1.mp4` — pengiriman jadi andal, tapi lalu
ketahuan *apa* yang dikirim sering salah. Kandidat koreksi harus lolos **semua**:

| Penjaga | Aturan | Kasus nyata |
|---|---|---|
| **Anti-degradasi** | Jumlah **angka** tidak boleh berkurang | `Z 1662 AV` → `Z 166 Z` = OCR memburuk saat kendaraan menjauh. Bacaan buruk yang konsisten **tetap lolos gate settle**, jadi settle saja tidak cukup |
| **Anti-kontaminasi** | Teks yang sudah terdaftar di baris **lain** → skip | Satu plat tidak mungkin milik dua baris |
| **Anti flip-flop** | `sent_plates`: teks yang pernah terkirim tidak pernah dikirim ulang | `L 1601 CF` ↔ `F 1601 CF` dulu mem-PATCH satu baris **5×** |
| **Koreksi "selevel"** | Jumlah angka sama (cuma huruf beda) → butuh **2× settle**; saat track mati → butuh **lebih banyak vote** dari yang sudah terkirim | Varian OCR yang berosilasi |
| **Veto track basi** | Lihat 4.1 | Antrean macet |

Filosofi yang konsisten di seluruh modul: **koreksi yang MENAMBAH angka (plat
makin lengkap) itu kasus yang diharapkan → syarat ringan. Koreksi yang cuma
menukar huruf → mencurigakan → syarat berat.**

---

### Tahap 5 — Pengiriman ([`src/uploader.py`](../src/uploader.py))

#### Kontrak v1 — tepat 6 field

| Field | Nilai |
|---|---|
| `plate_number` | string, boleh `""` |
| `vehicle_type` | `"mobil"` \| `"motor"` \| `"unknown"` |
| `is_electric` | `"bensin"` \| `"listrik"` \| `"unknown"` |
| `confidence_score` | float 0.0–1.0 (skor YOLO **kendaraan**, bukan skor fuel) |
| `detected_at` | ISO 8601 dengan offset TZ |
| `photo` | file JPG (server batasi 5 MB) |

**Nilai dalam Bahasa Indonesia, bukan Inggris.** Iterasi sebelumnya
menerjemahkan ke `car`/`motorcycle`/`gasoline`/`electric` — pemetaan itu
**dihapus dengan sengaja**. Jangan ditambahkan lagi.

> Inilah kenapa perubahan label **"bbm" → "konvensional"** hanya menyentuh
> **label tampilan** ([`src/utils.py:25-27`](../src/utils.py#L25-L27)), bukan
> payload. Nilai `bensin` di kawat adalah kontrak; nilai lain akan di-*coerce*
> server jadi `unknown`.

**`detected_at` = `created_ts` track (waktu first-seen), BUKAN waktu push.** Push
bisa telat beberapa detik karena antrean upload; tim web eksplisit menginginkan
**momen deteksi**. Format wajib TZ-aware (`+07:00`) — datetime naive itu
pelanggaran kontrak.

**Field yang sengaja TIDAK dikirim:** `camera_id`, `client_event_id`, `location`,
`lane` — server belum punya kolomnya, akan diabaikan diam-diam dan menciptakan
rasa percaya palsu bahwa itu "berfungsi".

Field internal berawalan `_` (`_track_id`, `_image_file`, `_pushed_at`) disaring
`_strip_internal()` sebelum POST; masuk ke **sidecar JSON lokal** saja.

#### Mekanik

Worker thread daemon, `Queue(maxsize=64)`, timeout 10 detik, 2× retry. **Deteksi
tidak pernah menunggu jaringan.** Mode `json` (tanpa foto) tersedia untuk uji
konektivitas. Autentikasi header `X-API-Key`, dibaca dari `api_key.txt` yang
di-gitignore — kunci tidak pernah masuk repo.

---

## 4. Komponen pendukung

**Preview server** ([`src/preview_server.py`](../src/preview_server.py)) — Flask
MJPEG di `:5001/preview`, thread daemon. Optimasi bagus untuk disebut: **encode
JPEG (5–15 ms @1080p!) hanya dikerjakan kalau ada client yang menonton**
([`src/preview_server.py:33`](../src/preview_server.py#L33)). Preview default ON
tapi biasanya tidak ditonton — tanpa gate ini tiap frame bayar encode percuma di
thread deteksi.

**Heartbeat** ([`src/registrar.py`](../src/registrar.py)) — POST
`/api/detector/register` tiap 15 detik berisi port preview. **Web mengambil IP
detector dari alamat sumber request**, jadi tidak perlu hardcode IP di sisi web.
Web anggap detector offline kalau tidak ada heartbeat > 60 detik. Base URL
diturunkan otomatis dari `--push-url`, tidak ada konfigurasi baru.

**GUI** ([`src/gui.py`](../src/gui.py)) — Tkinter. Model threading-nya perlu
disebut karena **Tkinter tidak thread-safe**: worker thread membaca frame +
memanggil `process_frame`, menyimpan hasil di `self._latest` di bawah lock,
**tidak menyentuh widget sama sekali**; main thread `_refresh()` tiap ~30 ms
membaca `_latest` → BGR→RGB→`PIL.ImageTk` → blit ke Label. Referensi `_imgtk`
**wajib** dipertahankan atau preview jadi blank (kena GC).

**Mock server** ([`tools/mock_server.py`](../tools/mock_server.py)) —
implementasi web tiruan lengkap dengan route PATCH, bisa simulasi latency &
packet loss. Berguna untuk demo sidang tanpa bergantung tim web.

---

## 5. Tabel parameter untuk dihafal

| Flag | Default | Fungsi |
|---|---|---|
| `--conf` | 0.4 | Ambang confidence kendaraan |
| `--imgsz` | 960 | Resolusi inference **plat** |
| `--vehicle-imgsz` | 640 | Resolusi inference **kendaraan** |
| `--ocr-interval` | 10 | Re-OCR plat sama tiap N frame |
| `--ocr-min-height` | 22 px | Jangan OCR plat lebih pendek dari ini |
| `--fuel-coverage` | 0.85 | Fraksi lebar strip biru minimum |
| `--fuel-smin` / `--fuel-vmin` | 75 / 95 | Guard median S/V "biru asli" |
| `--push-stable` | 5 | Min frame stabil sebelum push |
| `--push-min-height` | 100 px | Min tinggi bbox untuk di-push |
| `--push-plate-wait` | 20 | Batas frame menunggu plat settle |
| `--push-dedup-seconds` | 60 | Jendela anti-duplikat (detik) |
| `--push-timeout-frames` | 30 | Drop track setelah N frame hilang |
| `--track-max-overlap` | 0.5 | IoU di atas ini → frame "kotor" |
| `--track-predict-frames` | 10 | Batas penyelamatan lewat prediksi |

---

## 6. Evaluasi ([`evaluate.py`](../evaluate.py))

Menjalankan pipeline terhadap `eval_set/` + ground-truth CSV, melaporkan metrik
per-tahap:

| Metrik | Target | Exit code |
|---|---|---|
| Plate precision / recall / F1 | 0.97 | — |
| Fuel balanced accuracy | 0.95 | — |
| OCR exact match | 0.95 | — |
| **End-to-end accuracy** | **0.90** | **exit 2 kalau di bawah** |

*Balanced accuracy* untuk fuel, bukan akurasi mentah — karena dataset sangat
timpang (EV jauh lebih sedikit dari konvensional). Akurasi mentah bisa 95% hanya
dengan menebak "konvensional" untuk semua. **Ini pilihan metrik yang tepat,
sebutkan.**

```powershell
venv\Scripts\python.exe evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv --out eval_set/results.csv
```

**Tiga masalah yang harus diketahui:**

1. 🔴 **Skrip crash** (bug `blue_ratio_threshold`, lihat bagian 0).
2. 🟡 **Config eval beda dari runtime**: `imgsz=1280` (runtime 960),
   `preprocess=True` (runtime `False`). Angka evaluasi tidak mencerminkan sistem
   yang di-deploy. Penguji yang teliti akan menanyakan ini.
3. 🟡 `eval_set/` masih **template kosong**. Target ≥ 200 gambar lintas kondisi
   (siang/malam/hujan/sudut).

---

## 7. Keterbatasan — sampaikan duluan

Menyebutkan ini sendiri jauh lebih baik daripada ditemukan penguji:

1. **Klasifikasi bahan bakar murni visual.** EV yang platnya
   kotor/tertutup/ternaungi → terklasifikasi konvensional. Tidak ada fallback ke
   database registrasi.
2. **Strip EV malam gelap tertolak** oleh guard V ≥ 95. Trade-off sadar demi
   presisi.
3. **Kalibrasi HSV terikat kamera & lokasi.** Ganti kamera → harus ukur ulang S
   strip vs latar.
4. **Bus & truck tidak dideteksi** — hanya `car` dan `motorcycle`.
5. **Sisa edge case duplikat**: track kedua yang force-push (timeout tunggu plat)
   *sebelum* platnya settle masih bisa membuat baris yang baru belakangan
   ketahuan duplikat. Jarang, hanya saat OCR tertinggal dari push.
6. **Anti-kontaminasi adalah kontrol gejala**, bukan perbaikan akar, untuk
   kebocoran asosiasi plat di lalu lintas padat.
7. **Belum ada unit test otomatis** — hanya 2 self-check inline
   (`fuel_classifier.py`, `plate_reader.py`).
8. **Ground truth belum diisi**, jadi angka akurasi resmi belum ada.

---

## 8. Bank pertanyaan penguji

**"Kenapa YOLOv8n, bukan YOLOv8x yang lebih akurat?"**

> Real-time di GPU laptop. Kendaraan itu objek besar dan mudah — nano sudah
> cukup. Anggaran akurasi dialokasikan ke model **plat** yang custom-trained,
> karena di situlah letak kesulitannya.

**"Kenapa tidak pakai DeepSORT / ByteTrack?"**

> Keduanya mengoptimalkan metrik MOT (kontinuitas trajektori). Kebutuhan sistem
> ini beda: **jaminan tepat satu push per kendaraan dengan foto yang benar**.
> Yang dibutuhkan adalah gerbang push, memori dedup lintas-track, dan koreksi
> susulan — semuanya di luar cakupan tracker generik. Fondasinya tetap standar:
> Hungarian assignment + prediksi velocity, sama seperti SORT.

**"Kenapa Hungarian, bukan greedy IoU?"**

> Greedy menukar ID saat dua kendaraan bertumpuk. ID tertukar = **foto mobil A
> mendarat di baris mobil B**, gagal senyap. Hungarian meminimumkan total cost
> semua pasangan sekaligus, jadi tidak bisa "mencuri" pasangan yang sebenarnya
> lebih cocok untuk track lain.

**"Bagaimana kalau plat tidak terbaca sama sekali?"**

> Default `require_plate=True` — kendaraan tanpa plat **tidak pernah** dikirim ke
> web. Track menunggu sampai platnya terbaca; kalau sampai timeout tetap kosong,
> dibuang tanpa push. Web tidak pernah menerima baris kosong.

**"Kenapa strip biru, bukan model klasifikasi terlatih?"**

> Plat EV Indonesia punya penanda visual yang terstandar regulasi. Analisis
> warna: deterministik, bisa dijelaskan, tidak butuh dataset EV berlabel (yang
> langka), dan ~0 biaya komputasi. CNN butuh ratusan gambar EV yang tidak
> tersedia. Trade-off-nya sensitif pencahayaan — itu sebabnya ada tiga guard
> berlapis.

**"Berapa FPS-nya?"**

> Jalankan `venv\Scripts\python.exe src\main.py --source <video>` — angka tampil
> di overlay & log tiap 5 detik. Optimasi yang sudah dilakukan: vehicle imgsz
> 640, FP16 otomatis, OCR async, preview-encode gating, crop-on-capture,
> ekstraksi box tervektorisasi. **Ukur sebelum sidang dan hafalkan angkanya.**

**"Bagaimana Anda tahu perbaikan-perbaikan ini benar-benar bekerja?"**

> A/B run penuh pada `vid1.mp4`, membandingkan baris yang masuk ke web sebelum
> dan sesudah tiap penjaga. Kalibrasi fuel dari 84 capture berlabel (11 EV vs
> 73 FP). Optimasi yang tidak terbukti untung (`cudnn.benchmark`, upscale OCR,
> crop 22%) **ditolak dengan data**, bukan diterima karena "terdengar masuk
> akal".

---

## 9. Prioritas sebelum sidang

1. 🔴 **Perbaiki `evaluate.py`** — satu baris (`blue_ratio_threshold` →
   `strip_coverage_threshold`). Tanpa ini tidak ada angka akurasi sama sekali.
2. 🔴 **Isi `eval_set/`** dan jalankan evaluasi. Sidang tanpa angka itu berat.
3. 🟡 **Selaraskan default eval dengan runtime** (imgsz & preprocess), atau
   siapkan penjelasan kenapa berbeda.
4. 🟡 **Perbarui `CLAUDE.md`** (bus/truck, preprocessing OCR) supaya dokumen
   tidak bertentangan dengan kode.
5. 🟢 **Ukur & hafalkan FPS** di mesin yang akan dipakai demo.
