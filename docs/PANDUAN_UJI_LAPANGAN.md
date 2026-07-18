# Panduan Uji Lapangan (Live)

Panduan ini untuk menguji sistem saat **berjalan langsung** (kamera live / video),
bukan akurasi angka dari gambar statis. Untuk uji akurasi (deteksi, OCR, BBM) pakai
`evaluate.py` — lihat [`eval_set/PANDUAN_PENGUJIAN.md`](../eval_set/PANDUAN_PENGUJIAN.md).
Kedua dokumen saling melengkapi:

| Uji apa | Dokumen | Alat |
|---|---|---|
| Angka akurasi (presisi/recall/OCR exact-match) | `eval_set/PANDUAN_PENGUJIAN.md` | `evaluate.py` + gambar berlabel |
| Perilaku live (jarak, lux, kecepatan, kirim ke web, stabilitas) | **dokumen ini** | `src\main.py` / GUI + kamera/video |

---

## 0. Persiapan (lakukan SEBELUM mulai)

### 0.1 Dua hal yang bisa memblokir pengujian

1. **Endpoint koreksi plat di web** — uji "koreksi plat" (bagian 8) butuh
   `PATCH /api/detections/<id>/plate` sudah jadi di sisi web. Kalau belum, koreksi
   akan gagal diam-diam (`[Uploader] UPDATE HTTP 404`). **Cek dulu ke tim web.**
2. **Akses kendaraan listrik / plat biru** — uji klasifikasi BBM (bagian 5) tidak
   bisa dibuktikan tanpa minimal satu plat listrik asli. Siapkan dulu.

### 0.2 Alat yang perlu disiapkan

- Meteran / titik jarak yang sudah ditandai di lapangan (uji jarak).
- Aplikasi **lux meter** di HP (atau lux meter fisik) untuk uji pencahayaan.
- Kendaraan uji: campur **mobil & motor**, ada **plat listrik** dan **bensin**.
- Stopwatch / timer (uji stabilitas & latensi).
- Laptop web monitoring (atau `tools\mock_server.py` untuk simulasi).

### 0.3 Cara membaca log console (WAJIB paham — ini "alat ukur" utama)

Saat program jalan, perhatikan baris-baris ini di terminal:

| Baris log | Artinya |
|---|---|
| `[Tracker] PUSH #xxxx mobil/bensin plate="..."` | 1 kendaraan dikirim ke web (baris baru) |
| `[Tracker] UPDATE #xxxx plat A -> "B"` | koreksi plat dikirim (PATCH) ke baris lama |
| `[Tracker] DEDUP #xxxx plat "..." baru saja di-push` | duplikat dicegah — **tidak** bikin baris baru |
| `[Uploader] OK 201 ... server_id=N` | web terima kiriman, id record = N |
| `[Uploader] UPDATE SUCCESS id=N` | koreksi plat berhasil masuk web |
| `[Uploader] HTTP 401` | **ditolak** — API key salah/tidak ada |
| `[Uploader] error ...` | web tak terjangkau (jaringan) — akan di-retry |
| `[Heartbeat] Terdaftar di web` | auto-registrasi detector berhasil |

### 0.4 Cara mencatat bukti

- **Screenshot frame**: tekan `S` saat window aktif → tersimpan ke `output/`.
- **Log per-deteksi (CSV/JSONL)**: tambah `--log-jsonl output\uji.jsonl` → 1 baris
  per deteksi per frame (bisa dibuka untuk analisis).
- **Foto yang dikirim**: cek folder `output/captures/` (gambar + file `.json` sidecar).
- **Hasil akhir**: bandingkan dengan tampilan di **dashboard web**.

### 0.5 Kontrol keyboard saat window aktif

`Q` keluar · `Space` pause · `S` screenshot · `L` toggle loop · `←`/`→` step frame saat pause.

---

## 1. Uji Jarak

**Tujuan:** cari jarak maksimum di mana (a) kendaraan terdeteksi, (b) plat terdeteksi,
(c) plat terbaca benar.

**Langkah:**
1. Tandai jarak di tanah: mis. 2 m, 4 m, 6 m, 8 m, 10 m dari kamera.
2. Jalankan:
   ```powershell
   venv\Scripts\python.exe src\main.py --source 0
   ```
3. Tempatkan kendaraan di tiap titik jarak, diam sebentar. Catat di jarak berapa:
   kotak kendaraan muncul, kotak plat muncul, dan teks plat muncul & **benar**.

**Hasil diharapkan:** ada jarak wajar (sesuai penempatan kamera) di mana ketiganya
tercapai. Kendaraan sangat jauh **sengaja tidak dikirim** (filter `--push-min-height`
100 px) — itu perilaku benar, bukan bug.

**Catatan:** model kendaraan kini jalan di 640 px (`--vehicle-imgsz`), plat di 960 px.
Kalau butuh jangkauan lebih jauh untuk plat, coba `--imgsz 1280`; untuk kendaraan
jauh, `--vehicle-imgsz 960`.

| Jarak | Kendaraan terdeteksi? | Plat terdeteksi? | Plat benar? | Catatan |
|---|---|---|---|---|
| 2 m | | | | |
| 4 m | | | | |
| 6 m | | | | |
| 8 m | | | | |
| 10 m | | | | |

---

## 2. Uji Pencahayaan (Lux)

**Tujuan:** ukur performa di berbagai tingkat cahaya, dan cari batas minimum lux.

**Langkah:**
1. Ukur lux di lokasi plat pakai lux meter (HP), catat angkanya.
2. Uji di beberapa kondisi: terang (siang), redup (mendung/sore), gelap (malam +
   lampu), melawan cahaya (backlight/silau).
3. Di tiap kondisi, catat: kendaraan terdeteksi, plat terdeteksi, OCR benar, dan
   **BBM benar** (ini paling sensitif ke cahaya — lihat bagian 5).

**Hasil diharapkan:** performa turun bertahap seiring gelap; catat lux minimum di
mana OCR & BBM masih andal.

| Kondisi | Lux | Deteksi | OCR benar | BBM benar | Catatan |
|---|---|---|---|---|---|
| Siang terang | | | | | |
| Sore/redup | | | | | |
| Malam + lampu | | | | | |
| Backlight/silau | | | | | |

---

## 3. Uji Sudut Kamera

**Tujuan:** cari sudut kemiringan maksimum plat yang masih terbaca (OCR punya tahap
deskew — ada batasnya).

**Langkah:** posisikan kendaraan/kamera pada beberapa sudut horizontal (mis. 0°, 15°,
30°, 45°) dan vertikal (kamera dari atas). Catat sudut terakhir di mana plat masih
kebaca benar.

| Sudut | Plat terdeteksi? | Plat terbaca benar? | Catatan |
|---|---|---|---|
| 0° (lurus) | | | |
| 15° | | | |
| 30° | | | |
| 45° | | | |

---

## 4. Uji Kecepatan Kendaraan

**Tujuan:** ini uji terpenting untuk logika pengiriman. Kendaraan cepat berisiko
terkirim sebelum plat terbaca; kendaraan diam/pelan berisiko deteksi putus & bikin
baris dobel.

**Langkah:** jalankan dengan push aktif (lihat bagian 7 untuk setup web), lalu uji
tiga skenario dan amati log + dashboard:

1. **Parkir/diam** — kendaraan diam di depan kamera beberapa detik.
2. **Pelan** — lewat perlahan (~kecepatan orang jalan).
3. **Cepat** — lewat cepat.

**Hasil diharapkan:**
- Semua tetap terkirim (`[Tracker] PUSH`).
- **Tidak ada baris dobel** untuk satu kendaraan (cek dashboard).
- Plat cepat-lewat boleh awalnya kosong lalu terisi via `[Tracker] UPDATE` (kalau
  endpoint PATCH web sudah ada).

| Skenario | Terkirim? | Plat kebaca? | Baris dobel? | Catatan |
|---|---|---|---|---|
| Parkir/diam | | | | |
| Pelan | | | | |
| Cepat | | | | |

**Kalau kendaraan cepat sering tak terbaca platnya:** naikkan `--push-plate-wait`
(tunggu plat lebih lama sebelum kirim). Kalau muncul baris dobel: laporkan, mungkin
perlu menyetel `--push-dedup-seconds` (default 60).

---

## 5. Uji Klasifikasi BBM (Bensin vs Listrik)

**Tujuan:** buktikan sistem membedakan plat listrik (strip biru) vs bensin. Berbasis
rasio warna biru (HSV) — **sangat sensitif ke cahaya**, jadi uji silang dengan lux.

**Langkah:**
1. Uji minimal 1 plat **listrik** (biru) dan beberapa plat **bensin**.
2. Amati label `is_electric` di dashboard / field `fuel_type` di log.
3. Ulangi di kondisi cahaya berbeda (siang & redup) untuk plat listrik yang sama.

**Hasil diharapkan:** plat listrik → `listrik`, plat bensin → `bensin`. Tidak ada
plat bensin yang salah jadi `listrik` (false positive) dan sebaliknya.

| Plat | BBM sebenarnya | Hasil (terang) | Hasil (redup) | Catatan |
|---|---|---|---|---|
| (listrik) | listrik | | | |
| (bensin 1) | bensin | | | |
| (bensin 2) | bensin | | | |

---

## 6. Uji Akurasi OCR (angka)

Untuk **exact-match rate** dan metrik akurasi resmi, jangan hitung manual — pakai
`evaluate.py`. Ikuti [`eval_set/PANDUAN_PENGUJIAN.md`](../eval_set/PANDUAN_PENGUJIAN.md).
Ringkasnya:
```powershell
venv\Scripts\python.exe evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv
```

**Khusus dicek di sini:** apakah ada plat **format sah tapi tidak biasa** (plat dinas,
plat khusus) yang tampil **kosong**. Sistem menyaring teks ke pola plat standar
(`PLATE_TEMPLATE_RE`); plat di luar pola sengaja dikosongkan (bukan ditampilkan
salah). Catat kalau ketemu plat sah yang tertolak.

---

## 7. Uji Konektivitas ke Web

### 7.1 Setup

Cari IP laptop web (`python tools\get_lan_ip.py` di laptop web), lalu di laptop
detektor:
```powershell
venv\Scripts\python.exe src\main.py --source 0 --push --push-url http://<IP_WEB>:5000/api/detections
```
API key terisi otomatis dari `api_key.txt`.

### 7.2 Kasus uji

| No | Kondisi | Cara | Hasil diharapkan |
|---|---|---|---|
| a | Kirim normal | jalankan dengan `--push` | `[Uploader] OK 201`, data muncul di dashboard |
| b | API key benar | (default, key dari `api_key.txt`) | 201, bukan 401 |
| c | API key salah/kosong | tambah `--api-key ""` | `[Uploader] HTTP 401`, data **tidak** masuk (bukti proteksi jalan) |
| d | Web mati lalu hidup | matikan server web di tengah sesi, lalu hidupkan lagi | detektor tetap jalan, antre, lalu terkirim saat web balik (tidak crash) |
| e | Jaringan putus | cabut kabel/Wi-Fi sebentar | `[Uploader] error` lalu pulih sendiri, tidak crash |
| f | Foto sampai | cek dashboard | foto kendaraan tampil, tidak melebihi 5 MB |
| g | Heartbeat/registrasi | amati log saat start | `[Heartbeat] Terdaftar di web` |

### 7.3 Simulasi LAN jelek (opsional, tanpa laptop web)

```powershell
# terminal 1: web tiruan dengan latensi & packet loss
python tools\mock_server.py --port 5000 --latency-ms 30 --drop-rate 0.1
# terminal 2: detektor menembak ke sana
venv\Scripts\python.exe src\main.py --source video.mp4 --push --push-url http://127.0.0.1:5000/api/detections
```
Atau pakai `tools\sim_laptop_b.bat` (web tiruan) + `tools\sim_laptop_a.bat video.mp4`.

**Hasil diharapkan:** meski ada delay/drop, detektor tidak macet; kiriman yang gagal
di-retry, FPS deteksi tidak ikut turun (upload jalan di thread terpisah).

---

## 8. Uji Koreksi Plat (PATCH) — perlu endpoint web

**Tujuan:** buktikan satu baris bisa ter-update, bukan bikin baris baru.
**Prasyarat:** `PATCH /api/detections/<id>/plate` sudah ada di web (bagian 0.1).

**Langkah:** dengan push aktif, uji dua kasus (paling mudah dengan video di mana
kendaraan mendekat sehingga plat makin jelas):

1. **Kosong → terisi**: kendaraan terkirim saat plat belum terbaca, lalu plat kebaca.
   - Amati: `[Tracker] PUSH ... plate="(no plate)"` → lalu `[Tracker] UPDATE ... plat (kosong) -> "..."`.
2. **Salah → benar**: plat awalnya terbaca salah lalu terkoreksi (mis. `D 1 S` → `D 15 NW`).
   - Amati: `[Tracker] UPDATE ... plat D 1 S -> "D 15 NW"`.

**Hasil diharapkan:** di dashboard, **baris yang sama** berubah platnya — **bukan**
muncul baris kedua. Log uploader: `[Uploader] UPDATE SUCCESS`.

| Kasus | Log UPDATE muncul? | Baris di web ter-update? | Baris dobel? | Catatan |
|---|---|---|---|---|
| Kosong → terisi | | | | |
| Salah → benar | | | | |

---

## 9. Uji Anti-Duplikat (satu kendaraan = satu baris)

**Tujuan:** validasi akhir — hitung kendaraan nyata vs baris di dashboard.

**Langkah:**
1. Jalankan sesi berisi sejumlah kendaraan lewat (catat jumlah sebenarnya).
2. Setelah selesai, hitung jumlah baris di dashboard.

**Hasil diharapkan:** jumlah baris = jumlah kendaraan. Perhatikan kasus yang dulu
bermasalah: kendaraan yang deteksinya sempat putus (terhalang/menjauh) lalu muncul
lagi — harus tetap 1 baris (`[Tracker] DEDUP` muncul di log).

| Sesi | Jumlah kendaraan nyata | Jumlah baris di web | Cocok? | Catatan |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |

---

## 10. Uji FPS (Beban Nyata)

**Tujuan:** ukur FPS pada kondisi kerja penuh, bukan frame kosong.

**Langkah:** jalankan program utama (FPS tampil di window), uji beberapa kondisi:

| Kondisi | Perintah tambahan | FPS |
|---|---|---|
| Penuh (OCR + deteksi) | (default) | |
| Tanpa OCR | `--no-ocr` | |
| Banyak kendaraan sekaligus | (arahkan ke jalan ramai) | |
| Dengan push aktif | `--push --push-url ...` | |

**Hasil diharapkan:** selisih "penuh" vs "tanpa OCR" kecil (OCR sudah async). Kalau
selisihnya besar, laporkan. FP16 & vehicle-imgsz 640 sudah aktif secara default.

---

## 11. Uji Stabilitas Jangka Panjang

**Tujuan:** pastikan sistem tahan jalan berjam-jam (ini pembeda prototipe vs siap-pakai).

**Langkah:**
1. Jalankan headless berjam-jam:
   ```powershell
   venv\Scripts\python.exe src\main.py --source video.mp4 --loop --no-window --log-jsonl output\uji_lama.jsonl --push --push-url http://<IP_WEB>:5000/api/detections
   ```
2. Pantau tiap ~30 menit: pemakaian RAM (Task Manager), GPU (`nvidia-smi`), dan FPS.

**Hasil diharapkan:** RAM stabil (tidak naik terus = tidak ada memory leak dari
thread OCR/queue/cache), FPS tidak turun drastis (kalau turun karena laptop panas =
thermal throttle, catat), tidak ada crash.

| Jam ke- | RAM | GPU mem | FPS | Catatan |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |

---

## Ringkasan urutan disarankan

1. Uji bench cepat dulu: **jarak, sudut, lux** (statis, mudah diulang).
2. Lalu **BBM** (butuh plat listrik) dan **akurasi OCR** (`evaluate.py`).
3. Nyalakan web → **konektivitas**, lalu **kecepatan**, **koreksi plat**, **anti-duplikat**.
4. Terakhir **FPS beban nyata** & **stabilitas jangka panjang**.
