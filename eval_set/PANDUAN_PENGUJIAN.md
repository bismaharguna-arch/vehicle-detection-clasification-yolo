# Panduan Pengujian Akurasi Sistem

Panduan ini menjelaskan cara mengukur **akurasi sistem** (deteksi kendaraan, deteksi
plat, klasifikasi listrik/bensin, dan OCR) memakai program `evaluate.py`, lalu
menyimpan hasilnya ke folder **`hasil deteksi/`** berupa grafik + catatan angka.

> Singkatnya: kamu siapkan **gambar** + **kunci jawaban** → jalankan 1 perintah →
> angka akurasi & grafik keluar otomatis untuk laporan.

---

## Gambaran alur

```
1. Kumpulkan gambar   ->  eval_set/  (img001.jpg, img002.jpg, ...)
2. Isi kunci jawaban  ->  eval_set/ground_truth.csv
3. Jalankan program   ->  python evaluate.py ...
4. Ambil hasil        ->  hasil deteksi/ (grafik + catatan akurasi)
```

---

## Langkah 1 — Kumpulkan gambar uji

Taruh foto-foto kendaraan hasil pengujianmu ke dalam folder `eval_set/`.

- Format: `.jpg` (boleh juga `.png`).
- Beri nama urut biar rapi: `img001.jpg`, `img002.jpg`, dst.
- **Sesuai batasan masalah TA**, cukup kondisi **siang hari** di sekitar Telkom
  University. Tapi tetap variasikan biar hasil meyakinkan:
  - campuran **mobil dan motor**
  - beberapa **kendaraan listrik** (plat ada strip biru) supaya klasifikasi energi
    benar-benar teruji — usahakan minimal beberapa buah
  - variasi **sudut** dan **jarak** kamera
  - sertakan sedikit foto **tanpa kendaraan** (untuk uji salah-deteksi)

Semakin banyak gambar, semakin dipercaya angkanya. Untuk laporan TA, usahakan
**puluhan sampai ratusan gambar**.

---

## Langkah 2 — Isi kunci jawaban (`ground_truth.csv`)

File ini berisi **jawaban benar** tiap gambar, yang nanti dibandingkan dengan
tebakan sistem. Buka `ground_truth.csv` (bisa pakai Excel / Notepad), lalu isi
satu baris per gambar.

### Arti tiap kolom

| Kolom | Isi | Contoh |
|---|---|---|
| `filename` | nama file gambar (persis, termasuk `.jpg`) | `img001.jpg` |
| `vehicle_type` | jenis kendaraan: `mobil` / `motor` / kosong | `mobil` |
| `plate_text` | teks plat yang benar (spasi diabaikan). Kosong = plat tak terbaca | `B 1234 ABC` |
| `fuel_type` | tipe energi: `bensin` / `listrik` / kosong | `listrik` |
| `plate_x1` | koordinat X pojok **kiri-atas** kotak plat | `520` |
| `plate_y1` | koordinat Y pojok **kiri-atas** kotak plat | `610` |
| `plate_x2` | koordinat X pojok **kanan-bawah** kotak plat | `720` |
| `plate_y2` | koordinat Y pojok **kanan-bawah** kotak plat | `660` |

### Cara mengisi tiap kasus

- **Kendaraan + plat terlihat** → isi semua kolom (lihat `img001`–`img004`).
- **Kendaraan listrik** → `fuel_type` = `listrik` (plat ada strip biru).
- **Kendaraan ada tapi plat tak jelas / tak ada** → isi `vehicle_type` saja,
  sisanya kosong (lihat `img005`).
- **Foto tanpa kendaraan** → isi `filename` saja, kolom lain kosong (lihat `img006`).

### Cara mendapat koordinat kotak plat (bagian yang agak teknis)

Koordinat memakai satuan **piksel**, dihitung dari **pojok kiri-atas gambar (0,0)**.
Yang kamu butuhkan cuma 2 titik: **pojok kiri-atas** dan **pojok kanan-bawah** plat.

Cara paling gampang tanpa aplikasi tambahan:

1. Buka gambar dengan **Paint** (bawaan Windows).
2. Arahkan kursor ke **pojok kiri-atas plat** → lihat angka koordinat di kiri-bawah
   jendela Paint (mis. `520, 610`). Itu `plate_x1, plate_y1`.
3. Arahkan kursor ke **pojok kanan-bawah plat** → catat angkanya (mis. `720, 660`).
   Itu `plate_x2, plate_y2`.

> Tidak perlu presisi sempurna — asal kotak menutupi plat dengan wajar sudah cukup,
> karena penilaian memakai ambang tumpang-tindih (IoU) 50%.

---

## Langkah 3 — Jalankan pengujian

Buka PowerShell di folder proyek, lalu jalankan:

```powershell
venv\Scripts\python.exe evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv
```

Program akan memproses semua gambar dan mencetak laporan akurasi di layar.

Opsi tambahan (opsional):
- `--result-dir "nama folder"` → ganti folder output (default: `hasil deteksi`).
- `--ocr-cpu` → paksa OCR pakai CPU kalau GPU bermasalah.

---

## Langkah 4 — Ambil hasil untuk laporan

Setelah selesai, cek folder **`hasil deteksi/`**. Isinya:

| File | Untuk apa |
|---|---|
| `grafik_akurasi.png` | Grafik batang semua akurasi → **tempel sebagai Gambar di TA** |
| `ringkasan_akurasi.txt` | Catatan angka lengkap (bisa dibaca langsung) |
| `ringkasan_akurasi.csv` | Angka ringkas per metrik → **salin ke tabel laporan** |
| `hasil_per_gambar.csv` | Rincian benar/salah tiap gambar (untuk analisis kesalahan) |

Cara baca grafik: batang **hijau** = mencapai target, **oranye** = di bawah target,
garis putus-putus merah = target.

---

## Catatan penting

- **FPS (kecepatan)** tidak diukur di sini, melainkan saat menjalankan program utama
  (`src\main.py` atau GUI) — angkanya tampil di layar.
- Angka akurasi hanya **sevalid gambar ujimu**. Kalau gambar sedikit atau semua
  kondisinya sama, angkanya kurang meyakinkan.
- Kolom koordinat plat (`plate_x1..y2`) hanya perlu diisi untuk gambar yang ada
  platnya. Untuk foto tanpa plat/tanpa kendaraan, biarkan kosong.
