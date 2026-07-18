# Teknologi yang Digunakan

Program ini mendeteksi kendaraan dari kamera, membaca plat nomornya, menebak
jenis bahan bakar, lalu mengirim datanya ke website. Berikut alat-alat yang dipakai:

## Bahasa Pemrograman
- **Python** — bahasa utama program ini.

## Untuk Mengenali Kendaraan & Plat (Kecerdasan Buatan / AI)
- **YOLOv8** — "mata" program: mendeteksi ada kendaraan dan menemukan letak plat nomor di gambar.
- **PaddleOCR** — membaca tulisan/angka pada plat nomor.
- **OpenCV** — mengolah gambar dari kamera dan menebak jenis bahan bakar dari warna plat (plat hijau/biru = listrik).
- **PyTorch** — mesin penggerak di balik YOLOv8 agar bisa berpikir cepat (dibantu kartu grafis/GPU).

## Untuk Menampilkan & Mengirim Data
- **Flask** — menampilkan tayangan langsung kamera lewat browser (live preview).
- **requests** — mengirim hasil deteksi (data + foto) ke website pemantau.

## Alat Pendukung
- **NumPy, Pillow, scikit-image** — membantu mengolah gambar.
- **pandas** — mengolah data untuk pengujian/evaluasi.
- **matplotlib** — membuat grafik hasil.

---

**Singkatnya:** program ini berbasis **Python**, memakai **YOLOv8** dan
**PaddleOCR** untuk mengenali kendaraan & membaca plat, **OpenCV** untuk
mengolah gambar, lalu **Flask** dan **requests** untuk menampilkan dan
mengirim datanya ke website.
