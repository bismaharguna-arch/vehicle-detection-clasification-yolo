# Handoff — Sub-bab 4.2.7 Pengujian Operasional (data 3 Agustus 2026)

Ditulis 5 Agustus 2026. Sesi ini berjalan di repo **deteksi_kendaraan**, padahal
pembahasan buku kemungkinan seharusnya di repo **server monitoring**. Datanya sendiri
memang berada di repo deteksi, jadi seluruh angka di bawah dihitung dari sini.

---

## 1. Konteks

Penguji meminta dua hal:

1. Gambar hasil cetak laporan pada buku terlalu kecil untuk dibaca.
2. Data hasil monitoring tanggal **3 Agustus 2026** supaya **diolah lagi**, bukan
   ditempel mentah.

Poin 1 sudah selesai dijawab (ringkasannya di bagian 6). Sesi ini fokus ke poin 2.

## 2. Keputusan yang sudah diambil

| Pertanyaan | Keputusan | Alasan |
|---|---|---|
| Sub-bab sendiri atau nempel ke yang lama? | **Sub-bab sendiri** | Jenis pengujiannya beda: 4.2.1–4.2.6 terkendali (satu variabel diubah), ini operasional (tanpa variabel dikendalikan). Kalau diselipkan, jadi anomali metodologis |
| Nomornya 4.3 atau 4.2.7? | **4.2.7 Pengujian Operasional Sistem** | Ini tetap sebuah pengujian, jadi hierarkinya di bawah 4.2. Naik ke 4.3 hanya perlu kalau isinya dipecah bersubjudul — itu memaksa *heading* level 4 (4.2.7.1) yang di luar template TA |
| Dampak ke penomoran tabel | **Nol** | Posisinya paling akhir di BAB IV, tabel barunya menyambung jadi Tabel 4.13 dst. |
| Tabel verifikasi plat masuk sini atau ke 4.2.6? | **Tetap di 4.2.7**, tapi judulnya diganti | 4.2.6 mengukur kinerja modul OCR pada kondisi terkendali; tabel ini mengukur mutu 208 baris data yang benar-benar tersimpan. Beda pertanyaan, tidak saling mengulang |

**Perubahan judul yang belum diterapkan** (sudah disetujui konsepnya, belum diedit ke
berkas): "Tabel 4.15 Verifikasi Ketepatan Pembacaan Nomor Plat" → **"Tabel 4.15 Ketepatan
Data Hasil Pendataan Operasional"**, dengan pembingkaian "jumlah data belum menggambarkan
mutu data" supaya jelas bedanya dengan 4.2.6.

## 3. Angka hasil olahan (jangan dihitung ulang, sudah terverifikasi)

Sumber: 208 berkas JSON pendamping `output/captures/*20260803*.json` (+208 JPG = 416 berkas).

**Ringkasan (Tabel 4.13):**

- Durasi 2 jam 23 menit (09.09–11.32 WIB) — 82 kendaraan pukul 09, 94 pukul 10, 32 pukul 11
- 208 kendaraan, rata-rata 87 kendaraan/jam
- Tipe kendaraan: **208 mobil, 0 motor**
- Tipe energi: **206 konvensional, 2 listrik** (0,96%)
- Plat terbaca: 208/208 (100%); sesuai pola plat Indonesia: 208/208 (100%)
- Plat unik 205; 3 nomor terekam dua kali
- *Confidence*: min 0,40 · rata-rata 0,61 · maks 0,95
- Jeda deteksi→kirim: median 3 detik, maks 16 detik
- `frames_seen`: median 47, min 5, maks 361 (16 data di bawah 12 bingkai)
- Ukuran foto rata-rata 25 KB

**Sebaran confidence (Tabel 4.14):**

| Selang | Jumlah | % |
|---|---|---|
| 0,40–0,49 | 78 | 37,5 |
| 0,50–0,59 | 40 | 19,2 |
| 0,60–0,69 | 27 | 13,0 |
| 0,70–0,79 | 18 | 8,7 |
| 0,80–0,89 | 22 | 10,6 |
| 0,90–1,00 | 23 | 11,1 |

Turunan penting: **55 dari 208 (26,4%) berada di bawah 0,46.**

**Plat ganda (bukan bug):**

| Nomor | Waktu | Selisih |
|---|---|---|
| B 392 EB | 09.13.33 & 11.19.07 | 2 jam 6 menit |
| D 1051 AMM | 09.53.55 & 10.07.26 | 14 menit |
| D 1024 AER | 10.20.12 & 10.43.22 | 23 menit |

Semuanya jauh melampaui jendela pencegahan data ganda (60 detik), jadi ini kendaraan yang
memang melintas dua kali — bukan kegagalan dedup. Tidak ada data ganda dari satu lintasan.

## 4. Berkas yang dihasilkan sesi ini

| Berkas | Isi |
|---|---|
| `docs/REVISI_PENGUJI.md` → bagian **REVISI 5** | Draf lengkap 4.2.7 siap tempel: paragraf pengantar, Tabel 4.13/4.14/4.15, paragraf pembahasan, paragraf data ganda, paragraf keterbatasan, paragraf penutup |
| `docs/verifikasi_sampel_20260803.csv` | Lembar kerja verifikasi manual, 52 baris (sampel sistematis setiap kendaraan ke-4) |
| `docs/HANDOFF_4-2-7.md` | Berkas ini |

Judul `REVISI_PENGUJI.md` sudah diubah dari "4 Poin Penguji" → "5 Poin Penguji".

## 5. Yang belum selesai

1. **Tabel 4.15 masih kosong** — wajib diisi manual. Buka
   `docs/verifikasi_sampel_20260803.csv`, cocokkan tiap `berkas_foto` dengan fotonya di
   `output/captures/`, isi kolom `plat_sebenarnya_ISI_MANUAL`, `benar_1_salah_0`, dan
   `tipe_energi_sebenarnya_ISI_MANUAL`. Perkiraan 30–40 menit. Angka ini sekaligus mengisi
   *placeholder* **[XX]%** di Abstrak.
2. **Kurung siku di paragraf pengantar** — lokasi pengujian, posisi & ketinggian kamera,
   jarak kamera ke lajur, kondisi cuaca. Satu-satunya bagian yang tidak bisa diambil dari data.
3. **Penerapan perubahan judul Tabel 4.15** (lihat bagian 2).
4. **PERTANYAAN TERBUKA — buku siapa?** Belum dijawab, dan ini menentukan bentuk akhirnya:
   - Kalau **buku deteksi**: draf REVISI 5 sudah pas, tinggal tempel.
   - Kalau **buku web monitoring**: sudut pandangnya harus digeser ke sisi server — jumlah
     entri diterima, keberhasilan POST/PATCH, keutuhan data tersimpan, fungsi filter &
     cetak laporan. Bagian sebaran *confidence* dan pembahasan ambang 0,46 **dikeluarkan**,
     karena itu ranah program deteksi dan akan dipertanyakan penguji sebagai bukan bagiannya.
   - Petunjuk yang belum dikonfirmasi: gambar "Gambar 4.11 Hasil Cetak Laporan" itu fitur
     sisi web — perlu dipastikan ada di buku yang mana.

## 6. Peringatan penting

**Betulkan dulu klaim ambang 0,46 di sub-bab 4.2.1 dan 4.2.3 sebelum Tabel 4.14 masuk.**
Laporan sekarang menyebut 0,46 sebagai *"nilai yang sama persis dengan yang dipakai program
saat berjalan"*, padahal nilai bawaan `--conf` di `src/main.py` adalah **0,4**. Tabel 4.14
memuat 78 kendaraan di selang 0,40–0,49 — itu bukti tertulis program berjalan di 0,4,
sehingga klaim lama akan terbantah oleh tabel di halaman berikutnya. Perbaikan yang
disarankan: *"0,46 sebagai hasil analisis kurva F1, sedangkan program dijalankan pada ambang
0,4"*. Paragraf pembahasan Tabel 4.14 di draf sudah ditulis dengan pengandaian ini sudah
dibetulkan. (Lihat juga Temuan Tambahan no. 1 di `REVISI_PENGUJI.md`.)

**Jangan menarik akurasi klasifikasi tipe energi dari data ini** — hanya ada 2 sampel
kendaraan listrik. Akurasi klasifikasi tetap mengacu ke pengujian terkendali.

## 7. Poin 1 penguji (gambar kekecilan) — sudah dijawab, tinggal dikerjakan

Urutan perbaikan: (a) ambil ulang screenshot dengan zoom browser 150–175% supaya huruf
dirender lebih besar, bukan di-*upscale*; (b) crop tombol "Cetak/Simpan PDF", scrollbar,
margin kosong, dan potong tabel di ~8–10 baris; (c) di Word pasang selebar area teks
(15–16 cm, *in line with text*); (d) kalau masih kecil, pecah jadi dua gambar atau buat
halaman itu *landscape*. Isi laporan tidak perlu memperlihatkan semua 208 baris — narasi
sudah menyebut jumlahnya.

## 8. Cara menghitung ulang kalau perlu

```powershell
cd d:\deteksi_kendaraan\output\captures
d:\deteksi_kendaraan\venv\Scripts\python.exe -c "import json,glob,statistics as st; rows=[json.load(open(f,encoding='utf-8')) for f in glob.glob('*20260803*.json')]; print(len(rows), '%.2f'%st.mean(r['confidence_score'] for r in rows))"
```

Gunakan `venv\Scripts\python.exe`, bukan `python` sistem.
