# Draf Revisi Sidang TA — 5 Poin Penguji

Semua angka di bawah diambil langsung dari kode (`src/`), bukan perkiraan. Rujukan berkas
disebutkan di setiap bagian agar mudah diverifikasi.

---

## REVISI 1 — Tahapan Pembangunan Sistem

**Penempatan (sudah diputuskan):** sub-bab baru **3.3 Tahapan Pembangunan Sistem** di BAB III,
setelah 3.2 Pemodelan Sistem dan sebelum BAB IV. Saat ini laporan melompat dari Arsitektur
(BAB III) ke Implementasi (BAB IV) tanpa menjelaskan *urutan kerja* pembangunannya — itu
yang ditanyakan penguji.

**Catatan teknis Word:**
- Ketik judulnya `Tahapan Pembangunan Sistem` saja dengan style **Heading 2**. Jangan ketik
  angka "3.3" — nomor sub-bab level 2 di dokumen ini dihasilkan otomatis oleh Word
  (lihat 3.1 dan 3.2 yang juga tidak diketik nomornya). Beda dengan heading level 3
  ("3.1.1", "4.1.1") yang nomornya diketik manual.
- BAB III belum punya tabel sama sekali (nomor tabel di dokumen melompat dari 1.1 ke 4.1),
  jadi tabel ini menjadi **Tabel 3.1** tanpa menggeser nomor tabel mana pun.

Susunan sub-bab: **paragraf pengantar → Tabel 3.1 → paragraf penutup**.

### 3.3 Tahapan Pembangunan Sistem

**[Paragraf pengantar]**

> Pembangunan sistem dilakukan secara bertahap dan berurutan, di mana keluaran satu tahap
> menjadi masukan bagi tahap berikutnya. Tahapan ini disusun mengikuti pendekatan
> eksperimental dengan siklus perbaikan (iteratif), yaitu apabila hasil pengujian pada
> suatu tahap belum memenuhi target yang diharapkan, proses dikembalikan ke tahap
> sebelumnya untuk diperbaiki. Penyusunan tahapan secara berurutan ini bertujuan
> memastikan setiap komponen sistem dibangun di atas rancangan yang telah ditetapkan,
> sehingga proses pengembangan dapat ditelusuri dan diulang kembali. Secara keseluruhan,
> pembangunan sistem terbagi menjadi delapan tahap sebagaimana ditunjukkan pada Tabel 3.1.

**Tabel 3.1 Tahapan Pembangunan Sistem**

Kolom **Aktivitas** menjawab "apa yang dikerjakan", kolom **Keterangan** menjawab "kenapa
begitu". Tidak ada rujukan ke gambar/tabel BAB IV — disengaja, karena penomoran BAB IV
masih akan bergeser saat Tabel 4.2 (spek kamera) ditambahkan.

| No | Tahapan | Aktivitas | Keterangan |
|----|---------|-----------|------------|
| 1 | Studi literatur dan analisis kebutuhan | Mengkaji penelitian terdahulu tentang deteksi kendaraan, ALPR, dan analisis warna HSV; mengidentifikasi kebutuhan pendataan kendaraan listrik di lingkungan kampus | Menjadi dasar penetapan celah penelitian, yaitu klasifikasi tipe energi kendaraan yang belum dibahas pada penelitian terdahulu |
| 2 | Perancangan arsitektur dan pemodelan sistem | Menyusun blok diagram sistem, gambaran sistem, dan flowchart alur program | Perancangan diselesaikan sebelum implementasi agar pembagian modul program mengikuti alur data yang telah ditetapkan |
| 3 | Penyiapan lingkungan dan perangkat | Instalasi Python beserta *virtual environment*, NVIDIA CUDA Toolkit 11.8, PyTorch, Ultralytics YOLOv8, PaddleOCR, dan OpenCV; pemasangan serta pengujian kamera USB | Seluruh pustaka diisolasi di dalam *virtual environment* agar versi dependensi tetap konsisten. Akselerasi GPU diperlukan agar proses inferensi dapat berjalan secara *real-time* |
| 4 | Pengumpulan dan pelabelan dataset plat nomor | Mengumpulkan citra plat nomor kendaraan Indonesia, melakukan anotasi *bounding box* format YOLO, lalu membagi data latih dan data uji dengan perbandingan 80:20 | Pelabelan hanya dilakukan untuk kelas plat nomor, karena deteksi kendaraan menggunakan model bawaan YOLOv8n sesuai batasan masalah |
| 5 | Pelatihan model deteksi plat nomor | Melatih model YOLOv8n selama 100 *epoch* pada ukuran citra 640 piksel, kemudian mengevaluasi hasilnya melalui *confusion matrix*, kurva Precision–Recall, dan kurva F1 | Pelatihan diulang dengan menambah variasi data latih apabila hasil evaluasi dinilai belum memadai |
| 6 | Implementasi modul program deteksi | Membangun *pipeline* lima tahap: deteksi kendaraan dan plat, klasifikasi tipe energi berbasis HSV, pembacaan teks plat, pelacakan kendaraan, dan pengiriman data; ditambah antarmuka pengguna desktop | Modul dibangun secara berurutan mengikuti alur *pipeline*, di mana keluaran satu modul menjadi masukan bagi modul berikutnya |
| 7 | Integrasi dengan web monitoring | Menyepakati kontrak data dengan tim web, mengimplementasikan pengiriman HTTP POST beserta pembaruan hasil pembacaan plat melalui HTTP PATCH, dan menguji koneksi menggunakan server tiruan | Kontrak data disepakati lebih dahulu agar sisi program deteksi dan sisi web dapat dikembangkan secara paralel. Pengujian awal memakai server tiruan agar tidak bergantung pada kesiapan jaringan |
| 8 | Pengujian dan evaluasi sistem | Pengujian akurasi model, kecepatan pemrosesan, *confidence score*, jarak deteksi, intensitas cahaya, dan sudut pembacaan plat | Hasil pengujian digunakan kembali untuk menyetel parameter sistem, sehingga tahap ini bersifat iteratif terhadap tahap-tahap sebelumnya |

**[Paragraf penutup]**

> Berdasarkan Tabel 3.1, tahap 1 dan tahap 2 merupakan tahap perencanaan yang menghasilkan
> rancangan sistem, sedangkan tahap 3 sampai tahap 7 merupakan tahap realisasi yang
> mewujudkan rancangan tersebut menjadi program yang berfungsi. Tahap 8 berperan sebagai
> tahap evaluasi sekaligus penghubung kembali ke tahap sebelumnya, karena hasil pengujian
> digunakan untuk menyetel ulang parameter sistem. Sebagai contoh, hasil analisis kurva F1
> pada tahap pengujian menjadi dasar penetapan nilai ambang keyakinan yang diterapkan
> kembali pada program deteksi. Dengan demikian, tahap 4 sampai tahap 8 tidak berjalan satu
> arah, melainkan berulang hingga sistem mencapai kinerja yang memadai. Realisasi tahap 3
> sampai tahap 8 diuraikan secara rinci pada Bab IV.

**Dua hal yang harus kamu isi/cek sendiri:**

1. **Tahap 4 — jumlah dan sumber dataset belum ada.** Folder `dataset/` di repo ini kosong
   (hanya template), jadi pelatihan memang dilakukan di notebook terpisah sesuai sub-bab
   4.1.7. Isi berapa total citra dan dari mana sumbernya. Angka yang sudah kamu punya hanya
   sisi ujinya: 274 gambar berisi 397 plat.
2. **Tahap 5 — soal augmentasi.** Klausa augmentasi ("tanpa pembalikan vertikal, rotasi
   maksimum 5°") sudah saya **hapus** dari tabel, karena nilai itu berasal dari
   `train_plate.py` di repo — dan skrip itu bukan yang kamu pakai (defaultnya YOLOv8s @
   1280 px, sedangkan laporanmu YOLOv8n @ 640 px). Tambahkan kembali hanya jika notebook-mu
   memang mengatur augmentasi secara khusus dan bisa kamu tunjukkan.

**Alternatif bentuk tabel:** kalau 4 kolom terlalu padat untuk margin A4, gabungkan Aktivitas
dan Keterangan menjadi satu kolom **Keterangan** (kalimat aktivitas, lalu kalimat alasan).
Lebih lega dibaca, tetapi kehilangan ketegasan visual antara "apa" dan "kenapa".

---

## REVISI 2 — Spesifikasi Kamera

**Penempatan (sudah diputuskan):** spesifikasi kamera **dimasukkan ke dalam Tabel 4.1 yang
sudah ada**, bukan tabel baru. Konsekuensinya nol — tidak ada nomor tabel BAB IV yang
bergeser.

Spesifikasi kamera dari user (Agustus 2026): Camtech QHD Webcam, resolusi 2K 1440p
(4 megapiksel), lensa sudut lebar 106°, mikrofon internal.

Spesifikasi laptop diverifikasi langsung dari mesin (`Win32_ComputerSystem`,
`Win32_VideoController`, `Win32_Processor`): ASUS TUF Gaming F15 FX506HC, i5-11400H
6 core / 12 thread, RTX 3050 Laptop GPU 4 GB, RAM 16 GB. Cocok dengan yang tertulis di
laporan.

**Tabel 4.1 Lingkungan Sistem (Environment) — versi revisi**

| Komponen Sistem | Spesifikasi / Platform Terpasang |
|---|---|
| Laptop | ASUS TUF Gaming F15 FX506HC |
| Prosesor | Intel Core i5-11400H (6 *core* / 12 *thread*, 2,70 GHz) |
| Kartu Grafis (GPU) | NVIDIA GeForce RTX 3050 Laptop GPU (VRAM 4 GB) |
| Memori (RAM) | 16 GB |
| Sistem Operasi | Windows 11 |
| Perangkat Input Gambar | Camtech QHD Webcam (kamera web USB) |
| — Resolusi | 2K 1440p (4 megapiksel) |
| — Lensa | Sudut lebar 106° |
| — Fitur tambahan | Mikrofon internal (tidak digunakan dalam penelitian ini) |

Tiga perubahan terhadap Tabel 4.1 yang lama:

1. **Baris "Akselerasi Grafis | NVIDIA CUDA Toolkit 11.8" diganti menjadi GPU.** CUDA Toolkit
   adalah perangkat lunak dan **sudah tercantum di Tabel 4.2** (Perangkat Lunak Modul
   Deteksi), jadi sebelumnya dobel. Sekarang Tabel 4.1 murni perangkat keras, Tabel 4.2
   murni perangkat lunak.
2. **GPU dan RAM ditambahkan** — sebelumnya tidak ada sama sekali, padahal batasan masalah
   menyebut *"hanya mengandalkan satu unit komputer/laptop dengan GPU NVIDIA"*.
3. **Mikrofon diberi keterangan "tidak digunakan"** agar tidak ditanya kenapa sistem visi
   komputer memerlukan mikrofon.

Alternatif ringkas (satu baris, kalau tidak mau bertingkat):
`Camtech QHD Webcam — resolusi 2K 1440p (4 MP), lensa sudut lebar 106°, mikrofon internal`.
Kurang disarankan: penguji menanyakan spek kamera secara khusus, kalau dijejalkan ke satu
sel terkesan tempelan.

### Tambahan untuk sub-bab 4.2.4 (Pengujian Jarak)

Lensa 106° adalah penjelasan teknis kenapa teks plat hanya terbaca sampai 3 m. Sisipkan:

> Keterbatasan jarak pembacaan teks juga dipengaruhi oleh karakteristik lensa kamera yang
> digunakan, yaitu lensa sudut lebar 106°. Lensa sudut lebar memberikan cakupan area
> pemantauan yang luas, tetapi menyebabkan objek tampak lebih kecil di dalam bingkai pada
> jarak yang sama dibandingkan lensa bersudut normal, sehingga jumlah piksel yang tersedia
> untuk setiap karakter plat berkurang lebih cepat seiring bertambahnya jarak.

Ini mengubah "OCR cuma sampai 3 m" dari kelemahan sistem menjadi konsekuensi pilihan
perangkat yang dipahami.

### ⚠️ Catatan: program tidak memaksa resolusi kamera

Di [main.py:254-267](src/main.py#L254-L267) yang diatur hanya `CAP_PROP_FPS`;
`CAP_PROP_FRAME_WIDTH`/`HEIGHT` tidak pernah di-*set*. Artinya resolusi yang benar-benar
masuk ke YOLO adalah **resolusi bawaan penggerak kamera**, bukan otomatis 1440p. Kalau
penguji bertanya "QHD-nya kepakai atau tidak", jawaban jujurnya: kepakai hanya jika
penggerak kamera memang membuka pada resolusi itu secara bawaan.

Untuk memastikan, colok kamera Camtech lalu jalankan:

```powershell
venv\Scripts\python.exe -c "import cv2; c=cv2.VideoCapture(0,cv2.CAP_DSHOW); print('default',c.get(3),'x',c.get(4),'fps',c.get(5)); c.set(3,2560); c.set(4,1440); print('maks',c.get(3),'x',c.get(4)); c.release()"
```

Kalau hasilnya bukan 2560 × 1440, resolusi efektif sistem lebih rendah dari spesifikasi
kamera — dan itu boleh saja ditulis apa adanya. Kalau mau dipaksa ke 1440p (untuk
memperjauh jarak baca plat), itu perubahan kecil di `open_source()`: tambah opsi
`--cam-width` / `--cam-height`. **Tapi kalau pengujian di laporan sudah selesai pada
kondisi sekarang, jangan diubah** — mengubah resolusi membuat seluruh angka FPS dan jarak
di BAB IV tidak lagi sahih.

---

## REVISI 3 — Penjelasan Tracking

Penguji menanyakan ini karena *tracking* muncul di blok diagram (Gambar 3.1) dan di
kesimpulan, tetapi **tidak punya dasar teori di BAB II maupun sub-bab implementasi di
BAB IV**, padahal modul lain (deteksi, klasifikasi, OCR, pengiriman) punya keduanya.
Ada dua tambahan.

**BAB III tidak perlu diubah** — tracking sudah dijelaskan di level arsitektur pada
Gambar 3.1 (blok "Tracker Anti-spam") dan flowchart Gambar 3.3 (validasi pelacakan).

**Rencana penomoran sub-bab 4.1 (sekalian membetulkan 4.1.3/4.1.4 yang tertukar):**

| Sekarang | Jadi | Tindakan |
|---|---|---|
| 4.1.1 Lingkungan Sistem | 4.1.1 | tetap |
| 4.1.2 Antarmuka Pengguna Desktop | 4.1.2 | tetap |
| **4.1.4** Klasifikasi Tipe Energi | 4.1.4 | **pindahkan ke bawah** |
| **4.1.3** Deteksi Kendaraan | 4.1.3 | **pindahkan ke atas** |
| 4.1.5 Pembacaan Teks Plat Nomor | 4.1.5 | tetap |
| — | **4.1.6 Pelacakan Kendaraan (Tracking)** | **baru** |
| 4.1.6 Pengiriman Data ke Web Monitoring | 4.1.7 | ketik ulang nomor |
| 4.1.7 Flowchart Pelatihan Model YOLOv8 | 4.1.8 | ketik ulang nomor |

Label "4.1.3" dan "4.1.4" sebenarnya sudah benar — yang salah hanya urutan bloknya, jadi
cukup dipindah tanpa mengetik ulang nomor. Nomor gambar di caption memakai *field* SEQ
otomatis (terverifikasi dari `word/document.xml`: 15 SEQ Gambar, 12 SEQ Tabel), jadi
Gambar 4.2–4.5 menyesuaikan sendiri setelah `Ctrl+A` lalu `F9`. Sudah dicek: tidak ada
kalimat di badan teks yang menyebut "Gambar 4.1"–"Gambar 4.9", jadi tidak ada rujukan
yang menjadi salah.

### (a) BAB II — sub-bab baru 2.2.7 Object Tracking

*(sisipkan setelah 2.2.6 Analisis Warna Berbasis Ruang Warna HSV, sebelum BAB III —
posisi terakhir sehingga tidak ada nomor yang perlu digeser)*

> *Object tracking* adalah teknik dalam *computer vision* yang bertujuan mengenali objek
> yang sama pada rangkaian bingkai video yang berurutan, sehingga objek tersebut memiliki
> identitas (ID) yang konsisten selama berada di dalam jangkauan kamera. Berbeda dengan
> *object detection* yang bekerja secara independen pada setiap bingkai, *object tracking*
> menghubungkan hasil deteksi antarbingkai sehingga sistem dapat membedakan "satu kendaraan
> yang terlihat pada banyak bingkai" dari "banyak kendaraan yang berbeda". Kemampuan ini
> menjadi kebutuhan mendasar pada sistem pendataan kendaraan otomatis, karena tanpa
> pelacakan, satu kendaraan yang melintas selama beberapa detik akan tercatat berkali-kali
> sebagai data yang berbeda. Pemanfaatan pelacakan objek untuk kebutuhan survei lalu lintas
> telah ditunjukkan oleh Bihanda dkk. [13] yang memadukan model deteksi dengan algoritma
> ByteTrack untuk menghitung kendaraan secara otomatis.
>
> Salah satu pendekatan yang umum digunakan dalam pelacakan adalah penugasan (*assignment*)
> antara daftar objek yang sedang dilacak dan daftar hasil deteksi pada bingkai terbaru.
> Penugasan tersebut dapat dilakukan secara global menggunakan algoritma Hungarian
> (*Hungarian algorithm*), yaitu metode optimasi yang mencari pasangan dengan total biaya
> paling kecil atas seluruh kemungkinan pasangan sekaligus, bukan memilih pasangan terbaik
> satu per satu. Pendekatan global ini lebih tahan terhadap kesalahan penukaran identitas
> (*ID switch*) ketika dua objek saling berdekatan atau saling menutupi.

### (b) BAB IV — sub-bab baru 4.1.6 Pelacakan Kendaraan (Tracking)

*(sisipkan setelah 4.1.5 Pembacaan Teks Plat Nomor, sebelum sub-bab Pengiriman Data;
nomor sub-bab berikutnya digeser)*

> Tahap ini bertugas mengenali kendaraan yang sama pada bingkai-bingkai video yang
> berurutan, sehingga satu kendaraan hanya dikirim datanya satu kali ke web monitoring.
> Modul ini diwujudkan pada berkas `vehicle_tracker.py`.
>
> **Cara kerja umum.** Setiap kendaraan yang terdeteksi diberi sebuah catatan pelacakan
> (*track*) yang menyimpan identitas, posisi kotak deteksi, hasil pembacaan plat, tipe
> energi, dan foto terbaik kendaraan tersebut. Pada setiap bingkai baru, sistem
> mencocokkan seluruh hasil deteksi terhadap seluruh catatan pelacakan yang sedang aktif.
> Apabila cocok, catatan diperbarui; apabila tidak ada yang cocok, sistem menganggapnya
> sebagai kendaraan baru dan membuat catatan baru. Catatan yang tidak terlihat lagi selama
> 30 bingkai berturut-turut akan dihapus.
>
> **Cara mencocokkan kendaraan.** Pencocokan tidak dilakukan satu per satu berdasarkan
> tumpang tindih kotak saja, melainkan menggunakan penugasan global dengan algoritma
> Hungarian, yaitu sistem mencari kombinasi pasangan yang totalnya paling masuk akal untuk
> seluruh kendaraan sekaligus. Cara ini dipilih karena pencocokan satu per satu terbukti
> mudah menukar identitas ketika dua kendaraan saling bertumpuk — dan identitas yang
> tertukar berarti foto yang dikirim ke web adalah foto kendaraan yang salah. Kemiripan
> antara sebuah catatan pelacakan dan sebuah hasil deteksi dihitung dari tiga hal dengan
> bobot yang ditunjukkan pada Tabel 4.x.
>
> **Tabel 4.x Bobot dan Parameter Pencocokan pada Modul Pelacakan**
>
> | Parameter | Nilai | Keterangan |
> |-----------|-------|------------|
> | Bobot tumpang tindih kotak (IoU) | 0,60 | Seberapa besar kotak deteksi berimpit dengan kotak prediksi |
> | Bobot jarak titik pusat | 0,25 | Jarak pusat kotak, dinormalisasi terhadap diagonal kotak |
> | Bobot kemiripan bentuk | 0,15 | Kemiripan ukuran dan rasio lebar–tinggi kotak |
> | Ambang IoU minimum | 0,30 | Batas tumpang tindih agar langsung dianggap kendaraan yang sama |
> | Jarak pusat maksimum | 1,2 × diagonal kotak | Jalur cadangan saat kendaraan sesaat tertutup (IoU = 0) |
> | Biaya gabungan maksimum | 0,80 | Di atas nilai ini pasangan ditolak |
> | Batas penyelamatan lewat prediksi | 10 bingkai | Setelah itu pencocokan wajib benar-benar bertumpuk |
> | Batas waktu catatan pelacakan | 30 bingkai | Catatan dihapus bila tidak terlihat selama ini |
>
> Kendaraan dengan jenis berbeda (mobil dan motor) tidak pernah dipasangkan satu sama lain.
> Selain itu, posisi kotak setiap catatan pelacakan diperkirakan terlebih dahulu
> berdasarkan arah dan kecepatan geraknya pada dua bingkai terakhir, sehingga kendaraan
> yang sesaat tertutup kendaraan lain tetap dapat dikenali kembali sebagai kendaraan yang
> sama ketika muncul lagi.
>
> **Syarat pengiriman data (anti-spam).** Data sebuah kendaraan baru dikirim ke web
> monitoring apabila seluruh syarat pada Tabel 4.y terpenuhi. Setelah terkirim, catatan
> pelacakan ditandai sudah terkirim sehingga kendaraan yang sama tidak dikirim ulang.
>
> **Tabel 4.y Syarat Pengiriman Data Kendaraan**
>
> | Syarat | Nilai | Alasan |
> |--------|-------|--------|
> | Jumlah bingkai minimum terlihat | 5 bingkai | Mencegah deteksi sekejap (*flicker*) ikut terkirim |
> | Tinggi kotak kendaraan minimum | 100 piksel | Kendaraan yang terlalu jauh menghasilkan foto dan plat yang tidak terbaca |
> | Tipe energi | Harus sudah diketahui | Data tanpa tipe energi tidak berguna bagi tujuan penelitian |
> | Teks plat nomor | Tidak boleh kosong | Web tidak pernah menerima baris data tanpa nomor plat |
> | Kestabilan bacaan plat | 12 bingkai berturut-turut sama | Bukti bahwa OCR sudah membaca ulang dan hasilnya konsisten |
> | Batas waktu menunggu plat | 20 bingkai | Agar kendaraan yang melintas cepat tetap terkirim |
>
> **Pemilihan foto yang dikirim.** Foto yang dikirim bukan foto sembarang bingkai,
> melainkan bingkai dengan kotak kendaraan paling besar (paling dekat ke kamera) dari
> bingkai-bingkai yang "bersih", yaitu bingkai di mana kotak kendaraan tersebut tidak
> bertumpuk berat dengan kotak kendaraan lain (tumpang tindih di atas 0,5 dianggap kotor).
> Bingkai kotor hanya digunakan apabila kendaraan tersebut tidak pernah sekalipun terlihat
> dalam kondisi bersih. Foto kemudian diperkecil hingga sisi terpanjangnya maksimal 800
> piksel dan disimpan dalam format JPEG dengan kualitas 85.
>
> **Pencegahan data ganda antarcatatan.** Selain pencocokan antarbingkai, sistem juga
> menyimpan ingatan nomor plat yang sudah pernah dikirim selama 60 detik terakhir. Apabila
> sebuah kendaraan sempat hilang cukup lama lalu terdeteksi kembali sebagai kendaraan baru,
> nomor platnya akan dikenali sebagai plat yang sudah pernah dikirim sehingga sistem tidak
> membuat data baru, melainkan menempel pada data yang sudah ada.
>
> **Pembaruan hasil pembacaan plat.** Karena hasil OCR umumnya membaik ketika kendaraan
> semakin dekat, hasil pembacaan plat yang lebih baik dapat muncul setelah data terlanjur
> dikirim. Dalam kondisi tersebut sistem tidak membuat data baru, melainkan mengirim
> pembaruan ke server menggunakan metode HTTP PATCH agar nomor plat pada data yang sudah
> tersimpan diperbaiki. Pembaruan hanya dilakukan apabila hasil bacaan yang baru benar-benar
> lebih baik, yaitu jumlah digitnya tidak berkurang dan teks tersebut belum pernah dikirim
> sebelumnya untuk data yang sama.

---

## REVISI 4 — Nilai HSV yang Dipakai

**Penempatan (sudah diputuskan):** dasar teorinya sudah ada di **2.2.6 Analisis Warna
Berbasis Ruang Warna HSV** — yang hilang hanya angkanya. Jadi tidak perlu sub-bab baru;
sisipkan ke sub-bab **4.1.4 Klasifikasi Tipe Energi Kendaraan** yang sudah ada.

Titik sisip persisnya: **setelah** paragraf *"Penentuan tipe energi didasarkan pada ciri
khas plat nomor sesuai aturan Peraturan Polri No. 7 Tahun 2021…"* dan **sebelum** paragraf
*"Agar hasilnya tidak keliru, sistem dirancang untuk membedakan garis biru asli…"*.
Alasannya: paragraf sebelumnya menyatakan prinsipnya, paragraf sesudahnya menyatakan
pengamanannya — tabel angka duduk pas di tengah, dan paragraf terakhir tinggal diperluas
dengan angka 1,1 / 75 / 95 sehingga kalimat kualitatif yang ada sekarang punya dasar.

Sumber angka: [fuel_classifier.py:14-22](src/fuel_classifier.py#L14-L22) dan
[main.py:98-110](src/main.py#L98-L110).

> Pemeriksaan warna dilakukan pada ruang warna HSV, yaitu ruang warna yang memisahkan jenis
> warna (*Hue*), tingkat kejenuhan (*Saturation*), dan tingkat kecerahan (*Value*). Ruang
> warna ini dipilih karena lebih tahan terhadap perubahan pencahayaan di luar ruangan
> dibandingkan ruang warna RGB. Nilai parameter yang digunakan sistem ditunjukkan pada
> Tabel 4.3. Perlu dicatat bahwa pustaka OpenCV yang digunakan menerapkan skala *Hue*
> 0–179, bukan 0–359 seperti pada skala derajat pada umumnya, sedangkan *Saturation* dan
> *Value* menggunakan skala 0–255.
>
> **Tabel 4.3 Parameter Analisis Warna HSV pada Klasifikasi Tipe Energi**
>
> | Parameter | Nilai | Keterangan |
> |-----------|-------|------------|
> | Hue (H) | 95 – 135 | Rentang warna biru pada skala OpenCV 0–179 (setara 190°–270° pada skala 0–359) |
> | Saturation (S) | 35 – 255 | Batas bawah 35 dipilih karena strip biru asli tampak pudar di kamera (S ± 40–90), sedangkan bodi plat putih dan latar belakang berada di bawah 25 |
> | Value (V) | 60 – 255 | Menyingkirkan piksel yang terlalu gelap |
> | Area yang diperiksa | 40% bagian bawah plat | Lokasi strip biru kendaraan listrik |
> | Perpanjangan kotak plat ke bawah | 50% dari tinggi kotak | Model deteksi memotong plat tepat pada baris nomor, sedangkan strip biru berada di bawahnya |
> | Ambang cakupan strip | 0,85 | Minimal 85% lebar plat pada satu baris piksel harus terdeteksi biru |
> | Rasio kejenuhan strip terhadap area teks | 1,1 | Strip harus lebih pekat daripada area teks plat |
> | Median Saturation minimum strip | 75 | Penyaring "biru asli" |
> | Median Value minimum strip | 95 | Penyaring "biru asli" |
>
> Berdasarkan Tabel 4.3, klasifikasi tidak ditentukan semata-mata oleh banyaknya piksel
> biru, melainkan melalui empat langkah pemeriksaan berurutan:
>
> 1. **Pemeriksaan rentang warna.** Seluruh piksel pada area plat yang diperpanjang diubah
>    ke ruang warna HSV, kemudian ditandai piksel yang nilainya berada di dalam rentang
>    H 95–135, S 35–255, dan V 60–255.
> 2. **Pemeriksaan bentuk strip.** Sistem tidak memakai jumlah total piksel biru,
>    melainkan mencari baris piksel yang paling biru dan mengukur berapa persen lebar plat
>    yang tertutup warna biru pada baris tersebut. Nilai ini harus mencapai minimal 0,85.
>    Cara ini dipakai karena strip kendaraan listrik berbentuk garis mendatar selebar plat,
>    sedangkan sumber biru lain (huruf gelap, tepi plat) hanya biru pada sebagian kecil
>    lebar sehingga tidak lolos.
> 3. **Pemeriksaan anti-semburat warna kamera.** Apabila area teks plat di bagian atas juga
>    terbaca biru dan bagian bawah tidak lebih pekat setidaknya 1,1 kali dibanding bagian
>    atas, kondisi itu dianggap sebagai semburat warna dari pengaturan *white balance*
>    kamera, bukan strip kendaraan listrik.
> 4. **Pemeriksaan biru asli.** Nilai tengah (median) *Saturation* piksel biru pada zona
>    strip harus minimal 75 dan nilai tengah *Value*-nya minimal 95. Pemeriksaan ini
>    ditetapkan berdasarkan kalibrasi terhadap 84 citra hasil deteksi berlabel, yang terdiri
>    atas 11 plat kendaraan listrik asli dan 73 kesalahan deteksi. Pada plat listrik asli,
>    median *Saturation* terukur 148 dan median *Value* terukur 181, sedangkan pada
>    kesalahan deteksi median *Saturation*-nya hanya sekitar 60.
>
> Kendaraan dikelompokkan sebagai listrik hanya apabila keempat langkah tersebut terpenuhi.
> Apabila salah satu langkah tidak terpenuhi, kendaraan dikelompokkan sebagai konvensional.
> Ketiga pemeriksaan tambahan disusun setelah ditemukan bahwa pemeriksaan rentang warna saja
> menghasilkan kesalahan klasifikasi yang tinggi, terutama pada plat berwarna gelap yang
> ternaungi dan plat putih yang terkena silau cahaya. Seluruh nilai ambang tersebut dapat
> diubah melalui parameter program `--fuel-coverage`, `--fuel-smin`, dan `--fuel-vmin`
> tanpa mengubah kode.

### Penomoran tabel BAB IV setelah penambahan

Tiga tabel baru (1 di 4.1.4, 2 di 4.1.6) menggeser seluruh tabel pengujian:

| Baru | Isi | Sebelumnya |
|---|---|---|
| Tabel 4.1 | Lingkungan Sistem (sudah termasuk spek kamera) | 4.1 |
| Tabel 4.2 | Perangkat Lunak Modul Deteksi | 4.2 |
| **Tabel 4.3** | **Parameter Analisis Warna HSV** | baru |
| **Tabel 4.4** | **Bobot dan Parameter Pencocokan Pelacakan** | baru |
| **Tabel 4.5** | **Syarat Pengiriman Data Kendaraan** | baru |
| Tabel 4.6 | Pengujian Akurasi Model Deteksi Plat | 4.3 |
| Tabel 4.7 | Pengujian Kecepatan Pemrosesan (FPS) | 4.4 |
| Tabel 4.8 | Pengujian Confidence Score | 4.5 |
| Tabel 4.9 | Pengujian Jarak — mobil | 4.6 |
| Tabel 4.10 | Pengujian Jarak — motor | 4.7 |
| Tabel 4.11 | Pengujian Intensitas Cahaya | 4.8 |
| Tabel 4.12 | Pengujian Sudut Pembacaan Teks Plat | 4.9 |

Caption dan Daftar Tabel memakai *field* SEQ otomatis → benar sendiri setelah `Ctrl+A`
lalu `F9`. Yang manual hanya rujukan di dalam kalimat:

| Sub-bab | Kalimat | Tertulis | Seharusnya |
|---|---|---|---|
| 4.2.1 | "Dari **Tabel 4.4**, kinerja model terlihat seimbang…" | ⚠️ 4.4 | 4.6 |
| 4.2.1 | "…menghitung ulang angka pada **Tabel 4.2**" | ⚠️ 4.2 | 4.6 |
| 4.2.2 | "Hasil pengujian disajikan pada **Tabel 4.4**." | 4.4 | 4.7 |
| 4.2.2 | "Berdasarkan **Tabel 4.4**, kecepatan sistem menurun…" | 4.4 | 4.7 |
| 4.2.2 | "…nilai FPS pada **Tabel 4.3**" | ⚠️ 4.3 | 4.7 |
| 4.2.3 | "Hasil pengujian disajikan pada **Tabel 4.6**." | ⚠️ 4.6 | 4.8 |
| 4.2.3 | "Berdasarkan **Tabel 4.4**, mobil memperoleh…" | ⚠️ 4.4 | 4.8 |
| 4.2.4 | "Hasil pengujian disajikan pada **Tabel 4.7**." | ⚠️ 4.7 | 4.9 **dan Tabel 4.10** |
| 4.2.5 | "Hasil pengukuran disajikan pada **Tabel 4.8**." | 4.8 | 4.11 |
| 4.2.5 | "Berdasarkan **Tabel 4.8**, pengujian dilakukan…" | 4.8 | 4.11 |
| 4.2.6 | "Hasil pengujian disajikan pada **Tabel 4.9**." | 4.9 | 4.12 |
| 4.2.6 | "Berdasarkan **Tabel 4.9**, sistem berhasil membaca…" | 4.9 | 4.12 |

⚠️ = sudah salah sejak sekarang, terlepas dari pergeseran. Dari 12 titik, 7 memang sudah
salah — pergeseran ini hanya menambah 5 perbaikan baru. Disarankan pakai *Referensi Silang*
Word agar masalah ini tertutup permanen.

Catatan: rujukan di 4.2.4 menyebut **satu** tabel padahal tabelnya **dua** (mobil dan
motor) — sekalian dibetulkan.

**Alternatif tanpa pergeseran:** tulis nilai HSV sebagai kalimat berpoin di dalam paragraf,
bukan tabel. Nol pergeseran nomor, tetapi 8 parameter dalam bentuk kalimat berat dibaca dan
terkesan disebut sambil lalu — padahal penguji secara eksplisit menanyakan nilainya.

---

## REVISI 5 — Pengolahan Data Hasil Monitoring 3 Agustus 2026

**Penempatan (sudah diputuskan):** sub-bab baru **4.2.7 Pengujian Operasional Sistem**, di
akhir 4.2 setelah 4.2.6 Pengujian Sudut Pembacaan. Diletakkan sebagai anak 4.2 (bukan 4.3)
karena ini tetap sebuah pengujian; naik ke 4.3 hanya perlu jika isinya dipecah bersubjudul,
yang akan memaksa *heading* level 4 (4.2.7.1). Karena posisinya paling akhir, tidak ada
nomor tabel BAB IV yang bergeser — tabel baru langsung menyambung: **Tabel 4.13** dan
seterusnya.

**Sumber data:** 208 berkas JSON pendamping di `output/captures/` bertanggal 20260803
(masing-masing berpasangan dengan satu foto JPG, total 416 berkas). Seluruh angka di bawah
dihitung langsung dari berkas tersebut, bukan disalin dari tampilan web.

### Beda sub-bab ini dengan 4.2.1–4.2.6

Harus dinyatakan eksplisit di paragraf pembuka, karena inilah yang akan ditanya: 4.2.1–4.2.6
adalah pengujian **terkendali** (satu variabel diubah, sisanya ditahan), sedangkan 4.2.7
adalah pengujian **operasional** — sistem dibiarkan berjalan menerus pada lalu lintas
sebenarnya tanpa variabel yang dikendalikan. Tujuannya berbeda: bukan mengukur batas
kemampuan tiap modul, melainkan membuktikan seluruh rantai (deteksi → klasifikasi → OCR →
pelacakan → pengiriman) bekerja utuh dalam pemakaian nyata.

### 4.2.7 Pengujian Operasional Sistem

**[Paragraf pengantar]**

> Pengujian operasional dilakukan untuk mengetahui kinerja sistem secara menyeluruh pada
> kondisi pemakaian sebenarnya. Berbeda dengan pengujian pada sub-bab sebelumnya yang
> mengubah satu parameter uji secara terkendali, pada pengujian ini sistem dibiarkan
> berjalan secara menerus terhadap arus lalu lintas yang sesungguhnya tanpa pengaturan
> khusus, sehingga hasil yang diperoleh mencerminkan kemampuan seluruh rantai proses mulai
> dari deteksi kendaraan hingga penyimpanan data pada web monitoring. Pengujian
> dilaksanakan pada hari Senin, 3 Agustus 2026, pukul 09.09 sampai 11.32 WIB, bertempat di
> **[ISI: lokasi]**, dengan kamera dipasang **[ISI: posisi/ketinggian/jarak ke lajur]** dan
> kondisi cuaca **[ISI]**. Program dijalankan menggunakan seluruh nilai bawaan sebagaimana
> tercantum pada Tabel 4.2 dan terhubung ke web monitoring melalui jaringan lokal. Ringkasan
> hasil pengujian ditunjukkan pada Tabel 4.13.

**Tabel 4.13 Ringkasan Hasil Pengujian Operasional Sistem**

| Aspek yang Diamati | Hasil |
|---|---|
| Durasi pengujian | 2 jam 23 menit (09.09–11.32 WIB) |
| Jumlah kendaraan terdata | 208 kendaraan |
| Laju pendataan rata-rata | 87 kendaraan/jam |
| Tipe kendaraan terdeteksi | Mobil 208 (100%), motor 0 (0%) |
| Tipe energi terklasifikasi | Konvensional 206 (99,04%), listrik 2 (0,96%) |
| Data dengan nomor plat terbaca | 208 dari 208 (100%) |
| Nomor plat sesuai pola plat Indonesia | 208 dari 208 (100%) |
| Nomor plat unik | 205 nomor (3 nomor terekam dua kali) |
| *Confidence score* deteksi kendaraan | Terendah 0,40; rata-rata 0,61; tertinggi 0,95 |
| Jeda antara waktu deteksi dan waktu kirim | Median 3 detik; terlama 16 detik |
| Ukuran rata-rata foto terkirim | 25 KB |

**[Paragraf pembahasan Tabel 4.13]**

> Berdasarkan Tabel 4.13, sistem berhasil mendata 208 kendaraan selama 2 jam 23 menit, atau
> rata-rata 87 kendaraan per jam, dan seluruhnya terkirim ke web monitoring dengan jeda
> pengiriman median 3 detik. Jeda tersebut berasal dari mekanisme antrean pengiriman yang
> berjalan pada utas terpisah, sehingga proses pengiriman tidak menghambat pemrosesan
> bingkai berikutnya. Seluruh data yang terkirim memuat nomor plat dan seluruhnya sesuai
> pola penomoran plat Indonesia; hal ini merupakan akibat langsung dari penyaringan pola
> pada modul pelacakan, yang tidak meloloskan hasil pembacaan di luar pola tersebut untuk
> dikirim ke web monitoring. Perlu ditegaskan bahwa angka 100% pada baris tersebut
> menyatakan seluruh data memiliki nomor plat yang terbaca dan berpola benar, bukan
> menyatakan seluruh nomor plat terbaca dengan tepat; ketepatan pembacaan diuji secara
> terpisah melalui verifikasi sampel pada Tabel 4.15.

**Tabel 4.14 Sebaran *Confidence Score* pada Pengujian Operasional**

| Selang *Confidence Score* | Jumlah Kendaraan | Persentase |
|---|---|---|
| 0,40 – 0,49 | 78 | 37,5% |
| 0,50 – 0,59 | 40 | 19,2% |
| 0,60 – 0,69 | 27 | 13,0% |
| 0,70 – 0,79 | 18 | 8,7% |
| 0,80 – 0,89 | 22 | 10,6% |
| 0,90 – 1,00 | 23 | 11,1% |
| **Jumlah** | **208** | **100%** |

**[Paragraf pembahasan Tabel 4.14]**

> Berdasarkan Tabel 4.14, sebaran *confidence score* terpusat pada selang terendah, yaitu
> 37,5% kendaraan berada pada selang 0,40–0,49, sedangkan kendaraan dengan keyakinan di atas
> 0,80 hanya 21,7%. Pola ini sejalan dengan kondisi pengujian, karena kendaraan mulai
> terdeteksi sejak masih berjarak jauh sehingga sebagian besar data terkumpul pada saat
> ukuran objek di dalam bingkai masih kecil. Sebanyak 55 dari 208 kendaraan (26,4%) memiliki
> keyakinan di bawah 0,46, yang berarti kendaraan-kendaraan tersebut tidak akan terdata
> apabila ambang keyakinan dinaikkan ke nilai 0,46 sebagaimana hasil analisis kurva F1 pada
> sub-bab 4.2.1. Hal ini menunjukkan adanya pertukaran (*trade-off*) antara ketelitian dan
> kelengkapan pendataan: ambang yang lebih tinggi menghasilkan deteksi yang lebih meyakinkan,
> tetapi mengurangi jumlah kendaraan yang berhasil didata.

**Tabel 4.15 Hasil Verifikasi Ketepatan Pembacaan Nomor Plat** *(WAJIB DIISI — lihat catatan
di bawah)*

| Aspek | Jumlah | Persentase |
|---|---|---|
| Sampel diverifikasi | 52 | 100% |
| Nomor plat terbaca tepat seluruh karakter | **[ISI]** | **[ISI]%** |
| Nomor plat terbaca sebagian (1–2 karakter salah) | **[ISI]** | **[ISI]%** |
| Nomor plat terbaca salah | **[ISI]** | **[ISI]%** |
| Tipe energi terklasifikasi tepat | **[ISI]** | **[ISI]%** |

**[Paragraf pengantar Tabel 4.15]**

> Ketepatan pembacaan nomor plat tidak dapat dinilai dari data keluaran sistem itu sendiri,
> sehingga dilakukan verifikasi manual dengan membandingkan nomor plat hasil pembacaan
> sistem terhadap nomor plat yang terlihat pada foto yang tersimpan. Verifikasi dilakukan
> terhadap sampel sistematis, yaitu setiap kendaraan ke-4 dari keseluruhan data yang
> terurut berdasarkan waktu deteksi, sehingga diperoleh 52 sampel yang tersebar merata
> sepanjang durasi pengujian. Hasil verifikasi ditunjukkan pada Tabel 4.15.

**[Paragraf temuan data ganda]**

> Dari 208 data yang tercatat, terdapat 205 nomor plat unik dan 3 nomor plat yang terekam
> dua kali, yaitu B 392 EB (pukul 09.13 dan 11.19), D 1051 AMM (pukul 09.53 dan 10.07),
> serta D 1024 AER (pukul 10.20 dan 10.43). Selisih waktu antarpasangan tersebut berkisar
> 14 menit hingga 2 jam 6 menit, jauh melampaui jendela pencegahan data ganda yang
> ditetapkan selama 60 detik. Dengan demikian ketiga pasangan data tersebut bukan merupakan
> kegagalan mekanisme pencegahan data ganda, melainkan kendaraan yang memang melintas dua
> kali dalam rentang waktu pengujian. Tidak ditemukan data ganda yang berasal dari satu
> kendaraan yang sama pada satu kali lintasan.

**[Paragraf keterbatasan — JANGAN dihapus, ini yang menyelamatkanmu saat ditanya]**

> Pengujian operasional ini memiliki beberapa keterbatasan yang perlu dinyatakan. Pertama,
> seluruh kendaraan yang terdata berjenis mobil dan tidak ada satu pun sepeda motor yang
> terdata, sehingga kemampuan sistem terhadap sepeda motor pada kondisi operasional belum
> terwakili. Kedua, dari 208 kendaraan hanya 2 kendaraan yang terklasifikasi sebagai
> kendaraan listrik, sehingga jumlah tersebut terlalu sedikit untuk dijadikan dasar
> pengukuran ketelitian klasifikasi tipe energi; pengukuran ketelitian klasifikasi tetap
> mengacu pada pengujian terkendali pada sub-bab sebelumnya. Kondisi ini sekaligus
> menggambarkan keadaan sebenarnya di lokasi pengujian, yaitu populasi kendaraan listrik
> yang masih sangat sedikit dibandingkan kendaraan konvensional. Ketiga, pengujian
> dilakukan pada satu lokasi, satu rentang waktu pagi hari, dan satu kondisi cuaca,
> sehingga hasilnya belum dapat digeneralisasi untuk kondisi malam hari maupun hujan.

**[Paragraf penutup]**

> Berdasarkan keseluruhan hasil pada sub-bab ini, sistem terbukti mampu berjalan secara
> menerus selama lebih dari dua jam tanpa gangguan dan mendata 208 kendaraan beserta nomor
> plat, tipe kendaraan, tipe energi, dan foto masing-masing ke dalam basis data web
> monitoring. Hal ini menunjukkan bahwa seluruh tahapan *pipeline* yang dirancang pada Bab
> III dapat bekerja secara terpadu pada kondisi pemakaian sebenarnya, bukan hanya pada
> pengujian yang terkendali.

### Yang wajib kamu kerjakan sendiri

1. **Isi Tabel 4.15.** Berkas kerjanya sudah dibuat: `docs/verifikasi_sampel_20260803.csv`
   (52 baris, sampel sistematis setiap kendaraan ke-4). Buka di Excel, buka foto yang
   namanya tertera di kolom `berkas_foto` dari `output/captures/`, lalu isi kolom
   `plat_sebenarnya_ISI_MANUAL`, `benar_1_salah_0`, dan
   `tipe_energi_sebenarnya_ISI_MANUAL`. Angka Tabel 4.15 dihitung dari kolom-kolom itu.
   Ini juga yang mengisi *placeholder* **[XX]%** di Abstrak (lihat Temuan Tambahan no. 6).
2. **Isi kurung siku di paragraf pengantar**: lokasi, posisi/ketinggian kamera, jarak kamera
   ke lajur, dan kondisi cuaca. Ini satu-satunya bagian yang tidak bisa diambil dari data.
3. **Betulkan dulu klaim ambang 0,46 di sub-bab 4.2.1 dan 4.2.3** (Temuan Tambahan no. 1)
   sebelum Tabel 4.14 masuk. Tabel 4.14 memuat 78 kendaraan dengan keyakinan 0,40–0,49 —
   itu bukti tertulis bahwa program berjalan pada ambang 0,4, sehingga kalimat *"nilai yang
   sama persis dengan yang dipakai program saat berjalan"* akan terbantah oleh tabel di
   halaman berikutnya. Paragraf pembahasan Tabel 4.14 di atas sudah ditulis dengan
   pengandaian klaim tersebut sudah dibetulkan menjadi "0,46 hasil analisis kurva F1,
   program dijalankan pada 0,4".

### Gambar pendukung

Screenshot hasil cetak laporan (208 entri, filter 3 Agustus, tipe mobil) cocok diletakkan
di sub-bab ini sebagai bukti data benar-benar tersimpan dan terpanggil kembali dari basis
data — **bukan** sebagai bukti kinerja deteksi. Rujuk dari paragraf penutup. Kalau gambar
itu memang milik buku partner web, cukup rujuk silang dan jangan digandakan.

---

## Temuan Tambahan (tidak diminta penguji, tapi rawan ditanya)

Ini saya temukan saat mencocokkan dokumen dengan kode dan dengan tabelnya sendiri.
Beberapa di antaranya kalau ketahuan penguji lebih repot daripada empat revisi di atas.

**Perlu diverifikasi ke kode:**

1. **Ambang keyakinan 0,46 vs 0,4.** Sub-bab 4.2.1 dan 4.2.3 menyatakan nilai 0,46 adalah
   *"nilai yang sama persis dengan yang dipakai program saat berjalan"*. Padahal nilai
   bawaan `--conf` di [main.py:94](src/main.py#L94) adalah **0,4**, bukan 0,46. Dua pilihan:
   (a) jalankan program dengan `--conf 0.46` dan ubah nilai bawaannya di kode, atau
   (b) ubah kalimat di laporan menjadi "0,46 sebagai hasil analisis kurva F1, sedangkan
   program dijalankan pada ambang 0,4". Pilihan (a) lebih rapi.

**Kesalahan rujukan dan penomoran (mudah dibetulkan):**

2. Penomoran sub-bab BAB IV kacau: urutannya **4.1.1 → 4.1.2 → 4.1.4 (Klasifikasi) →
   4.1.3 (Deteksi Kendaraan) → 4.1.5 → 4.1.6 → 4.1.7**. Sub-bab 4.1.4 muncul sebelum 4.1.3.
3. Rujukan tabel salah di beberapa tempat: teks menulis *"Dari Tabel 4.4"* padahal maksudnya
   Tabel 4.3; *"bukan untuk menghitung ulang angka pada Tabel 4.2"* maksudnya Tabel 4.3;
   *"Hasil pengujian disajikan pada Tabel 4.6"* padahal tabelnya Tabel 4.5; *"nilai FPS pada
   Tabel 4.3"* maksudnya Tabel 4.4; *"Berdasarkan Tabel 4.4, mobil memperoleh…"* maksudnya
   Tabel 4.5; *"Hasil pengujian disajikan pada Tabel 4.7"* maksudnya Tabel 4.6.
4. Kalimat terpotong di sub-bab 4.2.1: *"Sementara itu, mAP tidak bergantung pada 4.4."*
   Sepertinya seharusnya *"…tidak bergantung pada ambang keyakinan tersebut."*
5. Kalimat kurang kata di sub-bab 4.1.3: *"Tahap ini bertugas menemukan pada gambar yang
   ditangkap kamera"* → kurang kata "kendaraan".
6. Abstrak dan Abstract masih memuat *placeholder* **[XX]%** untuk akurasi deteksi, akurasi
   klasifikasi, dan FPS. Ini wajib diisi sebelum dikumpulkan.

**Ketidakcocokan data:**

7. **Tabel jarak vs narasinya.** Narasi menyebut mobil terdeteksi hingga 40 m dan motor
   hingga 10 m, tetapi Tabel 4.6 dan Tabel 4.7 hanya memuat data sampai 10 m. Data 15–55 m
   ada di Lampiran (Tabel 6.1 dan 6.2). Sebaiknya narasi merujuk lampirannya secara
   eksplisit, atau tabel di BAB IV digabung dengan data lampiran.
8. **Tabel 4.8 (Lux) tidak konsisten.** Baris "Cahaya siang redup" tercatat maksimum
   **3637 Lux** — lebih tinggi daripada kondisi cerah (2113 Lux) — dengan rata-rata 749 Lux
   yang justru mendekati nilai minimumnya (616). Angkanya perlu dicek ulang; kalau memang
   segitu, perlu ada penjelasan (misalnya ada satu pengukuran saat matahari sempat muncul).
9. Judul berkas dokumen memakai kata **"Bensin"**, sedangkan isi laporan konsisten memakai
   **"Konvensional"**. Samakan.
10. Salah ketik pada judul gambar: *"gambaram sistem Deteksi"* (Gambar 3.2), *"Deteksi
    Knedaraan motor"* (Gambar 4.5), *"Confusion matrik"* (Gambar 4.10), *"kurva
    Precision-Recal"* (Gambar 4.11), *"Cahay siang redup"* (Tabel 4.8).
