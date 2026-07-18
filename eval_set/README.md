# Eval Set

Kumpulan gambar untuk evaluasi pipeline end-to-end. Hasil `evaluate.py` hanya
sevalid dataset ini — kalau eval set kecil atau bias, angka akurasi menipu.

## Target ukuran

- **Minimum**: 200 gambar
- **Sebar kondisi**:
  - 50 siang terang
  - 50 malam / lampu jalan
  - 50 hujan / kontras rendah
  - 50 sudut miring / jauh
- **Sebar kelas**:
  - mix mobil + motor
  - minimal 20% plat listrik (strip biru) supaya fuel classifier benar-benar diuji
  - sertakan beberapa gambar TANPA kendaraan (untuk uji false positive)

## Format `ground_truth.csv`

Header wajib persis seperti template:

```
filename,vehicle_type,plate_text,fuel_type,plate_x1,plate_y1,plate_x2,plate_y2
```

| Kolom          | Isi                                                                  |
| -------------- | -------------------------------------------------------------------- |
| `filename`     | nama file relatif terhadap folder ini (mis. `img001.jpg`)            |
| `vehicle_type` | `mobil` / `motor` / kosong                                           |
| `plate_text`   | teks plat ground-truth. Spasi diabaikan. Kosong = no plate           |
| `fuel_type`    | `bensin` / `listrik` / kosong                                        |
| `plate_x1..y2` | bbox plat ground-truth (int piksel). Kosongi semua kalau no plate.   |

Anotasi bbox bisa pakai [LabelImg](https://github.com/heartexlabs/labelImg) atau
[Roboflow](https://roboflow.com/) lalu konversi manual ke format CSV ini.

## Run

```
python evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv --out eval_set/results.csv
```

Exit code:
- `0` jika end-to-end >= 0.90 (release-ready)
- `2` jika di bawah target (block release di CI)
