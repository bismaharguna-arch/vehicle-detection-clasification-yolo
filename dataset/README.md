# Dataset Training Detektor Plat

Struktur folder mengikuti konvensi YOLO (Ultralytics):

```
dataset/
├── data.yaml
├── images/
│   ├── train/   *.jpg / *.png
│   ├── val/
│   └── test/    (opsional)
└── labels/
    ├── train/   *.txt   (satu per gambar, format YOLO)
    ├── val/
    └── test/
```

## Format label YOLO

Tiap gambar punya file `.txt` dengan nama sama. Tiap baris satu bbox:

```
<class_id> <cx> <cy> <w> <h>
```

Semua nilai **dinormalisasi 0-1** terhadap ukuran gambar. Untuk dataset ini
hanya ada satu kelas, jadi `class_id` selalu `0`.

## Anotasi

Cara paling cepat (rekomendasi): **Roboflow** — upload gambar, bounding box di
browser, export ke "YOLOv8" format, lalu unzip ke `dataset/`.

Alternatif lokal: **LabelImg** atau **CVAT** (self-hosted Docker).

## Target ukuran

- Minimum: 500 gambar plat Indonesia (mix mobil + motor + listrik)
- Ideal: 1000-2000 gambar dengan variasi waktu/cuaca/sudut
- Split: 80% train / 15% val / 5% test

## Run training

```
python train_plate.py
```

Default: 100 epoch, imgsz 1280, batch 16, base model yolov8s.pt. Output di
`runs/detect/train*/weights/best.pt`. Setelah selesai:

```
copy runs\detect\train\weights\best.pt models\license_plate_v2.pt
```

lalu update `--plate-model` di `main.py` / `evaluate.py`.

## Tips

- Plat kecil di gambar full-frame: pakai `imgsz=1280` minimum (jangan turun ke
  640, plat akan jadi <16px).
- Augmentasi default Ultralytics (mosaic + hsv + flip) sudah cukup untuk plat.
  **Jangan** aktifkan vertical flip — plat tidak pernah upside-down.
- Kalau val mAP@0.5 plateau di <0.95 setelah 100 epoch, kemungkinan dataset
  bias (mis. semua siang). Tambah variasi, jangan tambah epoch.
