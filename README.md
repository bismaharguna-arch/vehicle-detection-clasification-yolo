# vehicle-detection-clasification-yolo

Deteksi & klasifikasi kendaraan listrik dan bensin menggunakan YOLOv8 + PaddleOCR.

- **Deteksi kendaraan & plat**: YOLOv8 (dual-model: `models/yolov8n.pt` + `models/plate_best (1).pt`)
- **Klasifikasi bahan bakar**: strip biru plat (HSV) → `bensin` / `listrik`
- **OCR plat**: PaddleOCR (async, dengan preprocessing deskew/CLAHE/unsharp)
- **Tracking & anti-spam push**: Hungarian matching, push satu kali per kendaraan ke web monitoring

## Menjalankan

```powershell
# siapkan environment (sekali saja)
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# GUI desktop
venv\Scripts\python.exe src\gui.py

# CLI (webcam default)
venv\Scripts\python.exe src\main.py

# Video + push ke web monitoring
venv\Scripts\python.exe src\main.py --source video.mp4 --push --push-url http://<ip>:5000/api/detections
```

Detail arsitektur pipeline dan kontrak API ada di [CLAUDE.md](CLAUDE.md) dan [TECHSTACK.md](TECHSTACK.md).

> Catatan: `api_key.txt` (kunci X-API-Key web monitoring) tidak ikut di repo — buat manual di root proyek.
