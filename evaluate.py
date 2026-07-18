"""
evaluate.py
Evaluasi end-to-end pipeline deteksi-plat-OCR-bbm pada eval set tetap.

Usage:
    python evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv

Format ground_truth.csv (header wajib):
    filename,vehicle_type,plate_text,fuel_type,plate_x1,plate_y1,plate_x2,plate_y2

    - filename     : nama file relatif terhadap --eval-set (mis. img001.jpg)
    - vehicle_type : 'mobil' / 'motor' / kosong (kalau gak relevan)
    - plate_text   : teks plat ground-truth (alfanumerik + spasi). Kosong = no plate.
    - fuel_type    : 'bensin' / 'listrik' / kosong (kalau no plate)
    - plate_x1..y2 : bbox plat ground-truth (int). Kosongi semua kalau no plate.

Metrik yang dihitung:
    - Vehicle detection : recall@IoU>=0.5 (per kelas) — butuh kolom vehicle_type
    - Plate detection   : precision/recall/F1 @ IoU>=0.5
    - Fuel classifier   : balanced accuracy (cuma sample yang plate-detected & GT cocok)
    - OCR exact match   : (predicted text == GT text) — strip spasi, uppercase
    - End-to-end        : 1 sample dianggap benar kalau plate-detected DAN
                          fuel benar DAN OCR exact match.

Catatan:
    - Vehicle metric pakai recall (tidak punya bbox GT untuk vehicle, cuma class label).
      Untuk pipeline target >90% end-to-end, plate+OCR jauh lebih penting.
    - Sample tanpa plate (plate_text kosong) dipakai untuk hitung false positive plate.
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from detector import VehicleDetector
from fuel_classifier import FuelClassifier
from plate_reader import PlateReader


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate pipeline end-to-end.')
    p.add_argument('--eval-set', required=True, help='Folder berisi gambar eval')
    p.add_argument('--gt', required=True, help='Path ground_truth.csv')
    p.add_argument('--vehicle-model', default='models/yolov8n.pt')
    p.add_argument('--plate-model', default='models/plate_best (1).pt')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--iou-thresh', type=float, default=0.5,
                   help='IoU minimum untuk plate match (default 0.5)')
    p.add_argument('--blue-threshold', type=float, default=0.05)
    p.add_argument('--ocr-cpu', action='store_true')
    p.add_argument('--no-preprocess', action='store_true',
                   help='Matikan preprocessing OCR (CLAHE/deskew/sharpen)')
    p.add_argument('--out', default='', help='Optional: path CSV output per-sample')
    p.add_argument('--result-dir', default='hasil deteksi',
                   help='Folder output hasil akurasi akhir (grafik + catatan). '
                        'Default: "hasil deteksi".')
    return p.parse_args()


def iou(b1, b2):
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a = (ax2 - ax1) * (ay2 - ay1)
    b = (bx2 - bx1) * (by2 - by1)
    return inter / (a + b - inter) if (a + b - inter) > 0 else 0.0


def normalize_text(s):
    return ''.join(c for c in (s or '').upper() if c.isalnum())


def load_gt(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                bbox = None
                if r.get('plate_x1') and r.get('plate_y1') and r.get('plate_x2') and r.get('plate_y2'):
                    bbox = (int(r['plate_x1']), int(r['plate_y1']),
                            int(r['plate_x2']), int(r['plate_y2']))
                rows.append({
                    'filename': r['filename'].strip(),
                    'vehicle_type': (r.get('vehicle_type') or '').strip().lower(),
                    'plate_text': normalize_text(r.get('plate_text')),
                    'fuel_type': (r.get('fuel_type') or '').strip().lower(),
                    'bbox': bbox,
                })
            except Exception as e:
                print(f'[WARN] skip row {r}: {e}')
    return rows


def save_accuracy_chart(metrics, out_path, title='Hasil Akurasi Deteksi Sistem'):
    """Simpan grafik batang akurasi akhir ke file gambar (PNG).

    metrics: list of (label, value 0-1, target 0-1). Bar hijau kalau >= target,
    oranye kalau di bawah target. Nilai ditulis di atas tiap batang.
    """
    # Import di dalam fungsi supaya evaluate tetap jalan walau matplotlib absen.
    import matplotlib
    matplotlib.use('Agg')  # non-interaktif, aman headless
    import matplotlib.pyplot as plt

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    targets = [m[2] for m in metrics]
    colors = ['#2e9e5b' if v >= t else '#e08a2b' for v, t in zip(values, targets)]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.3), 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='#333', linewidth=0.6)

    # Label nilai (%) di atas batang.
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f'{v * 100:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Penanda target tiap metrik (garis pendek).
    for i, t in enumerate(targets):
        ax.plot([i - 0.4, i + 0.4], [t, t], color='#c0392b',
                linestyle='--', linewidth=1.2)

    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Akurasi')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axhline(0, color='#333', linewidth=0.8)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(handles=[
        plt.Line2D([0], [0], color='#2e9e5b', lw=8, label='Mencapai target'),
        plt.Line2D([0], [0], color='#e08a2b', lw=8, label='Di bawah target'),
        plt.Line2D([0], [0], color='#c0392b', lw=1.2, linestyle='--', label='Target'),
    ], loc='lower right', fontsize=9)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def best_plate_match(predictions, gt_bbox, iou_thresh):
    """Cari prediksi plat dengan IoU tertinggi terhadap GT (>= iou_thresh)."""
    best = None
    best_iou = 0.0
    for det in predictions:
        if det['class_name'] != 'plat':
            continue
        i = iou(det['bbox'], gt_bbox)
        if i > best_iou and i >= iou_thresh:
            best_iou = i
            best = det
    return best, best_iou


def main():
    args = parse_args()

    if not os.path.isdir(args.eval_set):
        print(f'[ERROR] eval-set folder not found: {args.eval_set}')
        sys.exit(1)
    if not os.path.isfile(args.gt):
        print(f'[ERROR] ground_truth.csv not found: {args.gt}')
        sys.exit(1)

    gt_rows = load_gt(args.gt)
    print(f'[Eval] Loaded {len(gt_rows)} ground-truth rows')

    fuel_classifier = FuelClassifier(blue_ratio_threshold=args.blue_threshold)
    plate_reader = PlateReader(gpu=not args.ocr_cpu, preprocess=not args.no_preprocess)
    detector = VehicleDetector(
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        imgsz=args.imgsz,
        smooth_window=1,  # eval per-image; tidak ada temporal smoothing
        fuel_classifier=fuel_classifier,
        plate_reader=plate_reader,
        ocr_interval=1,   # OCR setiap kali (tidak ada cache antar image)
    )

    # Counters
    n_total = 0
    n_with_plate_gt = 0

    # Plate detection
    plate_tp = plate_fp = plate_fn = 0

    # Fuel classification (sample yang plate-detected DAN GT punya plate)
    fuel_correct = 0
    fuel_total = 0
    fuel_per_class = defaultdict(lambda: [0, 0])  # class -> [correct, total]

    # OCR
    ocr_correct = 0
    ocr_total = 0

    # Vehicle recall (per kelas)
    vehicle_per_class = defaultdict(lambda: [0, 0])  # class -> [hit, total]

    # End-to-end
    e2e_correct = 0
    e2e_total = 0

    per_sample = []
    t0 = time.time()

    for i, gt in enumerate(gt_rows, 1):
        img_path = os.path.join(args.eval_set, gt['filename'])
        if not os.path.isfile(img_path):
            print(f'[WARN] image not found: {img_path}')
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            print(f'[WARN] cannot read: {img_path}')
            continue

        # Reset cache plat tiap gambar (eval per-image, bukan tracking video)
        detector._plate_cache = []
        detector.detection_history.clear()
        dets = detector.detect(frame)

        n_total += 1
        has_gt_plate = gt['bbox'] is not None

        # Vehicle recall
        if gt['vehicle_type']:
            vehicle_per_class[gt['vehicle_type']][1] += 1
            if any(d['class_name'] == gt['vehicle_type'] for d in dets):
                vehicle_per_class[gt['vehicle_type']][0] += 1

        # Plate detection
        sample_row = {
            'filename': gt['filename'],
            'gt_text': gt['plate_text'],
            'pred_text': '',
            'gt_fuel': gt['fuel_type'],
            'pred_fuel': '',
            'plate_iou': 0.0,
            'plate_correct': False,
            'fuel_correct': False,
            'ocr_correct': False,
            'e2e_correct': False,
        }

        n_pred_plates = sum(1 for d in dets if d['class_name'] == 'plat')

        if has_gt_plate:
            n_with_plate_gt += 1
            best, best_iou = best_plate_match(dets, gt['bbox'], args.iou_thresh)
            sample_row['plate_iou'] = round(best_iou, 3)
            if best is not None:
                plate_tp += 1
                # FP = sisa prediksi plat yg tidak match GT
                plate_fp += max(0, n_pred_plates - 1)
                sample_row['plate_correct'] = True

                # Fuel
                if gt['fuel_type']:
                    fuel_total += 1
                    fuel_per_class[gt['fuel_type']][1] += 1
                    pred_fuel = best.get('fuel_type', '')
                    sample_row['pred_fuel'] = pred_fuel
                    if pred_fuel == gt['fuel_type']:
                        fuel_correct += 1
                        fuel_per_class[gt['fuel_type']][0] += 1
                        sample_row['fuel_correct'] = True

                # OCR
                if gt['plate_text']:
                    ocr_total += 1
                    pred_text = normalize_text(best.get('text', ''))
                    sample_row['pred_text'] = pred_text
                    if pred_text == gt['plate_text']:
                        ocr_correct += 1
                        sample_row['ocr_correct'] = True
            else:
                plate_fn += 1
                plate_fp += n_pred_plates  # semua prediksi adalah FP

            # End-to-end: hanya hitung kalau GT punya teks DAN fuel
            if gt['plate_text'] and gt['fuel_type']:
                e2e_total += 1
                if (sample_row['plate_correct']
                        and sample_row['fuel_correct']
                        and sample_row['ocr_correct']):
                    e2e_correct += 1
                    sample_row['e2e_correct'] = True
        else:
            # Tidak ada plat di GT -> semua prediksi plat adalah FP
            plate_fp += n_pred_plates

        per_sample.append(sample_row)

        if i % 25 == 0 or i == len(gt_rows):
            print(f'[Eval] {i}/{len(gt_rows)} done')

    elapsed = time.time() - t0

    # ===== Hitung agregat =====
    plate_precision = plate_tp / (plate_tp + plate_fp) if (plate_tp + plate_fp) else 0.0
    plate_recall = plate_tp / (plate_tp + plate_fn) if (plate_tp + plate_fn) else 0.0
    plate_f1 = (2 * plate_precision * plate_recall / (plate_precision + plate_recall)
                if (plate_precision + plate_recall) else 0.0)

    fuel_acc = fuel_correct / fuel_total if fuel_total else 0.0
    if fuel_per_class:
        per_class_acc = [c / t for c, t in fuel_per_class.values() if t > 0]
        fuel_balanced_acc = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0
    else:
        fuel_balanced_acc = 0.0

    ocr_acc = ocr_correct / ocr_total if ocr_total else 0.0
    e2e_acc = e2e_correct / e2e_total if e2e_total else 0.0

    mobil_recall = (vehicle_per_class.get('mobil', [0, 0])[0]
                    / vehicle_per_class['mobil'][1]) if vehicle_per_class.get('mobil', [0, 0])[1] else 0.0
    motor_recall = (vehicle_per_class.get('motor', [0, 0])[0]
                    / vehicle_per_class['motor'][1]) if vehicle_per_class.get('motor', [0, 0])[1] else 0.0

    # ===== Susun laporan (dikumpulkan ke list -> bisa dicetak & disimpan) =====
    def fmt(name, value, target):
        mark = '[OK]' if value >= target else '[X]'
        return f'  {mark} {name:<32s} {value:.3f}  (target {target:.2f})'

    lines = []
    lines.append('=' * 60)
    lines.append('  LAPORAN AKURASI DETEKSI (EVALUATION REPORT)')
    lines.append('=' * 60)
    lines.append(f'  Waktu evaluasi:      {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'  Samples processed:   {n_total}')
    lines.append(f'  With plate GT:       {n_with_plate_gt}')
    if elapsed > 0:
        lines.append(f'  Wall time:           {elapsed:.1f}s ({n_total / elapsed:.2f} img/s)')
    lines.append('')
    lines.append('  --- Deteksi kendaraan (recall per kelas) ---')
    lines.append(f'    mobil       {mobil_recall:.3f}  ({vehicle_per_class.get("mobil", [0, 0])[0]}/{vehicle_per_class.get("mobil", [0, 0])[1]})')
    lines.append(f'    motor       {motor_recall:.3f}  ({vehicle_per_class.get("motor", [0, 0])[0]}/{vehicle_per_class.get("motor", [0, 0])[1]})')
    lines.append('')
    lines.append('  --- Deteksi plat ---')
    lines.append(f'    TP={plate_tp}  FP={plate_fp}  FN={plate_fn}')
    lines.append(fmt('Plate precision', plate_precision, 0.97))
    lines.append(fmt('Plate recall', plate_recall, 0.97))
    lines.append(fmt('Plate F1', plate_f1, 0.97))
    lines.append('')
    lines.append('  --- Klasifikasi bahan bakar ---')
    lines.append(fmt('Fuel accuracy (raw)', fuel_acc, 0.95))
    lines.append(fmt('Fuel balanced accuracy', fuel_balanced_acc, 0.95))
    for cls, (c, t) in fuel_per_class.items():
        acc = c / t if t else 0.0
        lines.append(f'    {cls:<10s}  {acc:.3f}  ({c}/{t})')
    lines.append('')
    lines.append('  --- OCR ---')
    lines.append(fmt('OCR exact match', ocr_acc, 0.95))
    lines.append('')
    lines.append('  --- END-TO-END ---')
    lines.append(fmt('Pipeline accuracy', e2e_acc, 0.90))
    lines.append(f'    {e2e_correct}/{e2e_total} samples correct end-to-end')
    lines.append('=' * 60)

    report_text = '\n'.join(lines)
    print('\n' + report_text)

    # ===== Simpan hasil akhir ke folder "hasil deteksi" =====
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    # 1) Catatan akurasi (teks)
    txt_path = os.path.join(result_dir, 'ringkasan_akurasi.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text + '\n')
    print(f'[Eval] Catatan akurasi   -> {txt_path}')

    # 2) Catatan akurasi (CSV ringkas, mudah dimasukkan ke tabel laporan)
    summary_csv = os.path.join(result_dir, 'ringkasan_akurasi.csv')
    with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metrik', 'nilai', 'nilai_persen', 'target'])
        for name, val, tgt in [
            ('recall_deteksi_mobil', mobil_recall, 0.0),
            ('recall_deteksi_motor', motor_recall, 0.0),
            ('plate_precision', plate_precision, 0.97),
            ('plate_recall', plate_recall, 0.97),
            ('plate_f1', plate_f1, 0.97),
            ('fuel_balanced_accuracy', fuel_balanced_acc, 0.95),
            ('ocr_exact_match', ocr_acc, 0.95),
            ('end_to_end_accuracy', e2e_acc, 0.90),
        ]:
            w.writerow([name, f'{val:.4f}', f'{val * 100:.1f}%', f'{tgt:.2f}'])
    print(f'[Eval] Ringkasan CSV     -> {summary_csv}')

    # 3) Grafik akurasi (gambar PNG untuk laporan)
    chart_metrics = [
        ('Recall\nMobil', mobil_recall, 0.90),
        ('Recall\nMotor', motor_recall, 0.90),
        ('Plat\nPrecision', plate_precision, 0.97),
        ('Plat\nRecall', plate_recall, 0.97),
        ('Plat\nF1', plate_f1, 0.97),
        ('Klasifikasi\nBBM', fuel_balanced_acc, 0.95),
        ('OCR\nExact', ocr_acc, 0.95),
        ('End-to-\nEnd', e2e_acc, 0.90),
    ]
    chart_path = os.path.join(result_dir, 'grafik_akurasi.png')
    try:
        save_accuracy_chart(chart_metrics, chart_path)
        print(f'[Eval] Grafik akurasi    -> {chart_path}')
    except Exception as e:
        print(f'[Eval] WARNING: gagal membuat grafik ({e}). '
              f'Catatan teks & CSV tetap tersimpan.')

    # 4) Hasil per-gambar (CSV rinci) -> selalu di folder hasil, plus --out kalau diminta
    per_sample_path = os.path.join(result_dir, 'hasil_per_gambar.csv')
    if per_sample:
        csv_targets = [per_sample_path] + ([args.out] if args.out else [])
        for path in csv_targets:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
            with open(path, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
                w.writeheader()
                for row in per_sample:
                    w.writerow(row)
        print(f'[Eval] Hasil per-gambar  -> {per_sample_path}')

    # Exit non-zero kalau e2e di bawah target -> bisa dipakai di CI/release gate
    if e2e_total > 0 and e2e_acc < 0.90:
        sys.exit(2)


if __name__ == '__main__':
    main()
