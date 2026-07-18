"""
train_plate.py
Fine-tune YOLO untuk detektor plat Indonesia.

Usage:
    python train_plate.py
    python train_plate.py --epochs 150 --batch 8 --base yolov8s.pt

Output: runs/detect/train*/weights/best.pt
        copy ke models/license_plate_v2.pt setelah validasi.

Pra-syarat:
    - dataset/ terisi sesuai dataset/README.md
    - GPU CUDA tersedia (training di CPU akan SANGAT lambat)
"""

import argparse
import os
import sys

import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset/data.yaml',
                   help='Path data.yaml')
    p.add_argument('--base', default='yolov8s.pt',
                   help='Base model (yolov8n.pt = cepat, yolov8s.pt = balance, '
                        'yolov8m.pt = akurasi tinggi tapi lambat)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=1280,
                   help='Image size (jangan <640 untuk plat)')
    p.add_argument('--batch', type=int, default=16,
                   help='Batch size. Turunkan kalau OOM.')
    p.add_argument('--patience', type=int, default=20,
                   help='Early stopping patience (epoch tanpa improvement)')
    p.add_argument('--device', default='', help='cuda:0 / cpu / kosong=auto')
    p.add_argument('--project', default='runs/detect')
    p.add_argument('--name', default='plate_v2')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.data):
        print(f'[ERROR] data.yaml not found: {args.data}')
        print('Lihat dataset/README.md untuk struktur yang dibutuhkan.')
        sys.exit(1)

    if not args.device:
        args.device = '0' if torch.cuda.is_available() else 'cpu'

    if args.device == 'cpu':
        print('[WARN] Training di CPU akan sangat lambat (puluhan jam). '
              'Pertimbangkan colab / GPU lokal.')

    print('=' * 60)
    print(f'  Base model : {args.base}')
    print(f'  Data       : {args.data}')
    print(f'  Epochs     : {args.epochs}')
    print(f'  Image size : {args.imgsz}')
    print(f'  Batch      : {args.batch}')
    print(f'  Device     : {args.device}')
    print('=' * 60)

    model = YOLO(args.base)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentation tuning untuk plat:
        flipud=0.0,        # plat tidak pernah terbalik vertikal
        fliplr=0.5,        # mirror OK
        mosaic=1.0,        # default Ultralytics, bagus untuk small object
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,       # rotasi kecil saja, plat biasanya hampir lurus
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        cache=False,       # set True kalau RAM cukup (mempercepat IO)
        amp=True,          # mixed precision -> lebih cepat di GPU modern
    )

    print('\n[Train] Selesai.')
    print('[Train] Best weights -> runs/detect/<name>/weights/best.pt')
    print('[Train] Lakukan: copy runs\\detect\\<name>\\weights\\best.pt models\\license_plate_v2.pt')
    print('[Train] Lalu jalankan evaluate.py untuk verifikasi mAP & end-to-end.')


if __name__ == '__main__':
    main()
