"""
logger.py
Logger structured (JSONL) untuk hasil deteksi.

Tiap baris = satu deteksi pada satu frame:
    {"ts": 1715000000.123, "frame": 42, "source": "video.mp4",
     "class": "plat", "conf": 0.91, "bbox": [x1,y1,x2,y2],
     "fuel_type": "bensin", "blue_ratio": 0.02, "text": "B 1234 ABC"}

Pakai JSONL (bukan satu JSON array besar) supaya bisa di-tail / di-stream.
"""

import json
import time
import os


class DetectionLogger:
    def __init__(self, log_path, source='', flush_every=1):
        """
        Args:
            log_path: file output (.jsonl). Folder dibuat otomatis kalau belum ada.
            source: identifier source (video path / 'webcam'), masuk ke tiap row.
            flush_every: flush ke disk tiap N writes. 1 = real-time tail-able.
        """
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or '.', exist_ok=True)
        self.path = log_path
        self.source = source
        self.flush_every = max(1, flush_every)
        self._fh = open(log_path, 'a', encoding='utf-8', buffering=1)
        self._counter = 0
        print(f'[Logger] JSONL -> {log_path}')

    def log(self, frame_idx, detections):
        """Tulis satu baris JSON per detection."""
        ts = time.time()
        for det in detections:
            row = {
                'ts': ts,
                'frame': frame_idx,
                'source': self.source,
                'class': det.get('class_name', ''),
                'conf': round(float(det.get('confidence', 0.0)), 4),
                'bbox': [int(v) for v in det.get('bbox', (0, 0, 0, 0))],
            }
            if det.get('class_name') == 'plat':
                row['fuel_type'] = det.get('fuel_type', '')
                row['blue_ratio'] = round(float(det.get('blue_ratio', 0.0)), 4)
                row['text'] = det.get('text', '')
            self._fh.write(json.dumps(row, ensure_ascii=False) + '\n')

        self._counter += 1
        if self._counter % self.flush_every == 0:
            self._fh.flush()

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
