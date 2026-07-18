"""
mock_server.py
Simulasi web monitoring (Laptop B) di mesin yang sama dengan detector.

Implementasi PERSIS kontrak /api/detections:
    POST /api/detections (multipart/form-data)
        plate_number     : string (boleh kosong)
        is_electric      : "electric" | "gasoline" | "unknown"
        vehicle_type     : "car" | "motorcycle" | "truck" | "unknown"
        confidence_score : float
        photo            : file JPG/PNG

    -> 201 {"status": "SUCCESS", "id": <int>}
    -> 500 {"status": "ERROR", "message": "..."}

Selain validasi dasar, server ini juga:
- simpan foto + JSON sidecar ke folder log
- print ringkasan setiap request (mirip dashboard realtime)
- expose GET /api/detections untuk lihat semua entri
- expose GET / untuk halaman dashboard sederhana

Untuk simulasi delay jaringan antar-laptop, pakai --latency-ms.

Usage:
    python tools/mock_server.py --host 0.0.0.0 --port 5000
    python tools/mock_server.py --latency-ms 50   # simulasi LAN
"""

import argparse
import json
import os
import time
from threading import Lock

from flask import Flask, request, jsonify


# Kontrak v1 (Mei 2026): Indonesia, drop English mapping, drop truck.
VALID_VEHICLE = {'mobil', 'motor', 'unknown'}
VALID_FUEL = {'bensin', 'listrik', 'unknown'}


def create_app(log_dir, latency_ms=0, drop_rate=0.0,
               max_content_mb=5, accept_detected_at=True):
    app = Flask(__name__)
    # Match planned hard limit di server real (5 MB).
    app.config['MAX_CONTENT_LENGTH'] = int(max_content_mb * 1024 * 1024)

    os.makedirs(log_dir, exist_ok=True)
    state = {
        'next_id': 1,
        'records': [],
        'lock': Lock(),
    }

    @app.route('/api/detections', methods=['POST'])
    def receive_detection():
        # Simulasi latensi jaringan LAN
        if latency_ms > 0:
            time.sleep(latency_ms / 1000.0)

        # Simulasi packet drop / server error
        if drop_rate > 0:
            import random
            if random.random() < drop_rate:
                return jsonify({'status': 'ERROR',
                                'message': 'simulated drop'}), 500

        # Tarik field. Multipart -> request.form, JSON -> request.json
        if request.is_json:
            data = request.get_json() or {}
            photo = None
        else:
            data = request.form
            photo = request.files.get('photo')

        plate_number = (data.get('plate_number') or '').strip()
        is_electric = (data.get('is_electric') or 'unknown').strip().lower()
        vehicle_type = (data.get('vehicle_type') or 'unknown').strip().lower()
        try:
            conf = float(data.get('confidence_score') or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        # detected_at: optional, ISO 8601 dengan offset TZ. Server saat ini
        # menerima tapi belum dipakai authoritative (forward-compat).
        detected_at = (data.get('detected_at') or '').strip() if accept_detected_at else ''

        # Nilai di luar enum -> log warning lalu paksa 'unknown' (server real
        # sebaiknya juga begini supaya stat tidak dipolusi orphan label).
        if is_electric not in VALID_FUEL:
            print(f'[Server] WARN unknown is_electric={is_electric!r} '
                  f'-> treat as unknown')
            is_electric = 'unknown'
        if vehicle_type not in VALID_VEHICLE:
            print(f'[Server] WARN unknown vehicle_type={vehicle_type!r} '
                  f'-> treat as unknown')
            vehicle_type = 'unknown'
        conf = max(0.0, min(1.0, conf))

        # Persist
        with state['lock']:
            new_id = state['next_id']
            state['next_id'] += 1

        ts_received = time.time()
        ts_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts_received))
        record = {
            'id': new_id,
            'received_at': time.strftime('%Y-%m-%dT%H:%M:%S%z',
                                         time.localtime(ts_received)),
            # Kalau detector kirim detected_at, log juga -- ini yang nanti
            # dipakai sebagai timestamp authoritative.
            'detected_at': detected_at,
            'plate_number': plate_number,
            'is_electric': is_electric,
            'vehicle_type': vehicle_type,
            'confidence_score': conf,
            'is_corrected': False,
        }

        if photo is not None:
            ext = os.path.splitext(photo.filename or '')[1] or '.jpg'
            photo_path = os.path.join(log_dir,
                                      f'{ts_str}_{new_id:05d}{ext}')
            try:
                photo.save(photo_path)
                record['photo_file'] = os.path.basename(photo_path)
            except Exception as e:
                return jsonify({'status': 'ERROR',
                                'message': f'failed save photo: {e}'}), 500

        # Sidecar JSON (audit trail)
        side_path = os.path.join(log_dir, f'{ts_str}_{new_id:05d}.json')
        with open(side_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        with state['lock']:
            state['records'].append(record)

        # Print ringkas (gaya dashboard)
        print(f'[Server] +id={new_id:<4d} {vehicle_type:<10s} '
              f'{is_electric:<8s} conf={conf:.2f} '
              f'photo={"YES" if photo else "no":<3s} '
              f'plate={plate_number or "(none)"}')

        return jsonify({'status': 'SUCCESS', 'id': new_id}), 201

    @app.route('/api/detections/<int:rec_id>/plate', methods=['PATCH'])
    def correct_plate(rec_id):
        """Koreksi plat susulan dari detector (plat telat terbaca / terkoreksi
        saat kendaraan mendekat). Mirror aturan server real:
          - hanya plate_number (+ is_electric kalau saat POST masih 'unknown')
          - record yang sudah dikoreksi ADMIN tidak ditimpa -> 200 SKIPPED
          - id tak dikenal -> 404
        Tanpa rute ini, jalur PATCH detector tidak bisa diuji lokal (dulu selalu
        404 dan koreksi plat kelihatan 'tidak pernah sampai')."""
        if latency_ms > 0:
            time.sleep(latency_ms / 1000.0)
        if drop_rate > 0:
            import random
            if random.random() < drop_rate:
                return jsonify({'status': 'ERROR',
                                'message': 'simulated drop'}), 500

        data = request.get_json(silent=True) or {}
        plate_number = (data.get('plate_number') or '').strip()
        is_electric = (data.get('is_electric') or '').strip().lower()

        with state['lock']:
            rec = next((r for r in state['records'] if r['id'] == rec_id), None)
            if rec is None:
                return jsonify({'status': 'ERROR',
                                'message': f'id {rec_id} not found'}), 404
            if rec.get('is_corrected'):
                print(f'[Server] ~id={rec_id:<4d} SKIPPED (sudah dikoreksi admin)')
                return jsonify({'status': 'SKIPPED', 'id': rec_id}), 200
            old = rec['plate_number'] or '(none)'
            if plate_number:
                rec['plate_number'] = plate_number
            if is_electric in VALID_FUEL and rec['is_electric'] == 'unknown':
                rec['is_electric'] = is_electric

        print(f'[Server] ~id={rec_id:<4d} plate {old} -> '
              f'{rec["plate_number"] or "(none)"} '
              f'({rec["is_electric"]})')
        return jsonify({'status': 'SUCCESS', 'id': rec_id}), 200

    @app.route('/api/detections', methods=['GET'])
    def list_detections():
        with state['lock']:
            return jsonify({
                'count': len(state['records']),
                'records': state['records'][-50:],  # 50 terbaru
            })

    @app.route('/api/stats', methods=['GET'])
    def stats():
        with state['lock']:
            recs = state['records']
        n = len(recs)
        listrik = sum(1 for r in recs if r['is_electric'] == 'listrik')
        bensin = sum(1 for r in recs if r['is_electric'] == 'bensin')
        mobil = sum(1 for r in recs if r['vehicle_type'] == 'mobil')
        motor = sum(1 for r in recs if r['vehicle_type'] == 'motor')
        return jsonify({
            'total': n,
            'by_fuel': {'listrik': listrik, 'bensin': bensin,
                        'unknown': n - listrik - bensin},
            'by_vehicle': {'mobil': mobil, 'motor': motor,
                           'unknown': n - mobil - motor},
        })

    @app.route('/', methods=['GET'])
    def dashboard():
        with state['lock']:
            recs = list(reversed(state['records'][-30:]))
        def _conf_badge(c):
            if c < 0.65:
                return f'<span style="color:#fa3">{c:.2f} low</span>'
            return f'{c:.2f}'

        def _plate_cell(s):
            if not s:
                return '<span style="color:#888">— belum di-OCR</span>'
            return s

        rows_html = ''.join(
            f'<tr><td>{r["id"]}</td>'
            f'<td>{r.get("detected_at") or r.get("received_at", "")}</td>'
            f'<td>{r["vehicle_type"]}</td><td>{r["is_electric"]}</td>'
            f'<td>{_conf_badge(r["confidence_score"])}</td>'
            f'<td>{_plate_cell(r.get("plate_number"))}</td>'
            f'<td>{"YES" if r.get("photo_file") else "-"}</td></tr>'
            for r in recs
        )
        return f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Mock Web Monitoring</title>
<meta http-equiv="refresh" content="2">
<style>
body {{ font-family: monospace; background: #111; color: #eee; padding: 20px; }}
h1 {{ color: #4cf; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
th {{ background: #222; color: #4cf; }}
tr:nth-child(even) {{ background: #181818; }}
.badge {{ padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
</style></head>
<body>
<h1>Mock Web Monitoring (Laptop B simulator)</h1>
<p>Total deteksi: <b>{len(state["records"])}</b> | Auto-refresh tiap 2 detik.</p>
<table>
<tr><th>ID</th><th>Waktu</th><th>Kendaraan</th><th>BBM</th>
    <th>Conf</th><th>Plat</th><th>Foto</th></tr>
{rows_html}
</table>
</body></html>'''

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'status': 'ERROR',
                        'message': f'photo too large '
                                   f'(max {max_content_mb}MB)'}), 413

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500

    return app


def parse_args():
    p = argparse.ArgumentParser(description='Mock web monitoring server.')
    p.add_argument('--host', default='0.0.0.0',
                   help='Bind host. 0.0.0.0 supaya bisa diakses dari LAN '
                        '(simulasi multi-device).')
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--log-dir', default='output/mock_server',
                   help='Folder simpan foto + sidecar JSON.')
    p.add_argument('--latency-ms', type=int, default=0,
                   help='Delay artifisial per request (simulasi LAN).')
    p.add_argument('--drop-rate', type=float, default=0.0,
                   help='Probabilitas (0-1) bales 500 untuk uji retry.')
    p.add_argument('--max-mb', type=int, default=5,
                   help='Hard limit ukuran request body (MB). Default 5 '
                        '(match planned limit di server real).')
    return p.parse_args()


def main():
    args = parse_args()
    print('=' * 60)
    print('  Mock Web Monitoring Server (Laptop B simulator)')
    print('=' * 60)
    print(f'  Endpoint  : http://{args.host}:{args.port}/api/detections')
    print(f'  Dashboard : http://localhost:{args.port}/')
    print(f'  Log dir   : {args.log_dir}')
    print(f'  Max body  : {args.max_mb} MB')
    print(f'  Vehicle   : {sorted(VALID_VEHICLE)}')
    print(f'  Fuel      : {sorted(VALID_FUEL)}')
    if args.latency_ms:
        print(f'  Latency   : {args.latency_ms} ms (simulasi LAN)')
    if args.drop_rate:
        print(f'  Drop rate : {args.drop_rate} (simulasi packet loss)')
    print('=' * 60)
    print()

    app = create_app(args.log_dir, args.latency_ms, args.drop_rate,
                     max_content_mb=args.max_mb)
    # Werkzeug dev server cukup untuk simulasi single-client.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
