# Simulasi 2-Laptop di 1 Mesin

Simulasi setup `Laptop A (detector) -> Laptop B (web monitoring)` tanpa
perlu hardware kedua. Pakai 2 terminal di mesin yang sama, dengan IP LAN
asli (bukan `localhost`) supaya pola HTTP-nya identik dengan deployment
nyata.

## Cara pakai (paling singkat)

Buka **2 terminal Windows**, di folder root project:

**Terminal 1 — Laptop B (Web Monitoring)**
```
tools\sim_laptop_b.bat
```
- Listen di `0.0.0.0:5000`
- Print IP LAN aktif (mis. `192.168.1.10`)
- Buka browser: `http://localhost:5000/` untuk dashboard live

**Terminal 2 — Laptop A (Detector)**
```
tools\sim_laptop_a.bat
```
- Default source = `video.mp4`, server IP = auto-detect IP LAN sendiri
- Override:
  ```
  tools\sim_laptop_a.bat test_images\sample.mp4
  tools\sim_laptop_a.bat 0  192.168.1.10        # webcam, IP eksplisit
  ```

Setiap kali ada kendaraan stabil terdeteksi, terminal A akan print
`[Tracker] PUSH ...` dan terminal B akan print `[Server] +id=...`.
Dashboard di browser auto-refresh tiap 2 detik.

## Apa yang disimulasikan

| Aspek                | Realita 2-laptop          | Simulasi sini                          |
| -------------------- | ------------------------- | -------------------------------------- |
| Endpoint             | `http://<ip-B>:5000/...`  | `http://<ip-LAN-sendiri>:5000/...`     |
| Bind address         | `0.0.0.0` (semua iface)   | sama (bukan `127.0.0.1`)               |
| Multipart kontrak    | identik                   | identik                                |
| Latensi LAN          | 5-50 ms                   | `--latency-ms 30` di mock              |
| Packet loss          | jarang                    | `--drop-rate 0.1` opsional             |
| Firewall             | wajib buka 5000           | tidak perlu (loopback via LAN IP OK)   |

## Komponen

### `mock_server.py`
Flask app implementasi kontrak `/api/detections`:
- POST multipart -> 201 + `{"status":"SUCCESS","id":<int>}`
- GET `/` -> dashboard HTML auto-refresh
- GET `/api/detections` -> 50 entri terakhir
- GET `/api/stats` -> rekap by-fuel & by-vehicle
- Foto + sidecar JSON disimpan ke `output/mock_server/`

Flag berguna untuk uji ketahanan detector:
```
python tools\mock_server.py --latency-ms 100  # lambat, simulasi WAN
python tools\mock_server.py --drop-rate 0.2   # 20% request bales 500
                                              # uji retry uploader
```

### `get_lan_ip.py`
Helper print IP LAN aktif (bukan loopback). Dipakai oleh kedua bat file
supaya simulasinya pakai IP nyata, bukan `127.0.0.1`.

### `sim_laptop_a.bat` / `sim_laptop_b.bat`
Wrapper. Auto-set argumen yang masuk akal untuk uji integrasi cepat.
Edit kalau mau ganti video/source default.

## Naik ke deployment 2-laptop sungguhan

Cuma tukar 2 hal:
1. Di **Laptop B**, jalankan `mock_server.py` (atau web monitoring asli)
   dengan `--host 0.0.0.0`. Pastikan firewall inbound TCP 5000 dibuka:
   ```
   New-NetFirewallRule -DisplayName "WebMonitor 5000" -Direction Inbound ^
       -Protocol TCP -LocalPort 5000 -Action Allow
   ```
2. Di **Laptop A**, panggil `sim_laptop_a.bat <source> <ip-laptop-B>`
   atau langsung `python src/main.py --source ... --push --push-url
   http://<ip-laptop-B>:5000/api/detections`.

Tidak ada perubahan kode di sisi detector.

## Smoke test cepat (tanpa video)

Cek server hidup tanpa run detector:
```
curl -i -X POST http://localhost:5000/api/detections ^
    -F "plate_number=" ^
    -F "is_electric=electric" ^
    -F "vehicle_type=car" ^
    -F "confidence_score=0.9" ^
    -F "photo=@test_plate.jpg"
```
Harusnya balas `HTTP/1.0 201 CREATED` + `{"status":"SUCCESS","id":1}`.
