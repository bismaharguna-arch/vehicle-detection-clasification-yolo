@echo off
REM ============================================================
REM Simulasi Laptop B (Web Monitoring)
REM Jalankan di terminal terpisah dari sim_laptop_a.bat.
REM Akan listen di SEMUA interface (0.0.0.0:5000) supaya
REM Laptop A (atau detector di mesin yang sama) bisa konek.
REM ============================================================

cd /d "%~dp0\.."

REM Aktifkan venv (Flask & cv2 ada di sini, bukan di system Python)
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

echo.
echo ============================================================
echo   LAPTOP B SIMULATOR - Web Monitoring
echo ============================================================
for /f %%i in ('python tools\get_lan_ip.py') do set LAN_IP=%%i
echo   IP LAN aktif : %LAN_IP%
echo   Endpoint     : http://%LAN_IP%:5000/api/detections
echo   Dashboard    : http://%LAN_IP%:5000/
echo   (atau dari mesin ini juga bisa: http://localhost:5000/)
echo ============================================================
echo.

REM Latency 30ms = mirip latensi WiFi LAN nyata.
python tools\mock_server.py --host 0.0.0.0 --port 5000 --latency-ms 30

pause
