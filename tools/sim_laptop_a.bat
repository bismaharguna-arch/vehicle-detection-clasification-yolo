@echo off
REM ============================================================
REM Simulasi Laptop A (Detector)
REM Jalankan SETELAH sim_laptop_b.bat sudah running di terminal lain.
REM Auto-detect IP LAN aktif supaya simulasinya mirip jaringan nyata
REM (bukan localhost).
REM ============================================================

cd /d "%~dp0\.."

REM Aktifkan venv supaya cv2/ultralytics/paddleocr terpakai
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

REM ----- Argumen opsional -----
REM   %1 = source video / nomor webcam (default: video.mp4)
REM   %2 = override IP server (default: auto-detect IP LAN sendiri)
set SOURCE=%1
if "%SOURCE%"=="" set SOURCE=video.mp4

set SERVER_IP=%2
if "%SERVER_IP%"=="" (
    for /f %%i in ('python tools\get_lan_ip.py') do set SERVER_IP=%%i
)

set ENDPOINT=http://%SERVER_IP%:5000/api/detections

echo.
echo ============================================================
echo   LAPTOP A SIMULATOR - Detector
echo ============================================================
echo   Source   : %SOURCE%
echo   Target   : %ENDPOINT%
echo   OCR      : OFF (uji konektivitas dulu)
echo   Push     : ON  (one-shot per kendaraan)
echo ============================================================
echo.

python src\main.py ^
    --source "%SOURCE%" ^
    --push ^
    --push-url %ENDPOINT% ^
    --push-mode multipart ^
    --push-stable 5 ^
    --push-timeout 30

pause
