@echo off
REM ============================================================
REM Start script - Aktifkan venv dan siap kerja
REM ============================================================

echo.
echo ============================================================
echo   PROJECT DETEKSI KENDARAAN
echo ============================================================
echo.

REM Pindah ke folder script (biar bisa di-run dari mana aja)
cd /d "%~dp0"

REM Cek venv ada
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Folder venv tidak ditemukan!
    echo Pastikan kamu menjalankan ini dari folder D:\deteksi_kendaraan
    echo.
    pause
    exit /b 1
)

REM Aktifkan venv
echo [Info] Mengaktifkan virtual environment...
call venv\Scripts\activate.bat

REM Tampilkan info
echo.
echo [OK] Virtual environment AKTIF
echo.
python --version
echo.
echo Folder kerja: %CD%
echo.
echo ============================================================
echo   READY! Perintah yang sering dipakai:
echo ============================================================
echo.
echo   python src\main.py                                          (webcam + preview :5001)
echo   python src\main.py --source video.mp4                       (video file)
echo   python src\main.py --no-preview                             (matikan MJPEG preview)
echo   python src\main.py --push --push-url http://IP:5000/api/detections   (push ke web)
echo.
echo   Live preview MJPEG:   http://localhost:5001/preview         (embed di dashboard web)
echo   Health check:         http://localhost:5001/health
echo.
echo   python diagnose.py                                          (cek model)
echo   python test_ocr.py                                          (test OCR)
echo   python evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv   (eval pipeline)
echo.
echo   deactivate                                                  (keluar venv)
echo   exit                                                        (tutup terminal)
echo.
echo ============================================================
echo.

REM Buka shell baru di dalam venv
cmd /k