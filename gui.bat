@echo off
REM Launcher GUI desktop deteksi kendaraan. Jalankan dari root project.
cd /d "%~dp0"
venv\Scripts\python.exe src\gui.py %*
