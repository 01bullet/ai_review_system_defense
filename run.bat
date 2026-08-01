@echo off
chcp 65001 >nul
title AI Review System
echo ============================================================
echo   AI Review System — 论文审稿与攻击防御
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ first.
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Install requirements (if needed)
if not exist ".requirements_installed" (
    echo [1/3] Installing Python dependencies...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if %errorlevel% neq 0 (
        echo [ERROR] pip install failed. Check your network and try again.
        pause
        exit /b 1
    )
    echo. > ".requirements_installed"
    echo [OK] Dependencies installed.
) else (
    echo [1/3] Dependencies already installed.
)

:: Ensure base model downloaded
echo.
echo [2/3] Checking base model (Qwen2.5-7B ~15GB, downloaded on first run)...
python ensure_model.py
if %errorlevel% neq 0 (
    echo [ERROR] Model download failed. See above for manual download instructions.
    pause
    exit /b 1
)

:: Start server
echo.
echo [3/3] Starting AI Review System...
echo.
echo   Open http://localhost:8000 in your browser
echo   Press Ctrl+C to stop
echo.
python review_app.py

pause
