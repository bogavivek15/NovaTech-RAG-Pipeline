@echo off
echo ========================================
echo   NovaTech RAG Chatbot - Quick Start
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    pause
    exit /b 1
)

echo Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting NovaTech Chatbot...
echo.
echo Chat Interface: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
timeout /t 2 /nobreak >nul
start http://localhost:8000
echo.
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
