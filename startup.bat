@echo off
cd /d "%~dp0"

REM === Pick Python: bundled first, system fallback ===
set PY_CMD=
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" -c "import fastapi, uvicorn" >nul 2>&1
    if not errorlevel 1 set PY_CMD=%~dp0python\python.exe
)
if "%PY_CMD%"=="" (
    python -c "import fastapi, uvicorn" >nul 2>&1
    if not errorlevel 1 set PY_CMD=python
)
if "%PY_CMD%"=="" (
    echo [ERROR] Python dependencies not found. Run: pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [Init] Created .env from .env.example
    ) else (
        echo [ERROR] .env.example file missing
        pause
        exit /b 1
    )
)

if not exist "files" mkdir files
if not exist "data\uploads" mkdir data\uploads
if not exist "data\qdrant_db" mkdir data\qdrant_db
if not exist "data\reports" mkdir data\reports
if not exist "data\logs" mkdir data\logs

echo.
echo   ==========================================
echo     Enterprise RAG Knowledge Platform
echo     Hybrid Retrieval + RBAC + Metrics
echo   ==========================================
echo.
echo   Login: admin / admin123
echo   Loading model (CPU) - please wait...
echo.

"%PY_CMD%" launcher.py
pause
