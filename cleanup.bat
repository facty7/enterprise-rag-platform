@echo off
cd /d "%~dp0"
echo.
echo   ==========================================
echo     Enterprise RAG Knowledge Platform - Cleanup
echo   ==========================================
echo.
echo   This will DELETE:
echo     - All uploaded documents
echo     - Vector database (Qdrant)
echo     - Chat history
echo     - Audit logs
echo     - QR lock file
echo     - Test documents
echo.
echo   The following will be KEPT:
echo     - User accounts (config/users.json)
echo     - Model files (models/)
echo     - System code (app/)
echo.
set /p CONFIRM="  Type 'YES' to confirm: "
if not "%CONFIRM%"=="YES" (
    echo   Cancelled.
    pause
    exit /b 0
)

echo.
echo   Cleaning...

if exist "data\qdrant_db" (
    rmdir /s /q "data\qdrant_db" >nul 2>&1
    echo   [OK] Qdrant database removed
)
if exist "data\uploads" (
    rmdir /s /q "data\uploads" >nul 2>&1
    echo   [OK] Uploaded files removed
)
if exist "data\test_docs" (
    rmdir /s /q "data\test_docs" >nul 2>&1
    echo   [OK] Test documents removed
)
if exist "data\chat_history.json" (
    del /q "data\chat_history.json" >nul 2>&1
    echo   [OK] Chat history removed
)
if exist "data\logs\audit.jsonl" (
    del /q "data\logs\audit.jsonl" >nul 2>&1
    echo   [OK] Audit logs removed
)
if exist "data\.wheels_installed" (
    del /q "data\.wheels_installed" >nul 2>&1
)
if exist "data\eval_result.json" (
    del /q "data\eval_result.json" >nul 2>&1
)
if exist "data\comparison_result.json" (
    del /q "data\comparison_result.json" >nul 2>&1
)
if exist "config\qr_lock.txt" (
    del /q "config\qr_lock.txt" >nul 2>&1
    echo   [OK] QR lock removed
)

rem Recreate empty dirs
if not exist "data\uploads" mkdir "data\uploads"
if not exist "data\logs" mkdir "data\logs"

echo.
echo   ==========================================
echo     Cleanup complete!
echo     Ready for fresh deployment.
echo   ==========================================
echo.
pause
