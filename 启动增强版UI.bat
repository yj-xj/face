@echo off
chcp 65001 >nul
echo ========================================
echo   人脸替换应用 - 增强版
echo ========================================
echo.
echo 正在启动应用程序...
echo.

cd /d "%~dp0"

"E:\anaconda\envs\face_swap\python.exe" "e:\face\src\face_swap_ui_enhanced.py"

if errorlevel 1 (
    echo.
    echo [错误] 应用程序启动失败！
    echo.
    pause
)
