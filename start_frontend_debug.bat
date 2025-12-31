@echo off
chcp 65001 >nul
echo ====================================
echo   启动 PyQt 前端界面 (调试模式)
echo ====================================
echo.

echo [1/2] 检查后端状态...
curl -s http://localhost:8000/api/ >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 后端服务器未运行！
    echo 请先运行 start_backend.bat 启动后端
    echo.
    pause
    exit /b 1
)
echo [OK] 后端服务器运行正常
echo.

cd /d E:\face\frontend
echo 工作目录: %CD%
echo.
echo [2/2] 正在启动 PyQt 前端 (调试模式)...
echo - 所有调试信息将显示在控制台
echo - 界面会自动弹出
echo.
echo 按 Ctrl+C 关闭窗口
echo ====================================
echo.

E:\anaconda\envs\face_swap\python.exe -u main.py

pause
