@echo off
chcp 65001 >nul
echo ====================================
echo   启动 Django 后端服务
echo ====================================
echo.
cd /d E:\face\backend-django
echo 工作目录: %CD%
echo.
echo 正在启动 Django 后端...
echo 后端地址: http://localhost:8000/
echo API 文档: http://localhost:8000/api/docs/
echo 管理后台: http://localhost:8000/admin/
echo.
echo 按 Ctrl+C 停止服务
echo ====================================
echo.

E:\anaconda\envs\face_swap\python.exe manage.py runserver 0.0.0.0:8000

pause
