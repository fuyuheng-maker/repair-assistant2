@echo off
chcp 65001 >nul
title 设备检修知识助手

echo ==============================================
echo          设备检修知识助手 - 启动脚本
echo ==============================================
echo.

:: 检查并释放端口 8000
echo [1/3] 检查端口占用情况...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000"') do (
    if not "%%a"=="0" (
        echo 发现端口 8000 被进程 %%a 占用，正在释放...
        taskkill /f /pid %%a >nul 2>&1
        timeout /t 1 /nobreak >nul
    )
)

:: 等待端口释放
echo [2/3] 等待端口释放...
set "retry=0"
:wait_port
netstat -ano | findstr ":8000" >nul
if %errorlevel% equ 0 (
    set /a retry+=1
    if %retry% lss 10 (
        timeout /t 1 /nobreak >nul
        goto wait_port
    )
    echo 警告：端口释放超时，尝试强制释放...
    taskkill /f /fi "imagename eq python.exe" >nul 2>&1
    timeout /t 2 /nobreak >nul
)

:: 启动服务
echo [3/3] 启动设备检修知识助手...
echo.
echo 服务将在 http://127.0.0.1:8000 运行
echo 按 Ctrl+C 停止服务
echo.

"D:\conda\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000

pause