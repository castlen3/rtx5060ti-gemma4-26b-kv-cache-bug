@echo off
setlocal
REM PASSING CONTROL: Minimal 12K matched KV test, K=q8_0, V=q8_0
REM Edit LLAMA_DIR and MODEL for your machine.
set "LLAMA_DIR=C:\Users\castlen3\llama-cuda-5060ti-release"
set "MODEL=C:\Users\castlen3\.lmstudio\models\lmstudio-community\gemma-4-26B-A4B-it-QAT-GGUF\gemma-4-26B-A4B-it-QAT-Q4_0.gguf"
set "OUTDIR=%~dp0..\logs"
set "LOG=%OUTDIR%\server-12k-q8q8-good.log"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
echo ============================================
echo PASSING CONTROL - 12K - K=q8_0 / V=q8_0
echo Gemma 4 26B A4B - RTX 5060 Ti CUDA
echo Log: %LOG%
echo ============================================
taskkill /F /IM llama-server.exe >nul 2>nul
taskkill /F /IM llama-cli.exe >nul 2>nul
taskkill /F /IM llama-bench.exe >nul 2>nul
timeout /t 5 /nobreak >nul
nvidia-smi
"%LLAMA_DIR%\llama-server.exe" -m "%MODEL%" -ngl 99 --device CUDA0 -t 8 -c 12288 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -fit off --no-mmap --n-cpu-moe 0 -b 4096 -ub 1024 --threads-batch 12 -np 1 --host 0.0.0.0 --port 8080 --log-file "%LOG%" --log-timestamps
pause
