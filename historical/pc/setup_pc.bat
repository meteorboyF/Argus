@echo off
REM ============================================================================
REM  ARGUS - Windows PC dependency installer (PC TEST PHASE ONLY)
REM ============================================================================
REM  This sets up the PC-side test environment (GTX 1060) used by
REM  argus_pc_test.ipynb. It is NOT the Jetson installer.
REM
REM  The Jetson runs Ubuntu/ARM64 and CANNOT run a .bat file. On the Jetson use:
REM      ./scripts/setup_jetson.sh
REM
REM  Usage (from an Anaconda Prompt, in this folder):
REM      setup_pc.bat
REM ============================================================================
setlocal

echo.
echo === ARGUS PC setup (GTX 1060 / Windows) ===
echo.

REM 1. Create + activate conda env
echo [1/3] Creating conda env "argus" (python 3.11)...
call conda create -n argus python=3.11 -y
if errorlevel 1 goto :err
call conda activate argus
if errorlevel 1 goto :err

REM 2. CUDA build of PyTorch for the Pascal 1060
echo [2/3] Installing CUDA PyTorch (cu121)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :err

REM 3. Rest of the PC-test dependencies
echo [3/3] Installing runtime/test dependencies...
pip install ultralytics opencv-python faster-whisper openwakeword onnx onnxruntime sounddevice soundfile piper-tts
if errorlevel 1 goto :err

echo.
echo === Done. Verify CUDA: ===
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
echo.
echo Next: open argus_pc_test.ipynb in VS Code, select the "argus" kernel, run top to bottom.
goto :eof

:err
echo.
echo *** Setup failed. See the error above. ***
exit /b 1
