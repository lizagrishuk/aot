@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title AI Detective - Laboratory 6 - Local Streamlit

echo ==================================================
echo      AI Detective - research stand - Windows
echo ==================================================
echo.

if not exist "app.py" goto missing

py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if errorlevel 1 goto checkpath
set "PY_CMD=py -3.12"
goto found

:checkpath
python -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if errorlevel 1 goto nopython
set "PY_CMD=python"

:found
if exist ".venv\Scripts\python.exe" goto havevenv
echo [1/3] Creating virtual environment
%PY_CMD% -m venv .venv
if errorlevel 1 goto fail

:havevenv
echo [2/3] Checking Python and Streamlit version
".venv\Scripts\python.exe" -c "import sys,streamlit,rank_bm25; assert sys.version_info[:2] == (3,12); assert streamlit.__version__ == '1.56.0'" >nul 2>&1
if not errorlevel 1 goto already
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

:already
echo [3/3] Starting interactive stand
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 goto fail
goto end

:missing
echo ERROR: app.py not found. Unpack the complete ZIP first.
goto fail

:nopython
echo ERROR: Python 3.12 is required.
echo Install from https://www.python.org/downloads/
goto fail

:fail
echo.
echo ERROR: Stand did not start. Copy the lines above to the chat.
pause
:end
endlocal
