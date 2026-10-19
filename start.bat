@echo off
rem Safe-Cadence Windows entry point. Keep ASCII + CRLF for cmd.exe.
rem Use Python UTF-8 mode for Chinese docs, filenames and logs.
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
cd /d "%~dp0"
rem Branch with GOTO, not block expansion, to preserve the process exit code.
if exist ".venv\Scripts\python.exe" goto runvenv
where py >nul 2>nul
if not errorlevel 1 goto runpy
where python >nul 2>nul
if not errorlevel 1 goto runpython
echo ERROR: Python 3.10+ was not found. Please install Python first.
set "EXITCODE=9009"
goto done
:runvenv
".venv\Scripts\python.exe" -X utf8 "%~dp0launcher.py" %*
goto capture
:runpy
py -3 -X utf8 "%~dp0launcher.py" %*
goto capture
:runpython
python -X utf8 "%~dp0launcher.py" %*
:capture
rem Capture ERRORLEVEL *after* Python exits (outside parentheses).
set "EXITCODE=%ERRORLEVEL%"
:done
rem Only pause interactive double-click; command-line automation must return.
if "%~1"=="" pause
exit /b %EXITCODE%
