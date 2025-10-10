@echo off
setlocal enableextensions enabledelayedexpansion
:: =============================================================================
:: WALT — Wrapper for Adaptive LLM Tuning
:: Windows CLI Launcher (init/run/stop/check/... one-file)
::
:: Copyright (c) 2025 Wanseok Kim (김완석)
:: License: MIT
:: Contact: ws-kim@naver.com, wskim4home@gmail.com
:: =============================================================================

:: ------------------------
:: Defaults (can override by .env or your shell env)
:: ------------------------
set "WALT_HOME=%USERPROFILE%\.walt"
set "WALT_PORT=5000"
set "WALT_HOST=0.0.0.0"
set "WALT_AUTH_TOKEN=s3cr3t"
set "WALT_OFFLINE=0"
set "WALT_APP_FILE=walt.py"
set "WALT_LOG=%WALT_HOME%\walt_debug.log"
set "WALT_PID=%WALT_HOME%\walt.pid"
set "WALT_ENV_FILE=%WALT_HOME%\walt.env"

:: ------------------------
:: Entry
:: ------------------------
if "%~1"=="" goto :help

:: normalize subcommand to lower
set "SUB=%~1"
call :tolower SUB

:: allow flags after subcommand (e.g., run --port=8001 --offline --token=XXX)
set "FLAGS="
if not "%~2"=="" (
  set "FLAGS=%*"
  set "FLAGS=%FLAGS:* =%"
)

:: Load .env if present (key=value)
if exist "%WALT_ENV_FILE%" call :load_env "%WALT_ENV_FILE%"

:: Ensure folder
if not exist "%WALT_HOME%" mkdir "%WALT_HOME%" >nul 2>&1

:: Dispatch
if "%SUB%"=="init"         goto :init
if "%SUB%"=="run"          goto :run
if "%SUB%"=="stop"         goto :stop
if "%SUB%"=="check"        goto :check
if "%SUB%"=="status"       goto :status
if "%SUB%"=="retrain"      goto :retrain
if "%SUB%"=="switch-model" goto :switchmodel
if "%SUB%"=="stream-test"  goto :stream
if "%SUB%"=="logs"         goto :logs
if "%SUB%"=="update"       goto :update
if "%SUB%"=="clean"        goto :clean
if "%SUB%"=="env"          goto :env
if "%SUB%"=="help"         goto :help

echo [WALT] Unknown command: %~1
goto :help

:: =============================================================================
:: Helpers
:: =============================================================================

:tolower
setlocal enabledelayedexpansion
set "varname=%~1"
for /f "delims=" %%a in ('powershell -nop -c "'%2'.ToLowerInvariant()"') do set "lower=%%~a"
endlocal & set "%varname%=%lower%"
goto :eof

:load_env
:: simple KEY=VALUE parser
for /f "usebackq tokens=1,* delims==" %%A in ("%~1") do (
  if not "%%A"=="" if not "%%B"=="" set "%%A=%%B"
)
goto :eof

:detect_python
:: Prefer venv python if exists
set "VENV=%WALT_HOME%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"
if exist "%PY%" (
  goto :eof
)
:: else try system python
for /f "delims=" %%p in ('where python 2^>nul') do (
  set "PY=%%p"
  goto :gotpy
)
:: else try py -3
where py >nul 2>&1
if not errorlevel 1 (
  set "PY=py -3"
  goto :gotpy
)
echo [WALT] Python not found. Please install Python 3.10+ and retry.
exit /b 1
:gotpy
goto :eof

:ensure_venv
if exist "%WALT_HOME%\venv\Scripts\python.exe" goto :eof
echo [WALT] Creating venv at %WALT_HOME%\venv ...
call :detect_python || exit /b 1
if /i "%PY%"=="py -3" (
  %PY% -m venv "%WALT_HOME%\venv"
) else (
  "%PY%" -m venv "%WALT_HOME%\venv"
)
if errorlevel 1 (
  echo [WALT] venv creation failed.
  exit /b 1
)
set "PY=%WALT_HOME%\venv\Scripts\python.exe"
set "PIP=%WALT_HOME%\venv\Scripts\pip.exe"
goto :eof

:install_reqs
call :ensure_venv || exit /b 1
set "PY=%WALT_HOME%\venv\Scripts\python.exe"
set "PIP=%WALT_HOME%\venv\Scripts\pip.exe"
echo [WALT] Upgrading pip ...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 echo [WALT] pip upgrade failed (continuing) ...
echo [WALT] Installing core deps (may take a while) ...
"%PIP%" install --upgrade flask flask-cors transformers datasets psutil nltk beautifulsoup4 pypdf2 requests
:: Try torch (CPU wheel is fine for most). If you want CUDA, install manually following PyTorch site.
"%PIP%" install --upgrade torch || echo [WALT] torch install failed (you can retry later)
:: Optional: faiss-cpu (Windows wheels can be flaky; ignore failure)
"%PIP%" install --upgrade faiss-cpu 2>nul 1>nul || echo [WALT] faiss-cpu unavailable on this platform (ok)
:: Cache dir
if not exist "%WALT_HOME%\hf_cache" mkdir "%WALT_HOME%\hf_cache" >nul 2>&1
echo [WALT] Done.
goto :eof

:parse_flags_run
:: Supported flags: --port=, --host=, --offline, --token=, --app=, --env=FILE
for %%A in (%FLAGS%) do (
  set "ARG=%%~A"
  echo !ARG! | findstr /b /c:"--port=" >nul && for /f "tokens=2 delims==" %%P in ("!ARG!") do set "WALT_PORT=%%~P"
  echo !ARG! | findstr /b /c:"--host=" >nul && for /f "tokens=2 delims==" %%H in ("!ARG!") do set "WALT_HOST=%%~H"
  if "!ARG!"=="--offline" set "WALT_OFFLINE=1"
  echo !ARG! | findstr /b /c:"--token=" >nul && for /f "tokens=2 delims==" %%T in ("!ARG!") do set "WALT_AUTH_TOKEN=%%~T"
  echo !ARG! | findstr /b /c:"--app=" >nul && for /f "tokens=2 delims==" %%F in ("!ARG!") do set "WALT_APP_FILE=%%~F"
  echo !ARG! | findstr /b /c:"--env=" >nul && for /f "tokens=2 delims==" %%E in ("!ARG!") do set "WALT_ENV_FILE=%%~E"
)
goto :eof

:curl_or_pwsh_json
:: %1=url
:: tries curl, else powershell
where curl >nul 2>&1
if not errorlevel 1 (
  curl -s -S "%~1"
  goto :eof
)
powershell -nop -c "try { irm -UseBasicParsing '%~1' | ConvertTo-Json -Depth 6 } catch { exit 1 }"
goto :eof

:curl_or_pwsh_post_bearer
:: %1=url %2=token
where curl >nul 2>&1
if not errorlevel 1 (
  curl -s -S -X POST -H "Authorization: Bearer %~2" "%~1"
  goto :eof
)
powershell -nop -c "try { irm -Method Post -Headers @{Authorization='Bearer %~2'} -Uri '%~1' } catch { exit 1 }"
goto :eof

:curl_or_pwsh_post_json
:: %1=url %2=json
where curl >nul 2>&1
if not errorlevel 1 (
  curl -s -S -H "Content-Type: application/json" -d "%~2" "%~1"
  goto :eof
)
powershell -nop -c "try { irm -Method Post -ContentType 'application/json' -Body '%~2' -Uri '%~1' } catch { exit 1 }"
goto :eof

:find_pid_by_port
:: sets FOUND_PID var if any
set "FOUND_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /i ":%WALT_PORT% " ^| findstr /i "LISTENING"') do (
  set "FOUND_PID=%%P"
)
goto :eof

:: =============================================================================
:: Commands
:: =============================================================================

:init
echo [WALT] Initializing...
call :install_reqs || exit /b 1
:: Save defaults to .env if missing
if not exist "%WALT_ENV_FILE%" (
  >"%WALT_ENV_FILE%" (
    echo WALT_HOME=%WALT_HOME%
    echo WALT_PORT=%WALT_PORT%
    echo WALT_HOST=%WALT_HOST%
    echo WALT_AUTH_TOKEN=%WALT_AUTH_TOKEN%
    echo WALT_OFFLINE=%WALT_OFFLINE%
    echo WALT_APP_FILE=%WALT_APP_FILE%
  )
  echo [WALT] Wrote %WALT_ENV_FILE%
)
echo [WALT] init complete.
exit /b 0

:run
call :parse_flags_run
call :ensure_venv || exit /b 1

:: Resolve app file (prefer walt.py, fallback to saltl.py)
if not exist "%WALT_APP_FILE%" (
  if exist "walt.py" (set "WALT_APP_FILE=walt.py") else (
    if exist "saltl.py" (set "WALT_APP_FILE=saltl.py") else (
      echo [WALT] Cannot find app file (walt.py/saltl.py or --app=PATH).
      exit /b 1
    )
  )
)

:: Kill previous instance on same port if needed
call :find_pid_by_port
if not "%FOUND_PID%"=="" (
  echo [WALT] Port %WALT_PORT% is already in use by PID %FOUND_PID%. Attempting to stop it...
  powershell -nop -c "try { Stop-Process -Id %FOUND_PID% -Force } catch {}"
  timeout /t 1 >nul
)

:: Export env to child proc only
set "AUTH_TOKEN=%WALT_AUTH_TOKEN%"
set "SALT_BASE_DIR=%WALT_HOME%"
set "WALT_OFFLINE=%WALT_OFFLINE%"
set "PORT=%WALT_PORT%"

echo [WALT] Starting server on http://127.0.0.1:%WALT_PORT%  (host=%WALT_HOST%)
echo [WALT] Logs: %WALT_LOG%

:: Start detached via PowerShell to capture PID
powershell -nop -c ^
  "$env:AUTH_TOKEN='%WALT_AUTH_TOKEN%';$env:SALT_BASE_DIR='%WALT_HOME%';$env:WALT_OFFLINE='%WALT_OFFLINE%';$env:PORT='%WALT_PORT%';" ^
  "$p=Start-Process -FilePath '%WALT_HOME%\venv\Scripts\python.exe' -ArgumentList '%CD%\%WALT_APP_FILE%' -PassThru -WindowStyle Hidden;" ^
  "Set-Content -Path '%WALT_PID%' -Value $p.Id;"

if exist "%WALT_PID%" (
  for /f "usebackq delims=" %%i in ("%WALT_PID%") do set "PID=%%~i"
  echo [WALT] PID: %PID%
) else (
  echo [WALT] Failed to write PID. Is PowerShell allowed?
)

:: Wait a moment then check health
timeout /t 2 >nul
call :check
exit /b 0

:stop
if exist "%WALT_PID%" (
  for /f "usebackq delims=" %%i in ("%WALT_PID%") do set "PID=%%~i"
  if not "%PID%"=="" (
    echo [WALT] Stopping PID %PID% ...
    powershell -nop -c "try { Stop-Process -Id %PID% -Force } catch {}"
    del /q "%WALT_PID%" >nul 2>&1
    echo [WALT] Stopped.
    exit /b 0
  )
)
:: fallback by port
call :find_pid_by_port
if not "%FOUND_PID%"=="" (
  echo [WALT] Stopping by port (PID %FOUND_PID%) ...
  powershell -nop -c "try { Stop-Process -Id %FOUND_PID% -Force } catch {}"
  echo [WALT] Stopped.
  exit /b 0
)
echo [WALT] No running process found.
exit /b 0

:check
call :curl_or_pwsh_json "http://127.0.0.1:%WALT_PORT%/health" >nul 2>&1
if errorlevel 1 (
  echo [WALT] Health: DOWN
  exit /b 1
) else (
  echo [WALT] Health: OK
  exit /b 0
)

:status
echo [WALT] Fetching /status ...
call :curl_or_pwsh_json "http://127.0.0.1:%WALT_PORT%/status"
echo.
exit /b 0

:retrain
:: usage: waltcli retrain [TOKEN]
set "TOK=%~2"
if "%TOK%"=="" set "TOK=%WALT_AUTH_TOKEN%"
echo [WALT] POST /retrain
call :curl_or_pwsh_post_bearer "http://127.0.0.1:%WALT_PORT%/retrain" "%TOK%"
echo.
exit /b 0

:switchmodel
:: usage: waltcli switch-model <hf_model_id>
if "%~2"=="" (
  echo Usage: waltcli switch-model ^<hf_model_id^>
  echo e.g.   waltcli switch-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  exit /b 1
)
set "MODEL=%~2"
echo [WALT] Switching model to %MODEL% (via /chat warm call) ...
set "PAYLOAD={\"message\":\"모델 스위치 워밍업\"}"
call :curl_or_pwsh_post_json "http://127.0.0.1:%WALT_PORT%/chat?model=%MODEL%" "%PAYLOAD%"
echo.
exit /b 0

:stream
:: quick SSE test
set "PAYLOAD={\"message\":\"스트리밍 테스트 문장입니다. 문장 단위로 출력되는지 확인합니다.\"}"
echo [WALT] /chat?stream=true
where curl >nul 2>&1
if not errorlevel 1 (
  curl -N -H "Content-Type: application/json" -d "%PAYLOAD%" "http://127.0.0.1:%WALT_PORT%/chat?stream=true"
) else (
  echo PowerShell does not easily show SSE chunks continuously; using one-shot Invoke-RestMethod is not suitable.
  echo Please install curl or use a browser hitting: http://127.0.0.1:%WALT_PORT%  and enable "스트리밍".
)
echo.
exit /b 0

:logs
if exist "%WALT_LOG%" (
  echo [WALT] Last 200 lines of %WALT_LOG%
  powershell -nop -c "Get-Content -Path '%WALT_LOG%' -Tail 200"
) else (
  echo [WALT] Log file not found: %WALT_LOG%
)
exit /b 0

:update
:: If in a git repo, try git pull; else just upgrade libs
where git >nul 2>&1
if not errorlevel 1 (
  git rev-parse --is-inside-work-tree >nul 2>&1
  if not errorlevel 1 (
    echo [WALT] git pull ...
    git pull
  )
)
call :ensure_venv
"%WALT_HOME%\venv\Scripts\pip.exe" install --upgrade flask flask-cors transformers datasets psutil nltk beautifulsoup4 pypdf2 requests torch
echo [WALT] update done.
exit /b 0

:clean
echo [WALT] This will remove venv and caches under %WALT_HOME%.
choice /m "Proceed?" /c YN >nul
if errorlevel 2 (
  echo [WALT] Aborted.
  exit /b 0
)
call :stop >nul 2>&1
rmdir /s /q "%WALT_HOME%\venv" 2>nul
rmdir /s /q "%WALT_HOME%\hf_cache" 2>nul
del /q "%WALT_LOG%" 2>nul
echo [WALT] Clean complete.
exit /b 0

:env
echo WALT_HOME=%WALT_HOME%
echo WALT_PORT=%WALT_PORT%
echo WALT_HOST=%WALT_HOST%
echo WALT_AUTH_TOKEN=%WALT_AUTH_TOKEN%
echo WALT_OFFLINE=%WALT_OFFLINE%
echo WALT_APP_FILE=%WALT_APP_FILE%
echo WALT_ENV_FILE=%WALT_ENV_FILE%
echo LOG=%WALT_LOG%
echo PID=%WALT_PID%
exit /b 0

:help
echo.
echo WALT CLI (waltcli.cmd) — Wrapper for Adaptive LLM Tuning
echo.
echo Usage:
echo   waltcli init                     ^> venv 생성, 필수 패키지 설치, 기본 .env 작성
echo   waltcli run [--port=5000] [--host=0.0.0.0] [--offline] [--token=XXX] [--app=FILE]
echo   waltcli stop                     ^> 서버 중지(PID 또는 포트 기반)
echo   waltcli check                    ^> /health 확인
echo   waltcli status                   ^> /status JSON 출력
echo   waltcli retrain [TOKEN]          ^> /retrain 호출(Bearer)
echo   waltcli switch-model MODEL_ID    ^> /chat로 수동 모델 스위치 워밍업
echo   waltcli stream-test              ^> SSE 스트리밍 테스트(curl 필요)
echo   waltcli logs                     ^> 최근 로그 200줄
echo   waltcli update                   ^> 라이브러리 업데이트(또는 git pull)
echo   waltcli clean                    ^> venv/캐시/로그 정리
echo   waltcli env                      ^> 현재 환경 변수 표시
echo   waltcli help                     ^> 본 도움말
echo.
echo Examples:
echo   waltcli init
echo   waltcli run --port=8001 --offline --token=mytoken
echo   waltcli retrain mytoken
echo   waltcli switch-model codellama/CodeLlama-7b-Instruct-hf
echo   waltcli stream-test
echo.
exit /b 0