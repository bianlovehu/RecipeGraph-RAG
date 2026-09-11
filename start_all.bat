@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

set "VENV_PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo 未找到虚拟环境 Python: %VENV_PY%
  echo 请先创建虚拟环境或修改脚本中的路径。
  pause
  exit /b 1
)

echo [1/4] 启动 Milvus(含 etcd/minio)...
docker compose up -d
if errorlevel 1 (
  echo Milvus 启动失败，请检查 Docker Desktop。
  pause
  exit /b 1
)

echo [2/4] 启动 Neo4j...
pushd "data"
docker compose up -d
if errorlevel 1 (
  echo Neo4j 启动失败，请检查 Docker Desktop。
  popd
  pause
  exit /b 1
)
popd

echo [3/4] 检查关键依赖...
"%VENV_PY%" -c "import neo4j,pymilvus,fastapi,uvicorn" >nul 2>nul
if errorlevel 1 (
  echo 检测到依赖缺失，正在安装 requirements.txt ...
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo 依赖安装失败，请手动执行："%VENV_PY%" -m pip install -r requirements.txt
    pause
    exit /b 1
  )
)

echo [4/4] 启动 Web 服务（含前端）: http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
"%VENV_PY%" -m uvicorn api.app:app --host 0.0.0.0 --port 8000

endlocal
