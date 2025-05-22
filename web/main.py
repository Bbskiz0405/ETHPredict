import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
WEB_STATIC = Path(__file__).parent / "static"
WEB_TEMPLATES = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=WEB_STATIC), name="static")
templates = Jinja2Templates(directory=str(WEB_TEMPLATES))

import subprocess

@app.post("/run-predict")
async def run_predict():
    try:
        result = subprocess.run([
            "python3", str(BASE_DIR / "src/eth_future_prediction.py")
        ], cwd=str(BASE_DIR), capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # 執行完畢後同步靜態圖表
            copy_results_to_static()
            return {"success": True}
        else:
            return {"success": False, "detail": result.stderr or result.stdout}
    except Exception as e:
        return {"success": False, "detail": str(e)}

from datetime import datetime
import shutil

BACKTEST_RESULTS_DIR = BASE_DIR / "backtest_results"

@app.post("/run-backtest")
async def run_backtest():
    try:
        # 執行時間字串
        run_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 確保目錄存在
        BACKTEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 執行 lstm_backtest.py 並傳遣 run_id 參數
        print(f"[DEBUG] Running backtest with run_id: {run_time_str}")
        result = subprocess.run([
            "python3", 
            str(BASE_DIR / "src/lstm_backtest.py"),
            run_time_str  # 傳遣 run_id 作為命令列參數
        ], cwd=str(BASE_DIR), capture_output=True, text=True, timeout=1200)
        
        print(f"[DEBUG] Backtest process completed with return code: {result.returncode}")
        if result.returncode != 0:
            print(f"[ERROR] Backtest error: {result.stderr}")
        
        # 同步靜態檔案
        copy_backtest_to_static()
        
        if result.returncode == 0:
            return {"success": True}
        else:
            return {"success": False, "detail": result.stderr or result.stdout}
    except Exception as e:
        print(f"[ERROR] Exception in run_backtest: {e}")
        return {"success": False, "detail": str(e)}

# 掉載回測靜態檔案
def copy_backtest_to_static():
    static_base = WEB_STATIC / "backtest"
    static_base.mkdir(exist_ok=True)
    print(f"[DEBUG] Copying backtest results from {BACKTEST_RESULTS_DIR} to {static_base}")
    if BACKTEST_RESULTS_DIR.exists():
        for d in BACKTEST_RESULTS_DIR.iterdir():
            if d.is_dir():
                static_subdir = static_base / d.name
                static_subdir.mkdir(exist_ok=True)
                print(f"[DEBUG] Processing backtest directory: {d.name}")
                for f in d.iterdir():
                    dst = static_subdir / f.name
                    if f.is_file() and not dst.exists():
                        print(f"[DEBUG] Copying file: {f.name} to {dst}")
                        shutil.copy(str(f), str(dst))

@app.get("/api/backtest_runs")
async def get_backtest_runs():
    runs = []
    if BACKTEST_RESULTS_DIR.exists():
        for d in sorted(BACKTEST_RESULTS_DIR.iterdir(), reverse=True):
            if d.is_dir() and any(d.iterdir()):
                runs.append(d.name)
    return {"runs": runs}

@app.get("/backtest/{run_id}", response_class=HTMLResponse)
async def backtest_detail(request: Request, run_id: str):
    run_folder = BACKTEST_RESULTS_DIR / run_id
    files = []
    file_url_prefix = f"/static/backtest/{run_id}/"
    if run_folder.exists():
        files = [f.name for f in run_folder.iterdir() if f.is_file()]
    return templates.TemplateResponse(
        "backtest_detail.html",
        {"request": request, "run_id": run_id, "files": files, "file_url_prefix": file_url_prefix}
    )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # 列出所有 results 子資料夾（排序新到舊）
    runs = []
    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
            if d.is_dir() and any(d.iterdir()):
                runs.append(d.name)
    return templates.TemplateResponse("index.html", {"request": request, "runs": runs})

@app.get("/result/{run_id}", response_class=HTMLResponse)
async def result_detail(request: Request, run_id: str):
    run_folder = RESULTS_DIR / run_id
    prediction_json = run_folder / "prediction_results.json"
    chart_img = run_folder / "eth_future_prediction.png"
    # 讀取預測結果
    prediction_data = []
    if prediction_json.exists():
        with open(prediction_json, "r", encoding="utf-8") as f:
            prediction_data = json.load(f)
    chart_url = f"/static/{run_id}/eth_future_prediction.png" if chart_img.exists() else None
    return templates.TemplateResponse(
        "result_detail.html",
        {"request": request, "run_id": run_id, "prediction_data": prediction_data, "chart_url": chart_url}
    )

# 靜態檔案掛載（每次執行的圖表）
# 將每個 results 子資料夾下的圖表複製到 static 下對應資料夾（啟動時處理）
def copy_results_to_static():
    if not WEB_STATIC.exists():
        WEB_STATIC.mkdir(parents=True)
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            if d.is_dir():
                static_subdir = WEB_STATIC / d.name
                static_subdir.mkdir(exist_ok=True)
                img_src = d / "eth_future_prediction.png"
                img_dst = static_subdir / "eth_future_prediction.png"
                if img_src.exists() and not img_dst.exists():
                    try:
                        import shutil
                        shutil.copy(str(img_src), str(img_dst))
                    except Exception:
                        pass

copy_results_to_static()
