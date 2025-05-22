# ETH 期貨預測與交易系統

基於 LSTM 神經網路的 ETH 期貨價格預測和回測系統。

## 專案結構

```
.
├── src/
│   ├── eth_future_prediction.py   # ETH 期貨預測模型
│   └── lstm_backtest.py          # 回測系統
├── web/
│   ├── main.py                   # FastAPI 網頁伺服器
│   ├── static/                   # 網頁靜態檔案
│   └── templates/
│       ├── index.html           # 主頁面
│       ├── backtest_detail.html # 回測結果頁面
│       └── result_detail.html   # 預測結果頁面
├── backtest_results/            # 存放回測結果
│   └── [timestamp]/
│       ├── equity_curve.png     # 權益曲線圖
│       ├── eth_price_signals.png # 價格信號圖
│       ├── backtest_performance.json # 回測績效資料
│       ├── lstm_model.keras     # LSTM 模型檔案
│       └── backtest_data.csv    # 回測數據
└── models/                      # 訓練好的 LSTM 模型

## 主要功能

- 基於 LSTM 神經網路的 ETH 期貨價格即時預測
- 整合比特幣價格數據以提高預測準確性
- 技術指標分析（包括 RSI、MACD、布林帶等）
- 完整的回測系統
- 圖形化網頁介面進行模型訓練和回測
- 交易信號和績效指標的視覺化展示
- 自動組織和顯示結果

## 系統需求

- Python 3.9+
- FastAPI
- TensorFlow
- yfinance
- pandas
- numpy
- scikit-learn
- ta（技術分析庫）
- matplotlib

## 安裝設置

1. 克隆專案庫
2. 安裝依賴套件：
   ```bash
   pip install -r requirements.txt
   ```
3. 啟動網頁伺服器：
   ```bash
   cd web
   uvicorn main:app --reload
   ```

## 使用方法

1. 在瀏覽器中訪問 `http://localhost:8000`
2. 使用訓練按鈕訓練新的 LSTM 模型
3. 使用回測按鈕執行回測
4. 在回測結果頁面查看詳細結果

## 檔案組織

- 系統會根據時間戳自動組織結果
- 回測結果存放在 `backtest_results/[timestamp]/`
- 包含的檔案：
  - 權益曲線圖
  - 價格和信號圖表
  - JSON 格式的績效指標
  - 訓練好的模型
  - 交易數據

## API 端點

- `/run-predict`: 執行 ETH 價格預測
- `/run-backtest`: 執行回測
- `/`: 主頁面
- `/backtest/{run_id}`: 回測結果詳細頁面
- `/result/{run_id}`: 預測結果詳細頁面

## 注意事項

- 系統使用 72 小時的序列長度進行預測
- 預測時間範圍為未來 48 小時（12 個 4 小時週期）
- 交易信號根據可配置的閾值生成
- 所有結果會自動同步到網頁介面

---

# ETH Price Prediction and Trading System

A web-based system for Ethereum price prediction and backtesting using LSTM neural networks.

## Project Structure

```
.
├── src/
│   ├── eth_future_prediction.py   # ETH price prediction model
│   └── lstm_backtest.py          # Backtesting system
├── web/
│   ├── main.py                   # FastAPI web server
│   ├── static/                   # Static files for web interface
│   └── templates/
│       ├── index.html           # Main page
│       ├── backtest_detail.html # Backtest results view
│       └── result_detail.html   # Prediction results view
├── backtest_results/            # Stores backtest results
│   └── [timestamp]/
│       ├── equity_curve.png
│       ├── eth_price_signals.png
│       ├── backtest_performance.json
│       ├── lstm_model.keras
│       └── backtest_data.csv
└── models/                      # Trained LSTM models
```

## Features

- Real-time Ethereum price prediction using LSTM neural networks
- Integration with Bitcoin price data for better prediction accuracy
- Technical indicators including RSI, MACD, and Bollinger Bands
- Comprehensive backtesting system
- Web interface for model training and backtesting
- Visualization of trading signals and performance metrics
- Automatic result organization and display

## Requirements

- Python 3.9+
- FastAPI
- TensorFlow
- yfinance
- pandas
- numpy
- scikit-learn
- ta (Technical Analysis library)
- matplotlib

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the web server:
   ```bash
   cd web
   uvicorn main:app --reload
   ```

## Usage

1. Access the web interface at `http://localhost:8000`
2. Use the training button to train a new LSTM model
3. Use the backtest button to run backtesting with the latest model
4. View detailed results in the backtest detail page

## File Organization

- The system automatically organizes results by timestamp
- Backtest results are stored in `backtest_results/[timestamp]/`
- Results include:
  - Equity curve plots
  - Price and signal charts
  - Performance metrics in JSON format
  - Trained models
  - Trading data

## API Endpoints

- `/run-predict`: Runs ETH price prediction
- `/run-backtest`: Executes backtesting
- `/`: Main page
- `/backtest/{run_id}`: Detailed backtest results view
- `/result/{run_id}`: Detailed prediction results view

## Notes

- The system uses a 72-hour sequence length for predictions
- Predictions are made for the next 48 hours (12 4-hour periods)
- Trading signals are generated based on a configurable threshold
- All results are automatically synchronized to the web interface
