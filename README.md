# ETH Futures Prediction Project

ETH Futures Prediction 是一個基於 LSTM 深度學習模型的 ETH 期貨價格預測系統。本專案整合了數據獲取、預處理、模型訓練、預測生成和回測分析等功能，旨在為 ETH 期貨交易提供數據驅動的決策支持。

## 主要功能
- **數據獲取與處理**
  - 自動獲取 ETH 期貨歷史數據
  - 實時數據更新與同步
  - 數據清洗與預處理

- **模型訓練**
  - 基於 LSTM 的時間序列預測模型
  - 自動超參數優化
  - 模型性能評估

- **預測分析**
  - 短期與長期價格預測
  - 交易信號生成
  - 風險評估與管理

- **回測系統**
  - 策略回測與優化
  - 權益曲線分析
  - 交易績效評估

## 目錄結構
- `models/` - 存放訓練好的模型檔案
  - `lstm_backtest_model.keras` - LSTM 模型的權重檔案
- `results/` - 存放預測結果和分析圖表
  - `equity_curve.png` - 權益曲線圖
  - `eth_future_prediction.png` - ETH 期貨預測圖
  - `eth_price_signals.png` - ETH 價格信號圖
  - `prediction_results.json` - 預測結果數據
- `src/` - 主要的程式碼檔案
  - `eth_future_prediction.py` - ETH 期貨預測主程式
  - `lstm_backtest.py` - LSTM 模型訓練和回測程式

## 安裝依賴
本專案需要以下 Python 套件：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 環境準備
1. 確保 Python 3.8+ 已安裝
2. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

### 2. 預測程式
運行 ETH 期貨預測程式：
```bash
python src/eth_future_prediction.py
```

### 3. 回測分析
運行 LSTM 模型訓練和回測：
```bash
python src/lstm_backtest.py
```

## 輸出檔案說明

### 預測結果圖表
- `equity_curve.png`: 顯示模型的投資績效曲線，包括總收益和最大回撤
- `eth_future_prediction.png`: 顯示 ETH 期貨價格的預測結果和實際價格對比
- `eth_price_signals.png`: 顯示交易信號和價格走勢

### 預測數據
- `prediction_results.json`: 包含預測結果的詳細數據，包括：
  - 預測日期
  - 預測價格
  - 實際價格
  - 交易信號
  - 收益率

## 注意事項
1. 本專案使用 LSTM 模型進行時間序列預測，建議使用 GPU 加速訓練過程
2. 預測結果僅供參考，不構成投資建議
3. 模型性能可能會受到市場環境和數據質量的影響

## 開發者指南

### 環境配置
建議使用 virtualenv 或 conda 建立獨立的開發環境：
```bash
# 使用 virtualenv
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 或使用 conda
conda create -n ethtrain python=3.8
conda activate ethtrain
```

### 常見問題
1. **數據獲取問題**
   - 確保網路連接正常
   - 檢查 API 限制

2. **模型訓練問題**
   - 調整學習率
   - 增加訓練批次
   - 調整模型結構
