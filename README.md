# 以深度學習驅動的以太坊量化交易系統

## 專案摘要

此專案為一套完整的以太坊(ETH)量化交易解決方案，透過深度學習技術建立預測模型並實現策略回測框架。本系統結合了LSTM神經網路、技術分析指標與跨資產相關性分析，旨在捕捉加密貨幣市場的非線性模式並產生具有統計優勢的交易信號。

### 為何適合「優式AI量化新星計畫 2025」

本專案完美對應UC Capital實習需求：
- **量化交易實踐**：從原始市場數據到可執行交易策略的完整工作流程
- **深度學習應用**：運用LSTM架構建立時間序列預測模型
- **策略回測**：嚴謹的策略評估框架與績效指標視覺化
- **跨資產分析**：結合BTC與ETH數據以提升預測準確性
- **全端系統**：整合後端算法與前端視覺化介面的完整解決方案

## 核心技術與功能

### 量化模型設計
- LSTM神經網路架構用於捕捉價格時間序列的長短期依賴關係
- 融合跨資產數據(ETH+BTC)以提高預測穩定性
- 滾動時間窗口訓練與預測模式確保模型適應性
- 動態閾值調整的交易信號生成機制

### 回測系統與績效評估
- 完整的策略回測框架，支持做多/做空策略
- 關鍵績效指標計算與視覺化：
  - 累積報酬率曲線
  - 信號準確率分析
  - 單筆交易盈虧分布
- 交易信號與價格走勢疊加圖表

### 網頁應用與視覺化
- FastAPI打造的高效後端服務
- Bootstrap響應式前端設計
- 即時模型訓練與回測執行
- 歷史回測結果瀏覽與比較

## 系統架構

```
.
├── src/                         # 核心算法模組
│   ├── eth_future_prediction.py # ETH預測模型
│   └── lstm_backtest.py         # 策略回測引擎
├── web/                         # 網頁應用框架
│   ├── main.py                  # FastAPI服務
│   ├── static/                  # 靜態資源
│   └── templates/               # 前端模板
├── backtest_results/            # 回測結果存儲
│   └── [timestamp]/             # 時間戳分類
│       ├── equity_curve.png     # 權益曲線
│       ├── eth_price_signals.png# 價格信號圖
│       ├── backtest_performance.json # 績效指標
│       ├── lstm_model.keras     # 訓練模型
│       └── backtest_data.csv    # 交易數據
└── models/                      # 模型庫
```

## 技術實現細節

### 數據處理與特徵工程
- **多時間尺度特徵**：結合4小時K線與移動平均指標
- **跨資產關聯**：引入BTC價格數據作為輔助特徵
- **標準化處理**：MinMaxScaler確保數據尺度一致性
- **序列構建**：72小時(18個時間窗口)的滑動窗口序列化

### 深度學習模型
- **LSTM架構**：64個神經元的單層LSTM網絡
- **Dropout正則化**：0.2的丟棄率防止過擬合
- **早停機制**：基於驗證集損失的模型訓練自動停止
- **批量訓練**：批量大小32優化訓練效率

### 交易策略邏輯
- **預測比較信號**：預測價格與當前價格差異驅動的交易決策
- **雙向策略**：支持做多(看漲)與做空(看跌)操作
- **模型持久化**：訓練模型保存與載入機制

### 結果呈現與分析
- **自動數據同步**：結果自動歸類與網頁同步
- **多維度視覺化**：權益曲線、信號疊加圖等多種圖表
- **性能指標JSON**：標準化的績效數據導出

## 技術棧

- **Python 3.9+**: 核心程式語言
- **TensorFlow/Keras**: 深度學習框架
- **FastAPI**: 網頁後端服務
- **Pandas/NumPy**: 數據處理
- **Matplotlib**: 數據視覺化
- **yfinance**: 市場數據獲取
- **Bootstrap**: 前端框架

## 專案亮點與創新

1. **端到端解決方案**：從數據獲取、模型訓練到策略回測的完整流程
2. **跨資產關聯性**：利用BTC與ETH之間的相關性提升預測效果
3. **模型持久化與復用**：訓練模型的保存與載入機制增強系統效率
4. **即時執行與視覺化**：網頁界面支持一鍵執行複雜算法並視覺化結果
5. **分類存儲系統**：結構化的結果存儲與管理機制

## 發展潛力與未來計劃

- 擴展支持更多加密貨幣與傳統資產
- 整合更多技術指標與基本面數據
- 引入強化學習優化交易策略
- 開發實時交易API連接功能
- 建立策略組合與風險管理框架

---

# Deep Learning-Powered Ethereum Quantitative Trading System

## Project Summary

This project presents a comprehensive Ethereum (ETH) quantitative trading solution, leveraging deep learning techniques to build predictive models and implement a strategy backtesting framework. The system combines LSTM neural networks, technical indicators, and cross-asset correlation analysis to capture non-linear patterns in cryptocurrency markets and generate statistically advantageous trading signals.

### Why It's Ideal for UC Capital's AI Quantitative Trading Internship 2025

This project perfectly aligns with UC Capital's internship requirements:
- **Quantitative Trading Implementation**: Complete workflow from raw market data to executable trading strategies
- **Deep Learning Application**: LSTM architecture for time series prediction modeling
- **Strategy Backtesting**: Rigorous strategy evaluation framework and performance metrics visualization
- **Cross-Asset Analysis**: Integration of BTC and ETH data to enhance prediction accuracy
- **Full-Stack System**: Complete solution integrating backend algorithms and frontend visualization interfaces

## Core Technologies and Features

### Quantitative Model Design
- LSTM neural network architecture for capturing long and short-term dependencies in price time series
- Cross-asset data fusion (ETH+BTC) for improved prediction stability
- Rolling time window training and prediction mode ensuring model adaptability
- Dynamic threshold adjustment for trade signal generation

### Backtesting System and Performance Evaluation
- Complete strategy backtesting framework supporting long/short strategies
- Key performance indicator calculation and visualization:
  - Cumulative return curves
  - Signal accuracy analysis
  - Single trade profit/loss distribution
- Trading signals and price trend overlay charts

### Web Application and Visualization
- Efficient backend services built with FastAPI
- Responsive frontend design with Bootstrap
- Real-time model training and backtesting execution
- Historical backtest result browsing and comparison

## Technical Implementation Details

### Data Processing and Feature Engineering
- **Multi-timeframe features**: Combination of 4-hour candles and moving average indicators
- **Cross-asset correlation**: Introduction of BTC price data as auxiliary features
- **Normalization**: MinMaxScaler ensuring data scale consistency
- **Sequence construction**: 72-hour (18 time windows) sliding window sequencing

### Deep Learning Model
- **LSTM architecture**: Single-layer LSTM network with 64 neurons
- **Dropout regularization**: 0.2 dropout rate to prevent overfitting
- **Early stopping**: Automatic training termination based on validation loss
- **Batch training**: Batch size 32 optimizing training efficiency

### Trading Strategy Logic
- **Prediction comparison signals**: Trading decisions driven by difference between predicted and current prices
- **Bidirectional strategy**: Support for long (bullish) and short (bearish) operations
- **Model persistence**: Training model saving and loading mechanism

## Technical Stack

- **Python 3.9+**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **FastAPI**: Web backend services
- **Pandas/NumPy**: Data processing
- **Matplotlib**: Data visualization
- **yfinance**: Market data acquisition
- **Bootstrap**: Frontend framework

## Project Highlights and Innovation

1. **End-to-end solution**: Complete workflow from data acquisition to model training to strategy backtesting
2. **Cross-asset correlation**: Leveraging correlation between BTC and ETH to enhance prediction effects
3. **Model persistence and reuse**: Saving and loading mechanisms for trained models to enhance system efficiency
4. **Real-time execution and visualization**: Web interface supporting one-click execution of complex algorithms and result visualization
5. **Classified storage system**: Structured result storage and management mechanism
