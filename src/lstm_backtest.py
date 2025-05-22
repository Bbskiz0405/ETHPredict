import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import shutil

# ========== 參數 ==========
SEQ_LEN = 72
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
BACKTEST_RESULTS_BASE = os.path.join(BASE_DIR, 'backtest_results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BACKTEST_RESULTS_BASE, exist_ok=True)

def run_lstm_backtest(run_id=None):
    """
    執行 LSTM 回測與繪圖，圖片儲存於 backtest_results/{run_id}/。
    模型存於 models/。
    run_id: 執行 ID，用於建立專屬資料夾，若為 None 則使用當前時間戳
    """
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 建立儲存目錄
    result_dir = os.path.join(BACKTEST_RESULTS_BASE, run_id)
    os.makedirs(result_dir, exist_ok=True)
    print(f"[DEBUG] Results will be saved in: {result_dir}")
    # 資料處理
    ETH = yf.download('ETH-USD', period='180d', interval='4h')
    BTC = yf.download('BTC-USD', period='180d', interval='4h')
    ETH['MA5'] = ETH['Close'].rolling(window=5).mean()
    ETH['MA10'] = ETH['Close'].rolling(window=10).mean()
    BTC['MA5'] = BTC['Close'].rolling(window=5).mean()
    BTC['MA10'] = BTC['Close'].rolling(window=10).mean()
    features = pd.DataFrame(index=ETH.index)
    features['ETH_Close'] = ETH['Close']
    features['ETH_MA5'] = ETH['MA5']
    features['ETH_MA10'] = ETH['MA10']
    features['BTC_Close'] = BTC['Close']
    features['BTC_MA5'] = BTC['MA5']
    features['BTC_MA10'] = BTC['MA10']
    features = features.dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN - 1):
        X.append(scaled[i:(i+SEQ_LEN)])
        y.append(scaled[i+SEQ_LEN, 0])  # ETH_Close
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 模型訓練/載入
    model_path = os.path.join(MODEL_DIR, 'lstm_backtest_model.keras')
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential([
            LSTM(64, input_shape=(SEQ_LEN, X.shape[2])),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[es])
        model.save(model_path)

    preds = model.predict(X_test)
    eth_close_min = scaler.data_min_[0]
    eth_close_max = scaler.data_max_[0]
    y_test_real = y_test * (eth_close_max - eth_close_min) + eth_close_min
    preds_real = preds.flatten() * (eth_close_max - eth_close_min) + eth_close_min
    current_real = X_test[:, -1, 0] * (eth_close_max - eth_close_min) + eth_close_min
    signal = np.where(preds_real > current_real, 1, -1)
    future_real = np.roll(y_test_real, -1)
    ret = (future_real - current_real) * signal
    ret = ret[:-1]
    equity = np.cumsum(ret)
    total_return = equity[-1]
    win_rate = np.mean(ret > 0)
    print(f'Total Return: {total_return:.2f} USD')
    print(f'Win Rate: {win_rate*100:.2f}%')



    # 畫圖，只產生一張主要結果圖（equity_curve.png）
    x_dates = features.index[-len(equity):]
    plt.figure(figsize=(12,5))
    plt.plot(x_dates, equity, label='Equity Curve')
    plt.title('LSTM Backtest Equity Curve (Long/Short)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (USD)')
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    save_path = os.path.join(result_dir, 'equity_curve.png')
    plt.savefig(save_path)
    print(f"[DEBUG] Saved equity curve plot to: {save_path}")
    plt.close()
    
    # 畫ETH價格與交易訊號圖
    plt.figure(figsize=(12,5))
    plt.plot(x_dates, y_test_real[-len(equity):], label='ETH Price')
    plt.scatter(x_dates, y_test_real[-len(equity):], c=signal[:-1], cmap='coolwarm', alpha=0.7, label='Signals')
    plt.title('ETH Price and Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('ETH Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    save_path = os.path.join(result_dir, 'eth_price_signals.png')
    plt.savefig(save_path)
    print(f"[DEBUG] Saved price signals plot to: {save_path}")
    plt.close()

    # 儲存績效數據
    perf = {
        'total_return': float(total_return),
        'win_rate': float(win_rate),
    }
    perf_path = os.path.join(result_dir, 'backtest_performance.json')
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(perf, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] Saved performance json to: {perf_path}")
    print(f"Backtest results saved to {result_dir}")
    
    # 儲存訓練資料到CSV
    df_results = pd.DataFrame({
        'Date': x_dates,
        'ETH_Price': y_test_real[-len(equity):],
        'Signal': signal[:-1],
        'Return': ret,
        'Equity': equity
    })
    csv_path = os.path.join(result_dir, 'backtest_data.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"[DEBUG] Saved backtest data to: {csv_path}")
    
    # 儲存模型到結果目錄
    model_result_path = os.path.join(result_dir, 'lstm_model.keras')
    model.save(model_result_path)
    print(f"[DEBUG] Saved model to: {model_result_path}")
    
    # 回傳結果路徑
    return {
        'result_dir': result_dir
    }

# ========== CLI 執行入口 ==========
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
        print(f"[DEBUG] Received run_id from CLI: {run_id}")
    else:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"[DEBUG] No run_id passed, using current timestamp: {run_id}")
    try:
        run_lstm_backtest(run_id)
    except Exception as e:
        print(f"[ERROR] Exception in run_lstm_backtest: {e}")
        import traceback
        traceback.print_exc()

    import pandas as pd
    import numpy as np
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    import os

    SEQ_LEN = 72
    MODEL_PATH = 'lstm_backtest_model.keras'

    ETH = yf.download('ETH-USD', period='180d', interval='4h')
    BTC = yf.download('BTC-USD', period='180d', interval='4h')

    ETH['MA5'] = ETH['Close'].rolling(window=5).mean()
    ETH['MA10'] = ETH['Close'].rolling(window=10).mean()
    BTC['MA5'] = BTC['Close'].rolling(window=5).mean()
    BTC['MA10'] = BTC['Close'].rolling(window=10).mean()

    features = pd.DataFrame(index=ETH.index)
    features['ETH_Close'] = ETH['Close']
    features['ETH_MA5'] = ETH['MA5']
    features['ETH_MA10'] = ETH['MA10']
    features['BTC_Close'] = BTC['Close']
    features['BTC_MA5'] = BTC['MA5']
    features['BTC_MA10'] = BTC['MA10']
    features = features.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN - 1):
        X.append(scaled[i:(i+SEQ_LEN)])
        y.append(scaled[i+SEQ_LEN, 0])  # ETH_Close
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 模型訓練/載入
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = Sequential([
            LSTM(64, input_shape=(SEQ_LEN, X.shape[2])),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[es])
        model.save(MODEL_PATH)

    preds = model.predict(X_test)

    eth_close_min = scaler.data_min_[0]
    eth_close_max = scaler.data_max_[0]
    y_test_real = y_test * (eth_close_max - eth_close_min) + eth_close_min
    preds_real = preds.flatten() * (eth_close_max - eth_close_min) + eth_close_min

    # 決策：預測>現價做多，預測<現價做空
    current_real = X_test[:, -1, 0] * (eth_close_max - eth_close_min) + eth_close_min
    signal = np.where(preds_real > current_real, 1, -1)
    # 報酬：下根真實收盤-現價
    future_real = np.roll(y_test_real, -1)
    ret = (future_real - current_real) * signal
    ret = ret[:-1]  # 最後一筆無法計算

    # 資產曲線
    equity = np.cumsum(ret)

    # 統計績效
    total_return = equity[-1]
    win_rate = np.mean(ret > 0)
    print(f'Total Return: {total_return:.2f} USD')
    print(f'Win Rate: {win_rate*100:.2f}%')

    # 檢查儲存目錄
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        prefix = os.path.join(save_dir, '')
    else:
        prefix = ''

    # 畫圖
    x_dates = features.index[-len(equity):]
    plt.figure(figsize=(12,5))
    plt.plot(x_dates, equity, label='Equity Curve')
    plt.title('LSTM Backtest Equity Curve (Long/Short)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (USD)')
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig(f'{prefix}equity_curve.png')
    plt.close()

    # 訊號標註在ETH收盤價圖
    eth_close_test = features['ETH_Close'][-len(equity):].values
    signal_plot = signal[:len(equity)]
    plt.figure(figsize=(12,5))
    plt.plot(x_dates, eth_close_test, label='ETH Close Price')
    plt.scatter(x_dates[signal_plot == 1], eth_close_test[signal_plot == 1], color='green', marker='^', label='Buy Signal')
    plt.scatter(x_dates[signal_plot == -1], eth_close_test[signal_plot == -1], color='red', marker='v', label='Sell Signal')
    plt.title('ETH Price with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig(f'{prefix}eth_price_signals.png')
    plt.close()

    # 進階訊號圖
    plt.figure(figsize=(14,6))
    plt.plot(x_dates, eth_close_test, label='ETH True Close', color='black')
    long_idx = np.where(signal_plot == 1)[0]
    plt.scatter(np.array(x_dates)[long_idx], eth_close_test[long_idx], marker='^', color='green', label='Long Signal')
    short_idx = np.where(signal_plot == -1)[0]
    plt.scatter(np.array(x_dates)[short_idx], eth_close_test[short_idx], marker='v', color='red', label='Short Signal')
    plt.title('ETH Price with LSTM Trade Signals')
    plt.xlabel('Date')
    plt.ylabel('ETH Close (USD)')
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig(f'{prefix}eth_price_signals_v2.png')
    plt.close()

    # 儲存績效數據
    import json
    perf = {
        'total_return': float(total_return),
        'win_rate': float(win_rate),
    }
    with open(f'{prefix}backtest_performance.json', 'w', encoding='utf-8') as f:
        json.dump(perf, f, ensure_ascii=False, indent=2)
    print(f"Backtest results saved to {save_dir if save_dir else os.getcwd()}")

# ========== CLI 執行入口 ==========
if __name__ == "__main__":
    run_lstm_backtest()

