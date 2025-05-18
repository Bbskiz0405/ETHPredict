import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# ========== 參數 ==========
SEQ_LEN = 72
MODEL_PATH = 'lstm_backtest_model.keras'

# ========== 資料處理 ==========
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

# ========== 模型訓練/載入 ==========
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

# ========== 回測策略 ==========
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

# 畫圖
# 取得測試集對應的日期（與equity長度一致）
x_dates = features.index[-len(equity):]

plt.figure(figsize=(12,5))
plt.plot(x_dates, equity, label='Equity Curve')
plt.title('LSTM Backtest Equity Curve (Long/Short)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (USD)')
plt.legend()
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.savefig('equity_curve.png')
plt.close()

# === 新增訊號標註在ETH收盤價圖 ===
eth_close_test = features['ETH_Close'][-len(equity):].values
signal_plot = signal[:len(equity)]  # 保證長度一致

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
plt.savefig('eth_price_signals.png')
plt.close()

# === 新增訊號標註在ETH收盤價圖 ===
eth_close_test = features['ETH_Close'][-len(equity):].values
signal_plot = signal[:len(equity)]  # 保證長度一致

plt.figure(figsize=(14,6))
plt.plot(x_dates, eth_close_test, label='ETH True Close', color='black')
# 做多點
long_idx = np.where(signal_plot == 1)[0]
plt.scatter(np.array(x_dates)[long_idx], eth_close_test[long_idx], marker='^', color='green', label='Long Signal')
# 做空點
short_idx = np.where(signal_plot == -1)[0]
plt.scatter(np.array(x_dates)[short_idx], eth_close_test[short_idx], marker='v', color='red', label='Short Signal')
plt.title('ETH Price with LSTM Trade Signals')
plt.xlabel('Date')
plt.ylabel('ETH Close (USD)')
plt.legend()
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.show()

