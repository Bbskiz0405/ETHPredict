import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import ta
import json

# ========== 參數設定 ==========
SEQ_LEN = 72  # 與訓練時保持一致
PREDICTION_STEPS = 12  # 預測未來48小時（12個4小時週期）
THRESHOLD = 0.005  # 交易訊號閾值
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

# 確保目錄存在
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_latest_model():
    """載入最新模型"""
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    if not models:
        raise FileNotFoundError("找不到訓練好的模型，請先執行訓練程式")
    
    # 按照日期排序找最新的模型
    latest_model = sorted(models)[-1]
    model_path = os.path.join(MODEL_DIR, latest_model)
    print(f"加載模型: {model_path}")
    return load_model(model_path)

def add_technical_indicators(df):
    """為數據添加技術指標"""
    # 移動平均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_middle'] = rolling_mean
    df['BB_lower'] = rolling_mean - (rolling_std * 2)
    
    # ATR
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 漲跌幅
    df['Return'] = df['Close'].pct_change()
    
    return df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1)

def create_features_df(eth_data, btc_data):
    """整合ETH和BTC數據創建特徵資料框"""
    # 確保索引一致
    common_index = eth_data.index.intersection(btc_data.index)
    if len(common_index) == 0:
        print("沒有共同的時間索引，無法進行預測")
        return None
    
    eth_data = eth_data.loc[common_index]
    btc_data = btc_data.loc[common_index]
    
    # 創建特徵資料框
    features = pd.DataFrame(index=common_index)
    
    # 添加ETH特徵
    features['ETH_Close'] = eth_data['Close']
    features['ETH_MA5'] = eth_data['MA5']
    features['ETH_MA10'] = eth_data['MA10']
    
    # 添加BTC特徵
    features['BTC_Close'] = btc_data['Close']
    features['BTC_MA5'] = btc_data['MA5']
    features['BTC_MA10'] = btc_data['MA10']
    
    # 檢查資料框是否有 NaN 值
    if features.isnull().values.any():
        print("移除 NaN 值...")
        features = features.dropna()
    
    # 檢查資料框是否為空
    if len(features) == 0:
        print("資料框為空，無法進行預測")
        return None
    
    return features

def load_latest_model():
    """加載最新的LSTM模型"""
    # 首先嘗試從當前目錄加載模型
    if os.path.exists('lstm_backtest_model.keras'):
        print("加載模型: lstm_backtest_model.keras")
        return load_model('lstm_backtest_model.keras')
    
    # 如果當前目錄沒有模型，再嘗試從models目錄加載
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("模型目錄不存在")
        return None
    
    # 找到最新的模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    if not model_files:
        print("沒有找到模型文件")
        return None
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    print(f"加載模型: {latest_model}")
    return load_model(os.path.join(model_dir, latest_model))

def generate_future_predictions(model, sequence, scaler, steps=PREDICTION_STEPS):
    """
    生成未來的預測
    """
    predictions = []
    
    # 預測未來6個時間點
    for _ in range(steps):
        pred = model.predict(sequence, verbose=0)
        predictions.append(pred[0][0])
        # 更新序列數據
        sequence = np.roll(sequence, -1, axis=1)
        sequence[0, -1] = pred[0][0]
    
    return np.array(predictions)

def main():
    """主函數"""
    print("下載最新市場數據中...")
    ETH = yf.download('ETH-USD', period='180d', interval='4h')
    BTC = yf.download('BTC-USD', period='180d', interval='4h')
    
    # 檢查數據是否下載成功
    if ETH.empty or BTC.empty:
        print("數據下載失敗，請檢查網路連接")
        return
    
    # 添加技術指標
    ETH = add_technical_indicators(ETH)
    BTC = add_technical_indicators(BTC)
    
    # 創建特徵資料框
    features = create_features_df(ETH, BTC)
    
    # 檢查特徵資料框是否為空
    if features is None:
        return
    
    # 標準化數據
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    
    # 準備序列數據
    X = []
    for i in range(len(scaled) - SEQ_LEN - 1):
        X.append(scaled[i:(i+SEQ_LEN)])
    X = np.array(X)
    
    # 載入模型
    print("載入模型中...")
    model = load_latest_model()
    if model is None:
        print("沒有找到可用的模型，請先訓練模型")
        return
    
    # 獲取最後一個序列
    last_sequence = scaled[-SEQ_LEN:]
    last_sequence = last_sequence.reshape(1, SEQ_LEN, -1)
    
    # 預測未來價格
    print("預測未來價格中...")
    predictions = generate_future_predictions(model, last_sequence, scaler, steps=PREDICTION_STEPS)
    
    # 逆轉換預測值
    predictions = scaler.inverse_transform(np.hstack([predictions.reshape(-1, 1), np.zeros((len(predictions), features.shape[1] - 1))]))[:, 0]
    
    # 獲取當前時間和價格
    current_time = features.index[-1]
    current_price = features['ETH_Close'].iloc[-1]
    
    # 生成交易訊號
    signals = []
    for pred in predictions:
        price_change = pred - current_price
        signal = ""
        if price_change > 0:
            if price_change > 100:
                signal = "強烈建議做多 ▲▲"
            elif price_change > 50:
                signal = "建議做多 ▲"
            else:
                signal = "中立"
        else:
            if price_change < -100:
                signal = "強烈建議做空 ▼▼"
            elif price_change < -50:
                signal = "建議做空 ▼"
            else:
                signal = "中立"
        signals.append(signal)
    
    # 輸出結果
    print(f"\n預測結果:")
    print(f"當前時間: {current_time}")
    print(f"最後一個歷史價格: {current_price:.2f} USD")
    print("\n未來預測:")
    for i, pred in enumerate(predictions, 1):
        future_time = current_time + pd.Timedelta(hours=4 * i)
        print(f"{i}個4小時後 ({future_time}): {pred:.2f} USD")
    
    print("\n交易訊號:")
    for i, signal in enumerate(signals, 1):
        print(f"{i}個4小時後: {signal}")
    
    # 繪製圖表
    plt.figure(figsize=(14, 6))
    plt.plot(features.index, features['ETH_Close'], label='Historical ETH Price', color='blue')
    
    # 確保預測時間和數據形狀匹配
    future_times = [current_time + pd.Timedelta(hours=4 * i) for i in range(1, PREDICTION_STEPS + 1)]
    predictions = predictions[:PREDICTION_STEPS]
    
    plt.plot(future_times, predictions, label='Future Predictions', color='red', linestyle='--')
    plt.title('ETH Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig('eth_future_prediction.png')
    plt.close()
    
    # 保存預測結果為JSON
    prediction_data = []
    for i in range(len(predictions)):
        prediction_data.append({
            "時間點": future_times[i].strftime('%Y-%m-%d %H:%M:%S'),
            "預測價格": f"{predictions[i]:.2f} USD",
            "價格變化": f"{(predictions[i] - current_price):.2f} USD",
            "交易訊號": signals[i]
        })
    
    with open(os.path.join(RESULTS_DIR, 'prediction_results.json'), 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n分析結果已保存至 {RESULTS_DIR} 目錄")

if __name__ == "__main__":
    main()