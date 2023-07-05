from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

# api key
api_key = 'XX'
api_secret = 'XX'

client = Client(api_key, api_secret)

# 取得範圍內 H1 的數據
klines = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1HOUR, "1 May, 2021", "1 Jun, 2023")

df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])


df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Close'] = df['Close'].astype(float)
df['Volume'] = df['Volume'].astype(float)
df['Price Change'] = df['Close'].diff()
#label每一天的價格相對於昨天是上漲還是下跌
df['Label'] = df['Price Change'].apply(lambda x: 1 if x > 0 else 0)

# macd
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
dif = ema_12 - ema_26
macd_signal = dif.ewm(span=9, adjust=False).mean()
macd_histogram = dif - macd_signal

df['DIF'] = dif
df['MACD Signal'] = macd_signal
df['MACD Histogram'] = macd_histogram

# mfi
def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    positive_flow_df = pd.DataFrame(positive_flow)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    negative_flow_df = pd.DataFrame(negative_flow)
    money_ratio = positive_flow_df.rolling(window=period).sum() / negative_flow_df.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

mfi = calculate_mfi(df)
df['MFI'] = mfi

# fs
series = df['Close']
period = 9
price = 0.5 * (series + 1)
price = np.maximum(np.minimum(price, 0.9999), -0.9999)  # 限制价格范围在(-0.9999, 0.9999)
fisher_transform = 0.5 * np.log((1 + price) / (1 - price))
fisher_transform = fisher_transform.rolling(window=period).mean()  # 对计算结果进行平滑处理
df['FS'] = fisher_transform

features = ['Close', 'Volume', 'MACD Signal']
labels = 'Label'

# 提取特徵和標籤數據

data_close = df['Close'].values
data_macd = df['MACD Signal'].values
data_volume = df['Volume'].values
data_mfi = df['MFI'].values
data_fs = df['FS'].values


num_samples = len(data_close) - 10 + 1 #總共的數據共有 總數 - 10 天 + 1
num_features = 50 #總共有50個特徵值 price, volume, macd, mfi, fs 各10天

# 創建空數组X和y
X = np.zeros((num_samples, 10, num_features))
y = np.zeros(num_samples)

# 填充數據到X和y
for i in range(num_samples-10):
    X[i, :, :10] = data_close[i:i+10]
    X[i, :, 10:20] = data_macd[i:i+10]
    X[i, :, 20:30] = data_volume[i:i+10]
    X[i, :, 30:40] = data_mfi[i:i+10]
    X[i, :, 40:] = data_fs[i:i+10]
    y[i] = 1 if data_close[i+10] > data_close[i] else 0

# 因為 macd 要到 26天後才有數據
X = X[26:]
y = y[26:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建LSTM模型
model = keras.Sequential()
model.add(layers.LSTM(units=64, input_shape=(10, num_features)))
model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 对测试集进行预测
y_pred = model.predict(X_test)


# 模型評估
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# # plt.plot(history.history['loss'], label='Training Loss')
# # plt.plot(history.history['val_loss'], label='Validation Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()

# # # 繪製訓練和驗證準確率曲線
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# # 預測測試集的標籤
# y_pred_prob = model.predict(X_test)

# # 將概率分佈轉換為預測類別（0或1）
# y_pred = np.round(y_pred_prob).flatten()

# # 計算混淆矩陣
# cm = confusion_matrix(y_test, y_pred)

# # 繪製混淆矩陣
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()
