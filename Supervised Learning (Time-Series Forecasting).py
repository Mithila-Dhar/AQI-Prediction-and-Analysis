import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('Ariyalur_AQIBulletins.csv')
df.head()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')

target = "Index Value"

ts = df[target].values.reshape(-1, 1)

scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts)

def create_windowed_data(data, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

WINDOW = 7
X, y = create_windowed_data(ts_scaled, WINDOW)

print(X.shape, y.shape)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, activation='tanh', return_sequences=False, input_shape=(WINDOW, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
y_test_original = scaler.inverse_transform(y_test)

mse = mean_squared_error(y_test_original, pred)
mae = mean_absolute_error(y_test_original, pred)

print("MSE:", mse)
print("MAE:", mae)

plt.figure(figsize=(12,6))
plt.plot(y_test_original, label="Actual Index Value")
plt.plot(pred, label="Predicted Index Value")
plt.xlabel("Time")
plt.ylabel("Index Value")
plt.title("Actual vs Predicted Index Value")
plt.legend()
plt.grid(True)
plt.show()
