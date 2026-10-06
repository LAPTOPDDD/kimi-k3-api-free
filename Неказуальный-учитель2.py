import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- 1. Настройки ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
BARS_TOTAL = 10500
BARS_TRAIN = 10000
LAGS = 50
SIGMA = 5

# --- 2. Загрузка данных ---
print("Подключение к MetaTrader 5...")
if not mt5.initialize():
    print("Ошибка инициализации MT5")
    mt5.shutdown()
    exit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_TOTAL)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("Ошибка загрузки данных")
    exit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
close = df['close'].values
print(f"Загружено {len(close)} баров {SYMBOL}")

# --- 3. Целевая функция (НЕКАЗУАЛЬНАЯ) ---
print("Построение неказуальной производной Гаусса...")
smoothed = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
target = np.zeros_like(smoothed, dtype=np.float64)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
target[0] = target[1]
target[-1] = target[-2]

# МАСШТАБИРОВАНИЕ ЦЕЛИ (критически важно!)
# Умножаем на 10000, чтобы значения были порядка ~1.0
TARGET_SCALE = 10000.0
target_scaled = target * TARGET_SCALE

print(f"Target range: [{target.min():.6f}, {target.max():.6f}]")
print(f"Target scaled range: [{target_scaled.min():.4f}, {target_scaled.max():.4f}]")

# --- 4. Подготовка фичей ---
print("Подготовка фичей...")
# Дополнительные фичи: returns, volatility, diff
returns = np.diff(close, prepend=close[0])
volatility = np.array([np.std(close[max(0, i-20):i+1]) for i in range(len(close))])

X = np.zeros((len(close), LAGS * 3))  # 3 типа фичей на каждый лаг
for i in range(LAGS, len(close)):
    lag_close = close[i-LAGS:i][::-1]
    lag_returns = returns[i-LAGS:i][::-1]
    lag_vol = volatility[i-LAGS:i][::-1]
    X[i] = np.concatenate([lag_close, lag_returns, lag_vol])

X = X[LAGS:]
y = target_scaled[LAGS:]  # Масштабированная цель!
time_index = df.index[LAGS:]

# Нормализация фичей
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. Разделение ---
X_train = X_scaled[:BARS_TRAIN]
y_train = y[:BARS_TRAIN]
X_test = X_scaled[BARS_TRAIN:]
y_test_true_scaled = y[BARS_TRAIN:]
test_time = time_index[BARS_TRAIN:]

# Сохраняем не масштабированную цель для визуализации
y_test_true = target[LAGS:][BARS_TRAIN:]

print(f"Обучение на {len(X_train)} барах...")
print(f"Тест на {len(X_test)} барах...")

# --- 6. Обучение модели ---
print("Обучение LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.1,        # Увеличили!
    num_leaves=127,           # Больше листьев
    max_depth=10,             # Ограничение глубины
    min_data_in_leaf=10,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.01,
    lambda_l2=0.01,
    objective='regression',
    metric='rmse',
    random_state=42,
    verbosity=1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test_true_scaled)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# --- 7. Инференс ---
print("Инференс (rolling forecast)...")
predictions_scaled = []
for i in range(len(X_test)):
    pred = model.predict(X_test[i].reshape(1, -1))
    predictions_scaled.append(pred[0])
predictions_scaled = np.array(predictions_scaled)

# Обратное масштабирование!
predictions = predictions_scaled / TARGET_SCALE

# --- 8. Визуализация ---
PLOT_OFFSET = 50
plot_start = PLOT_OFFSET

plt.figure(figsize=(16, 8))

plt.plot(test_time[plot_start:], y_test_true[plot_start:],
         label='True Non-Causal Derivative (Teacher)', color='blue', alpha=0.8, linewidth=1.5)
plt.plot(test_time[plot_start:], predictions[plot_start:],
         label='LightGBM Approximation (Causal)', color='red', linestyle='--', linewidth=1.5)

plt.title('Approximation of Non-Causal Gaussian Derivative\n(Strict Causal Inference, 10000 bars, scaled target)', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Derivative Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 9. Метрики ---
mse = mean_squared_error(y_test_true[plot_start:], predictions[plot_start:])
mae = mean_absolute_error(y_test_true[plot_start:], predictions[plot_start:])
r2 = r2_score(y_test_true[plot_start:], predictions[plot_start:])

print("\n--- Результаты на тестовом участке ---")
print(f"MSE: {mse:.10f}")
print(f"MAE: {mae:.10f}")
print(f"R2:  {r2:.4f}")

# --- 10. Дополнительный график: расхождение ---
residuals = y_test_true - predictions
plt.figure(figsize=(16, 4))
plt.bar(test_time[plot_start:], residuals[plot_start:], color='gray', alpha=0.6, width=0.02)
plt.title('Residuals (True - Predicted)')
plt.ylabel('Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()