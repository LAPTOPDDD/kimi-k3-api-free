import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- 1. Настройки ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
BARS_TOTAL = 1200  # Нужно 1000 обучение + 150 тест + буфер
BARS_TRAIN = 1000
LAGS = 50          # Количество лагов (Close_t-1 ... Close_t-50)
SIGMA = 5          # Параметр сглаживания Гаусса

# --- 2. Загрузка данных из MT5 ---
print("Подключение к MetaTrader 5...")
if not mt5.initialize():
    print("Не удалось инициализировать MT5")
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
# Симметричный Гаусс (использует будущее для сглаживания, но это эталон)
smoothed = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
# Первая производная (центральная разность)
target = np.zeros_like(smoothed)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
target[0] = target[1]
target[-1] = target[-2]

# --- 4. Подготовка фичей (КАУЗАЛЬНЫЕ ЛАГИ) ---
print("Подготовка фичей...")
X = np.zeros((len(close), LAGS))
for i in range(LAGS, len(close)):
    X[i] = close[i-LAGS:i][::-1]  # Лаги от 1 до 50

# Удаляем строки, где недостаточно истории
X = X[LAGS:]
y = target[LAGS:]
time_index = df.index[LAGS:]

# --- 5. Разделение ---
# Обучающая выборка: первые 1000 баров (после лагов)
train_size = BARS_TRAIN
X_train = X[:train_size]
y_train = y[:train_size]

# Тестовая выборка: оставшиеся бары (но начнем визуализировать с отступом 50)
X_test = X[train_size:]
y_test_true = y[train_size:]  # Истинный неказуальный учитель
test_time = time_index[train_size:]

print(f"Обучение на {len(X_train)} барах...")
print(f"Тест на {len(X_test)} барах...")

# --- 6. Обучение модели ---
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    verbosity=-1
)
model.fit(X_train, y_train)

# --- 7. Инференс (Строго по одному бару, каузально) ---
print("Запуск инференса (rolling forecast)...")
predictions = []

# Мы начинаем с бара train_size + LAGS (у нас уже есть лаги в X_test)
# Но по ТЗ: "отступаем 50 баров от края" -> это уже учтено в размере теста
for i in range(len(X_test)):
    # Берем текущий вектор признаков (он уже каузальный)
    current_features = X_test[i].reshape(1, -1)
    pred = model.predict(current_features)
    predictions.append(pred[0])

predictions = np.array(predictions)

# --- 8. Визуализация ---
# Отступаем 50 баров от начала теста для чистоты графика (как договаривались)
PLOT_OFFSET = 50
plot_start = PLOT_OFFSET

plt.figure(figsize=(14, 7))

plt.plot(test_time[plot_start:], y_test_true[plot_start:], 
         label='True Non-Causal Derivative (Teacher)', 
         color='blue', alpha=0.7, linewidth=1.5)

plt.plot(test_time[plot_start:], predictions[plot_start:], 
         label='LightGBM Approximation (Causal)', 
         color='red', linestyle='--', linewidth=1.5)

plt.title('Approximation of Non-Causal Gaussian Derivative\n(Strict Causal Inference)', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Derivative Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 9. Метрики на тесте ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_true[plot_start:], predictions[plot_start:])
mae = mean_absolute_error(y_test_true[plot_start:], predictions[plot_start:])
r2 = r2_score(y_test_true[plot_start:], predictions[plot_start:])

print("\n--- Результаты на тестовом участке ---")
print(f"MSE: {mse:.8f}")
print(f"MAE: {mae:.8f}")
print(f"R2:  {r2:.4f}")