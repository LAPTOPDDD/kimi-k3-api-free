import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# НАСТРОЙКИ (изменяйте здесь)
# ==========================================

SYMBOL = "EURUSD"                  # Торговый символ
TIMEFRAME = mt5.TIMEFRAME_H1       # Таймфрейм (H1, M15, D1 и т.д.)
TOTAL_BARS = 5000                  # Всего баров для загрузки
BARS_TEST = 100                    # Количество тестовых баров

# Параметры цели (Target)
SIGMA = 5                          # Сглаживание цены (гауссов фильтр)
TARGET_SCALE = 10000               # Масштабирование цели

# Параметры модели LightGBM
N_ESTIMATORS = 250
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 30

# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ ИЗ MT5
# ==========================================

print(f"📥 Загрузка данных {SYMBOL}...")

if not mt5.initialize():
    print("❌ Ошибка инициализации MT5")
    exit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_BARS)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("❌ Не удалось загрузить данные")
    exit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df[df.index.dayofweek < 5].copy()  # Убираем выходные

print(f"✅ Загружено {len(df)} баров")

# ==========================================
# 2. ПОДГОТОВКА ПРИЗНАКОВ (FEATURES)
# ==========================================

df['returns'] = df['close'].pct_change()
df['sma_20'] = df['close'].rolling(20).mean()
df['price_to_sma'] = df['close'] / df['sma_20']
df['std_20'] = df['close'].rolling(20).std()
df['cv_20'] = df['std_20'] / df['sma_20']
df['skew_20'] = df['close'].rolling(20).skew()
df['kurt_20'] = df['close'].rolling(20).kurt()
df['low_20'] = df['close'].rolling(20).min()
df['high_20'] = df['close'].rolling(20).max()
df['range_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-9)
df['ret_mean_20'] = df['returns'].rolling(20).mean()
df['ret_std_20'] = df['returns'].rolling(20).std()
df['sharpe_20'] = df['ret_mean_20'] / (df['ret_std_20'] + 1e-9) * np.sqrt(252)
df['vol_mean_20'] = df['tick_volume'].rolling(20).mean()
df['vol_ratio'] = df['tick_volume'] / (df['vol_mean_20'] + 1e-9)

df = df.dropna().copy()
close = df['close'].values
feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 
                                                          'tick_volume', 'spread', 'real_volume']]

# ==========================================
# 3. РАЗДЕЛЕНИЕ НА TRAIN / TEST
# ==========================================

BARS_TRAIN = len(df) - BARS_TEST

X_train = df[feature_cols].iloc[:BARS_TRAIN].values
X_test = df[feature_cols].iloc[BARS_TRAIN:].values
y_train_raw = close[:BARS_TRAIN]
y_test_raw = close[BARS_TRAIN:]

print(f"📊 Train: {BARS_TRAIN} баров | Test: {BARS_TEST} баров")

# ==========================================
# 4. СОЗДАНИЕ ЦЕЛИ (TARGET)
# ==========================================

smoothed = gaussian_filter1d(y_train_raw, sigma=SIGMA, mode='reflect')
target = np.zeros_like(smoothed)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
target_scaled = target * TARGET_SCALE

min_len = min(len(X_train), len(target_scaled))
X_fit = X_train[-min_len:]
y_fit = target_scaled[-min_len:]

print(f"🎯 Цель (target) создана | sigma={SIGMA}, scale={TARGET_SCALE}")

# ==========================================
# 5. НОРМАЛИЗАЦИЯ И ОБУЧЕНИЕ МОДЕЛИ
# ==========================================

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_fit)
X_test_s = scaler.transform(X_test)

model = lgb.LGBMRegressor(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    min_child_samples=MIN_CHILD_SAMPLES,
    n_jobs=-1,
    verbosity=-1,
    random_state=42
)
model.fit(X_train_s, y_fit)

print(f"🤖 Модель обучена | n_estimators={N_ESTIMATORS}, lr={LEARNING_RATE}")

# ==========================================
# 6. ПРЕДСКАЗАНИЕ НА ТЕСТОВЫХ БАРАХ
# ==========================================

preds_scaled = model.predict(X_test_s)
preds = preds_scaled / TARGET_SCALE  # Выход модели в том же масштабе, что и цель

# ==========================================
# 7. ГРАФИКИ: ЦЕЛЬ vs ВЫХОД МОДЕЛИ
# ==========================================

# Вычисляем реальное будущее движение цены (forward returns на 2 бара вперёд)
# для сравнения с предсказанием и целью
actual_forward = np.zeros(BARS_TEST)
for i in range(BARS_TEST - 2):
    actual_forward[i] = (close[BARS_TRAIN + i + 2] - close[BARS_TRAIN + i]) / 2.0

# Корреляция
corr_target_pred = np.corrcoef(preds, actual_forward[:len(preds)])[0, 1]

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

# --- Верхний график: Цена на тестовых барах ---
ax1 = axes[0]
x_bars = range(1, BARS_TEST + 1)
ax1.plot(x_bars, y_test_raw, color='black', linewidth=1.5, label=f'Цена {SYMBOL}')
ax1.fill_between(x_bars, y_test_raw.min(), y_test_raw.max(), alpha=0.05, color='gray')
ax1.set_title(f'{SYMBOL} — Цена на последних {BARS_TEST} тестовых барах', fontsize=14)
ax1.set_ylabel('Цена', fontsize=11)
ax1.set_xlabel('Номер тестового бара', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(alpha=0.3)

# --- Нижний график: Цель vs Предсказание ---
ax2 = axes[1]
ax2.plot(x_bars, preds, color='blue', linewidth=1.5, alpha=0.8, 
         label=f'Выход модели (prediction)')
ax2.plot(x_bars, actual_forward, color='red', linewidth=1.0, alpha=0.7,
         label=f'Реальное движение (actual)')
ax2.axhline(0, color='gray', lw=1, alpha=0.5)
ax2.set_title(f'Цель vs Выход модели (корреляция: {corr_target_pred:.3f})', fontsize=14)
ax2.set_ylabel('Значение (нормализованное)', fontsize=11)
ax2.set_xlabel('Номер тестового бара', fontsize=11)
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('target_vs_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 8. ВАЖНОСТЬ ПРИЗНАКОВ
# ==========================================

importances = model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n📊 Топ-10 важных признаков:")
print(feat_imp.head(10).to_string(index=False))

# ==========================================
# 9. ИТОГОВАЯ СТАТИСТИКА
# ==========================================

print("\n" + "="*50)
print("📈 ИТОГОВАЯ СТАТИСТИКА НА ТЕСТЕ:")
print("="*50)
print(f"  Символ:              {SYMBOL}")
print(f"  Тестовых баров:      {BARS_TEST}")
print(f"  Корреляция:          {corr_target_pred:.4f}")
print(f"  Средний сигнал:      {np.mean(preds):.6f}")
print(f"  Std сигнала:         {np.std(preds):.6f}")
print(f"  Max сигнал:          {np.max(preds):.6f}")
print(f"  Min сигнал:          {np.min(preds):.6f}")

# Покупки/продажи по порогу
threshold = np.std(preds) * 0.5
long_signals = np.sum(preds > threshold)
short_signals = np.sum(preds < -threshold)
neutral = BARS_TEST - long_signals - short_signals

print(f"\n  Сигналы (порог={threshold:.4f}):")
print(f"    LONG  (buy):       {long_signals}")
print(f"    SHORT (sell):      {short_signals}")
print(f"    NEUTRAL:           {neutral}")
print("="*50)