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
# НАСТРОЙКИ
# ==========================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
TOTAL_BARS = 5000
BARS_TEST = 100

SIGMA = 2                    # ✅ выбрано по графику Гаусса
TARGET_SCALE = 10000

N_ESTIMATORS = 250
LEARNING_RATE = 0.05
MIN_CHILD_SAMPLES = 30

# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================================
if not mt5.initialize():
    print("❌ MT5 init error"); exit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_BARS)
mt5.shutdown()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df[df.index.dayofweek < 5].copy()
print(f"✅ Загружено {len(df)} баров")

# ==========================================
# 2. ПРИЗНАКИ
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
feature_cols = [c for c in df.columns if c not in
                ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]

# ==========================================
# 3. TRAIN / TEST
# ==========================================
BARS_TRAIN = len(df) - BARS_TEST
X_train = df[feature_cols].iloc[:BARS_TRAIN].values
X_test  = df[feature_cols].iloc[BARS_TRAIN:].values
y_train_raw = close[:BARS_TRAIN]
y_test_raw  = close[BARS_TRAIN:]

# ==========================================
# 4. ЦЕЛЬ (σ=2)
# ==========================================
smoothed = gaussian_filter1d(y_train_raw, sigma=SIGMA, mode='reflect')
target = np.zeros_like(smoothed)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
target_scaled = target * TARGET_SCALE

# ==========================================
# 5. ОБУЧЕНИЕ
# ==========================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                          min_child_samples=MIN_CHILD_SAMPLES,
                          n_jobs=-1, verbosity=-1, random_state=42)
model.fit(X_train_s, target_scaled)
preds = model.predict(X_test_s) / TARGET_SCALE

# ==========================================
# 6. ДИАГНОСТИКА: ТРИ КОРРЕЛЯЦИИ
# ==========================================
# Идеал на тесте (оракул — только для оценки!)
smoothed_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
ideal_all = np.zeros_like(smoothed_all)
ideal_all[1:-1] = (smoothed_all[2:] - smoothed_all[:-2]) / 2.0
ideal_test = ideal_all[BARS_TRAIN:]

# Реальное движение вперёд
actual_forward = np.zeros(BARS_TEST)
for i in range(BARS_TEST - 2):
    actual_forward[i] = (close[BARS_TRAIN + i + 2] - close[BARS_TRAIN + i]) / 2.0

n = BARS_TEST - 2
pred_n, ideal_n, actual_n = preds[:n], ideal_test[:n], actual_forward[:n]

c_pi = np.corrcoef(pred_n, ideal_n)[0, 1]
c_ia = np.corrcoef(ideal_n, actual_n)[0, 1]
c_pa = np.corrcoef(pred_n, actual_n)[0, 1]

def xcorr(a, b, max_lag=6):
    best = (0, -2.0)
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:   xa, ya = a[:len(a)-lag], b[lag:]
        elif lag < 0: xa, ya = a[-lag:], b[:len(b)+lag]
        else:         xa, ya = a, b
        r = np.corrcoef(xa, ya)[0, 1]
        if r > best[1]: best = (lag, r)
    return best

lag_pa, r_pa = xcorr(pred_n, actual_n)

print("=" * 52)
print(f"  corr(модель, идеал)      = {c_pi:+.3f}  ← аппроксимация")
print(f"  corr(идеал, реальность)  = {c_ia:+.3f}  ← ценность оракула")
print(f"  corr(модель, реальность) = {c_pa:+.3f}  ← итог")
print(f"  best lag (модель/реальн) = {lag_pa:+d} бар (r={r_pa:+.3f})")
print(f"  std: модель={np.std(pred_n):.6f} | идеал={np.std(ideal_n):.6f}")
print("=" * 52)

# ==========================================
# 7. ГРАФИК: ЦЕНА + ТРИ КРИВЫЕ
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                         gridspec_kw={'height_ratios': [1.5, 1]})
x = np.arange(1, BARS_TEST + 1)

axes[0].plot(x, y_test_raw, color='black', lw=1.2)
axes[0].set_title(f'{SYMBOL} — цена на тесте | σ={SIGMA}')
axes[0].set_ylabel('Цена'); axes[0].grid(alpha=0.3)

axes[1].plot(x[:n], actual_n, color='red',   alpha=0.5, lw=1,   label='Реальность')
axes[1].plot(x[:n], ideal_n,  color='green', lw=2,      label='ИДЕАЛ (оракул)')
axes[1].plot(x[:n], pred_n,   color='blue',  lw=1.5,    label='Модель')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].set_title(f'модель↔идеал={c_pi:.2f} | идеал↔реальн={c_ia:.2f} | модель↔реальн={c_pa:.2f}')
axes[1].set_xlabel('Тестовый бар'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()