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

SIGMA = 2
TARGET_SCALE = 10000

# Модель (под быстрый сигнал: меньше min_child, больше деревьев)
N_ESTIMATORS = 400
LEARNING_RATE = 0.03
MIN_CHILD_SAMPLES = 15

# Тест торгового правила
NB_LIST = [3, 5, 8, 10]
COMMISSION = 0.5
SLIPPAGE = 0.5

# ==========================================
# 1. ДАННЫЕ
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_BARS)
mt5.shutdown()
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df = df[df.index.dayofweek < 5].copy()

# ==========================================
# 2. ФИЧИ: медленные (20) + БЫСТРЫЕ (3/5/8 под σ=2)
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
df['vol_ratio'] = df['tick_volume'] / (df['tick_volume'].rolling(20).mean() + 1e-9)

for w in (3, 5, 8):   # ⚡ быстрые фичи
    df[f'ret_{w}'] = df['close'].pct_change(w)
    df[f'price_to_sma_{w}'] = df['close'] / df['close'].rolling(w).mean()
    df[f'std_{w}'] = df['close'].rolling(w).std()
    lo, hi = df['close'].rolling(w).min(), df['close'].rolling(w).max()
    df[f'range_pos_{w}'] = (df['close'] - lo) / (hi - lo + 1e-9)

df = df.dropna().copy()
close = df['close'].values
feature_cols = [c for c in df.columns if c not in
                ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]

# ==========================================
# 3. TRAIN / TEST + ЦЕЛЬ
# ==========================================
BARS_TRAIN = len(df) - BARS_TEST
X_train = df[feature_cols].iloc[:BARS_TRAIN].values
X_test  = df[feature_cols].iloc[BARS_TRAIN:].values
y_train_raw, y_test_raw = close[:BARS_TRAIN], close[BARS_TRAIN:]

smoothed = gaussian_filter1d(y_train_raw, sigma=SIGMA, mode='reflect')
target = np.zeros_like(smoothed)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0

# ==========================================
# 4. ОБУЧЕНИЕ + КАЛИБРОВКА АМПЛИТУДЫ (на train!)
# ==========================================
scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                          min_child_samples=MIN_CHILD_SAMPLES,
                          n_jobs=-1, verbosity=-1, random_state=42)
model.fit(X_train_s, target * TARGET_SCALE)

pred_train = model.predict(X_train_s) / TARGET_SCALE
k = np.std(target) / (np.std(pred_train) + 1e-9)      # калибровка сжатия
preds = model.predict(X_test_s) / TARGET_SCALE * k
print(f"✅ Калибровка амплитуды: k={k:.2f}")

# ==========================================
# 5. ДИАГНОСТИКА
# ==========================================
smoothed_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
ideal_all = np.zeros_like(smoothed_all)
ideal_all[1:-1] = (smoothed_all[2:] - smoothed_all[:-2]) / 2.0
ideal_test = ideal_all[BARS_TRAIN:]

actual_forward = np.zeros(BARS_TEST)
for i in range(BARS_TEST - 2):
    actual_forward[i] = (close[BARS_TRAIN + i + 2] - close[BARS_TRAIN + i]) / 2.0

n = BARS_TEST - 2
pred_n, ideal_n, actual_n = preds[:n], ideal_test[:n], actual_forward[:n]

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
print("=" * 56)
print(f"  corr(модель, идеал)      = {np.corrcoef(pred_n, ideal_n)[0,1]:+.3f}")
print(f"  corr(идеал, реальность)  = {np.corrcoef(ideal_n, actual_n)[0,1]:+.3f}")
print(f"  corr(модель, реальность) = {np.corrcoef(pred_n, actual_n)[0,1]:+.3f}")
print(f"  best lag                 = {lag_pa:+d} (r={r_pa:+.3f})")
print("=" * 56)

# ==========================================
# 6. ТОРГОВОЕ ПРАВИЛО: знак → вход, hold nb
# ==========================================
def sign_equity(sig, close_t, nb):
    eq, pos, entry, held, trades = 0.0, 0, 0.0, 0, 0
    curve = []
    for t in range(len(sig)):
        p = close_t[t]
        if pos != 0:
            held += 1
            if held >= nb:
                eq += (p - SLIPPAGE*1e-4*pos - entry)*pos - COMMISSION
                trades += 1; pos = 0
        if pos == 0 and t <= len(sig) - nb - 1:
            s = np.sign(sig[t])
            if s != 0: pos, entry, held = s, p + SLIPPAGE*1e-4*s, 0
        curve.append(eq)
    return np.array(curve), eq, trades

print(f"\n{'nb':>3} | {'PnL модели':>11} {'сделок':>6} | {'PnL ИДЕАЛА':>11} {'сделок':>6}")
eq_curves = {}
for nb in NB_LIST:
    eq_m, pnl_m, tr_m = sign_equity(pred_n,  y_test_raw[:n], nb)
    eq_i, pnl_i, tr_i = sign_equity(ideal_n, y_test_raw[:n], nb)
    eq_curves[nb] = (eq_m, eq_i)
    print(f"{nb:>3} | {pnl_m:>11.2f} {tr_m:>6} | {pnl_i:>11.2f} {tr_i:>6}")

# ==========================================
# 7. ГРАФИК
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                         gridspec_kw={'height_ratios': [1.4, 1, 0.9]})
x = np.arange(1, BARS_TEST + 1)
axes[0].plot(x, y_test_raw, 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} | σ={SIGMA} | быстрые фичи + калибровка'); axes[0].grid(alpha=0.3)

axes[1].plot(x[:n], actual_n, 'r', alpha=0.5, lw=1, label='Реальность')
axes[1].plot(x[:n], ideal_n, 'g', lw=2, label='ИДЕАЛ')
axes[1].plot(x[:n], pred_n, 'b', lw=1.5, label='Модель')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].legend(); axes[1].grid(alpha=0.3)

nb_show = NB_LIST[1]
axes[2].plot(eq_curves[nb_show][0], 'b', lw=1.5, label=f'Модель nb={nb_show}')
axes[2].plot(eq_curves[nb_show][1], 'g', lw=1.5, label=f'ИДЕАЛ nb={nb_show} (потолок)')
axes[2].axhline(0, color='gray', lw=0.8); axes[2].legend(); axes[2].grid(alpha=0.3)
axes[2].set_xlabel('Тестовый бар')
plt.tight_layout(); plt.show()