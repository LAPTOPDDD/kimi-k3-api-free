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

SIGMA = 1
SHIFT = 0              # 0 = обычная цель, 1 = цель на 1 бар вперёд (компенсация лага)
TARGET_SCALE = 10000

N_ESTIMATORS = 400
LEARNING_RATE = 0.03
MIN_CHILD_SAMPLES = 15

NB_LIST = [ 4 ]    # твои удержания
COMMISSION = 0.0       # БЕЗ КОМИССИЙ, как ты просил
SLIPPAGE = 0.0

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
# 2. ФИЧИ: медленные (20) + быстрые (3,5,8 под σ=2)
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

for w in (3, 5, 8):   # ⚡ быстрые фичи под быстрый сигнал σ=2
    df[f'ret_{w}'] = df['close'].pct_change(w)
    df[f'price_to_sma_{w}'] = df['close'] / df['close'].rolling(w).mean()
    df[f'std_{w}'] = df['close'].rolling(w).std()
    lo = df['close'].rolling(w).min()
    hi = df['close'].rolling(w).max()
    df[f'range_pos_{w}'] = (df['close'] - lo) / (hi - lo + 1e-9)

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
# 4. ЦЕЛЬ СО СДВИГОМ
# ==========================================
smoothed = gaussian_filter1d(y_train_raw, sigma=SIGMA, mode='reflect')
d = np.zeros_like(smoothed)
d[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0

if SHIFT > 0:
    y_fit_raw = d[SHIFT:]
    X_fit = X_train[:-SHIFT]
else:
    y_fit_raw = d
    X_fit = X_train

# ==========================================
# 5. ОБУЧЕНИЕ + КАЛИБРОВКА
# ==========================================
scaler = StandardScaler()
X_fit_s = scaler.fit_transform(X_fit)
X_test_s = scaler.transform(X_test)

model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                          min_child_samples=MIN_CHILD_SAMPLES,
                          n_jobs=-1, verbosity=-1, random_state=42)
model.fit(X_fit_s, y_fit_raw * TARGET_SCALE)

pred_train = model.predict(X_fit_s) / TARGET_SCALE
k = np.std(y_fit_raw) / (np.std(pred_train) + 1e-9)
preds = model.predict(X_test_s) / TARGET_SCALE * k
print(f"✅ SHIFT={SHIFT} | калибровка амплитуды k={k:.2f}")

# ==========================================
# 6. ДИАГНОСТИКА
# ==========================================
smoothed_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
ideal_all = np.zeros_like(smoothed_all)
ideal_all[1:-1] = (smoothed_all[2:] - smoothed_all[:-2]) / 2.0
ideal_test = ideal_all[BARS_TRAIN + SHIFT: BARS_TRAIN + SHIFT + BARS_TEST]
if len(ideal_test) < BARS_TEST:
    ideal_test = np.pad(ideal_test, (0, BARS_TEST - len(ideal_test)))

actual_forward = np.zeros(BARS_TEST)
for i in range(BARS_TEST - 2):
    actual_forward[i] = (close[BARS_TRAIN + i + 2] - close[BARS_TRAIN + i]) / 2.0

n = BARS_TEST - 2
pred_n, ideal_n, actual_n = preds[:n], ideal_test[:n], actual_forward[:n]

print("=" * 56)
print(f"  corr(модель, идеал_сдв)  = {np.corrcoef(pred_n, ideal_n)[0,1]:+.3f}")
print(f"  corr(идеал_сдв, реальн.) = {np.corrcoef(ideal_n, actual_n)[0,1]:+.3f}")
print(f"  corr(модель, реальность) = {np.corrcoef(pred_n, actual_n)[0,1]:+.3f}")
print("=" * 56)

# ==========================================
# 7. ТВОЁ ПРАВИЛО (без комиссий)
# ==========================================
def user_rule(sig, close_t, nb):
    """Кросс → вход | держу N баров | выход по новому сигналу или по сроку"""
    eq, pos, entry, held, trades = 0.0, 0, 0.0, 0, 0
    curve, prev_s = [], 0
    for t in range(len(sig)):
        p = close_t[t]
        s = int(np.sign(sig[t]))
        cross = (s != 0) and (s != prev_s)
        if pos != 0:
            held += 1
            if held >= nb or (cross and s == -pos):
                eq += (p - entry) * pos
                trades += 1; pos = 0
        if cross and pos == 0:
            pos, entry, held = s, p, 0
        prev_s = s
        curve.append(eq)
    if pos != 0:
        eq += (close_t[-1] - entry) * pos
        trades += 1
    return np.array(curve), eq, trades

print(f"\nПРАВИЛО: кросс нуля + удержание N | БЕЗ КОМИССИЙ")
print(f"{'N':>2} | {'модель PnL':>10} {'сделок':>6} | {'идеал PnL':>10} {'сделок':>6}")
eqs = {}
for nb in NB_LIST:
    cm, pm, tm = user_rule(pred_n, y_test_raw[:n], nb)
    ci, pi, ti = user_rule(ideal_n, y_test_raw[:n], nb)
    eqs[nb] = (cm, ci)
    print(f"{nb:>2} | {pm:>10.2f} {tm:>6} | {pi:>10.2f} {ti:>6}")

# ==========================================
# 8. ГРАФИКИ
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                         gridspec_kw={'height_ratios': [1.4, 1, 0.9]})
x = np.arange(1, BARS_TEST + 1)

axes[0].plot(x, y_test_raw, 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} | σ={SIGMA} SHIFT={SHIFT} | твое правило')
axes[0].grid(alpha=0.3)

axes[1].plot(x[:n], actual_n, 'r', alpha=0.5, lw=1, label='Реальность')
axes[1].plot(x[:n], ideal_n, 'g', lw=2, label='ИДЕАЛ (сдв.)')
axes[1].plot(x[:n], pred_n, 'b', lw=1.5, label='Модель')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].legend(); axes[1].grid(alpha=0.3)

nb_show = 4
axes[2].plot(eqs[nb_show][0], 'b', lw=1.5, label=f'Модель N={nb_show}')
axes[2].plot(eqs[nb_show][1], 'g', lw=1.5, label=f'ИДЕАЛ N={nb_show} (потолок)')
axes[2].axhline(0, color='gray', lw=0.8)
axes[2].set_title('Эквити без комиссий')
axes[2].legend(); axes[2].grid(alpha=0.3)
axes[2].set_xlabel('Тестовый бар')

plt.tight_layout()
plt.show()