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
SHIFT = 0
TARGET_SCALE = 10000
NB = 4                          # выбран walk-forward'ом

LEARNING_RATE = 0.03
N_ESTIMATORS = 400

WF_WINDOWS = 10
WF_SIZE = 100

# СЕТКА КОНФИГУРАЦИЙ ВЫХОДА МОДЕЛИ
CONFIGS = [
    {'name': 'base (1 seed, без сглаж.)', 'seeds': 1, 'ema': 0, 'mc': 15},
    {'name': 'ensemble 5',                'seeds': 5, 'ema': 0, 'mc': 15},
    {'name': 'ensemble 5 + EMA2',         'seeds': 5, 'ema': 2, 'mc': 15},
    {'name': 'ens 5 + EMA2 + mc30',       'seeds': 5, 'ema': 2, 'mc': 30},
]

# ==========================================
# 1. ДАННЫЕ + ФИЧИ (как раньше)
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_BARS)
mt5.shutdown()
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df = df[df.index.dayofweek < 5].copy()

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
for w in (3, 5, 8):
    df[f'ret_{w}'] = df['close'].pct_change(w)
    df[f'price_to_sma_{w}'] = df['close'] / df['close'].rolling(w).mean()
    df[f'std_{w}'] = df['close'].rolling(w).std()
    lo, hi = df['close'].rolling(w).min(), df['close'].rolling(w).max()
    df[f'range_pos_{w}'] = (df['close'] - lo) / (hi - lo + 1e-9)

df = df.dropna().copy()
close = df['close'].values
feature_cols = [c for c in df.columns if c not in
                ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
print(f"✅ {len(df)} баров, {len(feature_cols)} фич")

# ==========================================
# 2. ОБУЧЕНИЕ С КОНФИГУРАЦИЕЙ ВЫХОДА
# ==========================================
def train_and_predict(X_tr, X_te, y_tr, cfg):
    sm = gaussian_filter1d(y_tr, sigma=SIGMA, mode='reflect')
    dd = np.zeros_like(sm)
    dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    yf, Xf = (dd[SHIFT:], X_tr[:-SHIFT]) if SHIFT > 0 else (dd, X_tr)

    sc = StandardScaler()
    Xf_s, Xte_s = sc.fit_transform(Xf), sc.transform(X_te)

    tr_p, te_p = [], []
    for seed in range(cfg['seeds']):
        m = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                              min_child_samples=cfg['mc'], n_jobs=-1,
                              verbosity=-1, random_state=42 + seed)
        m.fit(Xf_s, yf * TARGET_SCALE)
        tr_p.append(m.predict(Xf_s) / TARGET_SCALE)
        te_p.append(m.predict(Xte_s) / TARGET_SCALE)

    tr_m, te_m = np.mean(tr_p, axis=0), np.mean(te_p, axis=0)
    k = np.std(yf) / (np.std(tr_m) + 1e-9)
    pr = te_m * k
    if cfg['ema'] > 0:                       # каузальное сглаживание выхода
        pr = pd.Series(pr).ewm(span=cfg['ema'], adjust=False).mean().values
    return pr

# ==========================================
# 3. ПРАВИЛО (без гистерезиса)
# ==========================================
def user_rule(sig, close_t, nb):
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
        eq += (close_t[-1] - entry) * pos; trades += 1
    return np.array(curve), eq, trades

# ==========================================
# 4. WF-СРАВНЕНИЕ КОНФИГУРАЦИЙ
# ==========================================
print(f"\n🔄 WF-сравнение выходов модели (N={NB})...")
print(f"{'конфигурация':<28} | {'mean':>8} {'winrate':>7} {'score':>7}")
best_cfg, best_score = None, -1e9
results = {}
for cfg in CONFIGS:
    totals = []
    for w in range(WF_WINDOWS):
        te = len(df) - w * WF_SIZE
        ts = te - WF_SIZE
        if ts < 1500: break
        pr = train_and_predict(df[feature_cols].iloc[:ts].values,
                               df[feature_cols].iloc[ts:te].values,
                               close[:ts], cfg)
        _, pm, _ = user_rule(pr[:WF_SIZE-2], close[ts:ts+WF_SIZE-2], NB)
        totals.append(pm)
    p = np.array(totals)
    score = p.mean() / (p.std() + 1e-12)
    results[cfg['name']] = (p.mean(), np.mean(p > 0), score)
    print(f"{cfg['name']:<28} | {p.mean():>8.5f} {np.mean(p > 0):>6.0%} {score:>7.2f}")
    if score > best_score: best_score, best_cfg = score, cfg

print(f"\n🏆 ЛУЧШИЙ ВЫХОД: {best_cfg['name']}")

# ==========================================
# 5. ФИНАЛЬНЫЙ ТЕСТ С ЛУЧШЕЙ КОНФИГУРАЦИЕЙ
# ==========================================
BARS_TRAIN = len(df) - BARS_TEST
preds = train_and_predict(df[feature_cols].iloc[:BARS_TRAIN].values,
                          df[feature_cols].iloc[BARS_TRAIN:].values,
                          close[:BARS_TRAIN], best_cfg)

sm_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
d_all = np.zeros_like(sm_all)
d_all[1:-1] = (sm_all[2:] - sm_all[:-2]) / 2.0
ideal_test = d_all[BARS_TRAIN:BARS_TRAIN + BARS_TEST]

n = BARS_TEST - 2
cm, pm, tm = user_rule(preds[:n], close[BARS_TRAIN:BARS_TRAIN+n], NB)
ci, pi, ti = user_rule(ideal_test[:n], close[BARS_TRAIN:BARS_TRAIN+n], NB)

print(f"\nФИНАЛ: модель PnL={pm:.4f}/{tm} сделок | идеал PnL={pi:.4f}/{ti} сделок")
print(f"модель берёт {pm/pi*100:.0f}% от потолка оракула")

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
x = np.arange(1, BARS_TEST + 1)
axes[0].plot(x, close[BARS_TRAIN:], 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} | {best_cfg["name"]} | N={NB}')
axes[0].grid(alpha=0.3)
axes[1].plot(cm, 'b', lw=1.8, label=f'Модель: {pm:.4f}')
axes[1].plot(ci, 'g', lw=1.8, label=f'ИДЕАЛ: {pi:.4f}')
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_xlabel('Тестовый бар')
plt.tight_layout(); plt.show()