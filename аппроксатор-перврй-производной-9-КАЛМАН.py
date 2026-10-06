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

SIGMA = 1
TARGET_SCALE = 10000
NB = 4

WF_WINDOWS = 5
WF_SIZE = 100
TRAIN_BARS = 3000

# ==========================================
# 1. ДАННЫЕ + ФИЧИ
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
fcols = [c for c in df.columns if c not in
         ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
X_all = df[fcols].values
T = len(df)
print(f"✅ {T} баров, {len(fcols)} фич")

# учитель
sm_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
ideal_all = np.zeros_like(sm_all)
ideal_all[1:-1] = (sm_all[2:] - sm_all[:-2]) / 2.0

# ==========================================
# 2. ПРАВИЛО: твоё (кросс + N + встречный)
# ==========================================
def user_rule(sig, close_t, nb):
    eq, pos, entry, held, trades = 0.0, 0, 0.0, 0, 0
    curve, prev_s = [], 0
    for t in range(len(sig)):
        p = close_t[t]; s = int(np.sign(sig[t]))
        cross = (s != 0) and (s != prev_s)
        if pos != 0:
            held += 1
            if held >= nb or (cross and s == -pos):
                eq += (p - entry) * pos; trades += 1; pos = 0
        if cross and pos == 0: pos, entry, held = s, p, 0
        prev_s = s; curve.append(eq)
    if pos != 0: eq += (close_t[-1] - entry) * pos; trades += 1
    return np.array(curve), eq, trades

# ==========================================
# 3. ОБУЧЕНИЕ НА ОКНЕ
# ==========================================
def fit_predict(start, end):
    X_tr, c_tr = X_all[max(0, start - TRAIN_BARS):start], close[max(0, start - TRAIN_BARS):start]
    X_te, c_te = X_all[start:end], close[start:end]
    sm = gaussian_filter1d(c_tr, sigma=SIGMA, mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    pred_te = m.predict(sc.transform(X_te)) / TARGET_SCALE * k
    pred_tr = m.predict(Xtr_s) / TARGET_SCALE * k
    return pred_tr, pred_te

# ==========================================
# 4. ГЕЙТ-ПРАВИЛО: торгуем качок только если предыдущий был сильным
# ==========================================
def extract_amps(sig):
    """амплитуды качков сигнала"""
    amps, ps, ca = [], 0, 0.0
    for v in sig:
        s = int(np.sign(v)); ca = max(ca, abs(v))
        if s != 0 and s != ps:
            amps.append(ca); ca = 0.0
        ps = s
    return amps

def gated_rule(sig, close_t, nb, thr):
    eq, pos, entry, held, trades = 0.0, 0, 0.0, 0, 0
    curve, prev_s = [], 0
    last_amp, cur_amp = 0.0, 0.0
    for t in range(len(sig)):
        p = close_t[t]; s = int(np.sign(sig[t]))
        cur_amp = max(cur_amp, abs(sig[t]))
        cross = (s != 0) and (s != prev_s)
        if pos != 0:
            held += 1
            if held >= nb or (cross and s == -pos):
                eq += (p - entry) * pos; trades += 1; pos = 0
        if cross and pos == 0:
            if last_amp >= thr:
                pos, entry, held = s, p, 0
            last_amp, cur_amp = cur_amp, 0.0
        prev_s = s; curve.append(eq)
    if pos != 0: eq += (close_t[-1] - entry) * pos; trades += 1
    return np.array(curve), eq, trades

# ==========================================
# 5. WF-СРАВНЕНИЕ: base vs gated
# ==========================================
print(f"\n🔄 Walk-forward: {WF_WINDOWS} окон...")
print(f"{'окно':>4} | {'base':>8} | {'gated':>8} | {'thr':>6}")
base_a, gated_a = [], []
gated_curves, base_curves = [], []

for w in range(WF_WINDOWS):
    te_end = T - w * WF_SIZE
    te_start = te_end - WF_SIZE
    if te_start < 1000: break

    pred_tr, pred_te = fit_predict(te_start, te_end)
    cl = close[te_start:te_end]
    n = min(len(pred_te), len(cl)) - 2

    # порог = медиана амплитуд качков на train
    amps = extract_amps(pred_tr)
    thr = np.median(amps) if amps else 0.0

    _, pb, tb = user_rule(pred_te[:n], cl[:n], NB)
    _, pg, tg = gated_rule(pred_te[:n], cl[:n], NB, thr)

    base_a.append(pb); gated_a.append(pg)
    base_c, _, _ = user_rule(pred_te[:n], cl[:n], NB)
    gate_c, _, _ = gated_rule(pred_te[:n], cl[:n], NB, thr)
    base_curves.append(base_c); gated_curves.append(gate_c)

    print(f"{w+1:>4} | {pb:>8.4f} | {pg:>8.4f} | {thr:>6.5f}")

base_a = np.array(base_a); gated_a = np.array(gated_a)

print(f"\n{'='*44}")
print(f"  BASE:  mean={base_a.mean():>8.5f} | winrate={np.mean(base_a > 0):>4.0%}")
print(f"  GATED: mean={gated_a.mean():>8.5f} | winrate={np.mean(gated_a > 0):>4.0%}")
print(f"{'='*44}")

if gated_a.mean() > base_a.mean() * 1.3 and np.mean(gated_a > 0) >= 0.6:
    print("✅ Гейт работает: торговля в сильных режимах улучшает результат")
elif np.abs(gated_a.mean() - base_a.mean()) < 1e-5:
    print("⚠️  Гейт не изменил: персистентности качков нет")
else:
    print("❌ Гейт ухудшил: режем слишком много")

# ==========================================
# 6. ФИНАЛ: графики на последнем окне
# ==========================================
te_end, te_start = T, T - WF_SIZE
pred_tr, pred_te = fit_predict(te_start, te_end)
cl = close[te_start:te_end]
n = min(len(pred_te), len(cl)) - 2

amps = extract_amps(pred_tr)
thr = np.median(amps) if amps else 0.0

base_c, pb, tb = user_rule(pred_te[:n], cl[:n], NB)
gate_c, pg, tg = gated_rule(pred_te[:n], cl[:n], NB, thr)

fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                         gridspec_kw={'height_ratios': [1.3, 1, 1]})
x = np.arange(1, len(cl) + 1)

axes[0].plot(x, cl, 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} | last {WF_SIZE} bars')
axes[0].set_ylabel('Цена'); axes[0].grid(alpha=0.3)

axes[1].plot(x[:n], ideal_all[te_start:te_start + n], 'g', lw=2, label='ИДЕАЛ')
axes[1].plot(x[:n], pred_te[:n], 'b', lw=1.5, label='Модель')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].axhline(thr, color='orange', ls='--', lw=1, label=f'Гейт-порог={thr:.5f}')
axes[1].axhline(-thr, color='orange', ls='--', lw=1)
axes[1].legend(); axes[1].set_ylabel('Сигнал'); axes[1].grid(alpha=0.3)

axes[2].plot(base_c, 'b--', lw=1.2, alpha=0.7, label=f'BASE: {pb:.4f} / {tb} сделок')
axes[2].plot(gate_c, 'r',   lw=1.8, label=f'GATED: {pg:.4f} / {tg} сделок')
axes[2].axhline(0, color='gray', lw=0.8)
axes[2].set_title(f'Эквити (без комиссий)')
axes[2].legend(); axes[2].set_ylabel('PnL'); axes[2].grid(alpha=0.3)
axes[2].set_xlabel('Тестовый бар')

plt.tight_layout()
plt.savefig('gated_vs_base.png', dpi=150)
plt.show()