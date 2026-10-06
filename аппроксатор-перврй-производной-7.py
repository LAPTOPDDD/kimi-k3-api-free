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

N_ESTIMATORS = 400
LEARNING_RATE = 0.03
MIN_CHILD_SAMPLES = 15

NB_LIST = [3, 4, 5, 6]
WF_WINDOWS = 10
WF_SIZE = 100

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
# 2. ФИЧИ: медленные (20) + быстрые (3,5,8)
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

for w in (3, 5, 8):
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
# 3. ОБУЧЕНИЕ + ПРЕДСКАЗАНИЕ НА ОКНЕ
# ==========================================
def train_and_predict(X_tr, X_te, y_tr):
    sm = gaussian_filter1d(y_tr, sigma=SIGMA, mode='reflect')
    dd = np.zeros_like(sm)
    dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    if SHIFT > 0:
        yf, Xf = dd[SHIFT:], X_tr[:-SHIFT]
    else:
        yf, Xf = dd, X_tr

    sc = StandardScaler()
    Xf_s, Xte_s = sc.fit_transform(Xf), sc.transform(X_te)

    m = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                          min_child_samples=MIN_CHILD_SAMPLES,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xf_s, yf * TARGET_SCALE)

    ptr = m.predict(Xf_s) / TARGET_SCALE
    kk = np.std(yf) / (np.std(ptr) + 1e-9)
    return m.predict(X_te) / TARGET_SCALE * kk if X_te.ndim == 1 else m.predict(Xte_s) / TARGET_SCALE * kk

# ==========================================
# 4. ПРАВИЛО: кросс нуля → вход, N баров, выход по сроку/встречному
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
        eq += (close_t[-1] - entry) * pos
        trades += 1
    return np.array(curve), eq, trades

# ==========================================
# 5. ИДЕАЛ НА ВСЕЙ СЕРИИ (оракул для диагностики)
# ==========================================
sm_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
d_all = np.zeros_like(sm_all)
d_all[1:-1] = (sm_all[2:] - sm_all[:-2]) / 2.0

# ==========================================
# 6. WALK-FORWARD: выбор N + диагностика режимов
# ==========================================
print(f"\n🔄 Walk-forward: {WF_WINDOWS} окон...")
agg = {nb: [] for nb in NB_LIST}
detail = []

for w in range(WF_WINDOWS):
    test_end = len(df) - w * WF_SIZE
    test_start = test_end - WF_SIZE
    if test_start < 1500: break

    X_tr = df[feature_cols].iloc[:test_start].values
    X_te = df[feature_cols].iloc[test_start:test_end].values
    close_te = close[test_start:test_end]

    pr = train_and_predict(X_tr, X_te, close[:test_start])
    nn = WF_SIZE - 2

    rets = np.diff(close_te) / close_te[:-1]
    vol = np.std(rets)
    trend = abs(np.mean(rets)) / (vol + 1e-12)

    rec = {'w': w + 1, 'vol': vol, 'trend': trend}
    for nb in NB_LIST:
        _, pm, _ = user_rule(pr[:nn], close_te[:nn], nb)
        _, pi, _ = user_rule(d_all[test_start:test_start + nn], close_te[:nn], nb)
        agg[nb].append(pm)
        rec[nb] = (pm, pi)
    detail.append(rec)
    print(f"  окно {w+1}/{WF_WINDOWS} ✓")

# --- выбор N ---
print(f"\n{'N':>2} | {'mean':>8} {'std':>7} {'winrate':>7} {'score':>7}")
best_nb, best_score = None, -1e9
for nb in NB_LIST:
    p = np.array(agg[nb])
    score = p.mean() / (p.std() + 1e-12)
    print(f"{nb:>2} | {p.mean():>8.5f} {p.std():>7.5f} {np.mean(p > 0):>6.0%} {score:>7.2f}")
    if score > best_score:
        best_score, best_nb = score, nb
print(f"🏆 N = {best_nb}")

# --- диагностика по окнам при выбранном N ---
print(f"\nДиагностика при N={best_nb}:")
print(f"{'окно':>4} | {'модель':>8} | {'идеал':>8} | {'vol':>7} {'trend':>6}")
pm_a, pi_a, vol_a, tr_a = [], [], [], []
for r in detail:
    pm, pi = r[best_nb]
    pm_a.append(pm); pi_a.append(pi); vol_a.append(r['vol']); tr_a.append(r['trend'])
    print(f"{r['w']:>4} | {pm:>8.4f} | {pi:>8.4f} | {r['vol']:>7.5f} {r['trend']:>6.3f}")

pm_a, pi_a = np.array(pm_a), np.array(pi_a)
vol_a, tr_a = np.array(vol_a), np.array(tr_a)
print(f"\ncorr(модель, идеал) по окнам = {np.corrcoef(pm_a, pi_a)[0,1]:+.2f}")
print(f"corr(идеал, vol)             = {np.corrcoef(pi_a, vol_a)[0,1]:+.2f}")
print(f"corr(идеал, trend)           = {np.corrcoef(pi_a, tr_a)[0,1]:+.2f}")
print(f"идеал: mean={pi_a.mean():.5f} winrate={np.mean(pi_a > 0):.0%}")

# ==========================================
# 7. ФИНАЛЬНЫЙ ТЕСТ: последние 100 баров
# ==========================================
BARS_TRAIN = len(df) - BARS_TEST
X_tr = df[feature_cols].iloc[:BARS_TRAIN].values
X_te = df[feature_cols].iloc[BARS_TRAIN:].values
y_te = close[BARS_TRAIN:]

preds = train_and_predict(X_tr, X_te, close[:BARS_TRAIN])
ideal_test = d_all[BARS_TRAIN + SHIFT: BARS_TRAIN + SHIFT + BARS_TEST]
if len(ideal_test) < BARS_TEST:
    ideal_test = np.pad(ideal_test, (0, BARS_TEST - len(ideal_test)))

actual_forward = np.zeros(BARS_TEST)
for i in range(BARS_TEST - 2):
    actual_forward[i] = (close[BARS_TRAIN + i + 2] - close[BARS_TRAIN + i]) / 2.0
n = BARS_TEST - 2

print(f"\n{'='*52}")
print(f"  corr(модель, идеал)      = {np.corrcoef(preds[:n], ideal_test[:n])[0,1]:+.3f}")
print(f"  corr(идеал, реальность)  = {np.corrcoef(ideal_test[:n], actual_forward[:n])[0,1]:+.3f}")
print(f"  corr(модель, реальность) = {np.corrcoef(preds[:n], actual_forward[:n])[0,1]:+.3f}")

cm, pm, tm = user_rule(preds[:n], y_te[:n], best_nb)
ci, pi, ti = user_rule(ideal_test[:n], y_te[:n], best_nb)
print(f"  модель: PnL={pm:.4f} сделок={tm} | идеал: PnL={pi:.4f} сделок={ti}")
print(f"{'='*52}")

# ==========================================
# 8. ГРАФИКИ
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                         gridspec_kw={'height_ratios': [1.3, 1, 0.9]})
x = np.arange(1, BARS_TEST + 1)

axes[0].plot(x, y_te, 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} | σ={SIGMA} | N={best_nb} (без гистерезиса)')
axes[0].grid(alpha=0.3)

axes[1].plot(x[:n], actual_forward[:n], 'r', alpha=0.5, lw=1, label='Реальность')
axes[1].plot(x[:n], ideal_test[:n], 'g', lw=2, label='ИДЕАЛ')
axes[1].plot(x[:n], preds[:n], 'b', lw=1.5, label='Модель')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(cm, 'b', lw=1.8, label=f'Модель: {pm:.4f} / {tm} сделок')
axes[2].plot(ci, 'g', lw=1.8, label=f'ИДЕАЛ: {pi:.4f} / {ti} сделок')
axes[2].axhline(0, color='gray', lw=0.8)
axes[2].set_title('Эквити без комиссий')
axes[2].legend(loc='upper left'); axes[2].grid(alpha=0.3)
axes[2].set_xlabel('Тестовый бар')

plt.tight_layout()
plt.savefig('final_no_hysteresis.png', dpi=150)
plt.show()

print(f"\n📋 КОНФИГУРАЦИЯ: SYMBOL={SYMBOL} SIGMA={SIGMA} SHIFT={SHIFT} N={best_nb}")