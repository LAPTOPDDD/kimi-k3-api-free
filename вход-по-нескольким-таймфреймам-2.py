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
# (имя, TF, всего баров, sigma, train_cap)
TFS = [("M15", mt5.TIMEFRAME_M15, 20000, 1, 8000),
       ("M30", mt5.TIMEFRAME_M30, 12000, 1, 6000),
       ("H1",  mt5.TIMEFRAME_H1,   6000, 1, 4000),
       ("H4",  mt5.TIMEFRAME_H4,   3000, 2, 1500)]
TF_SIG = {n: s for n, _, _, s, _ in TFS}
TF_CAP = {n: c for n, _, _, _, c in TFS}

TARGET_SCALE = 10000
TEST_M15 = 400
WF_WINDOWS = 20          # ≈83 дня разных режимов
NB = 5                   # таймаут не срабатывает — не важен

# ==========================================
# 1. ЗАГРУЗКА ВСЕХ ТФ
# ==========================================
def prepare(df):
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
    return df.dropna().copy()

if not mt5.initialize(): print("❌ MT5"); exit()
data = {}
for name, tf, total, sig, cap in TFS:
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, total)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
    df = df[df.index.dayofweek < 5].copy()
    data[name] = prepare(df)
mt5.shutdown()
print("✅ Загружено:", {k: len(v) for k, v in data.items()})

fcols = [c for c in data['M15'].columns if c not in
         ['open','high','low','close','tick_volume','spread','real_volume']]

# ==========================================
# 2. МОДЕЛЬ НА ТФ + ПРЕДСКАЗАНИЕ ВСЕЙ СЕРИИ
# ==========================================
def tf_predictions(name, test_start_time):
    df = data[name]
    times = df.index.astype('int64').values
    tr_idx = np.where(df.index < test_start_time)[0]
    tr_idx = tr_idx[-TF_CAP[name]:] if len(tr_idx) > TF_CAP[name] else tr_idx
    X_tr, c_tr = df[fcols].values[tr_idx], df['close'].values[tr_idx]

    sm = gaussian_filter1d(c_tr, sigma=TF_SIG[name], mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    vals = m.predict(sc.transform(df[fcols].values)) / TARGET_SCALE * k
    return times, vals

def align(tf_times, tf_vals, m15_times):
    idx = np.searchsorted(tf_times, m15_times, side='right') - 1
    return tf_vals[np.clip(idx, 0, len(tf_vals) - 1)]

# ==========================================
# 3. ПРАВИЛО С КОНФЛЮЭНЦЕЙ
# ==========================================
def confluence_rule(act_long, act_short, close_t, nb):
    eq, pos, entry, held, tr = 0.0, 0, 0.0, 0, 0
    curve = []
    for t in range(len(close_t)):
        p = close_t[t]
        if pos == 1:
            held += 1
            if held >= nb or act_short[t] or not act_long[t]:
                eq += (p - entry); tr += 1; pos = 0
        elif pos == -1:
            held += 1
            if held >= nb or act_long[t] or not act_short[t]:
                eq += (entry - p); tr += 1; pos = 0
        if pos == 0:
            if act_long[t]: pos, entry, held = 1, p, 0
            elif act_short[t]: pos, entry, held = -1, p, 0
        curve.append(eq)
    if pos == 1: eq += (close_t[-1] - entry)
    if pos == -1: eq += (entry - close_t[-1])
    return np.array(curve), eq, tr

# ==========================================
# 4. WF-ТЕСТ: 20 ОКОН + БАЗЛАЙНЫ
# ==========================================
m15 = data['M15']
m15_times_all = m15.index.astype('int64').values
res = {v: [] for v in ['ALL4', 'H4only', 'M15only', 'BH']}
longs, shorts = 0, 0

for w in range(WF_WINDOWS):
    te_end = len(m15) - w * TEST_M15
    te_start = te_end - TEST_M15
    test_start_time = m15.index[te_start]

    aligned = {}
    for name, _, _, _, _ in TFS:
        t_times, t_vals = tf_predictions(name, test_start_time)
        aligned[name] = align(t_times, t_vals, m15_times_all[te_start:te_end])

    close_t = m15['close'].values[te_start:te_end]
    act4L = np.all([aligned[n] > 0 for n in aligned], axis=0)
    act4S = np.all([aligned[n] < 0 for n in aligned], axis=0)
    actHL, actHS = aligned['H4'] > 0, aligned['H4'] < 0
    act1L, act1S = aligned['M15'] > 0, aligned['M15'] < 0

    _, p4, t4 = confluence_rule(act4L, act4S, close_t, NB)
    _, pH, _  = confluence_rule(actHL, actHS, close_t, NB)
    _, p1, _  = confluence_rule(act1L, act1S, close_t, NB)
    pBH = close_t[-1] - close_t[0]

    res['ALL4'].append(p4); res['H4only'].append(pH)
    res['M15only'].append(p1); res['BH'].append(pBH)

    pos = 0
    for t in range(len(close_t)):
        if pos == 0 and act4L[t]: pos, longs = 1, longs + 1
        elif pos == 0 and act4S[t]: pos, shorts = -1, shorts + 1
        elif pos == 1 and (act4S[t] or not act4L[t]): pos = 0
        elif pos == -1 and (act4L[t] or not act4S[t]): pos = 0

    print(f"  окно {w+1:>2}: ALL4 {p4:>8.4f} | H4 {pH:>8.4f} | B&H {pBH:>8.4f}")

# ==========================================
# 5. АГРЕГАТ + РАЗБИВКА ПО РЕЖИМАМ
# ==========================================
print(f"\nсделки ALL4: лонг={longs} шорт={shorts}")
print(f"\n{'='*62}")
print(f"  {'вариант':>8} | {'mean':>8} {'std':>7} {'winrate':>7} {'score':>6}")
for v in ['ALL4', 'H4only', 'M15only', 'BH']:
    a = np.array(res[v])
    print(f"  {v:>8} | {a.mean():>8.5f} {a.std():>7.5f} "
          f"{np.mean(a > 0):>6.0%} {a.mean()/(a.std()+1e-12):>6.2f}")
print(f"{'='*62}")

a4, aH, aB = np.array(res['ALL4']), np.array(res['H4only']), np.array(res['BH'])
up, flat, down = np.where(aB > 0.002)[0], np.where(np.abs(aB) <= 0.002)[0], np.where(aB < -0.002)[0]
print(f"\n  режим  окон | ALL4 mean | H4 mean | B&H mean")
for nm, ix in [('UP', up), ('FLAT', flat), ('DOWN', down)]:
    m4 = a4[ix].mean() if len(ix) else 0.0
    mH = aH[ix].mean() if len(ix) else 0.0
    mB = aB[ix].mean() if len(ix) else 0.0
    print(f"  {nm:>4} {len(ix):>5} | {m4:>9.5f} | {mH:>7.5f} | {mB:>8.5f}")

# ==========================================
# 6. ГРАФИКИ: пооконно + кумулятив (в хронологии)
# ==========================================
order = list(range(WF_WINDOWS))[::-1]   # старые окна слева
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
xw = np.arange(1, WF_WINDOWS + 1)
w_ = 0.27
axes[0].bar(xw - w_, a4[order], w_, label='ALL4 (конфлюэнция)', color='b')
axes[0].bar(xw,     aH[order], w_, label='H4only', color='orange')
axes[0].bar(xw + w_, aB[order], w_, label='Buy&Hold', color='gray', alpha=0.6)
axes[0].axhline(0, color='k', lw=0.8); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title(f'{SYMBOL} | PnL по окнам (хронология слева→направо)')

axes[1].plot(xw, np.cumsum(a4[order]), 'b', lw=2.5, label='ALL4')
axes[1].plot(xw, np.cumsum(aH[order]), color='orange', lw=1.5, label='H4only')
axes[1].plot(xw, np.cumsum(aB[order]), color='gray', lw=1.2, alpha=0.7, label='Buy&Hold')
axes[1].axhline(0, color='k', lw=0.8); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_title('Кумулятивный PnL по окнам')
axes[1].set_xlabel('Окно (старое → новое)')
plt.tight_layout()
plt.savefig('confluence_20w.png', dpi=150)
plt.show()