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
TARGET_SCALE = 10000
TEST_M15 = 400
WF_WINDOWS = 4
NB_LIST = [3, 5, 10]

# ==========================================
# ЗАГРУЗКА ВСЕХ ТФ
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
    df = prepare(df)
    data[name] = df
mt5.shutdown()
print("✅ Загружено:", {k: len(v) for k, v in data.items()})

fcols = [c for c in data['M15'].columns if c not in
         ['open','high','low','close','tick_volume','spread','real_volume']]

# ==========================================
# МОДЕЛЬ НА ТФ + предсказание всей серии
# ==========================================
def tf_predictions(name, test_start_time):
    df = data[name]
    sig = dict((n, s) for n, _, _, s, _ in TFS)[name]
    cap = dict((n, c) for n, _, _, _, c in TFS)[name]
    times = df.index.astype('int64').values
    tr_mask = df.index < test_start_time
    tr_idx = np.where(tr_mask)[0]
    tr_idx = tr_idx[-cap:] if len(tr_idx) > cap else tr_idx
    X_tr = df[fcols].values[tr_idx]; c_tr = df['close'].values[tr_idx]

    sm = gaussian_filter1d(c_tr, sigma=sig, mode='reflect')
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
# ПРАВИЛО С КОНФЛЮЭНЦЕЙ
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
# WF-ТЕСТ: КОНФЛЮЭНЦИЯ vs M15-ОДИНОЧКА
# ==========================================
m15 = data['M15']
m15_times_all = m15.index.astype('int64').values
res = {('ALL4', nb): [] for nb in NB_LIST}
res.update({('M15only', nb): [] for nb in NB_LIST})
final_curves = {}

for w in range(WF_WINDOWS):
    te_end = len(m15) - w * TEST_M15
    te_start = te_end - TEST_M15
    test_start_time = m15.index[te_start]

    aligned = {}
    for name, _, _, _, _ in TFS:
        t_times, t_vals = tf_predictions(name, test_start_time)
        aligned[name] = align(t_times, t_vals, m15_times_all[te_start:te_end])

    close_t = m15['close'].values[te_start:te_end]
    act_long = np.all([aligned[n] > 0 for n in aligned], axis=0)
    act_short = np.all([aligned[n] < 0 for n in aligned], axis=0)
    act_long_1 = aligned['M15'] > 0
    act_short_1 = aligned['M15'] < 0

    for nb in NB_LIST:
        _, p4, t4 = confluence_rule(act_long, act_short, close_t, nb)
        _, p1, t1 = confluence_rule(act_long_1, act_short_1, close_t, nb)
        res[('ALL4', nb)].append(p4)
        res[('M15only', nb)].append(p1)
        if w == 0 and nb == 5:
            c4, _, _ = confluence_rule(act_long, act_short, close_t, nb)
            c1, _, _ = confluence_rule(act_long_1, act_short_1, close_t, nb)
            final_curves = {'ALL4': c4, 'M15only': c1}
    print(f"  окно {w+1}/{WF_WINDOWS} ✓ | ALL4 активен {np.mean(act_long | act_short):.0%} времени")

print(f"\n{'='*56}")
print(f"  {'вариант':>9} | {'NB':>2} | {'mean':>8} {'winrate':>7}")
for nb in NB_LIST:
    for var in ['ALL4', 'M15only']:
        a = np.array(res[(var, nb)])
        print(f"  {var:>9} | {nb:>2} | {a.mean():>8.5f} {np.mean(a > 0):>6.0%}")
print(f"{'='*56}")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(final_curves['ALL4'], 'b', lw=2, label='КОНФЛЮЭНЦИЯ: все 4 ТФ > 0')
ax.plot(final_curves['M15only'], 'r--', lw=1.2, alpha=0.7, label='M15 одиночка')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title(f'{SYMBOL} | вход на M15 при согласии M15+M30+H1+H4 | N=5')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()