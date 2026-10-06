import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

SYMBOL = "EURUSD"
COST = 0.0001
TFS = [("M15", mt5.TIMEFRAME_M15, 20000, 1, 8000),
       ("M30", mt5.TIMEFRAME_M30, 12000, 1, 6000),
       ("H1",  mt5.TIMEFRAME_H1,   6000, 1, 4000),
       ("H4",  mt5.TIMEFRAME_H4,   3000, 2, 1500)]
TF_MINUTES = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}
NAMES = ["M15", "M30", "H1", "H4"]
TARGET_SCALE = 10000
TEST_M15 = 400
WF_WINDOWS = 20

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
fcols = [c for c in data['M15'].columns if c not in
         ['open','high','low','close','tick_volume','spread','real_volume']]

def tf_predictions(name, test_start_time):
    df = data[name]
    sig = dict((n, s) for n, _, _, s, _ in TFS)[name]
    cap = dict((n, c) for n, _, _, _, c in TFS)[name]
    close_times = df.index + pd.Timedelta(minutes=TF_MINUTES[name])
    tr_idx = np.where(close_times <= test_start_time)[0]
    tr_idx = tr_idx[-cap:] if len(tr_idx) > cap else tr_idx
    X_tr, c_tr = df[fcols].values[tr_idx], df['close'].values[tr_idx]
    sm = gaussian_filter1d(c_tr, sigma=sig, mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    vals = m.predict(sc.transform(df[fcols].values)) / TARGET_SCALE * k
    return df.index.astype('int64').values, vals

def align(tf_name, tf_times, tf_vals, m15_times):
    tf_c  = tf_times  + TF_MINUTES[tf_name] * 60_000_000_000
    m15_c = m15_times + 15 * 60_000_000_000
    idx = np.searchsorted(tf_c, m15_c, side='right') - 1
    return tf_vals[np.clip(idx, 0, len(tf_vals) - 1)]

def eval_rule(d, close, cost):
    eq, tr, pos, entry = 0.0, 0, 0, 0.0
    for t in range(len(d)):
        tgt = d[t]
        if tgt != pos:
            if pos != 0: eq += (close[t] - entry) * pos - cost; tr += 1
            pos = tgt
            if pos != 0: entry = close[t]
    if pos != 0: eq += (close[-1] - entry) * pos - cost; tr += 1
    return eq, tr

# ==========================================
# 24 ПРАВИЛА: (старший a, младший b) × {лонг, шорт}
# ==========================================
RULES = []
for a in NAMES:
    for b in NAMES:
        if a != b:
            RULES.append((a, b, 'L'))   # лонг, пока a>0 и b<0
            RULES.append((a, b, 'S'))   # шорт, пока a<0 и b>0
res = {r: [] for r in RULES}

m15 = data['M15']; m15t = m15.index.astype('int64').values
for w in range(WF_WINDOWS):
    te_end = len(m15) - w * TEST_M15
    te_start = te_end - TEST_M15
    if te_start < 1600: break
    aligned = {}
    for name in NAMES:
        t_times, t_vals = tf_predictions(name, m15.index[te_start])
        aligned[name] = align(name, t_times, t_vals, m15t[te_start:te_end])
    close_t = m15['close'].values[te_start:te_end]

    for (a, b, side) in RULES:
        if side == 'L':
            cond = (aligned[a] > 0) & (aligned[b] < 0)
            d = np.where(cond, 1, 0)
        else:
            cond = (aligned[a] < 0) & (aligned[b] > 0)
            d = np.where(cond, -1, 0)
        eq, tr = eval_rule(d, close_t, COST)
        res[(a, b, side)].append(eq)
    print(f"  окно {w+1}/{WF_WINDOWS} ✓")

# ==========================================
# ТАБЛИЦА 24 ПРАВИЛ
# ==========================================
rows = []
for r in RULES:
    a = np.array(res[r])
    rows.append((f"{r[0]}>{r[1]}{'+' if r[2]=='L' else '-'}", a.mean(), a.std(),
                 np.mean(a > 0), a.mean() / (a.std() + 1e-12), len(a)))
rows.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'='*70}")
print(f"  24 ПРАВИЛА 'старший>0, младший<0' | каузально, 20 окон, косты {COST}")
print(f"  {'правило':>10} | {'mean':>9} {'std':>8} {'wr':>5} {'score':>6}")
for name, m, s, wr, sc, n in rows:
    print(f"  {name:>10} | {m:>9.5f} {s:>8.5f} {wr:>4.0%} {sc:>6.2f}")
print(f"{'='*70}")
alive = [r for r in rows if r[1] > 0 and r[3] >= 0.55]
print(f"  Живых (mean>0 и wr≥55%): {len(alive)}")
for r in alive: print(f"    → {r[0]}: mean={r[1]:.5f} wr={r[3]:.0%}")