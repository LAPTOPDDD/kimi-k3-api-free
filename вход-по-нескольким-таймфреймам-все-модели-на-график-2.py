import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SYMBOL = "EURUSD"
COST = 0.0001
TFS = [("M15", mt5.TIMEFRAME_M15, 20000, 1, 8000),
       ("M30", mt5.TIMEFRAME_M30, 12000, 1, 6000),
       ("H1",  mt5.TIMEFRAME_H1,   6000, 1, 4000),
       ("H4",  mt5.TIMEFRAME_H4,   3000, 2, 1500)]
TF_MINUTES = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}
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
    """d[t] — желаемая позиция на баре t; PnL = движения баров t+1..; кост за круг"""
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
# WF-ТЕСТ ПЯТИ ПРАВИЛ
# ==========================================
m15 = data['M15']; m15t = m15.index.astype('int64').values
RULES = ['ALL4', 'H4+M15', 'H4+H1', 'H4only', 'FADE_M15']
res = {r: [] for r in RULES}; trades = {r: 0 for r in RULES}

for w in range(WF_WINDOWS):
    te_end = len(m15) - w * TEST_M15
    te_start = te_end - TEST_M15
    if te_start < 1600: break
    aligned = {}
    for name, _, _, _, _ in TFS:
        t_times, t_vals = tf_predictions(name, m15.index[te_start])
        aligned[name] = align(name, t_times, t_vals, m15t[te_start:te_end])
    close_t = m15['close'].values[te_start:te_end]
    sM, sH1, sH4 = aligned['M15'], aligned['H1'], aligned['H4']
    gM, gH1, gH4 = np.sign(sM), np.sign(sH1), np.sign(sH4)
    allP = np.all([aligned[n] > 0 for n in aligned], axis=0)
    allS = np.all([aligned[n] < 0 for n in aligned], axis=0)

    d = {
        'ALL4':    np.where(allP, 1, np.where(allS, -1, 0)),
        'H4+M15':  np.where(gM == gH4, gH4, 0),
        'H4+H1':   np.where(gH1 == gH4, gH4, 0),
        'H4only':  gH4,
        'FADE_M15': np.where((gM != gH4) & (gH4 != 0), gH4, 0),
    }
    for r in RULES:
        eq, tr = eval_rule(d[r], close_t, COST)
        res[r].append(eq); trades[r] += tr
    print(f"  окно {w+1}/{WF_WINDOWS} ✓")

print(f"\n{'='*66}")
print(f"  {'правило':>9} | {'mean':>9} {'std':>8} {'winrate':>7} {'score':>6} | сделок")
for r in RULES:
    a = np.array(res[r])
    print(f"  {r:>9} | {a.mean():>9.5f} {a.std():>8.5f} {np.mean(a > 0):>6.0%} "
          f"{a.mean()/(a.std()+1e-12):>6.2f} | {trades[r]}")
print(f"{'='*66}")

fig, ax = plt.subplots(figsize=(16, 6))
for r in RULES:
    a = np.array(res[r])
    ax.plot(np.arange(1, len(a)+1), np.cumsum(a[::-1]), lw=2 if r == 'FADE_M15' else 1.2,
            label=f'{r}: {a.mean():.5f}')
ax.axhline(0, color='gray', lw=0.8); ax.legend(); ax.grid(alpha=0.3)
ax.set_title(f'{SYMBOL} | НОВЫЕ ПРАВИЛА ВХОДА | каузально, 20 окон, косты {COST}')
plt.tight_layout(); plt.savefig('new_rules.png', dpi=150); plt.show()