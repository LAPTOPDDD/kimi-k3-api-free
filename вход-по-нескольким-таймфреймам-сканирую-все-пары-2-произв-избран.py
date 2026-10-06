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

# ==========================================
# НАСТРОЙКИ
# ==========================================
SYMBOLS = ["MSFT", "AMD", "NVDA", "INTC", "XAUUSD"]
COST = {"MSFT": 0.10, "AMD": 0.05, "NVDA": 0.05, "INTC": 0.02, "XAUUSD": 0.60}  # ≈ спред за круг, правь под брокера
TFS = [("M15", mt5.TIMEFRAME_M15, 20000, 1, 8000),
       ("M30", mt5.TIMEFRAME_M30, 12000, 1, 6000),
       ("H1",  mt5.TIMEFRAME_H1,   6000, 1, 4000),
       ("H4",  mt5.TIMEFRAME_H4,   3000, 2, 1500)]
TF_MINUTES = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}
TARGET_SCALE = 10000
TEST_M15 = 400
WF_WINDOWS = 20          # ≈4–8 месяцев в зависимости от ликвидности
NB = 5

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

def load_symbol(sym):
    data = {}
    for name, tf, total, sig, cap in TFS:
        rates = mt5.copy_rates_from_pos(sym, tf, 0, total)
        if rates is None or len(rates) < 1000: return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
        df = df[df.index.dayofweek < 5].copy()
        df = prepare(df)
        if len(df) < 1000: return None
        data[name] = df
    return data

def tf_predictions(data, fcols, name, test_start_time, order):
    df = data[name]
    sig = dict((n, s) for n, _, _, s, _ in TFS)[name]
    cap = dict((n, c) for n, _, _, _, c in TFS)[name]
    close_times = df.index + pd.Timedelta(minutes=TF_MINUTES[name])
    tr_idx = np.where(close_times <= test_start_time)[0]
    tr_idx = tr_idx[-cap:] if len(tr_idx) > cap else tr_idx
    X_tr, c_tr = df[fcols].values[tr_idx], df['close'].values[tr_idx]
    dd = gaussian_filter1d(c_tr, sigma=sig, order=order, mode='reflect')
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

def confluence_rule(act_long, act_short, close_t, nb, cost):
    eq, pos, entry, tr = 0.0, 0, 0.0, 0
    for t in range(len(close_t)):
        p = close_t[t]
        if pos == 1 and (act_short[t] or not act_long[t]):
            eq += (p - entry) - cost; tr += 1; pos = 0
        elif pos == -1 and (act_long[t] or not act_short[t]):
            eq += (entry - p) - cost; tr += 1; pos = 0
        if pos == 0:
            if act_long[t]: pos, entry = 1, p
            elif act_short[t]: pos, entry = -1, p
    if pos == 1: eq += (close_t[-1] - entry) - cost
    if pos == -1: eq += (entry - close_t[-1]) - cost
    return eq, tr

# ==========================================
# ПРОГОН: символ × учитель
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
rows = []
fig, axes = plt.subplots(len(SYMBOLS), 1, figsize=(14, 4 * len(SYMBOLS)), squeeze=False)

for si_, sym in enumerate(SYMBOLS):
    data = load_symbol(sym)
    if data is None:
        print(f"{sym}: мало данных"); continue
    fcols = [c for c in data['M15'].columns if c not in
             ['open','high','low','close','tick_volume','spread','real_volume']]
    m15 = data['M15']; m15t = m15.index.astype('int64').values
    cost = COST.get(sym, 0.0)

    for order, lab in [(1, 'd1'), (2, 'd2')]:
        pcts, trs = [], 0
        for w in range(WF_WINDOWS):
            te_end = len(m15) - w * TEST_M15
            te_start = te_end - TEST_M15
            if te_start < 1600: break
            aligned = {}
            for name, _, _, _, _ in TFS:
                t_times, t_vals = tf_predictions(data, fcols, name, m15.index[te_start], order)
                aligned[name] = align(name, t_times, t_vals, m15t[te_start:te_end])
            close_t = m15['close'].values[te_start:te_end]
            actL = np.all([aligned[n] > 0 for n in aligned], axis=0)
            actS = np.all([aligned[n] < 0 for n in aligned], axis=0)
            pnl, tr = confluence_rule(actL, actS, close_t, NB, cost)
            pcts.append(pnl / np.mean(close_t) * 100); trs += tr
        a = np.array(pcts)
        if len(a) == 0: continue
        bh = (m15['close'].values[len(m15)-1] - m15['close'].values[len(m15)-TEST_M15])  # не используется
        rows.append(dict(sym=sym, teacher=lab, mean_pct=a.mean(), wr=np.mean(a > 0),
                         score=a.mean() / (a.std() + 1e-12), trades=trs, windows=len(a),
                         curve=np.cumsum(a[::-1])))
        print(f"{sym} {lab}: mean={a.mean():>7.3f}% wr={np.mean(a > 0):>4.0%} "
              f"score={a.mean()/(a.std()+1e-12):>5.2f} сделок={trs} окон={len(a)}")

    for r in [r for r in rows if r['sym'] == sym]:
        axes[si_][0].plot(r['curve'], lw=2, label=f"{r['teacher']}: {r['mean_pct']:.2f}%/окно")
    axes[si_][0].axhline(0, color='gray', lw=0.8)
    axes[si_][0].set_title(f"{sym} | % эквити по окнам (старое→новое) | косты учтены")
    axes[si_][0].legend(); axes[si_][0].grid(alpha=0.3)
mt5.shutdown()

plt.tight_layout(); plt.savefig('deep_dive_d1d2.png', dpi=120); plt.show()

df_res = pd.DataFrame(rows)[['sym','teacher','mean_pct','wr','score','trades','windows']]
print(f"\n{'='*70}")
print(df_res.to_string(index=False))
print(f"{'='*70}")