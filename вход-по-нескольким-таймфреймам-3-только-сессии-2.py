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
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
PIP = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01}
COST_PIPS = 1.0                    # спред+комиссия за круг
TARGET_SCALE = 10000
WF_WINDOWS = 45                    # ≈190 дней — вся доступная история
TEST_M15 = 400
H4_SIGMA, H4_CAP = 2, 1500

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

def load_pair(sym):
    out = {}
    for name, tf, total in [("M15", mt5.TIMEFRAME_M15, 20000), ("H4", mt5.TIMEFRAME_H4, 3000)]:
        rates = mt5.copy_rates_from_pos(sym, tf, 0, total)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
        df = df[df.index.dayofweek < 5].copy()
        out[name] = prepare(df)
    return out

def h4_signal(h4_df, fcols, test_start_time, m15_times):
    times = h4_df.index.astype('int64').values
    tr_idx = np.where(h4_df.index < test_start_time)[0]
    tr_idx = tr_idx[-H4_CAP:] if len(tr_idx) > H4_CAP else tr_idx
    X_tr, c_tr = h4_df[fcols].values[tr_idx], h4_df['close'].values[tr_idx]
    sm = gaussian_filter1d(c_tr, sigma=H4_SIGMA, mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    vals = m.predict(sc.transform(h4_df[fcols].values)) / TARGET_SCALE * k
    idx = np.searchsorted(times, m15_times, side='right') - 1
    return vals[np.clip(idx, 0, len(vals) - 1)]

def swing_rule(sig, close_t, cost):
    """направление по знаку sig; вход/выход на смене знака; кост за сделку"""
    eq, pos, entry, tr = 0.0, 0, 0.0, 0
    curve, prev_s = [], 0
    for t in range(len(close_t)):
        p = close_t[t]; s = int(np.sign(sig[t]))
        if pos != 0 and s != 0 and s != pos:
            eq += (p - entry) * pos - cost; tr += 1; pos = 0
        if pos == 0 and s != 0:
            pos, entry = s, p
        prev_s = s; curve.append(eq)
    if pos != 0: eq += (close_t[-1] - entry) * pos - cost
    return np.array(curve), eq, tr

# ==========================================
# WF-ТЕСТ ПО ПАРАМ (с костами)
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
pair_res = {}
for sym in PAIRS:
    d = load_pair(sym)
    m15, h4 = d['M15'], d['H4']
    fcols = [c for c in m15.columns if c not in
             ['open','high','low','close','tick_volume','spread','real_volume']]
    m15t = m15.index.astype('int64').values
    cost = COST_PIPS * PIP[sym]
    wins = []
    for w in range(WF_WINDOWS):
        te_end = len(m15) - w * TEST_M15
        te_start = te_end - TEST_M15
        if te_start < 1600: break
        sig = h4_signal(h4, fcols, m15.index[te_start], m15t[te_start:te_end])
        cl = m15['close'].values[te_start:te_end]
        _, pnl, tr = swing_rule(sig[:len(cl)-2], cl[:len(cl)-2], cost)
        wins.append(pnl)
    pair_res[sym] = np.array(wins)
    a = pair_res[sym]
    print(f"{sym}: mean={a.mean():>8.5f} std={a.std():>7.5f} "
          f"winrate={np.mean(a > 0):>4.0%} score={a.mean()/(a.std()+1e-12):>5.2f} | окон={len(a)}")
mt5.shutdown()

# ==========================================
# ПОРТФЕЛЬ ТРЁХ ПАР
# ==========================================
minlen = min(len(a) for a in pair_res.values())
port = sum(a[:minlen] for a in pair_res.values())
print(f"\n{'='*56}")
print(f"  ПОРТФЕЛЬ (H4-swing, 3 пары, С КОСТАМИ):")
print(f"    mean={port.mean():.5f} std={port.std():.5f} "
          f"winrate={np.mean(port > 0):.0%} score={port.mean()/(port.std()+1e-12):.2f}")
print(f"{'='*56}")

fig, ax = plt.subplots(figsize=(16, 6))
for sym, a in pair_res.items():
    ax.plot(np.arange(1, len(a)+1)[::-1], np.cumsum(a[::-1]), lw=1.3, alpha=0.8, label=sym)
ax.plot(np.arange(1, minlen+1)[::-1], np.cumsum(port[::-1]), 'k', lw=3, label='ПОРТФЕЛЬ')
ax.axhline(0, color='gray', lw=0.8); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('H4-swing с костами: пары и портфель (старое → новое)')
plt.tight_layout(); plt.show()