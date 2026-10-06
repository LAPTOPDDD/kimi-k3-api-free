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
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"]
TIMEFRAME = mt5.TIMEFRAME_H1
TOTAL_BARS = 5000
SIGMA = 1
TARGET_SCALE = 10000
NB = 4
WF_WINDOWS = 5
WF_SIZE = 100
TRAIN_BARS = 3000

# ==========================================
# ФУНКЦИИ
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

def user_rule_ret(sig, close_t, nb):
    """эквити в ПРОЦЕНТАХ (сравнимо между парами)"""
    eq, pos, entry, held = 0.0, 0, 1.0, 0
    curve, prev_s = [], 0
    for t in range(len(sig)):
        p = close_t[t]; s = int(np.sign(sig[t]))
        cross = (s != 0) and (s != prev_s)
        if pos != 0:
            held += 1
            if held >= nb or (cross and s == -pos):
                eq += (p / entry - 1) * pos; pos = 0
        if cross and pos == 0: pos, entry, held = s, p, 0
        prev_s = s; curve.append(eq)
    if pos != 0: eq += (close_t[-1] / entry - 1) * pos
    return np.array(curve), eq

def fit_predict(X_tr, c_tr, X_te):
    sm = gaussian_filter1d(c_tr, sigma=SIGMA, mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    return m.predict(sc.transform(X_te)) / TARGET_SCALE * k

# ==========================================
# WF-МАТРИЦА: пары × окна
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
pair_windows = {}
final_curves = {}

for sym in SYMBOLS:
    rates = mt5.copy_rates_from_pos(sym, TIMEFRAME, 0, TOTAL_BARS)
    if rates is None: continue
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
    df = df[df.index.dayofweek < 5].copy()
    df = prepare(df)
    close = df['close'].values
    fcols = [c for c in df.columns if c not in
             ['open','high','low','close','tick_volume','spread','real_volume']]
    X_all = df[fcols].values
    T = len(df)

    wins = []
    for w in range(WF_WINDOWS):
        te_end = T - w * WF_SIZE
        te_start = te_end - WF_SIZE
        if te_start < TRAIN_BARS + 100: break
        pred = fit_predict(X_all[te_start-TRAIN_BARS:te_start], close[te_start-TRAIN_BARS:te_start],
                           X_all[te_start:te_end])
        n = WF_SIZE - 2
        _, pnl = user_rule_ret(pred[:n], close[te_start:te_start+n], NB)
        wins.append(pnl)
        if w == 0:
            c, _ = user_rule_ret(pred[:n], close[te_start:te_start+n], NB)
            final_curves[sym] = c
    pair_windows[sym] = np.array(wins)
    print(f"{sym}: mean={np.mean(wins):>9.6f} winrate={np.mean(np.array(wins) > 0):.0%}")
mt5.shutdown()

# ==========================================
# ПОРТФЕЛЬ
# ==========================================
M = pd.DataFrame(pair_windows)          # окна × пары
port = M.sum(axis=1)

print(f"\n{'='*52}")
print(f"  ПОРТФЕЛЬ ({len(M.columns)} пар):")
print(f"    mean  = {port.mean():.6f}")
print(f"    std   = {port.std():.6f}")
print(f"    score = {port.mean() / (port.std() + 1e-12):.2f}")
print(f"    winrate = {np.mean(port > 0):.0%}")
print(f"\n  Корреляции пар (по окнам):")
print(M.corr().round(2).to_string())
print(f"{'='*52}")

if port.mean() / (port.std() + 1e-12) >= 0.5 and np.mean(port > 0) >= 0.6:
    print("✅ ПОРТФЕЛЬ ЖИВОЙ: слабый edge агрегируется — система существует")
else:
    print("❌ Портфель мёртв: истинный edge ≈ 0 → идея в текущем виде исчерпана")

# финальное окно: портфельная эквити
fig, ax = plt.subplots(figsize=(15, 6))
for sym, c in final_curves.items():
    ax.plot(c, lw=1, alpha=0.6, label=sym)
port_final = np.sum(list(final_curves.values()), axis=0)
ax.plot(port_final, 'k', lw=3, label=f'ПОРТФЕЛЬ: {port_final[-1]:.5f}')
ax.axhline(0, color='gray', lw=0.8)
ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Последнее окно: пары и портфель (в процентах)')
plt.tight_layout(); plt.show()