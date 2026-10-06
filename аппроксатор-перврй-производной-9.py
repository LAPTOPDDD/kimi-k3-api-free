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
N_ESTIMATORS = 400
LEARNING_RATE = 0.03

CFG = {'seeds': 5, 'ema': 2, 'mc': 30}      # лучший выход из прошлого прогона
SHIFTS = [0, 1, 2]
NB = 4
WF_WINDOWS = 15
WF_SIZE = 100

# ==========================================
# ДАННЫЕ + ФИЧИ (без изменений)
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
print(f"✅ {len(df)} баров")

sm_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
d_all = np.zeros_like(sm_all)
d_all[1:-1] = (sm_all[2:] - sm_all[:-2]) / 2.0

# ==========================================
# ОБУЧЕНИЕ (с параметром shift)
# ==========================================
def train_and_predict(X_tr, X_te, y_tr, shift):
    sm = gaussian_filter1d(y_tr, sigma=SIGMA, mode='reflect')
    dd = np.zeros_like(sm)
    dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    yf, Xf = (dd[shift:], X_tr[:-shift]) if shift > 0 else (dd, X_tr)

    sc = StandardScaler()
    Xf_s, Xte_s = sc.fit_transform(Xf), sc.transform(X_te)

    tr_p, te_p = [], []
    for seed in range(CFG['seeds']):
        m = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                              min_child_samples=CFG['mc'], n_jobs=-1,
                              verbosity=-1, random_state=42 + seed)
        m.fit(Xf_s, yf * TARGET_SCALE)
        tr_p.append(m.predict(Xf_s) / TARGET_SCALE)
        te_p.append(m.predict(Xte_s) / TARGET_SCALE)
    tr_m, te_m = np.mean(tr_p, axis=0), np.mean(te_p, axis=0)
    pr = te_m * (np.std(yf) / (np.std(tr_m) + 1e-9))
    if CFG['ema'] > 0:
        pr = pd.Series(pr).ewm(span=CFG['ema'], adjust=False).mean().values
    return pr

# ==========================================
# ДВА ПРАВИЛА
# ==========================================
def rule_nb(sig, close_t, nb):
    """кросс → вход, выход по сроку N или встречному"""
    eq, pos, entry, held, tr = 0.0, 0, 0.0, 0, 0
    curve, prev_s = [], 0
    for t in range(len(sig)):
        p = close_t[t]; s = int(np.sign(sig[t]))
        cross = (s != 0) and (s != prev_s)
        if pos != 0:
            held += 1
            if held >= nb or (cross and s == -pos):
                eq += (p - entry) * pos; tr += 1; pos = 0
        if cross and pos == 0: pos, entry, held = s, p, 0
        prev_s = s; curve.append(eq)
    if pos != 0: eq += (close_t[-1] - entry) * pos
    return np.array(curve), eq

def rule_cross(sig, close_t):
    """кросс → вход, держим ДО ВСТРЕЧНОГО кросса"""
    eq, pos, entry, tr = 0.0, 0, 0.0, 0
    curve, prev_s = [], 0
    for t in range(len(sig)):
        p = close_t[t]; s = int(np.sign(sig[t]))
        cross = (s != 0) and (s != prev_s)
        if pos != 0 and cross and s == -pos:
            eq += (p - entry) * pos; tr += 1; pos = 0
        if pos == 0 and cross: pos, entry = s, p
        prev_s = s; curve.append(eq)
    if pos != 0: eq += (close_t[-1] - entry) * pos
    return np.array(curve), eq

# ==========================================
# WF-СЕТКА: SHIFT × ПРАВИЛО
# ==========================================
print(f"\n🔄 WF {WF_WINDOWS} окон: SHIFT × правило...")
print(f"{'shift':>5} {'правило':>7} | {'m_mean':>8} {'win':>5} {'score':>6} | {'i_mean':>8}")
best, best_key = -1e9, None
for shift in SHIFTS:
    acc = {r: [] for r in ['N4', 'CROSS']}
    acc_i = {r: [] for r in ['N4', 'CROSS']}
    for w in range(WF_WINDOWS):
        te = len(df) - w * WF_SIZE
        ts = te - WF_SIZE
        if ts < 1500: break
        pr = train_and_predict(df[feature_cols].iloc[:ts].values,
                               df[feature_cols].iloc[ts:te].values, close[:ts], shift)
        idl = d_all[ts + shift: ts + shift + WF_SIZE]
        if len(idl) < WF_SIZE: idl = np.pad(idl, (0, WF_SIZE - len(idl)))
        nn = WF_SIZE - 2
        cl = close[ts:ts + nn]
        _, pm1 = rule_nb(pr[:nn], cl, NB);      _, pi1 = rule_nb(idl[:nn], cl, NB)
        _, pm2 = rule_cross(pr[:nn], cl);       _, pi2 = rule_cross(idl[:nn], cl)
        acc['N4'].append(pm1);  acc_i['N4'].append(pi1)
        acc['CROSS'].append(pm2); acc_i['CROSS'].append(pi2)
    for r in ['N4', 'CROSS']:
        p = np.array(acc[r]); pi = np.array(acc_i[r])
        score = p.mean() / (p.std() + 1e-12)
        print(f"{shift:>5} {r:>7} | {p.mean():>8.5f} {np.mean(p > 0):>4.0%} {score:>6.2f} | {pi.mean():>8.5f}")
        if score > best: best, best_key = score, (shift, r)

print(f"\n🏆 ЛУЧШАЯ СВЯЗКА: SHIFT={best_key[0]}, правило={best_key[1]}")