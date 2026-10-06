import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# НАСТРОЙКИ
# ==========================================
SYMBOL = "EURUSD"; TIMEFRAME = mt5.TIMEFRAME_H1; TOTAL_BARS = 5000
SIGMA = 1; TARGET_SCALE = 10000; NB = 4
WF_WINDOWS = 20; WF_SIZE = 100; TRAIN_BARS = 3000; K_LAGS = 20

# ==========================================
# ДАННЫЕ + ФИЧИ
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
         ['open','high','low','close','tick_volume','spread','real_volume']]
X_all = df[fcols].values
T = len(df)

diff_all = np.diff(close, prepend=close[0])
def lag_matrix(x, lags):
    n = len(x); M = np.zeros((n, lags))
    for j in range(lags):
        idx = np.arange(n) - j
        M[:, j] = np.where(idx >= 0, x[np.maximum(idx, 0)], 0.0)
    return M
LAG = lag_matrix(diff_all, K_LAGS)

sm_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
ideal_all = np.zeros_like(sm_all)
ideal_all[1:-1] = (sm_all[2:] - sm_all[:-2]) / 2.0

def gauss_deriv(c):
    sm = gaussian_filter1d(c, sigma=SIGMA, mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    return dd

def user_rule(sig, close_t, nb):
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
    return np.array(curve), eq, tr

# ==========================================
# УЧЕНИКИ
# ==========================================
def sig_lgbm(tr_c, tr_X, ev_X):
    dd = gauss_deriv(tr_c)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(tr_X)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    return m.predict(sc.transform(ev_X)) / TARGET_SCALE * k

def sig_ridge(tr_c, tr_idx, ev_idx):
    dd = gauss_deriv(tr_c)
    r = Ridge(alpha=1.0).fit(LAG[tr_idx], dd)
    return r.predict(LAG[ev_idx])

def kalman_slopes(series, sl, ss, si, x0=None, P0=None):
    q = np.array([[sl*sl, 0], [0, ss*ss]]); r = si*si
    F = np.array([[1., 1.], [0., 1.]]); H = np.array([[1., 0.]])
    x = x0 if x0 is not None else np.array([series[0], 0.0])
    P = P0 if P0 is not None else np.eye(2) * 1e6
    out = np.zeros(len(series))
    for t in range(len(series)):
        x = F @ x; P = F @ P @ F.T + q
        y = series[t] - H @ x
        S = H @ P @ H.T + r
        K = P @ H.T / S
        x = x + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P
        out[t] = x[1]
    return out, x, P

def sig_kalman(tr_c, ev_c, teacher_tr):
    best, bc = -2, None
    for sl in [0.01, 0.05]:
        for ss in [0.001, 0.005, 0.01]:
            for si in [0.002, 0.005]:
                s_tr, x, P = kalman_slopes(tr_c, sl, ss, si)
                c = np.corrcoef(s_tr[10:], teacher_tr[10:])[0, 1]
                if c > best: best, bc = c, (sl, ss, si, x, P)
    sl, ss, si, x, P = bc
    s_ev, _, _ = kalman_slopes(ev_c, sl, ss, si, x0=x, P0=P)
    return s_ev

def sig_mom(ev_idx):
    return (close[ev_idx] - close[ev_idx - 3]) / 3.0

# ==========================================
# WF-ЛИДЕРБОРД (с защитой от малых окон)
# ==========================================
models = ['LGBM', 'RIDGE', 'KALMAN', 'MOM3', 'IDEAL']
wf = {m: [] for m in models}
final = {}
MIN_TRAIN_BARS = max(K_LAGS + 100, 500)  # минимум для обучения
actual_windows = 0

for w in range(WF_WINDOWS):
    te_end = T - w * WF_SIZE
    te_start = te_end - WF_SIZE
    tr_start = max(0, te_start - TRAIN_BARS)
    
    # защита: выходим, если train слишком маленький
    if te_start - tr_start < MIN_TRAIN_BARS:
        print(f"⚠️  окно {w+1}: train слишком мал ({te_start - tr_start} баров) → стоп")
        break
    if te_start < 0:
        break
    
    tr_c = close[tr_start:te_start]
    tr_X = X_all[tr_start:te_start]
    ev_c = close[te_start:te_end]
    ev_X = X_all[te_start:te_end]
    tr_idx = np.arange(tr_start, te_start)
    ev_idx = np.arange(te_start, te_end)
    teacher_tr = gauss_deriv(tr_c)
    n = WF_SIZE - 2
    
    # защита от пустых данных
    if len(tr_c) < MIN_TRAIN_BARS or len(ev_c) < WF_SIZE:
        print(f"⚠️  окно {w+1}: недостаточно данных → стоп")
        break
    
    try:
        sigs = {
            'LGBM':   sig_lgbm(tr_c, tr_X, ev_X),
            'RIDGE':  sig_ridge(tr_c, tr_idx, ev_idx),
            'KALMAN': sig_kalman(tr_c, ev_c, teacher_tr),
            'MOM3':   sig_mom(ev_idx),
            'IDEAL':  ideal_all[te_start:te_end]
        }
    except Exception as e:
        print(f"⚠️  окно {w+1}: ошибка {e} → стоп")
        break
    
    for m, s in sigs.items():
        _, pnl, _ = user_rule(s[:n], ev_c[:n], NB)
        wf[m].append(pnl)
    
    if actual_windows == 0:
        final = sigs
        f_cl, f_idl = ev_c, ideal_all[te_start:te_end]
    actual_windows += 1

if actual_windows == 0:
    print("❌ Нет ни одного валидного окна — увеличь TOTAL_BARS или уменьши WF_WINDOWS")
    exit()

# реальные метрики
actual_f = np.zeros(WF_SIZE)
for i in range(WF_SIZE - 2):
    actual_f[i] = (close[T - actual_windows * WF_SIZE + i + 2] - 
                   close[T - actual_windows * WF_SIZE + i]) / 2.0
n = WF_SIZE - 2

print(f"\n{'='*66}")
print(f"  🎺 ДЕМБЕЛЬСКИЙ ЛИДЕРБОРД | {SYMBOL} H1 | учитель: Гаусс σ={SIGMA}")
print(f"  (на {actual_windows} валидных WF-окнах)")
print(f"{'='*66}")
print(f"  {'модель':>7} | {'WF mean':>8} {'winrate':>7} | {'corr идеал':>10} {'corr реальн':>11}")
for m in models:
    a = np.array(wf[m])
    ci = np.corrcoef(final[m][:n], f_idl[:n])[0, 1] if len(a) > 0 else 0.0
    ca = np.corrcoef(final[m][:n], actual_f[:n])[0, 1] if len(a) > 0 else 0.0
    print(f"  {m:>7} | {a.mean():>8.5f} {np.mean(a > 0):>6.0%} | {ci:>10.3f} {ca:>11.3f}")
print(f"{'='*66}")# ==========================================
# ФИНАЛЬНАЯ СЦЕНА
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                         gridspec_kw={'height_ratios': [1.2, 1.4]})
x = np.arange(1, WF_SIZE + 1)
axes[0].plot(x, f_cl, 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} | цена на последних {WF_SIZE} барах')
axes[0].grid(alpha=0.3)

axes[1].plot(x[:n], f_idl[:n], 'g', lw=3, alpha=0.9, label='УЧИТЕЛЬ (неказуальный Гаусс)')
axes[1].plot(x[:n], final['LGBM'][:n], 'b', lw=1.5, label='LightGBM (25 фич)')
axes[1].plot(x[:n], final['RIDGE'][:n], color='orange', lw=1.5, label='Ridge-Винер (лаги цен)')
axes[1].plot(x[:n], final['KALMAN'][:n], 'r', lw=1.5, label='Калман (local linear trend)')
axes[1].plot(x[:n], final['MOM3'][:n], 'm', ls='--', lw=1.2, alpha=0.7, label='MOM3 (наивный)')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].set_title('🎺 Дембельский аккорд: все ученики и Учитель')
axes[1].legend(loc='upper left', fontsize=9); axes[1].grid(alpha=0.3)
axes[1].set_xlabel('Тестовый бар')
plt.tight_layout()
plt.savefig('dembel_accord.png', dpi=150)
plt.show()