import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# НАСТРОЙКИ
# ==========================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
TOTAL_BARS = 5000

BARS_TEST = 100          # чистый тест
BARS_VAL = 100           # валидация для отбора Optuna

N_TRIALS = 80            # итераций Optuna
N_CHAMPIONS = 10         # размер ансамбля

SIGMA_MIN, SIGMA_MAX = 1, 5
NB_MIN, NB_MAX = 3, 8
TARGET_SCALE = 10000

# ==========================================
# 1. ДАННЫЕ + ФИЧИ
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

# ==========================================
# 2. РАЗДЕЛЕНИЕ TRAIN / VAL / TEST
# ==========================================
T = len(df)
test_start = T - BARS_TEST
val_start = test_start - BARS_VAL

X_train = df[feature_cols].iloc[:val_start].values
X_val   = df[feature_cols].iloc[val_start:test_start].values
X_test  = df[feature_cols].iloc[test_start:].values
close_train, close_val, close_test = close[:val_start], close[val_start:test_start], close[test_start:]
print(f"✅ train={len(close_train)} | val={len(close_val)} | test={len(close_test)}")

# ==========================================
# 3. ПРАВИЛО (кросс + N + встречный)
# ==========================================
def user_rule(sig, close_t, nb):
    eq, pos, entry, held, trades = 0.0, 0, 0.0, 0, 0
    curve, prev_s = [], 0
    for t in range(len(sig)):
        p = close_t[t]; s = int(np.sign(sig[t]))
        cross = (s != 0) and (s != prev_s)
        if pos != 0:
            held += 1
            if held >= nb or (cross and s == -pos):
                eq += (p - entry) * pos; trades += 1; pos = 0
        if cross and pos == 0: pos, entry, held = s, p, 0
        prev_s = s; curve.append(eq)
    if pos != 0: eq += (close_t[-1] - entry) * pos; trades += 1
    return np.array(curve), eq, trades

# ==========================================
# 4. ОБУЧЕНИЕ + ПРЕДСКАЗАНИЕ (val и test)
# ==========================================
def fit_predict(sigma, n_est, lr, mc):
    sm = gaussian_filter1d(close_train, sigma=sigma, mode='reflect')
    dd = np.zeros_like(sm)
    dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_train)
    m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=lr, min_child_samples=mc,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)

    ptr = m.predict(Xtr_s) / TARGET_SCALE
    k = np.std(dd) / (np.std(ptr) + 1e-9)
    val_p  = m.predict(sc.transform(X_val))  / TARGET_SCALE * k
    test_p = m.predict(sc.transform(X_test)) / TARGET_SCALE * k
    return val_p, test_p

# ==========================================
# 5. OPTUNA: отбор на VAL
# ==========================================
def objective(trial):
    sigma = trial.suggest_int('sigma', SIGMA_MIN, SIGMA_MAX)
    nb    = trial.suggest_int('nb', NB_MIN, NB_MAX)
    n_est = trial.suggest_int('n_estimators', 100, 400, step=50)
    lr    = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
    mc    = trial.suggest_int('min_child_samples', 15, 60, step=5)
    try:
        val_p, _ = fit_predict(sigma, n_est, lr, mc)
        _, pnl, trd = user_rule(val_p, close_val, nb)
        return pnl if trd >= 3 else pnl - 0.005
    except Exception:
        return -1e9

print(f"\n🔍 Optuna: {N_TRIALS} trials на VAL...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS)

champs = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                key=lambda t: t.value, reverse=True)[:N_CHAMPIONS]

# ==========================================
# 6. АНСАМБЛЬ НА ЧИСТОМ ТЕСТЕ (ИСПРАВЛЕНО)
# ==========================================
print(f"\n🏆 Топ-{N_CHAMPIONS} чемпионов → TEST:")
print(f"{'#':>2} {'sigma':>5} {'nb':>3} {'n_est':>5} {'lr':>7} {'mc':>3} | {'val':>8} {'test':>8} {'сдел':>4}")
curves, test_pnls = [], []
for i, t in enumerate(champs):
    p = t.params
    _, test_p = fit_predict(p['sigma'], p['n_estimators'], p['learning_rate'], p['min_child_samples'])
    curve, pnl, trd = user_rule(test_p, close_test, p['nb'])
    curves.append(curve); test_pnls.append(pnl)
    print(f"{i+1:>2} {p['sigma']:>5} {p['nb']:>3} {p['n_estimators']:>5} "
          f"{p['learning_rate']:>7.3f} {p['min_child_samples']:>3} | {t.value:>8.4f} {pnl:>8.4f} {trd:>4}")

agg = np.sum(curves, axis=0)
test_pnls = np.array(test_pnls)

# идеал для справки (ИСПРАВЛЕНО: curve, pnl, trades)
sm_all = gaussian_filter1d(close, sigma=2, mode='reflect')
d_all = np.zeros_like(sm_all)
d_all[1:-1] = (sm_all[2:] - sm_all[:-2]) / 2.0
ideal_test = d_all[test_start:test_start + BARS_TEST]
ideal_curve, ideal_pnl, _ = user_rule(ideal_test, close_test, 4)

print(f"\n{'='*52}")
print(f"  АНСАМБЛЬ: PnL = {agg[-1]:.4f}")
print(f"  чемпионов в плюсе на тесте: {np.mean(test_pnls > 0):.0%}")
print(f"  средний PnL чемпиона:       {test_pnls.mean():.4f}")
print(f"  ИДЕАЛ (справка):            {ideal_pnl:.4f}")
print(f"{'='*52}")

# ==========================================
# 7. ГРАФИК (ИСПРАВЛЕНО)
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                         gridspec_kw={'height_ratios': [1.2, 1]})
x = np.arange(1, BARS_TEST + 1)
axes[0].plot(x, close_test, 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} H1 | ансамбль {N_CHAMPIONS} чемпионов Optuna')
axes[0].grid(alpha=0.3)

for c in curves:
    axes[1].plot(c, color='lightblue', lw=0.8, alpha=0.6)
axes[1].plot(agg, 'b', lw=2.5, label=f'АНСАМБЛЬ: {agg[-1]:.4f}')
axes[1].plot(ideal_curve, 'g', lw=1.5, label=f'ИДЕАЛ: {ideal_pnl:.4f}')
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_xlabel('Тестовый бар')
plt.tight_layout(); plt.show()