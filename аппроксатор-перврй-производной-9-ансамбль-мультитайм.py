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
BARS_TEST = 100
BARS_VAL = 100
N_TRIALS = 60
N_CHAMPIONS = 10
SIGMA_MIN, SIGMA_MAX = 1, 5
NB_MIN, NB_MAX = 3, 8
TARGET_SCALE = 10000

# ВЕСА ЦЕЛИ OPTUNA:  score = PROFIT_W * sharpe_win - DD_W * dd_norm
PROFIT_W = 2.0        # макс прибыль + макс Sharpe (это одно и то же в окне)
DD_W = 3.0            # мин просадка

TIMEFRAMES = [
    ('M15', mt5.TIMEFRAME_M15, 20000),
    ('M30', mt5.TIMEFRAME_M30, 10000),
    ('H1',  mt5.TIMEFRAME_H1,  5000),
    ('H4',  mt5.TIMEFRAME_H4,  2000),
]

# ==========================================
# ОБЩИЕ ФУНКЦИИ
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

def trade_stats(curve):
    """Sharpe окна (= нормированная прибыль) и нормированная просадка"""
    ret = np.diff(curve, prepend=0.0)
    std = ret.std()
    n = len(ret)
    vol_win = std * np.sqrt(n) + 1e-12
    sharpe_win = (ret.mean() / (std + 1e-12)) * np.sqrt(n)   # = PnL / vol_win
    peak = np.maximum.accumulate(curve)
    dd_norm = np.max(peak - curve) / vol_win
    return sharpe_win, dd_norm

# ==========================================
# АНСАМБЛЬ НА ОДНОМ ТФ
# ==========================================
def run_ensemble(tf_name, tf_const, total_bars):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf_const, 0, total_bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
    df = df[df.index.dayofweek < 5].copy()
    df = prepare(df)
    close = df['close'].values
    fcols = [c for c in df.columns if c not in
             ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]

    T = len(df)
    ts, vs = T - BARS_TEST, T - BARS_TEST - BARS_VAL
    X_tr, X_v, X_te = df[fcols].iloc[:vs].values, df[fcols].iloc[vs:ts].values, df[fcols].iloc[ts:].values
    c_tr, c_v, c_te = close[:vs], close[vs:ts], close[ts:]

    def fit_predict(sigma, n_est, lr, mc):
        sm = gaussian_filter1d(c_tr, sigma=sigma, mode='reflect')
        dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
        m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=lr, min_child_samples=mc,
                              n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(Xtr_s, dd * TARGET_SCALE)
        k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
        return (m.predict(sc.transform(X_v)) / TARGET_SCALE * k,
                m.predict(sc.transform(X_te)) / TARGET_SCALE * k)

    def objective(trial):
        try:
            sigma = trial.suggest_int('sigma', SIGMA_MIN, SIGMA_MAX)
            nb    = trial.suggest_int('nb', NB_MIN, NB_MAX)
            n_est = trial.suggest_int('n_estimators', 100, 400, step=50)
            lr    = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
            mc    = trial.suggest_int('min_child_samples', 15, 60, step=5)
            vp, _ = fit_predict(sigma, n_est, lr, mc)
            curve, pnl, trd = user_rule(vp, c_v, nb)
            if trd < 3 or pnl <= 0:
                return -1e9
            sharpe_win, dd_norm = trade_stats(curve)
            trial.set_user_attr('val_pnl', pnl)
            return PROFIT_W * sharpe_win - DD_W * dd_norm
        except Exception:
            return -1e9

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    champs = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                    key=lambda t: t.value, reverse=True)[:N_CHAMPIONS]

    # ГЕЙТ 1: только чемпионы с положительной val-прибылью
    champs = [t for t in champs if t.user_attrs.get('val_pnl', 0) > 0]
    if not champs:
        print(f"{tf_name}: ❌ нет чемпионов с положительной val → ТФ отключён")
        return None

    # ГЕЙТ 2: ТФ в портфель только если сумма val чемпионов > 0
    val_sum = sum(t.user_attrs['val_pnl'] for t in champs)
    if val_sum <= 0:
        print(f"{tf_name}: ❌ ансамбль на val отрицательный → ТФ отключён")
        return None

    curves, pnls = [], []
    for t in champs:
        p = t.params
        _, tp = fit_predict(p['sigma'], p['n_estimators'], p['learning_rate'], p['min_child_samples'])
        curve, pnl, _ = user_rule(tp, c_te, p['nb'])
        curves.append(curve); pnls.append(pnl)

    agg = np.sum(curves, axis=0)
    print(f"{tf_name}: ансамбль={agg[-1]:>8.4f} | в плюсе {np.mean(np.array(pnls) > 0):.0%} "
          f"| val_sum={val_sum:.4f}")
    return agg

# ==========================================
# ПОРТФЕЛЬ
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
results = {}
for name, tf, total in TIMEFRAMES:
    r = run_ensemble(name, tf, total)
    if r is not None:
        results[name] = r
mt5.shutdown()

if not results:
    print("❌ Ни один ТФ не прошёл гейты"); exit()

portfolio = np.sum(list(results.values()), axis=0)

print(f"\n{'='*44}")
for name, agg in results.items():
    print(f"  {name:>4}: {agg[-1]:>9.4f}")
print(f"  {'ПОРТФЕЛЬ':>4}: {portfolio[-1]:>9.4f}")
print(f"{'='*44}")

fig, ax = plt.subplots(figsize=(15, 6))
for name, agg in results.items():
    ax.plot(agg, lw=1.2, alpha=0.7, label=name)
ax.plot(portfolio, 'k', lw=3, label=f'ПОРТФЕЛЬ: {portfolio[-1]:.4f}')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title(f'{SYMBOL} | Sharpe+прибыль−просадка | гейты по val')
ax.legend(); ax.grid(alpha=0.3)
ax.set_xlabel('Тестовый бар (свой для каждого ТФ)')
plt.tight_layout(); plt.show()