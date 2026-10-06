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
N_TRIALS = 40
N_CHAMPIONS = 10
SIGMA_MIN, SIGMA_MAX = 1, 5
NB_MIN, NB_MAX = 3, 8
TARGET_SCALE = 10000
PROFIT_W, DD_W = 2.0, 3.0
TRAIN_TAIL = 300          # хвост train для графика

TIMEFRAMES = [
    ('M15', mt5.TIMEFRAME_M15, 20000),
    ('M30', mt5.TIMEFRAME_M30, 10000),
    ('H1',  mt5.TIMEFRAME_H1,  5000),
    ('H4',  mt5.TIMEFRAME_H4,  2000),
]

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
    ret = np.diff(curve, prepend=0.0)
    std = ret.std(); n = len(ret)
    vol_win = std * np.sqrt(n) + 1e-12
    sharpe_win = (ret.mean() / (std + 1e-12)) * np.sqrt(n)
    peak = np.maximum.accumulate(curve)
    return sharpe_win, np.max(peak - curve) / vol_win

# ==========================================
# ВОРОНКА + КРИВЫЕ ЛУЧШЕГО ЧЕМПИОНА
# ==========================================
def run_funnel(tf_name, tf_const, total_bars):
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

    def fit_predict3(sigma, n_est, lr, mc):
        sm = gaussian_filter1d(c_tr, sigma=sigma, mode='reflect')
        dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
        m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=lr, min_child_samples=mc,
                              n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(Xtr_s, dd * TARGET_SCALE)
        k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
        return {'tr': m.predict(Xtr_s) / TARGET_SCALE * k,
                'v':  m.predict(sc.transform(X_v)) / TARGET_SCALE * k,
                'te': m.predict(sc.transform(X_te)) / TARGET_SCALE * k}

    def objective(trial):
        try:
            sigma = trial.suggest_int('sigma', SIGMA_MIN, SIGMA_MAX)
            nb    = trial.suggest_int('nb', NB_MIN, NB_MAX)
            n_est = trial.suggest_int('n_estimators', 100, 400, step=50)
            lr    = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
            mc    = trial.suggest_int('min_child_samples', 15, 60, step=5)
            p = fit_predict3(sigma, n_est, lr, mc)
            _, tr_pnl, _ = user_rule(p['tr'], c_tr, nb)
            v_curve, v_pnl, v_trd = user_rule(p['v'], c_v, nb)
            _, te_pnl, _ = user_rule(p['te'], c_te, nb)
            trial.set_user_attr('train_pnl', tr_pnl)
            trial.set_user_attr('val_pnl', v_pnl)
            trial.set_user_attr('test_pnl', te_pnl)
            if v_trd < 3: return -1e9
            sh, ddn = trade_stats(v_curve)
            return PROFIT_W * sh - DD_W * ddn
        except Exception:
            return -1e9

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    champs = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                    key=lambda t: t.value, reverse=True)[:N_CHAMPIONS]

    tr_a = np.array([t.user_attrs['train_pnl'] for t in champs])
    va_a = np.array([t.user_attrs['val_pnl']   for t in champs])
    te_a = np.array([t.user_attrs['test_pnl']  for t in champs])
    print(f"\n{'='*58}\n {tf_name} | воронка топ-{len(champs)}\n{'='*58}")
    print(f"  TRAIN: mean={tr_a.mean():>9.4f} | + {np.mean(tr_a > 0):>4.0%}")
    print(f"  VAL:   mean={va_a.mean():>9.4f} | + {np.mean(va_a > 0):>4.0%}")
    print(f"  TEST:  mean={te_a.mean():>9.4f} | + {np.mean(te_a > 0):>4.0%}")

    # кривые лучшего чемпиона: train(хвост) → val → test
    b = champs[0].params
    p = fit_predict3(b['sigma'], b['n_estimators'], b['learning_rate'], b['min_child_samples'])
    tr_c, _, _ = user_rule(p['tr'], c_tr, b['nb'])
    v_c,  _, _ = user_rule(p['v'],  c_v,  b['nb'])
    te_c, _, _ = user_rule(p['te'], c_te, b['nb'])
    return {'tr': tr_c, 'v': v_c, 'te': te_c}

# ==========================================
# ЗАПУСК + ГРАФИКИ
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
results = {}
for name, tf, total in TIMEFRAMES:
    results[name] = run_funnel(name, tf, total)
mt5.shutdown()

fig, axes = plt.subplots(len(results), 1, figsize=(15, 4.2 * len(results)))
for ax, (name, r) in zip(axes, results.items()):
    tr_t = r['tr'][-TRAIN_TAIL:]
    off1 = tr_t[-1]
    off2 = off1 + r['v'][-1]
    ax.plot(np.arange(-len(tr_t), 0), tr_t, 'b', lw=1, label='TRAIN (хвост 300)')
    ax.plot(np.arange(0, len(r['v'])), r['v'] + off1, 'g', lw=1.8, label='VAL (отбор)')
    ax.plot(np.arange(len(r['v']), len(r['v']) + len(r['te'])), r['te'] + off2, 'r', lw=1.8, label='TEST (чистый)')
    ax.axvline(0, color='gray', ls='--', alpha=0.7)
    ax.axvline(len(r['v']), color='gray', ls='--', alpha=0.7)
    ax.set_title(f'{name} | путь лучшего чемпиона: train → val → test')
    ax.legend(loc='upper left'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()