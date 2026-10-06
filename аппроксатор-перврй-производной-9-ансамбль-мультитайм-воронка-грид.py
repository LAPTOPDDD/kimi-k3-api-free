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
BARS_TEST = 100
K_VAL = 5                    # честность: 5 val-окон вместо одного
VAL_SIZE = 100
TRAIN_CAP = 8000             # предел train для скорости

SIGMA_GRID = [1, 2, 3, 4, 5]
NB_GRID = [3, 4, 5, 6, 7, 8]
N_EST_GRID = [200, 400]      # 2-я стадия greedy
LR_GRID = [0.03, 0.08]
MC_GRID = [20, 40]

N_CHAMPIONS = 10
TARGET_SCALE = 10000
DD_W = 3.0
MIN_FRAC_POS = 0.6           # чемпион обязан быть в плюсе на ≥60% окон

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

def window_score(curve, pnl, trd):
    if trd < 3 or pnl <= 0: return -5.0
    ret = np.diff(curve, prepend=0.0)
    std = ret.std(); n = len(ret)
    vol_win = std * np.sqrt(n) + 1e-12
    sh = (ret.mean() / (std + 1e-12)) * np.sqrt(n)
    dd = np.max(np.maximum.accumulate(curve) - curve) / vol_win
    return sh - DD_W * dd

# ==========================================
# GREEDY НА ОДНОМ ТФ
# ==========================================
def run_greedy(tf_name, tf_const, total_bars):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf_const, 0, total_bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
    df = df[df.index.dayofweek < 5].copy()
    df = prepare(df)
    close = df['close'].values
    fcols = [c for c in df.columns if c not in
             ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]

    T = len(df)
    ts = T - BARS_TEST
    val_zone = K_VAL * VAL_SIZE
    vs = ts - val_zone
    tr_start = max(0, vs - TRAIN_CAP)

    X_all = df[fcols].values
    X_tr = X_all[tr_start:vs]
    c_tr = close[tr_start:vs]
    X_te, c_te = X_all[ts:], close[ts:]
    val_wins = [(X_all[vs + i*VAL_SIZE: vs + (i+1)*VAL_SIZE],
                 close[vs + i*VAL_SIZE: vs + (i+1)*VAL_SIZE]) for i in range(K_VAL)]

    def eval_params(sigma, nb, n_est, lr, mc):
        sm = gaussian_filter1d(c_tr, sigma=sigma, mode='reflect')
        dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
        m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=lr, min_child_samples=mc,
                              n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(Xtr_s, dd * TARGET_SCALE)
        k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
        scores, pnls = [], []
        for X_w, c_w in val_wins:
            pred = m.predict(sc.transform(X_w)) / TARGET_SCALE * k
            curve, pnl, trd = user_rule(pred, c_w, nb)
            scores.append(window_score(curve, pnl, trd)); pnls.append(pnl)
        robust = float(np.mean(scores)); frac = float(np.mean(np.array(pnls) > 0))
        # тест (только отчёт, не отбор!)
        te_pred = m.predict(sc.transform(X_te)) / TARGET_SCALE * k
        _, te_pnl, _ = user_rule(te_pred, c_te, nb)
        return robust, frac, te_pnl, m, sc, k

    # --- СТАДИЯ 1: сетка sigma × nb (ML-параметры фиксированы) ---
    print(f"\n{'='*62}\n {tf_name}: СТАДИЯ 1 — ландшафт sigma×nb на {K_VAL} val-окнах\n{'='*62}")
    landscape = {}
    for sigma in SIGMA_GRID:
        row = []
        for nb in NB_GRID:
            robust, frac, _, _, _, _ = eval_params(sigma, nb, 300, 0.05, 30)
            landscape[(sigma, nb)] = (robust, frac)
            row.append(f"{robust:>6.2f}")
        print(f"  σ={sigma}: " + " ".join(row))
    print("        " + " ".join(f"nb={b:<3}" for b in NB_GRID))

    # --- СТАДИЯ 2: greedy-доточка ML-параметров для топ-5 ---
    top5 = sorted(landscape.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
    variants = []
    for (sigma, nb), _ in top5:
        for n_est in N_EST_GRID:
            for lr in LR_GRID:
                for mc in MC_GRID:
                    robust, frac, te_pnl, m, sc, k = eval_params(sigma, nb, n_est, lr, mc)
                    variants.append(dict(sigma=sigma, nb=nb, n_est=n_est, lr=lr, mc=mc,
                                         robust=robust, frac=frac, te_pnl=te_pnl))

    # --- ЧЕСТНЫЙ ОТБОР: robust>0 И frac>=0.6 ---
    champs = [v for v in variants if v['robust'] > 0 and v['frac'] >= MIN_FRAC_POS]
    champs.sort(key=lambda v: v['robust'], reverse=True)
    champs = champs[:N_CHAMPIONS]

    if not champs:
        print(f" {tf_name}: ❌ нет робастных чемпионов → ТФ отключён")
        return None

    print(f"\n {tf_name}: чемпионов={len(champs)}")
    print(f" {'σ':>2} {'nb':>3} {'n_est':>5} {'lr':>5} {'mc':>3} | {'robust':>7} {'frac':>5} | {'TEST':>8}")
    curves = []
    for v in champs:
        # пересобираем тестовую кривую
        sm = gaussian_filter1d(c_tr, sigma=v['sigma'], mode='reflect')
        dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
        m = lgb.LGBMRegressor(n_estimators=v['n_est'], learning_rate=v['lr'],
                              min_child_samples=v['mc'], n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(Xtr_s, dd * TARGET_SCALE)
        k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
        te_pred = m.predict(sc.transform(X_te)) / TARGET_SCALE * k
        curve, te_pnl, _ = user_rule(te_pred, c_te, v['nb'])
        curves.append(curve)
        print(f" {v['sigma']:>2} {v['nb']:>3} {v['n_est']:>5} {v['lr']:>5.2f} {v['mc']:>3} | "
              f"{v['robust']:>7.2f} {v['frac']:>4.0%} | {te_pnl:>8.4f}")

    agg = np.sum(curves, axis=0)
    te_mean = np.mean([c[-1] for c in curves])
    print(f" {tf_name}: АНСАМБЛЬ test={agg[-1]:.4f} | средний чемпион={te_mean:.4f}")
    return agg

# ==========================================
# ПОРТФЕЛЬ
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
results = {}
for name, tf, total in TIMEFRAMES:
    r = run_greedy(name, tf, total)
    if r is not None: results[name] = r
mt5.shutdown()

if results:
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
    ax.set_title(f'{SYMBOL} | GREEDY + {K_VAL}-оконный val | честный отбор')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
else:
    print("❌ Ни один ТФ не прошёл честный отбор")