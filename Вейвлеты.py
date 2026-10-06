import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# НАСТРОЙКИ
# ==========================================
OUT_DIR = "scan_charts_wavelet"
os.makedirs(OUT_DIR, exist_ok=True)

WAVELET_LEVEL = 3        # масштаб качания: период ~8-16 баров

TFS = [("M15", mt5.TIMEFRAME_M15, 20000, 8000),
       ("M30", mt5.TIMEFRAME_M30, 12000, 6000),
       ("H1",  mt5.TIMEFRAME_H1,   6000, 4000),
       ("H4",  mt5.TIMEFRAME_H4,   3000, 1500)]
TF_MINUTES = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}
TARGET_SCALE = 10000
TEST_M15 = 400
WF_WINDOWS = 20
NB = 5

# ==========================================
# À TROUS ВЕЙВЛЕТ (B3-сплайн, без библиотек)
# ==========================================
def atrous_detail(x, level):
    """деталь уровня level à trous-разложения (колебание масштаба 2^level)"""
    h = np.array([1., 4., 6., 4., 1.]) / 16.
    c = x
    detail = None
    for j in range(1, level + 1):
        gap = 2 ** (j - 1)
        n = len(c)
        pad = 4 * gap
        cp = np.pad(c, pad, mode='reflect')
        smooth = np.zeros(n)
        for k, w in zip(range(-2, 3), h):
            s = pad + k * gap
            smooth += w * cp[s:s + n]
        detail = c - smooth
        c = smooth
    return detail

# ==========================================
# ФИЧИ
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

def load_symbol(sym):
    data = {}
    for name, tf, total, cap in TFS:
        rates = mt5.copy_rates_from_pos(sym, tf, 0, total)
        if rates is None or len(rates) < 1000: return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
        df = df[df.index.dayofweek < 5].copy()
        df = prepare(df)
        if len(df) < 1000: return None
        data[name] = df
    return data

# ==========================================
# УЧЕНИК: учитель = вейвлет-деталь (лог-цен), фичи каузальные
# ==========================================
def tf_predictions(data, fcols, name, test_start_time):
    df = data[name]
    cap = dict((n, c) for n, _, _, c in TFS)[name]
    close_times = df.index + pd.Timedelta(minutes=TF_MINUTES[name])
    tr_idx = np.where(close_times <= test_start_time)[0]
    tr_idx = tr_idx[-cap:] if len(tr_idx) > cap else tr_idx
    X_tr = df[fcols].values[tr_idx]
    c_tr = np.log(df['close'].values[tr_idx])

    dd = atrous_detail(c_tr, WAVELET_LEVEL)          # ← ВЕЙВЛЕТ-УЧИТЕЛЬ
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

def confluence_rule(act_long, act_short, close_t, nb):
    eq, pos, entry, tr = 0.0, 0, 0.0, 0
    curve = []
    for t in range(len(close_t)):
        p = close_t[t]
        if pos == 1 and (act_short[t] or not act_long[t]):
            eq += (p - entry); tr += 1; pos = 0
        elif pos == -1 and (act_long[t] or not act_short[t]):
            eq += (entry - p); tr += 1; pos = 0
        if pos == 0:
            if act_long[t]: pos, entry = 1, p
            elif act_short[t]: pos, entry = -1, p
        curve.append(eq)
    if pos == 1: eq += (close_t[-1] - entry)
    if pos == -1: eq += (entry - close_t[-1])
    return np.array(curve), eq, tr

# ==========================================
# ОДНА ПАРА
# ==========================================
def run_symbol(sym):
    data = load_symbol(sym)
    if data is None: return None
    fcols = [c for c in data['M15'].columns if c not in
             ['open','high','low','close','tick_volume','spread','real_volume']]
    m15 = data['M15']; m15t = m15.index.astype('int64').values

    a4_list, a1_list, curves = [], [], None
    for w in range(WF_WINDOWS):
        te_end = len(m15) - w * TEST_M15
        te_start = te_end - TEST_M15
        if te_start < 1600: break
        aligned = {}
        for name, _, _, _ in TFS:
            t_times, t_vals = tf_predictions(data, fcols, name, m15.index[te_start])
            aligned[name] = align(name, t_times, t_vals, m15t[te_start:te_end])
        close_t = m15['close'].values[te_start:te_end]
        actL = np.all([aligned[n] > 0 for n in aligned], axis=0)
        actS = np.all([aligned[n] < 0 for n in aligned], axis=0)
        actL1, actS1 = aligned['M15'] > 0, aligned['M15'] < 0
        _, p4, _ = confluence_rule(actL, actS, close_t, NB)
        _, p1, _ = confluence_rule(actL1, actS1, close_t, NB)
        a4_list.append(p4); a1_list.append(p1)
        if w == 0:
            c4, _, _ = confluence_rule(actL, actS, close_t, NB)
            c1, _, _ = confluence_rule(actL1, actS1, close_t, NB)
            curves = (c4, c1)

    if not a4_list or curves is None: return None
    a4, a1 = np.array(a4_list), np.array(a1_list)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(curves[0], 'b', lw=2, label=f'ALL4 wavelet: mean={a4.mean():.5f}')
    ax.plot(curves[1], 'r--', lw=1.2, alpha=0.7, label=f'M15 wavelet: mean={a1.mean():.5f}')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title(f'{sym} | ВЕЙВЛЕТ-учитель | winrate ALL4={np.mean(a4 > 0):.0%}')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{sym}.png", dpi=100)
    plt.close(fig)
    return dict(sym=sym, a4=a4.mean(), wr4=np.mean(a4 > 0), a1=a1.mean(), wr1=np.mean(a1 > 0))

# ==========================================
# МАССОВЫЙ ПРОГОН
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
syms = sorted(s.name for s in mt5.symbols_get() if s.visible)
print(f"📋 Пар: {len(syms)} | учитель = à trous вейвлет, уровень {WAVELET_LEVEL} | {WF_WINDOWS} окон")

rows = []
for i, sym in enumerate(syms, 1):
    print(f"[{i}/{len(syms)}] {sym} ...", end=" ", flush=True)
    try:
        r = run_symbol(sym)
    except Exception:
        print("skip (ошибка)"); continue
    if r is None:
        print("skip (мало данных)"); continue
    rows.append(r)
    print(f"ALL4 mean={r['a4']:>9.5f} wr={r['wr4']:.0%}")
mt5.shutdown()

rows.sort(key=lambda r: r['a4'], reverse=True)
df_res = pd.DataFrame(rows)[['sym', 'a4', 'wr4', 'a1', 'wr1']]
df_res.columns = ['пара', 'WAVELET_mean', 'WAVELET_winrate', 'M15_mean', 'M15_winrate']
df_res.to_csv(f"{OUT_DIR}/leaderboard_wavelet.csv", index=False, encoding='utf-8-sig')

alive = df_res[(df_res['WAVELET_mean'] > 0) & (df_res['WAVELET_winrate'] >= 0.6)]
print(f"\n{'='*64}")
print("  ТОП-15 (ВЕЙВЛЕТ, 20 окон):")
print(df_res.head(15).to_string(index=False))
print(f"{'='*64}")
print(f"  Живых пар (mean>0 и winrate≥60%): {len(alive)}")
print(alive.to_string(index=False) if len(alive) else "  — пусто")
print(f"\n✅ {OUT_DIR}/ + leaderboard_wavelet.csv")