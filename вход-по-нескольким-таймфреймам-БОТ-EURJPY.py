import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time as _time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# НАСТРОЙКИ
# ==========================================
SYMBOL = "EURJPY"
TFS = [("M15", mt5.TIMEFRAME_M15, 20000, 1, 8000),
       ("M30", mt5.TIMEFRAME_M30, 12000, 1, 6000),
       ("H1",  mt5.TIMEFRAME_H1,   6000, 1, 4000),
       ("H4",  mt5.TIMEFRAME_H4,   3000, 2, 1500)]
TARGET_SCALE = 10000
TEST_M15 = 400
WF_WINDOWS = 4
NB_LIST = [3, 5, 10]

# --- БОТ ---
LOT = 0.01
MAGIC = 777002

# ==========================================
# ДАННЫЕ + ФИЧИ
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

if not mt5.initialize(): print("❌ MT5"); exit()
data = {}
for name, tf, total, sig, cap in TFS:
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, total)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
    df = df[df.index.dayofweek < 5].copy()
    data[name] = prepare(df)
mt5.shutdown()
print("✅ Загружено:", {k: len(v) for k, v in data.items()})

fcols = [c for c in data['M15'].columns if c not in
         ['open','high','low','close','tick_volume','spread','real_volume']]

# ==========================================
# МОДЕЛЬ + ВЫРАВНИВАНИЕ (ТВОЯ ВЕРСИЯ)
# ==========================================
def tf_predictions(name, test_start_time):
    df = data[name]
    sig = dict((n, s) for n, _, _, s, _ in TFS)[name]
    cap = dict((n, c) for n, _, _, _, c in TFS)[name]
    times = df.index.astype('int64').values
    tr_idx = np.where(df.index < test_start_time)[0]
    tr_idx = tr_idx[-cap:] if len(tr_idx) > cap else tr_idx
    X_tr = df[fcols].values[tr_idx]; c_tr = df['close'].values[tr_idx]
    sm = gaussian_filter1d(c_tr, sigma=sig, mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    vals = m.predict(sc.transform(df[fcols].values)) / TARGET_SCALE * k
    return times, vals

def align(tf_times, tf_vals, m15_times):
    idx = np.searchsorted(tf_times, m15_times, side='right') - 1
    return tf_vals[np.clip(idx, 0, len(tf_vals) - 1)]

# ==========================================
# ПРАВИЛО С КОНФЛЮЭНЦЕЙ
# ==========================================
def confluence_rule(act_long, act_short, close_t, nb):
    eq, pos, entry, held, tr = 0.0, 0, 0.0, 0, 0
    curve = []
    for t in range(len(close_t)):
        p = close_t[t]
        if pos == 1:
            held += 1
            if held >= nb or act_short[t] or not act_long[t]:
                eq += (p - entry); tr += 1; pos = 0
        elif pos == -1:
            held += 1
            if held >= nb or act_long[t] or not act_short[t]:
                eq += (entry - p); tr += 1; pos = 0
        if pos == 0:
            if act_long[t]: pos, entry, held = 1, p, 0
            elif act_short[t]: pos, entry, held = -1, p, 0
        curve.append(eq)
    if pos == 1: eq += (close_t[-1] - entry)
    if pos == -1: eq += (entry - close_t[-1])
    return np.array(curve), eq, tr

# ==========================================
# WF-ТЕСТ
# ==========================================
m15 = data['M15']
m15_times_all = m15.index.astype('int64').values
res = {('ALL4', nb): [] for nb in NB_LIST}
res.update({('M15only', nb): [] for nb in NB_LIST})
final_curves = {}

for w in range(WF_WINDOWS):
    te_end = len(m15) - w * TEST_M15
    te_start = te_end - TEST_M15
    test_start_time = m15.index[te_start]
    aligned = {}
    for name, _, _, _, _ in TFS:
        t_times, t_vals = tf_predictions(name, test_start_time)
        aligned[name] = align(t_times, t_vals, m15_times_all[te_start:te_end])
    close_t = m15['close'].values[te_start:te_end]
    act_long = np.all([aligned[n] > 0 for n in aligned], axis=0)
    act_short = np.all([aligned[n] < 0 for n in aligned], axis=0)
    act_long_1 = aligned['M15'] > 0
    act_short_1 = aligned['M15'] < 0
    for nb in NB_LIST:
        _, p4, t4 = confluence_rule(act_long, act_short, close_t, nb)
        _, p1, t1 = confluence_rule(act_long_1, act_short_1, close_t, nb)
        res[('ALL4', nb)].append(p4); res[('M15only', nb)].append(p1)
        if w == 0 and nb == 5:
            c4, _, _ = confluence_rule(act_long, act_short, close_t, nb)
            c1, _, _ = confluence_rule(act_long_1, act_short_1, close_t, nb)
            final_curves = {'ALL4': c4, 'M15only': c1}
    print(f"  окно {w+1}/{WF_WINDOWS} ✓ | ALL4 активен {np.mean(act_long | act_short):.0%}")

print(f"\n{'='*56}")
print(f"  {'вариант':>9} | {'NB':>2} | {'mean':>8} {'winrate':>7}")
for nb in NB_LIST:
    for var in ['ALL4', 'M15only']:
        a = np.array(res[(var, nb)])
        print(f"  {var:>9} | {nb:>2} | {a.mean():>8.5f} {np.mean(a > 0):>6.0%}")
print(f"{'='*56}")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(final_curves['ALL4'], 'b', lw=2, label='КОНФЛЮЭНЦИЯ: все 4 ТФ > 0')
ax.plot(final_curves['M15only'], 'r--', lw=1.2, alpha=0.7, label='M15 одиночка')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title(f'{SYMBOL} | эквити бота (твоя версия выравнивания)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# ==================================================
# ==================== БОТ =========================
# ==================================================
ans = input("\n🤖 Запустить БОТА? (Y/N): ").strip().upper()
if ans != 'Y':
    print("Завершение."); exit()

if not mt5.initialize(): print("❌ MT5 init"); exit()
ai = mt5.account_info()
if ai is None: print("❌ нет аккаунта"); exit()
if ai.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO:
    print(f"✅ ДЕМО #{ai.login}, баланс {ai.balance:.2f} {ai.currency}")
else:
    print(f"⚠️ РЕАЛЬНЫЙ счёт #{ai.login}!")
    if input("   Продолжить? Введите REAL: ").strip() != "REAL":
        mt5.shutdown(); print("Завершение."); exit()

si = mt5.symbol_info(SYMBOL)
lot = max(si.volume_min, LOT)
MODELS = {}
TRAIN_DATE = None

def refresh_data():
    out = {}
    for name, tf, total, sig, cap in TFS:
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, total)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
        df = df[df.index.dayofweek < 5].copy()
        out[name] = prepare(df)
    return out

def train_all(dfs, cutoff):
    mods = {}
    for name, _, _, sig, cap in TFS:
        df = dfs[name]
        tr_idx = np.where(df.index < cutoff)[0]
        tr_idx = tr_idx[-cap:] if len(tr_idx) > cap else tr_idx
        X_tr, c_tr = df[fcols].values[tr_idx], df['close'].values[tr_idx]
        sm = gaussian_filter1d(c_tr, sigma=sig, mode='reflect')
        dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, min_child_samples=30,
                              n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(Xtr_s, dd * TARGET_SCALE)
        k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
        mods[name] = (m, sc, k)
    return mods

def live_signal(dfs):
    m15d = dfs['M15']
    m15t = m15d.index.astype('int64').values
    sigs = {}
    for name, _, _, _, _ in TFS:
        m, sc, k = MODELS[name]
        df = dfs[name]
        vals = m.predict(sc.transform(df[fcols].values)) / TARGET_SCALE * k
        sigs[name] = align(df.index.astype('int64').values, vals, m15t)[-1]
    al4 = all(v > 0 for v in sigs.values())
    as4 = all(v < 0 for v in sigs.values())
    return al4, as4, sigs

def magic_positions():
    ps = mt5.positions_get(symbol=SYMBOL) or []
    return [p for p in ps if p.magic == MAGIC]

def close_all():
    for p in magic_positions():
        side = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        _send(side, p.volume, p.ticket)

def _send(side, vol, ticket=None):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if side == mt5.ORDER_TYPE_BUY else tick.bid
    for fill in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": vol,
               "type": side, "price": price, "deviation": 20, "magic": MAGIC,
               "comment": "conf_bot", "type_time": mt5.ORDER_TIME_GTC, "type_filling": fill}
        if ticket: req["position"] = ticket
        r = mt5.order_send(req)
        if r and r.retcode == mt5.TRADE_RETCODE_DONE: return r
    print("❌ order_send fail:", r)
    return None

def set_target(target):
    ps = magic_positions()
    cur = sum(p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume for p in ps)
    cur_sign = 1 if cur > 0 else (-1 if cur < 0 else 0)
    if cur_sign == target: return "hold"
    if cur_sign != 0: close_all()
    if target != 0:
        _send(mt5.ORDER_TYPE_BUY if target > 0 else mt5.ORDER_TYPE_SELL, lot)
    return f"→ {target}"

print("🤖 Бот запущен. Ctrl+C — остановка.")
last_bar = None
try:
    while True:
        _time.sleep(2)
        r = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 1)
        if r is None: continue
        bt = int(r[0]['time'])
        if bt == last_bar: continue
        last_bar = bt
        now = pd.to_datetime(bt, unit='s')
        try:
            dfs = refresh_data()
            if TRAIN_DATE != now.date():
                MODELS = train_all(dfs, now)
                TRAIN_DATE = now.date()
                print(f"🌅 {now} — модели переобучены")
            al4, as4, sigs = live_signal(dfs)
            target = 1 if al4 else (-1 if as4 else 0)
            act = set_target(target)
            print(f"{now} | " + " ".join(f"{n}:{s:+.4f}" for n, s in sigs.items())
                  + f" | target={target} | {act}")
        except Exception as e:
            print("⚠️ ошибка цикла:", e)
except KeyboardInterrupt:
    print("\n🛑 Бот остановлен. Позиции остаются.")
    mt5.shutdown()