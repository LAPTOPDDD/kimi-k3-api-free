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
SYMBOL = "GBPJPY"
TFS = [("M5",  mt5.TIMEFRAME_M5,  30000, 1, 12000),
       ("M15", mt5.TIMEFRAME_M15, 20000, 1, 8000),
       ("M30", mt5.TIMEFRAME_M30, 12000, 1, 6000),
       ("H1",  mt5.TIMEFRAME_H1,   6000, 1, 4000),
       ("H4",  mt5.TIMEFRAME_H4,   3000, 2, 1500)]
TARGET_SCALE = 10000
TEST_M15 = 400
WF_WINDOWS = 4
NB_LIST = [3, 5, 10]

# --- БОТ ---
LOT = 0.01
MAGIC = 777003
MAX_SPREAD = 50
SLIPPAGE = 30
MIN_BARS_BETWEEN_TRADES = 3

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

# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================
if not mt5.initialize():
    print("❌ MT5 initialize failed")
    exit()

data = {}
for name, tf, total, sig, cap in TFS:
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, total)
    if rates is None or len(rates) == 0:
        print(f"❌ Нет данных для {name}")
        continue
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[df.index.dayofweek < 5].copy()
    data[name] = prepare(df)
    print(f"✅ {name}: {len(df)} баров")

mt5.shutdown()

if len(data) == 0:
    print("❌ Нет данных для работы")
    exit()

print("✅ Загружено:", {k: len(v) for k, v in data.items()})

fcols = [c for c in data['M15'].columns if c not in
         ['open','high','low','close','tick_volume','spread','real_volume']]

# ==========================================
# МОДЕЛЬ + ВЫРАВНИВАНИЕ
# ==========================================
def tf_predictions(name, test_start_time):
    df = data[name]
    sig = dict((n, s) for n, _, _, s, _ in TFS)[name]
    cap = dict((n, c) for n, _, _, _, c in TFS)[name]
    times = df.index.astype('int64').values
    tr_idx = np.where(df.index < test_start_time)[0]
    tr_idx = tr_idx[-cap:] if len(tr_idx) > cap else tr_idx
    
    if len(tr_idx) < 50:
        print(f"⚠️ {name}: мало данных для обучения ({len(tr_idx)})")
        return times, np.zeros(len(times))
    
    X_tr = df[fcols].values[tr_idx]
    c_tr = df['close'].values[tr_idx]
    
    sm = gaussian_filter1d(c_tr, sigma=sig, mode='reflect')
    dd = np.zeros_like(sm)
    dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                          min_child_samples=30, n_jobs=-1,
                          verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    
    pred = m.predict(Xtr_s) / TARGET_SCALE
    k = np.std(dd) / (np.std(pred) + 1e-9)
    k = np.clip(k, 0.1, 10.0)
    
    vals = m.predict(sc.transform(df[fcols].values)) / TARGET_SCALE * k
    return times, vals

def align(tf_times, tf_vals, m15_times):
    idx = np.searchsorted(tf_times, m15_times, side='right') - 1
    idx = np.clip(idx, 0, len(tf_vals) - 1)
    return tf_vals[idx]

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
                eq += (p - entry)
                tr += 1
                pos = 0
        elif pos == -1:
            held += 1
            if held >= nb or act_long[t] or not act_short[t]:
                eq += (entry - p)
                tr += 1
                pos = 0
        if pos == 0:
            if act_long[t]:
                pos, entry, held = 1, p, 0
            elif act_short[t]:
                pos, entry, held = -1, p, 0
        curve.append(eq)
    if pos == 1:
        eq += (close_t[-1] - entry)
    if pos == -1:
        eq += (entry - close_t[-1])
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
        if name not in data:
            continue
        t_times, t_vals = tf_predictions(name, test_start_time)
        aligned[name] = align(t_times, t_vals, m15_times_all[te_start:te_end])
    
    close_t = m15['close'].values[te_start:te_end]
    
    if len(aligned) < len(TFS):
        print(f"⚠️ Окно {w+1}: не все ТФ загружены")
        continue
    
    act_long = np.all([aligned[n] > 0 for n in aligned], axis=0)
    act_short = np.all([aligned[n] < 0 for n in aligned], axis=0)
    act_long_1 = aligned['M15'] > 0
    act_short_1 = aligned['M15'] < 0
    
    for nb in NB_LIST:
        _, p4, t4 = confluence_rule(act_long, act_short, close_t, nb)
        _, p1, t1 = confluence_rule(act_long_1, act_short_1, close_t, nb)
        res[('ALL4', nb)].append(p4)
        res[('M15only', nb)].append(p1)
        if w == 0 and nb == 5:
            c4, _, _ = confluence_rule(act_long, act_short, close_t, nb)
            c1, _, _ = confluence_rule(act_long_1, act_short_1, close_t, nb)
            final_curves = {'ALL4': c4, 'M15only': c1}
    
    activity = np.mean(act_long | act_short)
    print(f"  окно {w+1}/{WF_WINDOWS} ✓ | ALL4 активен {activity:.0%}")

print(f"\n{'='*56}")
print(f"  {'вариант':>9} | {'NB':>2} | {'mean':>8} {'winrate':>7}")
for nb in NB_LIST:
    for var in ['ALL4', 'M15only']:
        a = np.array(res[(var, nb)])
        if len(a) > 0:
            print(f"  {var:>9} | {nb:>2} | {a.mean():>8.5f} {np.mean(a > 0):>6.0%}")
print(f"{'='*56}")

if final_curves:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(final_curves['ALL4'], 'b', lw=2, label='КОНФЛЮЭНЦИЯ: все ТФ > 0')
    ax.plot(final_curves['M15only'], 'r--', lw=1.2, alpha=0.7, label='M15 одиночка')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title(f'{SYMBOL} | эквити бота')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================================================
# ==================== БОТ =========================
# ==================================================
ans = input("\n🤖 Запустить БОТА? (Y/N): ").strip().upper()
if ans != 'Y':
    print("Завершение.")
    exit()

if not mt5.initialize():
    print("❌ MT5 init failed")
    exit()

ai = mt5.account_info()
if ai is None:
    print("❌ нет аккаунта")
    mt5.shutdown()
    exit()

if ai.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO:
    print(f"✅ ДЕМО #{ai.login}, баланс {ai.balance:.2f} {ai.currency}")
else:
    print(f"⚠️ РЕАЛЬНЫЙ счёт #{ai.login}!")
    if input("   Продолжить? Введите REAL: ").strip() != "REAL":
        mt5.shutdown()
        print("Завершение.")
        exit()

si = mt5.symbol_info(SYMBOL)
if si is None:
    print(f"❌ Символ {SYMBOL} не найден")
    mt5.shutdown()
    exit()

lot = max(si.volume_min, LOT)
lot = round(lot / si.volume_step) * si.volume_step

MODELS = {}
TRAIN_DATE = None
last_bar_time = None
last_trade_time = 0

def refresh_data():
    out = {}
    for name, tf, total, sig, cap in TFS:
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, total)
        if rates is None or len(rates) == 0:
            continue
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[df.index.dayofweek < 5].copy()
        out[name] = prepare(df)
    return out

def train_all(dfs, cutoff):
    mods = {}
    for name, _, _, sig, cap in TFS:
        if name not in dfs:
            continue
        df = dfs[name]
        tr_idx = np.where(df.index < cutoff)[0]
        tr_idx = tr_idx[-cap:] if len(tr_idx) > cap else tr_idx
        
        if len(tr_idx) < 50:
            print(f"⚠️ {name}: мало данных для обучения ({len(tr_idx)})")
            continue
        
        X_tr = df[fcols].values[tr_idx]
        c_tr = df['close'].values[tr_idx]
        sm = gaussian_filter1d(c_tr, sigma=sig, mode='reflect')
        dd = np.zeros_like(sm)
        dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
        
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X_tr)
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              min_child_samples=30, n_jobs=-1,
                              verbosity=-1, random_state=42)
        m.fit(Xtr_s, dd * TARGET_SCALE)
        
        pred = m.predict(Xtr_s) / TARGET_SCALE
        k = np.std(dd) / (np.std(pred) + 1e-9)
        k = np.clip(k, 0.1, 10.0)
        
        mods[name] = (m, sc, k)
    return mods

def live_signal(dfs):
    m15d = dfs['M15']
    m15t = m15d.index.astype('int64').values
    sigs = {}
    
    for name, _, _, _, _ in TFS:
        if name not in MODELS or name not in dfs:
            return False, False, {}
        
        m, sc, k = MODELS[name]
        df = dfs[name]
        vals = m.predict(sc.transform(df[fcols].values)) / TARGET_SCALE * k
        sigs[name] = align(df.index.astype('int64').values, vals, m15t)[-1]
    
    al4 = all(v > 0 for v in sigs.values())
    as4 = all(v < 0 for v in sigs.values())
    return al4, as4, sigs

def magic_positions():
    ps = mt5.positions_get(symbol=SYMBOL)
    if ps is None:
        return []
    return [p for p in ps if p.magic == MAGIC]

def close_all():
    ps = magic_positions()
    if not ps:
        return True
    
    success = True
    for p in ps:
        side = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        r = send_order(side, p.volume, p.ticket)
        if not r:
            success = False
            print(f"❌ Не удалось закрыть позицию {p.ticket}")
            _time.sleep(0.5)
    return success

def send_order(side, vol, ticket=None):
    """Отправка ордера с проверками"""
    si = mt5.symbol_info(SYMBOL)
    if si is None:
        print(f"❌ Символ {SYMBOL} не найден")
        return None
    
    if not si.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
        print(f"❌ Торговля {SYMBOL} запрещена")
        return None
    
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("❌ Нет данных по тику")
        return None
    
    # Проверка спреда
    spread = (tick.ask - tick.bid) / si.point
    if spread > MAX_SPREAD:
        print(f"⚠️ Спред слишком большой: {spread:.0f} пунктов")
        return None
    
    # Проверка средств - ИСПРАВЛЕНО
    ai = mt5.account_info()
    if ai is None:
        print("❌ Нет данных по аккаунту")
        return None
    
    # Проверка объема
    if vol < si.volume_min:
        vol = si.volume_min
    vol = round(vol / si.volume_step) * si.volume_step
    vol = min(vol, si.volume_max)
    
    # Проверка свободной маржи - ИСПРАВЛЕНО
    margin_req = si.margin_initial * vol
    free_margin = ai.margin_free  # Правильное свойство
    
    if free_margin < margin_req:
        print(f"⚠️ Недостаточно средств. Нужно {margin_req:.2f}, есть {free_margin:.2f}")
        return None
    
    price = tick.ask if side == mt5.ORDER_TYPE_BUY else tick.bid
    
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": vol,
        "type": side,
        "price": price,
        "deviation": SLIPPAGE,
        "magic": MAGIC,
        "comment": "conf_bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    if ticket:
        req["position"] = ticket
    
    for attempt in range(3):
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Ордер выполнен: {result.order}, цена: {result.price:.5f}")
            return result
        elif result:
            print(f"❌ Попытка {attempt+1}: {result.retcode} - {result.comment}")
            _time.sleep(1)
        else:
            print(f"❌ Попытка {attempt+1}: ошибка отправки")
            _time.sleep(1)
    
    return None

def set_target(target):
    """Управление позициями"""
    ps = magic_positions()
    
    # Проверяем отложенные ордера
    orders = mt5.orders_get(symbol=SYMBOL)
    if orders:
        pending = [o for o in orders if o.magic == MAGIC]
        for o in pending:
            mt5.order_delete(o.ticket)
            _time.sleep(0.1)
    
    # Определяем текущий знак позиции
    cur_volume = sum(p.volume if p.type == mt5.POSITION_TYPE_BUY else -p.volume for p in ps)
    cur_sign = 1 if cur_volume > 0.0001 else (-1 if cur_volume < -0.0001 else 0)
    
    # Если позиция уже есть и совпадает с сигналом
    if cur_sign == target:
        return f"hold ({abs(cur_volume):.2f})"
    
    # Закрываем текущую позицию
    if cur_sign != 0:
        print(f"🔄 Закрываем позицию (объем {abs(cur_volume):.2f})")
        for p in ps:
            side = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            r = send_order(side, abs(p.volume), p.ticket)
            if not r:
                print(f"❌ Не удалось закрыть позицию {p.ticket}")
                return "ошибка закрытия"
            _time.sleep(0.3)
    
    # Открываем новую позицию
    if target != 0:
        side = mt5.ORDER_TYPE_BUY if target > 0 else mt5.ORDER_TYPE_SELL
        r = send_order(side, lot)
        if not r:
            print(f"❌ Не удалось открыть позицию")
            return "ошибка открытия"
        global last_trade_time
        last_trade_time = _time.time()
    
    return f"→ {target}"

# ==================================================
# ==================== ЗАПУСК БОТА =================
# ==================================================
print("🤖 Бот запущен. Ctrl+C — остановка.")
last_bar_time = None
last_trade_time = 0

try:
    while True:
        _time.sleep(1)
        
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 1)
        if rates is None or len(rates) == 0:
            continue
        
        current_bar_time = rates[0]['time']
        
        if current_bar_time == last_bar_time:
            continue
        last_bar_time = current_bar_time
        
        now = pd.to_datetime(current_bar_time, unit='s')
        
        if now.dayofweek >= 5:
            continue
        
        try:
            dfs = refresh_data()
            if 'M15' not in dfs or len(dfs) < 3:
                print(f"⚠️ Недостаточно данных")
                continue
            
            if TRAIN_DATE != now.date():
                MODELS = train_all(dfs, now)
                if len(MODELS) >= len(TFS) - 1:
                    TRAIN_DATE = now.date()
                    print(f"🌅 {now} — модели переобучены ({len(MODELS)}/{len(TFS)})")
                else:
                    print(f"⚠️ Недостаточно моделей: {len(MODELS)}/{len(TFS)}")
                    continue
            
            al4, as4, sigs = live_signal(dfs)
            if not sigs:
                continue
            
            # Проверяем лимит частоты торгов
            if _time.time() - last_trade_time < MIN_BARS_BETWEEN_TRADES * 60 * 15:
                target = 0
                action = "wait (frequency limit)"
            else:
                target = 1 if al4 else (-1 if as4 else 0)
                action = set_target(target)
            
            sig_str = " ".join(f"{n}:{s:+.4f}" for n, s in sigs.items())
            positions = magic_positions()
            pos_str = f"поз:{len(positions)}" if positions else "нет поз"
            print(f"{now.strftime('%Y-%m-%d %H:%M')} | {sig_str} | target={target} | {action} | {pos_str}")
            
        except Exception as e:
            print(f"⚠️ ошибка цикла: {e}")
            import traceback
            traceback.print_exc()
            _time.sleep(5)

except KeyboardInterrupt:
    print("\n🛑 Бот остановлен. Позиции остаются.")
    mt5.shutdown()
    print("MT5 отключен")