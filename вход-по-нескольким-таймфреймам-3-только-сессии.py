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
TFS = [("M15", mt5.TIMEFRAME_M15, 20000, 1, 8000),
       ("M30", mt5.TIMEFRAME_M30, 12000, 1, 6000),
       ("H1",  mt5.TIMEFRAME_H1,   6000, 1, 4000),
       ("H4",  mt5.TIMEFRAME_H4,   3000, 2, 1500)]
TF_SIG = {n: s for n, _, _, s, _ in TFS}
TF_CAP = {n: c for n, _, _, _, c in TFS}

TARGET_SCALE = 10000

# ТОРГОВАЯ СЕССИЯ (UTC): лондонская 07:00-16:00
SESSION_START_HOUR = 7     # UTC
SESSION_END_HOUR = 16      # UTC

# ==========================================
# 1. ЗАГРУЗКА
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
# 2. РАЗБИВКА НА ТОРГОВЫЕ ДНИ (только часы сессии)
# ==========================================
m15 = data['M15']
m15['date'] = m15.index.date
m15['hour'] = m15.index.hour

# список торговых дней: каждый день = интервал 07:00–16:00 UTC
dates = sorted(m15['date'].unique())
# последние ~30 торговых дней для теста
trade_days = dates[-30:]
print(f"\nТорговых дней в тесте: {len(trade_days)}")
print(f"Первый: {trade_days[0]}  Последний: {trade_days[-1]}")

# ==========================================
# 3. УТРЕННЕЕ ОБУЧЕНИЕ НА КАЖДЫЙ ДЕНЬ
# ==========================================
def train_morning(name, session_start_time):
    df = data[name]
    tr_idx = np.where(df.index < session_start_time)[0]
    tr_idx = tr_idx[-TF_CAP[name]:] if len(tr_idx) > TF_CAP[name] else tr_idx
    X_tr, c_tr = df[fcols].values[tr_idx], df['close'].values[tr_idx]

    sm = gaussian_filter1d(c_tr, sigma=TF_SIG[name], mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    return m, sc, k

def predict_session(m, sc, k, df, start_t, end_t):
    mask = (df.index >= start_t) & (df.index < end_t)
    X = df[fcols].values[mask]
    preds = m.predict(sc.transform(X)) / TARGET_SCALE * k
    return preds, df['close'].values[mask]

# ==========================================
# 4. ПРАВИЛО СЕССИИ: вход по конфлюэнции, закрытие в конце сессии
# ==========================================
def session_trade(signals_4, close_t, mode):
    """mode: 'ALL4' | 'H4only' | 'M15only'
    Возвращает PnL дня + кривую"""
    # выбираем активные сигналы по режиму
    if mode == 'ALL4':
        actL = np.all([s > 0 for s in signals_4], axis=0)
        actS = np.all([s < 0 for s in signals_4], axis=0)
    elif mode == 'H4only':
        actL, actS = signals_4[3] > 0, signals_4[3] < 0
    else:
        actL, actS = signals_4[0] > 0, signals_4[0] < 0

    eq, pos, entry = 0.0, 0, 0.0
    curve = [0.0]
    for t in range(len(close_t)):
        p = close_t[t]
        # закрытие при смене режима (или противоположном сигнале)
        if pos == 1 and (actS[t] or not actL[t]):
            eq += (p - entry); pos = 0
        elif pos == -1 and (actL[t] or not actS[t]):
            eq += (entry - p); pos = 0
        # вход
        if pos == 0:
            if actL[t]: pos, entry = 1, p
            elif actS[t]: pos, entry = -1, p
        curve.append(eq + (p - entry) * pos if pos != 0 else eq)
    # принудительное закрытие в конце сессии
    if pos != 0:
        eq += (close_t[-1] - entry) * pos if pos == 1 else (entry - close_t[-1])
    return eq, np.array(curve[1:])

# ==========================================
# 5. ПОДНЕВНЫЙ WF-ТЕСТ
# ==========================================
results = {'ALL4': [], 'H4only': [], 'M15only': [], 'BH': []}
day_curves = {'ALL4': [], 'H4only': [], 'BH': []}

for date in trade_days:
    # определяем временные окна
    day_m15 = m15[m15['date'] == date]
    if len(day_m15) == 0: continue

    session_start = day_m15.index[0].replace(hour=SESSION_START_HOUR, minute=0, second=0)
    session_end = day_m15.index[0].replace(hour=SESSION_END_HOUR, minute=0, second=0)
    if session_start >= session_end: continue

    # УТРЕННЕЕ ОБУЧЕНИЕ: 4 модели на истории до 07:00 UTC
    signals = []
    try:
        for name, _, _, _, _ in TFS:
            m, sc, k = train_morning(name, session_start)
            pred, _ = predict_session(m, sc, k, data[name], session_start, session_end)
            # выравниваем по M15: берём последний доступный сигнал для каждого M15-бара
            df_tf = data[name]
            tf_idx = np.searchsorted(df_tf.index.values, day_m15.index[day_m15.index.hour >= SESSION_START_HOUR].values, side='right') - 1
            tf_idx = np.clip(tf_idx, 0, len(pred) - 1)
            # пересчитываем предсказание на M15-сетке сессии
            m15_session = day_m15[day_m15.index.hour >= SESSION_START_HOUR]
            tf_idx2 = np.searchsorted(df_tf.index.values, m15_session.index.values, side='right') - 1
            tf_idx2 = np.clip(tf_idx2, 0, len(pred) - 1)
            signals.append(pred[tf_idx2])
    except Exception as e:
        continue

    if not signals or len(set(len(s) for s in signals)) > 1: continue
    if len(signals[0]) < 2: continue

    close_t = m15['close'].loc[m15_session.index].values

    p4, c4 = session_trade(signals, close_t, 'ALL4')
    pH, cH = session_trade(signals, close_t, 'H4only')
    p1, c1 = session_trade(signals, close_t, 'M15only')
    pBH = close_t[-1] - close_t[0]

    results['ALL4'].append(p4); results['H4only'].append(pH)
    results['M15only'].append(p1); results['BH'].append(pBH)
    day_curves['ALL4'].append(c4); day_curves['H4only'].append(cH); day_curves['BH'].append(np.cumsum(np.diff(close_t, prepend=close_t[0])))

    print(f"  {date} | ALL4 {p4:>8.4f} | H4 {pH:>8.4f} | B&H {pBH:>8.4f}")

# ==========================================
# 6. АГРЕГАТ + РАЗБИВКА ПО РЕЖИМАМ
# ==========================================
print(f"\n{'='*62}")
print(f"  {'вариант':>8} | {'mean':>8} {'std':>7} {'winrate':>7} {'score':>6} | дней")
for v in ['ALL4', 'H4only', 'M15only', 'BH']:
    a = np.array(results[v])
    if len(a) == 0: continue
    print(f"  {v:>8} | {a.mean():>8.5f} {a.std():>7.5f} "
          f"{np.mean(a > 0):>6.0%} {a.mean()/(a.std()+1e-12):>6.2f} | {len(a)}")
print(f"{'='*62}")

a4, aH, aB = np.array(results['ALL4']), np.array(results['H4only']), np.array(results['BH'])
up, flat, down = np.where(aB > 0.002)[0], np.where(np.abs(aB) <= 0.002)[0], np.where(aB < -0.002)[0]
print(f"\n  режим  дней | ALL4 mean | H4 mean | B&H mean")
for nm, ix in [('UP', up), ('FLAT', flat), ('DOWN', down)]:
    m4 = a4[ix].mean() if len(ix) else 0.0
    mH = aH[ix].mean() if len(ix) else 0.0
    mB = aB[ix].mean() if len(ix) else 0.0
    print(f"  {nm:>4} {len(ix):>5} | {m4:>9.5f} | {mH:>7.5f} | {mB:>8.5f}")

# ==========================================
# 7. ГРАФИКИ
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
xw = np.arange(1, len(trade_days) + 1)
w_ = 0.27
axes[0].bar(xw - w_, a4, w_, label='ALL4', color='b')
axes[0].bar(xw,     aH, w_, label='H4only', color='orange')
axes[0].bar(xw + w_, aB, w_, label='Buy&Hold', color='gray', alpha=0.6)
axes[0].axhline(0, color='k', lw=0.8); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title(f'{SYMBOL} | ежедневный PnL (сессия {SESSION_START_HOUR}-{SESSION_END_HOUR} UTC)')

axes[1].plot(xw, np.cumsum(a4), 'b', lw=2.5, label='ALL4')
axes[1].plot(xw, np.cumsum(aH), color='orange', lw=1.5, label='H4only')
axes[1].plot(xw, np.cumsum(aB), color='gray', lw=1.2, alpha=0.7, label='Buy&Hold')
axes[1].axhline(0, color='k', lw=0.8); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_title('Кумулятивный PnL (переобучение каждое утро)')
axes[1].set_xlabel('Торговый день')
plt.tight_layout()
plt.savefig('confluence_daily_wf.png', dpi=150)
plt.show()