import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
TARGET_SCALE = 10000
TEST_BARS = 100  # последние 100 баров для визуализации

# ==========================================
# ЗАГРУЗКА
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
print("✅ Загружено")

fcols = [c for c in data['M15'].columns if c not in
         ['open','high','low','close','tick_volume','spread','real_volume']]

# ==========================================
# ОБУЧЕНИЕ НА ИСТОРИИ, ВЫХОДЫ НА ПОСЛЕДНИХ 100 БАРАХ
# ==========================================
def get_signals_test():
    signals = {}
    for name, _, _, sig, cap in TFS:
        df = data[name]
        # обучение на всей истории кроме последних TEST_BARS
        train_end = len(df) - TEST_BARS
        tr_idx = np.arange(max(0, train_end - cap), train_end)
        X_tr = df[fcols].values[tr_idx]
        c_tr = df['close'].values[tr_idx]
        
        sm = gaussian_filter1d(c_tr, sigma=sig, mode='reflect')
        dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr)
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, min_child_samples=30,
                              n_jobs=-1, verbosity=-1, random_state=42)
        m.fit(Xtr_s, dd * TARGET_SCALE)
        k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
        
        # предсказание на последних TEST_BARS барах
        X_test = df[fcols].values[-TEST_BARS:]
        vals = m.predict(sc.transform(X_test)) / TARGET_SCALE * k
        signals[name] = vals
    
    m15_test = data['M15'].iloc[-TEST_BARS:]
    return m15_test, signals

m15_test, signals = get_signals_test()

# ==========================================
# ГРАФИК: 2 ПАНЕЛИ, 100 баров теста
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1]})

# --- панель 1: СВЕЧИ (последние 100 баров) ---
times = m15_test.index
tnum = mdates.date2num(times.to_pydatetime())
o, h, l, c = m15_test['open'].values, m15_test['high'].values, m15_test['low'].values, m15_test['close'].values
width = (tnum[1] - tnum[0]) * 0.7
up = c >= o

ax1.vlines(tnum[up], l[up], h[up], color='green', lw=0.8)
ax1.vlines(tnum[~up], l[~up], h[~up], color='red', lw=0.8)
ax1.bar(tnum[up], (c - o)[up], bottom=o[up], width=width, color='green', edgecolor='green', lw=0.8)
ax1.bar(tnum[~up], (c - o)[~up], bottom=o[~up], width=width, color='red', edgecolor='red', lw=0.8)

ax1.set_ylabel('Цена', fontsize=12)
ax1.set_title(f'{SYMBOL} M15 | последние 100 свечей (обучение за кадром)', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# --- панель 2: ВЫХОДЫ МОДЕЛЕЙ (те же 100 баров) ---
colors = {'M15': 'red', 'M30': 'orange', 'H1': 'green', 'H4': 'blue'}
for name in ['M15', 'M30', 'H1', 'H4']:
    ax2.plot(tnum, signals[name], lw=1.5, alpha=0.85, color=colors[name], label=f'{name} выход')

ax2.axhline(0, color='gray', lw=1.2, ls='--', alpha=0.7)
ax2.set_ylabel('Выход модели', fontsize=12)
ax2.set_xlabel('Время', fontsize=12)
ax2.set_title('Выходы аппроксиматоров M15/M30/H1/H4 (тестовые 100 баров)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(alpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('test_100_bars.png', dpi=150, bbox_inches='tight')
plt.show()