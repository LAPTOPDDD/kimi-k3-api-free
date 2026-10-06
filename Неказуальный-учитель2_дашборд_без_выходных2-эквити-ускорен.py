import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- Конфигурация страницы ---
st.set_page_config(page_title="KIMI 100-Bar Test", layout="wide")
st.title("📈 LightGBM: Тест на последних 100 барах")

# --- Sidebar: Параметры ---
st.sidebar.header("⚙️ Параметры модели")
LAGS = st.sidebar.slider("Кол-во лагов", 5, 100, 50, step=5)
SIGMA = st.sidebar.slider("Sigma (Гаусс)", 1, 20, 5, step=1)
TARGET_SCALE = st.sidebar.selectbox("Масштаб цели (×)", [1000, 5000, 10000, 50000], index=2)

st.sidebar.subheader("LightGBM")
N_EST = st.sidebar.slider("n_estimators", 50, 1000, 300, step=50)
LR = st.sidebar.slider("learning_rate", 0.005, 0.3, 0.05, step=0.005)

st.sidebar.subheader("💰 Торговля")
NB = st.sidebar.slider("NB (удержание)", 1, 20, 3, step=1)
COMMISSION = st.sidebar.number_input("Комиссия ($)", value=0.0, step=0.01)

# --- Загрузка данных ---
@st.cache_data(ttl=600)
def get_mt5_data(symbol, timeframe, n_bars):
    if not mt5.initialize():
        return None
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

with st.spinner("Загрузка данных..."):
    # Берем с запасом, чтобы хватило на обучение и 100 баров теста
    df_raw = get_mt5_data("EURUSD", mt5.TIMEFRAME_H1, 10000)

if df_raw is None:
    st.error("❌ Ошибка MT5")
    st.stop()

df = df_raw.copy()
# Исключаем выходные
df = df[df.index.dayofweek < 5]

close = df['close'].values

# --- Target (Teacher) ---
smoothed = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
target = np.zeros_like(smoothed)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
target[0], target[-1] = target[1], target[-2]
target_scaled = target * TARGET_SCALE

# --- Feature Engineering (Оптимизировано) ---
returns = np.diff(close, prepend=close[0])
volatility = pd.Series(close).rolling(window=20, min_periods=1).std().fillna(0).values

X_list = []
for i in range(LAGS, len(close)):
    feat_row = np.concatenate([
        close[i-LAGS:i][::-1], 
        returns[i-LAGS:i][::-1], 
        volatility[i-LAGS:i][::-1]
    ])
    X_list.append(feat_row)

X = np.array(X_list)
y = target_scaled[LAGS:]
time_idx = df.index[LAGS:]

# --- РАЗДЕЛЕНИЕ: ТЕСТ РОВНО 100 БАРОВ ---
BARS_TEST = 100
BARS_TRAIN = len(X) - BARS_TEST

X_train, y_train = X[:BARS_TRAIN], y[:BARS_TRAIN]
X_test = X[BARS_TRAIN:]
y_true = target[LAGS:][BARS_TRAIN:] 
test_time = time_idx[BARS_TRAIN:]
test_close = close[LAGS:][BARS_TRAIN:]

st.info(f"Обучение: {BARS_TRAIN} баров | Тест: {BARS_TEST} баров")

# Масштабирование
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Обучение ---
with st.spinner("Обучение..."):
    model = lgb.LGBMRegressor(n_estimators=N_EST, learning_rate=LR, n_jobs=-1, verbosity=-1)
    model.fit(X_train_s, y_train)

# --- Инференс (Мгновенно) ---
preds = model.predict(X_test_s) / TARGET_SCALE

# --- Симуляция торговли ---
trades_data = []
equity_curve = [0.0]
position = 0 
entry_price = 0.0
entry_time = None
bars_held = 0

for t in range(len(preds)):
    current_price = test_close[t]
    current_time = test_time[t]
    
    if position != 0:
        bars_held += 1
        if bars_held >= NB:
            pnl = (current_price - entry_price) * position - COMMISSION
            equity_curve.append(equity_curve[-1] + pnl)
            trades_data.append({'Time': entry_time, 'PnL': pnl, 'Type': 'Long' if position==1 else 'Short'})
            position, bars_held = 0, 0
        else:
            equity_curve.append(equity_curve[-1])
    else:
        equity_curve.append(equity_curve[-1])
        if preds[t] > 0:
            position, entry_price, entry_time, bars_held = 1, current_price, current_time, 0
        elif preds[t] < 0:
            position, entry_price, entry_time, bars_held = -1, current_price, current_time, 0

equity_curve = np.array(equity_curve[1:])

# --- Визуализация ---
c1, c2, c3 = st.columns(3)
c1.metric("R² Score", f"{r2_score(y_true, preds):.4f}")
c2.metric("Total PnL", f"{sum(d['PnL'] for d in trades_data):.4f}")
c3.metric("Trades", len(trades_data))

# График 1: Учитель vs Модель (РОВНО 100 БАРОВ)
st.subheader("🎯 Прогноз на тестовом участке (100 баров)")
fig1, ax1 = plt.subplots(figsize=(16, 5))
ax1.plot(test_time, y_true, label='Teacher (Идеал)', color='blue', lw=2)
ax1.plot(test_time, preds, label='LightGBM (Прогноз)', color='red', linestyle='--')
ax1.axhline(0, color='black', lw=0.5)
ax1.set_title("Сравнение Teacher и Model на последних 100 барах")
ax1.legend()
ax1.grid(alpha=0.2)
st.pyplot(fig1)

# График 2: Эквити
st.subheader("💰 Результат торговли (Equity)")
fig2, ax2 = plt.subplots(figsize=(16, 4))
ax2.plot(equity_curve, color='black')
ax2.fill_between(range(len(equity_curve)), equity_curve, 0, where=equity_curve>=0, color='green', alpha=0.2)
ax2.fill_between(range(len(equity_curve)), equity_curve, 0, where=equity_curve<0, color='red', alpha=0.2)
ax2.set_title("Накопленная прибыль на 100 барах теста")
ax2.grid(alpha=0.2)
st.pyplot(fig2)

# Таблица
if trades_data:
    st.subheader("📋 Журнал сделок")
    st.dataframe(pd.DataFrame(trades_data), use_container_width=True)