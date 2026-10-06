import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="KIMI Causal + Equity", layout="wide")
st.title("📈 LightGBM vs Teacher + 💰 Equity Curve")

# --- Sidebar: Параметры модели ---
st.sidebar.header("⚙️ Параметры модели")
LAGS = st.sidebar.slider("Кол-во лагов", 5, 100, 50, step=5)
SIGMA = st.sidebar.slider("Sigma (Гаусс)", 1, 20, 5, step=1)
TARGET_SCALE = st.sidebar.selectbox("Масштаб цели (×)", [1000, 5000, 10000, 50000], index=2)

st.sidebar.subheader("LightGBM")
N_EST = st.sidebar.slider("n_estimators", 50, 1000, 300, step=50)
LR = st.sidebar.slider("learning_rate", 0.005, 0.3, 0.05, step=0.005)
NUM_LEAVES = st.sidebar.slider("num_leaves", 15, 255, 63, step=16)
MAX_DEPTH = st.sidebar.slider("max_depth", -1, 20, 10, step=1)
MIN_DATA = st.sidebar.slider("min_data_in_leaf", 1, 50, 10, step=1)
FEAT_FRAC = st.sidebar.slider("feature_fraction", 0.3, 1.0, 0.9, step=0.05)

# --- Фильтры ---
st.sidebar.subheader("🗓️ Фильтры")
filter_weekends = st.sidebar.checkbox("Исключить выходные (Сб-Вс)?", value=True)

# --- Параметры торговли ---
st.sidebar.subheader("💰 Параметры сделок")
NB = st.sidebar.slider("NB (баров держать позу)", 1, 20, 3, step=1)
COMMISSION = st.sidebar.number_input("Комиссия за сделку ($)", value=0.0, step=0.1)

# --- Загрузка данных ---
with st.spinner("Подключение к MT5..."):
    if not mt5.initialize():
        st.error("❌ MT5 не инициализирован")
        st.stop()
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 15000)
    mt5.shutdown()

if rates is None or len(rates) == 0:
    st.error("❌ Нет данных")
    st.stop()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

if filter_weekends:
    df = df[df.index.dayofweek < 5]
    st.info(f"Выходные исключены. Осталось {len(df)} баров.")

close = df['close'].values
BARS_TOTAL = len(close)

BARS_TRAIN = st.sidebar.slider("Обучающих баров", 500, BARS_TOTAL - 200, min(10000, BARS_TOTAL - 200), step=100)

st.success(f"✅ Загружено {BARS_TOTAL} баров")

# --- Target (Teacher) ---
smoothed = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
target = np.zeros_like(smoothed)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
target[0] = target[1]
target[-1] = target[-2]
target_scaled = target * TARGET_SCALE

# --- Features ---
returns = np.diff(close, prepend=close[0])
volatility = np.array([np.std(close[max(0, i-20):i+1]) for i in range(len(close))])

X = np.zeros((len(close), LAGS * 3))
for i in range(LAGS, len(close)):
    X[i] = np.concatenate([close[i-LAGS:i][::-1], returns[i-LAGS:i][::-1], volatility[i-LAGS:i][::-1]])
X = X[LAGS:]
y = target_scaled[LAGS:]
time_idx = df.index[LAGS:]

# Split
X_train, y_train = X[:BARS_TRAIN], y[:BARS_TRAIN]
X_test, y_true = X[BARS_TRAIN:], target[LAGS:][BARS_TRAIN:]
test_time = time_idx[BARS_TRAIN:]
test_close = close[LAGS:][BARS_TRAIN:]

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train ---
with st.spinner("Обучение..."):
    model = lgb.LGBMRegressor(n_estimators=N_EST, learning_rate=LR, num_leaves=NUM_LEAVES,
                               max_depth=MAX_DEPTH, min_data_in_leaf=MIN_DATA,
                               feature_fraction=FEAT_FRAC, bagging_fraction=0.8, bagging_freq=5,
                               lambda_l1=0.01, lambda_l2=0.01, random_state=42, verbosity=-1)
    model.fit(X_train_s, y_train)

# --- Inference ---
#with st.spinner("Инференс..."):
#    preds_s = [model.predict(X_test_s[i].reshape(1, -1))[0] for i in range(len(X_test_s))]
#    preds = np.array(preds_s) / TARGET_SCALE


# --- Inference ---
with st.spinner("Инферс..."):
    preds_s = []
    for i in range(len(X_test_s)):
        # Преобразуем в 2D массив и убираем warning
        pred = model.predict(X_test_s[i].reshape(1, -1), 
                             feature_name=['dummy']) # <-- ВОТ ВОЛШЕБНАЯ СТРОЧКА
        preds_s.append(pred[0])
    preds = np.array(preds_s) / TARGET_SCALE


# --- Metrics ---
mse = mean_squared_error(y_true, preds)
mae = mean_absolute_error(y_true, preds)
r2 = r2_score(y_true, preds)

# --- Equity Curve Simulation ---
st.subheader("💰 Симуляция торговли")

# Логика:
# Если preds[t] > 0 → LONG на NB баров
# Если preds[t] < 0 → SHORT на NB баров
# PnL = (цена_выхода - цена_входа) * направление - комиссия

n = len(preds)
equity = np.zeros(n)
position = 0  # 1 = long, -1 = short, 0 = flat
entry_price = 0.0
bars_held = 0

pnl_list = []
trade_times = []
equity_curve = [0.0]

for t in range(n):
    current_price = test_close[t]
    
    # Проверяем, нужно ли закрыть позу
    if position != 0 and bars_held >= NB:
        # Закрываем позу
        pnl = (current_price - entry_price) * position - COMMISSION
        equity_curve.append(equity_curve[-1] + pnl)
        pnl_list.append(pnl)
        position = 0
        bars_held = 0
    else:
        equity_curve.append(equity_curve[-1])
    
    # Проверяем сигнал на ВХОД (только если сейчас flat)
    if position == 0:
        if preds[t] > 0:
            position = 1  # LONG
            entry_price = current_price
            bars_held = 0
            trade_times.append(test_time[t])
        elif preds[t] < 0:
            position = -1  # SHORT
            entry_price = current_price
            bars_held = 0
            trade_times.append(test_time[t])
    else:
        bars_held += 1

# Если осталась открытая поза в конце — закрываем по последней цене
if position != 0:
    pnl = (test_close[-1] - entry_price) * position - COMMISSION
    equity_curve[-1] += pnl
    pnl_list.append(pnl)

equity_curve = np.array(equity_curve[1:])  # убираем начальный 0

total_trades = len(pnl_list)
winning_trades = sum(1 for p in pnl_list if p > 0)
win_rate = winning_trades / total_trades if total_trades > 0 else 0
total_pnl = equity_curve[-1] if len(equity_curve) > 0 else 0
max_drawdown = np.min(equity_curve) - np.max(equity_curve[:np.argmin(equity_curve)+1]) if len(equity_curve) > 0 else 0

# --- Plot 1: Teacher vs Model ---
PLOT_OFFSET = 50
PLOT_LIMIT = 500

fig1, ax1 = plt.subplots(figsize=(16, 6))
ax1.plot(test_time[PLOT_OFFSET:PLOT_OFFSET+PLOT_LIMIT], 
         y_true[PLOT_OFFSET:PLOT_OFFSET+PLOT_LIMIT], 
         label='True Non-Causal (Teacher)', color='blue', alpha=0.8, linewidth=1.5)
ax1.plot(test_time[PLOT_OFFSET:PLOT_OFFSET+PLOT_LIMIT], 
         preds[PLOT_OFFSET:PLOT_OFFSET+PLOT_LIMIT], 
         label='LightGBM (Causal)', color='red', linestyle='--', linewidth=1.5)
ax1.set_title(f"Teacher vs LightGBM | R²={r2:.3f} | MSE={mse:.2e}")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# --- Метрики ---
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("MSE", f"{mse:.2e}")
c2.metric("MAE", f"{mae:.2e}")
c3.metric("R²", f"{r2:.4f}")
c4.metric("Сделок", total_trades)
c5.metric("Win Rate", f"{win_rate:.1%}")
c6.metric("Total PnL", f"{total_pnl:.2f}", delta=f"{total_pnl:.2f}")

# --- Plot 2: Equity Curve ---
fig2, ax2 = plt.subplots(figsize=(16, 5))
ax2.fill_between(range(len(equity_curve)), equity_curve, 0, 
                  where=equity_curve >= 0, color='green', alpha=0.3, interpolate=True)
ax2.fill_between(range(len(equity_curve)), equity_curve, 0, 
                  where=equity_curve < 0, color='red', alpha=0.3, interpolate=True)
ax2.plot(equity_curve, color='black', linewidth=1.2)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax2.set_title(f"💰 Equity Curve | NB={NB} | Комиссия=${COMMISSION} | Max DD={max_drawdown:.2f}")
ax2.set_xlabel("Trade #")
ax2.set_ylabel("Cumulative PnL")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# --- Таблица сделок ---
if len(pnl_list) > 0:
    st.subheader("📋 Последние сделки")
    trades_df = pd.DataFrame({
        'Time': trade_times[:50],
        'PnL': pnl_list[:50]
    })
    st.dataframe(trades_df.tail(20))