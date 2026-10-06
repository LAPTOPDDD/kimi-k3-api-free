import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="KIMI Causal Dashboard", layout="wide")
st.title("📈 LightGBM vs Non-Causal Teacher (Weekdays Only)")

# --- Sidebar: Параметры, не зависящие от данных ---
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

# --- Загрузка данных ---
with st.spinner("Подключение к MT5 и загрузка данных..."):
    if not mt5.initialize():
        st.error("❌ Не удалось подключиться к MetaTrader 5.")
        st.stop()
    
    # Грузим чуть больше данных про запас, чтобы после фильтрации выходных хватило
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 15000)
    mt5.shutdown()

if rates is None or len(rates) == 0:
    st.error(f"❌ Нет данных для EURUSD")
    st.stop()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)

# --- Фильтр выходных ---
st.sidebar.subheader("🗓️ Фильтры")
filter_weekends = st.sidebar.checkbox("Исключить выходные (Сб-Вс)?", value=True)

if filter_weekends:
    # dt.dayofweek: Monday=0, Sunday=6. Оставляем 0-4 (Пн-Пт)
    mask = df.index.dayofweek < 5
    df = df[mask]
    st.info(f"Выходные исключены. Осталось {len(df)} баров.")

# Теперь, когда данные окончательны, настраиваем слайдеры размера выборки
st.sidebar.subheader("📊 Размеры выборок")
BARS_TOTAL = len(df)
BARS_TRAIN = st.sidebar.slider(
    "Обучающих баров", 
    min_value=500, 
    max_value=BARS_TOTAL - 200, # Оставляем место для теста
    value=min(10000, BARS_TOTAL - 200),
    step=100
)

# --- Подготовка данных ---
close = df['close'].values
st.success(f"✅ Используется {BARS_TOTAL} баров для анализа.")

# --- Target: неказуальная производная ---
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
    X[i] = np.concatenate([
        close[i-LAGS:i][::-1],
        returns[i-LAGS:i][::-1],
        volatility[i-LAGS:i][::-1]
    ])
X = X[LAGS:]
y = target_scaled[LAGS:]
time_idx = df.index[LAGS:]

# Split
X_train = X[:BARS_TRAIN]
y_train = y[:BARS_TRAIN]
X_test = X[BARS_TRAIN:]
y_true_scaled = y[BARS_TRAIN:]
y_true = target[LAGS:][BARS_TRAIN:]
test_time = time_idx[BARS_TRAIN:]

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train ---
with st.spinner("Обучение LightGBM..."):
    model = lgb.LGBMRegressor(
        n_estimators=N_EST, learning_rate=LR, num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH, min_data_in_leaf=MIN_DATA,
        feature_fraction=FEAT_FRAC, bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.01, lambda_l2=0.01, random_state=42, verbosity=-1
    )
    model.fit(X_train_s, y_train)

# --- Inference ---
with st.spinner("Инференс..."):
    preds_s = [model.predict(X_test_s[i].reshape(1, -1))[0] for i in range(len(X_test_s))]
    preds = np.array(preds_s) / TARGET_SCALE

# --- Metrics ---
mse = mean_squared_error(y_true, preds)
mae = mean_absolute_error(y_true, preds)
r2 = r2_score(y_true, preds)

# --- Plot ---
#PLOT_OFFSET = 50
#fig, ax = plt.subplots(figsize=(16, 7))
#ax.plot(test_time[PLOT_OFFSET:], y_true[PLOT_OFFSET:], label='True Non-Causal (Teacher)', color='blue', alpha=0.8, linewidth=1.5)
#ax.plot(test_time[PLOT_OFFSET:], preds[PLOT_OFFSET:], label='LightGBM (Causal)', color='red', linestyle='--', linewidth=1.5)
PLOT_OFFSET = 50
PLOT_LIMIT = 100   # сколько баров рисовать; можешь поставить 300, 1000, 2000

fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(
    test_time[PLOT_OFFSET:PLOT_OFFSET+PLOT_LIMIT],
    y_true[PLOT_OFFSET:PLOT_OFFSET+PLOT_LIMIT],
    label='True Non-Causal (Teacher)',
    color='blue',
    alpha=0.8,
    linewidth=1.5
)

ax.plot(
    test_time[PLOT_OFFSET:PLOT_OFFSET+PLOT_LIMIT],
    preds[PLOT_OFFSET:PLOT_OFFSET+PLOT_LIMIT],
    label='LightGBM (Causal)',
    color='red',
    linestyle='--',
    linewidth=1.5
)

ax.set_title(f"Teacher vs LightGBM | R²={r2:.3f} | MSE={mse:.2e}")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# --- Metrics ---
c1, c2, c3 = st.columns(3)
c1.metric("MSE", f"{mse:.2e}")
c2.metric("MAE", f"{mae:.2e}")
c3.metric("R²", f"{r2:.4f}")

# --- Feature Importance ---
st.subheader("🔍 Важность признаков")
fi = pd.Series(model.feature_importances_, name="importance")
lag_names = [f"close_lag_{i}" for i in range(1, LAGS+1)]
ret_names = [f"return_lag_{i}" for i in range(1, LAGS+1)]
vol_names = [f"vol_lag_{i}" for i in range(1, LAGS+1)]
fi.index = lag_names + ret_names + vol_names
st.bar_chart(fi.sort_values(ascending=False).head(30))