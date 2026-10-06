import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Настройки интерфейса ---
st.set_page_config(page_title="24h Market Predictor", layout="wide")
st.title("🎯 Модель на 24 часа: Прогноз и Сигнал")

# --- Параметры в сайдбаре ---
st.sidebar.header("⚙️ Настройки")
SYMBOL = st.sidebar.text_input("Инструмент", "EURUSD")
TIMEFRAME = mt5.TIMEFRAME_H1
LAGS = st.sidebar.slider("Лаги (память модели)", 5, 100, 35)
SIGMA = st.sidebar.slider("Sigma (плавность)", 1, 10, 2)
TARGET_SCALE = 10000 
BARS_TEST = 24  # Ровно сутки для H1

# --- Загрузка данных ---
@st.cache_data(ttl=300) # Обновление кэша каждые 5 минут
def get_fresh_data(symbol, n_bars):
    if not mt5.initialize():
        return None
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_bars)
    mt5.shutdown()
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.set_index('time')

with st.spinner("Получение свежих котировок..."):
    df = get_fresh_data(SYMBOL, 5000)

if df is None:
    st.error("Ошибка подключения к MT5")
    st.stop()

# --- Подготовка данных ---
close = df['close'].values
smoothed = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
target = np.zeros_like(smoothed)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
target_scaled = target * TARGET_SCALE

# Признаки
returns = np.diff(close, prepend=close[0])
volatility = pd.Series(close).rolling(window=20).std().fillna(0).values

X_list = []
for i in range(LAGS, len(close)):
    X_list.append(np.concatenate([
        close[i-LAGS:i][::-1], 
        returns[i-LAGS:i][::-1], 
        volatility[i-LAGS:i][::-1]
    ]))

X = np.array(X_list)
y = target_scaled[LAGS:]
time_idx = df.index[LAGS:]

# Разделение на Обучение и Тест (последние 24 часа)
X_train, y_train = X[:-BARS_TEST], y[:-BARS_TEST]
X_test = X[-BARS_TEST:]
y_true = target[LAGS:][-BARS_TEST:]
test_time = time_idx[-BARS_TEST:]

# --- Обучение ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.1, n_jobs=-1, verbosity=-1)
model.fit(X_train_s, y_train)

# Предсказания для теста
preds = model.predict(X_test_s) / TARGET_SCALE

# --- ТЕКУЩИЙ СИГНАЛ (на следующий бар) ---
# Берем самые последние данные для прогноза "вперед"
last_features = np.concatenate([
    close[-LAGS:][::-1],
    returns[-LAGS:][::-1],
    volatility[-LAGS:][::-1]
]).reshape(1, -1)
last_features_s = scaler.transform(last_features)
current_pred = model.predict(last_features_s)[0] / TARGET_SCALE

# --- ВИЗУАЛИЗАЦИЯ СИГНАЛА ---
st.subheader("📢 Текущий сигнал (на ближайший час)")
c1, c2 = st.columns([1, 3])

with c1:
    if current_pred > 0.00005:
        st.success("🚀 BUY (ВВЕРХ)")
    elif current_pred < -0.00005:
        st.error("🔻 SELL (ВНИЗ)")
    else:
        st.warning("Neutral (ЖДАТЬ)")
    st.write(f"Значение: {current_pred:.6f}")

with c2:
    st.info(f"Модель переобучена. Входные данные: {len(X_train)} баров. Тест выполнен на последних {BARS_TEST} часах.")

# --- Графики ---
col_left, col_right = st.columns(2)

with col_left:
    st.write("📊 Точность за последние 24 часа")
    fig1, ax1 = plt.subplots()
    ax1.plot(test_time, y_true, label="Рынок (Ideal)", color="blue")
    ax1.plot(test_time, preds, label="Прогноз", color="red", linestyle="--")
    ax1.legend()
    st.pyplot(fig1)

with col_right:
    st.write("💰 Эквити за последние 24 часа")
    # Простая симуляция: если предсказание > 0, покупаем
    pnl = np.diff(close[-BARS_TEST-1:]) # Реальное изменение цены
    # Сигнал был верным, если знаки совпали
    strategy_pnl = np.sign(preds[:-1]) * pnl[1:] 
    equity = np.cumsum(strategy_pnl)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(equity, color="green")
    ax2.axhline(0, color="black", lw=0.5)
    st.pyplot(fig2)

st.write(f"Последнее обновление данных: {df.index[-1]}")