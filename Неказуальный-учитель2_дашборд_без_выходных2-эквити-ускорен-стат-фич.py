import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- Конфигурация страницы ---
st.set_page_config(page_title="KIMI Stats Features Test", layout="wide")
st.title("📈 LightGBM: Статистические фичи + Тест на 100 барах")

# --- Sidebar: Параметры ---
st.sidebar.header("⚙️ Параметры модели")
SIGMA = st.sidebar.slider("Sigma (Гаусс)", 1, 20, 5, step=1)
TARGET_SCALE = st.sidebar.selectbox("Масштаб цели (×)", [1000, 5000, 10000, 50000], index=2)

st.sidebar.subheader("LightGBM")
N_EST = st.sidebar.slider("n_estimators", 50, 1000, 300, step=50)
LR = st.sidebar.slider("learning_rate", 0.005, 0.3, 0.05, step=0.005)
# Увеличиваем min_child_samples, так как фич стало больше (защита от переобучения)
MIN_CHILD = st.sidebar.slider("min_child_samples", 10, 100, 30, step=10)

st.sidebar.subheader("💰 Торговля")
NB = st.sidebar.slider("NB (удержание, баров)", 1, 20, 3, step=1)
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

with st.spinner("Загрузка и обработка данных..."):
    # Берем с запасом (10000 баров ~ 1.5 года на H1)
    df = get_mt5_data("EURUSD", mt5.TIMEFRAME_H1, 10000)

if df is None:
    st.error("❌ Ошибка MT5 или нет данных")
    st.stop()

# Исключаем выходные для чистоты статистики
df = df[df.index.dayofweek < 5].copy()

# --- 1. Целевая переменная (Teacher) ---
close = df['close'].values
smoothed = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
target = np.zeros_like(smoothed)
target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
target[0], target[-1] = target[1], target[-2]

df['target_raw'] = target
df['target_scaled'] = target * TARGET_SCALE

# --- 2. FEATURE ENGINEERING (Векторизовано и быстро) ---
# Базовые
df['returns'] = df['close'].pct_change()

# 1. Меры центральной тенденции
df['sma_20'] = df['close'].rolling(20).mean()
df['price_to_sma'] = df['close'] / df['sma_20']

# 2. Меры дисперсии (волатильность)
df['std_20'] = df['close'].rolling(20).std()
df['cv_20'] = df['std_20'] / df['sma_20']  # Коэффициент вариации

# 3. Меры формы распределения (Самые важные для ML!)
df['skew_20'] = df['close'].rolling(20).skew()
df['kurt_20'] = df['close'].rolling(20).kurt()

# 4. Квантили и диапазон
df['low_20'] = df['close'].rolling(20).min()
df['high_20'] = df['close'].rolling(20).max()
df['range_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-9)

# 5. Статистика доходности
df['ret_mean_20'] = df['returns'].rolling(20).mean()
df['ret_std_20'] = df['returns'].rolling(20).std()
df['sharpe_20'] = df['ret_mean_20'] / (df['ret_std_20'] + 1e-9) * np.sqrt(252)

# 6. Статистика объемов (tick_volume в MT5)
df['vol_mean_20'] = df['tick_volume'].rolling(20).mean()
df['vol_ratio'] = df['tick_volume'] / (df['vol_mean_20'] + 1e-9)

# Определяем список фич для модели (исключаем target и время)
feature_cols = [col for col in df.columns if col not in ['target_raw', 'target_scaled']]

# --- 3. ОЧИСТКА И РАЗДЕЛЕНИЕ ---
# Dropna удалит первые ~20 строк, где rolling-окна дают NaN
df_clean = df.dropna().copy()

st.info(f"Всего баров после очистки от NaN: {len(df_clean)} | Использовано фич: {len(feature_cols)}")

# ЖЕСТКОЕ разделение: последние 100 баров - тест
BARS_TEST = 100
BARS_TRAIN = len(df_clean) - BARS_TEST

# Данные для обучения
X_train = df_clean[feature_cols].iloc[:BARS_TRAIN].values
y_train = df_clean['target_scaled'].iloc[:BARS_TRAIN].values

# Данные для теста (ровно 100 баров)
X_test = df_clean[feature_cols].iloc[BARS_TRAIN:].values
y_test_scaled = df_clean['target_scaled'].iloc[BARS_TRAIN:].values
y_test_true = df_clean['target_raw'].iloc[BARS_TRAIN:].values # Для графиков в исходном масштабе

test_time = df_clean.index[BARS_TRAIN:]
test_close = df_clean['close'].iloc[BARS_TRAIN:].values

# Масштабирование
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- 4. Обучение ---
with st.spinner("Обучение LightGBM..."):
    model = lgb.LGBMRegressor(
        n_estimators=N_EST, 
        learning_rate=LR, 
        min_child_samples=MIN_CHILD, # Важно для стат. фич!
        n_jobs=-1, 
        verbosity=-1,
        random_state=42
    )
    model.fit(X_train_s, y_train)

# --- 5. Инференс ---
preds_scaled = model.predict(X_test_s)
preds = preds_scaled / TARGET_SCALE # Возвращаем к исходному масштабу для торговли

# --- 6. Симуляция торговли ---
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
            trades_data.append({
                'Time': entry_time.strftime('%Y-%m-%d %H:%M'), 
                'Type': 'Long' if position==1 else 'Short',
                'Entry': entry_price,
                'Exit': current_price,
                'PnL': pnl
            })
            position, bars_held = 0, 0
        else:
            equity_curve.append(equity_curve[-1])
    else:
        equity_curve.append(equity_curve[-1])
        # Входим в позицию, если прогноз сильный (можно добавить порог, например abs(preds[t]) > threshold)
        if preds[t] > 0:
            position, entry_price, entry_time, bars_held = 1, current_price, current_time, 0
        elif preds[t] < 0:
            position, entry_price, entry_time, bars_held = -1, current_price, current_time, 0

equity_curve = np.array(equity_curve[1:])

# --- 7. Визуализация ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² Score", f"{r2_score(y_test_true, preds):.4f}")
c2.metric("MAE", f"{mean_absolute_error(y_test_true, preds):.2e}")
c3.metric("Total PnL", f"{sum(d['PnL'] for d in trades_data):.2f} $")
c4.metric("Trades", len(trades_data))

# График 1: Учитель vs Модель
st.subheader("🎯 Прогноз на тестовом участке (последние 100 баров)")
fig1, ax1 = plt.subplots(figsize=(16, 5))
ax1.plot(test_time, y_test_true, label='Teacher (Идеал)', color='blue', lw=2)
ax1.plot(test_time, preds, label='LightGBM (Прогноз)', color='red', linestyle='--', lw=2)
ax1.axhline(0, color='black', lw=1, alpha=0.5)
ax1.set_title(f"Сравнение Teacher и Model (Sigma={SIGMA})")
ax1.legend()
ax1.grid(alpha=0.2)
st.pyplot(fig1)

# График 2: Эквити
st.subheader("💰 Результат торговли (Equity)")
fig2, ax2 = plt.subplots(figsize=(16, 4))
ax2.plot(equity_curve, color='black', lw=2)
ax2.fill_between(range(len(equity_curve)), equity_curve, 0, where=equity_curve>=0, color='green', alpha=0.3)
ax2.fill_between(range(len(equity_curve)), equity_curve, 0, where=equity_curve<0, color='red', alpha=0.3)
ax2.set_title(f"Накопленная прибыль (Удержание: {NB} баров, Комиссия: {COMMISSION}$)")
ax2.grid(alpha=0.2)
st.pyplot(fig2)

# Важность признаков
st.subheader("🧠 Важность статистических признаков (Feature Importance)")
importance = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15) # Топ 15

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1], color='purple', alpha=0.7)
ax3.set_xlabel('Важность')
ax3.set_title('Топ-15 самых полезных признаков для модели')
ax3.grid(axis='x', alpha=0.3)
st.pyplot(fig3)

# Таблица сделок
if trades_data:
    st.subheader("📋 Журнал сделок")
    st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
else:
    st.warning("Сделок не было. Попробуйте уменьшить NB или изменить Sigma.")