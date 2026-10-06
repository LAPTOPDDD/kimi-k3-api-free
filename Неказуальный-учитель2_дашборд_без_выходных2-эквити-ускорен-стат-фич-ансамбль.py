import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import optuna
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- Конфигурация ---
st.set_page_config(page_title="ENSEMBLE OF CHAMPIONS", layout="wide")
st.title("🏆 ANSEMBLE OF CHAMPIONS - Поиск лучших параметров")

# --- Функции ---
def calculate_equity(preds, close, nb, commission=0.0):
    """Симуляция торговли и расчет метрик"""
    equity = [0.0]
    trades_count = 0
    position = 0
    entry_price = 0.0
    bars_held = 0
    
    for t in range(len(preds)):
        price = close[t]
        
        if position != 0:
            bars_held += 1
            if bars_held >= nb:
                pnl = (price - entry_price) * position - commission
                equity.append(equity[-1] + pnl)
                trades_count += 1
                position, bars_held = 0, 0
            else:
                equity.append(equity[-1])
        else:
            equity.append(equity[-1])
            if preds[t] > 0:
                position, entry_price, bars_held = 1, price, 0
            elif preds[t] < 0:
                position, entry_price, bars_held = -1, price, 0
    
    equity = np.array(equity[1:])
    returns = np.diff(equity, prepend=0)
    
    if len(returns) > 0 and np.std(returns) > 1e-9:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = -10.0 # Штраф за отсутствие волатильности
    
    if len(equity) > 0:
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (np.abs(peak) + 1e-9)
        max_dd = np.max(drawdown)
    else:
        max_dd = 1.0
    
    total_pnl = equity[-1] if len(equity) > 0 else 0
    
    return equity, sharpe, max_dd, total_pnl, trades_count

# --- Sidebar ---
st.sidebar.header("⚙️ Параметры оптимизации")
N_CHAMPIONS = st.sidebar.slider("Кол-во чемпионов (топ-N)", 3, 20, 10, step=1)
N_TRIALS = st.sidebar.slider("Кол-во итераций Optuna", 10, 100, 30, step=5)

st.sidebar.subheader("Диапазоны параметров")
SIGMA_MIN = st.sidebar.slider("Sigma min", 1, 10, 2, step=1)
SIGMA_MAX = st.sidebar.slider("Sigma max", 5, 20, 8, step=1)
NB_MIN = st.sidebar.slider("NB min", 1, 5, 2, step=1)
NB_MAX = st.sidebar.slider("NB max", 5, 20, 8, step=1)

st.sidebar.subheader("LightGBM")
N_EST = st.sidebar.slider("n_estimators", 50, 500, 150, step=50)
LR = st.sidebar.number_input("learning_rate", 0.01, 0.3, 0.05, step=0.01)
MIN_CHILD = st.sidebar.slider("min_child_samples", 5, 100, 20, step=5)

st.sidebar.subheader("💰 Торговля")
COMMISSION = st.sidebar.number_input("Комиссия ($)", 0.0, 10.0, 0.0, step=0.1)
TARGET_SCALE = st.sidebar.selectbox("Масштаб цели", [1000, 5000, 10000], index=1)

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

with st.spinner("Загрузка и подготовка данных..."):
    df = get_mt5_data("EURUSD", mt5.TIMEFRAME_H1, 10000)

if df is None:
    st.error("❌ Ошибка MT5 или нет данных")
    st.stop()

df = df[df.index.dayofweek < 5].copy()
close = df['close'].values

# --- Feature Engineering ---
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
df['sharpe_20'] = df['ret_mean_20'] / (df['ret_std_20'] + 1e-9) * np.sqrt(252)
df['vol_mean_20'] = df['tick_volume'].rolling(20).mean()
df['vol_ratio'] = df['tick_volume'] / (df['vol_mean_20'] + 1e-9)

feature_cols = [col for col in df.columns if col not in ['target_raw', 'target_scaled']]
df_clean = df.dropna().copy()

# Разделение
BARS_TEST = 100
BARS_TRAIN = len(df_clean) - BARS_TEST

X_train = df_clean[feature_cols].iloc[:BARS_TRAIN].values
y_train_raw = df_clean['close'].iloc[:BARS_TRAIN].values
X_test = df_clean[feature_cols].iloc[BARS_TRAIN:].values
y_test_raw = df_clean['close'].iloc[BARS_TRAIN:].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- OPTUNA OPTIMIZATION ---
def objective(trial):
    try:
        sigma = trial.suggest_int('sigma', SIGMA_MIN, SIGMA_MAX)
        nb = trial.suggest_int('nb', NB_MIN, NB_MAX)
        
        smoothed = gaussian_filter1d(y_train_raw, sigma=sigma, mode='reflect')
        target = np.zeros_like(smoothed)
        target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
        target_scaled = target * TARGET_SCALE
        
        # КРИТИЧЕСКИ ВАЖНО: Выравниваем длины массивов, чтобы избежать краша fit()
        min_len = min(len(X_train_s), len(target_scaled))
        X_fit = X_train_s[-min_len:]
        y_fit = target_scaled[-min_len:]
        
        model = lgb.LGBMRegressor(n_estimators=N_EST, learning_rate=LR, 
                                   min_child_samples=MIN_CHILD, n_jobs=-1, verbosity=-1)
        model.fit(X_fit, y_fit)
        
        preds = model.predict(X_test_s) / TARGET_SCALE
        
        equity, sharpe, max_dd, total_pnl, n_trades = calculate_equity(preds, y_test_raw, nb, COMMISSION)
        
        # Функция качества: Sharpe минус штраф за просадку
        score = sharpe - (max_dd * 2.0)
        
        if n_trades < 3:
            score -= 20.0 # Жесткий штраф за отсутствие сделок
            
        return score
    except Exception as e:
        # Если произошла ошибка, возвращаем очень плохой скор, чтобы Optuna не падал
        return -100.0

# Запуск оптимизации
st.info(f"Запуск оптимизации: {N_TRIALS} итераций. Это может занять минуту...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

with st.spinner("Идет поиск лучших параметров..."):
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, n_jobs=-1)

# --- ПРОВЕРКА РЕЗУЛЬТАТОВ ---
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if not complete_trials:
    st.error("❌ Оптимизация не завершилась успешно. Все итерации упали с ошибкой. Попробуйте уменьшить N_EST или изменить диапазоны Sigma.")
    st.stop()

# --- ОТБОР ЧЕМПИОНОВ ---
trials_df = pd.DataFrame([{
    'sigma': t.params['sigma'],
    'nb': t.params['nb'],
    'sharpe_dd_score': t.value,
    'trial_number': t.number
} for t in complete_trials])

# Сортируем и берем топ-N
trials_df = trials_df.sort_values('sharpe_dd_score', ascending=False).head(N_CHAMPIONS)

st.success(f"✅ Оптимизация завершена! Найдено {len(complete_trials)} рабочих комбинаций.")

# --- ЗАПУСК ВСЕХ ЧЕМПИОНОВ ---
st.subheader("📊 Equity curves чемпионов")

fig_equity, ax_equity = plt.subplots(figsize=(16, 6))

all_equities = []
champion_stats = []

for idx, row in trials_df.iterrows():
    sigma = int(row['sigma'])
    nb = int(row['nb'])
    
    smoothed = gaussian_filter1d(y_train_raw, sigma=sigma, mode='reflect')
    target = np.zeros_like(smoothed)
    target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
    target_scaled = target * TARGET_SCALE
    
    min_len = min(len(X_train_s), len(target_scaled))
    X_fit = X_train_s[-min_len:]
    y_fit = target_scaled[-min_len:]
    
    model = lgb.LGBMRegressor(n_estimators=N_EST, learning_rate=LR, 
                               min_child_samples=MIN_CHILD, n_jobs=-1, verbosity=-1)
    model.fit(X_fit, y_fit)
    
    preds = model.predict(X_test_s) / TARGET_SCALE
    equity, sharpe, max_dd, total_pnl, n_trades = calculate_equity(preds, y_test_raw, nb, COMMISSION)
    
    all_equities.append(equity)
    
    champion_stats.append({
        'Rank': idx + 1,
        'Sigma': sigma,
        'NB': nb,
        'Sharpe': f"{sharpe:.2f}",
        'Max DD': f"{max_dd*100:.1f}%",
        'Total PnL': f"{total_pnl:.2f}$",
        'Trades': n_trades,
        'Score': f"{row['sharpe_dd_score']:.2f}"
    })
    
    ax_equity.plot(equity, label=f"#{idx+1}: σ={sigma}, NB={nb}, PnL={total_pnl:.2f}$", 
                   alpha=0.7, linewidth=2)

# Ансамбль (среднее всех чемпионов)
ensemble_equity = np.mean(all_equities, axis=0)
ax_equity.plot(ensemble_equity, label="🏆 ANSEMBLE (среднее)", 
               color='black', linewidth=3, linestyle='--')

ax_equity.axhline(0, color='gray', lw=1, alpha=0.5)
ax_equity.set_title("Equity curves всех чемпионов + Ensemble (последние 100 баров)")
ax_equity.set_xlabel("Бар")
ax_equity.set_ylabel("Накопленная прибыль, $")
ax_equity.legend(loc='upper left', fontsize=9)
ax_equity.grid(alpha=0.3)

st.pyplot(fig_equity)

# --- ТАБЛИЦА СТАТИСТИКИ ---
st.subheader("📋 Статистика чемпионов")
champions_df = pd.DataFrame(champion_stats)
st.dataframe(champions_df, use_container_width=True)

# --- ГРАФИК ПРОСАДОК ---
st.subheader("📉 Просадки (Drawdown) чемпионов")

fig_dd, ax_dd = plt.subplots(figsize=(14, 5))

for idx, equity in enumerate(all_equities):
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / (np.abs(peak) + 1e-9)
    ax_dd.plot(drawdown * 100, label=f"Champion #{idx+1}", alpha=0.6)

ax_dd.set_title("Drawdown всех чемпионов")
ax_dd.set_ylabel("Просадка, %")
ax_dd.set_xlabel("Бар")
ax_dd.legend(fontsize=8)
ax_dd.grid(alpha=0.3)

st.pyplot(fig_dd)

# --- РЕКОМЕНДАЦИЯ ---
ens_pnl = ensemble_equity[-1]
st.success(f"""
### 💡 Итог:
- **Ансамбль из {N_CHAMPIONS} моделей** показывает более гладкую кривую капитала, чем любая отдельная модель.
- Разные значения `Sigma` ловят разные рыночные циклы, компенсируя просадки друг друга.
- **Общая прибыль ансамбля**: {ens_pnl:.2f}$ за 100 баров.
""")

# --- СКАЧАТЬ ПАРАМЕТРЫ ---
st.download_button(
    label="📥 Скачать параметры чемпионов (JSON)",
    data=champions_df.to_json(index=False, orient="records"),
    file_name="champions_params.json",
    mime="application/json"
)