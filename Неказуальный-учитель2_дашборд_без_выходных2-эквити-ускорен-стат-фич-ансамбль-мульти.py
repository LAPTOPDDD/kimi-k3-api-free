import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import optuna
import matplotlib.pyplot as plt
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# --- Конфигурация ---
st.set_page_config(page_title="PRO ENSEMBLE TRADER", layout="wide")
st.title(" PRO ENSEMBLE - Оптимизация с кнопкой запуска")

# --- Инициализация session_state ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'optimizing' not in st.session_state:
    st.session_state.optimizing = False

# ======================== ФУНКЦИИ ========================

def calculate_equity(preds, close, nb, commission=0.5, slippage=0.5):
    """Симуляция торговли"""
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
                exit_price = price - (slippage * 0.0001 * position)
                pnl = (exit_price - entry_price) * position - commission
                equity.append(equity[-1] + pnl)
                trades_count += 1
                position, bars_held = 0, 0
            else:
                equity.append(equity[-1])
        else:
            equity.append(equity[-1])
            if preds[t] > 0:
                position = 1
                entry_price = price + slippage * 0.0001
                bars_held = 0
            elif preds[t] < 0:
                position = -1
                entry_price = price - slippage * 0.0001
                bars_held = 0
    
    equity = np.array(equity[1:])
    returns = np.diff(equity, prepend=0)
    
    if len(returns) > 0 and np.std(returns) > 1e-9:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = -10.0
    
    if len(equity) > 0:
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (np.abs(peak) + 1e-9)
        max_dd = np.max(drawdown)
    else:
        max_dd = 1.0
    
    total_pnl = equity[-1] if len(equity) > 0 else 0
    
    return equity, sharpe, max_dd, total_pnl, trades_count

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

def prepare_features(df):
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
    return df

def optimize_pair(symbol, df, feature_cols, n_trials, bars_test=100,
                  SIGMA_MIN=2, SIGMA_MAX=15, NB_MIN=3, NB_MAX=18,
                  COMMISSION=0.5, SLIPPAGE=0.5, TARGET_SCALE=10000,
                  progress_bar=None, status_text=None):
    """Оптимизация одной пары с прогресс-баром"""
    df_clean = df.dropna().copy()
    close = df_clean['close'].values
    
    BARS_TRAIN = len(df_clean) - bars_test
    X_train = df_clean[feature_cols].iloc[:BARS_TRAIN].values
    y_train_raw = close[:BARS_TRAIN]
    X_test = df_clean[feature_cols].iloc[BARS_TRAIN:].values
    y_test_raw = close[BARS_TRAIN:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    trials_count = [0]
    
    def objective(trial):
        try:
            sigma = trial.suggest_int('sigma', SIGMA_MIN, SIGMA_MAX)
            nb = trial.suggest_int('nb', NB_MIN, NB_MAX)
            n_est_local = trial.suggest_int('n_estimators', 100, 400, step=50)
            lr_local = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
            min_child_local = trial.suggest_int('min_child_samples', 15, 60, step=5)
            
            smoothed = gaussian_filter1d(y_train_raw, sigma=sigma, mode='reflect')
            target = np.zeros_like(smoothed)
            target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
            target_scaled = target * TARGET_SCALE
            
            min_len = min(len(X_train_s), len(target_scaled))
            X_fit = X_train_s[-min_len:]
            y_fit = target_scaled[-min_len:]
            
            model = lgb.LGBMRegressor(
                n_estimators=n_est_local,
                learning_rate=lr_local,
                min_child_samples=min_child_local,
                n_jobs=-1,
                verbosity=-1,
                random_state=trial.number
            )
            model.fit(X_fit, y_fit)
            
            preds = model.predict(X_test_s) / TARGET_SCALE
            equity, sharpe, max_dd, total_pnl, n_trades = calculate_equity(
                preds, y_test_raw, nb, COMMISSION, SLIPPAGE
            )
            
            score = sharpe - (max_dd * 3.0)
            if n_trades < 5:
                score -= 30.0
            
            trials_count[0] += 1
            if progress_bar:
                progress_bar.progress(trials_count[0] / n_trials)
            if status_text:
                status_text.text(f"🔍 {symbol}: trial {trials_count[0]}/{n_trials} | score={score:.2f}")
            
            return score
        except:
            return -100.0
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=-1)
    
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not complete_trials:
        return None
    
    trials_df = pd.DataFrame([{
        'sigma': t.params['sigma'],
        'nb': t.params['nb'],
        'n_estimators': t.params['n_estimators'],
        'learning_rate': t.params['learning_rate'],
        'min_child_samples': t.params['min_child_samples'],
        'score': t.value
    } for t in complete_trials])
    
    return trials_df

# ======================== SIDEBAR ========================

st.sidebar.header("⚙️ Параметры оптимизации")
N_CHAMPIONS = st.sidebar.slider("Кол-во чемпионов на пару", 5, 30, 15, step=1)
N_TRIALS = st.sidebar.slider("Итерации Optuna на пару", 20, 150, 50, step=10)

st.sidebar.subheader(" Диапазоны параметров")
SIGMA_MIN = st.sidebar.slider("Sigma min", 1, 10, 2, step=1)
SIGMA_MAX = st.sidebar.slider("Sigma max", 10, 25, 15, step=1)
NB_MIN = st.sidebar.slider("NB min (удержание)", 2, 10, 3, step=1)
NB_MAX = st.sidebar.slider("NB max (удержание)", 10, 30, 18, step=1)

st.sidebar.subheader("🤖 LightGBM")
N_EST = st.sidebar.slider("n_estimators (default)", 100, 500, 250, step=50)
LR = st.sidebar.number_input("learning_rate (default)", 0.01, 0.2, 0.05, step=0.01)
MIN_CHILD = st.sidebar.slider("min_child_samples (default)", 10, 100, 30, step=5)

st.sidebar.subheader("💰 Торговля")
LOT_SIZE = st.sidebar.number_input("Размер лота", 1, 100, 10, step=5)
COMMISSION = st.sidebar.number_input("Комиссия за сделку ($)", 0.0, 5.0, 0.5, step=0.1)
SLIPPAGE = st.sidebar.number_input("Проскальзывание (пипсы)", 0.0, 5.0, 0.5, step=0.1)
TARGET_SCALE = st.sidebar.selectbox("Масштаб цели", [1000, 5000, 10000, 50000], index=2)

st.sidebar.subheader("🌍 Валютные пары")
PAIRS = st.sidebar.multiselect(
    "Выберите пары",
    ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"],
    default=["EURUSD", "GBPUSD", "USDJPY"]
)

# ======================== КНОПКИ УПРАВЛЕНИЯ ========================

st.sidebar.markdown("---")
st.sidebar.subheader("🎮 Управление")

# Кнопка загрузки данных
load_btn = st.sidebar.button("📥 Загрузить данные", use_container_width=True)

# Кнопка запуска оптимизации
optimize_btn = st.sidebar.button("🔄 ПЕРЕОБУЧИТЬ", type="primary", use_container_width=True)

# Кнопка сброса
if st.sidebar.button("🗑️ Сбросить результаты", use_container_width=True):
    st.session_state.results = None
    st.rerun()

# ======================== ЗАГРУЗКА ДАННЫХ ========================

if load_btn or st.session_state.results is None:
    if not PAIRS:
        st.warning("⚠️ Выберите хотя бы одну валютную пару")
        st.stop()
    
    with st.spinner(f"Загрузка данных для {len(PAIRS)} пар..."):
        all_data = {}
        for pair in PAIRS:
            df = get_mt5_data(pair, mt5.TIMEFRAME_H1, 10000)
            if df is not None:
                df = df[df.index.dayofweek < 5].copy()
                all_data[pair] = prepare_features(df)
    
    if not all_data:
        st.error("❌ Не удалось загрузить данные")
        st.stop()
    
    st.session_state.all_data = all_data
    st.success(f"✅ Загружены данные: {', '.join(all_data.keys())}")
else:
    all_data = st.session_state.get('all_data', {})

# ======================== ОПТИМИЗАЦИЯ (ТОЛЬКО ПО КНОПКЕ) ========================

if optimize_btn:
    if not PAIRS:
        st.warning("⚠️ Выберите хотя бы одну валютную пару")
        st.stop()
    
    if not all_data:
        st.error("❌ Сначала загрузите данные")
        st.stop()
    
    st.session_state.optimizing = True
    
    # Прогресс-бар
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_champions = {}
    total_pairs = len(all_data)
    
    for pair_idx, (pair, df) in enumerate(all_data.items()):
        feature_cols = [col for col in df.columns if col not in ['target_raw', 'target_scaled']]
        
        status_text.text(f"🔍 Оптимизация {pair} ({pair_idx+1}/{total_pairs})...")
        
        champions = optimize_pair(
            pair, df, feature_cols, N_TRIALS,
            SIGMA_MIN=SIGMA_MIN, SIGMA_MAX=SIGMA_MAX,
            NB_MIN=NB_MIN, NB_MAX=NB_MAX,
            COMMISSION=COMMISSION, SLIPPAGE=SLIPPAGE,
            TARGET_SCALE=TARGET_SCALE,
            progress_bar=progress_bar,
            status_text=status_text
        )
        
        if champions is not None:
            # Берем топ-N
            champions = champions.sort_values('score', ascending=False).head(N_CHAMPIONS)
            all_champions[pair] = champions
        
        # Обновляем общий прогресс
        overall_progress = (pair_idx + 1) / total_pairs
        progress_bar.progress(overall_progress)
    
    # Сохраняем результаты в session_state
    st.session_state.results = {
        'champions': all_champions,
        'data': all_data,
        'params': {
            'N_CHAMPIONS': N_CHAMPIONS,
            'LOT_SIZE': LOT_SIZE,
            'COMMISSION': COMMISSION,
            'SLIPPAGE': SLIPPAGE,
            'TARGET_SCALE': TARGET_SCALE
        }
    }
    
    st.session_state.optimizing = False
    status_text.text("✅ Оптимизация завершена!")
    progress_bar.progress(1.0)
    
    st.rerun()

# ======================== ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ========================

if st.session_state.results is None:
    st.info("👈 Настройте параметры в сайдбаре и нажмите **🔄 ПЕРЕОБУЧИТЬ**")
    st.stop()

# Загружаем результаты из session_state
results = st.session_state.results
all_champions = results['champions']
all_data = results['data']
params = results['params']

if not all_champions:
    st.error(" Оптимизация не дала результатов. Попробуйте изменить диапазоны параметров.")
    st.stop()

st.success(f"✅ Найдено чемпионов: {sum(len(c) for c in all_champions.values())}")

# --- ЗАПУСК ВСЕХ ЧЕМПИОНОВ ---
st.subheader("📊 Equity Curves всех пар и чемпионов")

fig_main, ax_main = plt.subplots(figsize=(16, 7))

total_equity = None
all_stats = []

LOT_SIZE = params['LOT_SIZE']
COMMISSION = params['COMMISSION']
SLIPPAGE = params['SLIPPAGE']
TARGET_SCALE = params['TARGET_SCALE']

for pair_idx, (pair, champions) in enumerate(all_champions.items()):
    df = all_data[pair]
    feature_cols = [col for col in df.columns if col not in ['target_raw', 'target_scaled']]
    df_clean = df.dropna().copy()
    close = df_clean['close'].values
    
    BARS_TEST = 100
    BARS_TRAIN = len(df_clean) - BARS_TEST
    y_test_raw = close[BARS_TRAIN:]
    
    pair_equity_sum = np.zeros(BARS_TEST)
    
    # Сбрасываем индексы чемпионов, чтобы iloc работал корректно
    champions_reset = champions.reset_index(drop=True)
    
    for champion_idx in range(len(champions_reset)):
        row = champions_reset.iloc[champion_idx]
        sigma = int(row['sigma'])
        nb = int(row['nb'])
        n_est_local = int(row['n_estimators'])
        lr_local = float(row['learning_rate'])
        min_child_local = int(row['min_child_samples'])
        score = float(row['score'])
        
        y_train_raw = close[:BARS_TRAIN]
        smoothed = gaussian_filter1d(y_train_raw, sigma=sigma, mode='reflect')
        target = np.zeros_like(smoothed)
        target[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
        target_scaled = target * TARGET_SCALE
        
        X_train = df_clean[feature_cols].iloc[:BARS_TRAIN].values
        X_test = df_clean[feature_cols].iloc[BARS_TRAIN:].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        min_len = min(len(X_train_s), len(target_scaled))
        X_fit = X_train_s[-min_len:]
        y_fit = target_scaled[-min_len:]
        
        model = lgb.LGBMRegressor(
            n_estimators=n_est_local,
            learning_rate=lr_local,
            min_child_samples=min_child_local,
            n_jobs=-1,
            verbosity=-1,
            random_state=champion_idx
        )
        model.fit(X_fit, y_fit)
        
        preds = model.predict(X_test_s) / TARGET_SCALE
        equity, sharpe, max_dd, total_pnl, n_trades = calculate_equity(
            preds, y_test_raw, nb, COMMISSION, SLIPPAGE
        )
        
        equity_scaled = equity * LOT_SIZE
        pair_equity_sum += equity_scaled
        
        # СОХРАНЯЕМ ВСЕ ПАРАМЕТРЫ СРАЗУ — не нужно лезть в all_champions потом!
        all_stats.append({
            'Pair': pair,
            'Champion': champion_idx + 1,
            'Sigma': sigma,
            'NB': nb,
            'n_estimators': n_est_local,
            'learning_rate': lr_local,
            'min_child_samples': min_child_local,
            'Sharpe': float(sharpe),
            'Max DD': float(max_dd * 100),
            'PnL': float(total_pnl * LOT_SIZE),
            'Trades': int(n_trades),
            'Score': score
        })
        
        if champion_idx < 3:
            ax_main.plot(equity_scaled, alpha=0.5, linewidth=1,
                        label=f"{pair} #{champion_idx+1}" if pair_idx == 0 else "")
    
    ax_main.plot(pair_equity_sum, linewidth=2.5,
                label=f"{pair} TOTAL", linestyle='--')
    
    if total_equity is None:
        total_equity = pair_equity_sum
    else:
        total_equity += pair_equity_sum

ax_main.plot(total_equity, color='black', linewidth=3, linestyle='-',
            label="🏆 TOTAL PORTFOLIO")

ax_main.axhline(0, color='gray', lw=1, alpha=0.5)
ax_main.set_title(f"Equity Curves: {len(all_champions)} пар × {N_CHAMPIONS} чемпионов | Лот: {LOT_SIZE}")
ax_main.set_xlabel("Бар (из 100)")
ax_main.set_ylabel("Накопленная прибыль, $")
ax_main.legend(loc='upper left', fontsize=8)
ax_main.grid(alpha=0.3)

st.pyplot(fig_main)
# --- СТАТИСТИКА ---
st.subheader("📋 Статистика всех чемпионов")
stats_df = pd.DataFrame(all_stats)

# Форматируем для отображения
display_df = stats_df.copy()
display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
display_df['Max DD'] = display_df['Max DD'].apply(lambda x: f"{x:.1f}%")
display_df['PnL'] = display_df['PnL'].apply(lambda x: f"{x:.2f}$")
display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.2f}")

st.dataframe(display_df, use_container_width=True)

# --- ИТОГОВЫЕ МЕТРИКИ (теперь работают корректно!) ---
st.subheader("📊 Итоговые результаты")

col1, col2, col3, col4 = st.columns(4)

total_pnl = total_equity[-1]
total_trades = int(stats_df['Trades'].sum())
avg_sharpe = float(stats_df['Sharpe'].mean())

if len(total_equity) > 0:
    peak = np.maximum.accumulate(total_equity)
    drawdown = (peak - total_equity) / (np.abs(peak) + 1e-9)
    max_dd_portfolio = float(np.max(drawdown) * 100)
else:
    max_dd_portfolio = 0.0

col1.metric("Total PnL", f"{total_pnl:.2f}$")
col2.metric("Всего сделок", total_trades)
col3.metric("Средний Sharpe", f"{avg_sharpe:.2f}")
col4.metric("Max DD портфеля", f"{max_dd_portfolio:.1f}%")

# --- СОХРАНЕНИЕ В JSON ---
# --- СОХРАНЕНИЕ В JSON ---
st.subheader(" Сохранение параметров для бота")

champions_list = []
magic_base = 100000

for idx, row in stats_df.iterrows():
    champion = {
        'symbol': row['Pair'],
        'timeframe': 'H1',
        'magic': magic_base + idx,
        'sigma': int(row['Sigma']),
        'nb': int(row['NB']),
        'n_estimators': int(row['n_estimators']),
        'learning_rate': float(row['learning_rate']),
        'min_child_samples': int(row['min_child_samples']),
        'target_scale': TARGET_SCALE,
        'lot_size': LOT_SIZE,
        'commission': COMMISSION,
        'slippage': SLIPPAGE,
        'sharpe': float(row['Sharpe']),
        'max_dd': float(row['Max DD']),
        'pnl': float(row['PnL']),
        'score': float(row['Score'])
    }
    champions_list.append(champion)

json_data = json.dumps(champions_list, indent=2, ensure_ascii=False)

st.code(json_data[:500] + "\n..." if len(json_data) > 500 else json_data, language='json')

st.download_button(
    label="📥 Скачать champions.json",
    data=json_data,
    file_name=f"champions_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
    mime="application/json",
    use_container_width=True
)