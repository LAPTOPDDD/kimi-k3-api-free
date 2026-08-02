"""
Интерактивный аппроксиматор LightGBM (MT5, EURUSD M60) — V2
============================================================

Исследовательский инструмент (НЕ торговый бот):
LightGBM НЕ прогнозирует цену, а учится аппроксимировать производный сигнал:
    ПП = неказуальная симметричная гауссова производная Close
         (scipy.ndimage.gaussian_filter1d, order=1, mode='reflect').

Математически строгая схема (без утечек данных в признаках):

1. Цель (учитель):
   * gaussian_filter1d(close, sigma, order=1, mode='reflect') — симметричное
     ядро видит БУДУЩИЕ бары внутри окна гауссиана; это «идеальный» сигнал,
     недоступный в реальном времени. Look-ahead есть ТОЛЬКО в цели.
   * Вычисляется ОДИН раз на всём датасете. mode='reflect' отражает ряд
     на границах, поэтому учитель определён на всех барах.

2. Признаки (вход модели):
   * ТОЛЬКО прошлое: лаги Close(t-1..t-N), лаги доходностей и волатильности.
   * Никаких будущих данных в признаках. Никогда.

3. Обучение:
   * LightGBM обучается на первых BARS_TRAIN валидных samples.
   * Цель — неказуальная производная, признаки — каузальные лаги.

4. Инференс (строгий скользящий прогноз):
   * Предсказание начинается с бара BARS_TRAIN + LAGS в координатах исходного
     ряда (первые бары уходят на прогрев лагов).
   * На каждом шаге t модель получает только [t-1 ... t-LAGS].
   * predict() вызывается СТРОГО по одному бару в цикле
     (НЕ на всей матрице сразу) — имитация реального времени.

5. Бэктест (эквити на сегменте инференса):
   * pred > 0 -> LONG, pred < 0 -> SHORT (pred == 0 -> пропуск бара).
   * Вход по Close(t): признаки сигнала известны к закрытию бара t-1,
     поэтому сделка по Close(t) не заглядывает в будущее.
   * Позиция удерживается NB баров (слайдер), выход по Close(t+NB).
   * Одна позиция за раз: сигналы во время открытой позиции игнорируются.
   * Эквити — компаунд с 1.0, mark-to-market внутри сделки.
   * Комиссии и проскальзывание не учитываются.

Запуск:
    pip install -r requirements.txt
    streamlit run lgbm_pp_approximator_V2.py

Режим MT5 требует: Windows + запущенный терминал MetaTrader 5.
Без MT5 доступен демо-режим на синтетических данных.
"""

import io

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

# ---------------------------------------------------------------------------
# Константы проекта
# ---------------------------------------------------------------------------
SEED = 42
BARS_TRAIN = 1000   # обучающий сегмент (баров)
BARS_TEST = 150     # тестовый сегмент (баров, строго после train)
FETCH_BARS = 3000   # сколько баров запрашиваем (запас на прогрев признаков)
SYMBOL = "EURUSD"

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Загрузка котировок из MetaTrader 5...")
def load_mt5_data(symbol: str, n_bars: int) -> pd.DataFrame:
    """Забирает последние n_bars баров M60 (H1) по символу из MT5."""
    if not MT5_AVAILABLE:
        raise RuntimeError("Пакет MetaTrader5 не установлен (доступен только на Windows).")
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() не удался: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Символ {symbol} не найден в MT5.")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_bars)
    finally:
        mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 вернул пустой набор баров по {symbol} (M60).")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


@st.cache_data(show_spinner=False)
def make_synthetic_data(n_bars: int) -> pd.DataFrame:
    """Детерминированный синтетический ряд (демо-режим без MT5)."""
    rng = np.random.default_rng(SEED)
    rets = rng.normal(0.0, 0.0008, n_bars) + 0.0004 * np.sin(np.arange(n_bars) / 25.0)
    close = 1.10 * np.exp(np.cumsum(rets))
    times = pd.date_range("2023-01-01", periods=n_bars, freq=pd.Timedelta(hours=1))
    return pd.DataFrame({"time": times, "close": close})

# ---------------------------------------------------------------------------
# Целевая переменная и признаки
# ---------------------------------------------------------------------------
def gaussian_derivative(close: np.ndarray, sigma: float, truncate: float) -> np.ndarray:
    """Неказуальная симметричная гауссова производная — «идеальный» учитель.

    scipy.ndimage.gaussian_filter1d с order=1: свёртка с первой производной
    гауссиана = первая производная сглаженного ряда. Симметричное ядро видит
    будущие бары внутри окна; mode='reflect' отражает ряд на границах, поэтому
    учитель определён на ВСЕХ барах. Вычисляется один раз на всём датасете.
    """
    return gaussian_filter1d(close, sigma=sigma, order=1, mode="reflect", truncate=truncate)


@st.cache_data(show_spinner=False)
def build_dataset(df, max_lag, vol_window, gauss_sigma, gauss_truncate, norm_mode):
    """Строит признаки/цель на всей истории, затем режет по времени.

    Признаки — только каузальные лаги (lag >= 1). Цель — неказуальная
    гауссова производная (scipy, mode='reflect'), посчитанная один раз
    на всём датасете. Дополнительно возвращает Close тестового сегмента
    (нужен для бэктеста; в признаки НЕ входит).

    Возвращает dict с train/test или None, если истории не хватает.
    """
    close = df["close"].astype(float)
    returns = close.pct_change()
    volatility = returns.rolling(vol_window).std()

    # учитель: один раз на всём датасете
    target_raw = pd.Series(
        gaussian_derivative(close.to_numpy(), gauss_sigma, gauss_truncate),
        index=df.index,
    )

    feats = {}
    for lag in range(1, max_lag + 1):
        feats[f"close_lag{lag}"] = close.shift(lag)
        feats[f"ret_lag{lag}"] = returns.shift(lag)
        feats[f"vol_lag{lag}"] = volatility.shift(lag)
    X = pd.DataFrame(feats, index=df.index)
    feature_cols = list(X.columns)

    full = pd.concat([df["time"], df["close"], X, target_raw.rename("target")], axis=1)
    full = full.dropna().reset_index(drop=True)

    if len(full) < BARS_TRAIN + BARS_TEST:
        return None

    # строгий временной сплит: train -> test (test строго позже)
    full = full.iloc[-(BARS_TRAIN + BARS_TEST):].reset_index(drop=True)

    # нормализация ПП по статистикам ТОЛЬКО train-сегмента
    y = full["target"].to_numpy(dtype=float)
    y_train_raw = y[:BARS_TRAIN]
    if norm_mode == "z-score":
        mu = y_train_raw.mean()
        sd = y_train_raw.std(ddof=0)
        full["target_norm"] = (y - mu) / (sd if sd > 0 else 1.0)
    elif norm_mode == "min-max":
        lo, hi = y_train_raw.min(), y_train_raw.max()
        full["target_norm"] = (y - lo) / (hi - lo if hi > lo else 1.0)
    else:
        full["target_norm"] = y

    train = full.iloc[:BARS_TRAIN]
    test = full.iloc[BARS_TRAIN:]

    return {
        "X_train": train[feature_cols],
        "y_train": train["target_norm"].to_numpy(),
        "X_test": test[feature_cols],
        "y_test": test["target_norm"].to_numpy(),
        "test_close": test["close"].to_numpy(dtype=float),
        "train_time": train["time"].reset_index(drop=True),
        "test_time": test["time"].reset_index(drop=True),
        "n_features": len(feature_cols),
    }

# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, params):
    """Детерминированное обучение LightGBM Regressor."""
    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbose=-1,
        **params,
    )
    model.fit(X_train, y_train)
    return model


def rolling_predict(model, X_test) -> np.ndarray:
    """Строгий скользящий прогноз: predict() вызывается ПО ОДНОМУ бару.

    Намеренно НЕ используем векторный predict на всей матрице: покадровый
    цикл имитирует реальное время — на шаге t модель видит только лаги
    [t-1 ... t-LAGS] и ничего больше.
    """
    preds = np.empty(len(X_test), dtype=float)
    for i in range(len(X_test)):
        preds[i] = model.predict(X_test.iloc[i:i + 1])[0]
    return preds

# ---------------------------------------------------------------------------
# Бэктест (эквити на сегменте инференса)
# ---------------------------------------------------------------------------
def backtest_signals(close: np.ndarray, preds: np.ndarray, hold_bars: int):
    """Event-driven бэктест по знаку прогноза (без пирамидинга).

    Правила:
      * сигнал на баре t: pred > 0 -> LONG, pred < 0 -> SHORT (0 -> пропуск);
      * вход по Close(t) — признаки сигнала известны к закрытию бара t-1,
        поэтому сделка по Close(t) не заглядывает в будущее;
      * позиция удерживается ровно hold_bars баров, выход по Close(t+hold_bars)
        (если история кончается раньше — по Close последнего бара);
      * пока позиция открыта, новые сигналы игнорируются; следующий сигнал
        рассматривается со следующего бара после выхода;
      * доходность сделки: direction * (exit - entry) / entry;
      * эквити — компаунд: equity *= (1 + ret), старт с 1.0; внутри сделки
        эквити переоценивается по рынку (mark-to-market).

    Возвращает (equity_curve, trades).
    """
    n = len(close)
    equity = np.ones(n)
    trades = []
    eq = 1.0
    i = 0
    while i < n:
        if i >= n - 1:  # последний бар: сделку не открываем, эквити фиксируем
            equity[i] = eq
            i += 1
            continue
        p = preds[i]
        if p > 0:
            direction = 1
        elif p < 0:
            direction = -1
        else:
            equity[i] = eq
            i += 1
            continue
        entry = float(close[i])
        exit_idx = min(i + hold_bars, n - 1)
        for k in range(i, exit_idx + 1):  # mark-to-market внутри сделки
            equity[k] = eq * (1.0 + direction * (float(close[k]) - entry) / entry)
        exit_price = float(close[exit_idx])
        ret = direction * (exit_price - entry) / entry
        eq *= 1.0 + ret
        trades.append({
            "entry_idx": i,
            "exit_idx": exit_idx,
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "ret": ret,
        })
        i = exit_idx + 1
    return equity, trades

# ---------------------------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LightGBM ПП-аппроксиматор V2", layout="wide")

st.title("LightGBM-аппроксимация ПП + эквити-бэктест (V2)")
st.caption(
    "Исследовательский инструмент, НЕ торговый бот. "
    "Модель не прогнозирует цену: она аппроксимирует «идеальную» неказуальную "
    "производную сглаженного ряда по каузальным лаговым признакам. "
    "Инференс — строгий покадровый цикл на тестовом сегменте "
    "(150 баров сразу после обучающих 1000). По знаку прогноза строится "
    "эквити: pred > 0 — LONG, pred < 0 — SHORT, удержание NB баров."
)

with st.sidebar:
    st.header("Данные")
    if MT5_AVAILABLE:
        source = st.radio("Источник", ["MT5: EURUSD M60", "Синтетика (демо)"], index=0)
    else:
        st.warning("MetaTrader5 не установлен — доступен только демо-режим.")
        source = "Синтетика (демо)"
    if st.button("🔄 Обновить данные"):
        st.cache_data.clear()
        st.rerun()

    st.header("Признаки и цель")
    max_lag = st.slider("LAGS — макс. лаг", 1, 50, 10)
    vol_window = st.slider("Окно волатильности", 5, 120, 20)
    gauss_sigma = st.slider("Gaussian sigma (учитель)", 0.5, 10.0, 2.0, 0.5)
    gauss_truncate = st.slider(
        "Gaussian truncate (ширина ядра, в σ)", 1.0, 8.0, 4.0, 0.5,
        help="Ядро обрезается на расстоянии truncate * sigma от центра (как в scipy).",
    )
    norm_mode = st.selectbox("Нормализация ПП", ["z-score", "min-max", "без нормализации"])

    st.header("LightGBM")
    num_leaves = st.slider("num_leaves", 2, 256, 31)
    learning_rate = st.slider("learning_rate", 0.01, 0.30, 0.05, 0.01)
    n_estimators = st.slider("n_estimators", 50, 2000, 400, 50)
    max_depth = st.slider("max_depth (-1 = без лимита)", -1, 32, -1)
    min_data_in_leaf = st.slider("min_data_in_leaf", 1, 200, 20)
    feature_fraction = st.slider("feature_fraction", 0.3, 1.0, 0.9, 0.05)
    bagging_fraction = st.slider("bagging_fraction", 0.3, 1.0, 0.8, 0.05)
    lambda_l1 = st.slider("lambda_l1", 0.0, 10.0, 0.0, 0.1)
    lambda_l2 = st.slider("lambda_l2", 0.0, 10.0, 0.0, 0.1)

    st.header("Бэктест (эквити)")
    hold_bars = st.slider(
        "NB — удержание позиции (баров)", 1, 20, 2,
        help="Позиция открывается по Close(t) и закрывается по Close(t+NB). "
             "Одна позиция за раз: сигналы во время открытой позиции игнорируются.",
    )

lgbm_params = dict(
    num_leaves=num_leaves,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_child_samples=min_data_in_leaf,
    colsample_bytree=feature_fraction,
    subsample=bagging_fraction,
    subsample_freq=1,
    reg_alpha=lambda_l1,
    reg_lambda=lambda_l2,
)

# --- загрузка данных
try:
    if source.startswith("MT5"):
        raw = load_mt5_data(SYMBOL, FETCH_BARS)
    else:
        raw = make_synthetic_data(FETCH_BARS)
except Exception as exc:
    st.error(f"Ошибка загрузки данных: {exc}")
    st.stop()

# --- построение датасета
dataset = build_dataset(raw, max_lag, vol_window, gauss_sigma, gauss_truncate, norm_mode)
if dataset is None:
    st.error(
        "Недостаточно истории для выбранных окон/лагов. "
        "Уменьшите max_lag / окна или увеличьте FETCH_BARS в коде."
    )
    st.stop()

st.info(
    "Учитель — неказуальная гауссова производная (scipy, mode='reflect'): "
    "симметричное ядро видит будущие бары — это «идеальный» сигнал, а не "
    "торговый сигнал. Признаки — строго каузальные лаги; инференс — строгий "
    "покадровый цикл predict() по одному бару, как в реальном времени. "
    "Эквити ниже — результат простого бэктеста по знаку прогноза "
    "(без комиссий и проскальзывания)."
)

# --- обучение и строгий покадровый прогноз (индикатор загрузки)
with st.spinner("Обучение LightGBM и строгий покадровый прогноз..."):
    model = train_model(dataset["X_train"], dataset["y_train"], lgbm_params)
    y_pred = rolling_predict(model, dataset["X_test"])

# --- метрики аппроксимации на тесте
mae = mean_absolute_error(dataset["y_test"], y_pred)
rmse = float(np.sqrt(mean_squared_error(dataset["y_test"], y_pred)))
r2 = r2_score(dataset["y_test"], y_pred)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE (test)", f"{mae:.5f}")
c2.metric("RMSE (test)", f"{rmse:.5f}")
c3.metric("R² (test)", f"{r2:.4f}")
c4.metric("Признаков", dataset["n_features"])

st.caption(
    f"Источник: {source} | "
    f"Train: {dataset['train_time'].iloc[0]} — {dataset['train_time'].iloc[-1]} ({BARS_TRAIN} баров) | "
    f"Test: {dataset['test_time'].iloc[0]} — {dataset['test_time'].iloc[-1]} ({BARS_TEST} баров) | "
    f"Нормализация: {norm_mode} (по train) | "
    f"Инференс: покадровый цикл ({BARS_TEST} вызовов predict)"
)

# --- график (тестовый сегмент: истина vs покадровый прогноз)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dataset["test_time"], y=dataset["y_test"],
    name="ПП (истина, неказуальный учитель)", line=dict(color="#1f77b4", width=2),
))
fig.add_trace(go.Scatter(
    x=dataset["test_time"], y=y_pred,
    name="Аппроксимация LightGBM (покадровый прогноз)",
    line=dict(color="#ff7f0e", width=2, dash="dash"),
))
fig.add_hline(y=0, line_dash="dot", line_color="gray")
fig.update_layout(
    title="Тестовый сегмент (150 баров): истинная ПП (неказуальный учитель) vs покадровая аппроксимация",
    xaxis_title="Время",
    yaxis_title="ПП (норм.)",
    hovermode="x unified",
    height=540,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, use_container_width=True)

# --- бэктест: эквити по знаку прогноза на сегменте инференса
equity, trades = backtest_signals(dataset["test_close"], y_pred, hold_bars)

n_trades = len(trades)
win_rate = float(np.mean([t["ret"] > 0 for t in trades])) if trades else 0.0
running_max = np.maximum.accumulate(equity)
max_dd = float((equity / running_max - 1.0).min())
total_ret = float(equity[-1] - 1.0)

st.subheader("Эквити на сегменте инференса (бэктест по знаку прогноза)")

b1, b2, b3, b4 = st.columns(4)
b1.metric("Доходность (эквити)", f"{total_ret * 100:.2f}%")
b2.metric("Сделок", n_trades)
b3.metric("Win rate", f"{win_rate * 100:.1f}%")
b4.metric("Макс. просадка", f"{max_dd * 100:.2f}%")

st.caption(
    f"Правила: pred > 0 → LONG, pred < 0 → SHORT | вход по Close(t), "
    f"выход по Close(t+{hold_bars}) | одна позиция за раз | "
    f"компаунд с 1.0 | без комиссий и проскальзывания"
)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=dataset["test_time"], y=equity,
    name="Эквити (компаунд, старт = 1.0)",
    line=dict(color="#2ca02c", width=2),
))
if trades:
    exit_idx = [t["exit_idx"] for t in trades]
    exit_colors = ["#2ca02c" if t["ret"] > 0 else "#d62728" for t in trades]
    exit_text = [
        f"{'LONG' if t['direction'] > 0 else 'SHORT'}: {t['ret'] * 100:+.3f}%"
        for t in trades
    ]
    fig_eq.add_trace(go.Scatter(
        x=dataset["test_time"].iloc[exit_idx], y=equity[exit_idx],
        mode="markers", name="Закрытия сделок",
        marker=dict(color=exit_colors, size=8, symbol="circle"),
        text=exit_text, hovertemplate="%{x}<br>%{text}<br>Эквити: %{y:.4f}<extra></extra>",
    ))
fig_eq.add_hline(y=1.0, line_dash="dot", line_color="gray")
fig_eq.update_layout(
    title=f"Эквити (тестовый сегмент, NB = {hold_bars} бара удержания)",
    xaxis_title="Время",
    yaxis_title="Эквити",
    hovermode="x unified",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig_eq, use_container_width=True)

with st.expander("Журнал сделок (тестовый сегмент)"):
    if trades:
        trades_df = pd.DataFrame({
            "Вход (время)": [dataset["test_time"].iloc[t["entry_idx"]] for t in trades],
            "Выход (время)": [dataset["test_time"].iloc[t["exit_idx"]] for t in trades],
            "Направление": ["LONG" if t["direction"] > 0 else "SHORT" for t in trades],
            "Цена входа": [t["entry"] for t in trades],
            "Цена выхода": [t["exit"] for t in trades],
            "Доходность, %": [t["ret"] * 100 for t in trades],
        })
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.write("Сделок не было (все прогнозы нулевые или сегмент слишком короткий).")

# --- экспорт обученной модели
with st.expander("Экспорт обученной модели"):
    e1, e2 = st.columns(2)
    e1.download_button(
        "Скачать booster (.txt)",
        data=model.booster_.model_to_string(),
        file_name="lgbm_pp_approximator_V2.txt",
        mime="text/plain",
    )
    buf = io.BytesIO()
    joblib.dump(model, buf)
    e2.download_button(
        "Скачать модель (.pkl)",
        data=buf.getvalue(),
        file_name="lgbm_pp_approximator_V2.pkl",
        mime="application/octet-stream",
    )

with st.expander("Методология (защита от look-ahead)"):
    st.markdown(
        """
- **Цель (учитель)** — неказуальная симметричная гауссова производная Close:
  `scipy.ndimage.gaussian_filter1d(close, sigma, order=1, mode='reflect')`.
  Симметричное ядро видит будущие бары внутри окна — это «идеальный» сигнал,
  недоступный в реальном времени. Look-ahead есть **только в цели**, что
  допустимо для исследования. `mode='reflect'` отражает ряд на границах,
  поэтому учитель определён на всех барах и считается **один раз** на всём
  датасете.
- **Признаки** — только лаги `lag >= 1`: Close, Returns, Volatility.
  Никаких будущих данных в признаках. Никогда.
- **Обучение** — LightGBM на первых `BARS_TRAIN` валидных samples.
- **Инференс — строгий скользящий прогноз**: `predict()` вызывается в цикле
  по одному бару (`for i in range(len(X_test))`), а **не** на всей матрице
  сразу. На шаге `t` модель получает только `[t-1 ... t-LAGS]` — имитация
  реального времени. Первый предсказанный бар исходного ряда:
  `BARS_TRAIN + LAGS` (первые бары уходят на прогрев лагов).
- **Бэктест (эквити)** — на сегменте инференса: `pred > 0` → LONG,
  `pred < 0` → SHORT; вход по `Close(t)` (сигнал известен к закрытию бара
  `t-1` — без look-ahead), удержание `NB` баров, выход по `Close(t+NB)`;
  одна позиция за раз (сигналы во время открытой позиции игнорируются);
  доходность сделки `direction * (exit - entry) / entry`; эквити — компаунд
  с 1.0, mark-to-market внутри сделки; комиссии и проскальзывание не
  учитываются. Знак прогноза инвариантен к режиму нормализации ПП
  (z-score/min-max — монотонные преобразования).
- **Нормализация ПП** — среднее/σ (или min/max) считаются только по train-сегменту.
- **Сплит по времени**: train 1000 баров → test 150 баров строго позже.
- **Воспроизводимость**: фиксированный seed, `deterministic=True`, `n_jobs=1`.
        """
    )
