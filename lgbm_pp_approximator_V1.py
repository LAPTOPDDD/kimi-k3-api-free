"""
Интерактивный аппроксиматор LightGBM (MT5, EURUSD M60)
=======================================================

Исследовательский инструмент (НЕ торговый бот):
LightGBM НЕ прогнозирует цену, а учится аппроксимировать производный сигнал:
    ПП = первая производная Close после гауссовой фильтрации.

Два режима учителя (целевой ПП):
  * каузальный   — односторонний гауссов фильтр (только текущий и прошлые бары);
  * неказуальный — симметричный гауссов фильтр (использует и БУДУЩИЕ бары).
                   Это «идеальный» учитель: в реальном времени он недоступен,
                   но модель на лаговых признаках учится восстанавливать его
                   из прошлых данных.

Ключевой эффект неказуального учителя — «край» истории:
  на последних window//2 барах истинная неказуальная ПП ещё не вычислима
  (нужны будущие бары), но обученная модель получает на вход каузальные
  фичи из прошлого и выдаёт оценку «идеальной» ПП в реальном времени —
  именно так она и обучалась. Эти бары показаны на графике отдельно.

Защита от look-ahead bias (в обоих режимах):
  * признаки — только лаги (lag >= 1) Close / Returns / Volatility;
  * нормализация ПП — по статистикам ТОЛЬКО обучающего сегмента;
  * строгий временной сплит: train (1000 баров) -> test (150 баров сразу после).

Запуск:
    pip install -r requirements.txt
    streamlit run lgbm_pp_approximator_V1.py

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
N_TRAIN = 1000      # обучающий сегмент (баров)
N_TEST = 150        # тестовый сегмент (баров, строго после train)
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
def causal_gaussian_smooth(values: np.ndarray, sigma: float, window: int) -> np.ndarray:
    """Каузальная (односторонняя) гауссова фильтрация.

    Вес с индексом 0 относится к текущему бару, k > 0 — к прошлым барам.
    Будущие значения ряда НЕ используются (нет look-ahead).
    Первые window-1 значений не имеют достаточной истории -> NaN.
    """
    k = np.arange(window)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    # smoothed[t] = sum_k kernel[k] * values[t-k]
    smoothed = np.convolve(values, kernel, mode="full")[: len(values)]
    smoothed[: window - 1] = np.nan
    return smoothed


def noncausal_gaussian_smooth(values: np.ndarray, sigma: float, window: int) -> np.ndarray:
    """Неказуальная (симметричная) гауссова фильтрация — «идеальный» учитель.

    Ядро центрировано на текущем баре: используются и ПРОШЛЫЕ, и БУДУЩИЕ бары.
    Такой сигнал недоступен в реальном времени, но может служить учителем:
    модель на лаговых признаках учится восстанавливать «идеальную» ПП.
    Первые и последние half = window // 2 значений -> NaN (нет полного контекста).
    """
    half = window // 2
    k = np.arange(-half, half + 1)  # симметричное ядро нечётной длины
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    # smoothed[t] = sum_k kernel[k] * values[t+k]  (есть заглядывание вперёд)
    smoothed = np.convolve(values, kernel, mode="same")
    if half > 0:
        smoothed[:half] = np.nan
        smoothed[-half:] = np.nan
    return smoothed


@st.cache_data(show_spinner=False)
def build_dataset(df, max_lag, vol_window, gauss_sigma, gauss_window, norm_mode, teacher_mode):
    """Строит признаки/цель на всей истории, затем режет по времени.

    Признаки — всегда каузальные (только лаги). Цель (учитель) — каузальная
    или неказуальная ПП в зависимости от teacher_mode.

    Дополнительно возвращает «край» истории (X_edge / edge_time): бары, где
    признаки валидны, но учитель недоступен (только неказуальный режим —
    последние gauss_window//2 баров). Именно здесь обученная модель выдаёт
    оценку «идеальной» ПП по каузальным признакам в реальном времени.

    Возвращает dict с train/test/edge или None, если истории не хватает.
    """
    close = df["close"].astype(float)
    returns = close.pct_change()
    volatility = returns.rolling(vol_window).std()

    if teacher_mode == "неказуальный":
        smoothed = noncausal_gaussian_smooth(close.to_numpy(), gauss_sigma, gauss_window)
    else:
        smoothed = causal_gaussian_smooth(close.to_numpy(), gauss_sigma, gauss_window)
    target_raw = pd.Series(smoothed, index=df.index).diff()  # первая производная ПП

    feats = {}
    for lag in range(1, max_lag + 1):
        feats[f"close_lag{lag}"] = close.shift(lag)
        feats[f"ret_lag{lag}"] = returns.shift(lag)
        feats[f"vol_lag{lag}"] = volatility.shift(lag)
    X = pd.DataFrame(feats, index=df.index)
    feature_cols = list(X.columns)

    full = pd.concat([df["time"], X, target_raw.rename("target")], axis=1)

    # «край» истории: признаки валидны, но учитель NaN (неказуальный режим —
    # последние gauss_window//2 баров; в каузальном режиме таких баров нет).
    edge = full[full["target"].isna() & full[feature_cols].notna().all(axis=1)]

    full = full.dropna().reset_index(drop=True)

    if len(full) < N_TRAIN + N_TEST:
        return None

    # строгий временной сплит: train -> test (test строго позже)
    full = full.iloc[-(N_TRAIN + N_TEST):].reset_index(drop=True)

    # нормализация ПП по статистикам ТОЛЬКО train-сегмента
    y = full["target"].to_numpy(dtype=float)
    y_train_raw = y[:N_TRAIN]
    if norm_mode == "z-score":
        mu = y_train_raw.mean()
        sd = y_train_raw.std(ddof=0)
        full["target_norm"] = (y - mu) / (sd if sd > 0 else 1.0)
    elif norm_mode == "min-max":
        lo, hi = y_train_raw.min(), y_train_raw.max()
        full["target_norm"] = (y - lo) / (hi - lo if hi > lo else 1.0)
    else:
        full["target_norm"] = y

    train = full.iloc[:N_TRAIN]
    test = full.iloc[N_TRAIN:]

    return {
        "X_train": train[feature_cols],
        "y_train": train["target_norm"].to_numpy(),
        "X_test": test[feature_cols],
        "y_test": test["target_norm"].to_numpy(),
        "train_time": train["time"].reset_index(drop=True),
        "test_time": test["time"].reset_index(drop=True),
        "X_edge": edge[feature_cols].reset_index(drop=True),
        "edge_time": edge["time"].reset_index(drop=True),
        "n_features": len(feature_cols),
        "teacher_mode": teacher_mode,
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

# ---------------------------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LightGBM ПП-аппроксиматор", layout="wide")

st.title("LightGBM-аппроксимация ПП (производная гауссова фильтра Close)")
st.caption(
    "Исследовательский инструмент, НЕ торговый бот. "
    "Модель не прогнозирует цену: она аппроксимирует производную сглаженного ряда "
    "на тестовом сегменте (150 баров сразу после обучающих 1000)."
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
    teacher_mode = st.selectbox(
        "Учитель (целевая ПП)",
        ["каузальный", "неказуальный"],
        index=0,
        help=(
            "Каузальный: фильтр использует только текущий и прошлые бары — "
            "сигнал доступен в реальном времени. "
            "Неказуальный: симметричный фильтр использует и будущие бары — "
            "«идеальный» учитель, недоступный в реальном времени. "
            "На последних window//2 барах («край») учитель не вычислим, "
            "но обученная модель выдаёт там его оценку по каузальным фичам."
        ),
    )
    max_lag = st.slider("max_lag — макс. лаг", 1, 50, 10)
    vol_window = st.slider("Окно волатильности", 5, 120, 20)
    gauss_sigma = st.slider("Gaussian sigma", 0.5, 10.0, 2.0, 0.5)
    gauss_window = st.slider("Gaussian window", 8, 256, 32, 8)
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
dataset = build_dataset(raw, max_lag, vol_window, gauss_sigma, gauss_window, norm_mode, teacher_mode)
if dataset is None:
    st.error(
        "Недостаточно истории для выбранных окон/лагов. "
        "Уменьшите max_lag / окна или увеличьте FETCH_BARS в коде."
    )
    st.stop()

has_edge = len(dataset["X_edge"]) > 0

if teacher_mode == "неказуальный":
    st.info(
        f"Неказуальный учитель: целевая ПП использует будущие бары (look-ahead только в цели — "
        f"это учитель, а не торговый сигнал). Последние {len(dataset['edge_time'])} баров истории — "
        f"«край»: там истинная ПП ещё не вычислима, но обученная модель по каузальным фичам "
        f"выдаёт её оценку в реальном времени (зелёная линия на графике)."
    )

# --- обучение и аппроксимация (индикатор загрузки)
with st.spinner("Обучение LightGBM и аппроксимация ПП..."):
    model = train_model(dataset["X_train"], dataset["y_train"], lgbm_params)
    y_pred = model.predict(dataset["X_test"])
    y_edge = model.predict(dataset["X_edge"]) if has_edge else None

# --- метрики на тесте
mae = mean_absolute_error(dataset["y_test"], y_pred)
rmse = float(np.sqrt(mean_squared_error(dataset["y_test"], y_pred)))
r2 = r2_score(dataset["y_test"], y_pred)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE (test)", f"{mae:.5f}")
c2.metric("RMSE (test)", f"{rmse:.5f}")
c3.metric("R² (test)", f"{r2:.4f}")
c4.metric("Признаков", dataset["n_features"])

edge_note = f" | Край (оценка модели): {len(dataset['edge_time'])} баров" if has_edge else ""
st.caption(
    f"Источник: {source} | Учитель: {teacher_mode} | "
    f"Train: {dataset['train_time'].iloc[0]} — {dataset['train_time'].iloc[-1]} ({N_TRAIN} баров) | "
    f"Test: {dataset['test_time'].iloc[0]} — {dataset['test_time'].iloc[-1]} ({N_TEST} баров) | "
    f"Нормализация: {norm_mode} (по train){edge_note}"
)

# --- график (тестовый сегмент + «край» с оценкой модели)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dataset["test_time"], y=dataset["y_test"],
    name=f"ПП (истина, {teacher_mode} учитель)", line=dict(color="#1f77b4", width=2),
))
fig.add_trace(go.Scatter(
    x=dataset["test_time"], y=y_pred,
    name="Аппроксимация LightGBM", line=dict(color="#ff7f0e", width=2, dash="dash"),
))
if has_edge:
    # непрерывно продолжаем линию от последней точки теста
    edge_x = pd.concat([dataset["test_time"].iloc[-1:], dataset["edge_time"]], ignore_index=True)
    edge_y = np.concatenate([[y_pred[-1]], y_edge])
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        name="Оценка модели на краю (учитель недоступен)",
        mode="lines+markers",
        line=dict(color="#2ca02c", width=2, dash="dot"),
        marker=dict(size=5),
    ))
    fig.add_vline(
        x=dataset["edge_time"].iloc[0], line_dash="dot", line_color="#2ca02c",
        annotation_text="край: дальше учитель недоступен",
        annotation_position="top left",
    )
fig.add_hline(y=0, line_dash="dot", line_color="gray")
fig.update_layout(
    title=f"Тестовый сегмент (150 баров): истинная ПП ({teacher_mode} учитель) vs аппроксимация",
    xaxis_title="Время",
    yaxis_title="ПП (норм.)",
    hovermode="x unified",
    height=540,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig, use_container_width=True)

# --- экспорт обученной модели
with st.expander("Экспорт обученной модели"):
    e1, e2 = st.columns(2)
    e1.download_button(
        "Скачать booster (.txt)",
        data=model.booster_.model_to_string(),
        file_name="lgbm_pp_approximator_V1.txt",
        mime="text/plain",
    )
    buf = io.BytesIO()
    joblib.dump(model, buf)
    e2.download_button(
        "Скачать модель (.pkl)",
        data=buf.getvalue(),
        file_name="lgbm_pp_approximator_V1.pkl",
        mime="application/octet-stream",
    )

with st.expander("Методология (защита от look-ahead)"):
    st.markdown(
        """
- **Два режима учителя (целевой ПП)**:
  - *Каузальный* — односторонний гауссов фильтр: веса только на текущий и прошлые бары.
    Сигнал доступен в реальном времени.
  - *Неказуальный* — симметричный гауссов фильтр: использует и будущие бары.
    Это «идеальный» учитель, недоступный в реальном времени; look-ahead есть
    **только в цели**, что допустимо для исследования (модель учится
    восстанавливать идеальный сигнал из прошлого).
- **«Край» истории (неказуальный режим)**: на последних `window//2` барах истинная
  неказуальная ПП ещё не вычислима (нужны будущие бары). Но обученная модель
  получает на вход каузальные лаговые фичи и выдаёт **оценку идеальной ПП
  в реальном времени** — именно этому она и обучалась. Эти бары показаны
  на графике зелёной пунктирной линией; ground truth там нет по определению.
- **Признаки** — только лаги `lag >= 1`: Close, Returns, Volatility (в обоих режимах).
- **Нормализация ПП** — среднее/σ (или min/max) считаются только по train-сегменту.
- **Сплит по времени**: train 1000 баров → test 150 баров строго позже.
- **Воспроизводимость**: фиксированный seed, `deterministic=True`, `n_jobs=1`.
        """
    )
