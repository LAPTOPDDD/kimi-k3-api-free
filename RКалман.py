import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    HAS_UC = True
except ImportError:
    HAS_UC = False
    print("⚠️ statsmodels не установлен — используем ручной Калман")

# ==========================================
# НАСТРОЙКИ
# ==========================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
TOTAL_BARS = 5000

SIGMA_TEACHER = 1
TARGET_SCALE = 10000
NB = 4

WF_WINDOWS = 5
WF_SIZE = 100
TRAIN_BARS = 3000
MIN_FRAC_POS = 0.6
DD_W = 3.0

# Сетка Калмана
SIGMA_LEVEL_GRID = [0.001, 0.005, 0.01, 0.05]
SIGMA_SLOPE_GRID = [0.0001, 0.0005, 0.001, 0.005]
SIGMA_IRREG_GRID = [0.005, 0.01, 0.02, 0.05]

# ==========================================
# ДАННЫЕ
# ==========================================
if not mt5.initialize(): print("❌ MT5"); exit()
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_BARS)
mt5.shutdown()
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s'); df.set_index('time', inplace=True)
df = df[df.index.dayofweek < 5].copy()
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
df = df.dropna().copy()
close = df['close'].values
fcols = [c for c in df.columns if c not in
         ['open','high','low','close','tick_volume','spread','real_volume']]
X_all = df[fcols].values
T = len(df)
print(f"✅ {T} баров | statsmodels={'есть' if HAS_UC else 'нет (ручной Калман)'}")

# учитель
sm_all = gaussian_filter1d(close, sigma=SIGMA_TEACHER, mode='reflect')
ideal_all = np.zeros_like(sm_all)
ideal_all[1:-1] = (sm_all[2:] - sm_all[:-2]) / 2.0

# ==========================================
# ПРАВИЛО
# ==========================================
def user_rule(sig, close_t, nb):
    eq, pos, entry, held, tr = 0.0, 0, 0.0, 0, 0
    curve, prev_s = [], 0
    for t in range(len(sig)):
        p = close_t[t]; s = int(np.sign(sig[t]))
        cross = (s != 0) and (s != prev_s)
        if pos != 0:
            held += 1
            if held >= nb or (cross and s == -pos):
                eq += (p - entry) * pos; tr += 1; pos = 0
        if cross and pos == 0: pos, entry, held = s, p, 0
        prev_s = s; curve.append(eq)
    if pos != 0: eq += (close_t[-1] - entry) * pos
    return np.array(curve), eq, tr

def window_score(curve, pnl, trd):
    if trd < 3 or pnl <= 0: return -5.0
    ret = np.diff(curve, prepend=0.0)
    std = ret.std(); n = len(ret)
    vol_win = std * np.sqrt(n) + 1e-12
    sh = (ret.mean() / (std + 1e-12)) * np.sqrt(n)
    dd = np.max(np.maximum.accumulate(curve) - curve) / vol_win
    return sh - DD_W * dd

# ==========================================
# КАЛМАН: извлекаем slope_t (каузальная скорость)
# ==========================================
def kalman_slope(c_train, c_eval, sigma_level, sigma_slope, sigma_irreg):
    """Фильтр local linear trend → slope_t (каузально)"""
    if HAS_UC:
        try:
            mod = UnobservedComponents(
                c_train,
                level='local linear trend',
                irreg=True,
                stochastic_level=True,
                stochastic_trend=False,
            )
            res = mod.fit(disp=False, maxiter=300)
            # применяем С ФИКСИРОВАННЫМИ параметрами к eval
            mod_e = UnobservedComponents(
                c_eval,
                level='local linear trend',
                irreg=True,
            )
            # переносим параметры train → eval
            params = res.params.copy()
            # имена: 'sigma2.irregular', 'sigma2.level', 'sigma2.trend'
            out = mod_e.filter(params)
            states = out.filtered_state
            # состояния: [level, slope]
            slope = states[1] if states.shape[0] >= 2 else states[0]
            return slope
        except Exception:
            return _manual_kalman(c_train, c_eval, sigma_level, sigma_slope, sigma_irreg)
    else:
        return _manual_kalman(c_train, c_eval, sigma_level, sigma_slope, sigma_irreg)

def _manual_kalman(c_train, c_eval, sigma_level, sigma_slope, sigma_irreg):
    """Ручной Калман local linear trend (fallback)"""
    q = np.array([[sigma_level**2, 0],
                  [0, sigma_slope**2]])
    r = sigma_irreg**2
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    def filter_only(series):
        n = len(series)
        x = np.array([series[0], 0.0])
        P = np.eye(2) * 1e6
        slopes = np.zeros(n)
        for t in range(n):
            x = F @ x
            P = F @ P @ F.T + q
            y = series[t] - H @ x
            S = H @ P @ H.T + r
            K = P @ H.T / S
            x = x + K.flatten() * y
            P = (np.eye(2) - K @ H) @ P
            slopes[t] = x[1]
        return slopes, x, P

    # учим на train (только прогон, без подгонки — параметры уже переданы)
    _, x, P = filter_only(c_train)
    # продолжаем на eval, начиная с последнего состояния train
    n = len(c_eval)
    slopes = np.zeros(n)
    for t in range(n):
        x = F @ x
        P = F @ P @ F.T + q
        y = c_eval[t] - H @ x
        S = H @ P @ H.T + r
        K = P @ H.T / S
        x = x + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P
        slopes[t] = x[1]
    return slopes

# ==========================================
# LGBM-БАЗА (для честного сравнения)
# ==========================================
def lgbm_predict(c_train, X_train, X_eval):
    sm = gaussian_filter1d(c_train, sigma=SIGMA_TEACHER, mode='reflect')
    dd = np.zeros_like(sm); dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    sc = StandardScaler(); Xtr_s = sc.fit_transform(X_train)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, min_child_samples=30,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xtr_s, dd * TARGET_SCALE)
    k = np.std(dd) / (np.std(m.predict(Xtr_s) / TARGET_SCALE) + 1e-9)
    return m.predict(sc.transform(X_eval)) / TARGET_SCALE * k

# ==========================================
# GREEDY: оценка кандидата на 5 val-окнах
# ==========================================
def eval_kalman(sigma_l, sigma_s, sigma_i, te_start, te_end, val_wins):
    scores, pnls = [], []
    for (vs, ve), (c_tr_w, X_tr_w), (c_v_w, X_v_w) in val_wins:
        slope = kalman_slope(c_tr_w, c_v_w, sigma_l, sigma_s, sigma_i)
        curve, pnl, trd = user_rule(slope, c_v_w, NB)
        scores.append(window_score(curve, pnl, trd)); pnls.append(pnl)
    # тест (только отчёт)
    c_tr_main = close[max(0, te_start - TRAIN_BARS):te_start]
    slope_te = kalman_slope(c_tr_main, close[te_start:te_end], sigma_l, sigma_s, sigma_i)
    _, te_pnl, _ = user_rule(slope_te[:WF_SIZE-2], close[te_start:te_start+WF_SIZE-2], NB)
    return float(np.mean(scores)), float(np.mean(np.array(pnls) > 0)), te_pnl, slope_te

def eval_lgbm(te_start, te_end, val_wins):
    scores, pnls = [], []
    for (vs, ve), (c_tr_w, X_tr_w), (c_v_w, X_v_w) in val_wins:
        pred = lgbm_predict(c_tr_w, X_tr_w, X_v_w)
        curve, pnl, trd = user_rule(pred, c_v_w, NB)
        scores.append(window_score(curve, pnl, trd)); pnls.append(pnl)
    c_tr_main = close[max(0, te_start - TRAIN_BARS):te_start]
    X_tr_main = X_all[max(0, te_start - TRAIN_BARS):te_start]
    X_te = X_all[te_start:te_end]
    pred_te = lgbm_predict(c_tr_main, X_tr_main, X_te)
    _, te_pnl, _ = user_rule(pred_te[:WF_SIZE-2], close[te_start:te_start+WF_SIZE-2], NB)
    return float(np.mean(scores)), float(np.mean(np.array(pnls) > 0)), te_pnl, pred_te[:WF_SIZE-2]

# ==========================================
# ЗАПУСК: одно финальное окно (для скорости)
# ==========================================
te_end, te_start = T, T - WF_SIZE

# готовим val-окна
val_wins = []
for i in range(WF_WINDOWS):
    ve = te_start - i * WF_SIZE
    vs = ve - WF_SIZE
    if vs < TRAIN_BARS + 100: break
    c_tr_w = close[max(0, vs - TRAIN_BARS):vs]
    X_tr_w = X_all[max(0, vs - TRAIN_BARS):vs]
    c_v_w = close[vs:ve]
    X_v_w = X_all[vs:ve]
    val_wins.append(((vs, ve), (c_tr_w, X_tr_w), (c_v_w, X_v_w)))

print(f"\n🔄 GREEDY: {len(SIGMA_LEVEL_GRID)}×{len(SIGMA_SLOPE_GRID)}×{len(SIGMA_IRREG_GRID)} "
      f"= {len(SIGMA_LEVEL_GRID)*len(SIGMA_SLOPE_GRID)*len(SIGMA_IRREG_GRID)} вариантов "
      f"× {len(val_wins)} val-окон")

# --- LGBM baseline ---
lgbm_score, lgbm_frac, lgbm_te_pnl, lgbm_te_sig = eval_lgbm(te_start, te_end, val_wins)
print(f"\n  LGBM baseline: score={lgbm_score:>6.2f} frac={lgbm_frac:.0%} test_pnl={lgbm_te_pnl:>8.4f}")

# --- Калман: ландшафт ---
results = []
for sl in SIGMA_LEVEL_GRID:
    for ss in SIGMA_SLOPE_GRID:
        for si in SIGMA_IRREG_GRID:
            try:
                score, frac, te_pnl, slope_te = eval_kalman(sl, ss, si, te_start, te_end, val_wins)
                results.append(dict(sl=sl, ss=ss, si=si, score=score, frac=frac,
                                     te_pnl=te_pnl, slope_te=slope_te))
            except Exception as e:
                pass

df_res = pd.DataFrame(results)

# --- ландшафт по σ_slope × σ_level (усреднённый по σ_irreg) ---
print(f"\n📊 ЛАНДШАФТ score (усреднён по σ_irreg):")
pivot = df_res.groupby(['sl', 'ss'])['score'].mean().unstack(fill_value=-5)
print(pivot.round(2).to_string())

# --- честный отбор ---
good = df_res[(df_res['score'] > 0) & (df_res['frac'] >= MIN_FRAC_POS)]
print(f"\n🏆 Робастных кандидатов: {len(good)} / {len(df_res)}")

if len(good) == 0:
    print("❌ Нет робастных Калман-кандидатов — каузальный потолок подтверждён")
    best_slope = results[0]['slope_te']
    best_name = "первый (слабый)"
else:
    good = good.sort_values('score', ascending=False)
    print(f"  Топ-5:")
    print(good[['sl','ss','si','score','frac','te_pnl']].head().to_string(index=False))
    best_slope = good.iloc[0]['slope_te']
    best_name = f"σL={good.iloc[0]['sl']:.3f} σS={good.iloc[0]['ss']:.4f} σI={good.iloc[0]['si']:.3f}"

# ==========================================
# ФИНАЛЬНОЕ СРАВНЕНИЕ: LGBM vs лучший Калман vs идеал
# ==========================================
n = WF_SIZE - 2
cl = close[te_start:te_start + n]
idl = ideal_all[te_start:te_start + n]

c_lgbm, pnl_lgbm, _ = user_rule(lgbm_te_sig[:n], cl, NB)
c_kal, pnl_kal, _ = user_rule(best_slope[:n], cl, NB)
c_idl, pnl_idl, _ = user_rule(idl, cl, NB)

print(f"\n{'='*56}")
print(f"  ФИНАЛЬНОЕ ОКНО (без комиссий):")
print(f"    LGBM:  corr(идеал)={np.corrcoef(lgbm_te_sig[:n], idl)[0,1]:+.3f}  PnL={pnl_lgbm:>8.4f}")
print(f"    KALMAN:corr(идеал)={np.corrcoef(best_slope[:n], idl)[0,1]:+.3f}  PnL={pnl_kal:>8.4f}  ({best_name})")
print(f"    ИДЕАЛ: (оракул)                                      PnL={pnl_idl:>8.4f}")
print(f"{'='*56}")

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                         gridspec_kw={'height_ratios': [1.3, 1]})
x = np.arange(1, n + 1)

axes[0].plot(x, cl, 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} | LGBM vs Калман ({best_name})')
axes[0].grid(alpha=0.3)

axes[1].plot(x, idl, 'g', lw=2, label='ИДЕАЛ (учитель)')
axes[1].plot(x, lgbm_te_sig[:n], 'b', lw=1.5, label=f'LGBM (PnL={pnl_lgbm:.4f})')
axes[1].plot(x, best_slope[:n], 'r', lw=1.5, label=f'Калман (PnL={pnl_kal:.4f})')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_xlabel('Тестовый бар')
plt.tight_layout()
plt.savefig('kalman_vs_lgbm.png', dpi=150)
plt.show()

# ВЕРДИКТ
diff_corr = abs(np.corrcoef(lgbm_te_sig[:n], idl)[0,1] - np.corrcoef(best_slope[:n], idl)[0,1])
diff_pnl = abs(pnl_lgbm - pnl_kal)
if diff_corr < 0.1 and diff_pnl < 0.002:
    print("\n🎯 ВЕРДИКТ: Калман ≈ LGBM по всем метрикам")
    print("   → Подтверждён КАУЗАЛЬНЫЙ ПОТОЛОК: оба оптимальных ученика дают ~одно и то же.")
    print("   → Проблема не в ML, а в ЦЕНЕ КАУЗАЛЬНОСТИ учителя.")
elif pnl_kal > pnl_lgbm * 1.5:
    print("\n🎉 ВЕРДИКТ: Калман СИЛЬНЕЕ LGBM")
    print("   → Нашли ученика выше потолка! Развиваем этот путь.")
else:
    print("\n⚠️  ВЕРДИКТ: смешанный результат — нужен анализ")