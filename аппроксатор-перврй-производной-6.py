import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# НАСТРОЙКИ
# ==========================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
TOTAL_BARS = 5000
BARS_TEST = 100

SIGMA = 1                      # быстрая сигма
SHIFT = 0                      # 0 = обычная цель
TARGET_SCALE = 10000

# Модель
N_ESTIMATORS = 400
LEARNING_RATE = 0.03
MIN_CHILD_SAMPLES = 15

# Параметры walk-forward
NB_LIST = [3, 4, 5, 6]
EPS_LIST = [0.0, 0.1, 0.2, 0.3, 0.4]   # как доля std(pred_train)
WF_WINDOWS = 10
WF_SIZE = 100

# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================================
if not mt5.initialize():
    print("❌ MT5 init error"); exit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_BARS)
mt5.shutdown()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df[df.index.dayofweek < 5].copy()
print(f"✅ Загружено {len(df)} баров")

# ==========================================
# 2. ФИЧИ: медленные (20) + быстрые (3,5,8)
# ==========================================
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
    lo = df['close'].rolling(w).min()
    hi = df['close'].rolling(w).max()
    df[f'range_pos_{w}'] = (df['close'] - lo) / (hi - lo + 1e-9)

df = df.dropna().copy()
close = df['close'].values
feature_cols = [c for c in df.columns if c not in
                ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]

# ==========================================
# 3. ПОМОЩНИК: обучение + предсказание на одном окне
# ==========================================
def train_and_predict(X_tr, X_te, y_tr):
    sm = gaussian_filter1d(y_tr, sigma=SIGMA, mode='reflect')
    dd = np.zeros_like(sm)
    dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    if SHIFT > 0:
        yf, Xf = dd[SHIFT:], X_tr[:-SHIFT]
    else:
        yf, Xf = dd, X_tr

    sc = StandardScaler()
    Xf_s = sc.fit_transform(Xf)
    Xte_s = sc.transform(X_te)

    m = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                          min_child_samples=MIN_CHILD_SAMPLES,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xf_s, yf * TARGET_SCALE)

    ptr = m.predict(Xf_s) / TARGET_SCALE
    kk = np.std(yf) / (np.std(ptr) + 1e-9)
    pr = m.predict(Xte_s) / TARGET_SCALE * kk
    std_pred = np.std(ptr)
    return pr, std_pred, sc, m

# ==========================================
# 4. ПРАВИЛО ВХОДА С ГИСТЕРЕЗИСОМ
# ==========================================
def user_rule(sig, close_t, nb, eps=0.0):
    eq, pos, entry, held, trades = 0.0, 0, 0.0, 0, 0
    curve, prev_v = [], 0.0
    for t in range(len(sig)):
        p = close_t[t]
        v = sig[t]
        up   = (prev_v <=  eps) and (v >  eps)
        down = (prev_v >= -eps) and (v < -eps)
        if pos != 0:
            held += 1
            opposite = (pos == 1 and down) or (pos == -1 and up)
            if held >= nb or opposite:
                eq += (p - entry) * pos
                trades += 1; pos = 0
        if pos == 0:
            if up:   pos, entry, held =  1, p, 0
            if down: pos, entry, held = -1, p, 0
        prev_v = v
        curve.append(eq)
    if pos != 0:
        eq += (close_t[-1] - entry) * pos
        trades += 1
    return np.array(curve), eq, trades

# ==========================================
# 5. WALK-FORWARD: подбор N и eps
# ==========================================
print(f"\n🔄 Walk-forward: {WF_WINDOWS} окон × {WF_SIZE} баров...")
agg = {(nb, eps): [] for nb in NB_LIST for eps in EPS_LIST}

for w in range(WF_WINDOWS):
    test_end = len(df) - w * WF_SIZE
    test_start = test_end - WF_SIZE
    if test_start < 1500:
        break

    X_tr = df[feature_cols].iloc[:test_start].values
    X_te = df[feature_cols].iloc[test_start:test_end].values
    close_te = close[test_start:test_end]
    y_tr = close[:test_start]

    pr, std_pred, _, _ = train_and_predict(X_tr, X_te, y_tr)
    nn = WF_SIZE - 2
    sig, cl = pr[:nn], close_te[:nn]

    for nb in NB_LIST:
        for eps_f in EPS_LIST:
            _, pm, _ = user_rule(sig, cl, nb, eps_f * std_pred)
            agg[(nb, eps_f)].append(pm)
    print(f"  окно {w+1}/{WF_WINDOWS} ✓")

# ==========================================
# 6. АГРЕГАТ: выбираем лучшую пару (N, eps)
# ==========================================
print(f"\n{'N':>2} {'eps/std':>7} | {'mean':>8} {'std':>7} {'winrate':>7} {'score':>7}")
print("-" * 48)
best, best_score = None, -1e9
for nb in NB_LIST:
    for eps_f in EPS_LIST:
        p = np.array(agg[(nb, eps_f)])
        score = p.mean() / (p.std() + 1e-12)
        if score > best_score:
            best_score, best = score, (nb, eps_f)
        if eps_f in (0.0, 0.2, 0.3):   # показываем только ключевые
            print(f"{nb:>2} {eps_f:>7.1f} | {p.mean():>8.5f} {p.std():>7.5f} "
                  f"{np.mean(p > 0):>6.0%} {score:>7.2f}")

BEST_NB, BEST_EPS_F = best
print(f"\n🏆 ВЫБРАНО: N={BEST_NB}, eps/std={BEST_EPS_F:.2f}  (score={best_score:.2f})")

# ==========================================
# 7. ФИНАЛЬНЫЙ ТЕСТ: последние 100 баров
# ==========================================
print(f"\n{'='*56}")
print(f"ФИНАЛЬНЫЙ ТЕСТ: последние {BARS_TEST} баров")
print(f"{'='*56}")

BARS_TRAIN = len(df) - BARS_TEST
X_tr = df[feature_cols].iloc[:BARS_TRAIN].values
X_te = df[feature_cols].iloc[BARS_TRAIN:].values
y_tr = close[:BARS_TRAIN]
y_te = close[BARS_TRAIN:]

preds, std_pred, scaler, model = train_and_predict(X_tr, X_te, y_tr)

# идеал для теста
sm_all = gaussian_filter1d(close, sigma=SIGMA, mode='reflect')
d_all = np.zeros_like(sm_all)
d_all[1:-1] = (sm_all[2:] - sm_all[:-2]) / 2.0
ideal_test = d_all[BARS_TRAIN + SHIFT: BARS_TRAIN + SHIFT + BARS_TEST]
if len(ideal_test) < BARS_TEST:
    ideal_test = np.pad(ideal_test, (0, BARS_TEST - len(ideal_test)))

# корреляции
actual_forward = np.zeros(BARS_TEST)
for i in range(BARS_TEST - 2):
    actual_forward[i] = (close[BARS_TRAIN + i + 2] - close[BARS_TRAIN + i]) / 2.0
n = BARS_TEST - 2

print(f"  corr(модель, идеал)      = {np.corrcoef(preds[:n], ideal_test[:n])[0,1]:+.3f}")
print(f"  corr(идеал, реальность)  = {np.corrcoef(ideal_test[:n], actual_forward[:n])[0,1]:+.3f}")
print(f"  corr(модель, реальность) = {np.corrcoef(preds[:n], actual_forward[:n])[0,1]:+.3f}")

# эквити: БЕЗ гистерезиса и С гистерезисом
_, pm_raw, tm_raw = user_rule(preds[:n], y_te[:n], BEST_NB, 0.0)
_, pm_best, tm_best = user_rule(preds[:n], y_te[:n], BEST_NB, BEST_EPS_F * std_pred)
_, pi_ideal, ti_ideal = user_rule(ideal_test[:n], y_te[:n], BEST_NB, 0.0)

print(f"\n  {'вариант':>14} | {'PnL':>8} {'сделок':>6}")
print(f"  {'-'*38}")
print(f"  {'модель (raw)':>14} | {pm_raw:>8.4f} {tm_raw:>6}")
print(f"  {'модель + eps':>14} | {pm_best:>8.4f} {tm_best:>6}")
print(f"  {'идеал (raw)':>14} | {pi_ideal:>8.4f} {ti_ideal:>6}")

# ==========================================
# 8. ГРАФИКИ
# ==========================================
curve_raw, _, _ = user_rule(preds[:n], y_te[:n], BEST_NB, 0.0)
curve_best, _, _ = user_rule(preds[:n], y_te[:n], BEST_NB, BEST_EPS_F * std_pred)
curve_ideal, _, _ = user_rule(ideal_test[:n], y_te[:n], BEST_NB, 0.0)

fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                         gridspec_kw={'height_ratios': [1.3, 1, 0.9]})
x = np.arange(1, BARS_TEST + 1)

axes[0].plot(x, y_te, 'k', lw=1.2)
axes[0].set_title(f'{SYMBOL} | σ={SIGMA} | N={BEST_NB} eps/std={BEST_EPS_F:.2f}')
axes[0].grid(alpha=0.3)

axes[1].plot(x[:n], actual_forward[:n], 'r', alpha=0.5, lw=1, label='Реальность')
axes[1].plot(x[:n], ideal_test[:n], 'g', lw=2, label='ИДЕАЛ')
axes[1].plot(x[:n], preds[:n], 'b', lw=1.5, label='Модель')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(curve_raw,  'b--', lw=1, alpha=0.6, label=f'model raw: {pm_raw:.4f} / {tm_raw} сделок')
axes[2].plot(curve_best, 'b',   lw=1.8, label=f'model + eps: {pm_best:.4f} / {tm_best} сделок')
axes[2].plot(curve_ideal,'g',   lw=1.8, label=f'ideal: {pi_ideal:.4f} / {ti_ideal} сделок')
axes[2].axhline(0, color='gray', lw=0.8)
axes[2].set_title('Эквити (без комиссий)')
axes[2].legend(loc='upper left'); axes[2].grid(alpha=0.3)
axes[2].set_xlabel('Тестовый бар')

plt.tight_layout()
plt.savefig('final_test.png', dpi=150)
plt.show()

print(f"\n📋 ИТОГОВАЯ КОНФИГУРАЦИЯ ДЛЯ БОТА:")
print(f"   SYMBOL       = {SYMBOL}")
print(f"   SIGMA        = {SIGMA}")
print(f"   SHIFT        = {SHIFT}")
print(f"   N (удержание)= {BEST_NB}")
print(f"   eps/std      = {BEST_EPS_F:.3f}")