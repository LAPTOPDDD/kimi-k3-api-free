import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
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
SIGMA_LIST = [2, 5, 10, 15]     # сравниваем несколько сглаживаний

# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================
if not mt5.initialize():
    print("❌ MT5 init error"); exit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, TOTAL_BARS)
mt5.shutdown()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)
df = df[df.index.dayofweek < 5].copy()

close = df['close'].values
BARS_TRAIN = len(close) - BARS_TEST

x = np.arange(1, BARS_TEST + 1)
price_test = close[BARS_TRAIN:]

# ==========================================
# ГАУСС НА ЦЕНЕ + ПРОИЗВОДНЫЕ (оракул, вся серия)
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

# --- Верх: цена + сглаженные кривые ---
ax1 = axes[0]
ax1.plot(x, price_test, color='black', lw=1, alpha=0.6, label='Цена (raw)')

colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
for sigma, c in zip(SIGMA_LIST, colors):
    smoothed = gaussian_filter1d(close, sigma=sigma, mode='reflect')
    ax1.plot(x, smoothed[BARS_TRAIN:], color=c, lw=2, label=f'Гаусс σ={sigma}')

ax1.set_title(f'{SYMBOL} — Гаусс на ЦЕНЕ (тестовые {BARS_TEST} баров)')
ax1.set_ylabel('Цена')
ax1.legend(loc='upper left'); ax1.grid(alpha=0.3)

# --- Низ: первые производные (наша цель) ---
ax2 = axes[1]
print(f"{'sigma':>6} | {'std(произв.)':>12} | {'шершавость':>12}")
print("-" * 40)

for sigma, c in zip(SIGMA_LIST, colors):
    smoothed = gaussian_filter1d(close, sigma=sigma, mode='reflect')
    deriv = np.zeros_like(smoothed)
    deriv[1:-1] = (smoothed[2:] - smoothed[:-2]) / 2.0
    d_test = deriv[BARS_TRAIN:]

    ax2.plot(x, d_test, color=c, lw=1.5, label=f'произв. σ={sigma}')

    # метрика гладкости: RMS первых разностей производной
    roughness = np.sqrt(np.mean(np.diff(d_test) ** 2))
    print(f"{sigma:>6} | {np.std(d_test):>12.6f} | {roughness:>12.6f}")

ax2.axhline(0, color='gray', lw=0.8, alpha=0.5)
ax2.set_title('Первая производная Гаусса (целевой сигнал)')
ax2.set_xlabel('Номер тестового бара')
ax2.legend(loc='upper left'); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()