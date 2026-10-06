# ==========================================
# WALK-FORWARD ВЫБОР N (вместо одного окна)
# ==========================================
WF_WINDOWS = 10
WF_SIZE = 100

agg = {nb: [] for nb in NB_LIST}

for w in range(WF_WINDOWS):
    test_end = len(df) - w * WF_SIZE
    test_start = test_end - WF_SIZE
    if test_start < 1500:
        break

    X_tr = df[feature_cols].iloc[:test_start].values
    X_te = df[feature_cols].iloc[test_start:test_end].values
    close_tr = close[:test_start]
    close_te = close[test_start:test_end]

    # цель со сдвигом
    sm = gaussian_filter1d(close_tr, sigma=SIGMA, mode='reflect')
    dd = np.zeros_like(sm)
    dd[1:-1] = (sm[2:] - sm[:-2]) / 2.0
    if SHIFT > 0:
        yf, Xf = dd[SHIFT:], X_tr[:-SHIFT]
    else:
        yf, Xf = dd, X_tr

    sc = StandardScaler()
    Xf_s, Xte_s = sc.fit_transform(Xf), sc.transform(X_te)

    m = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
                          min_child_samples=MIN_CHILD_SAMPLES,
                          n_jobs=-1, verbosity=-1, random_state=42)
    m.fit(Xf_s, yf * TARGET_SCALE)

    ptr = m.predict(Xf_s) / TARGET_SCALE
    kk = np.std(yf) / (np.std(ptr) + 1e-9)
    pr = m.predict(Xte_s) / TARGET_SCALE * kk

    nn = WF_SIZE - 2
    for nb in NB_LIST:
        _, pm, _ = user_rule(pr[:nn], close_te[:nn], nb)
        agg[nb].append(pm)
    print(f"  окно {w+1}/{WF_WINDOWS} готово")

# ==========================================
# АГРЕГАТ И ВЫБОР
# ==========================================
print(f"\n{'N':>2} | {'mean PnL':>9} | {'std':>8} | {'winrate':>7} | {'score':>7}")
best_nb, best_score = None, -1e9
for nb in NB_LIST:
    p = np.array(agg[nb])
    score = p.mean() / (p.std() + 1e-12)     # "Sharpe по окнам"
    print(f"{nb:>2} | {p.mean():>9.5f} | {p.std():>8.5f} | {np.mean(p > 0):>6.0%} | {score:>7.2f}")
    if score > best_score:
        best_score, best_nb = score, nb

print(f"\n🏆 ВЫБРАН N = {best_nb}")