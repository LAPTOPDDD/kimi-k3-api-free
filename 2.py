def user_rule_eps(sig, close_t, nb, eps=0.0):
    """Твоё правило + гистерезис: вход при пробое ±eps"""
    eq, pos, entry, held, trades = 0.0, 0, 0.0, 0, 0
    curve, prev_v = [], 0.0
    for t in range(len(sig)):
        p = close_t[t]
        v = sig[t]
        up   = (prev_v <=  eps) and (v >  eps)   # пробил +eps снизу
        down = (prev_v >= -eps) and (v < -eps)   # пробил -eps сверху

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
        eq += (close_t[-1] - entry) * pos; trades += 1
    return np.array(curve), eq, trades

# ==========================================
# ПОДБОР ε: цель — сделок ≈ 13 (как у идеала) и максимум PnL
# ==========================================
std_p = np.std(pred_n)
print(f"{'eps/std':>7} | {'модель PnL':>10} {'сдел':>4} {'PnL/сдел':>9} | идеал: 13 сделок")
for f in [0.0, 0.1, 0.2, 0.3, 0.5]:
    _, pm, tm = user_rule_eps(pred_n, y_test_raw[:n], 4, f * std_p)
    print(f"{f:>7.1f} | {pm:>10.4f} {tm:>4} {pm/(tm+1e-9):>9.5f}")