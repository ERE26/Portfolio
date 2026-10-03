"""BACKTEST de MEAN-REVERSION intradía (familia OPUESTA al ORB, para la Opción 3).

Hipótesis: en stocks in play, cuando el precio se EXTIENDE mucho por DEBAJO del VWAP
(sobreventa intradía) y aparece un giro, reversión de vuelta al VWAP. Entrada con LIMIT en
el extremo (menos sensible a slippage que el breakout), objetivo = VWAP, stop por debajo del
mínimo reciente, salida EOD. Long-only (el caso sobreventa->rebote).

Es la familia contraria al momentum/ORB: debería funcionar en días de CHOP (rango), justo
donde el ORB falla. Mismo motor OOS, slippage modelado, multi-semilla.

  python backtest_meanrev.py [--tickers 60] [--dias 25] [--meses 12] [--seed 42] [--k 1.5]

k = cuántos ATR por debajo del VWAP para considerar sobreventa (extensión).
"""
import argparse
import random

import numpy as np
import pandas as pd

import backtest_intradia as bi
from backtest_orb5 import _opening_rvol
from fuentes import alpaca
from patrones import indicadores as ind

SLIP = 0.0004
SLIP_STOP = 0.0008


def _sim_mr(close, high, low, dia_pos, dist_vwap, atr_pct, k):
    """Primera señal de sobreventa+giro del día; reversión a VWAP. Devuelve R o None."""
    for n, j in enumerate(dia_pos):
        if n < 3 or n >= len(dia_pos) - 2:
            continue
        atrp = atr_pct[j]
        if not (atrp > 0):
            continue
        # sobreventa: cierre >= k ATR por debajo del VWAP
        if dist_vwap[j] > -k * atrp:
            continue
        # giro: la vela actual cierra verde y por encima de la anterior (rebote incipiente)
        if not (close[j] > close[j - 1] and close[j] > (high[j] + low[j]) / 2):
            continue
        entry = close[j] * (1 + SLIP)
        vwap_px = close[j] / (1 + dist_vwap[j])          # objetivo = VWAP implícito
        stop = min(low[j], low[j - 1]) * (1 - 0.001)     # stop bajo el mínimo reciente
        rdist = entry - stop
        if rdist <= 0 or vwap_px <= entry:
            continue
        tp = vwap_px                                      # objetivo: volver al VWAP
        dia_last = dia_pos[-1]
        exit_px, = (float(close[dia_last]),)
        for m in range(j + 1, dia_last + 1):
            if low[m] <= stop:
                exit_px = stop * (1 - SLIP_STOP); break
            if high[m] >= tp:
                exit_px = tp * (1 - SLIP); break
            if m == dia_last:
                exit_px = float(close[m]) * (1 - SLIP); break
        return (exit_px - entry) / rdist
    return None


def backtest(tickers_n=60, dias=25, meses=12, seed=42, k=1.5):
    rng = random.Random(seed)
    uni = bi._universo(n=150); rng.shuffle(uni); uni = uni[:tickers_n]
    desde = (pd.Timestamp.now() - pd.Timedelta(days=int(meses * 31))).strftime("%Y-%m-%d")
    R = []
    n_td = 0
    print(f"\n### MEAN-REVERSION intradía | {len(uni)} tickers x {dias} días | {meses} meses | seed {seed} | k={k} ATR")
    for t in uni:
        try:
            df = alpaca.historico(t, "5Min", desde=desde, max_paginas=20)
        except Exception:
            continue
        if len(df) < 500:
            continue
        close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
        idx = pd.DatetimeIndex(df.index)
        dv = bi.rasgos_intradia(df)["intra_dist_vwap"].to_numpy()
        atrp = (ind.atr(df["high"], df["low"], df["close"], 14) / df["close"]).to_numpy()
        orv = _opening_rvol(df)
        por_dia = pd.Series(np.arange(len(df)), index=idx).groupby(idx.normalize())
        dias_ok = [(d, g.to_numpy()) for d, g in por_dia if len(g) >= 20]
        for d, dia_pos in rng.sample(dias_ok, min(dias, len(dias_ok))):
            n_td += 1
            rv = orv.get(d, np.nan)
            if not (rv == rv) or rv < 1.0:               # stock in play
                continue
            r = _sim_mr(close, high, low, list(dia_pos), dv, atrp, k)
            if r is not None:
                R.append(r)
    R = np.array(R)
    print(f"ticker-días evaluados: {n_td} | operaciones: {len(R)}")
    if len(R) < 20:
        print("muestra insuficiente"); return
    w = R[R > 0]; l = R[R <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    lo, hi = bi._boot_ci(R)
    verd = "FUNCIONA (edge+)" if lo > 0 else ("sin edge claro" if hi > 0 else "pierde")
    print(f"  acierto {len(w)/len(R)*100:.0f}% | expectancy {R.mean():+.3f}R [IC95 {lo:+.3f},{hi:+.3f}] | "
          f"PF {pf:.2f} | R_win {w.mean() if len(w) else 0:+.2f}")
    print(f"  -> VEREDICTO: {verd}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--dias", type=int, default=25)
    ap.add_argument("--meses", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=float, default=1.5)
    a = ap.parse_args()
    backtest(a.tickers, a.dias, a.meses, a.seed, a.k)
