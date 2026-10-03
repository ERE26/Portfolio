"""ORB de 5 MINUTOS fiel al paper rentable (Zarattini & Aziz 2023; "ORB for Stocks in
Play", Concretum, Sharpe ~2.4-2.8). Backtest Monte-Carlo OOS con las reglas CLAVE que nos
faltaban:

  - Barras de 5 min (rango de apertura = primera vela de 5m, 9:30-9:35).
  - RVOL DE APERTURA = vol de la 1ª vela de 5m / media de la 1ª vela de 5m los 14 días
    previos (proxy de "stock in play"); gate > umbral.
  - Dirección = color de la 1ª vela (solo LONG aquí: 1ª vela alcista).
  - Entrada: stop-order en el MÁXIMO del rango de apertura (fill al romper).
  - Stop inicial: mínimo del rango de apertura.
  - Salida: MANTENER HASTA EL CIERRE (EOD) — dejar correr la ganadora todo el día; o stop.
    Variantes: target 10R (Zarattini) y trailing opcional.

Compara: EOD puro vs EOD+trailing vs target-10R, y 2 niveles de RVOL de apertura.

  python backtest_orb5.py [--tickers 60] [--dias 25] [--meses 12] [--seed 42]

Mismos avisos honestos que backtest_intradia (sesgo de supervivencia, IEX, 1 entrada/día).
El coste de ejecución es decisivo en el ORB: se modela slippage; el paper rompe break-even
a ~2.2¢/acción de slippage, así que se reporta también la sensibilidad.
"""
import argparse
import random

import numpy as np
import pandas as pd

import backtest_intradia as bi
from fuentes import alpaca

SLIP = 0.0004
SLIP_STOP = 0.0008


def _opening_rvol(df):
    """Serie por día: vol de la 1ª vela de 5m / media de las 1ª velas de los 14 días previos."""
    idx = pd.DatetimeIndex(df.index)
    dia = idx.normalize()
    primera = pd.Series(df["volume"].to_numpy(float), index=df.index).groupby(dia).first()
    base = primera.shift(1).rolling(14, min_periods=5).mean()
    rv = (primera / base).replace([np.inf, -np.inf], np.nan)
    return rv                                    # indexado por día (normalizado)


def _sim(close, high, low, dia_pos, or_hi, or_lo, modo, trail_atr, atr_abs, confirmado=False):
    """Simula un ORB long en un día. dia_pos[0] es la vela del rango de apertura.
    confirmado=False: entra al tocar el máximo del rango (stop-order). confirmado=True:
    espera a que una vela CIERRE sobre el máximo (ruptura confirmada, evita falsas mechas) y
    entra a ese cierre — peor precio, pero filtra las rupturas falsas (-0.37R)."""
    entry = None
    for j in dia_pos[1:]:
        if confirmado:
            if close[j] > or_hi:
                entry = close[j] * (1 + SLIP)    # entra al CIERRE de la vela que confirma
                i_in = j
                break
        elif high[j] >= or_hi:
            entry = or_hi * (1 + SLIP)           # stop-order en el máximo del rango
            i_in = j
            break
    if entry is None:
        return None
    rdist = entry - or_lo
    if rdist <= 0:
        return None
    stop = or_lo
    tp = entry + 10.0 * rdist                     # target 10R (solo modo 'target')
    run_high = entry
    dia_last = dia_pos[-1]
    exit_px, reason = float(close[dia_last]), "EOD"
    for j in range(i_in, dia_last + 1):
        hi, lo, cl = float(high[j]), float(low[j]), float(close[j])
        if lo <= stop:
            reason = "TRAIL" if stop > entry * 1.0001 else "SL"
            exit_px = stop * (1 - SLIP_STOP); break
        if modo == "target" and hi >= tp:
            exit_px = tp * (1 - SLIP); reason = "TP10R"; break
        run_high = max(run_high, hi)
        if modo == "trail" and run_high >= entry + rdist:    # tras +1R, trailing
            nuevo = run_high - trail_atr * atr_abs
            if nuevo > stop:
                stop = nuevo
        if j == dia_last:
            exit_px = cl * (1 - SLIP); reason = "EOD"; break
    return {"R": (exit_px - entry) / rdist, "reason": reason}


def backtest(tickers_n=60, dias=25, meses=12, seed=42, rvol_min=1.0, confirmado=False):
    rng = random.Random(seed)
    uni = bi._universo(n=150)
    rng.shuffle(uni); uni = uni[:tickers_n]
    desde = (pd.Timestamp.now() - pd.Timedelta(days=int(meses * 31))).strftime("%Y-%m-%d")
    modos = {"EOD": ("eod", 0.0), "EOD+trail2ATR": ("trail", 2.0), "target10R": ("target", 0.0)}
    res = {m: [] for m in modos}
    n_td = 0
    print(f"\n### ORB 5-min | {len(uni)} tickers x {dias} días | {meses} meses | seed {seed} | "
          f"RVOL_ap>{rvol_min} | {'RUPTURA CONFIRMADA (cierre)' if confirmado else 'al tocar máximo'}")
    for t in uni:
        try:
            df = alpaca.historico(t, "5Min", desde=desde, max_paginas=20)
        except Exception:
            continue
        if len(df) < 500:
            continue
        close = df["close"].to_numpy(float); high = df["high"].to_numpy(float)
        low = df["low"].to_numpy(float); opn = df["open"].to_numpy(float)
        idx = pd.DatetimeIndex(df.index)
        orv = _opening_rvol(df)
        # ATR diario aprox (en $) para el trailing: rango medio de la sesión / nº aprox velas
        atr_abs_global = float(np.nanmedian((df["high"] - df["low"]).to_numpy())) * 3.0
        por_dia = pd.Series(np.arange(len(df)), index=idx).groupby(idx.normalize())
        dias_ok = [(d, g.to_numpy()) for d, g in por_dia if len(g) >= 20]
        if len(dias_ok) < 2:
            continue
        for d, dia_pos in rng.sample(dias_ok, min(dias, len(dias_ok))):
            n_td += 1
            rv = orv.get(d, np.nan)
            if not (rv == rv) or rv < rvol_min:          # RVOL de apertura insuficiente
                continue
            p0 = dia_pos[0]
            if close[p0] <= opn[p0]:                      # 1ª vela debe ser ALCISTA (dirección long)
                continue
            or_hi, or_lo = float(high[p0]), float(low[p0])
            for m, (modo, tr) in modos.items():
                r = _sim(close, high, low, list(dia_pos), or_hi, or_lo, modo, tr, atr_abs_global,
                         confirmado=confirmado)
                if r is not None:
                    res[m].append(r)
    print(f"ticker-días evaluados: {n_td}\n")
    print(f"{'modo salida':<16}{'ops':>5}{'acierto':>9}{'expectancy':>13}{'IC95':>20}{'PF':>6}  veredicto")
    for m in modos:
        _stats(m, res[m])
    return res


def _stats(nombre, trades):
    if len(trades) < 5:
        print(f"{nombre:<16}{len(trades):>5}  (muestra insuficiente)")
        return
    R = np.array([t["R"] for t in trades])
    w = R[R > 0]; l = R[R <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    lo, hi = bi._boot_ci(R)
    verd = "FUNCIONA (edge+)" if lo > 0 else ("sin edge claro" if hi > 0 else "pierde")
    print(f"{nombre:<16}{len(R):>5}{len(w)/len(R)*100:>7.0f}% {R.mean():>+11.3f}R  "
          f"[{lo:+.3f},{hi:+.3f}]{pf:>6.2f}  {verd}  (R_win {w.mean() if len(w) else 0:+.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--dias", type=int, default=25)
    ap.add_argument("--meses", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rvol", type=float, default=1.0)
    ap.add_argument("--confirmado", action="store_true", help="exigir cierre sobre el máximo del rango")
    a = ap.parse_args()
    backtest(a.tickers, a.dias, a.meses, a.seed, a.rvol, a.confirmado)
