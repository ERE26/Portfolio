"""ESTRATEGIA 2 - GAP-FADE: desvanecer gaps grandes de apertura.

Hipotesis: en gaps de apertura grandes (|gap| sobre el cierre previo > umbral),
el precio tiende a revertir hacia el VWAP / cierre previo. Se FADEA con una pequena
confirmacion de giro:
  - gap UP  (> +umbral): CORTO cuando una vela cierra por DEBAJO del minimo de la
    1a vela (giro a la baja). Objetivo = VWAP de sesion. Stop = maximo del dia hasta
    la entrada (si rompe nuevos maximos, el fade fallo).
  - gap DOWN (< -umbral): LARGO cuando una vela cierra por ENCIMA del maximo de la
    1a vela. Objetivo = VWAP. Stop = minimo del dia hasta la entrada.
Modos: 'objGapFill' (cierra al rellenar el hueco = tocar el cierre previo, si no EOD) y
'EOD' (aguanta al cierre). El objetivo solo cuenta si queda en el lado rentable (si el
hueco ya se llenó antes de confirmar el giro, se aguanta a EOD).

El gap se mide con el cierre previo REAL (GROUPED del dia de sesion anterior, ajustado
-> sin falsos gaps por splits), un solo request por dia para todo el universo.
Point-in-time, sin supervivencia, 3 semillas, bootstrap IC95.

  python backtest_gapfade_massive.py [--umbral 0.03] [--tickers 35] [--dias 15] [--meses 20]
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import backtest_nuevos_massive as H
import universo_pit as U
from fuentes import massive


def _prev_session(fecha, cm):
    """Ultimo dia de sesion estrictamente anterior a `fecha`."""
    ts = pd.Timestamp(fecha)
    ini = (ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    spy = massive.barras_diarias("SPY", ini, fecha, cm=cm)
    prev = [d for d in spy.index if d < ts]
    return prev[-1].strftime("%Y-%m-%d") if prev else None


def _sim(close, high, low, obj, i_in, lado, stop):
    """Simula el fade desde la barra de entrada i_in. lado=+1 largo, -1 corto.
    obj = objetivo de gap-fill (cierre previo). Devuelve {objGapFill, EOD}."""
    n = len(close)
    entry = float(close[i_in]) * (1 + H.SLIP * lado)
    rdist = abs(entry - stop)
    if rdist <= 0:
        return None
    # El objetivo solo es valido si esta en el lado RENTABLE de la entrada (si el gap ya
    # se lleno antes de confirmar el giro, no hay objetivo -> se aguanta a EOD).
    obj_valido = (lado > 0 and obj > entry) or (lado < 0 and obj < entry)
    out = {}
    # modo objetivo gap-fill (cierre previo)
    exit_px = float(close[n - 1])
    for j in range(i_in, n):
        hi, lo, cl = float(high[j]), float(low[j]), float(close[j])
        if lado > 0 and lo <= stop:
            exit_px = stop * (1 - H.SLIP_STOP); break
        if lado < 0 and hi >= stop:
            exit_px = stop * (1 + H.SLIP_STOP); break
        if obj_valido and lado > 0 and hi >= obj:
            exit_px = obj * (1 - H.SLIP); break
        if obj_valido and lado < 0 and lo <= obj:
            exit_px = obj * (1 + H.SLIP); break
        if j == n - 1:
            exit_px = cl * (1 - H.SLIP * lado); break
    R = lado * (exit_px - entry) / rdist
    if lado < 0:
        R -= (H.BORROW_CORTO * entry) / rdist
    out["objGapFill"] = R
    # modo EOD (solo stop o cierre)
    exit_px = float(close[n - 1])
    for j in range(i_in, n):
        hi, lo, cl = float(high[j]), float(low[j]), float(close[j])
        if lado > 0 and lo <= stop:
            exit_px = stop * (1 - H.SLIP_STOP); break
        if lado < 0 and hi >= stop:
            exit_px = stop * (1 + H.SLIP_STOP); break
        if j == n - 1:
            exit_px = cl * (1 - H.SLIP * lado); break
    R = lado * (exit_px - entry) / rdist
    if lado < 0:
        R -= (H.BORROW_CORTO * entry) / rdist
    out["EOD"] = R
    return out


def run(umbral=0.03, tickers=35, dias=15, meses=20, seeds=(42, 7, 123), cm=None):
    cm = cm or H.cliente()
    modos = ["objGapFill", "EOD"]
    por_seed = {}
    for seed in seeds:
        pares = H.muestrear(seed, clase="CS", n_tickers=tickers, n_dias=dias, meses=meses, cm=cm)
        # cierres previos por dia (un GROUPED del dia anterior sirve para todo el universo)
        por_dia = defaultdict(list)
        for tk, f in pares:
            por_dia[f].append(tk)
        res = {m: [] for m in modos}
        n_gap = 0
        for fecha, tks in por_dia.items():
            prev = _prev_session(fecha, cm)
            if not prev:
                continue
            g = U.grouped_asof(prev, cm=cm)      # [ticker, close, volume, dollar_vol], parquet-cacheado
            if g.empty:
                continue
            pc = g.set_index("ticker")["close"]
            for tk in tks:
                if tk not in pc.index:
                    continue
                df = H.cargar(tk, fecha, cm=cm)
                if len(df) < 12:
                    continue
                opn = df["open"].to_numpy(float); close = df["close"].to_numpy(float)
                high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
                prev_close = float(pc.loc[tk])
                if prev_close <= 0:
                    continue
                gap = opn[0] / prev_close - 1.0
                if abs(gap) < umbral:
                    continue
                n_gap += 1
                or_hi, or_lo = float(high[0]), float(low[0])
                if gap > 0:                       # gap up -> fade corto al girar a la baja
                    lado, i_in, stop = -1, None, None
                    run_hi = or_hi
                    for j in range(1, len(close)):
                        run_hi = max(run_hi, float(high[j]))
                        if close[j] < or_lo:
                            i_in = j; stop = run_hi; break
                else:                             # gap down -> fade largo al girar al alza
                    lado, i_in, stop = 1, None, None
                    run_lo = or_lo
                    for j in range(1, len(close)):
                        run_lo = min(run_lo, float(low[j]))
                        if close[j] > or_hi:
                            i_in = j; stop = run_lo; break
                if i_in is None:
                    continue
                r = _sim(close, high, low, prev_close, i_in, lado, stop)
                if r:
                    for m in modos:
                        res[m].append({"R": r[m]})
        print(f"# seed {seed}: {len(pares)} pares, {n_gap} con |gap|>{umbral:.0%}")
        for m in modos:
            por_seed[(m, seed)] = res[m]
    return modos, por_seed, seeds


def informe(modos, por_seed, seeds):
    filas = []
    for m in modos:
        for seed in seeds:
            filas.append(H.resumen_R(m, por_seed[(m, seed)], seed))
        filas.append(H.resumen_R(m, sum((por_seed[(m, s)] for s in seeds), []), "pooled"))
    print(H.tabla_md(filas))
    print()
    for m in modos:
        perseed = [H.resumen_R(m, por_seed[(m, s)], s) for s in seeds]
        print(f"- **{m}**: {H.veredicto_global(perseed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.03)
    ap.add_argument("--tickers", type=int, default=35)
    ap.add_argument("--dias", type=int, default=15)
    ap.add_argument("--meses", type=int, default=20)
    a = ap.parse_args()
    modos, por_seed, seeds = run(a.umbral, a.tickers, a.dias, a.meses)
    print()
    informe(modos, por_seed, seeds)
