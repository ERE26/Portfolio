"""ESTRATEGIA 3 - MEAN-REVERSION en ETFs y LARGE-CAPS (NO in-play).

Distinta a lo ya descartado (aquel era mean-reversion sobre mid-caps "in play").
Aqui el universo es LIQUIDO y SIN problema de supervivencia:
  - ETFs de indice/sector (SPY,QQQ,IWM,DIA,XLK,...) -> fijos, todos vivos.
  - top-30 large-caps por dolar-volumen POINT-IN-TIME (del GROUPED del mes anterior).

Setup: fade de extensiones intradia respecto al VWAP de sesion. Extension
  z = (close - VWAP) / ATR_bar. Cuando z <= -k -> LARGO (precio deprimido bajo VWAP,
  objetivo VWAP). Cuando z >= +k -> CORTO (precio estirado sobre VWAP, objetivo VWAP).
  Stop = entry -/+ m*ATR. Una entrada por dia y lado (la primera extension valida).
  Se evita la media hora final (sin tiempo de reversion) y los 2 primeros bars
  (VWAP aun inestable).

Point-in-time, sin supervivencia, long y short por separado + pooled, 3 semillas.

  python backtest_meanrev_massive.py [--k 1.5] [--m 1.0] [--tickers 40] [--dias 15] [--meses 20]
"""
import argparse

import numpy as np
import pandas as pd

import backtest_nuevos_massive as H


def _atr_bar(df, n=14):
    """ATR por barra (serie, en $), para normalizar la extension respecto al VWAP."""
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    s = pd.Series(tr).rolling(n, min_periods=3).mean().to_numpy()
    return s


def _sim(close, high, low, vwap, i_in, lado, stop):
    n = len(close)
    entry = float(close[i_in]) * (1 + H.SLIP * lado)
    rdist = abs(entry - stop)
    if rdist <= 0:
        return None
    exit_px = float(close[n - 1])
    for j in range(i_in, n):
        hi, lo, cl, vw = float(high[j]), float(low[j]), float(close[j]), float(vwap[j])
        if lado > 0 and lo <= stop:
            exit_px = stop * (1 - H.SLIP_STOP); break
        if lado < 0 and hi >= stop:
            exit_px = stop * (1 + H.SLIP_STOP); break
        if lado > 0 and hi >= vw:               # objetivo: volver al VWAP
            exit_px = vw * (1 - H.SLIP); break
        if lado < 0 and lo <= vw:
            exit_px = vw * (1 + H.SLIP); break
        if j == n - 1:
            exit_px = cl * (1 - H.SLIP * lado); break
    R = lado * (exit_px - entry) / rdist
    if lado < 0:
        R -= (H.BORROW_CORTO * entry) / rdist
    return {"R": R}


def run(k=1.5, m=1.0, tickers=40, dias=15, meses=20, seeds=(42, 7, 123), cm=None):
    cm = cm or H.cliente()
    lados = {"LARGO": 1, "CORTO": -1}
    por_seed = {}
    for seed in seeds:
        pares = H.muestrear(seed, clase="LC_ETF", n_tickers=tickers, n_dias=dias,
                            meses=meses, cm=cm)
        res = {l: [] for l in lados}
        n_eval = 0
        for tk, fecha in pares:
            df = H.cargar(tk, fecha, cm=cm)
            if len(df) < 20:
                continue
            n_eval += 1
            close = df["close"].to_numpy(float); high = df["high"].to_numpy(float)
            low = df["low"].to_numpy(float)
            vwap = H.vwap_sesion(df)
            atrb = _atr_bar(df)
            z = (close - vwap) / np.where(atrb > 0, atrb, np.nan)
            ult = len(close) - 6                 # no abrir en la ultima media hora (6 barras de 5m)
            for nombre, lado in lados.items():
                i_in = None
                for j in range(3, max(4, ult)):
                    if not (z[j] == z[j]):
                        continue
                    if lado > 0 and z[j] <= -k:
                        i_in = j; break
                    if lado < 0 and z[j] >= k:
                        i_in = j; break
                if i_in is None:
                    continue
                atr_abs = float(atrb[i_in])
                if not (atr_abs > 0):
                    continue
                stop = close[i_in] - lado * m * atr_abs
                r = _sim(close, high, low, vwap, i_in, lado, stop)
                if r:
                    res[nombre].append(r)
        print(f"# seed {seed}: {len(pares)} pares ETF+LC, {n_eval} con datos")
        for l in lados:
            por_seed[(l, seed)] = res[l]
    return lados, por_seed, seeds


def informe(lados, por_seed, seeds):
    filas = []
    for l in lados:
        for seed in seeds:
            filas.append(H.resumen_R(l, por_seed[(l, seed)], seed))
        filas.append(H.resumen_R(l, sum((por_seed[(l, s)] for s in seeds), []), "pooled"))
    # pooled LONG+SHORT
    todos = sum((por_seed[(l, s)] for l in lados for s in seeds), [])
    filas.append(H.resumen_R("L+S", todos, "pooled"))
    print(H.tabla_md(filas))
    print()
    for l in lados:
        perseed = [H.resumen_R(l, por_seed[(l, s)], s) for s in seeds]
        print(f"- **{l}**: {H.veredicto_global(perseed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=1.5)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--tickers", type=int, default=40)
    ap.add_argument("--dias", type=int, default=15)
    ap.add_argument("--meses", type=int, default=20)
    a = ap.parse_args()
    lados, por_seed, seeds = run(a.k, a.m, a.tickers, a.dias, a.meses)
    print()
    informe(lados, por_seed, seeds)
