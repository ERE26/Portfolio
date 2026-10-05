"""ESTRATEGIA 1 - LADO CORTO (breakdown), simetrico al ORB largo que murio.

Reglas (espejo del ORB confirmado):
  - Rango de apertura = 1a vela de 5m (09:30-09:35). Debe ser BAJISTA (close<open):
    direccion = color de la 1a vela, como en el paper (solo short aqui).
  - Entrada: cuando una vela CIERRA POR DEBAJO del minimo del rango (ruptura
    confirmada a la baja). Se vende a ese cierre + slippage (peor precio).
  - Stop inicial: MAXIMO del rango de apertura (cubrir si rebota).
  - Salida: EOD (cubrir al cierre) y variante TRAILING (tras -1R a favor, stop baja
    bajo el minimo recorrido a trail*ATR).
  - Coste: SLIP por lado, SLIP_STOP en stops, + BORROW_CORTO por ser corto.

Point-in-time (universo_pit, sin supervivencia), 3 semillas (42/7/123), bootstrap IC95.

  python backtest_short_massive.py [--tickers 30] [--dias 15] [--meses 20]
"""
import argparse

import numpy as np

import backtest_nuevos_massive as H


def _sim_corto(close, high, low, or_hi, or_lo, modo, trail_atr, atr_abs):
    """Corto en un dia. Entra al cierre de la 1a vela que cierra bajo or_lo."""
    n = len(close)
    entry = None
    for j in range(1, n):
        if close[j] < or_lo:
            entry = close[j] * (1 - H.SLIP)       # vende al cierre que confirma (slippage en contra)
            i_in = j
            break
    if entry is None:
        return None
    rdist = or_hi - entry                          # distancia al stop (arriba)
    if rdist <= 0:
        return None
    stop = or_hi
    run_low = entry
    exit_px, reason = float(close[n - 1]), "EOD"
    for j in range(i_in, n):
        hi, lo, cl = float(high[j]), float(low[j]), float(close[j])
        if hi >= stop:                             # stop tocado: cubrir arriba
            reason = "TRAIL" if stop < entry * 0.9999 else "SL"
            exit_px = stop * (1 + H.SLIP_STOP); break
        run_low = min(run_low, lo)
        if modo == "trail" and run_low <= entry - rdist:   # tras -1R a favor, trailing
            nuevo = run_low + trail_atr * atr_abs
            if nuevo < stop:
                stop = nuevo
        if j == n - 1:
            exit_px = cl * (1 + H.SLIP); reason = "EOD"; break
    R = (entry - exit_px) / rdist                   # corto: gana si exit < entry
    R -= (H.BORROW_CORTO * entry) / rdist           # coste de borrow en unidades R
    return {"R": R, "reason": reason}


def run(tickers=30, dias=15, meses=20, seeds=(42, 7, 123), cm=None):
    cm = cm or H.cliente()
    modos = {"EOD": ("eod", 0.0), "EOD+trail2ATR": ("trail", 2.0)}
    por_seed = {}        # (modo, seed) -> lista R
    for seed in seeds:
        pares = H.muestrear(seed, clase="CS", n_tickers=tickers, n_dias=dias, meses=meses, cm=cm)
        res = {m: [] for m in modos}
        n_eval = 0
        for tk, fecha in pares:
            df = H.cargar(tk, fecha, cm=cm)
            if len(df) < 12:
                continue
            n_eval += 1
            close = df["close"].to_numpy(float); high = df["high"].to_numpy(float)
            low = df["low"].to_numpy(float); opn = df["open"].to_numpy(float)
            if close[0] >= opn[0]:                  # 1a vela debe ser BAJISTA
                continue
            or_hi, or_lo = float(high[0]), float(low[0])
            atr_abs = H.atr_dia(df)
            if not (atr_abs == atr_abs):
                continue
            for m, (modo, tr) in modos.items():
                r = _sim_corto(close, high, low, or_hi, or_lo, modo, tr, atr_abs)
                if r is not None:
                    res[m].append(r)
        print(f"# seed {seed}: {len(pares)} pares PIT, {n_eval} con datos")
        for m in modos:
            por_seed[(m, seed)] = res[m]
    return modos, por_seed, seeds


def informe(modos, por_seed, seeds):
    filas = []
    pooled_rows = []
    for m in modos:
        for seed in seeds:
            filas.append(H.resumen_R(f"{m}", por_seed[(m, seed)], seed))
        todos = sum((por_seed[(m, s)] for s in seeds), [])
        pr = H.resumen_R(m, todos, "pooled")
        pooled_rows.append(pr)
        filas.append(pr)
    print(H.tabla_md(filas))
    print()
    for m in modos:
        perseed = [H.resumen_R(m, por_seed[(m, s)], s) for s in seeds]
        print(f"- **{m}**: {H.veredicto_global(perseed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=30)
    ap.add_argument("--dias", type=int, default=15)
    ap.add_argument("--meses", type=int, default=20)
    a = ap.parse_args()
    modos, por_seed, seeds = run(a.tickers, a.dias, a.meses)
    print()
    informe(modos, por_seed, seeds)
