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
import random
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


def _evaluar(df, prev_close, umbral):
    """Evalua el fade sobre un (ticker, dia) ya cargado. Devuelve {objGapFill, EOD}
    o None si no hay gap suficiente / no se confirma el giro."""
    if len(df) < 12 or prev_close <= 0:
        return None
    opn = df["open"].to_numpy(float); close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    gap = opn[0] / prev_close - 1.0
    if abs(gap) < umbral:
        return None
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
        return None
    return _sim(close, high, low, prev_close, i_in, lado, stop)


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
                prev_close = float(pc.loc[tk])
                df = H.cargar(tk, fecha, cm=cm)
                if len(df) >= 12 and prev_close > 0 and abs(df["open"].iloc[0] / prev_close - 1.0) >= umbral:
                    n_gap += 1
                r = _evaluar(df, prev_close, umbral)
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


# ===================== MODO HISTORIA (prescreen de gaps) =====================
def enumerar_gaps(umbral, meses, cm, cs_set):
    """Enumera TODOS los eventos (ticker, dia, cierre_previo) con |gap|>umbral en la
    ventana, usando SOLO barras diarias GROUPED (baratas: 1 request/dia para todo el
    mercado, sin minuto). El gap se mide open(dia)/close(dia previo)-1 (ajustado). El
    ticker debe estar en el universo PIT de ese mes. Devuelve lista de eventos."""
    desde, hasta = H._ventana(meses)
    dias = H._dias_sesion(desde, hasta, cm)
    uni_mes, eventos, prev_g = {}, [], None
    for i, d in enumerate(dias):
        g = massive.grouped_diario(d, cm=cm)
        if not g.empty:
            g = g.set_index("ticker")
        if prev_g is not None and not g.empty and len(prev_g):
            pc = prev_g["close"]
            ts = pd.Timestamp(d); clave = ts.strftime("%Y-%m")
            if clave not in uni_mes:
                ref = (ts.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                uni_mes[clave] = list(H._universo_mes(ref, "CS", 120, cm, cs_set))
            uni = uni_mes[clave]
            sub = g.reindex(uni); pcl = pc.reindex(uni)
            o = sub["open"].to_numpy(float); p = pcl.to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                gap = o / p - 1.0
            ok = np.isfinite(gap) & (p > 0) & (o > 0) & (np.abs(gap) >= umbral)
            for tk, is_ok, pv in zip(uni, ok, p):
                if is_ok:
                    eventos.append((tk, d, float(pv)))
        if not g.empty:
            prev_g = g
        if (i + 1) % 120 == 0:
            print(f"  ...prescreen {i+1}/{len(dias)} dias, {len(eventos)} eventos", flush=True)
    return eventos


def run_historia(umbral=0.03, meses=54, cap=1200, seeds=(42, 7, 123), cm=None):
    """Re-test con MUCHA historia: prescreen barato de gaps -> descarga minuto SOLO de
    los eventos. Los eventos se barajan (semilla fija) y se parten en 3 folds disjuntos
    (= 3 muestras independientes para el veredicto de robustez)."""
    cm = cm or H.cliente()
    cs_set, _ = U.set_common_stocks(cm=cm)
    print(f"Prescreen de gaps |gap|>{umbral:.0%} en ~{meses} meses...", flush=True)
    eventos = enumerar_gaps(umbral, meses, cm, cs_set)
    print(f"# eventos candidatos: {len(eventos)}", flush=True)
    random.Random(0).shuffle(eventos)
    eventos = eventos[:cap]
    folds = {seeds[0]: eventos[0::3], seeds[1]: eventos[1::3], seeds[2]: eventos[2::3]}
    modos = ["objGapFill", "EOD"]
    por_seed, con_fecha = {}, {m: [] for m in modos}
    for seed in seeds:
        res = {m: [] for m in modos}
        for tk, d, prev_close in folds[seed]:
            r = _evaluar(H.cargar(tk, d, cm=cm), prev_close, umbral)
            if r:
                for m in modos:
                    res[m].append({"R": r[m]})
                    con_fecha[m].append((d, r[m]))
        print(f"# seed {seed}: {len(folds[seed])} eventos candidatos, {len(res['EOD'])} operados", flush=True)
        for m in modos:
            por_seed[(m, seed)] = res[m]
    return modos, por_seed, seeds, con_fecha


def informe_historia(modos, por_seed, seeds, con_fecha):
    informe(modos, por_seed, seeds)
    print("\n### Split cronológico (train = primeras, test = últimas por fecha)")
    print("| modo | tramo | ops | expectancy (R) | IC95 |")
    print("|---|---|---:|---:|---|")
    for m in modos:
        datos = sorted(con_fecha[m], key=lambda x: x[0])
        R = [r for _, r in datos]
        k = int(len(R) * 0.6)
        for et, seg in (("train 60%", R[:k]), ("test 40%", R[k:])):
            s = H.resumen_R(m, [{"R": x} for x in seg], et)
            ic = f"[{s['lo']:+.3f}, {s['hi']:+.3f}]" if s["ops"] >= 5 and s["exp"] == s["exp"] else "-"
            exp = f"{s['exp']:+.3f}" if s["ops"] >= 5 and s["exp"] == s["exp"] else "-"
            print(f"| {m} | {et} | {s['ops']} | {exp} | {ic} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.03)
    ap.add_argument("--tickers", type=int, default=35)
    ap.add_argument("--dias", type=int, default=15)
    ap.add_argument("--meses", type=int, default=20)
    ap.add_argument("--historia", action="store_true",
                    help="re-test con prescreen de gaps sobre mucha historia (minuto solo de eventos)")
    ap.add_argument("--cap", type=int, default=1200, help="tope de eventos a operar (modo historia)")
    a = ap.parse_args()
    if a.historia:
        modos, por_seed, seeds, con_fecha = run_historia(a.umbral, a.meses, a.cap)
        print()
        informe_historia(modos, por_seed, seeds, con_fecha)
    else:
        modos, por_seed, seeds = run(a.umbral, a.tickers, a.dias, a.meses)
        print()
        informe(modos, por_seed, seeds)
