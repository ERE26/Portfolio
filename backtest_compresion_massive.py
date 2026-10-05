"""VOLATILIDAD: COMPRESION -> EXPANSION (breakout intradia SOLO tras compresion de rango).

Familia NUEVA (no probada en el repo): la hipotesis clasica de que una EXPANSION de rango
(breakout con continuacion) es mas probable DESPUES de un periodo de COMPRESION de volatilidad
(dias de rango estrecho tipo NR7, o una apertura inusualmente tranquila). Es la condicion que
el ORB probaba SIN filtro (y murio al quitar la supervivencia): aqui la pregunta es si
CONDICIONAR el breakout a compresion previa lo resucita. Largo Y corto (simetrico).

Rigor identico al resto (reutiliza el harness `backtest_nuevos_massive`):
  - Universo POINT-IN-TIME survivorship-free (`muestrear`: acciones comunes del GROUPED del
    mes anterior, con deslistadas). Sin look-ahead: el filtro de compresion usa SOLO sesiones
    PREVIAS (NR7) o la MAÑANA del propio dia (apertura, antes de la entrada).
  - Breakout CONFIRMADO del rango de apertura (1a media hora): entra al CIERRE de la vela que
    rompe (largo sobre el maximo, corto bajo el minimo) + slippage; stop = extremo opuesto del
    rango; salida EOD (+ variante trailing 2*ATR). Coste de borrow en cortos.
  - 3 semillas (42/7/123), bootstrap IC95, veredicto ROBUSTO solo si IC95>0 en las 3.

Se evaluan TRES cohortes sobre la MISMA muestra para aislar el efecto de la compresion:
  - BASELINE : breakout en TODOS los dias muestreados (sin filtro).
  - NR7       : solo dias cuya sesion PREVIA fue la de rango mas estrecho de las ultimas 7.
  - OR_TIGHT  : solo dias cuyo rango de apertura (30 min) es <50% del ATR diario previo.
Si la compresion aporta, una cohorte filtrada debe BATIR al baseline con IC95>0 en las 3.

  python backtest_compresion_massive.py [--tickers 20] [--dias 12] [--meses 18]

SOLO research/paper. Cache de barras en datos/cache (gitignored).
"""
import argparse

import numpy as np
import pandas as pd

import backtest_nuevos_massive as H
from fuentes import massive as mv

OR_BARRAS = 6          # rango de apertura = primeras 6 velas de 5 min (09:30-10:00)
LOOKBACK_CAL = 45      # dias de calendario hacia atras (>=14 sesiones para NR7/ATR)
NR_N = 7               # NR7: rango mas estrecho de las ultimas 7 sesiones
OR_TIGHT_Q = 0.5       # OR_width / ATR_diario_previo < 0.5 -> apertura comprimida
MODOS = {"EOD": ("eod", 0.0), "EOD+trail2ATR": ("trail", 2.0)}


def _fetch_win(ticker, fecha, cm):
    """5-min RTH en [fecha-LOOKBACK, fecha] (incluye sesiones previas para el filtro)."""
    desde = (pd.Timestamp(fecha) - pd.Timedelta(days=LOOKBACK_CAL)).strftime("%Y-%m-%d")
    try:
        df = mv.barras_intradia(ticker, "5Min", desde, fecha, solo_regular=True, cm=cm, cache=True)
    except Exception:
        return None
    return df if len(df) >= 100 else None


def _rangos_diarios(df):
    """Serie fecha->rango (high-low) por sesion, en orden cronologico."""
    idx = pd.DatetimeIndex(df.index)
    g = df.groupby(idx.normalize())
    return g["high"].max() - g["low"].min()


def _sim_breakout(close, high, low, or_hi, or_lo, modo, trail_atr, atr_abs):
    """Primer breakout CONFIRMADO del rango de apertura (largo o corto). R neto o None."""
    n = len(close)
    lado = entry = i_in = None
    for j in range(OR_BARRAS, n):
        if close[j] > or_hi:
            lado, entry, i_in = +1, close[j] * (1 + H.SLIP), j; break
        if close[j] < or_lo:
            lado, entry, i_in = -1, close[j] * (1 - H.SLIP), j; break
    if lado is None:
        return None
    if lado > 0:
        stop = or_lo; rdist = entry - stop
    else:
        stop = or_hi; rdist = stop - entry
    if rdist <= 0:
        return None
    run_ext = entry
    exit_px, reason = float(close[n - 1]), "EOD"
    for j in range(i_in, n):
        hi, lo, cl = float(high[j]), float(low[j]), float(close[j])
        if lado > 0:
            if lo <= stop:
                reason = "TRAIL" if stop > entry * 1.0001 else "SL"
                exit_px = stop * (1 - H.SLIP_STOP); break
            run_ext = max(run_ext, hi)
            if modo == "trail" and run_ext >= entry + rdist:
                nuevo = run_ext - trail_atr * atr_abs
                if nuevo > stop:
                    stop = nuevo
        else:
            if hi >= stop:
                reason = "TRAIL" if stop < entry * 0.9999 else "SL"
                exit_px = stop * (1 + H.SLIP_STOP); break
            run_ext = min(run_ext, lo)
            if modo == "trail" and run_ext <= entry - rdist:
                nuevo = run_ext + trail_atr * atr_abs
                if nuevo < stop:
                    stop = nuevo
        if j == n - 1:
            exit_px = cl * (1 - H.SLIP) if lado > 0 else cl * (1 + H.SLIP)
            reason = "EOD"; break
    R = (exit_px - entry) / rdist if lado > 0 else (entry - exit_px) / rdist
    if lado < 0:
        R -= (H.BORROW_CORTO * entry) / rdist        # coste de borrow en unidades R
    return {"R": R, "lado": lado, "reason": reason}


def _eval_dia(df, fecha):
    """Devuelve dict con R por modo y las banderas de cohorte (nr7, or_tight), o None."""
    idx = pd.DatetimeIndex(df.index)
    d_norm = pd.Timestamp(fecha).normalize()
    pos = np.where(idx.normalize() == d_norm)[0]
    if len(pos) < OR_BARRAS + 3:
        return None
    rangos = _rangos_diarios(df)
    previos = rangos[rangos.index < d_norm]
    if len(previos) < NR_N:
        return None
    ult7 = previos.iloc[-NR_N:]
    nr7 = bool(ult7.iloc[-1] == ult7.min())          # la sesion previa fue la mas estrecha de 7
    atr_prev = float(previos.iloc[-14:].mean())
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    p = pos
    or_hi = float(high[p[:OR_BARRAS]].max()); or_lo = float(low[p[:OR_BARRAS]].min())
    or_w = or_hi - or_lo
    or_tight = bool(atr_prev > 0 and or_w / atr_prev < OR_TIGHT_Q)
    atr_abs = float(np.nanmedian((high[p] - low[p]))) * 3.0
    cp, ch, cl_ = close[p], high[p], low[p]
    out = {"nr7": nr7, "or_tight": or_tight, "R": {}}
    for m, (modo, tr) in MODOS.items():
        r = _sim_breakout(cp, ch, cl_, or_hi, or_lo, modo, tr, atr_abs)
        if r is not None:
            out["R"][m] = r["R"]
    return out if out["R"] else None


def run(tickers=20, dias=12, meses=18, seeds=(42, 7, 123)):
    cm = H.cliente()
    cohortes = ("BASELINE", "NR7", "OR_TIGHT")
    # res[cohorte][modo][seed] = lista de R
    res = {c: {m: {s: [] for s in seeds} for m in MODOS} for c in cohortes}
    ndias = {s: 0 for s in seeds}
    cache = {}
    for s in seeds:
        pares = H.muestrear(s, clase="CS", n_tickers=tickers, n_dias=dias, meses=meses, cm=cm)
        print(f"[seed {s}] {len(pares)} ticker-dias PIT (survivorship-free)")
        for k, (tk, d) in enumerate(pares):
            ev = cache.get((tk, d))
            if ev is None:
                df = _fetch_win(tk, d, cm)
                ev = _eval_dia(df, d) if df is not None else False
                cache[(tk, d)] = ev
                if (k + 1) % 100 == 0:
                    print(f"   ... {k+1}/{len(pares)} | llamadas={cm.llamadas}")
            if not ev:
                continue
            ndias[s] += 1
            for m, R in ev["R"].items():
                res["BASELINE"][m][s].append(R)
                if ev["nr7"]:
                    res["NR7"][m][s].append(R)
                if ev["or_tight"]:
                    res["OR_TIGHT"][m][s].append(R)
        print(f"   seed {s}: {ndias[s]} dias con breakout | llamadas acumuladas={cm.llamadas}")
    _informe(res, cohortes, seeds)
    return res


def _informe(res, cohortes, seeds):
    print("\n" + "=" * 96)
    print("COMPRESION -> EXPANSION | breakout confirmado del rango de apertura (largo+corto, EOD)")
    print("la pregunta: ¿CONDICIONAR a compresion previa (NR7 / apertura tight) da edge al breakout?")
    print("=" * 96)
    base = {}
    for c in cohortes:
        for m in MODOS:
            filas = [H.resumen_R(f"{c}|{m}", [{"R": r} for r in res[c][m][s]], s) for s in seeds]
            pooled = [r for s in seeds for r in res[c][m][s]]
            fp = H.resumen_R(f"{c}|{m}", [{"R": r} for r in pooled], "POOL")
            glob = H.veredicto_global(filas)
            exp = fp["exp"] if fp["ops"] >= 5 else float("nan")
            ic = (f"[{fp['lo']:+.3f},{fp['hi']:+.3f}]" if fp["ops"] >= 5 else "-")
            print(f"\n### {c} | {m}")
            print(H.tabla_md(filas + [fp]))
            print(f"-> GLOBAL: {glob}")
            if c == "BASELINE":
                base[m] = exp
            elif exp == exp and m in base and base[m] == base[m]:
                delta = exp - base[m]
                print(f"   (vs baseline {base[m]:+.3f}R -> {'MEJORA' if delta>0 else 'NO mejora'} {delta:+.3f}R)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=20)
    ap.add_argument("--dias", type=int, default=12)
    ap.add_argument("--meses", type=int, default=18)
    ap.add_argument("--seeds", type=str, default="42,7,123")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(","))
    run(a.tickers, a.dias, a.meses, seeds=seeds)
