"""BACKTEST MONTE-CARLO del intradía: coge TICKERS y DÍAS al azar y simula la estrategia
REAL (mismas features de gestionar.señal, los mismos gates de confluencia, el bracket
TP/SL, el break-even y el time-stop) sobre barras históricas de 15m, barra a barra y
POINT-IN-TIME (sin look-ahead). Mide si hay edge y COMPARA reglas NUEVAS vs VIEJAS.

  python backtest_intradia.py [--tickers 30] [--dias 10] [--meses 15] [--seed 42]

Por qué es fiel: las features intradía (VWAP, RVOL, ret_sesión, ATR) son CAUSALES —se
reinician por día y solo usan el pasado—, así que calcularlas una vez sobre el histórico
y leer la barra i equivale a haberlas calculado con datos solo hasta i. La entrada se
rellena al OPEN de la barra SIGUIENTE a la señal (nunca la misma barra). TP/SL se evalúan
intrabar con high/low (y si ambos caben en la barra, se asume SL primero: conservador).

Avisos honestos:
  (1) SESGO DE SUPERVIVENCIA: se muestrea el universo líquido ACTUAL (no incluye
      deslistados/quebrados) -> sesga AL ALZA. El veredicto es optimista por construcción.
  (2) Datos IEX (no SIP): menos volumen/cobertura que el feed de pago.
  (3) El SUELO DE CONVICCIÓN es un z-score de LOTE (relativo a los candidatos del mismo
      ciclo); no se puede replicar por-barra, así que aquí se testean los GATES DUROS
      (RVOL, VWAP, extensión, anti-parábola, fuerza relativa, sesión, momentum reciente,
      large-cap). Con el suelo en 0.0 esto no cambia nada hoy.
  (4) Una entrada por ticker-día (la primera válida): mide calidad de entrada, no re-entradas.
  (5) Slippage modelado (entrada/salida y extra en stops); comisión 0 (paper).
"""
import argparse
import random

import numpy as np
import pandas as pd

import gestionar as G
from fuentes import alpaca
from fuentes import tradingview as TV
from patrones import indicadores as ind
from patrones.intradia import rasgos_intradia, rvol

SLIP = 0.0004          # 4 bps por lado (entrada/salida a mercado)
SLIP_STOP = 0.0008     # 8 bps extra al tocar stop (peor fill, realista)
NY = "America/New_York"

# Juegos de parámetros a comparar
NUEVO = {"nombre": "NUEVO", "ext_max": G.EXT_MAX, "rvol_min": 2.0, "breakeven": True,
         "k_pt": G.CONFIG["15Min"]["k_pt"], "k_sl": G.CONFIG["15Min"]["k_sl"]}
VIEJO = {"nombre": "VIEJO", "ext_max": 4.0, "rvol_min": 1.5, "breakeven": False,
         "k_pt": G.CONFIG["15Min"]["k_pt"], "k_sl": G.CONFIG["15Min"]["k_sl"]}


def _universo(n=150, cap_min=1e9, cap_max=TV.MAX_CAP_INTRADIA):
    """Universo de muestreo: acciones US líquidas en [cap_min, cap_max] (mid-caps por
    defecto). cap_max=None -> sin tope (para el test de large-caps)."""
    cols = ["name", "Value.Traded", "market_cap_basic"]
    filtros = [
        {"left": "close", "operation": "in_range", "right": [5, 5000]},
        {"left": "Value.Traded", "operation": "greater", "right": 10_000_000},
        {"left": "average_volume_10d_calc", "operation": "greater", "right": 700_000},
        {"left": "type", "operation": "equal", "right": "stock"},
        {"left": "is_primary", "operation": "equal", "right": True},
    ]
    if cap_max:
        filtros.append({"left": "market_cap_basic", "operation": "in_range",
                        "right": [cap_min, cap_max]})
    else:
        filtros.append({"left": "market_cap_basic", "operation": "greater", "right": 50e9})
    rows, _ = TV.scan(cols, sort_by="Value.Traded", filtros=filtros, n=n)
    return [d["symbol"] for d in rows if d.get("symbol")]


def _precarga(df):
    """Precalcula (una vez por ticker) las series causales: dist_vwap, ret_sesion, rvol, atr."""
    r = rasgos_intradia(df)
    atr = ind.atr(df["high"], df["low"], df["close"], 14)
    return {"dist_vwap": r["intra_dist_vwap"].to_numpy(),
            "ret_sesion": r["intra_ret_sesion"].to_numpy(),
            "rvol": rvol(df).to_numpy(),
            "atr": atr.to_numpy(),
            "gap": r["intra_gap"].to_numpy()}            # gap de apertura (open vs cierre previo)


def _feats(df_close, pre, i):
    """Features point-in-time en la posición i (igual que gestionar.señal)."""
    precio = float(df_close[i])
    if precio <= 0 or i < 20:
        return None
    atr_pct = float(pre["atr"][i]) / precio if precio else 0.0
    dv = float(pre["dist_vwap"][i])
    ext = dv / atr_pct if atr_pct > 0 else 0.0
    k = min(4, i)
    ret_reciente = float(df_close[i] / df_close[i - k] - 1.0) if k >= 1 else 0.0
    y = np.log(df_close[i - 4:i + 1]); x = np.arange(len(y))
    pend = float(np.polyfit(x, y, 1)[0]) if len(y) >= 3 else 0.0
    return {"precio": precio, "dist_vwap": dv, "ext": ext, "atr_pct": atr_pct,
            "rvol": float(pre["rvol"][i]), "ret_sesion": float(pre["ret_sesion"][i]),
            "ret_reciente": ret_reciente, "pend": pend, "gap": float(pre["gap"][i])}


def _pasa_gates(f, fr, p):
    """Los mismos gates de confluencia que gestionar.rankear_conviccion + escanear."""
    if f["rvol"] < p["rvol_min"]:
        return False
    if f["dist_vwap"] <= 0:                              # sobre VWAP
        return False
    if not (0 < f["ext"] < p["ext_max"]):               # no sobre-extendido
        return False
    if f["ext"] >= G.EXT_PARAB and f["ret_sesion"] >= G.RET_PARAB:   # anti-parábola
        return False
    if fr <= G.MARGEN_FR:                                # fuerza relativa > mercado
        return False
    if f["ret_sesion"] < G.RET_MIN:                     # sesión con sustancia
        return False
    if f["ret_reciente"] <= 0:                          # momentum reciente (no gap-and-fade)
        return False
    return True


def _reversion(dv_arr, close, j):
    bajo = float((dv_arr[max(0, j - 3):j + 1] < 0).mean())
    y = np.log(close[max(0, j - 4):j + 1]); x = np.arange(len(y))
    pend = float(np.polyfit(x, y, 1)[0]) if len(y) >= 3 else 0.0
    return (bajo >= 0.75) and (pend < 0) and (dv_arr[j] < -0.003)


def _simular_dia(df_close, high, low, pre, dia_pos, spy_ret_pos, p):
    """Primera entrada válida del día y su simulación hasta la salida. None si no hay."""
    for pos in dia_pos[:-1]:                             # hace falta barra siguiente para entrar
        f = _feats(df_close, pre, pos)
        if f is None:
            continue
        fr = f["ret_sesion"] - spy_ret_pos.get(pos, 0.0)
        if not _pasa_gates(f, fr, p):
            continue
        return _simular_trade(df_close, high, low, pre, pos, dia_pos[-1], f, p)
    return None


def _simular_trade(close, high, low, pre, pos_sig, dia_last, f, p, sl_px=None):
    i_in = pos_sig + 1
    entry = float(close[i_in]) * (1 + SLIP)             # fill al open~close de la barra sig. + slippage
    atrp = f["atr_pct"]
    if atrp <= 0:
        return None
    if sl_px is not None and sl_px < entry:             # stop propio del setup (p.ej. mínimo del rango)
        rdist = entry - sl_px
    else:
        rdist = p["k_sl"] * atrp * entry                # = entry - sl (1R en $), por ATR
    if rdist <= 0:
        return None
    sl = entry - rdist
    tp = entry + p["k_pt"] * atrp * entry
    stop, movido, reason, exit_px = sl, False, "EOD", float(close[dia_last])
    for j in range(i_in, dia_last + 1):
        hi, lo, cl = float(high[j]), float(low[j]), float(close[j])
        mins = (j - i_in) * 15
        if lo <= stop:                                  # SL / BE (conservador: antes que TP)
            exit_px = stop * (1 - SLIP_STOP); reason = "BE" if movido else "SL"; break
        if hi >= tp:
            exit_px = tp * (1 - SLIP); reason = "TP"; break
        if p["breakeven"] and not movido and hi >= entry + rdist:   # +1R -> stop a entrada
            stop, movido = entry, True
        if _reversion(pre["dist_vwap"], close, j):
            exit_px = cl * (1 - SLIP); reason = "REV"; break
        if mins >= G.MAX_DURA_MIN:
            exit_px = cl * (1 - SLIP); reason = "TS_duro"; break
        if mins >= G.MAX_VERDE_MIN and (cl - entry) >= 0:
            exit_px = cl * (1 - SLIP); reason = "TS_verde"; break
        if j == dia_last:
            exit_px = cl * (1 - SLIP); reason = "EOD"; break
    pnl = exit_px - entry
    return {"R": pnl / rdist, "pnl_pct": pnl / entry, "reason": reason, "mins": (j - i_in) * 15}


def _spy_por_dia(meses):
    desde = (pd.Timestamp.now() - pd.Timedelta(days=int(meses * 31))).strftime("%Y-%m-%d")
    spy = alpaca.historico("SPY", "15Min", desde=desde, max_paginas=12)
    ser = rasgos_intradia(spy)["intra_ret_sesion"]
    return spy.index, ser.to_numpy()


def backtest(tickers_n=30, dias=10, meses=15, seed=42, cap_max=TV.MAX_CAP_INTRADIA, etiqueta="mid-caps"):
    rng = random.Random(seed)
    uni = _universo(n=150, cap_max=cap_max)
    rng.shuffle(uni)
    uni = uni[:tickers_n]
    desde = (pd.Timestamp.now() - pd.Timedelta(days=int(meses * 31))).strftime("%Y-%m-%d")
    # SPY: ret de sesión por timestamp (para la fuerza relativa)
    spy_idx, spy_ret = _spy_por_dia(meses)
    spy_map = pd.Series(spy_ret, index=pd.DatetimeIndex(spy_idx))
    res = {p["nombre"]: [] for p in (NUEVO, VIEJO)}
    n_td = 0
    print(f"\n### MUESTRA: {etiqueta} | {len(uni)} tickers x {dias} días al azar | últimos {meses} meses | seed {seed}")
    for t in uni:
        try:
            df = alpaca.historico(t, "15Min", desde=desde, max_paginas=12)
        except Exception:
            continue
        if len(df) < 300:
            continue
        close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
        pre = _precarga(df)
        idx = pd.DatetimeIndex(df.index)
        # spy ret alineado a las barras de este ticker (asof)
        spy_al = spy_map.reindex(idx, method="ffill").to_numpy()
        spy_ret_pos = {i: (0.0 if np.isnan(spy_al[i]) else float(spy_al[i])) for i in range(len(idx))}
        # días disponibles con sesión completa (>= 15 barras)
        por_dia = pd.Series(np.arange(len(df)), index=idx).groupby(idx.normalize())
        dias_ok = [(d, g.to_numpy()) for d, g in por_dia if len(g) >= 15]
        if len(dias_ok) < 2:
            continue
        muestra = rng.sample(dias_ok, min(dias, len(dias_ok)))
        for _, dia_pos in muestra:
            n_td += 1
            for p in (NUEVO, VIEJO):
                tr = _simular_dia(close, high, low, pre, list(dia_pos), spy_ret_pos, p)
                if tr is not None:
                    res[p["nombre"]].append(tr)
    print(f"ticker-días evaluados: {n_td}")
    for p in (NUEVO, VIEJO):
        _resumen(p["nombre"], res[p["nombre"]], n_td)
    return res


def _boot_ci(xs, n=2000, seed=1):
    if len(xs) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.array(xs)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _resumen(nombre, trades, n_td):
    if not trades:
        print(f"\n[{nombre}] 0 operaciones (gates demasiado estrictos sobre la muestra)")
        return
    R = np.array([t["R"] for t in trades])
    w = R[R > 0]; l = R[R <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    lo, hi = _boot_ci(R)
    veredicto = "FUNCIONA (edge+)" if lo > 0 else ("SIN EDGE CLARO" if hi > 0 else "PIERDE")
    print(f"\n[{nombre}]  {len(trades)} ops ({len(trades)/max(n_td,1)*100:.0f}% de los ticker-días dispararon)")
    print(f"   acierto {len(w)/len(R)*100:.0f}%  | expectancy {R.mean():+.3f}R  (IC95 [{lo:+.3f}, {hi:+.3f}])")
    print(f"   R medio win {w.mean() if len(w) else 0:+.2f} | R medio loss {l.mean() if len(l) else 0:+.2f} | PF {pf:.2f}")
    cierres = {}
    for t in trades:
        cierres[t["reason"]] = cierres.get(t["reason"], 0) + 1
    tot = len(trades)
    rr = {k: f"{v}({v/tot*100:.0f}%)" for k, v in sorted(cierres.items(), key=lambda x: -x[1])}
    print(f"   cómo cerró: {rr}")
    print(f"   -> VEREDICTO: {veredicto}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=30)
    ap.add_argument("--dias", type=int, default=10)
    ap.add_argument("--meses", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--largecaps", action="store_true", help="muestrea large-caps (>$50B) para comparar")
    a = ap.parse_args()
    if a.largecaps:
        backtest(a.tickers, a.dias, a.meses, a.seed, cap_max=None, etiqueta="LARGE-CAPS (>$50B)")
    else:
        backtest(a.tickers, a.dias, a.meses, a.seed, etiqueta="mid-caps (<$20B)")
