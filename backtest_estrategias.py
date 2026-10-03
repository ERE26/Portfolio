"""BAKE-OFF de hipótesis intradía: prueba varias LÓGICAS DE ENTRADA distintas sobre la
MISMA muestra aleatoria (tickers y días al azar), con el mismo motor de simulación OOS de
backtest_intradia (features point-in-time, bracket TP/SL + break-even + time-stop, slippage)
y las mismas salidas. Así se aísla el efecto de la ENTRADA y se ve cuál (si alguna) tiene
edge positivo out-of-sample.

Hipótesis:
  momentum      - la actual (ruptura/fuerza sobre VWAP a cualquier hora). BASELINE.
  momentum_am   - igual, pero SOLO en los primeros 90 min (la hora con más dirección).
  pullback_vwap - comprar el RETROCESO a un VWAP al alza (soporte), no la extensión.
  orb           - Opening-Range Breakout: romper el máximo de los primeros 30 min, en las
                  primeras 2h, con volumen y fuerza relativa.

  python backtest_estrategias.py [--tickers 40] [--dias 12] [--meses 15] [--seed 42]

Mismos avisos honestos que backtest_intradia (sesgo de supervivencia, IEX, 1 entrada/día).
"""
import argparse
import random

import numpy as np
import pandas as pd

import gestionar as G
import backtest_intradia as bi
from fuentes import alpaca


def e_momentum(f, fr, c):
    """BASELINE: los gates de confluencia actuales (ruptura/fuerza sobre VWAP)."""
    return bi._pasa_gates(f, fr, c["p"])


def e_momentum_am(f, fr, c):
    """Momentum actual pero SOLO en los primeros 90 min de sesión."""
    return c["mins"] <= 90 and bi._pasa_gates(f, fr, c["p"])


def e_pullback_vwap(f, fr, c):
    """Comprar el RETROCESO al VWAP en un valor fuerte: día alcista, MUY cerca del VWAP por
    arriba (no extendido), girando al alza, con volumen y fuerza relativa. Es comprar
    soporte, no perseguir la ruptura."""
    return (f["ret_sesion"] >= G.RET_MIN and fr > G.MARGEN_FR
            and f["rvol"] >= c["p"]["rvol_min"]
            and 0 < f["ext"] < 0.8            # pegado al VWAP (retroceso), no estirado
            and f["ret_reciente"] > 0 and f["pend"] > 0)   # rebotando / girando arriba


def e_orb(f, fr, c):
    """Opening-Range Breakout BIEN HECHO: romper el máximo de los primeros 30 min, dentro de
    las primeras 2h, con volumen y fuerza relativa. STOP en el MÍNIMO del rango de apertura
    (riesgo propio del setup, no un ATR genérico) -> se fija en ctx['sl_px']."""
    ok = (c["or_high"] is not None and 2 <= c["k"] <= 8
          and f["precio"] > c["or_high"]
          and f["rvol"] >= c["p"]["rvol_min"] and fr > G.MARGEN_FR
          and f["ret_reciente"] > 0)
    if ok:
        c["sl_px"] = c["or_low"]        # stop al mínimo del rango de apertura
    return ok


def e_gap_and_go(f, fr, c):
    """Gap-and-go: el valor ABRIÓ con hueco al alza >=3% (proxy de catalizador/noticia),
    y en la primera 1.5h sigue por encima del VWAP, con volumen, fuerza relativa y sin
    haberse girado (momentum reciente > 0). Es el momentum CON catalizador que de verdad
    continúa, no el genérico."""
    return (f["gap"] >= 0.03 and c["mins"] <= 90
            and f["dist_vwap"] > 0 and f["rvol"] >= c["p"]["rvol_min"]
            and fr > G.MARGEN_FR and f["ret_reciente"] > 0
            and 0 < f["ext"] < c["p"]["ext_max"])


def e_gap_orb(f, fr, c):
    """GAP-AND-GO textbook: abrió con hueco >=2% (catalizador) Y rompe el máximo del rango
    de apertura en las primeras 2h, con volumen y fuerza. Stop al mínimo del rango. Es el
    setup que de verdad operan los day-traders profesionales (catalizador + confirmación +
    riesgo propio)."""
    ok = (f["gap"] >= 0.02 and c["or_high"] is not None and 2 <= c["k"] <= 8
          and f["precio"] > c["or_high"] and f["dist_vwap"] > 0
          and f["rvol"] >= c["p"]["rvol_min"] and fr > G.MARGEN_FR and f["ret_reciente"] > 0)
    if ok:
        c["sl_px"] = c["or_low"]
    return ok


ESTRATEGIAS = {"momentum": e_momentum, "pullback_vwap": e_pullback_vwap,
               "orb": e_orb, "gap_and_go": e_gap_and_go, "gap_orb": e_gap_orb}


def _simular_dia(entrada_fn, close, high, low, pre, dia_pos, spy_ret_pos, p):
    or_high = max(high[dia_pos[0]], high[dia_pos[1]]) if len(dia_pos) >= 2 else None
    or_low = min(low[dia_pos[0]], low[dia_pos[1]]) if len(dia_pos) >= 2 else None
    for k, pos in enumerate(dia_pos[:-1]):
        f = bi._feats(close, pre, pos)
        if f is None:
            continue
        fr = f["ret_sesion"] - spy_ret_pos.get(pos, 0.0)
        ctx = {"p": p, "mins": k * 15, "k": k, "or_high": or_high, "or_low": or_low, "sl_px": None}
        if entrada_fn(f, fr, ctx):
            return bi._simular_trade(close, high, low, pre, pos, dia_pos[-1], f, p,
                                     sl_px=ctx.get("sl_px"))
    return None


def bakeoff(tickers_n=40, dias=12, meses=15, seed=42):
    rng = random.Random(seed)
    uni = bi._universo(n=150)
    rng.shuffle(uni)
    uni = uni[:tickers_n]
    desde = (pd.Timestamp.now() - pd.Timedelta(days=int(meses * 31))).strftime("%Y-%m-%d")
    spy_idx, spy_ret = bi._spy_por_dia(meses)
    spy_map = pd.Series(spy_ret, index=pd.DatetimeIndex(spy_idx))
    p = bi.NUEVO
    res = {k: [] for k in ESTRATEGIAS}
    n_td = 0
    print(f"\n### BAKE-OFF INTRADÍA | {len(uni)} tickers x {dias} días al azar | {meses} meses | seed {seed}")
    for t in uni:
        try:
            df = alpaca.historico(t, "15Min", desde=desde, max_paginas=12)
        except Exception:
            continue
        if len(df) < 300:
            continue
        close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
        pre = bi._precarga(df)
        idx = pd.DatetimeIndex(df.index)
        spy_al = spy_map.reindex(idx, method="ffill").to_numpy()
        spy_ret_pos = {i: (0.0 if np.isnan(spy_al[i]) else float(spy_al[i])) for i in range(len(idx))}
        por_dia = pd.Series(np.arange(len(df)), index=idx).groupby(idx.normalize())
        dias_ok = [g.to_numpy() for _, g in por_dia if len(g) >= 15]
        if len(dias_ok) < 2:
            continue
        for dia_pos in rng.sample(dias_ok, min(dias, len(dias_ok))):
            n_td += 1
            dp = list(dia_pos)
            for nombre, fn in ESTRATEGIAS.items():
                tr = _simular_dia(fn, close, high, low, pre, dp, spy_ret_pos, p)
                if tr is not None:
                    res[nombre].append(tr)
    print(f"ticker-días evaluados: {n_td}\n")
    print(f"{'estrategia':<15}{'ops':>5}{'acierto':>9}{'expectancy':>13}{'IC95':>20}{'PF':>6}  veredicto")
    filas = []
    for nombre in ESTRATEGIAS:
        filas.append((nombre, _stats(res[nombre])))
    for nombre, s in sorted(filas, key=lambda x: -(x[1]["exp"] if x[1] else -9)):
        if not s:
            print(f"{nombre:<15}{'0':>5}  (sin operaciones)")
            continue
        print(f"{nombre:<15}{s['n']:>5}{s['win']*100:>7.0f}% {s['exp']:>+11.3f}R  "
              f"[{s['lo']:+.3f},{s['hi']:+.3f}]{s['pf']:>6.2f}  {s['verd']}")
    return res


def _stats(trades):
    if len(trades) < 5:
        return None
    R = np.array([t["R"] for t in trades])
    w = R[R > 0]; l = R[R <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    lo, hi = bi._boot_ci(R)
    verd = "FUNCIONA (edge+)" if lo > 0 else ("sin edge claro" if hi > 0 else "pierde")
    return {"n": len(R), "win": len(w) / len(R), "exp": float(R.mean()), "lo": lo, "hi": hi,
            "pf": pf, "verd": verd}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=40)
    ap.add_argument("--dias", type=int, default=12)
    ap.add_argument("--meses", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    bakeoff(a.tickers, a.dias, a.meses, a.seed)
