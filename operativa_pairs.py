"""LIBRO PAIRS / STAT-ARB intradía en PAPER — la única SEÑAL intradía real del estudio.

El backtest (docs/statarb_fondo_resultado.md, docs/intradia_ronda2_resultado.md) mostró que el
spread de pares emparejados SÍ revierte intradía (+1,5 bp bruto, 62% acierto, robusto), pero
MUERE a coste de mercado (break-even ~1 bp/pata). La pregunta abierta: ¿con ejecución real y
size pequeño cruza a positivo, o el muro de ejecución lo mata también en vivo? Esto lo MIDE en
paper (como la Fase 0 del ORB midió su slippage).

Market-neutral: cuando el z-score del spread de un par se desvía >Z_IN, se va LARGO de la pata
barata y CORTO de la cara; se cierra al revertir (|z|<Z_OUT), al romper (|z|>Z_STOP) o en EOD.
Pares de ACCIONES líquidas (no ETFs) para NO colisionar con TSMOM (que opera SPY/QQQ/IWM).
Tamaño de MEDICIÓN pequeño; cierre a nivel de orden. PAPER siempre (claves PK).

  python operativa_pairs.py --entrar [--enviar]     # abre pares con |z|>=Z_IN
  python operativa_pairs.py --revisar [--enviar]    # cierra los que revierten / rompen / EOD
  python operativa_pairs.py --reconciliar           # fills y slippage
"""
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca
from fuentes import alpaca_trade as AT

LOG = "datos/pairs_paper.csv"
# Pares de acciones líquidas y económicamente emparejadas (sin solape con TSMOM/ETFs):
PARES = [("GOOGL", "GOOG"), ("V", "MA"), ("KO", "PEP"), ("XOM", "CVX"),
         ("JPM", "BAC"), ("HD", "LOW"), ("MSFT", "ORCL"), ("PEP", "MDLZ")]
Z_IN, Z_OUT, Z_STOP = 2.0, 0.5, 3.5
LOOKBACK = 120            # barras de 5m para beta + estadística del spread
NOTIONAL = 1000.0         # $ por pata (medición)
CAMPOS = ["ts", "par", "ticker", "etapa", "lado", "qty", "z", "order_id", "fill",
          "slippage_pct", "pl", "estado"]


def _ahora_et():
    return pd.Timestamp.now(tz="America/New_York")


def _min_al_cierre(now=None):
    now = now or _ahora_et()
    cierre = now.normalize() + pd.Timedelta(hours=16)
    return (cierre - now).total_seconds() / 60.0


def _log():
    if os.path.exists(LOG):
        try:
            return pd.read_csv(LOG)
        except Exception:
            pass
    return pd.DataFrame(columns=CAMPOS)


def _guardar(df):
    os.makedirs("datos", exist_ok=True)
    df.to_csv(LOG, index=False)


def _closes(ticker):
    desde = (pd.Timestamp.now() - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    try:
        df = alpaca.historico(ticker, "5Min", desde=desde, max_paginas=6)
    except Exception:
        return None
    return df["close"].astype(float) if len(df) else None


def _estado(a, b):
    """z-score del spread log(a)-beta*log(b) sobre las últimas LOOKBACK barras alineadas.
    beta por OLS causal (toda la ventana es pasado). Devuelve dict o None."""
    ca, cb = _closes(a), _closes(b)
    if ca is None or cb is None:
        return None
    df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, sort=True).dropna()
    if len(df) < LOOKBACK:
        return None
    w = df.tail(LOOKBACK)
    la, lb = np.log(w["a"].to_numpy()), np.log(w["b"].to_numpy())
    beta = float(np.polyfit(lb, la, 1)[0])
    spread = la - beta * lb
    mu, sd = float(spread[:-1].mean()), float(spread[:-1].std())
    if sd <= 0:
        return None
    z = (spread[-1] - mu) / sd
    return {"z": z, "beta": beta, "pa": float(w["a"].iloc[-1]), "pb": float(w["b"].iloc[-1])}


def _abiertos(df):
    """Pares con entrada sin salida (estado enviada/hecha)."""
    if not len(df):
        return set()
    viv = df[df["estado"].isin(["enviada", "hecha"])]
    out = set()
    for par in viv["par"].unique():
        ent = (viv[(viv.par == par) & (viv.etapa == "entrada")]).shape[0]
        sal = (viv[(viv.par == par) & (viv.etapa == "salida")]).shape[0]
        if ent > sal:
            out.add(par)
    return out


def entrar(confirmar=False):
    et = _ahora_et()
    print(f"PAIRS entrar | {et:%Y-%m-%d %H:%M} ET")
    if _min_al_cierre() < 60:
        print("  <60 min al cierre: no abro (no daría tiempo a revertir)."); return
    df = _log()
    abiertos = _abiertos(df)
    nuevas = []
    for a, b in PARES:
        par = f"{a}_{b}"
        if par in abiertos:
            continue
        s = _estado(a, b)
        if s is None:
            continue
        z = s["z"]
        if not (Z_IN <= abs(z) < Z_STOP):
            continue
        qa, qb = int(NOTIONAL / s["pa"]), int(NOTIONAL / s["pb"])
        if qa <= 0 or qb <= 0:
            continue
        # z>0: spread caro -> A cara -> CORTO A / LARGO B.  z<0: al revés.
        lado_a, lado_b = ("sell", "buy") if z > 0 else ("buy", "sell")
        patas = [(a, lado_a, qa), (b, lado_b, qb)]
        for tkr, lado, q in patas:
            fila = {"ts": et.isoformat(), "par": par, "ticker": tkr, "etapa": "entrada",
                    "lado": lado, "qty": q, "z": round(z, 2), "order_id": "", "fill": np.nan,
                    "slippage_pct": np.nan, "pl": np.nan, "estado": "pend"}
            if confirmar:
                r = AT.enviar_orden(tkr, q, lado, confirmar=True, libro="pairs")
                fila["order_id"] = r.get("id", "")
                fila["estado"] = "enviada" if r.get("id") else "error"
            nuevas.append(fila)
        print(f"  {'[dry] ' if not confirmar else ''}{par} z={z:+.2f} -> {lado_a.upper()} {a} x{qa} / {lado_b.upper()} {b} x{qb}")
    if confirmar and nuevas:
        _guardar(pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True))
    elif not nuevas:
        print("  (ningún par con |z|>=2 ahora)")


def revisar(confirmar=False):
    """Cierra los pares abiertos que revierten (|z|<=Z_OUT), rompen (|z|>=Z_STOP) o en EOD."""
    et = _ahora_et()
    df = _log()
    abiertos = _abiertos(df)
    if not abiertos:
        print("PAIRS revisar | sin pares abiertos."); return
    eod = _min_al_cierre() <= 8
    try:
        pos = AT.posiciones()
    except Exception as e:
        print(f"  error posiciones: {str(e)[:50]}"); return
    cerradas = []
    for par in abiertos:
        a, b = par.split("_")
        s = _estado(a, b)
        z = s["z"] if s else 0.0
        motivo = ("EOD" if eod else ("REV" if (s and abs(z) <= Z_OUT) else
                  ("STOP" if (s and abs(z) >= Z_STOP) else None)))
        if motivo is None:
            continue
        # cerrar cada pata: lado opuesto al de la entrada (última entrada del par)
        ent = df[(df.par == par) & (df.etapa == "entrada") & (df.estado.isin(["enviada", "hecha"]))]
        for _, e in ent.groupby("ticker").last().reset_index().iterrows():
            tkr = e["ticker"]; qty = int(e["qty"])
            lado_cierre = "buy" if e["lado"] == "sell" else "sell"
            fila = {"ts": et.isoformat(), "par": par, "ticker": tkr, "etapa": "salida",
                    "lado": lado_cierre, "qty": qty, "z": round(z, 2), "order_id": "",
                    "fill": np.nan, "slippage_pct": np.nan,
                    "pl": (pos.get(tkr, {}) or {}).get("pl"), "estado": "pend"}
            if confirmar:
                r = AT.enviar_orden(tkr, qty, lado_cierre, confirmar=True, libro="pairs")
                fila["order_id"] = r.get("id", "")
                fila["estado"] = "enviada" if r.get("id") else "error"
            cerradas.append(fila)
        print(f"  {'[dry] ' if not confirmar else ''}cerrar {par} z={z:+.2f} ({motivo})")
    if confirmar and cerradas:
        _guardar(pd.concat([df, pd.DataFrame(cerradas)], ignore_index=True))


def reconciliar():
    df = _log()
    if not len(df):
        return
    for i, r in df.iterrows():
        if r.get("order_id") and (pd.isna(r.get("fill")) or r.get("fill") == ""):
            try:
                fav = AT.orden(str(r["order_id"])).get("filled_avg_price")
                if fav:
                    df.at[i, "fill"] = float(fav)
                    df.at[i, "estado"] = "hecha"
            except Exception:
                continue
    _guardar(df)
    hechas = df[df["fill"].notna()]
    cerrados = df[(df.etapa == "salida") & df.pl.notna()]
    print(f"PAIRS reconciliado | fills: {len(hechas)} | patas cerradas: {len(cerrados)}"
          + (f" | PnL patas cerradas: {cerrados['pl'].astype(float).sum():+.2f}$" if len(cerrados) else ""))


if __name__ == "__main__":
    if "--reconciliar" in sys.argv:
        reconciliar()
    elif "--revisar" in sys.argv:
        revisar(confirmar="--enviar" in sys.argv)
    elif "--entrar" in sys.argv:
        entrar(confirmar="--enviar" in sys.argv)
    else:
        print(__doc__)
