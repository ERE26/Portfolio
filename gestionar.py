"""Gestion ACTIVA del intradia + escaneo de mas oportunidades (intradia y diario).

Pensado para ejecutarse cada ~25 min mientras el mercado esta abierto:
  1. Revisa cada posicion intradia (con bracket): si la tendencia se gira (pierde
     el VWAP o la sesion se vuelve negativa) la cierra para cortar perdida; si va
     bien la deja correr (el bracket protege TP/SL).
  2. Escanea un universo AMPLIADO buscando nuevas oportunidades intradia (1h/15m)
     y diarias (momentum), que todavia no esten en cartera.

Uso:
  python gestionar.py            # dry-run: revisa y reporta, no cierra ni abre
  python gestionar.py --gestionar  # cierra las intradia que se giran (no abre solo)
"""
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca, alpaca_trade as AT
from patrones import indicadores as ind
from patrones.intradia import rasgos_intradia, rvol
from operativa_intradia import CONFIG, LIQUIDOS

# universo ampliado para buscar mas oportunidades (grandes + mas sectores)
UNIVERSO = sorted(set(LIQUIDOS + [
    "ORCL", "CRM", "ADBE", "WFC", "C", "AXP", "V", "MA", "COP", "PFE", "MRK", "ABBV",
    "HD", "MCD", "NKE", "KO", "PEP", "PG", "WMT", "CAT", "BA", "DIA", "IWM", "XLK", "XLF",
    "DE", "GE", "MU", "QCOM", "INTC", "T", "VZ", "DIS", "NFLX", "PYPL"]))


def señal(ticker, intervalo="15Min"):
    cfg = CONFIG[intervalo]
    desde = (pd.Timestamp.now() - pd.Timedelta(days=cfg["dias"])).strftime("%Y-%m-%d")
    df = alpaca.historico(ticker, intervalo, desde=desde, max_paginas=6)
    if len(df) < 20:
        raise RuntimeError("pocos datos")
    r = rasgos_intradia(df)
    precio = float(df["close"].iloc[-1])
    atr = float(ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1])
    return {"precio": precio, "dist_vwap": float(r["intra_dist_vwap"].iloc[-1]),
            "rvol": float(rvol(df).iloc[-1]), "ret_sesion": float(r["intra_ret_sesion"].iloc[-1]),
            "atr_pct": atr / precio if precio else 0.0}


def revisar_intradia():
    """Decide mantener o cerrar cada posicion intradia (la que tiene bracket)."""
    ordenes = AT.ordenes_abiertas()
    simbolos = sorted(set(o["symbol"] for o in ordenes))
    pos = AT.posiciones()
    out = []
    for t in simbolos:
        if t not in pos:
            continue
        try:
            s = señal(t, "15Min")
        except Exception:
            continue
        # cortar si se gira: por debajo del VWAP o sesion ya negativa
        cerrar = (s["dist_vwap"] < -0.0015) or (s["ret_sesion"] < -0.001)
        out.append({"ticker": t, "pl": pos[t]["pl"], "cerrar": cerrar,
                    "dist_vwap": s["dist_vwap"], "ret_sesion": s["ret_sesion"], "rvol": s["rvol"]})
    return out


def escanear(intervalo="15Min", excluir=(), n=6):
    """Nuevas oportunidades intradia en el universo ampliado (no en cartera)."""
    cfg = CONFIG[intervalo]
    filas = []
    for t in UNIVERSO:
        if t in excluir:
            continue
        try:
            s = señal(t, intervalo)
        except Exception:
            continue
        cand = (s["dist_vwap"] > 0) and (s["ret_sesion"] > 0) and (s["rvol"] >= cfg["rvol_min"])
        if cand:
            filas.append({"ticker": t, "score": s["rvol"] * s["ret_sesion"], **s})
    d = pd.DataFrame(filas)
    return d.sort_values("score", ascending=False).head(n).reset_index(drop=True) if len(d) else d


def escaneo_diario(n=8):
    """Top momentum diario del universo (oportunidades swing)."""
    from fuentes.descarga import cargar_mercado
    from patrones import transversal as TV
    pre, _ = cargar_mercado("datos/mercado", intervalo="1d")
    panel = TV.panel_precios(pre)
    mom = TV.score_momentum(panel).dropna().sort_values(ascending=False)
    return mom.head(n)


def main(gestionar=False, rotar=False, reserva_por_pos=20000.0):
    pos = AT.posiciones()
    rev = revisar_intradia()
    print(f"Posiciones intradia (bracket): {len(rev)}")
    a_cerrar = []
    for d in rev:
        estado = "CERRAR (se gira)" if d["cerrar"] else "mantener"
        print(f"  {d['ticker']:<6} PL ${d['pl']:+.0f} | VWAP {d['dist_vwap']*100:+.2f}% "
              f"sesion {d['ret_sesion']*100:+.2f}% RVOL {d['rvol']:.2f} -> {estado}")
        if d["cerrar"]:
            a_cerrar.append(d["ticker"])
    if a_cerrar and gestionar:
        print(f"Cerrando {a_cerrar} para cortar perdidas...")
        print(" ", AT.cerrar_posiciones(a_cerrar, confirmar=True))
    elif a_cerrar:
        print(f"[dry-run] cerraria: {a_cerrar}")

    print("\nNUEVAS OPORTUNIDADES INTRADIA (15m, universo ampliado):")
    opo = escanear("15Min", excluir=set(pos))
    if len(opo):
        for _, s in opo.iterrows():
            print(f"  {s['ticker']:<6} VWAP {s['dist_vwap']*100:+.2f}% RVOL {s['rvol']:.2f} "
                  f"sesion {s['ret_sesion']*100:+.2f}%")
    else:
        print("  (ninguna ahora)")

    # rotacion: por cada posicion cerrada, abrir la mejor oportunidad nueva (bracket)
    if gestionar and rotar and a_cerrar and len(opo):
        cfg = CONFIG["15Min"]
        nuevos = opo.head(len(a_cerrar))
        print("\nRotando capital a nuevas oportunidades:")
        for _, s in nuevos.iterrows():
            qty = int(reserva_por_pos / s["precio"])
            if qty <= 0:
                continue
            tp = round(s["precio"] * (1 + cfg["k_pt"] * s["atr_pct"]), 2)
            sl = round(s["precio"] * (1 - cfg["k_sl"] * s["atr_pct"]), 2)
            try:
                r = AT.enviar_bracket(s["ticker"], qty, tp, sl, confirmar=True)
                print(f"  LARGO {qty} {s['ticker']} -> TP ${tp} / SL ${sl}  "
                      f"{'OK '+r.get('id','')[:8] if r.get('id') else r}")
            except Exception as e:
                print(f"  {s['ticker']}: error {str(e)[:70]}")

    print("\nTOP MOMENTUM DIARIO (oportunidades swing):")
    for t, m in escaneo_diario().items():
        print(f"  {t:<6} momento {m*100:+.1f}%" + ("  [ya en cartera]" if t in pos else ""))


if __name__ == "__main__":
    main(gestionar="--gestionar" in sys.argv, rotar="--rotar" in sys.argv)
