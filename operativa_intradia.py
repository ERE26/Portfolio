"""Operativa INTRADIA para la cuenta paper: estrategia 'intraday momentum en
stocks in play' (largo si el valor esta sobre su VWAP, con volumen anormal (RVOL)
y sesion positiva), ejecutada con ordenes BRACKET (TP/SL en ATR) que Alpaca cierra
solas. Usa una reserva del capital, separada del book de momentum (swing).

Uso:
  python operativa_intradia.py            # dry-run
  python operativa_intradia.py --enviar   # envia las ordenes bracket al paper
"""
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca, alpaca_trade as AT
from patrones import indicadores as ind
from patrones.intradia import rasgos_intradia, rvol

# universo intradia liquido (las mismas grandes y ETFs que se usaron en el estudio)
LIQUIDOS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO",
            "JPM", "BAC", "GS", "XOM", "CVX", "UNH", "LLY", "COST", "SPY", "QQQ", "XLE"]

# configuracion por temporalidad (de lo analizado en el estudio):
#  - 1h: RVOL filtro, continuacion; R:R 1.5:1 en ATR.
#  - 15m: mas ruidoso -> RVOL mas exigente; sin forzar cierre (el bracket decide).
CONFIG = {
    "1Hour": {"k_pt": 1.5, "k_sl": 1.0, "rvol_min": 1.1, "dias": 8},
    "15Min": {"k_pt": 1.3, "k_sl": 0.9, "rvol_min": 1.3, "dias": 4},
}


def señales(tickers=LIQUIDOS, intervalo="1Hour", verbose=True):
    """Señales intradia actuales: para cada ticker, VWAP/RVOL/sesion y si es
    candidato largo. Devuelve DataFrame ordenado por fuerza."""
    cfg = CONFIG[intervalo]
    desde = (pd.Timestamp.now() - pd.Timedelta(days=cfg["dias"])).strftime("%Y-%m-%d")
    filas = []
    for t in tickers:
        try:
            df = alpaca.historico(t, intervalo, desde=desde, max_paginas=6)
        except Exception as e:
            if verbose:
                print(f"  {t}: sin datos ({str(e)[:40]})")
            continue
        if len(df) < 20:
            continue
        r = rasgos_intradia(df)
        rv = float(rvol(df).iloc[-1])
        atr = float(ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1])
        precio = float(df["close"].iloc[-1])
        atr_pct = atr / precio if precio else 0.0
        dist_vwap = float(r["intra_dist_vwap"].iloc[-1])
        ret_ses = float(r["intra_ret_sesion"].iloc[-1])
        candidato = (dist_vwap > 0) and (ret_ses > 0) and (rv >= cfg["rvol_min"]) and (atr_pct > 0.001)
        filas.append({"ticker": t, "precio": precio, "atr_pct": atr_pct,
                      "dist_vwap": dist_vwap, "rvol": rv, "ret_sesion": ret_ses,
                      "score": rv * max(ret_ses, 0), "candidato": candidato})
    d = pd.DataFrame(filas)
    return d.sort_values("score", ascending=False).reset_index(drop=True) if len(d) else d


def plan_intradia(reserva, intervalo="1Hour", n_max=5, tickers=LIQUIDOS, verbose=True):
    """Objetivo intradia: hasta n_max candidatos largos, repartiendo 'reserva' $,
    con precios de entrada/TP/SL. Devuelve (DataFrame de señales, lista de ordenes)."""
    cfg = CONFIG[intervalo]
    d = señales(tickers, intervalo=intervalo, verbose=verbose)
    if len(d) == 0:
        return d, []
    cand = d[d["candidato"]].head(n_max)
    if len(cand) == 0:
        return d, []
    por_pos = reserva / len(cand)
    ordenes = []
    for _, s in cand.iterrows():
        qty = int(por_pos / s["precio"])
        if qty <= 0:
            continue
        tp = s["precio"] * (1 + cfg["k_pt"] * s["atr_pct"])
        sl = s["precio"] * (1 - cfg["k_sl"] * s["atr_pct"])
        ordenes.append({"ticker": s["ticker"], "qty": qty, "precio": s["precio"],
                        "tp": round(tp, 2), "sl": round(sl, 2),
                        "rvol": s["rvol"], "ret_sesion": s["ret_sesion"]})
    return d, ordenes


def ejecutar(reserva=25000.0, intervalo="1Hour", enviar=False):
    cta = AT.cuenta()
    print(f"Cuenta PAPER: equity ${float(cta['equity']):,.0f} | reserva {intervalo} ${reserva:,.0f}")
    import requests
    clock = requests.get("https://paper-api.alpaca.markets/v2/clock", headers=AT._headers(),
                         timeout=20).json()
    print(f"Mercado {'ABIERTO' if clock.get('is_open') else 'CERRADO (las ordenes quedan en cola)'}")
    d, ordenes = plan_intradia(reserva, intervalo=intervalo)
    if len(d):
        print(f"\nSEÑALES {intervalo} (top por fuerza):")
        for _, s in d.head(8).iterrows():
            marca = "LARGO" if s["candidato"] else "  -  "
            print(f"  {marca} {s['ticker']:<6} VWAP {s['dist_vwap']*100:+5.2f}% | RVOL {s['rvol']:.2f} "
                  f"| sesion {s['ret_sesion']*100:+5.2f}% | ATR {s['atr_pct']*100:.2f}%")
    print(f"\nPLAN INTRADIA ({len(ordenes)} bracket):")
    for o in ordenes:
        print(f"  LARGO {o['qty']:>4} {o['ticker']:<6} entrada ~${o['precio']:.2f} "
              f"-> TP ${o['tp']:.2f} / SL ${o['sl']:.2f}  (RVOL {o['rvol']:.2f})")
    if not ordenes:
        print("  (ningun candidato intradia ahora mismo)")
        return
    if not enviar:
        print("\n[DRY-RUN] No se ha enviado nada. Ejecuta con --enviar para colocar los brackets.")
        return
    print("\nEnviando ordenes bracket a la cuenta PAPER...")
    ok = 0
    for o in ordenes:
        try:
            r = AT.enviar_bracket(o["ticker"], o["qty"], o["tp"], o["sl"], confirmar=True)
            if r.get("id"):
                ok += 1
        except Exception as e:
            print(f"  error en {o['ticker']}: {str(e)[:90]}")
    print(f"Enviadas {ok}/{len(ordenes)} ordenes bracket. Revisa el panel de Alpaca.")


if __name__ == "__main__":
    iv = "15Min" if "--15m" in sys.argv else "1Hour"
    rsv = 25000.0
    for a in sys.argv:
        if a.startswith("--reserva="):
            rsv = float(a.split("=")[1])
    ejecutar(reserva=rsv, intervalo=iv, enviar="--enviar" in sys.argv)
