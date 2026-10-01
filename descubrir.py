"""Descubrimiento: analiza tickers NUEVOS (no vistos en el universo del estudio)
para encontrar nuevas oportunidades intradia, con sus noticias. Se ejecuta entre
ciclos de gestion para ampliar el radar sin tocar la cartera.

Uso:  python descubrir.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca, noticias_vivo
from patrones import indicadores as ind
from patrones.intradia import rasgos_intradia, rvol
from operativa_intradia import LIQUIDOS
from gestionar import UNIVERSO

WATCHLIST = "datos/watchlist.json"


def _cargar_watchlist(dias=3):
    """Lee la watchlist persistente y purga lo mas viejo de 'dias'."""
    try:
        wl = json.load(open(WATCHLIST))
    except Exception:
        return {}
    ahora = pd.Timestamp.now()
    return {t: ts for t, ts in wl.items()
            if (ahora - pd.Timestamp(ts)) <= pd.Timedelta(days=dias)}


def watchlist_vigente(dias=3):
    """Tickers descubiertos recientemente (para que el radar crezca y rote, en vez
    de mirar siempre la misma lista fija)."""
    return sorted(_cargar_watchlist(dias).keys())


def guardar_en_watchlist(d, rvol_min=1.5, sent_min=-0.1, dias=3):
    """Apunta en la watchlist los descubrimientos PROMETEDORES (RVOL fuerte y sin
    noticia claramente negativa). Devuelve los nombres nuevos anotados."""
    wl = _cargar_watchlist(dias)
    if d is None or len(d) == 0:
        json.dump(wl, open(WATCHLIST, "w")); return []
    os.makedirs("datos", exist_ok=True)
    ahora = pd.Timestamp.now().isoformat()
    nuevos = []
    for _, s in d.iterrows():
        if float(s["rvol"]) >= rvol_min and float(s.get("sent_noticias", 0.0)) >= sent_min:
            if str(s["ticker"]) not in wl:
                nuevos.append(str(s["ticker"]))
            wl[str(s["ticker"])] = ahora
    json.dump(wl, open(WATCHLIST, "w"))
    return nuevos

# liquidos que NO estan en el universo del estudio (acciones nuevas para el radar)
NUEVOS = ["SHOP", "PLTR", "SNOW", "CRWD", "DDOG", "NET", "ABNB", "UBER", "COIN", "RBLX",
          "ROKU", "PANW", "ANET", "SMCI", "ARM", "DELL", "TXN", "LRCX", "KLAC", "ADI",
          "BKNG", "MAR", "CMG", "SBUX", "LOW", "TGT", "LULU", "DKNG", "F", "GM",
          "GILD", "AMGN", "VRTX", "REGN", "ISRG", "MRNA", "LMT", "RTX", "MU", "QCOM"]


def candidatos_nuevos(intervalo="15Min", n=8, con_noticias=True):
    """Escanea los tickers nuevos y devuelve los candidatos intradia mas fuertes,
    con su sentimiento de noticias."""
    ya = set(UNIVERSO) | set(LIQUIDOS)
    universo = [t for t in NUEVOS if t not in ya] or NUEVOS
    desde = (pd.Timestamp.now() - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    filas = []
    for t in universo:
        try:
            df = alpaca.historico(t, intervalo, desde=desde, max_paginas=6)
            if len(df) < 20:
                continue
            r = rasgos_intradia(df)
            precio = float(df["close"].iloc[-1])
            atr = float(ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1])
            dv = float(r["intra_dist_vwap"].iloc[-1])
            rv = float(rvol(df).iloc[-1])
            rs = float(r["intra_ret_sesion"].iloc[-1])
        except Exception:
            continue
        if (dv > 0) and (rs > 0) and (rv >= 1.3):
            filas.append({"ticker": t, "precio": precio, "dist_vwap": dv, "rvol": rv,
                          "ret_sesion": rs, "atr_pct": atr / precio if precio else 0.0,
                          "score": rv * rs})
    d = pd.DataFrame(filas)
    if len(d) == 0:
        return d
    d = d.sort_values("score", ascending=False).head(n).reset_index(drop=True)
    if con_noticias:
        notis = []
        for t in d["ticker"]:
            nv = noticias_vivo.noticias(t, n=4)
            notis.append({"sent_noticias": nv["sentimiento"], "n_noticias": nv["n"],
                          "tono": nv["etiqueta"]})
        d = pd.concat([d, pd.DataFrame(notis)], axis=1)
    return d


def main():
    d = candidatos_nuevos()
    if len(d) == 0:
        print("Sin candidatos nuevos ahora mismo.")
        return
    print(f"NUEVOS TICKERS CON OPORTUNIDAD INTRADIA ({len(d)}):")
    for _, s in d.iterrows():
        extra = f" | noticias {s.get('tono','')} ({s.get('sent_noticias',0):+.2f})" if "tono" in d else ""
        print(f"  {s['ticker']:<6} VWAP {s['dist_vwap']*100:+.2f}% RVOL {s['rvol']:.2f} "
              f"sesion {s['ret_sesion']*100:+.2f}%{extra}")
    # memoria del radar: los prometedores entran en la watchlist y pasan a ser
    # candidatos operables en los proximos ciclos (el universo crece y rota).
    nuevos = guardar_en_watchlist(d)
    vig = watchlist_vigente()
    if nuevos:
        print(f"  -> anadidos a la watchlist operable: {nuevos}")
    if vig:
        print(f"  watchlist vigente ({len(vig)}): {vig}")


if __name__ == "__main__":
    main()
