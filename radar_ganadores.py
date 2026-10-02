"""RADAR DE GANADORES DEL MERCADO (cada ~15 min). Mira los MAYORES ganadores de TODO el
mercado US (no solo nuestro universo), qué han hecho en los últimos ~15 min, y POR QUÉ
nosotros no los compramos o no estaban en el radar. Sirve para detectar PUNTOS CIEGOS
(valores fuera de nuestro universo) y revisar los gates con casos reales.

Auto-limitado: no repite si el último pase fue hace < dias_min minutos (estado en
datos/radar_last.json), para que el bucle pueda llamarlo cada ciclo sin coste.

  python radar_ganadores.py          # corre si han pasado >= 14 min
  python radar_ganadores.py --force  # fuerza
"""
import json
import os
import sys

import pandas as pd

import gestionar as G
from operativa_intradia import CONFIG
from fuentes import alpaca_trade as AT
from fuentes.http import Cliente

ESTADO = "datos/radar_last.json"
URL = ("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
       "?scrIds=day_gainers&count=25")


def _ultimo():
    try:
        return pd.Timestamp(json.load(open(ESTADO))["ts"])
    except Exception:
        return None


def _marcar():
    os.makedirs("datos", exist_ok=True)
    json.dump({"ts": pd.Timestamp.now().isoformat()}, open(ESTADO, "w"))


def _ganadores_mercado(cli, n=15):
    """Mayores ganadores del mercado US hoy (screener gratuito de Yahoo)."""
    d = cli.get_json(URL, headers={"User-Agent": "Mozilla/5.0"}, cache=False)
    q = d["finance"]["result"][0]["quotes"]
    out = []
    for x in q:
        out.append({"t": x.get("symbol"), "dia": float(x.get("regularMarketChangePercent", 0.0)),
                    "precio": x.get("regularMarketPrice"), "vol": float(x.get("regularMarketVolume", 0) or 0),
                    "nombre": (x.get("shortName") or "")[:26]})
    return out[:n]


def _delta_15m(ticker):
    """Movimiento en los últimos ~15 min (última vela 15m vs la previa). None si no hay."""
    try:
        from fuentes import alpaca
        dsd = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = alpaca.historico(ticker, "15Min", desde=dsd, max_paginas=2)
        c = df["close"].astype(float)
        if len(c) >= 2:
            return float(c.iloc[-1] / c.iloc[-2] - 1.0)
    except Exception:
        pass
    return None


def _motivo_no_compra(t, universo, held, cooldown, mkt, rvol_fuerte):
    """Por qué NO lo compramos (si está en el universo); o que es punto ciego (fuera)."""
    if t not in universo:
        return "FUERA DE UNIVERSO", "no está en nuestros ~114 (punto ciego)"
    if t in held:
        return "en cartera", "ya la tenemos"
    if t in cooldown:
        return "cooldown", "cerrada hace <60m"
    try:
        s = G.señal(t, "15Min")
    except Exception:
        return "sin datos 15m", "no hay velas suficientes"
    rv, dv, rs = s["rvol"], s["dist_vwap"], s["ret_sesion"]
    atr = s.get("atr_pct", 0.0)
    fr = rs - mkt
    ext = dv / atr if atr > 0 else 0.0
    if rv < rvol_fuerte:
        return "descartada", f"RVOL {rv:.2f} < {rvol_fuerte} (volumen no anormal p/ nosotros)"
    if dv <= 0:
        return "descartada", f"bajo VWAP ({dv*100:+.2f}%)"
    if not (0 < ext < G.EXT_MAX):
        return "descartada", f"sobre-extendida {ext:.1f} ATR"
    if fr <= G.MARGEN_FR:
        return "descartada", f"fuerza rel {fr*100:+.2f}% <= +{G.MARGEN_FR*100:.1f}%"
    if rs < G.RET_MIN:
        return "descartada", f"sesión {rs*100:+.2f}% < +{G.RET_MIN*100:.1f}%"
    return "PASARÍA", "pasa la confluencia (no entró por hueco lleno / REDUCE o cooldown de slot)"


def ejecutar(force=False, dias_min=14):
    ult = _ultimo()
    if not force and ult is not None and (pd.Timestamp.now() - ult).total_seconds() < dias_min * 60:
        mins = (pd.Timestamp.now() - ult).total_seconds() / 60
        print(f"RADAR: último pase hace {mins:.0f} min (<{dias_min}); no toca.")
        return
    cli = Cliente()
    try:
        gan = _ganadores_mercado(cli)
    except Exception as e:
        print(f"RADAR: no se pudo leer el screener de ganadores ({str(e)[:60]}).")
        return
    universo = set(G.universo_dinamico())
    held = set(AT.posiciones())
    from registro import tickers_en_cooldown, cargar_umbral
    cooldown = tickers_en_cooldown(minutos=60)
    mkt = G._ret_sesion_mercado("15Min")
    rvol_fuerte = cargar_umbral(1.5)

    print(f"RADAR DE GANADORES DEL MERCADO | top {len(gan)} del día | mercado (SPY) {mkt*100:+.2f}%")
    print(f"  {'ticker':<7}{'día':>8}{'15min':>8}{'vol':>8}  por qué NO nosotros")
    fuera = 0
    for g in gan:
        t = g["t"]
        d15 = _delta_15m(t)
        est, motivo = _motivo_no_compra(t, universo, held, cooldown, mkt, rvol_fuerte)
        if est == "FUERA DE UNIVERSO":
            fuera += 1
        d15s = f"{d15*100:+.1f}%" if d15 is not None else "—"
        print(f"  {t:<7}{g['dia']:>+7.1f}%{d15s:>8}{g['vol']/1e6:>6.1f}M  [{est}] {motivo}")
    print(f"\n  Resumen: {fuera}/{len(gan)} de los mayores ganadores están FUERA de nuestro "
          f"universo (punto ciego). El resto, por qué no, arriba.")
    if fuera >= len(gan) * 0.6:
        print("  -> La mayoría de los grandes movimientos del día son valores que NO miramos "
              "(small/mid-caps). Decisión a valorar: ampliar universo o aceptar que operamos "
              "solo líquidos grandes (menos sorpresa, menos oportunidad).")
    _marcar()


if __name__ == "__main__":
    ejecutar(force="--force" in sys.argv)
