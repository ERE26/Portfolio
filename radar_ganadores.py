"""RADAR DE GANADORES DE LA ÚLTIMA HORA (cada ~15 min), sobre TODO EL MERCADO US.

Fuente primaria: TradingView (escáner público, ~2.5-3k valores en una llamada) rankeados
por el cambio intradía de la última hora (sin el hueco de apertura), filtrados por liquidez
(precio, $-volumen, RVOL). Si TradingView falla, cae a un pool de Yahoo (most_actives +
gainers) con el 1h calculado por barras de Alpaca.

Para cada top-gainer dice POR QUÉ no operamos ahí: dentro de nuestro universo -> el gate
que falla / en cartera / cooldown / 'PASARÍA pero sin hueco/REDUCE'; fuera -> PUNTO CIEGO.

Auto-limitado (datos/radar_last.json, >= dias_min min) para que el bucle lo llame cada ciclo.

  python radar_ganadores.py [--force]
"""
import json
import os
import sys

import pandas as pd

import gestionar as G

ESTADO = "datos/radar_last.json"


def _ultimo():
    try:
        return pd.Timestamp(json.load(open(ESTADO))["ts"])
    except Exception:
        return None


def _marcar():
    os.makedirs("datos", exist_ok=True)
    json.dump({"ts": pd.Timestamp.now().isoformat()}, open(ESTADO, "w"))


def _motivo_no_compra(t, universo, pool, held, cooldown, mkt, rvol_fuerte):
    """Por qué NO operamos un top-gainer. Distingue el PUNTO CIEGO REAL (fuera del pool
    líquido que de verdad escaneamos: no pasa el filtro de liquidez/RVOL) de 'LA VEMOS y
    la descartamos' (sí entra en el escáner, pero falla un gate concreto). `pool` = pool
    intradía en juego ahora (TradingView en_juego); `universo` = universo dinámico."""
    if t in held:
        return "en cartera", "ya la tenemos"
    if t in cooldown:
        return "cooldown", "cerrada hace <60m"
    # ¿la VERÍA siquiera nuestro escáner? El pool intradía es TODO el mercado en juego;
    # si no está ni en él ni en el universo, es un punto ciego REAL (filtro de liquidez).
    if (t not in pool) and (t not in universo):
        return "FUERA", "punto ciego REAL: fuera del pool líquido en juego (no pasa el filtro de liquidez/RVOL del escáner)"
    # la vemos -> ¿qué gate falla?
    try:
        s = G.señal(t, "15Min")
    except Exception:
        return "sin datos", "en el pool pero sin velas 15m frescas"
    rv, dv, rs = s["rvol"], s["dist_vwap"], s["ret_sesion"]
    atr = s.get("atr_pct", 0.0)
    fr = rs - mkt
    ext = dv / atr if atr > 0 else 0.0
    if rv < rvol_fuerte:
        return "descartada", f"RVOL {rv:.2f} < {rvol_fuerte}"
    if dv <= 0:
        return "descartada", f"bajo VWAP ({dv*100:+.2f}%)"
    if not (0 < ext < G.EXT_MAX):
        return "descartada", f"sobre-extendida {ext:.1f} ATR"
    if ext >= G.EXT_PARAB and rs >= G.RET_PARAB:
        return "descartada", f"parábola: {ext:.1f} ATR sobre VWAP + sesión {rs*100:+.1f}% (agotada)"
    if fr <= G.MARGEN_FR:
        return "descartada", f"fuerza rel {fr*100:+.2f}% <= +{G.MARGEN_FR*100:.1f}%"
    if rs < G.RET_MIN:
        return "descartada", f"sesión {rs*100:+.2f}% < +{G.RET_MIN*100:.1f}%"
    return "PASARÍA", "pasa confluencia; no entró por hueco lleno / REDUCE"


def _ganadores(top):
    """(filas, fuente, total_mercado). Intenta TradingView (todo el mercado); si falla, Yahoo."""
    try:
        from fuentes import tradingview as TV
        g, total = TV.ganadores_intradia(tf="60", n=top)
        if g:
            filas = [{"t": x["symbol"], "h1": (x["chg_tf"] or 0) / 100.0,
                      "m15": (x["chg_15"] or 0) / 100.0, "dia": (x["chg_dia"] or 0) / 100.0,
                      "rvol": x.get("rvol"), "dvol": x["dvol"], "sector": x["sector"]} for x in g]
            return filas, f"TradingView · {total} valores del mercado", total
    except Exception as e:
        print(f"  (TradingView no disponible: {str(e)[:60]}; uso pool de Yahoo)")
    # ---- fallback: pool Yahoo + 1h por barras de Alpaca ----
    from fuentes.http import Cliente
    from fuentes import alpaca
    cli = Cliente()
    syms, nombre = [], {}
    for scr in ["most_actives", "day_gainers", "small_cap_gainers"]:
        try:
            d = cli.get_json(f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds={scr}&count=50",
                             headers={"User-Agent": "Mozilla/5.0"}, cache=False)
            for q in d["finance"]["result"][0]["quotes"]:
                s = q.get("symbol")
                if s and s not in nombre:
                    nombre[s] = 1; syms.append(s)
        except Exception:
            continue
    desde = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    filas = []
    for t in syms[:120]:
        try:
            cl = alpaca.historico(t, "15Min", desde=desde, max_paginas=2)["close"].astype(float)
            if len(cl) >= 5:
                filas.append({"t": t, "h1": float(cl.iloc[-1] / cl.iloc[-5] - 1), "m15": float(cl.iloc[-1] / cl.iloc[-2] - 1),
                              "dia": float("nan"), "dvol": float("nan"), "sector": ""})
        except Exception:
            continue
    filas.sort(key=lambda x: -x["h1"])
    return filas[:top], f"Yahoo pool · {len(syms)} activos", len(syms)


def ejecutar(force=False, dias_min=14, top=12):
    ult = _ultimo()
    if not force and ult is not None and (pd.Timestamp.now() - ult).total_seconds() < dias_min * 60:
        print(f"RADAR 1h: último pase hace {(pd.Timestamp.now()-ult).total_seconds()/60:.0f} min (<{dias_min}); no toca.")
        return
    filas, fuente, _ = _ganadores(top)
    if not filas:
        print("RADAR 1h: sin datos.")
        _marcar()
        return
    from fuentes import alpaca_trade as AT
    from registro import tickers_en_cooldown, cargar_umbral
    universo = set(G.universo_dinamico())
    try:                                              # pool intradía REAL que escaneamos
        from fuentes import tradingview as TV
        pool = set(TV.en_juego())
    except Exception:
        pool = set()
    held = set(AT.posiciones())
    cooldown = tickers_en_cooldown(minutos=60)
    mkt = G._ret_sesion_mercado("15Min")
    rvol_fuerte = cargar_umbral(1.5)

    print(f"RADAR · MAYORES GANADORES DE LA ÚLTIMA HORA (sin apertura) | fuente: {fuente} | "
          f"mercado (SPY) {mkt*100:+.2f}%")
    print(f"  {'ticker':<7}{'1h':>8}{'15m':>8}{'RVOL':>6}{'$vol(M)':>9}  por qué NO nosotros")
    fuera = 0
    pasarian = []
    for g in filas:
        est, mot = _motivo_no_compra(g["t"], universo, pool, held, cooldown, mkt, rvol_fuerte)
        if est == "FUERA":
            fuera += 1
        if est == "PASARÍA":
            pasarian.append(g["t"])
        fl = "↑" if (g["m15"] or 0) > 0.001 else "↓" if (g["m15"] or 0) < -0.001 else "→"
        rv = f"{g['rvol']:.1f}" if g.get("rvol") else "—"
        dv = f"{g['dvol']/1e6:.1f}" if g.get("dvol") and g["dvol"] == g["dvol"] else "—"
        sec = f"  · {g['sector']}" if g.get("sector") else ""
        print(f"  {g['t']:<7}{g['h1']*100:>+7.2f}%{g['m15']*100:>+7.2f}%{fl}{rv:>6}{dv:>9}  [{est}] {mot}{sec}")
    print(f"\n  Resumen: de los {len(filas)} que más subieron la última hora, {fuera} son PUNTO "
          f"CIEGO REAL (fuera del pool líquido que escaneamos); {len(filas)-fuera} SÍ las vimos "
          f"(descartadas por un gate / en cartera / cooldown).")
    if pasarian:
        print(f"  OJO: {pasarian} pasaban la confluencia y no entraron solo por hueco/REDUCE "
              f"-> ahí sí perdimos una entrada válida.")
    _marcar()


if __name__ == "__main__":
    ejecutar(force="--force" in sys.argv)
