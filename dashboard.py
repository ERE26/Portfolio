"""Terminal: reune TODO el estado del sistema (cuenta, riesgo, posiciones, curva de
equity, monitor de mercado, noticias en vivo, tear-sheet) en un JSON y lo inyecta en
la plantilla HTML -> web/dashboard.html (un terminal estilo mesa de trading).

  python dashboard.py            # genera web/dashboard.html + datos/dashboard.json
El bucle puede regenerarlo cada ciclo -> cuasi tiempo real. No opera; solo lee.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca_trade as AT
from patrones import riesgo as R
from patrones import analitica as A

# monitor de mercado: indices/activos clave (via Yahoo)
MERCADO = [
    ("^GSPC", "S&P 500"), ("^NDX", "Nasdaq 100"), ("^VIX", "VIX"),
    ("^STOXX50E", "Euro Stoxx"), ("^IBEX", "IBEX 35"), ("^GDAXI", "DAX"),
    ("EURUSD=X", "EUR/USD"), ("GC=F", "Oro"), ("CL=F", "Petroleo"), ("BTC-USD", "Bitcoin"),
]


def _madrid():
    try:
        return pd.Timestamp.now(tz="Europe/Madrid").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def _clock():
    try:
        import requests
        r = requests.get("https://paper-api.alpaca.markets/v2/clock", headers=AT._headers(), timeout=20)
        d = r.json()
        return {"abierto": bool(d.get("is_open")), "next_open": d.get("next_open", ""),
                "next_close": d.get("next_close", "")}
    except Exception:
        return {"abierto": None, "next_open": "", "next_close": ""}


def _ordenes():
    try:
        import requests
        r = requests.get(AT._url("orders"), params={"status": "all", "limit": 500},
                         headers=AT._headers(), timeout=30)
        return r.json()
    except Exception:
        return []


def _mercado(cli):
    from fuentes import yahoo
    filas = []
    for t, nombre in MERCADO:
        try:
            df, _ = yahoo.historico(t, intervalo="1d", rango="5d", cliente=cli)
            c = df["close"].astype(float)
            chg = float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else 0.0
            filas.append({"ticker": t, "nombre": nombre, "precio": float(c.iloc[-1]), "chg": chg})
        except Exception:
            continue
    return filas


def _noticias(tickers, cli, max_total=14):
    from fuentes import noticias_vivo
    vistos, out = set(), []
    for t in tickers:
        try:
            r = noticias_vivo.noticias(t, n=4)
        except Exception:
            continue
        for it in r.get("titulares", []):
            tit = it.get("titular", "")
            if tit and tit not in vistos:
                vistos.add(tit)
                out.append({"ticker": t, "titular": tit, "fuente": it.get("fuente", ""),
                            "fecha": it.get("fecha", ""), "punt": it.get("punt", 0.0),
                            "url": it.get("url", "")})
    out.sort(key=lambda x: x.get("fecha", ""), reverse=True)
    return out[:max_total]


def reunir():
    from fuentes.http import Cliente
    cli = Cliente()
    cta = AT.cuenta()
    pos = AT.posiciones()
    ordenes = _ordenes()
    equity = float(cta.get("equity", 0.0))

    # pico HWM real desde la historia de la cuenta (para drawdown)
    curva = []
    try:
        h = AT.historia_cuenta(period="6M", timeframe="1D")
        if len(h):
            est = R.cargar_estado()
            hwm = float(h["equity"].max())
            if hwm > float(est.get("pico_equity", 0.0) or 0.0):
                est["pico_equity"] = hwm
                R.guardar_estado(est)
            curva = [{"fecha": d.strftime("%Y-%m-%d"), "equity": float(e)}
                     for d, e in h["equity"].items()]
    except Exception:
        pass

    exp = R.exposiciones(pos, equity)
    v95 = R.var_cartera(pos, equity, nivel=0.95) if pos else {"var_pct": float("nan"), "var_usd": 0, "corr_media": float("nan"), "vol_diaria": float("nan"), "n_cubiertos": 0}
    ev = R.evaluar_riesgo(cta, pos, ordenes, var_pct=(v95["var_pct"] if np.isfinite(v95.get("var_pct", float("nan"))) else None))
    libros = R.atribuir_libros(pos, ordenes)
    exp_libro = R.exposicion_por_libro(pos, equity, ordenes)

    px = AT.precio_actual(list(pos)) if pos else {}
    posiciones = []
    for t, p in sorted(pos.items(), key=lambda x: -abs(float(x[1].get("valor", 0)))):
        val = float(p.get("valor", 0.0))
        posiciones.append({"ticker": t, "qty": float(p.get("qty", 0)), "valor": val,
                           "pl": float(p.get("pl", 0.0)),
                           "peso": val / equity if equity else 0.0,
                           "precio": float(px.get(t, 0.0)),
                           "libro": libros.get(t, "sin_clasificar")})

    # tear-sheet + SPY normalizado para la curva
    eqs = pd.Series({pd.Timestamp(c["fecha"]): c["equity"] for c in curva}) if curva else pd.Series(dtype=float)
    met = A.metricas(eqs) if len(eqs) >= 3 else {"insuficiente": True, "n": len(eqs)}
    spy = []
    if len(eqs) >= 2:
        try:
            from fuentes import yahoo
            sp, _ = yahoo.historico("SPY", intervalo="1d", desde=str(eqs.index[0].date()), cliente=cli)
            spn = sp["close"].astype(float).reindex(eqs.index, method="ffill")
            spy = [{"fecha": d.strftime("%Y-%m-%d"), "v": float(x)} for d, x in spn.items() if np.isfinite(x)]
        except Exception:
            pass

    from registro import LOG, _fills
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    try:
        fills = _fills()
    except Exception:
        fills = []
    st = A.stats_trades(log)
    slp, _ = A.slippage(log, fills)

    # noticias: posiciones + indices clave
    tickers_news = [p["ticker"] for p in posiciones][:8] + ["SPY", "QQQ"]
    noticias = _noticias(list(dict.fromkeys(tickers_news)), cli)

    return {
        "ts": _madrid(),
        "clock": _clock(),
        "cuenta": {"equity": equity, "cash": float(cta.get("cash", 0.0)),
                   "pl_dia": ev["metricas"]["pl_dia"], "drawdown": ev["metricas"]["drawdown"],
                   "pico": ev["metricas"]["pico_equity"]},
        "riesgo": {"decision": ev["decision"], "razones": ev["razones"],
                   "bruta": exp["bruta"], "neta": exp["neta"], "n": exp["n"],
                   "peso_max": exp["peso_max"], "hhi": exp["hhi"],
                   "var95_pct": v95.get("var_pct"), "var95_usd": v95.get("var_usd"),
                   "vol_diaria": v95.get("vol_diaria"), "corr_media": v95.get("corr_media")},
        "posiciones": posiciones,
        "libros": [{"libro": k, "bruta": v["bruta"], "pl": v["pl"], "n": v["n"]}
                   for k, v in sorted(exp_libro.items(), key=lambda x: -x[1]["bruta"])],
        "curva": curva, "spy": spy,
        "tearsheet": met,
        "ops": {**st, "slippage_bps": slp.get("medio_bps")},
        "mercado": _mercado(cli),
        "noticias": noticias,
    }


def generar(datos):
    base = os.path.dirname(os.path.abspath(__file__))
    plantilla = open(os.path.join(base, "web", "plantilla.html"), encoding="utf-8").read()
    html = plantilla.replace("__DATOS__", json.dumps(datos, ensure_ascii=False))
    os.makedirs(os.path.join(base, "web"), exist_ok=True)
    with open(os.path.join(base, "web", "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
    os.makedirs(os.path.join(base, "datos"), exist_ok=True)
    with open(os.path.join(base, "datos", "dashboard.json"), "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    d = reunir()
    if "--solo-json" in sys.argv:
        print(json.dumps(d, ensure_ascii=False, indent=2)[:3000])
    else:
        generar(d)
        print(f"web/dashboard.html generado | equity ${d['cuenta']['equity']:,.0f} | "
              f"{len(d['posiciones'])} pos | {len(d['noticias'])} noticias | "
              f"{len(d['mercado'])} mercado | riesgo {d['riesgo']['decision']}")
