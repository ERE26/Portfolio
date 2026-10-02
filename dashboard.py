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


def _ohlc(ticker, cli, n=70, nombre=None):
    """Velas diarias (OHLC) de un instrumento para el grafico de velas. Yahoo."""
    try:
        from fuentes import yahoo
        df, _ = yahoo.historico(ticker, intervalo="1d", rango="4mo", cliente=cli)
        df = df.tail(n)
        bars = [{"f": d.strftime("%Y-%m-%d"), "o": float(r["open"]), "h": float(r["high"]),
                 "l": float(r["low"]), "c": float(r["close"])} for d, r in df.iterrows()]
        if len(bars) < 2:
            return {}
        last, prev = bars[-1]["c"], bars[-2]["c"]
        return {"ticker": ticker, "nombre": nombre or ticker, "bars": bars,
                "o": bars[-1]["o"], "h": bars[-1]["h"], "l": bars[-1]["l"], "c": last,
                "chg": last / prev - 1 if prev else 0.0}
    except Exception:
        return {}


def _regimen(cli):
    """Overlay de regimen (gestion de vol + VIX). Multiplicador de exposicion sugerido."""
    try:
        from patrones.regimen import multiplicador
        m = multiplicador(cli=cli)
        vx = m.get("vix", {})
        return {"mult": m["mult"], "vol_managed": m["vol_managed"], "factor_vix": m.get("factor_vix"),
                "vol_realizada": m.get("vol_realizada"), "vix": vx.get("vix"), "vix3m": vx.get("vix3m"),
                "ratio": vx.get("ratio"), "estado": vx.get("estado"), "risk_on": vx.get("risk_on")}
    except Exception:
        return {}


def _modelo():
    """Cache de inteligencia del modelo (importancias + edge matrix), generado por
    modelo.py (lento, se regenera a diario). Si no existe, se omite la seccion."""
    try:
        return json.load(open("datos/modelo_cache.json"))
    except Exception:
        return {}


def _watchlist():
    try:
        d = json.load(open("datos/watchlist.json"))
        return sorted(d.keys()) if isinstance(d, dict) else []
    except Exception:
        return []


def _calendario(cli, max_ev=8):
    """Calendario economico de la semana (Forex Factory): eventos de impacto alto/medio
    en USD y EUR, proximos primero."""
    try:
        from fuentes import forexfactory as FF
        partes = []
        for w in ("esta_semana", "proxima_semana"):
            try:
                partes.append(FF.calendario(w, cliente=cli))
            except Exception:
                pass
        df = pd.concat(partes, ignore_index=True) if partes else None
    except Exception:
        return []
    if df is None or len(df) == 0:
        return []
    try:
        df = df[df["pais"].isin(["USD", "EUR"])]
        df = df[df["impacto"].astype(str).str.lower().isin(["high", "medium"])]
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", utc=True)
        ahora = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
        df = df[df["fecha"] >= ahora].sort_values("fecha")      # solo PROXIMOS
        out = []
        for _, r in df.head(max_ev).iterrows():
            out.append({"fecha": str(r.get("fecha", ""))[:16], "pais": str(r.get("pais", "")),
                        "evento": str(r.get("evento", "")), "impacto": str(r.get("impacto", "")),
                        "prevision": str(r.get("prevision", "")), "anterior": str(r.get("anterior", ""))})
        return out
    except Exception:
        return []


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


def _nn(x):
    """float o None (descarta NaN) para construir el motivo sin basura."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _bitacora_dia():
    """Operaciones INTRADÍA de la sesión de HOY (America/New_York) con su MOTIVO y un
    resumen analítico. Lee datos/trades_intradia.csv (entrada, salida, pnl, señales)."""
    ruta = "datos/trades_intradia.csv"
    if not os.path.exists(ruta):
        return {"trades": [], "resumen": {}}
    try:
        df = pd.read_csv(ruta)
    except Exception:
        return {"trades": [], "resumen": {}}
    if df.empty:
        return {"trades": [], "resumen": {}}
    from bitacora import describir_motivo
    ny = "America/New_York"
    # la columna ts mezcla timestamps naive y con zona -> format="mixed" (si no, NaT)
    ent = pd.to_datetime(df["ts"], format="mixed", utc=True, errors="coerce").dt.tz_convert(ny)
    sal = pd.to_datetime(df.get("salida_ts"), format="mixed", utc=True, errors="coerce").dt.tz_convert(ny)
    hoy = pd.Timestamp.now(tz=ny).date()
    mask = (ent.dt.date == hoy) | (sal.dt.date == hoy)
    d = df[mask].copy()
    if d.empty:
        return {"trades": [], "resumen": {}}
    d["_ent"], d["_sal"] = ent[mask], sal[mask]
    trades = []
    for _, r in d.sort_values("_ent").iterrows():
        cerrado = str(r.get("resultado")) in ("win", "loss") and _nn(r.get("salida")) is not None
        hold = None
        if cerrado and pd.notna(r["_sal"]) and pd.notna(r["_ent"]):
            hold = max(0.0, (r["_sal"] - r["_ent"]).total_seconds() / 60.0)
        feats = {"rvol": _nn(r.get("rvol")), "dist_vwap": _nn(r.get("dist_vwap")),
                 "ret_sesion": _nn(r.get("ret_sesion")), "sent_noticias": _nn(r.get("sent_noticias"))}
        hora_m = r["_ent"].tz_convert("Europe/Madrid").strftime("%H:%M") if pd.notna(r["_ent"]) else str(r.get("hora", ""))
        trades.append({
            "hora": hora_m, "ticker": str(r.get("ticker", "")),
            "qty": int(_nn(r.get("qty")) or 0), "entrada": _nn(r.get("entrada")),
            "salida": _nn(r.get("salida")) if cerrado else None,
            "pnl": _nn(r.get("pnl")) if cerrado else None,
            "resultado": str(r.get("resultado")) if cerrado else "abierta",
            "hold_min": round(hold) if hold is not None else None,
            "rvol": _nn(r.get("rvol")),
            "motivo": describir_motivo(feats),
        })
    cerr = [t for t in trades if t["resultado"] in ("win", "loss")]
    pnls = [t["pnl"] for t in cerr if t["pnl"] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [t["hold_min"] for t in cerr if t["hold_min"] is not None]
    rvs = [t["rvol"] for t in trades if t["rvol"] is not None]
    resumen = {
        "n_total": len(trades), "n_cerradas": len(cerr), "n_abiertas": len(trades) - len(cerr),
        "ganadoras": len(wins), "perdedoras": len(losses),
        "win_rate": (len(wins) / len(pnls)) if pnls else None,
        "pnl_total": float(sum(pnls)) if pnls else 0.0,
        "pnl_medio": (float(sum(pnls)) / len(pnls)) if pnls else None,
        "mejor": max(pnls) if pnls else None, "peor": min(pnls) if pnls else None,
        "profit_factor": (sum(wins) / abs(sum(losses))) if (losses and sum(losses) != 0) else None,
        "hold_medio_min": (float(np.mean(holds)) if holds else None),
        "rvol_medio": (float(np.mean(rvs)) if rvs else None),
    }
    return {"trades": trades, "resumen": resumen}


def _auditoria_sl():
    """Post-mortem de los STOPS: de las intradías cerradas por SL, cuántas fueron sacadas
    en falso (stop tight, análisis bueno) vs nos salvaron (error de compra). Lee el log que
    genera auditoria_sl.py (datos/sl_auditoria.csv)."""
    ruta = "datos/sl_auditoria.csv"
    if not os.path.exists(ruta):
        return {"resumen": {"n": 0, "falso": 0, "salvo": 0, "neutro": 0}, "recientes": []}
    try:
        d = pd.read_csv(ruta)
    except Exception:
        return {"resumen": {"n": 0, "falso": 0, "salvo": 0, "neutro": 0}, "recientes": []}
    v = d["veredicto"].value_counts().to_dict()
    res = {"n": int(len(d)), "falso": int(v.get("SACADA_EN_FALSO", 0)),
           "salvo": int(v.get("SL_NOS_SALVO", 0)), "neutro": int(v.get("NEUTRO", 0))}
    recientes = []
    for _, r in d.tail(8).iloc[::-1].iterrows():
        recientes.append({"ticker": str(r.get("ticker", "")), "veredicto": str(r.get("veredicto", "")),
                          "detalle": str(r.get("detalle", ""))})
    return {"resumen": res, "recientes": recientes}


def _curva_intradia():
    """Evolución del equity SOLO de la sesión de HOY (intradía, 5-min), no la histórica.
    Devuelve puntos {hora (Madrid), equity, pl_pct vs apertura de sesión}."""
    try:
        h = AT.historia_cuenta(period="1D", timeframe="5Min")
    except Exception:
        return []
    if not len(h):
        return []
    idx = h.index
    try:
        idx = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert("Europe/Madrid")
    except Exception:
        pass
    eq = h["equity"].astype(float)
    base = float(eq.iloc[0]) if len(eq) else 0.0
    return [{"hora": t.strftime("%H:%M"), "equity": float(e),
             "pl_pct": (float(e) / base - 1.0) if base else 0.0}
            for t, e in zip(idx, eq)]


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
    try:
        from fuentes import universo as U
        cat = U.catalogo()
    except Exception:
        cat = {}
    posiciones = []
    for t, p in sorted(pos.items(), key=lambda x: -abs(float(x[1].get("valor", 0)))):
        val = float(p.get("valor", 0.0))
        pl = float(p.get("pl", 0.0))
        coste = val - pl
        posiciones.append({"ticker": t, "qty": float(p.get("qty", 0)), "valor": val, "pl": pl,
                           "pl_pct": (pl / coste) if coste else 0.0,
                           "peso": val / equity if equity else 0.0,
                           "precio": float(px.get(t, 0.0)),
                           "sector": cat.get(t, {}).get("sector", "otros"),
                           "libro": libros.get(t, "sin_clasificar")})
    # asignacion por sector (sobre el valor absoluto)
    sect = {}
    for p in posiciones:
        sect[p["sector"]] = sect.get(p["sector"], 0.0) + abs(p["valor"])
    tot_s = sum(sect.values()) or 1.0
    sectores = [{"sector": k, "valor": v, "peso": v / tot_s}
                for k, v in sorted(sect.items(), key=lambda x: -x[1])]
    # top movers del dia (por P&L%)
    movers = sorted([p for p in posiciones if p["valor"]], key=lambda x: -x["pl_pct"])
    top_movers = movers[:3] + movers[-3:] if len(movers) > 6 else movers

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
        "intradia": _bitacora_dia(),
        "auditoria_sl": _auditoria_sl(),
        "curva_intradia": _curva_intradia(),
        "grafico": _ohlc("SPY", cli, 70, "S&P 500 (SPY)"),
        "tearsheet": met,
        "ops": {**st, "slippage_bps": slp.get("mediana_bps", slp.get("medio_bps"))},
        "sectores": sectores,
        "top_movers": top_movers,
        "watchlist": _watchlist(),
        "regimen": _regimen(cli),
        "modelo": _modelo(),
        "calendario": _calendario(cli),
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
