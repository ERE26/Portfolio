"""Smell test de Massive (plan gratis Stocks Basic) antes de pagar Starter.

Solo lectura: no envia ordenes ni toca brokers. ~12 peticiones (~3 min por el
limite de 5 req/min). Uso:  MASSIVE_API_KEY=... python validar_massive.py

Comprueba:
  1. Conector: la clave autentica y la respuesta trae la forma esperada.
  2. EOD: barras diarias de SPY y AAPL contra Yahoo (retornos, cierre, volumen,
     sesiones que faltan) y profundidad real de historial del plan gratis.
  3. Splits: el ajuste adjusted=true cuadra con el ratio del split.
  4. Survivorship: hay deslistados con fecha de baja y sus barras se pueden bajar.
"""
import datetime as dt
import json
import sys

import pandas as pd

from fuentes import massive as mv
from fuentes import yahoo

HOY = dt.date.today()
TICKERS = ["SPY", "AAPL"]
veredicto = {}


def seccion(t):
    print(f"\n=== {t} ===")


def comparar_con_yahoo(cm, t, desde, hasta):
    m = mv.barras_diarias(t, desde, hasta, ajustado=True, cm=cm)
    y, _ = yahoo.historico(t, "1d", desde=desde, hasta=hasta)
    comun = m.index.intersection(y.index)
    solo_y = y.index.difference(m.index)
    solo_m = m.index.difference(y.index)
    rm, ry = m["close"].pct_change(), y["close"].pct_change()
    dr = (rm - ry).reindex(comun).dropna().abs()
    # Yahoo ajusta por dividendos y Massive solo por splits: el cierre diverge
    # poco a poco (drift ~ yield), los retornos solo el dia ex-dividendo.
    dcierre = ((m["close"] - y["close"]) / y["close"]).reindex(comun).abs()
    dvol = ((m["volume"] - y["volume"]) / y["volume"]).reindex(comun).dropna()
    r = {"barras_massive": len(m), "barras_yahoo": len(y), "comunes": len(comun),
         "faltan_en_massive": [str(d.date()) for d in solo_y[-10:]],
         "faltan_en_yahoo": [str(d.date()) for d in solo_m[-10:]],
         "ret_dif_mediana_pb": round(dr.median() * 1e4, 2),
         "ret_dif_p99_pb": round(dr.quantile(.99) * 1e4, 2),
         "dias_ret_dif_>10pb": int((dr > 1e-3).sum()),
         "cierre_dif_ultimo_%": round(dcierre.iloc[-1] * 100, 3),
         "vol_dif_mediana_%": round(dvol.median() * 100, 2),
         "primera_fecha": str(m.index.min().date()) if len(m) else None,
         "ultima_fecha": str(m.index.max().date()) if len(m) else None}
    print(json.dumps({t: r}, indent=1))
    return r


def main():
    cm = mv.ClienteMassive()

    seccion("1. Conector")
    d = cm.get("/v3/reference/tickers/AAPL")
    ok = d.get("status") == "OK" and d.get("results", {}).get("ticker") == "AAPL"
    print("detalle AAPL:", {k: d.get("results", {}).get(k)
                            for k in ("ticker", "name", "primary_exchange", "cik", "active")})
    veredicto["conector"] = ok

    seccion("2. EOD vs Yahoo (ultimos ~120 dias)")
    desde = (HOY - dt.timedelta(days=180)).isoformat()
    eod = {t: comparar_con_yahoo(cm, t, desde, HOY.isoformat()) for t in TICKERS}
    veredicto["eod"] = all(r["ret_dif_mediana_pb"] < 1 and r["dias_ret_dif_>10pb"] <= 3
                           and not r["faltan_en_massive"] for r in eod.values())
    # retraso: el plan gratis es fin de dia; la ultima barra deberia ser la sesion previa
    veredicto["ultima_barra"] = eod["SPY"]["ultima_fecha"]

    seccion("2b. Profundidad de historial (pido desde 2015)")
    h = mv.barras_diarias("SPY", "2015-01-01", HOY.isoformat(), cm=cm)
    ini = h.index.min().date() if len(h) else None
    print(f"SPY: {len(h)} barras, primera {ini}")
    veredicto["historial_desde"] = str(ini)

    seccion("3. Splits en la ventana accesible")
    s = mv.splits((HOY - dt.timedelta(days=700)).isoformat(), HOY.isoformat(), limite=50, cm=cm)
    veredicto["split_ok"] = None
    if len(s):
        # un split de un valor liquido (ticker corto y sin sufijos raros)
        cand = s[s["ticker"].str.fullmatch(r"[A-Z]{1,4}")].copy()
        cand = cand[(cand["split_from"] != cand["split_to"])]
        if len(cand):
            sp = cand.iloc[0]
            fe = pd.Timestamp(sp["execution_date"])
            a, b = (fe - pd.Timedelta(days=15)).date().isoformat(), (fe + pd.Timedelta(days=15)).date().isoformat()
            adj = mv.barras_diarias(sp["ticker"], a, b, ajustado=True, cm=cm)
            raw = mv.barras_diarias(sp["ticker"], a, b, ajustado=False, cm=cm)
            antes = raw.index < fe
            if antes.any() and len(adj):
                ratio = (raw["close"][antes] / adj["close"].reindex(raw.index)[antes]).median()
                esperado = sp["split_to"] / sp["split_from"]
                veredicto["split_ok"] = bool(abs(ratio / esperado - 1) < 0.01)
                print(f"{sp['ticker']} {sp['split_from']}:{sp['split_to']} el {fe.date()} -> "
                      f"ratio raw/adj {ratio:.4f} (esperado {esperado:.4f})")

    seccion("4. Survivorship: deslistados")
    dl = mv.tickers_referencia(activos=False, paginas=2, cm=cm, type="CS")
    print(f"deslistados (CS) en 2 paginas: {len(dl)}  columnas: {list(dl.columns)[:12]}")
    veredicto["deslistados_listados"] = int(len(dl))
    veredicto["deslistados_con_fecha_%"] = None
    veredicto["deslistados_con_barras"] = None
    if len(dl) and "delisted_utc" in dl:
        fechas = pd.to_datetime(dl["delisted_utc"], utc=True, errors="coerce")
        veredicto["deslistados_con_fecha_%"] = round(fechas.notna().mean() * 100, 1)
        print("rango de bajas:", fechas.min(), "->", fechas.max())
        # deslistados recientes (dentro del historial del plan gratis): pido sus barras
        recientes = dl.assign(f=fechas).dropna(subset=["f"])
        recientes = recientes[recientes["f"] > pd.Timestamp(HOY - dt.timedelta(days=500), tz="UTC")]
        muestra = recientes.sort_values("f", ascending=False).head(3)
        con = 0
        for _, row in muestra.iterrows():
            fin = row["f"].date()
            b = mv.barras_diarias(row["ticker"], (fin - dt.timedelta(days=90)).isoformat(),
                                  fin.isoformat(), ajustado=True, cm=cm)
            print(f"  {row['ticker']:6s} {str(row.get('name'))[:35]:35s} baja {fin}  "
                  f"barras ultimos 90d: {len(b)}  ultima {b.index.max().date() if len(b) else '-'}")
            con += len(b) > 0
        veredicto["deslistados_con_barras"] = f"{con}/{len(muestra)}"

    seccion("VEREDICTO")
    print(json.dumps(veredicto, indent=1, default=str))
    print(f"peticiones a Massive: {cm.llamadas}")
    return 0 if veredicto["conector"] else 1


if __name__ == "__main__":
    sys.exit(main())
