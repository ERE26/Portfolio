"""Yahoo Finance: precios OHLCV historicos y titulares por valor.

Usa el endpoint publico /v8/finance/chart, el mismo que alimenta los graficos de
finance.yahoo.com. Devuelve los datos en el mismo formato OHLCV que usa el resto
del proyecto, asi que un DataFrame de aqui entra directo en patrones.ventanas.
"""
import datetime as dt

import numpy as np
import pandas as pd

from .http import Cliente

BASE = "https://query1.finance.yahoo.com"

# rango maximo que Yahoo sirve por temporalidad (aprox.); pedir mas es tirar datos
RANGO_MAX = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "60m": "730d",
             "1h": "730d", "1d": "max", "1wk": "max", "1mo": "max"}
COLUMNAS = ["open", "high", "low", "close", "volume"]


def historico(ticker, intervalo="1d", rango=None, desde=None, hasta=None, cliente=None):
    """OHLCV de un valor. 'intervalo' en {1d,1wk,1mo,1h,15m,5m,1m,...}.

    Se puede pedir por 'rango' (p.ej. '2y', 'max') o por fechas 'desde'/'hasta'.
    """
    cli = cliente or Cliente()
    params = {"interval": intervalo, "includePrePost": "false", "events": "div,split"}
    if desde is not None:
        params["period1"] = int(pd.Timestamp(desde).timestamp())
        params["period2"] = int(pd.Timestamp(hasta or dt.datetime.now()).timestamp())
    else:
        params["range"] = rango or RANGO_MAX.get(intervalo, "2y")

    d = cli.get_json(f"{BASE}/v8/finance/chart/{ticker}", params=params)
    res = d.get("chart", {}).get("result")
    if not res:
        err = d.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo sin datos para {ticker}: {err}")
    res = res[0]
    ts = res.get("timestamp")
    if not ts:
        raise RuntimeError(f"Yahoo no devolvio velas para {ticker} (intervalo {intervalo})")
    q = res["indicators"]["quote"][0]
    # cierre ajustado por dividendos/splits si esta disponible (mejor para el largo plazo)
    ajust = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    idx = pd.to_datetime(np.array(ts), unit="s")
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"],
                       "close": q["close"], "volume": q["volume"]}, index=idx)
    if ajust is not None:
        factor = pd.Series(ajust, index=idx) / df["close"]
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * factor
    df.index.name = "fecha"
    # en diaria/semanal/mensual el timestamp trae la hora de apertura del mercado;
    # se normaliza a fecha limpia para que encaje con el calendario del pipeline.
    if intervalo in ("1d", "1wk", "1mo"):
        df.index = df.index.normalize()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    meta = {"symbol": res["meta"].get("symbol", ticker),
            "moneda": res["meta"].get("currency"),
            "tipo": res["meta"].get("instrumentType"),
            "exchange": res["meta"].get("exchangeName")}
    return df[COLUMNAS], meta


def universo(tickers, intervalo="1d", rango="5y", cliente=None, sectores=None, verbose=True):
    """Descarga varios valores y devuelve (precios, meta) listos para el pipeline."""
    cli = cliente or Cliente()
    precios, filas = {}, []
    for t in tickers:
        try:
            df, m = historico(t, intervalo=intervalo, rango=rango, cliente=cli)
            if len(df) < 60:
                if verbose:
                    print(f"  {t}: solo {len(df)} velas, se salta")
                continue
            precios[t] = df
            filas.append({"ticker": t, "sector": (sectores or {}).get(t, "desconocido"),
                          "nombre": m["symbol"], "moneda": m["moneda"],
                          "exchange": m["exchange"]})
            if verbose:
                print(f"  {t}: {len(df)} velas {df.index[0].date()} a {df.index[-1].date()}"
                      f" ({m['exchange']}, {m['moneda']})")
        except Exception as e:
            if verbose:
                print(f"  {t}: fallo -> {e}")
    if not precios:
        raise RuntimeError("Yahoo no devolvio datos de ningun ticker")
    return precios, pd.DataFrame(filas)


def titulares(ticker, n=10, cliente=None):
    """Titulares recientes del valor. OJO: son recientes, sin historico profundo,
    asi que sirven para la prediccion EN VIVO, no para entrenar (ver noticias.py)."""
    cli = cliente or Cliente()
    d = cli.get_json(f"{BASE}/v1/finance/search",
                     params={"q": ticker, "newsCount": n, "quotesCount": 0},
                     cache=True)
    filas = []
    for n_ in d.get("news", []):
        t = n_.get("providerPublishTime")
        filas.append({"fecha": pd.to_datetime(t, unit="s") if t else pd.NaT,
                      "ticker": ticker, "titular": n_.get("title", ""),
                      "fuente": n_.get("publisher", ""), "url": n_.get("link", "")})
    return pd.DataFrame(filas).dropna(subset=["titular"])
