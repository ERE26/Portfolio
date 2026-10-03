"""Massive (antes Polygon.io): barras EOD y universo de referencia CON deslistados.

Para que sirve aqui: quitar el sesgo de supervivencia. Yahoo solo sirve los
valores que siguen cotizando; Massive lista tambien los tickers deslistados
(active=false, con fecha de baja) y, segun plan, sus barras historicas.

La clave se lee de la variable de entorno MASSIVE_API_KEY y viaja en la cabecera
Authorization (no en la URL), para que no acabe en logs ni en mensajes de error.
El plan gratis (Stocks Basic) limita a ~5 peticiones/minuto: el cliente espaciado
de este modulo respeta ese ritmo.
"""
import os
import time

import pandas as pd

from .http import Cliente

BASE = "https://api.massive.com"
COLUMNAS = ["open", "high", "low", "close", "volume"]
PAUSA_MIN = 12.5  # s entre peticiones: 60/5 req/min + margen


def clave():
    k = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not k:
        raise RuntimeError("Falta la variable de entorno MASSIVE_API_KEY")
    return k


class ClienteMassive:
    """Envuelve fuentes.http.Cliente con autenticacion y limite de ritmo."""

    def __init__(self, pausa_min=PAUSA_MIN, cache_dir="datos/cache", ttl=24 * 3600):
        self.cli = Cliente(cache_dir=cache_dir, ttl=ttl, reintentos=4, pausa=15.0)
        self.cli.sesion.headers["Authorization"] = f"Bearer {clave()}"
        self.pausa_min = pausa_min
        self._ultima = 0.0
        self.llamadas = 0

    def get(self, ruta, params=None):
        url = ruta if ruta.startswith("http") else f"{BASE}{ruta}"
        espera = self.pausa_min - (time.time() - self._ultima)
        if espera > 0:
            time.sleep(espera)
        try:
            return self.cli.get_json(url, params=params)
        finally:
            self._ultima = time.time()
            self.llamadas += 1


def barras_diarias(ticker, desde, hasta, ajustado=True, cm=None):
    """OHLCV diario (indice = fecha de sesion NY). ajustado=True -> solo splits."""
    cm = cm or ClienteMassive()
    d = cm.get(f"/v2/aggs/ticker/{ticker}/range/1/day/{desde}/{hasta}",
               {"adjusted": str(ajustado).lower(), "sort": "asc", "limit": 50000})
    res = d.get("results") or []
    if not res:
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.DataFrame(res)
    idx = (pd.to_datetime(df["t"], unit="ms", utc=True)
           .dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize())
    out = pd.DataFrame({"open": df["o"].values, "high": df["h"].values,
                        "low": df["l"].values, "close": df["c"].values,
                        "volume": df["v"].values}, index=pd.DatetimeIndex(idx, name="fecha"))
    return out[~out.index.duplicated(keep="last")]


def tickers_referencia(activos=True, paginas=1, limite=1000, cm=None, **filtros):
    """Universo de referencia. activos=False -> deslistados (con 'delisted_utc')."""
    cm = cm or ClienteMassive()
    params = {"market": "stocks", "active": str(activos).lower(), "limit": limite, **filtros}
    filas, d = [], cm.get("/v3/reference/tickers", params)
    for i in range(paginas):
        filas += d.get("results") or []
        nxt = d.get("next_url")
        if not nxt or i == paginas - 1:
            break
        d = cm.get(nxt)  # next_url ya trae el cursor; la clave va en la cabecera
    return pd.DataFrame(filas)


def splits(desde, hasta, limite=50, cm=None, **filtros):
    cm = cm or ClienteMassive()
    d = cm.get("/v3/reference/splits", {"execution_date.gte": desde,
                                        "execution_date.lte": hasta, "limit": limite, **filtros})
    return pd.DataFrame(d.get("results") or [])
