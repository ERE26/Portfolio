"""Massive (antes Polygon.io): barras EOD y universo de referencia CON deslistados.

Para que sirve aqui: quitar el sesgo de supervivencia. Yahoo solo sirve los
valores que siguen cotizando; Massive lista tambien los tickers deslistados
(active=false, con fecha de baja) y, segun plan, sus barras historicas.

La clave se lee de la variable de entorno MASSIVE_API_KEY (con respaldo en
datos/.massive_key, que esta en .gitignore) y viaja en la cabecera Authorization
(no en la URL), para que no acabe en logs ni en mensajes de error. Nunca se
hardcodea. El plan gratis (Stocks Basic) limita a ~5 peticiones/minuto: el
cliente espaciado de este modulo respeta ese ritmo.
"""
import os
import time

import pandas as pd

from .http import Cliente

BASE = "https://api.massive.com"
COLUMNAS = ["open", "high", "low", "close", "volume"]
PAUSA_MIN = 12.5  # s entre peticiones: 60/5 req/min + margen


def _claves(k=None):
    """API key de Massive. Prioridad: argumento -> env MASSIVE_API_KEY ->
    primera linea de datos/.massive_key o ~/.massive_key (gitignored). Nunca se
    escribe en la URL ni se registra; solo viaja en la cabecera Authorization."""
    k = (k or os.environ.get("MASSIVE_API_KEY") or "").strip()
    if not k:
        for ruta in ("datos/.massive_key", os.path.expanduser("~/.massive_key")):
            if os.path.exists(ruta):
                lineas = open(ruta).read().split()
                if lineas:
                    k = lineas[0].strip()
                    break
    if not k:
        raise RuntimeError("Falta la API key de Massive (env MASSIVE_API_KEY "
                           "o datos/.massive_key)")
    return k


# Alias retrocompatible: antes la funcion se llamaba clave().
clave = _claves


class ClienteMassive:
    """Envuelve fuentes.http.Cliente con autenticacion y limite de ritmo."""

    def __init__(self, pausa_min=PAUSA_MIN, cache_dir="datos/cache", ttl=24 * 3600):
        self.cli = Cliente(cache_dir=cache_dir, ttl=ttl, reintentos=4, pausa=15.0)
        self.cli.sesion.headers["Authorization"] = f"Bearer {_claves()}"
        self.pausa_min = pausa_min
        self._ultima = 0.0
        self.llamadas = 0

    def get(self, ruta, params=None, cache=True):
        url = ruta if ruta.startswith("http") else f"{BASE}{ruta}"
        espera = self.pausa_min - (time.time() - self._ultima)
        if espera > 0:
            time.sleep(espera)
        try:
            return self.cli.get_json(url, params=params, cache=cache)
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


_TF_MIN = {"1Min": 1, "5Min": 5, "15Min": 15, "1min": 1, "5min": 5, "15min": 15}


def barras_intradia(ticker, tf, desde, hasta, ajustado=True, cm=None,
                    solo_regular=True, max_paginas=200, cache=True):
    """OHLCV intradia (barras de minuto agregadas) para un valor, con PAGINACION.

    tf: '5Min' -> 5, '1Min' -> 1 (tambien '15Min'). El endpoint es
    GET /v2/aggs/ticker/{t}/range/{mult}/minute/{from}/{to}. Se sigue next_url
    hasta agotar (cada pagina trae hasta `limit` barras). El indice es datetime
    convertido UTC->America/New_York y tz-naive; por defecto se FILTRA a horario
    regular 09:30-16:00 (Massive sirve tambien premarket/afterhours). Columnas
    open, high, low, close, volume. Mismo estilo defensivo que barras_diarias.
    """
    mult = _TF_MIN.get(tf)
    if mult is None:
        raise ValueError(f"tf no soportado: {tf!r} (usa '1Min'/'5Min'/'15Min')")
    cm = cm or ClienteMassive()
    ruta = f"/v2/aggs/ticker/{ticker}/range/{mult}/minute/{desde}/{hasta}"
    params = {"adjusted": str(ajustado).lower(), "sort": "asc", "limit": 50000}
    filas, d = [], cm.get(ruta, params, cache=cache)
    for i in range(max_paginas):
        filas += d.get("results") or []
        nxt = d.get("next_url")
        if not nxt or i == max_paginas - 1:
            break
        d = cm.get(nxt, cache=cache)  # next_url ya trae el cursor; la clave va en la cabecera
    if not filas:
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.DataFrame(filas)
    idx = (pd.to_datetime(df["t"], unit="ms", utc=True)
           .dt.tz_convert("America/New_York").dt.tz_localize(None))
    out = pd.DataFrame({"open": df["o"].values, "high": df["h"].values,
                        "low": df["l"].values, "close": df["c"].values,
                        "volume": df["v"].values}, index=pd.DatetimeIndex(idx, name="fecha"))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if solo_regular:
        t = out.index.time
        import datetime as _dt
        out = out[(t >= _dt.time(9, 30)) & (t < _dt.time(16, 0))]
    return out


def grouped_diario(fecha, ajustado=True, cm=None):
    """TODAS las acciones US de un dia (1 llamada). Incluye los que ESE dia
    cotizaban aunque hoy esten deslistados -> base para universo point-in-time
    sin sesgo de supervivencia. GET /v2/aggs/grouped/locale/us/market/stocks/{d}.
    Devuelve DataFrame con ticker, open, high, low, close, volume (vacio si el
    plan no lo incluye o la fecha no es sesion). Lanza si la API devuelve error.
    """
    cm = cm or ClienteMassive()
    d = cm.get(f"/v2/aggs/grouped/locale/us/market/stocks/{fecha}",
               {"adjusted": str(ajustado).lower()})
    res = d.get("results") or []
    if not res:
        return pd.DataFrame(columns=["ticker"] + COLUMNAS)
    df = pd.DataFrame(res)
    return pd.DataFrame({"ticker": df["T"].values, "open": df["o"].values,
                         "high": df["h"].values, "low": df["l"].values,
                         "close": df["c"].values, "volume": df["v"].values})


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
