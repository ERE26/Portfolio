"""Alpaca Market Data: barras intradia con mas historico que Yahoo (feed IEX gratis).

El feed IEX (gratuito) da barras intradia desde ~julio 2020 (~6 anos), frente a
los 2 anos de 1h / 2 meses de 15m de Yahoo. IEX es un exchange pequeno, asi que
el VOLUMEN es parcial (una fraccion del mercado), pero el OHLC es correcto y el
volumen relativo (RVOL) sigue siendo un proxy valido. Suficiente para un estudio
academico de intradia.

Claves: env APCA_API_KEY_ID / APCA_API_SECRET_KEY, o archivo datos/.alpaca_key
(dos lineas: key id y secret). Nunca se suben al repositorio.
"""
import os
import time

import pandas as pd

from .http import Cliente

BASE = "https://data.alpaca.markets/v2/stocks"
COLUMNAS = ["open", "high", "low", "close", "volume"]


def _claves(kid=None, sec=None):
    kid = kid or os.environ.get("APCA_API_KEY_ID")
    sec = sec or os.environ.get("APCA_API_SECRET_KEY")
    if not (kid and sec):
        for ruta in ("datos/.alpaca_key", os.path.expanduser("~/.alpaca_key")):
            if os.path.exists(ruta):
                partes = open(ruta).read().split()
                if len(partes) >= 2:
                    kid, sec = partes[0], partes[1]
                    break
    if not (kid and sec):
        raise RuntimeError("Faltan las claves de Alpaca (APCA_API_KEY_ID / APCA_API_SECRET_KEY "
                           "o datos/.alpaca_key)")
    return kid, sec


def historico(ticker, intervalo="1Hour", desde="2020-08-01", hasta=None, feed="iex",
              kid=None, sec=None, cliente=None, max_paginas=120):
    """Barras intradia de un valor, siguiendo la paginacion de Alpaca.

    intervalo: '1Hour', '15Min', '5Min', '1Day'. Devuelve OHLCV indexado por fecha
    (hora de mercado, UTC->naive). Solo horas regulares de mercado.
    """
    kid, sec = _claves(kid, sec)
    cli = cliente or Cliente(ttl=30 * 24 * 3600)
    h = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    hasta = hasta or pd.Timestamp.now().strftime("%Y-%m-%d")
    filas, token, paginas = [], None, 0
    while paginas < max_paginas:
        params = {"timeframe": intervalo, "start": desde, "end": hasta, "feed": feed,
                  "limit": 10000, "adjustment": "all"}
        if token:
            params["page_token"] = token
        d = cli.get_json(f"{BASE}/{ticker}/bars", params=params, headers=h,
                         cache=(token is None))   # solo cachea la 1a pagina (clave estable)
        filas.extend(d.get("bars", []))
        token = d.get("next_page_token")
        paginas += 1
        if not token:
            break
        time.sleep(0.05)
    if not filas:
        raise RuntimeError(f"Alpaca sin datos para {ticker}")
    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.set_index("fecha").sort_index()
    # solo sesion regular (9:30-16:00 ET)
    hm = df.index.hour * 60 + df.index.minute
    df = df[(hm >= 570) & (hm <= 960)]
    return df[COLUMNAS]


def diario(ticker, desde="2023-01-01", hasta=None, feed="iex", kid=None, sec=None, cliente=None):
    """Barras DIARIAS (1Day) de un valor, SIN filtrar por horario (las usa, p.ej., el SMA200
    del libro overnight). Devuelve OHLCV indexado por fecha (naive), ajustado por splits/div."""
    kid, sec = _claves(kid, sec)
    cli = cliente or Cliente(ttl=12 * 3600)
    h = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    hasta = hasta or pd.Timestamp.now().strftime("%Y-%m-%d")
    filas, token, paginas = [], None, 0
    while paginas < 20:
        params = {"timeframe": "1Day", "start": desde, "end": hasta, "feed": feed,
                  "limit": 10000, "adjustment": "all"}
        if token:
            params["page_token"] = token
        d = cli.get_json(f"{BASE}/{ticker}/bars", params=params, headers=h, cache=(token is None))
        filas.extend(d.get("bars", []))
        token = d.get("next_page_token")
        paginas += 1
        if not token:
            break
        time.sleep(0.05)
    if not filas:
        raise RuntimeError(f"Alpaca sin datos diarios para {ticker}")
    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df.set_index("fecha").sort_index()[COLUMNAS]


def descargar_universo(tickers, intervalo="1Hour", desde="2020-08-01", carpeta="datos/intraday",
                       feed="iex", pausa=0.3, verbose=True, minimo=200):
    """Descarga barras intradia del universo y guarda un CSV por ticker."""
    sub = os.path.join(carpeta, intervalo)
    os.makedirs(sub, exist_ok=True)
    cli = Cliente(ttl=30 * 24 * 3600)
    informe = []
    for i, t in enumerate(tickers):
        ruta = os.path.join(sub, f"{_safe(t)}.csv")
        if os.path.exists(ruta) and (time.time() - os.path.getmtime(ruta) < 7 * 24 * 3600):
            informe.append((t, "cache"))
            continue
        try:
            df = historico(t, intervalo, desde, feed=feed, cliente=cli)
            if len(df) < minimo:
                informe.append((t, f"pocos:{len(df)}"))
                continue
            df.to_csv(ruta)
            informe.append((t, len(df)))
            if verbose and (i % 20 == 0 or i == len(tickers) - 1):
                print(f"  [{intervalo}] {i+1}/{len(tickers)}: {t} ({len(df)} barras)")
        except Exception as e:
            informe.append((t, f"error:{str(e)[:30]}"))
        time.sleep(pausa)
    return informe


def _safe(t):
    return t.replace("=", "_eq_").replace("^", "_ix_").replace("/", "_")
