"""Descarga por lotes del universo, multi-temporalidad, con cache y throttling.

Yahoo tolera bastantes peticiones pero no ilimitadas; entre ticker y ticker se
deja una pausa corta y todo queda cacheado en disco, de modo que repetir la
descarga el mismo dia no vuelve a pegar a la API.

Limites REALES de historico intradia en Yahoo (no es cosa del codigo):
  5m/15m/30m  ~60 dias      1h  ~730 dias      1d/1wk/1mo  decadas
Por eso "todo el historico" significa cosas distintas segun la temporalidad, y
se documenta en el propio guardado.
"""
import os
import time

import pandas as pd

from . import yahoo
from .http import Cliente
from .universo import catalogo

RANGO_POR_INTERVALO = {"5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d",
                       "60m": "730d", "1d": "max", "1wk": "max", "1mo": "max"}


def descargar_universo(tickers=None, intervalos=("1d",), carpeta="datos/mercado",
                       pausa=0.4, cliente=None, verbose=True, minimo=60):
    """Descarga y guarda un CSV por (ticker, intervalo). Devuelve un informe."""
    cat = catalogo()
    tickers = tickers or list(cat)
    cli = cliente or Cliente(ttl=6 * 3600)
    os.makedirs(carpeta, exist_ok=True)
    informe = []
    for iv in intervalos:
        rango = RANGO_POR_INTERVALO.get(iv, "2y")
        sub = os.path.join(carpeta, iv)
        os.makedirs(sub, exist_ok=True)
        ok = 0
        for i, t in enumerate(tickers):
            ruta = os.path.join(sub, f"{_safe(t)}.csv")
            if os.path.exists(ruta) and (time.time() - os.path.getmtime(ruta) < 6 * 3600):
                ok += 1
                informe.append({"ticker": t, "intervalo": iv, "velas": _nfilas(ruta),
                                "estado": "cache"})
                continue
            try:
                df, m = yahoo.historico(t, intervalo=iv, rango=rango, cliente=cli)
                if len(df) < minimo:
                    informe.append({"ticker": t, "intervalo": iv, "velas": len(df),
                                    "estado": "pocos_datos"})
                    continue
                df.to_csv(ruta)
                ok += 1
                informe.append({"ticker": t, "intervalo": iv, "velas": len(df),
                                "estado": "ok", "desde": df.index[0].date(),
                                "hasta": df.index[-1].date()})
                if verbose and (i % 20 == 0 or i == len(tickers) - 1):
                    print(f"  [{iv}] {i + 1}/{len(tickers)} descargados, ultimo {t} ({len(df)} velas)")
            except Exception as e:
                informe.append({"ticker": t, "intervalo": iv, "velas": 0,
                                "estado": f"error: {str(e)[:40]}"})
            time.sleep(pausa)
        if verbose:
            print(f"  [{iv}] completado: {ok}/{len(tickers)} con datos")
    inf = pd.DataFrame(informe)
    inf.to_csv(os.path.join(carpeta, "_informe_descarga.csv"), index=False)
    return inf


def cargar_mercado(carpeta="datos/mercado", intervalo="1d", tickers=None):
    """Carga los CSV descargados de una temporalidad como dict de DataFrames + meta."""
    from .universo import catalogo
    cat = catalogo()
    sub = os.path.join(carpeta, intervalo)
    if not os.path.isdir(sub):
        raise RuntimeError(f"No existe {sub}. Ejecuta antes la descarga.")
    precios, filas = {}, []
    for f in sorted(os.listdir(sub)):
        if not f.endswith(".csv"):
            continue
        t = _unsafe(f[:-4])
        if tickers and t not in tickers:
            continue
        df = pd.read_csv(os.path.join(sub, f), parse_dates=["fecha"]).set_index("fecha")
        if len(df) < 60:
            continue
        precios[t] = df
        info = cat.get(t, {"clase": "desconocida", "sector": "desconocido"})
        filas.append({"ticker": t, "clase": info["clase"], "sector": info["sector"],
                      "nombre": t})
    return precios, pd.DataFrame(filas)


def _safe(t):
    return t.replace("=", "_eq_").replace("^", "_ix_").replace("/", "_")


def _unsafe(s):
    return s.replace("_eq_", "=").replace("_ix_", "^").replace("_", "/") \
        if ("_eq_" in s or "_ix_" in s) else s


def _nfilas(ruta):
    try:
        return sum(1 for _ in open(ruta)) - 1
    except Exception:
        return 0
