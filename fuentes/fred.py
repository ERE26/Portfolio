"""FRED (Reserva Federal de St. Louis): datos macro de regimen.

Requiere una clave gratuita (env FRED_API_KEY). Aporta lo que el grafico de un
valor no contiene: el estado del mercado en su conjunto.

Series usadas:
  VIXCLS    indice de volatilidad implicita (miedo del mercado)
  DGS10 / DGS2   tipos a 10 y 2 anos -> pendiente de la curva (senal de ciclo)
  BAMLH0A0HYM2   spread de bonos high yield (estres de credito)
Todas son diarias y point-in-time (el dato de una fecha se conocia esa fecha).
"""
import os

import numpy as np
import pandas as pd

from .http import Cliente

BASE = "https://api.stlouisfed.org/fred/series/observations"
SERIES = {"vix": "VIXCLS", "tipo10": "DGS10", "tipo2": "DGS2", "hy": "BAA10Y"}


def _clave(clave):
    clave = clave or os.environ.get("FRED_API_KEY")
    if not clave:
        # archivo local (gitignored) como ultimo recurso, para no exponer el secreto
        for ruta in ("datos/.fred_key", os.path.expanduser("~/.fred_key")):
            if os.path.exists(ruta):
                clave = open(ruta).read().strip()
                if clave:
                    break
    if not clave:
        raise RuntimeError("Falta FRED_API_KEY (consíguela gratis en "
                           "fred.stlouisfed.org/docs/api/api_key.html)")
    return clave


def serie(series_id, clave=None, cliente=None, desde="2005-01-01"):
    cli = cliente or Cliente(ttl=24 * 3600)
    d = cli.get_json(BASE, params={"series_id": series_id, "api_key": _clave(clave),
                                   "file_type": "json", "observation_start": desde})
    obs = d.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).set_index("date")["value"]


def rasgos_macro(fechas, clave=None, cliente=None, desde="2005-01-01"):
    """Features macro alineadas al calendario de mercado (ffill point-in-time)."""
    cols = ["macro_vix", "macro_vix_cambio", "macro_pendiente_curva",
            "macro_spread_credito", "macro_tendencia_tipos"]
    try:
        clave = _clave(clave)
    except RuntimeError:
        return pd.DataFrame({c: 0.0 for c in cols}, index=fechas)
    cli = cliente or Cliente(ttl=24 * 3600)
    try:
        s = {k: serie(v, clave, cli, desde) for k, v in SERIES.items()}
    except Exception:
        # fallo de red a mitad: se devuelve neutro en lugar de romper la construccion
        return pd.DataFrame({c: 0.0 for c in cols}, index=fechas)
    idx = pd.DatetimeIndex(fechas)
    vix = s["vix"].reindex(idx, method="ffill")
    t10 = s["tipo10"].reindex(idx, method="ffill")
    t2 = s["tipo2"].reindex(idx, method="ffill")
    hy = s["hy"].reindex(idx, method="ffill")
    out = pd.DataFrame(index=idx)
    out["macro_vix"] = (vix / 20.0 - 1.0).fillna(0.0)                    # centrado en 20
    out["macro_vix_cambio"] = vix.pct_change(5).fillna(0.0)
    out["macro_pendiente_curva"] = (t10 - t2).fillna(0.0)               # curva 10-2
    out["macro_spread_credito"] = (hy / 2.0 - 1.0).fillna(0.0)          # Baa-Tesoro, centrado en 2%
    out["macro_tendencia_tipos"] = t10.diff(20).fillna(0.0)
    return out.fillna(0.0)
