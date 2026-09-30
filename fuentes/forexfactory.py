"""Forex Factory: calendario economico (feed JSON publico).

Eventos macro de la semana (tipos de interes, empleo, IPC, PMIs...) con su pais,
impacto (Low/Medium/High), la prevision y el dato anterior. Es informacion de
REGIMEN macro, no titulares de una empresa: util para forex e indices, y como
contexto general de riesgo (p.ej. "hoy hay un evento de alto impacto en USD").
"""
import pandas as pd

from .http import Cliente

FEEDS = {
    "esta_semana": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "semana_pasada": "https://nfs.faireconomy.media/ff_calendar_lastweek.json",
    "proxima_semana": "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
}
PESO_IMPACTO = {"High": 1.0, "Medium": 0.5, "Low": 0.2, "Holiday": 0.0}


def calendario(cual="esta_semana", cliente=None):
    cli = cliente or Cliente(ttl=3600)
    d = cli.get_json(FEEDS[cual])
    df = pd.DataFrame(d)
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["peso_impacto"] = df["impact"].map(PESO_IMPACTO).fillna(0.2)
    return df.rename(columns={"country": "pais", "impact": "impacto", "title": "evento",
                              "forecast": "prevision", "previous": "anterior"})[
        ["fecha", "pais", "impacto", "peso_impacto", "evento", "prevision", "anterior"]
    ].sort_values("fecha").reset_index(drop=True)


def riesgo_por_dia(monedas=None, cliente=None):
    """Riesgo macro agregado por dia y moneda: suma de pesos de impacto.

    Sirve como feature de contexto: un dia con varios eventos de alto impacto en
    la moneda del valor es un dia de mayor incertidumbre.
    """
    marcos = []
    for cual in ("semana_pasada", "esta_semana", "proxima_semana"):
        try:
            marcos.append(calendario(cual, cliente))
        except Exception:
            pass
    if not marcos:
        return pd.DataFrame(columns=["dia", "pais", "riesgo_macro"])
    cal = pd.concat(marcos, ignore_index=True)
    if monedas:
        cal = cal[cal["pais"].isin(monedas)]
    cal["dia"] = cal["fecha"].dt.normalize()
    return (cal.groupby(["dia", "pais"])["peso_impacto"].sum()
            .reset_index().rename(columns={"peso_impacto": "riesgo_macro"}))
