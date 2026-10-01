"""Reglas de SALIDA para el libro diario/swing. Cuando vender importa tanto como
cuando comprar.

El momentum transversal (patrones/transversal.py) decide QUE comprar (los de mas
fuerza relativa) y su unica salida implicita es "caer del quintil al rebalanceo
mensual". Eso deja dos agujeros caros:
  1. entre rebalanceos, una posicion puede desplomarse y la mantienes semanas;
  2. el momentum sufre CRASHES en los giros de mercado (Daniel-Moskowitz 2016):
     justo cuando el mercado rebota desde un suelo, los 'ganadores' (que eran
     defensivos) se hunden. Sin filtro de regimen, el momentum da sus peores
     rachas ahi.

Salidas ROBUSTAS (evidencia, no sobreajuste; ver docs/aprendizaje_quant.md):
  - Trailing stop tipo Chandelier (max de cierres - k*ATR): deja correr al
    ganador y protege lo ganado. Mejor que un objetivo fijo, que corta la cola
    derecha de la que vive el momentum.
  - Rotura de tendencia propia: cierre por debajo de su media larga (200d).
  - Filtro de REGIMEN: fuera (a liquidez) cuando el mercado esta por debajo de su
    media de 200 sesiones -> esquiva los crashes de momentum.
Evitamos las "mejoras" fragiles (breakeven agresivo, objetivos finos): suben el
% de acierto pero bajan la esperanza.
"""
import numpy as np
import pandas as pd

from . import indicadores as ind


def sma(close, n=200):
    return ind.sma(pd.Series(close, dtype=float), n)


def chandelier_stop(df, n_atr=3.0, lookback=63, atr_n=14):
    """Nivel de trailing stop: maximo de cierres de las ultimas 'lookback' sesiones
    menos n_atr veces el ATR. Sube con el precio, nunca da mas que n_atr*ATR."""
    c = df["close"].astype(float)
    atr = ind.atr(df["high"].astype(float), df["low"].astype(float), c, atr_n)
    techo = c.rolling(lookback, min_periods=max(5, lookback // 3)).max()
    return techo - n_atr * atr


def salida_chandelier(df, **kw):
    """True en las barras donde el cierre pierde el trailing stop (senal de salida)."""
    c = df["close"].astype(float)
    stop = chandelier_stop(df, **kw)
    return (c < stop) & stop.notna()


def salida_tendencia(close, n=200):
    """True cuando el cierre esta por debajo de su media larga (tendencia rota)."""
    c = pd.Series(close, dtype=float)
    return (c < sma(c, n)) & sma(c, n).notna()


def regimen_alcista(indice_close, n=200):
    """True cuando el indice esta por encima de su media de n sesiones (risk-on).
    Sirve de overlay: en regimen bajista se recorta el libro de momentum."""
    c = pd.Series(indice_close, dtype=float)
    return (c > sma(c, n)) & sma(c, n).notna()


def indice_equiponderado(panel):
    """Proxy de 'el mercado' si no hay un indice: media equiponderada normalizada
    de todos los valores del panel (base 1 al inicio)."""
    rel = panel.div(panel.iloc[0])
    return rel.mean(axis=1)


def estado_salida(df, indice_close=None, mom=None, n_tend=200, **kw_chand):
    """Diagnostico de salida de UNA posicion (para el reporte en vivo): nivel de
    trailing stop, si lo ha perdido, si ha roto tendencia, momento, y regimen."""
    c = df["close"].astype(float)
    stop = chandelier_stop(df, **kw_chand)
    sal_ch = bool(c.iloc[-1] < stop.iloc[-1]) if stop.notna().iloc[-1] else False
    sal_te = bool(salida_tendencia(c, n_tend).iloc[-1])
    out = {"precio": float(c.iloc[-1]),
           "trailing_stop": float(stop.iloc[-1]) if stop.notna().iloc[-1] else float("nan"),
           "pierde_trailing": sal_ch, "rompe_tendencia": sal_te,
           "momentum": float(mom) if mom is not None else float("nan")}
    if indice_close is not None:
        out["regimen_alcista"] = bool(regimen_alcista(indice_close).iloc[-1])
    return out
