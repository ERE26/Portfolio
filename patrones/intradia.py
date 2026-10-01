"""Indicadores especificos de intradia, calculables con OHLCV y marca de tiempo.

El intradia es un regimen distinto del diario: domina la reversion a la media
(no el momentum), importan la hora del dia, el VWAP (el precio de referencia
institucional) y la posicion dentro del rango de la sesion. Estos indicadores no
tienen sentido en diario y son la base para operar intradia.

Todos usan solo el pasado de la sesion en curso (nunca el futuro). La sesion se
identifica por la fecha (dia natural) de cada barra.
"""
import numpy as np
import pandas as pd


def es_intradia(index):
    """True si el indice tiene varias barras por dia (temporalidad intradia)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return False
    por_dia = pd.Series(1, index=idx).groupby(idx.normalize()).sum()
    return por_dia.median() > 1.5


def rasgos_intradia(df):
    """DataFrame de features intradia alineado al indice de df (OHLCV con hora).

    Columnas:
      intra_dist_vwap   distancia relativa del cierre al VWAP de la sesion
      intra_pos_rango   posicion en el rango [min, max] de la sesion hasta ahora
      intra_hora_sin    hora del dia codificada ciclicamente (seno)
      intra_hora_cos    hora del dia codificada ciclicamente (coseno)
      intra_vol_rel     volumen de la barra vs volumen medio a esa hora del dia
      intra_gap         gap de apertura de la sesion (primera barra vs cierre previo)
      intra_ret_sesion  retorno acumulado de la sesion hasta la barra actual
    """
    idx = pd.DatetimeIndex(df.index)
    dia = idx.normalize()
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    tp = (h + l + c) / 3.0                      # precio tipico por barra
    out = pd.DataFrame(index=df.index)

    g = pd.Series(np.arange(len(df)), index=df.index).groupby(dia)
    # VWAP acumulado dentro de la sesion (solo pasado de la sesion)
    pv = pd.Series(tp * v, index=df.index)
    cum_pv = pv.groupby(dia).cumsum()
    cum_v = pd.Series(v, index=df.index).groupby(dia).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    out["intra_dist_vwap"] = (c - vwap) / vwap.replace(0, np.nan)

    # posicion en el rango de la sesion hasta ahora
    hi_dia = pd.Series(h, index=df.index).groupby(dia).cummax()
    lo_dia = pd.Series(l, index=df.index).groupby(dia).cummin()
    rango = (hi_dia - lo_dia).replace(0, np.nan)
    out["intra_pos_rango"] = (c - lo_dia) / rango

    # hora del dia, codificada ciclicamente (minuto del dia / jornada)
    minuto = idx.hour * 60 + idx.minute
    frac = (minuto - minuto.min()) / max(minuto.max() - minuto.min(), 1)
    out["intra_hora_sin"] = np.sin(2 * np.pi * frac)
    out["intra_hora_cos"] = np.cos(2 * np.pi * frac)

    # volumen relativo a esa hora del dia (media movil por hora, solo pasado)
    vser = pd.Series(v, index=df.index)
    media_hora = vser.groupby(idx.hour).transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        out["intra_vol_rel"] = np.log(vser / media_hora.replace(0, np.nan))

    # gap de apertura: primera barra de la sesion vs ultimo cierre de la anterior
    primera = g.transform("min")
    es_primera = pd.Series(np.arange(len(df)), index=df.index) == primera
    cierre_prev = pd.Series(c, index=df.index).groupby(dia).last().shift(1)
    cierre_prev = dia.map(cierre_prev)
    gap = np.where(es_primera, (o - cierre_prev) / cierre_prev, np.nan)
    out["intra_gap"] = pd.Series(gap, index=df.index).groupby(dia).transform("first")

    # retorno acumulado de la sesion
    apertura = pd.Series(o, index=df.index).groupby(dia).transform("first")
    out["intra_ret_sesion"] = c / apertura - 1.0

    # volumen relativo (RVOL): interes anormal a esta hora -> filtro clave intradia
    out["intra_rvol"] = np.log(rvol(df).clip(0.1, 10))

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


NOMBRES_INTRA = ["intra_dist_vwap", "intra_pos_rango", "intra_hora_sin", "intra_hora_cos",
                 "intra_vol_rel", "intra_gap", "intra_ret_sesion", "intra_rvol"]


def spread_corwin_schultz(df, ventana=20):
    """Estimador del bid-ask spread solo con maximos y minimos (Corwin-Schultz 2012).

    Permite meter el coste REAL de operar en el backtest sin datos de libro de
    ordenes. Devuelve el spread estimado (fraccion del precio) suavizado en una
    ventana. En intradia el coste decide la rentabilidad, asi que esta estimacion
    por activo es esencial.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.log(df["high"].clip(lower=1e-9).to_numpy(float))
        l = np.log(df["low"].clip(lower=1e-9).to_numpy(float))
    hl = (h - l) ** 2
    beta = pd.Series(hl, index=df.index).rolling(2).sum().to_numpy()  # barra t y t-1
    h2 = np.maximum(h[1:], h[:-1])
    l2 = np.minimum(l[1:], l[:-1])
    gamma = np.concatenate([[np.nan], (h2 - l2) ** 2])
    den = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = pd.Series(s, index=df.index).clip(lower=0)      # el spread no es negativo
    return s.rolling(ventana, min_periods=5).median().fillna(s.median())


def barras_hasta_cierre(index):
    """Numero de barras que quedan hasta la ultima barra de la sesion (dia natural).

    0 = esta barra ES el cierre de la sesion. Es lo que permite forzar la salida
    al cierre (EOD): una operacion intradia no puede sostenerse mas alla del final
    del dia, asi que el horizonte efectivo de cada caso se recorta a este numero.
    Se calcula de los datos reales, asi que respeta medias sesiones (cierres
    anticipados) sin tener que asumir una rejilla horaria fija.
    """
    idx = pd.DatetimeIndex(index)
    dia = idx.normalize()
    s = pd.Series(np.arange(len(idx)), index=range(len(idx)))
    g = s.groupby(pd.Series(dia, index=s.index))
    pos_en_dia = g.cumcount().to_numpy()
    n_dia = g.transform("size").to_numpy()
    return (n_dia - 1 - pos_en_dia).astype(int)


def rvol(df, lookback=20):
    """Relative Volume: volumen acumulado del dia hasta la barra t frente a la media
    del volumen acumulado a la MISMA hora en los ultimos 'lookback' dias.

    RVOL alto = interes anormal ahora (noticia/evento). Es el filtro que hace
    funcionar los breakouts y los gaps; sin el, el edge intradia se diluye.
    """
    idx = pd.DatetimeIndex(df.index)
    dia = idx.normalize()
    cumv = pd.Series(df["volume"].to_numpy(float), index=df.index).groupby(dia).cumsum()
    # media del cumv a la misma hora del dia, solo con dias anteriores
    hora = idx.hour * 100 + idx.minute
    out = cumv.groupby(hora).transform(
        lambda s: s.shift(1).rolling(lookback, min_periods=3).mean())
    return (cumv / out.replace(0, np.nan)).fillna(1.0).clip(0, 10)
