"""Asignador de capital: combina varias alfas en una sola cartera, walk-forward.

Tener dos algoritmos descorrelados (patrones + momentum) solo sirve si se COMBINAN
bien. Este modulo reparte el capital entre ellos mes a mes usando SOLO informacion
pasada (volatilidad y correlacion recientes), nunca el futuro. Es el paso que
convierte varias alfas en una estrategia de cartera.

  - Cada alfa se lleva a riesgo comparable dividiendo por su volatilidad trailing
    (vol-targeting por alfa), para que ninguna domine por tener mas escala.
  - Los pesos salen de risk parity / inverse-vol sobre la ventana reciente.
  - El retorno combinado de cada mes usa los pesos fijados con datos hasta el mes
    anterior: sin look-ahead.
"""
import numpy as np
import pandas as pd


def combinar(series, ventana=18, metodo="inv_vol", min_obs=6):
    """Combina un dict {nombre: serie_mensual} en una cartera walk-forward.

    metodo: 'inv_vol' (risk parity naive), 'riesgo' (HRP/covarianza shrunk) o
    'igual' (50/50 a riesgo igual). Devuelve (serie combinada, DataFrame de pesos).
    """
    df = pd.DataFrame(series).dropna()
    cols = list(df.columns)
    if len(df) < min_obs + 2 or len(cols) < 2:
        return pd.Series(dtype=float), pd.DataFrame()
    ret_comb, pesos_hist, idx = [], [], []
    for i in range(min_obs, len(df)):
        tr = df.iloc[max(0, i - ventana):i]          # historia hasta t-1
        vol = tr.std().replace(0, np.nan)
        escala = (1.0 / vol).fillna(0.0)             # lleva cada alfa a ~unidad de riesgo
        sr = tr * escala                             # alfas re-escaladas (historia)
        if metodo == "igual":
            w = pd.Series(1.0 / len(cols), index=cols)
        elif metodo == "riesgo":
            from .cartera import pesos_riesgo
            cov = np.cov(sr.dropna().to_numpy(), rowvar=False)
            w = pd.Series(pesos_riesgo(np.atleast_2d(cov), "hrp"), index=cols)
        else:  # inv_vol sobre las alfas re-escaladas (que ya tienen vol ~1 -> ~igual)
            v2 = sr.std().replace(0, np.nan)
            iw = (1.0 / v2).fillna(0.0)
            w = iw / iw.sum()
        # retorno del mes i con las alfas re-escaladas (riesgo comparable) y los pesos
        r_i = float((df.iloc[i] * escala * w).sum())
        ret_comb.append(r_i); pesos_hist.append(w); idx.append(df.index[i])
    serie = pd.Series(ret_comb, index=idx)
    pesos = pd.DataFrame(pesos_hist, index=idx)
    return serie, pesos


def metricas(serie, periodos=12):
    r = np.asarray(serie, float); r = r[np.isfinite(r)]
    if len(r) < 6 or r.std(ddof=1) == 0:
        return {}
    eq = np.cumsum(r)
    return {"meses": int(len(r)),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(periodos)),
            "max_drawdown": float((eq - np.maximum.accumulate(eq)).min()),
            "pct_positivos": float((r > 0).mean())}
