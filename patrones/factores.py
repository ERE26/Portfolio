"""Normalizacion cross-sectional de los factores (estilo AQR / Fama-French).

Un factor crudo (p.ej. E/P = 2.6%) no es comparable entre acciones ni entre
epocas: un E/P del 2% significa cosas distintas segun el nivel general de
valoracion. La solucion estandar es normalizar EN SECCION CRUZADA: en cada fecha,
sobre todas las acciones, se convierte el factor a rango y luego a z-score. Asi:

  - factores de unidades distintas quedan en una escala comun (media 0, desv 1);
  - se elimina el nivel y la dispersion que cambian con el regimen;
  - el rango-luego-z (metodo de Quality-Minus-Junk) es robusto a outliers y a
    relaciones no lineales, critico con fundamentales ruidosos.

No hay fuga temporal: la normalizacion de una fecha usa solo valores de esa
misma fecha (todos ya point-in-time).
"""
import numpy as np
import pandas as pd

from .ventanas import GRUPOS_ESCALARES


def normalizar_cross_sectional(banco, min_por_fecha=8):
    """Normaliza en seccion cruzada las columnas de factores del banco, in place.

    Para cada fecha con al menos 'min_por_fecha' acciones, cada factor fac_* se
    transforma a z-score de su rango entre las acciones de esa fecha. Las fechas
    con pocas acciones o los factores ausentes quedan en 0 (neutro).
    """
    cols = [c for c in GRUPOS_ESCALARES["factores"] if c.startswith("fac_")]
    idx_cols = [banco.nombres_feat.index(c) for c in cols]
    feat = banco.feat
    # agrupar por MES: con submuestreo las fechas exactas no se alinean entre
    # acciones, pero en un mes hay muchas acciones comparables en seccion cruzada.
    periodo = pd.PeriodIndex(banco.meta["fecha"], freq="M").astype(str).to_numpy()
    orden = np.argsort(periodo, kind="stable")
    p_orden = periodo[orden]
    ini = 0
    for j in range(1, len(orden) + 1):
        if j == len(orden) or p_orden[j] != p_orden[ini]:
            grupo = orden[ini:j]
            if len(grupo) >= min_por_fecha:
                for ci in idx_cols:
                    vals = feat[grupo, ci]
                    fin = np.isfinite(vals)
                    if fin.sum() >= min_por_fecha:
                        z = np.zeros(len(grupo))
                        r = pd.Series(vals[fin]).rank().to_numpy()
                        r = (r - r.mean()) / (r.std() + 1e-9)
                        z[fin] = np.clip(r, -3, 3)
                        feat[grupo, ci] = z
                    else:
                        feat[grupo, ci] = 0.0
            else:
                feat[np.ix_(grupo, idx_cols)] = 0.0
            ini = j
    # cualquier NaN residual (fechas sueltas) a 0
    feat[:, idx_cols] = np.nan_to_num(feat[:, idx_cols], nan=0.0)
    return banco
