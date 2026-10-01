"""Dimensionado de posiciones: cuanto arriesgar en cada operacion.

Hasta aqui el sistema decidia QUE operar (direccion + seleccion del meta) y media
cada operacion a tamano fijo. El sizing traduce la CONFIANZA calibrada y el riesgo
en un tamano, que es de lo que mas depende el crecimiento a largo plazo.

  - Kelly fraccional: f* = p - (1-p)/b maximiza el crecimiento logaritmico, con p
    la probabilidad (calibrada) de acierto y b la ratio recompensa/riesgo
    (objetivo/stop). Full Kelly es ruinoso con p mal estimada, asi que se usa una
    fraccion (0.2-0.3). Kelly (1956); Thorp (2006); MacLean-Thorp-Ziemba (2011).
  - Volatility targeting: escalar la posicion inverso a la volatilidad para que
    cada operacion aporte un riesgo parecido. Harvey et al. (2018).

El Sharpe y el profit factor son invariantes a la escala global, asi que lo que
cambia el resultado es el peso RELATIVO entre operaciones (mas tamano donde hay
mas convicción), no la fraccion de Kelly en si.
"""
import numpy as np


def kelly_fraccional(p, b, fraccion=0.25, tope=1.0):
    """Fraccion de Kelly para una apuesta binaria, recortada a [0, tope].

    p: probabilidad (calibrada) de acierto. b: ratio recompensa/riesgo (objetivo/stop).
    f* = p - (1-p)/b; se multiplica por 'fraccion' (fractional Kelly) por seguridad.
    Donde f* <= 0 (sin ventaja) el tamano es 0: no operar.
    """
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    f = p - (1.0 - p) / max(float(b), 1e-9)
    return np.clip(f * float(fraccion), 0.0, float(tope))


def escala_vol_objetivo(atr_pct, objetivo, limites=(0.25, 4.0)):
    """Escalar inverso a la volatilidad (ATR%): menos tamano en lo mas volatil.

    objetivo: ATR% de referencia (p.ej. la mediana). Se recorta el apalancamiento
    a 'limites' para no tomar posiciones enormes en activos muy tranquilos.
    """
    atr = np.asarray(atr_pct, float)
    s = objetivo / np.where(atr > 1e-6, atr, np.nan)
    return np.clip(np.nan_to_num(s, nan=1.0), *limites)


def pesos_posicion(sel, reglas, b):
    """Peso (tamano) de cada operacion seleccionada segun el modo de sizing.

    modos: 'igual' (tamano fijo, por defecto), 'confianza' (lineal en el meta),
    'kelly' (Kelly fraccional sobre la prob. calibrada). Opcionalmente se multiplica
    por el escalado de volatilidad si reglas['vol_objetivo'] esta puesto.
    """
    modo = reglas.get("sizing", "igual")
    n = len(sel)
    if modo == "confianza" and "prob_meta" in sel:
        w = np.clip((sel["prob_meta"].to_numpy(float) - 0.5) * 2.0, 0.0, 1.0)
    elif modo == "kelly" and "prob_meta" in sel:
        w = kelly_fraccional(sel["prob_meta"].to_numpy(float), b,
                             reglas.get("kelly_fraccion", 0.25))
    else:
        w = np.ones(n)
    if reglas.get("vol_objetivo") and "atr_pct" in sel:
        w = w * escala_vol_objetivo(sel["atr_pct"].to_numpy(float), reglas["vol_objetivo"])
    return w
