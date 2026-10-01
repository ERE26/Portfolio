"""Modelo de costes de transaccion realistas (spread + comision + slippage).

El backtest base usa un coste PLANO en unidades de ATR (p.ej. 0.05). Es una
aproximacion pobre y puede enganar sobre donde esta el edge neto. La razon: los
costes reales se pagan en PRECIO (puntos basicos), asi que al expresarlos en
unidades de RIESGO (ATR, que es como se mide el retorno del sistema) escalan
INVERSAMENTE con la volatilidad del valor.

    coste_atr (ida y vuelta) = (spread + comision) / atr_pct + slippage_atr

  - spread:   fraccion del precio por cruzar el diferencial (ida + vuelta).
  - comision: fraccion del precio (p.ej. 2 * $0.0035/accion / precio).
  - slippage_atr: impacto por agresividad/participacion, ya en unidades de ATR.

Consecuencia practica (y por que importa): operar un valor de atr_pct=0.8%
cuesta, en riesgo, mas del doble que uno de atr_pct=2.0%. Un coste plano
infravalora el trading de baja volatilidad y sobreestima el de alta. Modelar
esto bien EMPUJA la seleccion hacia los "stocks in play" (alta volatilidad,
catalizador) y penaliza el churn de baja volatilidad: exactamente el principio
calidad > cantidad, pero demostrado con numeros, no por intuicion.

Referencias: la evidencia robusta (Barber & Odean 2000; Lopez de Prado) es que la
mayoria de los "edges" de backtest se evaporan al meter costes honestos; ver
docs/aprendizaje_quant.md.
"""
import numpy as np

# Defaults para acciones US liquidas (large caps). Conservadores pero realistas.
SPREAD_BPS = 4.0            # diferencial ida+vuelta en puntos basicos del precio
COMISION_ACCION = 0.0035    # $/accion por lado (tarifa tipica de broker)
SLIPPAGE_ATR = 0.02         # impacto/slippage como fraccion del ATR (estrategia selectiva)
SUELO_ATR_PCT = 0.002       # suelo de atr_pct (0.2%) para no dividir por ~0


def coste_atr(atr_pct, precio=None, spread_bps=SPREAD_BPS,
              comision_accion=COMISION_ACCION, comision_bps=None,
              slippage_atr=SLIPPAGE_ATR, suelo_atr_pct=SUELO_ATR_PCT):
    """Coste de ida y vuelta de UNA operacion, en unidades de ATR.

    atr_pct: ATR como fraccion del precio (p.ej. 0.015 = 1.5%). Escalar o array.
    precio:  para la comision por accion; si None, se asume ~$100 o se usa
             comision_bps si se pasa.
    Devuelve el coste en ATR (mismo escalar/forma que atr_pct)."""
    ap = np.maximum(np.asarray(atr_pct, dtype=float), suelo_atr_pct)
    spread_frac = spread_bps / 1e4                      # ida+vuelta, fraccion del precio
    if comision_bps is not None:
        com_frac = comision_bps / 1e4
    elif precio is not None:
        com_frac = 2.0 * comision_accion / np.maximum(np.asarray(precio, float), 1e-6)
    else:
        com_frac = 2.0 * comision_accion / 100.0        # asume ~$100 si no hay precio
    coste_precio = spread_frac + com_frac                # fraccion del precio, ida+vuelta
    return coste_precio / ap + slippage_atr


def coste_retorno(atr_pct, precio=None, **kw):
    """Coste de ida y vuelta en unidades de RETORNO (fraccion del precio), para
    restarlo a retornos en precio (p.ej. la etiqueta triple-barrera tb_ret).

    Es el coste en ATR multiplicado por atr_pct:
        coste_retorno = (spread + comision) + slippage_atr * atr_pct
    A diferencia de un coste plano en bps, el termino de slippage escala con la
    volatilidad del valor (mover un valor volatil cuesta mas), y el spread+comision
    es fijo en precio. Asi el coste depende del valor, no es una constante."""
    ap = np.asarray(atr_pct, dtype=float)
    return coste_atr(ap, precio, **kw) * np.maximum(ap, kw.get("suelo_atr_pct", SUELO_ATR_PCT))


def coste_por_fila(df, atr_col="atr_pct", precio_col="precio", defecto_plano=0.05, **kw):
    """Coste realista por fila de un DataFrame de operaciones (array alineado al df).

    Usa atr_pct (y precio si esta) de cada fila. Si no hay columna de volatilidad,
    cae al coste plano por defecto (no se puede afinar sin volatilidad)."""
    import pandas as pd
    if atr_col not in df or not pd.Series(df[atr_col]).notna().any():
        return np.full(len(df), float(defecto_plano))
    atr_pct = pd.Series(df[atr_col], dtype=float).to_numpy()
    precio = pd.Series(df[precio_col], dtype=float).to_numpy() if precio_col in df else None
    c = coste_atr(atr_pct, precio, **kw)
    # filas sin atr_pct -> coste plano (no castigar de mas por falta de dato)
    return np.where(np.isfinite(atr_pct), c, float(defecto_plano))
