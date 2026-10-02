"""Fuente UNICA de verdad del PRESUPUESTO DE CAPITAL por libro (el 'SO del pod-shop').

El reparto sale del presupuesto de RIESGO entre libros (HRP sobre su historia de
retornos, via riesgo.presupuesto_riesgo). Hoy es un PRIOR de BACKTEST (2008-2026, ver
`presupuesto_cross_book.py`); cuando se acumule historia VIVA por libro (atribuida via
client_order_id) el mismo motor se re-alimenta con retornos REALES y actualiza los pesos.

Por que combinar los libros (meses comunes 2008-2026):
  corr B_swing<->D_rotacion ~ -0.02 (casi ortogonales: momentum US vs rotacion Europa)
  corr C_tsmom<->D_rotacion ~  0.41 ; corr B<->C ~ 0.25
  Combinar los 3 NO sube mucho el Sharpe (0.61 vs 0.59 del mejor en solitario) pero
  PARTE el drawdown (-14% combinado vs -28/-33% por libro) y baja la vol a ~9%. Ese es
  el premio real para un mandato de preservacion de capital, no mas retorno.

El intradia (Libro A) va como SLEEVE FIJO fuera del HRP (horizonte intradia, cash-heavy).

Honestidad: HRP reparte por RIESGO y diversificacion, NO por retorno; por eso carga al
swing (la pata de menor vol y mejor diversificadora) aunque su Sharpe en solitario sea el
mas flojo (0.34). Es deliberado: el objetivo es minimizar drawdown, no maximizar retorno.
"""
import numpy as np
import pandas as pd

# Sleeve fijo del intradia (fuera del HRP; horizonte intradia, cash-heavy).
SLEEVE_INTRADIA = 0.35

# PRIOR de backtest: pesos de RIESGO HRP entre los libros de horizonte largo
# (presupuesto_cross_book.py, 213 meses comunes 2008-2026). Suman 1.0 ENTRE ELLOS.
PRIOR_HRP = {"swing": 0.519, "tsmom": 0.255, "rotacion": 0.226}

# Fallback por libro si todo falla (valores historicos hardcodeados que sustituye).
FALLBACK_CAPITAL = {"intradia": 0.35, "swing": 0.33, "tsmom": 0.17, "rotacion": 0.15}

LIBROS_LARGOS = ("swing", "tsmom", "rotacion")


def _capital_desde_riesgo(w_riesgo):
    """Traduce pesos de RIESGO (entre libros de horizonte largo, suman 1) a fraccion de
    CAPITAL total, reservando SLEEVE_INTRADIA para el intradia."""
    s = sum(w_riesgo.get(k, 0.0) for k in LIBROS_LARGOS)
    if s <= 0:
        return dict(FALLBACK_CAPITAL)
    resto = 1 - SLEEVE_INTRADIA
    cap = {k: w_riesgo.get(k, 0.0) / s * resto for k in LIBROS_LARGOS}
    cap["intradia"] = SLEEVE_INTRADIA
    return cap


def objetivo(retornos_vivos=None, min_obs=24):
    """(capital_frac_por_libro, nota). Usa historia VIVA por libro si hay suficiente de
    los TRES libros largos (>=min_obs cada uno); si no, el PRIOR de backtest.
    Fail-safe: ante cualquier problema, cae al prior/fallback."""
    try:
        if retornos_vivos:
            series = {k: pd.Series(v).dropna() for k, v in retornos_vivos.items()}
            tienen = {k for k in LIBROS_LARGOS if len(series.get(k, [])) >= min_obs}
            if tienen >= set(LIBROS_LARGOS):
                from patrones import riesgo
                w, nota = riesgo.presupuesto_riesgo(
                    {k: series[k] for k in LIBROS_LARGOS}, metodo="hrp", min_obs=min_obs)
                if w and set(w) >= set(LIBROS_LARGOS):
                    return _capital_desde_riesgo(w), f"HRP vivo ({nota})"
    except Exception:
        pass
    return _capital_desde_riesgo(PRIOR_HRP), "prior de backtest (HRP 2008-2026)"


def frac_libro(libro, retornos_vivos=None):
    """Fraccion de capital objetivo para UN libro (con fallback seguro)."""
    cap, _ = objetivo(retornos_vivos)
    return float(cap.get(libro, FALLBACK_CAPITAL.get(libro, 0.15)))
