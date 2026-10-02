"""Capa de REGIMEN / gestion de volatilidad: escala la exposicion segun el riesgo del
mercado. Dos motores validados (SPY 1993-2026, DSR 1.0):

  1. Gestion de volatilidad (Moreira-Muir 2017): exposicion = vol_objetivo /
     vol_realizada reciente (sin look-ahead). Sube en calma, baja en turbulencia.
     Solo: Sharpe 0.65 -> 0.76, maxDD -55% -> -44%.
  2. Filtro VIX por estructura temporal: risk-off cuando VIX > VIX3M (backwardation =
     estres) o VIX muy alto. Solo: Sharpe -> 0.78, maxDD -> -31%.
  Combinado: Sharpe 0.79, maxDD -22% (mas que halves el drawdown), CAGR ~igual.

El `multiplicador()` da un escalar de exposicion en [0, cap] que los libros usan para
dimensionar: en calma/contango ~1x (o algo mas), en estres recorta. No mira el futuro.
"""
import numpy as np
import pandas as pd

VOL_OBJ = 0.14
CAP = 1.5
PER = 252


def vol_managed_escala(precios, lb=21, vol_obj=VOL_OBJ, cap=CAP):
    """Escala de exposicion por gestion de volatilidad (ultimo valor): vol_obj /
    vol realizada de los ultimos `lb` dias (desfasada 1 dia, sin look-ahead)."""
    c = pd.Series(precios).astype(float)
    r = c.pct_change()
    rv = r.rolling(lb).std().shift(1).iloc[-1] * np.sqrt(PER)
    if not np.isfinite(rv) or rv <= 0:
        return 1.0, float("nan")
    return float(np.clip(vol_obj / rv, 0.0, cap)), float(rv)


def regimen_vix(vix=None, vix3m=None, cli=None):
    """Estado del VIX y su estructura temporal. ratio = VIX/VIX3M: >1 backwardation
    (estres, risk-off), <1 contango (normal, risk-on). Tambien marca VIX alto."""
    if vix is None or vix3m is None:
        from fuentes import yahoo
        from fuentes.http import Cliente
        c = cli or Cliente()
        desde = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        try:
            vix = float(yahoo.historico("^VIX", intervalo="1d", desde=desde, cliente=c)[0]["close"].iloc[-1])
            vix3m = float(yahoo.historico("^VIX3M", intervalo="1d", desde=desde, cliente=c)[0]["close"].iloc[-1])
        except Exception:
            return {"vix": float("nan"), "vix3m": float("nan"), "ratio": float("nan"),
                    "estado": "s/d", "risk_on": True}
    ratio = vix / vix3m if vix3m else float("nan")
    backwardation = np.isfinite(ratio) and ratio >= 1.0
    vix_alto = np.isfinite(vix) and vix >= 28.0
    risk_on = not (backwardation or vix_alto)
    return {"vix": float(vix), "vix3m": float(vix3m), "ratio": float(ratio),
            "estado": "backwardation" if backwardation else ("VIX alto" if vix_alto else "contango"),
            "risk_on": bool(risk_on)}


def multiplicador(precios_mercado=None, cli=None, vol_obj=VOL_OBJ, cap=CAP, piso_riskoff=0.5):
    """Multiplicador de exposicion combinado en [0, cap] para dimensionar los libros.
    = escala_vol_managed(mercado) * factor_vix. factor_vix = 1.0 en contango, `piso_riskoff`
    en estres (el backtest usaba 0 = efectivo total; en vivo se usa un piso para evitar
    whipsaw, configurable). precios_mercado: serie de SPY; si falta, se descarga."""
    if precios_mercado is None:
        from fuentes import yahoo
        from fuentes.http import Cliente
        c = cli or Cliente()
        desde = (pd.Timestamp.now() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        try:
            precios_mercado = yahoo.historico("SPY", intervalo="1d", desde=desde, cliente=c)[0]["close"]
        except Exception:
            return {"mult": 1.0, "vol_managed": 1.0, "vix": {}, "nota": "sin datos de mercado"}
    esc, rv = vol_managed_escala(precios_mercado, vol_obj=vol_obj, cap=cap)
    vx = regimen_vix(cli=cli)
    fvix = 1.0 if vx["risk_on"] else piso_riskoff
    mult = float(np.clip(esc * fvix, 0.0, cap))
    return {"mult": mult, "vol_managed": esc, "vol_realizada": rv, "factor_vix": fvix,
            "vix": vx, "nota": f"vol-managed {esc:.2f}x · VIX {vx['estado']} (×{fvix})"}
