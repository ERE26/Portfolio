"""Estrategia cross-sectional momentum (long-short), un algoritmo quant clasico.

A diferencia del sistema de patrones (que mira el grafico de UN valor y decide
cuando operar), esta estrategia es TRANSVERSAL: cada mes compara TODOS los valores
entre si, se pone largo en los de mas momento y corto en los de menos, y se
mantiene neutral a mercado. Es el momentum de Jegadeesh-Titman (1993), base de los
fondos cuantitativos (AQR, etc.).

Se construye solo con precios. Nunca mira al futuro: en cada rebalanceo el score
usa datos HASTA esa fecha y el retorno se realiza DESPUES. El coste incluye la
rotacion (turnover) de la cartera.
"""
import numpy as np
import pandas as pd


def panel_precios(precios):
    """DataFrame (fechas x tickers) de cierres, alineado y rellenado con huecos cortos."""
    panel = pd.DataFrame({t: df["close"].astype(float) for t, df in precios.items()})
    return panel.sort_index().ffill(limit=5)


def score_momentum(panel_hasta, lookback=252, skip=21):
    """Momento 12-1 por ticker al final de 'panel_hasta' (log del precio hace
    'skip' sesiones frente a hace 'lookback'). Salta el ultimo mes (reversion)."""
    if len(panel_hasta) < lookback + skip + 1:
        return pd.Series(dtype=float)
    reciente = panel_hasta.iloc[-1 - skip]
    lejano = panel_hasta.iloc[-1 - lookback]
    return np.log(reciente / lejano)


def backtest(precios, lookback=252, skip=21, q=0.2, largo_corto=True, coste_bps=10.0,
             vol_objetivo=None, freq="ME", min_tickers=20):
    """Backtest mensual de la estrategia. Devuelve (retornos mensuales, metricas, pesos).

    q: fraccion de cada extremo (0.2 = quintil superior largo, inferior corto).
    vol_objetivo: si se da, escala la exposicion a esa volatilidad anual (vol targeting).
    """
    panel = panel_precios(precios)
    # ultimo dia de bolsa REAL de cada mes (no el fin de mes natural, que puede caer
    # en fin de semana y no existir en el indice)
    rebal = list(pd.Series(panel.index, index=panel.index).resample(freq).last().dropna())
    filas, w_prev = [], pd.Series(0.0, index=panel.columns)
    for i in range(len(rebal) - 1):
        t0, t1 = rebal[i], rebal[i + 1]
        p_hasta = panel.loc[:t0]
        mom = score_momentum(p_hasta, lookback, skip).dropna()
        # solo tickers con precio valido en t0 y t1 (sin look-ahead de supervivencia)
        ok = panel.loc[t0].reindex(mom.index).notna() & panel.loc[t1].reindex(mom.index).notna()
        mom = mom[ok.reindex(mom.index).fillna(False)]
        if len(mom) < min_tickers:
            continue
        n = max(int(len(mom) * q), 1)
        orden = mom.sort_values(ascending=False)
        largos = orden.index[:n]
        w = pd.Series(0.0, index=panel.columns)
        w[largos] = 1.0 / n
        if largo_corto:
            cortos = orden.index[-n:]
            w[cortos] = -1.0 / n
        r_ind = np.log(panel.loc[t1] / panel.loc[t0]).reindex(w.index).fillna(0.0)
        ret_bruto = float((w * r_ind).sum())
        turnover = float((w - w_prev).abs().sum())
        filas.append({"fecha": t1, "ret_bruto": ret_bruto,
                      "ret": ret_bruto - coste_bps / 1e4 * turnover, "turnover": turnover})
        w_prev = w
    if len(filas) < 12:
        return pd.DataFrame(filas), {}, w_prev
    d = pd.DataFrame(filas).set_index("fecha")
    r = d["ret"].to_numpy()
    if vol_objetivo:
        # vol targeting ex-ante: escala por la vol trailing de la propia estrategia
        vol_tr = d["ret"].rolling(12, min_periods=6).std().shift(1) * np.sqrt(12)
        escala = (vol_objetivo / vol_tr).clip(0.2, 3.0).fillna(1.0).to_numpy()
        r = r * escala
        d["ret"] = r
    met = _metricas(r, d)
    return d, met, w_prev


def _metricas(r, d):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    mu, sd = r.mean(), r.std(ddof=1)
    eq = np.cumsum(r)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    pos = r[r > 0].sum(); neg = -r[r < 0].sum()
    return {
        "meses": int(len(r)),
        "ret_anual": float(mu * 12),
        "vol_anual": float(sd * np.sqrt(12)),
        "sharpe": float(mu / sd * np.sqrt(12)) if sd > 0 else float("nan"),
        "max_drawdown": dd,
        "profit_factor": float(pos / max(neg, 1e-9)),
        "pct_meses_positivos": float((r > 0).mean()),
        "turnover_medio": float(d["turnover"].mean()),
    }


def serie_para_combinar(precios, **kw):
    """Retornos mensuales de la estrategia, indexados por fecha, para combinarla
    con otra alfa (p.ej. el sistema de patrones) a nivel de cartera."""
    d, _, _ = backtest(precios, **kw)
    return d["ret"] if len(d) else pd.Series(dtype=float)
