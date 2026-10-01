"""Time-Series Momentum (TSMOM / CTA): el motor de los fondos de futuros gestionados
(Man AHL, Winton, AQR Managed Futures; Moskowitz-Ooi-Pedersen 2012).

A diferencia del momentum TRANSVERSAL (patrones/transversal.py, que compara valores
entre si), el TSMOM mira CADA activo contra SU PROPIO pasado: largo si ha subido en
su horizonte, corto si ha bajado, con el tamano escalado por su volatilidad para
igualar la contribucion de riesgo. Se promedian varios horizontes.

Por que anadirlo: es un flujo de retorno DIVERGENTE (gana en crisis prolongadas,
'crisis alpha') y poco correlacionado con los libros long-biased (momentum
transversal + intradia). La leccion de los multi-manager no es una senal genial,
son varias senales descorrelacionadas con presupuesto de riesgo. Se aplica a una
CESTA CROSS-ASSET de ETFs liquidos (renta variable, bonos, oro, credito), no a
acciones sueltas (ahi es ruidoso).

Solo precios (OHLCV). Nunca mira el futuro: la senal en t usa datos hasta t y el
retorno se realiza despues. El coste incluye la rotacion.
"""
import numpy as np
import pandas as pd


def panel_precios(precios):
    panel = pd.DataFrame({t: df["close"].astype(float) for t, df in precios.items()})
    return panel.sort_index().ffill(limit=5)


def signal_tsmom(panel_hasta, horizontes=(21, 63, 126, 252)):
    """Señal por activo en [-1, 1]: media del SIGNO del retorno en cada horizonte.
    +1 = sube en todos los horizontes (largo pleno); -1 = baja en todos (corto)."""
    sig = {}
    for t in panel_hasta.columns:
        s = panel_hasta[t].dropna()
        if len(s) < max(horizontes) + 1:
            continue
        signos = [np.sign(s.iloc[-1] / s.iloc[-1 - h] - 1.0) for h in horizontes
                  if len(s) > h]
        if signos:
            sig[t] = float(np.mean(signos))
    return pd.Series(sig)


def backtest(precios, horizontes=(21, 63, 126, 252), vol_objetivo=0.10,
             vol_lookback=63, coste_bps=10.0, freq="ME", cap_peso=1.5, min_activos=3,
             vol_cartera=0.12):
    """Backtest del TSMOM sobre una cesta. Posicion = señal * (vol_objetivo /
    vol_realizada del activo), acotada a +/-cap_peso, y con un ESCALADO FINAL de la
    cartera a 'vol_cartera' (vol targeting ex-ante con la vol trailing de la propia
    estrategia; sin mirar el futuro). Sin esto el libro queda sobre-apalancado.
    Rebalanceo 'freq'. Devuelve (retornos por periodo, metricas)."""
    panel = panel_precios(precios)
    ret_d = panel.pct_change()
    vol_d = ret_d.rolling(vol_lookback, min_periods=vol_lookback // 2).std() * np.sqrt(252)
    rebal = list(pd.Series(panel.index, index=panel.index).resample(freq).last().dropna())
    rebal = [d for d in rebal if d in set(panel.index)]
    filas, w_prev = [], pd.Series(0.0, index=panel.columns)
    for i in range(len(rebal) - 1):
        t0, t1 = rebal[i], rebal[i + 1]
        sig = signal_tsmom(panel.loc[:t0], horizontes)
        if len(sig) < min_activos:
            continue
        vol = vol_d.loc[t0].reindex(sig.index)
        escala = (vol_objetivo / vol).clip(upper=cap_peso).fillna(0.0)
        w = (sig * escala).clip(-cap_peso, cap_peso)
        w = w.reindex(panel.columns).fillna(0.0)
        r_ind = (panel.loc[t1] / panel.loc[t0] - 1.0).reindex(w.index).fillna(0.0)
        ret_bruto = float((w * r_ind).sum())
        turnover = float((w - w_prev).abs().sum())
        filas.append({"fecha": t1, "ret": ret_bruto - coste_bps / 1e4 * turnover,
                      "ret_bruto": ret_bruto, "turnover": turnover,
                      "exposicion_bruta": float(w.abs().sum())})
        w_prev = w
    if len(filas) < 12:
        return pd.DataFrame(filas), {}
    d = pd.DataFrame(filas).set_index("fecha")
    per = 12 if freq in ("ME", "M") else 52
    if vol_cartera:
        # vol targeting ex-ante de la cartera: escala por la vol trailing de la
        # propia estrategia (desfasada, sin look-ahead). Lleva la vol al objetivo.
        vol_tr = d["ret"].rolling(12, min_periods=6).std().shift(1) * np.sqrt(per)
        escala = (vol_cartera / vol_tr).clip(0.1, 3.0).fillna(1.0)
        d["ret"] = d["ret"] * escala.to_numpy()
    r = d["ret"].to_numpy()
    mu, sd = r.mean(), r.std(ddof=1)
    eq = np.cumprod(1 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    met = {"periodos": len(r), "ret_anual": mu * per, "vol_anual": sd * np.sqrt(per),
           "sharpe": mu / sd * np.sqrt(per) if sd > 0 else float("nan"),
           "max_drawdown": dd, "pct_positivos": float((r > 0).mean()),
           "turnover_medio": float(d["turnover"].mean())}
    return d, met


def serie_para_combinar(precios, **kw):
    """Retornos por periodo del TSMOM, para combinarlo/correlacionarlo con otros libros."""
    d, _ = backtest(precios, **kw)
    return d["ret"] if len(d) else pd.Series(dtype=float)
