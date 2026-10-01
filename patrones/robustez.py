"""Metricas anti-sobreajuste: Deflated Sharpe Ratio y Probability of Backtest
Overfitting. Son el reality-check: con muchas pruebas (ratios de barrera,
umbrales, 100+ tickers...), el mejor Sharpe esta casi seguro inflado por azar.

  - Deflated Sharpe Ratio (DSR): probabilidad de que el Sharpe verdadero sea > 0
    teniendo en cuenta el numero de pruebas, la asimetria y la curtosis de los
    retornos y el tamano de la muestra. Bailey & Lopez de Prado (2014).
  - Probability of Backtest Overfitting (PBO): via CSCV, probabilidad de que la
    configuracion elegida como mejor en in-sample quede por debajo de la mediana
    out-of-sample. Bailey, Borwein, Lopez de Prado & Zhu (2017).

Ninguna arregla un sistema; sirven para creerse (o no) una mejora.
"""
import itertools

import numpy as np
from scipy.stats import norm

EULER = 0.5772156649015329


def sharpe(r, periodos=1):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periodos))


def sharpe_no_anualizado(r):
    """Sharpe por periodo (sin anualizar), que es el que usan las formulas DSR."""
    return sharpe(r, periodos=1)


def sr_esperado_max(sr_trials, n_trials=None):
    """Sharpe maximo esperado por PURO AZAR entre N pruebas independientes.

    sr0 = sqrt(Var(SRs)) * [ (1-g)*Z(1-1/N) + g*Z(1-1/(N*e)) ]  (Bailey-LdP).
    sr_trials: Sharpes (por periodo) de todas las pruebas que se hicieron.
    """
    sr = np.asarray(sr_trials, float)
    sr = sr[np.isfinite(sr)]
    N = int(n_trials or len(sr))
    if N < 2 or len(sr) < 2:
        return 0.0
    var = sr.var(ddof=1)
    z1 = norm.ppf(1 - 1.0 / N)
    z2 = norm.ppf(1 - 1.0 / (N * np.e))
    return float(np.sqrt(var) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe_ratio(retornos, sr_trials, sr_benchmark=None):
    """Deflated Sharpe Ratio en [0,1]: prob. de que el Sharpe real supere el
    umbral esperado por azar. >0.95 (o 0.90) para creerselo.

    retornos: serie de retornos por operacion/periodo de la estrategia ELEGIDA.
    sr_trials: Sharpes (por periodo, no anualizados) de TODAS las pruebas hechas.
    sr_benchmark: umbral; por defecto el Sharpe maximo esperado por azar.
    """
    from scipy.stats import kurtosis, skew
    r = np.asarray(retornos, float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 10:
        return float("nan")
    sr_hat = sharpe_no_anualizado(r)
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))      # curtosis normal = 3
    sr0 = sr_esperado_max(sr_trials) if sr_benchmark is None else float(sr_benchmark)
    denom = np.sqrt(max(1 - g3 * sr_hat + (g4 - 1) / 4.0 * sr_hat ** 2, 1e-12))
    z = (sr_hat - sr0) * np.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def probabilidad_sobreajuste(M, S=10):
    """Probability of Backtest Overfitting via CSCV.

    M: matriz (T periodos x N configuraciones) de retornos por periodo. Cada
    columna es una variante de la estrategia. Devuelve PBO en [0,1]: fraccion de
    particiones en las que la mejor configuracion in-sample queda por debajo de la
    mediana out-of-sample. >0.5 = probablemente sobreajustado.
    """
    M = np.asarray(M, float)
    T, N = M.shape
    if N < 2 or T < 2 * S:
        return float("nan")
    S = S - (S % 2)                      # S par
    cortes = np.array_split(np.arange(T), S)
    bloques = [M[idx] for idx in cortes]
    logits = []
    for comb in itertools.combinations(range(S), S // 2):
        is_idx = set(comb)
        J_is = np.vstack([bloques[b] for b in range(S) if b in is_idx])
        J_oos = np.vstack([bloques[b] for b in range(S) if b not in is_idx])
        sr_is = np.array([sharpe_no_anualizado(J_is[:, n]) for n in range(N)])
        sr_oos = np.array([sharpe_no_anualizado(J_oos[:, n]) for n in range(N)])
        if not np.isfinite(sr_is).any():
            continue
        n_star = int(np.nanargmax(sr_is))
        # rango relativo OOS de la mejor IS (1 = mejor, 0 = peor)
        orden = np.argsort(np.argsort(np.nan_to_num(sr_oos, nan=-np.inf)))
        w = (orden[n_star] + 1) / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    if not logits:
        return float("nan")
    logits = np.array(logits)
    return float((logits <= 0).mean())


def matriz_variantes_por_periodo(res, aplicar_fn, variantes, freq="ME"):
    """Construye la matriz T x N para el PBO: retorno por periodo (mes) de cada
    variante de reglas. aplicar_fn(res, reglas) -> (sel con 'ret_op' y 'fecha').
    variantes: lista de dicts de reglas. Devuelve (M, nombres)."""
    import pandas as pd
    cols, nombres = [], []
    base = None
    for i, reglas in enumerate(variantes):
        sel, _ = aplicar_fn(res, reglas=reglas)
        if "ret_op" not in sel or len(sel) < 10:
            continue
        s = sel.copy()
        s["fecha"] = pd.to_datetime(s["fecha"])
        serie = s.set_index("fecha")["ret_op"].resample(freq).mean()
        cols.append(serie)
        nombres.append(f"v{i}")
    if len(cols) < 2:
        return np.empty((0, 0)), []
    import pandas as pd
    M = pd.concat(cols, axis=1).fillna(0.0)
    return M.to_numpy(), nombres
