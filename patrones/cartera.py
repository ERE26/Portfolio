"""Modelo de riesgo y construccion de cartera.

Hasta aqui cada operacion se media por separado y a tamano fijo. Pero cuando hay
varias posiciones ABIERTAS A LA VEZ, lo que importa es el riesgo del CONJUNTO: dos
tecnologicas muy correladas no diversifican. Este modulo pondera las operaciones
concurrentes por su riesgo, no por igual.

  - Covarianza shrunk (Ledoit-Wolf): la covarianza muestral es ruidosa y casi
    singular con muchos activos; se encoge hacia una estructura estable.
    Ledoit & Wolf (2004).
  - Hierarchical Risk Parity (HRP): reparte el riesgo por clusters sin invertir la
    covarianza (mas estable fuera de muestra que Markowitz). Lopez de Prado (2016).
  - Inverse-vol (risk parity naive): peso ~ 1/volatilidad. Simple y robusto.

El modelo de riesgo no sube el acierto; suaviza el equity y evita concentrar
riesgo. Se mide honestamente si mejora el Sharpe del conjunto.
"""
import numpy as np
import pandas as pd


def covarianza_shrunk(retornos):
    """Covarianza con shrinkage de Ledoit-Wolf (robusta con pocas observaciones)."""
    from sklearn.covariance import LedoitWolf
    X = np.asarray(retornos, float)
    if X.shape[0] < 5 or X.shape[1] < 2:
        return np.cov(X, rowvar=False) if X.shape[1] > 1 else np.array([[np.var(X) + 1e-9]])
    return LedoitWolf().fit(X).covariance_


def pesos_inv_vol(cov):
    """Peso proporcional a 1/volatilidad (risk parity naive), normalizado a 1."""
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    w = 1.0 / vol
    return w / w.sum()


def _ivp(cov, idx):
    """Inverse-variance portfolio de un subconjunto (para HRP)."""
    sub = cov[np.ix_(idx, idx)]
    iv = 1.0 / np.clip(np.diag(sub), 1e-12, None)
    return iv / iv.sum()


def pesos_hrp(cov):
    """Hierarchical Risk Parity (Lopez de Prado 2016): cluster + biseccion
    recursiva repartiendo varianza inversa. No invierte la covarianza."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    # quasi-diagonalizacion: orden de las hojas del dendrograma
    orden = _orden_cluster(link, n)
    # biseccion recursiva
    w = pd.Series(1.0, index=orden)
    clusters = [orden]
    while clusters:
        nuevos = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mitad = len(c) // 2
            izq, der = c[:mitad], c[mitad:]
            var_izq = _var_cluster(cov, izq)
            var_der = _var_cluster(cov, der)
            alfa = 1 - var_izq / (var_izq + var_der)
            for i in izq:
                w[i] *= alfa
            for i in der:
                w[i] *= (1 - alfa)
            nuevos += [izq, der]
        clusters = nuevos
    return w.reindex(range(n)).to_numpy()


def _var_cluster(cov, idx):
    w = _ivp(cov, idx)
    sub = cov[np.ix_(idx, idx)]
    return float(w @ sub @ w)


def _orden_cluster(link, n):
    """Orden de las hojas tras el clustering jerarquico (quasi-diagonal)."""
    link = link.astype(int)
    orden = [link[-1, 0], link[-1, 1]]
    while max(orden) >= n:
        nuevo = []
        for v in orden:
            if v < n:
                nuevo.append(v)
            else:
                nuevo += [link[v - n, 0], link[v - n, 1]]
        orden = nuevo
    return orden


def pesos_riesgo(cov, metodo="hrp"):
    if metodo == "hrp":
        return pesos_hrp(cov)
    if metodo == "inv_vol":
        return pesos_inv_vol(cov)
    n = cov.shape[0]
    return np.full(n, 1.0 / n)        # igual


def backtest_cartera(res_sel, precios, metodo="hrp", freq="W", ventana=120,
                     col_ret="ret_op"):
    """Agrupa las operaciones por cohortes temporales (p.ej. semana de entrada) y,
    dentro de cada cohorte, pondera por riesgo las operaciones CONCURRENTES usando
    la covarianza shrunk de sus tickers. Devuelve (serie de retornos por cohorte,
    metricas).

    metodo: 'igual' | 'inv_vol' | 'hrp'. Compara el conjunto ponderado por riesgo
    frente al equiponderado."""
    d = res_sel.copy()
    d["fecha"] = pd.to_datetime(d["fecha"])
    # retornos diarios por ticker para estimar covarianza
    rets = {}
    for t, df in precios.items():
        c = df["close"].astype(float)
        rets[t] = np.log(c / c.shift(1))
    cohortes = d.groupby(pd.Grouper(key="fecha", freq=freq))
    filas = []
    for fecha, g in cohortes:
        g = g[g["ticker"].isin(rets)]
        if len(g) == 0:
            continue
        tickers = list(g["ticker"])
        r_op = g[col_ret].to_numpy(float)
        if len(tickers) == 1:
            w = np.array([1.0])
        else:
            # ventana de retornos previos a la fecha de la cohorte
            M = []
            fin = fecha
            for t in tickers:
                s = rets[t]
                s = s[s.index < fin].tail(ventana)
                M.append(s)
            R = pd.concat(M, axis=1).dropna()
            if len(R) < 20:
                w = np.full(len(tickers), 1.0 / len(tickers))
            else:
                cov = covarianza_shrunk(R.to_numpy())
                w = pesos_riesgo(cov, metodo)
        filas.append({"fecha": fecha, "ret": float(np.dot(w, r_op)), "n": len(tickers)})
    if not filas:
        return pd.Series(dtype=float), {}
    serie = pd.DataFrame(filas).set_index("fecha")["ret"]
    from .robustez import sharpe_no_anualizado
    r = serie.to_numpy()
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    met = {"cohortes": len(serie), "ret_medio": float(r.mean()),
           "sharpe_op": sharpe_no_anualizado(r),
           "sharpe_anual": sharpe_no_anualizado(r) * np.sqrt(52 if freq == "W" else 12),
           "profit_factor": float(pos / max(neg, 1e-9)),
           "pct_positivas": float((r > 0).mean())}
    return serie, met
