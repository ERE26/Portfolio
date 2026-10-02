"""Modelos de volatilidad condicional: la volatilidad NO es constante, llega en
rachas (clustering). Capturar eso mejora el sizing, el riesgo y la selección.

Dos niveles, a propósito:

1. `vol_ewma` / `rasgos_vol_condicional`  —  NUMPY PURO, sin dependencias. Es la
   familia GARCH en su forma RiskMetrics (IGARCH con lambda fijo): la varianza de
   hoy es una mezcla exponencial de la de ayer y el shock de ayer. Barato, estable
   y sin nada que compilar -> apto para la ruta de PRODUCCIÓN (features del meta,
   que se recalculan en cada ciclo en un contenedor que arranca limpio).

2. `garch_ajustar` / `pronostico_vol` / `var_garch`  —  GARCH(1,1) formal por
   máxima verosimilitud (libreria `arch`). Mejor para ANÁLISIS y gestión de riesgo
   (pronóstico de vol, Value-at-Risk): estima omega/alpha/beta y la persistencia.
   La importación es perezosa: si `arch` no está, el resto del módulo sigue vivo.

Por qué importa (lección cuant): el error típico es usar la vol histórica simple
(ventana fija, equiponderada), que reacciona tarde y se queda "pegada" tras un
shock. EWMA/GARCH ponderan lo reciente y modelan la reversión de la vol a su media
-> mejor estimación del riesgo AHORA, que es lo que alimenta el vol-targeting y el
Value-at-Risk. Referencias: Engle (1982, ARCH); Bollerslev (1986, GARCH);
J.P. Morgan RiskMetrics (1996, EWMA lambda=0.94).
"""
import numpy as np
import pandas as pd

LAMBDA_RISKMETRICS = 0.94   # decaimiento diario estandar de RiskMetrics
ANUAL = np.sqrt(252)


def vol_ewma(retornos, lam=LAMBDA_RISKMETRICS, anualizar=True):
    """Volatilidad condicional EWMA (RiskMetrics). Serie alineada a `retornos`.

    sigma2_t = lam * sigma2_{t-1} + (1-lam) * r_{t-1}^2   (usa solo el PASADO: la
    vol en t no mira r_t, no hay look-ahead). Devuelve desviacion tipica.
    """
    r = pd.Series(retornos).astype(float).fillna(0.0).to_numpy()
    n = len(r)
    s2 = np.empty(n)
    if n == 0:
        return pd.Series(dtype=float)
    # semilla: varianza de las primeras observaciones disponibles
    s2[0] = np.var(r[:min(n, 20)]) if n > 1 else r[0] ** 2
    for t in range(1, n):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2    # solo r hasta t-1
    sd = np.sqrt(s2)
    if anualizar:
        sd = sd * ANUAL
    idx = retornos.index if isinstance(retornos, pd.Series) else None
    return pd.Series(sd, index=idx)


def rasgos_vol_condicional(close, lam=LAMBDA_RISKMETRICS, lb_largo=100):
    """Rasgos de régimen de volatilidad para el meta-modelo, por barra y SIN
    look-ahead. Todo se calcula con informacion hasta t-1.

    - vol_cond: vol condicional EWMA anualizada (nivel de riesgo actual)
    - vol_ratio: vol condicional / vol de largo plazo (>1 = régimen agitado)
    - vol_subiendo: pendiente reciente de la vol condicional (clustering al alza)
    - ret_estandarizado: retorno de ayer en unidades de vol condicional (z-score)
    """
    c = pd.Series(close).astype(float)
    r = c.pct_change()
    vc = vol_ewma(r, lam=lam, anualizar=True)
    vc_diaria = vc / ANUAL
    vol_largo = r.rolling(lb_largo, min_periods=lb_largo // 2).std() * ANUAL
    ratio = (vc / vol_largo).replace([np.inf, -np.inf], np.nan)
    subiendo = vc.pct_change(5)
    # retorno de AYER estandarizado por la vol condicional de AYER (ambos < t)
    z = (r.shift(1) / vc_diaria.shift(1)).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({
        "vol_cond": vc,
        "vol_ratio": ratio,
        "vol_subiendo": subiendo,
        "ret_estandarizado": z,
    })


# --------------------------------------------------------------- GARCH formal
def garch_ajustar(retornos, p=1, q=1, dist="t", reescalar=True):
    """Ajusta un GARCH(p,q) por máxima verosimilitud (libreria `arch`). Devuelve el
    resultado ajustado o lanza ImportError si `arch` no esta instalado."""
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError("GARCH formal requiere `arch` (pip install arch). Para la "
                          "ruta de produccion usa vol_ewma / rasgos_vol_condicional.") from e
    r = pd.Series(retornos).dropna().astype(float)
    # `arch` prefiere retornos en % (mejor condicionamiento numerico del optimizador)
    escala = 100.0 if reescalar else 1.0
    am = arch_model(r * escala, mean="Constant", vol="GARCH", p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    res._escala_ilegalisimo = escala
    return res


def pronostico_vol(res, horizonte=1, anualizar=True):
    """Pronóstico de volatilidad para los próximos `horizonte` días a partir de un
    GARCH ajustado. Devuelve la vol (no la varianza), desescalada y anualizada."""
    escala = getattr(res, "_escala_ilegalisimo", 1.0)
    f = res.forecast(horizon=horizonte, reindex=False)
    var = f.variance.to_numpy()[-1]                 # varianza por horizonte, en %^2
    sd = np.sqrt(var) / escala                       # vuelve a fraccion
    if anualizar:
        sd = sd * ANUAL
    return sd if horizonte > 1 else float(sd[0])


def persistencia(res):
    """alpha+beta de un GARCH(1,1): cuan persistente es la volatilidad (cerca de 1
    = los shocks de vol duran mucho; media-vida = ln(0.5)/ln(alpha+beta))."""
    pr = res.params
    a = sum(v for k, v in pr.items() if k.startswith("alpha"))
    b = sum(v for k, v in pr.items() if k.startswith("beta"))
    ab = float(a + b)
    media_vida = np.log(0.5) / np.log(ab) if 0 < ab < 1 else np.inf
    return {"alpha": float(a), "beta": float(b), "persistencia": ab, "media_vida_dias": media_vida}


# --------------------------------------------------------------- Value at Risk
def var_historico(retornos, nivel=0.95):
    """VaR historico (no parametrico): el cuantil empirico de la cola de pérdidas.
    Devuelve una pérdida POSITIVA (p.ej. 0.021 = 2.1% en el peor 5% de los dias)."""
    r = pd.Series(retornos).dropna().astype(float)
    return float(-np.quantile(r, 1 - nivel)) if len(r) else float("nan")


def var_garch(retornos, nivel=0.95, dist="t"):
    """VaR a 1 dia con la vol CONDICIONAL de un GARCH(1,1) y el cuantil de la
    distribucion ajustada (t de Student capta mejor las colas gordas que la normal).
    Mucho mas reactivo que el VaR historico tras un cambio de régimen."""
    res = garch_ajustar(retornos, dist=dist)
    sd = pronostico_vol(res, horizonte=1, anualizar=False)     # vol diaria pronosticada
    mu = float(res.params.get("mu", 0.0)) / getattr(res, "_escala_ilegalisimo", 1.0)
    try:
        from scipy import stats
        nu = float(res.params.get("nu", np.nan))
        if np.isfinite(nu):
            q = stats.t.ppf(1 - nivel, df=nu) / np.sqrt(nu / (nu - 2))   # t estandarizada
        else:
            q = stats.norm.ppf(1 - nivel)
    except Exception:
        from scipy import stats
        q = stats.norm.ppf(1 - nivel)
    return float(-(mu + sd * q))
