"""Analisis de rentabilidad: barreras asimetricas y dimensionado por meta.

Un sistema puede ser rentable aunque acierte menos del 50%: basta con que las
ganancias sean mayores que las perdidas (asimetria riesgo/recompensa) y con
operar mas fuerte donde se tiene mas confianza. Este modulo mide eso sobre las
senales que produce la evaluacion meta, usando los recorridos ya guardados.
"""
import numpy as np
import pandas as pd


def _sharpe(r, periodos=25):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periodos))


def _max_drawdown(equity):
    return float((equity - np.maximum.accumulate(equity)).min())


def etiqueta_desde_recorrido(sel, k_pt, k_sl, cap=None):
    """Recalcula el retorno (en ATR) de cada senal para un ratio objetivo/stop,
    usando el recorrido de maximos/minimos guardado en las columnas cam_*.

    cap: horizonte efectivo por caso (numero de barras futuras que se pueden
    usar). Sirve para la SALIDA AL CIERRE (EOD): una operacion intradia no puede
    sostenerse mas alla del final de la sesion, asi que solo cuentan las barreras
    tocadas dentro de la sesion y, si no se toca ninguna, se cierra al precio de
    la ultima barra del dia (no al horizonte completo). Con cap=None se usa el
    horizonte completo (comportamiento del sistema diario, sin cambios).
    """
    hi = np.vstack(sel["cam_high"].to_numpy())
    lo = np.vstack(sel["cam_low"].to_numpy())
    cc = np.vstack(sel["cam_close"].to_numpy())
    lado = sel["lado"].to_numpy()[:, None]
    # para un largo, objetivo arriba y stop abajo; para un corto, al reves
    fav = np.where(lado > 0, hi, -lo)       # recorrido a favor del lado (ATR)
    adv = np.where(lado > 0, lo, -hi)       # recorrido en contra
    n, H = hi.shape
    GR = H + 10
    if cap is None:
        heff = np.full(n, H, int)
    else:
        heff = np.clip(np.nan_to_num(np.asarray(cap, float), nan=H).astype(int), 1, H)
    dentro = np.arange(H)[None, :] < heff[:, None]   # barras dentro de la sesion
    obj = (fav >= k_pt) & dentro
    stp = (adv <= -k_sl) & dentro
    t_obj = np.where(obj.any(1), obj.argmax(1), GR)
    t_stop = np.where(stp.any(1), stp.argmax(1), GR)
    # vencimiento: cierre de la ultima barra admisible (cierre de sesion si EOD)
    cc_salida = cc[np.arange(n), heff - 1]
    ret = lado.ravel() * cc_salida
    ret = np.where(t_obj < t_stop, k_pt, np.where(t_stop < GR, -k_sl, ret))
    return ret.astype(float)


def resultado_operaciones(res, k_pt, k_sl, coste_atr=0.05,
                          dimensionar=True, umbral_meta=0.5):
    """Retorno por operacion (en ATR) con un ratio dado y seleccion del meta.

    El tamano de la posicion es proporcional a la confianza del meta si dimensionar.
    """
    sel = res[res["prob_meta"] >= umbral_meta].copy()
    if len(sel) == 0:
        return sel, {}
    r = etiqueta_desde_recorrido(sel, k_pt, k_sl)
    tam = np.clip((sel["prob_meta"].to_numpy() - 0.5) * 2, 0.0, 1.0) if dimensionar else np.ones(len(sel))
    r_neto = r * tam - coste_atr * tam            # coste proporcional al tamano
    sel["ret_op_atr"] = r_neto
    sel["tam"] = tam
    equity = np.cumsum(r_neto)
    met = {
        "operaciones": int(len(sel)), "cobertura": len(sel) / len(res),
        "ret_medio_atr": float(np.mean(r_neto)), "ret_total_atr": float(np.sum(r_neto)),
        "pct_ganadoras": float((r > 0).mean()),
        "sharpe": _sharpe(r_neto), "max_dd_atr": _max_drawdown(equity),
        "profit_factor": float(r_neto[r_neto > 0].sum() / max(-r_neto[r_neto < 0].sum(), 1e-9)),
    }
    return sel, met


def barrido_barreras(res, ratios, coste_atr=0.05, umbral_meta=0.55, dimensionar=True):
    """Prueba varios ratios objetivo/stop y devuelve sus metricas de rentabilidad."""
    filas = []
    for k_pt, k_sl in ratios:
        _, m = resultado_operaciones(res, k_pt, k_sl, coste_atr, dimensionar, umbral_meta)
        if m:
            m.update({"k_pt": k_pt, "k_sl": k_sl, "ratio": f"{k_pt}:{k_sl}"})
            filas.append(m)
    return pd.DataFrame(filas)


def edge_por_segmento(res, k_pt=2.5, k_sl=1.0, coste_atr=0.05, min_n=40):
    """Desglosa el rendimiento por segmentos para ver DONDE funciona el sistema.

    Segmenta por regimen de volatilidad, figura chartista, sector y lado
    (largo/corto), y para cada grupo da el acierto del meta, el retorno medio por
    operacion (ratio dado) y el profit factor. Responde 'en que condiciones
    conviene operar y en cuales no'.
    """
    d = res.copy()
    d["ret_op"] = etiqueta_desde_recorrido(d, k_pt, k_sl) - coste_atr
    d["gana"] = d["ret_op"] > 0
    # regimen de volatilidad: por atr_pct si existe, si no por la amplitud del
    # recorrido (max menos min en ATR), que siempre esta disponible.
    vol = d["atr_pct"] if ("atr_pct" in d and d["atr_pct"].notna().any()) else pd.Series(
        [float(np.vstack([h])[0].max() - np.vstack([l])[0].min())
         for h, l in zip(d["cam_high"], d["cam_low"])], index=d.index)
    if vol.notna().any():
        d["regimen_vol"] = pd.qcut(vol.rank(method="first"), 3,
                                   labels=["baja", "media", "alta"])
    d["lado_txt"] = np.where(d["lado"] > 0, "largo", "corto")

    out = {}
    for dim in ["regimen_vol", "patron", "sector", "lado_txt"]:
        if dim not in d or d[dim].isna().all():
            continue
        filas = []
        for val, g in d.groupby(dim, observed=True):
            if len(g) < min_n:
                continue
            pos = g[g["ret_op"] > 0]["ret_op"].sum()
            neg = -g[g["ret_op"] < 0]["ret_op"].sum()
            filas.append({"segmento": str(val), "n": len(g),
                          "acierto_meta": float(g["acierto_primario"].mean()),
                          "ret_medio_atr": float(g["ret_op"].mean()),
                          "pct_ganadoras": float(g["gana"].mean()),
                          "profit_factor": float(pos / max(neg, 1e-9))})
        if filas:
            out[dim] = pd.DataFrame(filas).sort_values("ret_medio_atr", ascending=False)
    return out


# Reglas de operativa decididas A PRIORI (principios generales, no ajustadas al
# test): asimetria riesgo/recompensa, sesgo largo en renta variable, evitar la
# baja volatilidad y operar a favor de la tendencia del mercado. El filtro por
# sector NO se incluye: elegir "buenos sectores" mirando el test es sobreajuste.
REGLAS_SISTEMA = {
    "k_pt": 3.0, "k_sl": 1.5, "umbral_meta": 0.55,
    "solo_largos": True, "evitar_vol_baja": True, "solo_mercado_alcista": True,
    "permitir_cortos": True,    # largos si el mercado sube, cortos si cae (robusto en ambos)
    "salida_cierre": False,     # intradia: forzar el cierre de la posicion al final del dia (EOD)
    "rvol_min": None,           # intradia: operar solo con volumen relativo >= rvol_min
    "cuantil_meta": None,       # seleccionar el top-x por ranking del meta (robusto a la calibracion)
    "sizing": "igual",          # tamano: 'igual' | 'confianza' | 'kelly' (frac. de Kelly)
    "kelly_fraccion": 0.25,     # fraccion de Kelly si sizing='kelly'
    "vol_objetivo": None,       # ATR% de referencia para escalar inverso a la volatilidad
    "solo_sectores": None,      # lista de sectores a operar (p.ej. ciclicos); None = todos
    "caps_horizonte": None,     # ensemble: lista de horizontes de salida a promediar (p.ej. [5,10,15])
    "conviccion_min": None,     # exigir conviccion del primario alta (cuantil de |prob-0.5|)
}


def _coste_vector(d, coste_atr):
    """Resuelve el coste por operacion (en ATR). Admite:
      - un float: coste PLANO (comportamiento por defecto, retrocompatible);
      - "realista": coste por fila desde patrones.costes (spread+comision escalan
        inverso a la volatilidad; ver docs/aprendizaje_quant.md);
      - un callable f(d)->array: coste a medida por fila."""
    if isinstance(coste_atr, str) and coste_atr == "realista":
        from .costes import coste_por_fila
        return coste_por_fila(d)
    if callable(coste_atr):
        return np.asarray(coste_atr(d), float)
    return float(coste_atr)


def aplicar_sistema(res, reglas=None, coste_atr=0.05, invertir=False):
    """Aplica las reglas de operativa a las senales y devuelve (operadas, metricas).

    Decide que operar con filtros defendibles a priori y mide el resultado por
    operacion (en ATR) y por pliegue, para comprobar que es estable en el tiempo.

    coste_atr: float (coste plano, por defecto 0.05) o "realista" (coste por fila
    que escala inverso a la volatilidad; mas honesto, ver patrones.costes).
    """
    r = {**REGLAS_SISTEMA, **(reglas or {})}
    d = res.copy()
    if invertir:                    # reversion: operar lo contrario de la senal
        d["lado"] = -d["lado"]
    cv = _coste_vector(d, coste_atr)      # coste en ATR: escalar o array alineado a d
    # salida al cierre (EOD): el horizonte de cada operacion se recorta al final
    # de la sesion. Requiere la columna 'barras_hasta_cierre' (banco intradia).
    cap = d["barras_hasta_cierre"].to_numpy() if (
        r.get("salida_cierre") and "barras_hasta_cierre" in d) else None
    if r.get("caps_horizonte"):
        # ensemble de horizontes de salida: promedia el resultado a varios
        # horizontes (menos dependencia del timing exacto de salida).
        n = len(d)
        rets = [etiqueta_desde_recorrido(d, r["k_pt"], r["k_sl"], cap=np.full(n, c))
                for c in r["caps_horizonte"]]
        d["ret_op"] = np.mean(rets, axis=0) - cv
    else:
        d["ret_op"] = etiqueta_desde_recorrido(d, r["k_pt"], r["k_sl"], cap=cap) - cv
    vol = pd.Series([np.max(h) - np.min(l) for h, l in zip(d["cam_high"], d["cam_low"])],
                    index=d.index)
    vol_alta = pd.qcut(vol.rank(method="first"), 3, labels=["baja", "media", "alta"]) != "baja"

    # seleccion por ranking (cuantil) o por umbral absoluto del meta. El cuantil es
    # robusto a la calibracion: la isotonica es monotona, asi que el top-x no cambia.
    if r.get("cuantil_meta"):
        umbral = d["prob_meta"].quantile(1.0 - r["cuantil_meta"])
        m = d["prob_meta"] >= umbral
    else:
        m = d["prob_meta"] >= r["umbral_meta"]
    if r["evitar_vol_baja"]:
        m &= vol_alta
    if r.get("salida_cierre") and "barras_hasta_cierre" in d:
        m &= d["barras_hasta_cierre"] >= 1       # sin barra posterior no hay operacion
    if r.get("rvol_min") is not None and "rvol" in d:
        m &= d["rvol"] >= r["rvol_min"]          # filtro "stocks in play": volumen anormal
    if r.get("solo_sectores") and "sector" in d:
        m &= d["sector"].isin(r["solo_sectores"])  # p.ej. operar solo sectores ciclicos
    if r.get("conviccion_min") is not None and "prob" in d:
        # exigir que el primario se incline con fuerza (no una señal 50/50): operar
        # solo donde la conviccion esta en el cuantil superior. Mejora la calidad.
        conv = (d["prob"] - 0.5).abs()
        m &= conv >= conv.quantile(r["conviccion_min"])
    if r.get("permitir_cortos") and "mkt_alcista" in d:
        # a favor del regimen: largos cuando el mercado sube, cortos cuando cae.
        alcista = d["mkt_alcista"] > 0.5
        m &= ((alcista & (d["lado"] > 0)) | (~alcista & (d["lado"] < 0)))
    else:
        if r["solo_largos"]:
            m &= d["lado"] > 0
        if r["solo_mercado_alcista"] and "mkt_alcista" in d:
            m &= d["mkt_alcista"] > 0.5
    sel = d[m].copy()
    if len(sel) < 20:
        return sel, {"operaciones": len(sel)}
    # tamano de cada operacion (Kelly/confianza/igual) x escalado de volatilidad.
    # El retorno dimensionado escala coste y resultado por igual (coste ya restado).
    from .sizing import pesos_posicion
    w = pesos_posicion(sel, r, b=r["k_pt"] / max(r["k_sl"], 1e-9))
    # tamano 0 = sin ventaja (Kelly): no es una operacion, se descarta de las metricas.
    keep = w > 1e-9
    sel, w = sel[keep].copy(), w[keep]
    if len(sel) < 20:
        return sel, {"operaciones": len(sel)}
    sel["tam"] = w
    sel["ret_sized"] = sel["ret_op"].to_numpy() * w
    rr = sel["ret_sized"].to_numpy()
    pos, neg = rr[rr > 0].sum(), -rr[rr < 0].sum()
    por_pliegue = {int(pl): float(g["ret_sized"].mean()) for pl, g in sel.groupby("pliegue")}
    met = {
        "operaciones": int(len(sel)), "cobertura": len(sel) / len(res),
        "ret_medio_atr": float(rr.mean()), "sharpe": _sharpe(rr),
        "profit_factor": float(pos / max(neg, 1e-9)),
        "pct_ganadoras": float((rr > 0).mean()),
        "tam_medio": float(np.mean(w)),
        "pliegues_rentables": int(sum(v > 0 for v in por_pliegue.values())),
        "n_pliegues": len(por_pliegue), "ret_por_pliegue": por_pliegue,
    }
    return sel, met
