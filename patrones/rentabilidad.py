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


def etiqueta_desde_recorrido(sel, k_pt, k_sl):
    """Recalcula el retorno (en ATR) de cada senal para un ratio objetivo/stop,
    usando el recorrido de maximos/minimos guardado en las columnas cam_*."""
    hi = np.vstack(sel["cam_high"].to_numpy())
    lo = np.vstack(sel["cam_low"].to_numpy())
    cc = np.vstack(sel["cam_close"].to_numpy())
    lado = sel["lado"].to_numpy()[:, None]
    # para un largo, objetivo arriba y stop abajo; para un corto, al reves
    fav = np.where(lado > 0, hi, -lo)       # recorrido a favor del lado (ATR)
    adv = np.where(lado > 0, lo, -hi)       # recorrido en contra
    H = hi.shape[1]; GR = H + 10
    t_obj = np.where((fav >= k_pt).any(1), (fav >= k_pt).argmax(1), GR)
    t_stop = np.where((adv <= -k_sl).any(1), (adv <= -k_sl).argmax(1), GR)
    ret = (lado.ravel() * cc[:, -1])        # vencimiento: cierre a favor del lado
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
}


def aplicar_sistema(res, reglas=None, coste_atr=0.05):
    """Aplica las reglas de operativa a las senales y devuelve (operadas, metricas).

    Decide que operar con filtros defendibles a priori y mide el resultado por
    operacion (en ATR) y por pliegue, para comprobar que es estable en el tiempo.
    """
    r = {**REGLAS_SISTEMA, **(reglas or {})}
    d = res.copy()
    d["ret_op"] = etiqueta_desde_recorrido(d, r["k_pt"], r["k_sl"]) - coste_atr
    vol = pd.Series([np.max(h) - np.min(l) for h, l in zip(d["cam_high"], d["cam_low"])],
                    index=d.index)
    vol_alta = pd.qcut(vol.rank(method="first"), 3, labels=["baja", "media", "alta"]) != "baja"

    m = d["prob_meta"] >= r["umbral_meta"]
    if r["solo_largos"]:
        m &= d["lado"] > 0
    if r["evitar_vol_baja"]:
        m &= vol_alta
    if r["solo_mercado_alcista"] and "mkt_alcista" in d:
        m &= d["mkt_alcista"] > 0.5
    sel = d[m]
    if len(sel) < 20:
        return sel, {"operaciones": len(sel)}
    rr = sel["ret_op"].to_numpy()
    pos, neg = rr[rr > 0].sum(), -rr[rr < 0].sum()
    por_pliegue = {int(pl): float(g["ret_op"].mean()) for pl, g in sel.groupby("pliegue")}
    met = {
        "operaciones": int(len(sel)), "cobertura": len(sel) / len(res),
        "ret_medio_atr": float(rr.mean()), "sharpe": _sharpe(rr),
        "profit_factor": float(pos / max(neg, 1e-9)), "pct_ganadoras": float((rr > 0).mean()),
        "pliegues_rentables": int(sum(v > 0 for v in por_pliegue.values())),
        "n_pliegues": len(por_pliegue), "ret_por_pliegue": por_pliegue,
    }
    return sel, met
