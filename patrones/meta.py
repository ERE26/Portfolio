"""Meta-labeling: aprender CUAL de las senales elegir, no solo la direccion.

La memoria de casos da una senal primaria (direccion + confianza). El meta-modelo
es un segundo clasificador que, mirando las caracteristicas de esa senal (cuanto
se parecen los vecinos, si estan de acuerdo, el regimen de mercado, la figura...),
predice la probabilidad de que la senal ACIERTE. Se opera solo cuando esa
probabilidad supera un umbral.

Idea (Lopez de Prado, 'Advances in Financial Machine Learning', cap. 3): el
modelo primario fija la direccion; el meta-modelo fija el tamano/el si-operar.
Sube la PRECISION de las operaciones elegidas aunque el acierto bruto no cambie,
que es exactamente "no solo acertar, sino elegir cual".
"""
import numpy as np
import pandas as pd

from .figuras import describir

# figuras principales que el meta-modelo mira como variables indicadoras
FIGURAS_META = ["doble_techo", "doble_suelo", "hombro_cabeza_hombro",
                "hombro_cabeza_hombro_invertido", "canal_alcista", "canal_bajista",
                "triangulo", "rotura_resistencia_alcista", "rotura_soporte_bajista"]


def features_meta(pred, banco=None, i=None):
    """Vector de caracteristicas de una prediccion, para el meta-modelo.

    Captura 'que tan fiable es esta senal': fuerza, conviccion, acuerdo entre
    vecinos y entre grupos, cercania, regimen y figura.
    """
    v = pred.vecinos
    rets = v["ret"].to_numpy() if len(v) else np.array([0.0])
    f = {
        "prob_cruda": pred.prob_sube_cruda,
        "conviccion": abs(2 * pred.prob_sube_cruda - 1),
        "confianza": pred.confianza,
        "dist_media": pred.distancia_media if np.isfinite(pred.distancia_media) else 1.0,
        "n_vecinos": float(len(v)),
        "disp_ret_vecinos": float(np.std(rets)) if len(rets) > 1 else 1.0,
        "rango_ret_vecinos": float(np.ptp(rets)) if len(rets) > 1 else 1.0,
        "frac_vecinos_suben": float((rets > 0).mean()) if len(rets) else 0.5,
        "mfe_esp": pred.mfe_esperado,
        "mae_esp": pred.mae_esperado,
        "ratio_mfe_mae": pred.mfe_esperado / (abs(pred.mae_esperado) + 1e-6),
    }
    # si hay modelo GBM, su probabilidad y su acuerdo con la memoria son señal:
    # dos modelos distintos que coinciden dan mas confianza.
    if np.isfinite(getattr(pred, "prob_gbm", float("nan"))):
        f["prob_gbm"] = pred.prob_gbm
        f["acuerdo_modelos"] = 1.0 - abs(pred.prob_gbm - pred.prob_sube_cruda)
        f["conviccion_gbm"] = abs(2 * pred.prob_gbm - 1)
    # acuerdo entre grupos: cuanto coinciden las probs por grupo con la global
    if pred.probs_grupo:
        pg = np.array(list(pred.probs_grupo.values()))
        f["disp_grupos"] = float(np.std(pg))
        f["frac_grupos_acuerdo"] = float(np.mean((pg > 0.5) == (pred.prob_sube_cruda > 0.5)))
    else:
        f["disp_grupos"] = 0.5
        f["frac_grupos_acuerdo"] = 0.5
    # regimen y figura del tramo consultado
    if banco is not None and i is not None:
        for c in ("atr_pct",):
            if c in banco.meta.columns:
                f[c] = float(banco.meta[c].iloc[i])
        d = describir(banco.formas[i], banco.formas_vol[i])
        f["pendiente_fig"] = d.get("pendiente", 0.0)
        for fig in FIGURAS_META:
            f[f"fig_{fig}"] = 1.0 if fig in d.get("figuras", []) else 0.0
    return f


def etiqueta_acierto(pred, y_fila):
    """¿La senal primaria acerto? Usa la etiqueta triple-barrera si existe.

    La senal predice subida (prob>0.5) o bajada. Acierta si el lado coincide con
    el desenlace: tb_label=+1 es beneficio al alza, -1 perdida. Para las vencidas
    (0) se usa el signo del retorno realizado.
    """
    lado = 1 if pred.prob_sube > 0.5 else -1
    tb = y_fila.get("tb_label", np.nan)
    if pd.notna(tb) and tb != 0:
        real = 1 if tb > 0 else -1
    else:
        real = 1 if y_fila.get("ret", 0.0) > 0 else -1
    return int(lado == real)


class MetaModelo:
    """Clasificador que predice P(la senal primaria acierta)."""

    def __init__(self, semilla=0):
        self.modelo = None
        self.cols = None
        self.semilla = semilla

    def entrenar(self, filas_features, aciertos, pesos=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = pd.DataFrame(filas_features).fillna(0.0)
        self.cols = list(X.columns)
        y = np.asarray(aciertos, int)
        self.modelo = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
            min_samples_leaf=30, early_stopping=True, validation_fraction=0.15,
            random_state=self.semilla)
        self.modelo.fit(X.to_numpy(), y, sample_weight=pesos)
        return self

    def prob_acierto(self, filas_features):
        X = pd.DataFrame(filas_features).reindex(columns=self.cols).fillna(0.0)
        return self.modelo.predict_proba(X.to_numpy())[:, 1]

    def importancias(self, filas_features, aciertos, n=12):
        from sklearn.inspection import permutation_importance
        X = pd.DataFrame(filas_features).reindex(columns=self.cols).fillna(0.0)
        r = permutation_importance(self.modelo, X.to_numpy(), np.asarray(aciertos, int),
                                   n_repeats=5, random_state=self.semilla, scoring="roc_auc")
        return dict(sorted(zip(self.cols, r.importances_mean), key=lambda kv: -kv[1])[:n])
