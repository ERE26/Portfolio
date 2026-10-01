"""Modelos primarios intercambiables: memoria de casos, GBM y ensamble.

El modelo primario da la señal de direccion. Hasta ahora era la memoria de casos
(k-vecinos): explicable y rica en features de acuerdo, pero lineal en la metrica
de similitud. El gradient boosting (GBM) aprende relaciones no lineales entre las
54 features (formas + indicadores + factores) y la direccion.

El ENSAMBLE combina ambos: la memoria aporta los vecinos y la explicabilidad; el
GBM aporta potencia predictiva. Dos modelos distintos que coinciden dan mas
confianza, y ese acuerdo es una feature que el meta-modelo aprovecha.

Todos exponen la misma interfaz .ajustar(banco) y .predecir(banco, i) -> Prediccion,
asi que la evaluacion walk-forward y el meta-labeling funcionan igual con cualquiera.
"""
import numpy as np
import pandas as pd

from .memoria import MemoriaPatrones, Prediccion, Calibrador
from .modelos import matriz_supervisada
from .evaluacion import submuestreo_no_solapado


class PrimarioGBM:
    """Gradient boosting sobre todas las features. Interfaz como la memoria."""

    def __init__(self, cfg, umbral_confianza=0.35, semilla=0, submuestreo=True):
        self.cfg = cfg
        self.umbral_confianza = umbral_confianza
        self.semilla = semilla
        self.submuestreo = submuestreo
        self.modelo = None
        self.calibrador = Calibrador()
        self._cache = {}

    def ajustar(self, banco):
        from sklearn.ensemble import HistGradientBoostingClassifier
        # submuestreo no solapado para no sobre-representar ventanas contiguas
        tr = submuestreo_no_solapado(banco, max(1, self.cfg.horizonte // 2)) \
            if self.submuestreo else banco
        X, y, self.nombres = matriz_supervisada(tr)
        self.modelo = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
            min_samples_leaf=40, early_stopping=True, validation_fraction=0.15,
            random_state=self.semilla)
        self.modelo.fit(X, y)
        return self

    def _matriz(self, banco):
        clave = id(banco)
        if clave not in self._cache:
            X, _, _ = matriz_supervisada(banco)
            self._cache = {clave: self.modelo.predict_proba(X)[:, 1]}  # solo el ultimo banco
        return self._cache[clave]

    def _prob(self, banco, i):
        return float(self._matriz(banco)[i])

    def predecir(self, banco, i, **kw):
        prob = self._prob(banco, i)
        prob_c = self.calibrador.calibrar(prob)
        conv = abs(2 * prob - 1)
        h = self.cfg.horizonte
        return Prediccion(
            ticker=str(banco.meta["ticker"].iloc[i]), fecha=pd.Timestamp(banco.meta["fecha"].iloc[i]),
            prob_sube_cruda=prob, prob_sube=prob_c, ret_esperado=0.0,
            mfe_esperado=0.0, mae_esperado=0.0, camino=np.zeros(h),
            camino_p10=np.zeros(h), camino_p90=np.zeros(h),
            confianza=float(np.clip(conv, 0, 1)), abstencion=conv < 0.05,
            distancia_media=float("nan"),
            vecinos=pd.DataFrame(columns=["ticker", "fecha", "distancia", "peso", "ret", "mfe", "mae"]),
            probs_grupo={}, atr_pct=float(banco.meta["atr_pct"].iloc[i]) if "atr_pct" in banco.meta else float("nan"))

    def aprender_del_error(self, pred, ret_real):
        self.calibrador.observar(pred.prob_sube_cruda, ret_real > 0)


class PrimarioEnsamble:
    """Combina memoria de casos y GBM: la memoria explica, el GBM potencia."""

    def __init__(self, cfg, k=30, peso_gbm=0.5, umbral_confianza=0.35, pesos=None,
                 semilla=0, **kw_mem):
        self.cfg = cfg
        self.peso_gbm = peso_gbm
        self.mem = MemoriaPatrones(cfg, k=k, umbral_confianza=umbral_confianza,
                                   pesos=pesos, **kw_mem)
        self.gbm = PrimarioGBM(cfg, umbral_confianza, semilla)
        self.umbral_confianza = umbral_confianza

    def ajustar(self, banco):
        self.mem.ajustar(banco)
        self.gbm.ajustar(banco)
        return self

    def predecir(self, banco, i, **kw):
        p = self.mem.predecir(banco, i, **kw)       # trae vecinos, probs_grupo, caminos
        pg = self.gbm._prob(banco, i)
        # NO se promedian las probabilidades (eso diluye la señal discriminativa del
        # GBM). La prediccion primaria sigue siendo la de la memoria (con sus
        # vecinos); la prob del GBM se expone como canal aparte y el META-modelo
        # aprende a combinar ambas. El acuerdo entre modelos sube la confianza.
        p.prob_gbm = pg
        acuerdo = 1.0 - abs(pg - p.prob_sube_cruda)
        p.confianza = float(np.clip(p.confianza * (0.7 + 0.3 * acuerdo), 0, 1))
        p.abstencion = p.confianza < self.umbral_confianza
        return p

    def aprender_del_error(self, pred, ret_real):
        self.mem.aprender_del_error(pred, ret_real)


def crear_primario(nombre, cfg, k=30, pesos=None, umbral_confianza=0.35, **kw):
    if nombre == "memoria":
        return MemoriaPatrones(cfg, k=k, pesos=pesos, umbral_confianza=umbral_confianza, **kw)
    if nombre == "gbm":
        return PrimarioGBM(cfg, umbral_confianza=umbral_confianza)
    if nombre == "ensamble":
        return PrimarioEnsamble(cfg, k=k, pesos=pesos, umbral_confianza=umbral_confianza, **kw)
    raise ValueError(f"primario desconocido: {nombre}")
