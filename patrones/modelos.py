"""Modelos supervisados de contraste.

La memoria de casos tiene que competir contra algo. Si una regresion logistica
sobre los mismos datos lo hace igual de bien, la memoria no aporta nada y hay
que decirlo en la memoria del trabajo.
"""
import numpy as np

from .ventanas import GRUPOS_ESCALARES


def matriz_supervisada(banco, incluir_formas=True, incluir_tf=True):
    """Aplana un BancoCasos a la matriz X que espera sklearn."""
    bloques, nombres = [], []
    if incluir_formas:
        bloques += [banco.formas, banco.formas_vol]
        nombres += [f"forma_{i}" for i in range(banco.formas.shape[1])]
        nombres += [f"forma_vol_{i}" for i in range(banco.formas_vol.shape[1])]
    if incluir_tf:
        for k, v in banco.tf.items():
            bloques.append(v)
            nombres += [f"{k}_{i}" for i in range(v.shape[1])]
    bloques.append(banco.feat)
    nombres += list(banco.nombres_feat)
    X = np.concatenate(bloques, axis=1).astype(np.float32)
    y = (banco.y["ret"].to_numpy(float) > 0).astype(int)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y, nombres


class ModeloSupervisado:
    """Envoltorio fino sobre sklearn: logistica o arboles con gradiente."""

    def __init__(self, tipo="gbm", semilla=0):
        self.tipo = tipo
        self.semilla = semilla
        self.modelo = None
        self.nombres = None

    def _crear(self):
        if self.tipo == "logistica":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=0.05,
                                                    random_state=self.semilla))
        if self.tipo == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0,
                min_samples_leaf=40, early_stopping=True, validation_fraction=0.15,
                random_state=self.semilla)
        raise ValueError(f"tipo desconocido: {self.tipo}")

    def entrenar(self, banco):
        X, y, self.nombres = matriz_supervisada(banco)
        self.modelo = self._crear()
        self.modelo.fit(X, y)
        return self

    def prob_sube(self, banco):
        X, _, _ = matriz_supervisada(banco)
        return self.modelo.predict_proba(X)[:, 1]

    def importancias(self, banco, n=15, semilla=0):
        """Importancia por permutacion, agrupada por bloque de features.

        Interesa el grupo, no la columna suelta: saber que "volumen" aporta mas
        que "noticias" dice algo; saber que forma_17 aporta 0.003 no dice nada.
        """
        from sklearn.inspection import permutation_importance
        X, y, nombres = matriz_supervisada(banco)
        r = permutation_importance(self.modelo, X, y, n_repeats=5,
                                   random_state=semilla, scoring="roc_auc")
        grupos = {}
        for nom, imp in zip(nombres, r.importances_mean):
            if nom.startswith("forma_vol"):
                g = "forma_volumen"
            elif nom.startswith("forma_"):
                g = "forma"
            elif nom.startswith("semanal_"):
                g = "tf_semanal"
            elif nom.startswith("mensual_"):
                g = "tf_mensual"
            else:
                g = next((k for k, cols in GRUPOS_ESCALARES.items() if nom in cols), "otros")
            grupos[g] = grupos.get(g, 0.0) + float(imp)
        return dict(sorted(grupos.items(), key=lambda kv: -kv[1])[:n])
