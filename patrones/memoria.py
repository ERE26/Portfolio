"""Memoria de patrones: predice por casos parecidos y aprende de sus errores.

Como funciona, en una frase: guarda miles de tramos de grafico junto con lo que
paso despues, y cuando le das un tramo nuevo busca los que mas se le parecen y
promedia sus desenlaces.

Tres cosas lo separan de un k-vecinos de manual, y son las que hacen que los
resultados sean creibles:

1. PURGA TEMPORAL. Un caso solo puede ser vecino si su resultado ya se conocia
   antes de la fecha de la consulta. Si no, estarias prediciendo el lunes con
   informacion del viernes siguiente.

2. SEPARACION ENTRE VECINOS. Dos ventanas consecutivas del mismo valor se
   solapan en 39 de 40 barras: son casi el mismo caso. Si no lo controlas, tus
   "10 vecinos independientes" son en realidad uno repetido diez veces y la
   confianza que reportas es falsa. Aqui se exige separacion minima en el tiempo
   entre vecinos del mismo valor.

3. APRENDE DEL ERROR DE DOS FORMAS. Reajusta el peso de cada grupo de features
   segun quien acerto (Hedge / gradiente exponenciado) y recalibra sus
   probabilidades contra la frecuencia real observada.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ventanas import GRUPOS_ESCALARES, BancoCasos, Config

EPS = 1e-9

PESOS_INICIALES = {
    "forma": 0.25, "tf_semanal": 0.10, "tf_mensual": 0.05, "forma_volumen": 0.08,
    "osciladores": 0.12, "momento": 0.12, "volumen": 0.08, "volatilidad": 0.07,
    "contexto": 0.08, "noticias": 0.05,
}


def percentil_ponderado(x, w, q):
    """Percentil con pesos. Se usa para el cono de la prediccion."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    orden = np.argsort(x)
    x, w = x[orden], w[orden]
    acum = np.cumsum(w) / max(w.sum(), EPS)
    return float(np.interp(q, acum, x))


class Calibrador:
    """Convierte la probabilidad cruda en una probabilidad honesta.

    Si el sistema dice '70%' y de esas veces solo acierta el 55%, esta
    mintiendo. El calibrador guarda lo que paso de verdad en cada tramo de
    probabilidad y corrige, mezclando con el valor crudo segun cuantos casos
    tenga (poca muestra -> se fia mas del valor crudo).
    """

    def __init__(self, n_tramos=10, fuerza=25.0):
        self.bordes = np.linspace(0.0, 1.0, n_tramos + 1)
        self.fuerza = fuerza
        self.aciertos = np.zeros(n_tramos)
        self.total = np.zeros(n_tramos)

    def _tramo(self, p):
        return int(np.clip(np.digitize(p, self.bordes) - 1, 0, len(self.total) - 1))

    def observar(self, p_cruda, subio):
        i = self._tramo(p_cruda)
        self.total[i] += 1
        self.aciertos[i] += 1.0 if subio else 0.0

    def calibrar(self, p_cruda):
        i = self._tramo(p_cruda)
        n = self.total[i]
        if n == 0:
            return float(p_cruda)
        empirica = self.aciertos[i] / n
        peso = n / (n + self.fuerza)
        return float(np.clip(peso * empirica + (1 - peso) * p_cruda, 0.01, 0.99))

    def tabla(self):
        centros = (self.bordes[:-1] + self.bordes[1:]) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            frec = np.where(self.total > 0, self.aciertos / np.maximum(self.total, 1), np.nan)
        return pd.DataFrame({"prob_predicha": centros, "frec_real": frec, "n": self.total})

    def estado(self):
        return {"bordes": self.bordes.tolist(), "fuerza": self.fuerza,
                "aciertos": self.aciertos.tolist(), "total": self.total.tolist()}

    @classmethod
    def desde_estado(cls, d):
        c = cls(len(d["total"]), d["fuerza"])
        c.bordes = np.array(d["bordes"])
        c.aciertos = np.array(d["aciertos"])
        c.total = np.array(d["total"])
        return c


@dataclass
class Prediccion:
    ticker: str
    fecha: pd.Timestamp
    prob_sube_cruda: float
    prob_sube: float
    ret_esperado: float
    mfe_esperado: float
    mae_esperado: float
    camino: np.ndarray
    camino_p10: np.ndarray
    camino_p90: np.ndarray
    confianza: float
    abstencion: bool
    distancia_media: float
    vecinos: pd.DataFrame
    probs_grupo: dict = field(default_factory=dict)
    atr_pct: float = float("nan")

    def informe(self, n=6):
        """El informe en texto que se imprime por partida."""
        signo = "SUBE" if self.prob_sube > .5 else "BAJA"
        lineas = [
            f"{self.ticker}  {pd.Timestamp(self.fecha).date()}",
            f"  Prediccion: {signo}  prob_subida {self.prob_sube:.1%} "
            f"(cruda {self.prob_sube_cruda:.1%})",
            f"  Retorno esperado a {len(self.camino)} barras: {self.ret_esperado * 100:+.2f}%",
            f"  Recorrido esperado en ATR: a favor {self.mfe_esperado:+.2f} / "
            f"en contra {self.mae_esperado:+.2f}",
            f"  Confianza {self.confianza:.2f}"
            + ("  -> ME ABSTENGO, no reconozco esta forma" if self.abstencion else ""),
            f"  Distancia media a los vecinos: {self.distancia_media:.3f}",
        ]
        if len(self.vecinos):
            lineas.append(f"  Se parece a ({min(n, len(self.vecinos))} de {len(self.vecinos)} vecinos):")
            for _, v in self.vecinos.head(n).iterrows():
                etq = f"  [{v['patron']}]" if v.get("patron") not in (None, "", float("nan")) \
                    and isinstance(v.get("patron"), str) else ""
                lineas.append(f"    {v['ticker']:<8} {pd.Timestamp(v['fecha']).date()}  "
                              f"dist {v['distancia']:.3f}  peso {v['peso']:.2f}  "
                              f"hizo {v['ret'] * 100:+.2f}%{etq}")
            if "patron" in self.vecinos.columns:
                vc = self.vecinos["patron"].dropna()
                vc = vc[vc.astype(str) != ""]
                if len(vc):
                    top = vc.value_counts().head(2)
                    lineas.append("  Figuras dominantes entre los vecinos: " +
                                  ", ".join(f"{k} x{v}" for k, v in top.items()))
        return "\n".join(lineas)


class MemoriaPatrones:
    def __init__(self, cfg=None, k=40, pesos=None, umbral_confianza=0.35,
                 temperatura=0.35, separacion_dias=45, eta=0.35, peso_minimo=0.01):
        self.cfg = cfg or Config()
        self.k = k
        self.pesos = dict(pesos or PESOS_INICIALES)
        self.umbral_confianza = umbral_confianza
        self.temperatura = temperatura
        self.separacion_dias = separacion_dias
        self.eta = eta
        self.peso_minimo = peso_minimo
        self.calibrador = Calibrador()
        self.banco = None
        self.mu = None
        self.sd = None
        self.historial_pesos = []
        self.historial_errores = []

    # ------------------------------------------------------------- ajuste
    def ajustar(self, banco):
        """Carga los casos iniciales y fija la normalizacion de los indicadores.

        La media y la desviacion salen SOLO de estos casos. En la evaluacion
        walk-forward esto es entrenamiento, nunca test: si normalizases con todo
        el historico estarias colando informacion del futuro por la puerta de atras.
        """
        self.banco = banco
        self.mu = banco.feat.mean(axis=0)
        self.sd = banco.feat.std(axis=0)
        self.sd = np.where(self.sd > EPS, self.sd, 1.0)
        self._reindexar()
        self._calibrar_escala()
        return self

    def _calibrar_escala(self, n_muestra=120, semilla=0):
        """Aprende que distancia es 'normal' en esta memoria.

        La confianza no puede compararse contra un numero inventado a mano: la
        distancia tipica depende de cuantos casos hay y de cuantos grupos pesan.
        Asi que se muestrean casos de la propia memoria, se mira a que distancia
        media quedan sus k vecinos, y esa distribucion se usa despues como
        referencia. 'parecido 0.9' pasa a significar "esta forma se parece mas
        que el 90% de las consultas tipicas", que si es interpretable.
        """
        n = len(self.banco)
        if n < 200:
            self.ref_distancias = np.array([0.3, 0.5, 0.7])
            return
        rng = np.random.default_rng(semilla)
        # solo casos de la segunda mitad: los primeros no tienen pasado suficiente
        muestra = rng.choice(np.arange(n // 2, n), size=min(n_muestra, n - n // 2), replace=False)
        refs = []
        for i in muestra:
            p = self.predecir(self.banco, int(i), calibrar_confianza=False)
            if np.isfinite(p.distancia_media):
                refs.append(p.distancia_media)
        self.ref_distancias = np.sort(np.array(refs)) if refs else np.array([0.3, 0.5, 0.7])

    def _reindexar(self):
        b = self.banco
        self.fechas = b.meta["fecha"].to_numpy("datetime64[ns]")
        self.tickers = b.meta["ticker"].to_numpy()
        self.feat_z = np.clip((b.feat - self.mu) / self.sd, -5, 5).astype(np.float32)
        self.ret = b.y["ret"].to_numpy(float)
        self.mfe = b.y["mfe"].to_numpy(float)
        self.mae = b.y["mae"].to_numpy(float)
        self.patron = (b.meta["patron"].to_numpy() if "patron" in b.meta.columns
                       else np.array([""] * len(b.meta), object))
        dias = self.fechas.astype("datetime64[D]").astype(np.int64)
        self.cubo = dias // max(int(self.separacion_dias), 1)

    def anadir(self, banco_nuevo):
        """Mete casos nuevos en la memoria (esto es 'aprender la forma')."""
        if self.banco is None:
            return self.ajustar(banco_nuevo)
        b, n = self.banco, banco_nuevo
        cols = sorted(set(b.meta.columns) | set(n.meta.columns))
        self.banco = BancoCasos(
            b.cfg, np.concatenate([b.formas, n.formas]),
            np.concatenate([b.formas_vol, n.formas_vol]),
            {k: np.concatenate([b.tf[k], n.tf[k]]) for k in b.tf},
            np.concatenate([b.feat, n.feat]),
            pd.concat([b.meta.reindex(columns=cols), n.meta.reindex(columns=cols)],
                      ignore_index=True),
            pd.concat([b.y, n.y], ignore_index=True),
            np.concatenate([b.caminos, n.caminos]))
        self._reindexar()
        return self

    def __len__(self):
        return 0 if self.banco is None else len(self.banco)

    # --------------------------------------------------------- distancias
    def _consulta(self, banco, i):
        """Extrae de un banco el caso i en el formato que usan las distancias."""
        return {
            "forma": banco.formas[i],
            "forma_vol": banco.formas_vol[i],
            "tf": {k: v[i] for k, v in banco.tf.items()},
            "feat": np.clip((banco.feat[i] - self.mu) / self.sd, -5, 5).astype(np.float32),
        }

    def _distancias_por_grupo(self, q, cand):
        """Distancia por grupo. Formas: 1 - correlacion. Indicadores: euclidea normalizada."""
        b = self.banco
        d = {}
        L = b.formas.shape[1]
        d["forma"] = 1.0 - (b.formas[cand] @ q["forma"]) / L
        d["forma_volumen"] = 1.0 - (b.formas_vol[cand] @ q["forma_vol"]) / L
        for nombre, M in b.tf.items():
            d[f"tf_{nombre}"] = 1.0 - (M[cand] @ q["tf"][nombre]) / M.shape[1]
        for nombre, cols in GRUPOS_ESCALARES.items():
            idx = [b.nombres_feat.index(c) for c in cols if c in b.nombres_feat]
            if not idx:
                continue
            dif = self.feat_z[np.ix_(cand, idx)] - q["feat"][idx]
            d[nombre] = np.sqrt((dif ** 2).mean(axis=1))
        return d

    def _elegir_vecinos(self, orden, cand, k=None):
        """Coge los k mas parecidos exigiendo separacion temporal por valor.

        Sin esto los vecinos serian ventanas solapadas del mismo valor: el mismo
        caso contado k veces, con la confianza inflada que eso implica.

        La separacion se impone con cubos de 'separacion_dias'. Se acepta un
        candidato solo si su cubo y los dos contiguos estan libres para ese
        valor, lo que garantiza separacion >= separacion_dias con una sola
        consulta a un set por candidato en vez de recorrer los ya elegidos.
        """
        k = k or self.k
        elegidos, ocupados = [], set()
        for j in orden:
            g = cand[j]
            clave = (self.tickers[g], self.cubo[g])
            if clave in ocupados:
                continue
            t = clave[0]
            ocupados.update({(t, clave[1] - 1), clave, (t, clave[1] + 1)})
            elegidos.append(j)
            if len(elegidos) >= k:
                break
        return np.array(elegidos, int)

    # ---------------------------------------------------------- prediccion
    def predecir(self, banco, i, fecha=None, solo_sector=None, excluir_ticker=None,
                 embargo_dias=None, calibrar_confianza=True, grupos_independientes=True):
        q = self._consulta(banco, i)
        fecha = np.datetime64(pd.Timestamp(fecha if fecha is not None
                                           else banco.meta["fecha"].iloc[i]))
        ticker = str(banco.meta["ticker"].iloc[i])

        # PURGA: solo casos cuyo resultado ya se conocia antes de la consulta.
        if embargo_dias is None:
            embargo_dias = int(self.cfg.horizonte * 1.6) + 1
        limite = fecha - np.timedelta64(int(embargo_dias), "D")
        cand = np.flatnonzero(self.fechas <= limite)
        if solo_sector is not None and "sector" in self.banco.meta.columns:
            cand = cand[self.banco.meta["sector"].to_numpy()[cand] == solo_sector]
        if excluir_ticker is not None:
            cand = cand[self.tickers[cand] != excluir_ticker]
        if len(cand) < 5:
            return self._sin_datos(ticker, fecha, banco, i)

        dg = self._distancias_por_grupo(q, cand)
        total = np.zeros(len(cand))
        suma_w = 0.0
        for g, w in self.pesos.items():
            if g in dg and w > 0:
                total += w * dg[g]
                suma_w += w
        total /= max(suma_w, EPS)

        orden = np.argsort(total, kind="stable")[: self.k * 12]
        sel = self._elegir_vecinos(orden, cand)
        if len(sel) < 3:
            return self._sin_datos(ticker, fecha, banco, i)

        gidx = cand[sel]
        d = total[sel]
        w = np.exp(-d / max(self.temperatura, EPS))
        w = w / max(w.sum(), EPS)

        r = self.ret[gidx]
        prob_cruda = float(np.clip((w * (r > 0)).sum(), 0.001, 0.999))
        prob = self.calibrador.calibrar(prob_cruda)
        caminos = self.banco.caminos[gidx]
        camino = (w[:, None] * caminos).sum(axis=0)
        p10 = np.array([percentil_ponderado(caminos[:, t], w, .10) for t in range(caminos.shape[1])])
        p90 = np.array([percentil_ponderado(caminos[:, t], w, .90) for t in range(caminos.shape[1])])

        # Confianza = se parecen mucho Y estan de acuerdo Y son suficientes.
        if calibrar_confianza and getattr(self, "ref_distancias", None) is not None:
            # percentil inverso dentro de las distancias tipicas de esta memoria
            pos = np.searchsorted(self.ref_distancias, d.mean()) / len(self.ref_distancias)
            parecido = float(np.clip(1.0 - pos, 0.0, 1.0))
        else:
            parecido = float(np.clip(1.0 - d.mean() / 0.6, 0.0, 1.0))
        acuerdo = abs(2 * prob_cruda - 1.0)
        cobertura = min(len(sel) / self.k, 1.0)
        confianza = float(np.clip((parecido ** .5) * (0.35 + 0.65 * acuerdo) * cobertura, 0, 1))

        # Que habria dicho CADA grupo por su cuenta, eligiendo sus propios vecinos.
        # Hay que dejarle elegir: si se le obligan los vecinos que eligio la
        # distancia combinada, el reparto de culpas del Hedge queda sesgado.
        probs_grupo = {}
        if grupos_independientes:
            for g in self.pesos:
                if g not in dg:
                    continue
                og = np.argsort(dg[g], kind="stable")[: self.k * 12]
                sg = self._elegir_vecinos(og, cand)
                if len(sg) < 3:
                    continue
                wg = np.exp(-dg[g][sg] / max(self.temperatura, EPS))
                wg = wg / max(wg.sum(), EPS)
                probs_grupo[g] = float(np.clip((wg * (self.ret[cand[sg]] > 0)).sum(), 0.001, 0.999))

        cols_vec = {"ticker": self.tickers[gidx], "fecha": self.fechas[gidx],
                    "distancia": d, "peso": w, "ret": r,
                    "mfe": self.mfe[gidx], "mae": self.mae[gidx],
                    "patron": self.patron[gidx]}
        # distancia por grupo de cada vecino: es el "por que se parece"
        for g in dg:
            cols_vec[f"d_{g}"] = dg[g][sel]
        vecinos = pd.DataFrame(cols_vec).sort_values("distancia").reset_index(drop=True)

        return Prediccion(
            ticker=ticker, fecha=pd.Timestamp(fecha), prob_sube_cruda=prob_cruda,
            prob_sube=prob, ret_esperado=float((w * r).sum()),
            mfe_esperado=float((w * self.mfe[gidx]).sum()),
            mae_esperado=float((w * self.mae[gidx]).sum()),
            camino=camino, camino_p10=p10, camino_p90=p90,
            confianza=confianza, abstencion=confianza < self.umbral_confianza,
            distancia_media=float(d.mean()), vecinos=vecinos, probs_grupo=probs_grupo,
            atr_pct=float(banco.meta["atr_pct"].iloc[i]))

    def _sin_datos(self, ticker, fecha, banco, i):
        h = self.cfg.horizonte
        return Prediccion(ticker, pd.Timestamp(fecha), 0.5, 0.5, 0.0, 0.0, 0.0,
                          np.zeros(h), np.zeros(h), np.zeros(h), 0.0, True, float("nan"),
                          pd.DataFrame(columns=["ticker", "fecha", "distancia", "peso", "ret",
                                                "mfe", "mae", "patron"]),
                          {}, float(banco.meta["atr_pct"].iloc[i]))

    # ----------------------------------------------------- aprender del error
    def aprender_del_error(self, pred, ret_real):
        """Se le revela lo que paso de verdad y ajusta dos cosas.

        1. Los pesos de los grupos, con Hedge: el grupo que habria acertado mas
           gana peso, el que fallo lo pierde. Es la version honesta de "ahora doy
           mas importancia al volumen": se mide, no se decide a dedo.
        2. La calibracion: guarda que frecuencia real de subidas hubo cuando dijo
           un 70%, para no volver a decir 70% si la realidad era 55%.
        """
        subio = bool(ret_real > 0)
        self.calibrador.observar(pred.prob_sube_cruda, subio)

        if pred.probs_grupo:
            y = 1.0 if subio else 0.0
            perdidas = {g: (p - y) ** 2 for g, p in pred.probs_grupo.items()}
            media = float(np.mean(list(perdidas.values())))
            for g, perd in perdidas.items():
                self.pesos[g] = self.pesos.get(g, 0.0) * float(np.exp(-self.eta * (perd - media)))
            for g in self.pesos:
                self.pesos[g] = max(self.pesos[g], self.peso_minimo)
            s = sum(self.pesos.values())
            self.pesos = {g: w / s for g, w in self.pesos.items()}
            self.historial_pesos.append(dict(self.pesos, fecha=pd.Timestamp(pred.fecha)))

        acerto = (pred.prob_sube > .5) == subio
        self.historial_errores.append({
            "fecha": pd.Timestamp(pred.fecha), "ticker": pred.ticker,
            "prob": pred.prob_sube, "prob_cruda": pred.prob_sube_cruda,
            "ret_esperado": pred.ret_esperado, "ret_real": float(ret_real),
            "acerto": bool(acerto), "confianza": pred.confianza,
            "abstencion": bool(pred.abstencion),
            "brier": float((pred.prob_sube - (1.0 if subio else 0.0)) ** 2)})
        return acerto

    # ------------------------------------------------------------- persistencia
    def guardar(self, ruta):
        b = self.banco
        extra = {f"tf_{k}": v for k, v in b.tf.items()}
        np.savez_compressed(
            ruta, formas=b.formas, formas_vol=b.formas_vol, feat=b.feat,
            caminos=b.caminos, mu=self.mu, sd=self.sd,
            meta=b.meta.to_json(orient="split", date_format="iso"),
            y=b.y.to_json(orient="split"),
            estado=json.dumps({
                "pesos": self.pesos, "k": self.k, "umbral_confianza": self.umbral_confianza,
                "temperatura": self.temperatura, "separacion_dias": self.separacion_dias,
                "eta": self.eta, "peso_minimo": self.peso_minimo,
                "calibrador": self.calibrador.estado(),
                "cfg": {"largo": self.cfg.largo, "horizonte": self.cfg.horizonte,
                        "largo_semanal": self.cfg.largo_semanal,
                        "largo_mensual": self.cfg.largo_mensual},
                "tf": list(b.tf)}),
            errores=pd.DataFrame(self.historial_errores).to_json(orient="split",
                                                                 date_format="iso"),
            **extra)
        return ruta

    @classmethod
    def cargar(cls, ruta):
        d = np.load(ruta, allow_pickle=False)
        est = json.loads(str(d["estado"]))
        cfg = Config(**est["cfg"])
        m = cls(cfg, k=est["k"], pesos=est["pesos"],
                umbral_confianza=est["umbral_confianza"], temperatura=est["temperatura"],
                separacion_dias=est["separacion_dias"], eta=est["eta"],
                peso_minimo=est["peso_minimo"])
        m.calibrador = Calibrador.desde_estado(est["calibrador"])
        meta = pd.read_json(io_str(d["meta"]), orient="split")
        meta["fecha"] = pd.to_datetime(meta["fecha"])
        banco = BancoCasos(cfg, d["formas"], d["formas_vol"],
                           {k: d[f"tf_{k}"] for k in est["tf"]}, d["feat"],
                           meta, pd.read_json(io_str(d["y"]), orient="split"), d["caminos"])
        m.banco = banco
        m.mu, m.sd = d["mu"], d["sd"]
        m._reindexar()
        m._calibrar_escala()
        if "errores" in d:
            err = pd.read_json(io_str(d["errores"]), orient="split")
            m.historial_errores = err.to_dict("records")
        return m

    @staticmethod
    def existe(ruta):
        return os.path.exists(ruta)


def io_str(x):
    """pandas 3 exige un buffer, no una cadena suelta."""
    import io
    return io.StringIO(str(x))
