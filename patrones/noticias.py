"""Canal de noticias: sentimiento de titulares con decaimiento temporal.

Aviso importante y que conviene dejar escrito en la memoria del trabajo: este
es el componente mas debil del sistema. Para entrenar hace falta un historico
de titulares con marca de tiempo fiable (point-in-time). Si usas titulares
recientes descargados hoy para etiquetar barras de hace tres anos, estas
metiendo informacion del futuro y los resultados saldran buenos por un motivo
que no existe. Por eso aqui el sentimiento SIEMPRE se calcula con titulares
estrictamente anteriores a la barra que se esta evaluando.
"""
import math
import re

import numpy as np
import pandas as pd

# Lexico financiero minimo, espanol e ingles. Pesos en [-1, 1].
LEXICO = {
    # negativo
    "caida": -.7, "caidas": -.7, "cae": -.6, "desploma": -1.0, "desplome": -1.0,
    "hundimiento": -.9, "perdidas": -.8, "perdida": -.7, "recorta": -.5, "recorte": -.5,
    "despidos": -.7, "quiebra": -1.0, "concurso": -.9, "fraude": -1.0, "investigacion": -.6,
    "multa": -.7, "demanda": -.6, "rebaja": -.7, "profit warning": -1.0, "deuda": -.4,
    "dimite": -.5, "dimision": -.5, "advertencia": -.6, "debilidad": -.5, "riesgo": -.4,
    "decepciona": -.8, "incumple": -.8, "retraso": -.5, "huelga": -.6, "sancion": -.7,
    "fall": -.6, "falls": -.6, "plunge": -1.0, "slump": -.8, "loss": -.8, "losses": -.8,
    "downgrade": -.8, "layoffs": -.7, "bankruptcy": -1.0, "probe": -.6, "lawsuit": -.6,
    "miss": -.7, "misses": -.7, "warning": -.6, "cut": -.5, "weak": -.6, "bearish": -.7,
    # positivo
    "sube": .6, "subida": .6, "suben": .6, "dispara": 1.0, "maximos": .7, "record": .8,
    "beneficios": .7, "beneficio": .6, "ganancias": .7, "crece": .6, "crecimiento": .6,
    "mejora": .7, "supera": .8, "bate": .9, "dividendo": .5, "recompra": .7,
    "adquisicion": .5, "contrato": .6, "alianza": .5, "aprobacion": .7, "expansion": .6,
    "optimista": .6, "fortaleza": .6, "recuperacion": .6, "eleva": .6, "alcista": .7,
    "rise": .6, "rises": .6, "surge": 1.0, "jump": .9, "profit": .7, "profits": .7,
    "beat": .9, "beats": .9, "upgrade": .8, "growth": .6, "strong": .6, "rally": .8,
    "bullish": .7, "approval": .7, "buyback": .7, "raises": .6, "outperform": .8,
}
# Entradas de varias palabras: hay que buscarlas ANTES de partir en tokens, porque
# si no "profit warning" se lee como "profit" (+) y "warning" (-) y se anulan.
MULTIPALABRA = {
    "profit warning": -1.0, "guerra comercial": -.7, "recorte de plantilla": -.7,
    "maximos historicos": .9, "suspension de pagos": -1.0, "opa": .6,
}
LEXICO.update(MULTIPALABRA)

UNIPALABRA = {k: v for k, v in LEXICO.items() if " " not in k}

NEGACIONES = {"no", "sin", "nunca", "tampoco", "not", "without", "fails", "falla", "descarta"}
INTENSIFICADORES = {"muy": 1.4, "fuerte": 1.3, "fuertemente": 1.4, "historico": 1.3,
                    "very": 1.4, "sharply": 1.4, "strongly": 1.3, "record": 1.3}


def _normalizar(texto):
    t = str(texto).lower()
    for a, b in zip("áéíóúü", "aeiouu"):
        t = t.replace(a, b)
    return t


def puntuar_titular(texto):
    """Devuelve el sentimiento del titular en [-1, 1]. 0 = neutro o sin palabras conocidas."""
    t = _normalizar(texto)
    total, n = 0.0, 0
    for frase, peso in MULTIPALABRA.items():
        veces = t.count(frase)
        if veces:
            total += peso * veces
            n += veces
            t = t.replace(frase, " ")
    tok = re.findall(r"[a-zn]+", t)
    if not tok and n == 0:
        return 0.0
    for i, w in enumerate(tok):
        peso = UNIPALABRA.get(w)
        if peso is None:
            continue
        mult = 1.0
        for j in range(max(0, i - 3), i):
            if tok[j] in NEGACIONES:
                mult *= -1.0
            mult *= INTENSIFICADORES.get(tok[j], 1.0)
        total += peso * mult
        n += 1
    if n == 0:
        return 0.0
    return float(np.clip(total / n, -1.0, 1.0))


class FuenteNoticias:
    """Interfaz. Debe devolver SOLO titulares anteriores a 'hasta'."""

    def titulares(self, ticker, hasta, dias=30):
        raise NotImplementedError


class SinNoticias(FuenteNoticias):
    def titulares(self, ticker, hasta, dias=30):
        return pd.DataFrame(columns=["fecha", "titular", "puntuacion"])


class NoticiasCSV(FuenteNoticias):
    """CSV con columnas fecha, ticker, titular. Es la via recomendada para datos reales."""

    def __init__(self, ruta):
        df = pd.read_csv(ruta)
        df.columns = [c.strip().lower() for c in df.columns]
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
        df = df.dropna(subset=["fecha", "titular"])
        df["puntuacion"] = df["titular"].map(puntuar_titular)
        self.df = df.sort_values("fecha").reset_index(drop=True)

    def titulares(self, ticker, hasta, dias=30):
        d = self.df
        m = (d["ticker"].astype(str) == str(ticker)) & (d["fecha"] < hasta) & \
            (d["fecha"] >= hasta - pd.Timedelta(days=dias))
        return d.loc[m, ["fecha", "titular", "puntuacion"]]


class NoticiasSinteticas(FuenteNoticias):
    """Titulares de juguete para probar la tuberia sin depender de ninguna API.

    Por defecto se construyen SOLO con el pasado del precio (senal=0.0), asi que
    no inflan los resultados: no saben nada que el grafico no sepa ya. Subiendo
    'senal' se inyecta a proposito una ventaja predictiva conocida, lo que sirve
    para el experimento de ablacion "cuanto aportaria un canal de noticias que
    de verdad anticipase algo".
    """
    PLANTILLAS_POS = ["{n} bate previsiones y eleva su guia", "{n} firma un contrato historico",
                      "Los analistas mejoran su recomendacion sobre {n}",
                      "{n} anuncia recompra de acciones", "{n} marca maximos de 52 semanas"]
    PLANTILLAS_NEG = ["{n} decepciona con sus resultados trimestrales", "{n} recorta su dividendo",
                      "Una investigacion sobre {n} golpea al valor",
                      "{n} lanza un profit warning", "{n} anuncia despidos por debilidad de la demanda"]
    PLANTILLAS_NEU = ["{n} celebra su junta de accionistas", "{n} participa en una conferencia del sector",
                      "Cambios en el consejo de {n}", "{n} refinancia deuda a vencimiento"]

    def __init__(self, precios, meta=None, semilla=11, por_mes=4, senal=0.0, horizonte=10):
        self.precios = precios
        self.senal = float(senal)
        self.horizonte = int(horizonte)
        nombres = {}
        if meta is not None and "ticker" in getattr(meta, "columns", []):
            nombres = dict(zip(meta["ticker"], meta.get("nombre", meta["ticker"])))
        rng = np.random.default_rng(semilla)
        filas = []
        for t, df in precios.items():
            c = df["close"]
            r_pasado = np.log(c / c.shift(5)).fillna(0.0)
            r_futuro = np.log(c.shift(-self.horizonte) / c).fillna(0.0)
            paso = max(1, int(21 / max(por_mes, 1)))
            for i in range(20, len(c) - 1, paso):
                tono = float(np.tanh(r_pasado.iloc[i] * 12)) * (1 - self.senal) \
                    + float(np.sign(r_futuro.iloc[i])) * self.senal
                tono += rng.normal(0, 0.55)
                base = self.PLANTILLAS_POS if tono > .35 else \
                    self.PLANTILLAS_NEG if tono < -.35 else self.PLANTILLAS_NEU
                titular = base[rng.integers(len(base))].format(n=nombres.get(t, t))
                filas.append({"fecha": c.index[i], "ticker": t, "titular": titular,
                              "puntuacion": puntuar_titular(titular)})
        self.df = pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)

    def titulares(self, ticker, hasta, dias=30):
        d = self.df
        m = (d["ticker"] == ticker) & (d["fecha"] < hasta) & (d["fecha"] >= hasta - pd.Timedelta(days=dias))
        return d.loc[m, ["fecha", "titular", "puntuacion"]]


def rasgos_sentimiento(fuente, ticker, fechas, dias=30, semivida=7.0):
    """Serie de sentimiento con decaimiento exponencial, para cada fecha del calendario.

    Devuelve tres columnas: sentimiento medio ponderado, cuantos titulares habia
    y cuanto se contradicen entre si (dispersion). La dispersion importa: diez
    titulares que dicen cosas opuestas no son lo mismo que diez de acuerdo.
    """
    cols = ["noticias_sentimiento", "noticias_n", "noticias_dispersion"]
    if isinstance(fuente, SinNoticias):
        return pd.DataFrame({c: 0.0 for c in cols}, index=fechas)

    # Via rapida: si la fuente tiene la tabla entera en memoria, se ordena una vez
    # y se corta con searchsorted. La via generica hacia una mascara de pandas
    # sobre toda la tabla por cada fecha, que es cuadratico y no hace falta.
    tabla = getattr(fuente, "df", None)
    if tabla is not None and len(tabla):
        sub = tabla[tabla["ticker"].astype(str) == str(ticker)].sort_values("fecha")
        f_not = sub["fecha"].to_numpy("datetime64[ns]")
        p_not = sub["puntuacion"].to_numpy(float)
        objetivo = pd.DatetimeIndex(fechas).to_numpy("datetime64[ns]")
        desde = objetivo - np.timedelta64(int(dias), "D")
        ini = np.searchsorted(f_not, desde, side="left")
        fin = np.searchsorted(f_not, objetivo, side="left")   # estrictamente anteriores
        filas = np.zeros((len(objetivo), 3))
        for j, (a, b, ref) in enumerate(zip(ini, fin, objetivo)):
            if b <= a:
                continue
            edad = (ref - f_not[a:b]) / np.timedelta64(1, "D")
            w = np.exp(-np.log(2) * edad / semivida)
            pp = p_not[a:b]
            sw = w.sum()
            if sw <= 0:
                continue
            media = float((pp * w).sum() / sw)
            disp = float(np.sqrt(max((w * (pp - media) ** 2).sum() / sw, 0.0)))
            filas[j] = (media, float(b - a), disp)
        return pd.DataFrame(filas, index=fechas, columns=cols)

    filas = []
    for f in fechas:
        t = fuente.titulares(ticker, f, dias)
        if len(t) == 0:
            filas.append((0.0, 0.0, 0.0))
            continue
        edad = (f - t["fecha"]).dt.total_seconds() / 86400.0
        w = np.exp(-np.log(2) * edad / semivida)
        p = t["puntuacion"].to_numpy(float)
        sw = w.sum()
        media = float((p * w).sum() / sw) if sw > 0 else 0.0
        disp = float(np.sqrt(max((w * (p - media) ** 2).sum() / sw, 0.0))) if sw > 0 else 0.0
        filas.append((media, float(len(t)), disp))
    return pd.DataFrame(filas, index=fechas, columns=cols)
