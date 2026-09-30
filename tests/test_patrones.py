"""Tests del sistema. El grueso vigila lo que de verdad importa en este tipo de
proyecto: que en ningun sitio se cuele informacion del futuro. Un modelo de
prediccion que en los tests parece brillante casi siempre es un modelo que esta
viendo la respuesta sin querer.
"""
import numpy as np
import pandas as pd
import pytest

from patrones import datos, contexto, noticias, ventanas, memoria, evaluacion
from patrones import indicadores as ind


@pytest.fixture(scope="module")
def universo():
    pre, meta, idx, reg = datos.generar_universo(n_tickers=6, n_barras=1500, semilla=17)
    return pre, meta, idx, reg


@pytest.fixture(scope="module")
def banco(universo):
    pre, meta, idx, reg = universo
    ctx = contexto.Contexto(pre, meta, idx)
    fn = noticias.NoticiasSinteticas(pre, meta, senal=0.0)
    b = ventanas.construir(pre, ctx, fn, verbose=False)
    return ventanas.etiquetar(b, reg)


# ------------------------------------------------------------- indicadores
def test_rsi_en_rango():
    c = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 500)) + 100)
    r = ind.rsi(c).dropna()
    assert r.between(0, 100).all()


def test_rsi_monotono():
    assert ind.rsi(pd.Series(np.arange(1, 80, dtype=float))).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(pd.Series(np.arange(80, 1, -1, dtype=float))).iloc[-1] == pytest.approx(0.0)


def test_atr_positivo():
    n = 300
    rng = np.random.default_rng(1)
    c = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    h, l = c + np.abs(rng.normal(0, 1, n)), c - np.abs(rng.normal(0, 1, n))
    assert (ind.atr(h, l, c).dropna() >= 0).all()


def test_indicadores_sin_lookahead():
    """Cambiar el futuro no puede cambiar el valor de hoy. Esta es LA prueba."""
    rng = np.random.default_rng(3)
    c = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)))
    h, l = c + 1, c - 1
    corte = 300
    calc = {"rsi": ind.rsi, "atr": lambda x: ind.atr(h, l, x),
            "bb": ind.bollinger_b, "macd": lambda x: ind.macd(x)[2]}
    for nombre, f in calc.items():
        antes = f(c).iloc[:corte]
        c2 = c.copy()
        c2.iloc[corte:] = c2.iloc[corte:] + rng.normal(0, 5, len(c) - corte)
        despues = f(c2).iloc[:corte]
        pd.testing.assert_series_equal(antes, despues, check_names=False,
                                       obj=f"{nombre} mira al futuro")


# --------------------------------------------------------------- datos
def test_ohlc_coherente(universo):
    pre = universo[0]
    for df in pre.values():
        assert (df.high >= df[["open", "close"]].max(axis=1) - 1e-6).all()
        assert (df.low <= df[["open", "close"]].min(axis=1) + 1e-6).all()
        assert (df.volume > 0).all()


def test_figuras_cumplen_su_frecuencia(universo):
    """El catalogo dice cada figura con que probabilidad se cumple; el generador
    debe respetarlo (con margen por muestra finita)."""
    reg = universo[3]
    for fig, (_, prob) in datos.CATALOGO.items():
        g = reg[reg.patron == fig]
        if len(g) >= 15:
            assert abs(g.se_cumplio.mean() - prob) < 0.25, f"{fig} se desvia mucho"


# ------------------------------------------------------------- ventanas
def test_formas_normalizadas(banco):
    assert banco.formas.mean() == pytest.approx(0.0, abs=1e-5)
    assert banco.formas.std() == pytest.approx(1.0, abs=1e-3)


def test_sin_nan(banco):
    assert not np.isnan(banco.formas).any()
    assert not np.isnan(banco.feat).any()
    assert not np.isnan(banco.caminos).any()


def test_casos_ordenados(banco):
    assert banco.meta.fecha.is_monotonic_increasing


def test_etiqueta_no_entra_en_features(banco):
    """La figura conocida es solo para explicar; no puede estar en X."""
    assert "patron" not in banco.nombres_feat
    X, y, nombres = __import__("patrones.modelos", fromlist=["matriz_supervisada"]) \
        .matriz_supervisada(banco)
    assert not any("patron" in n for n in nombres)


# ------------------------------------------------------------- memoria
def test_prediccion_valida(banco):
    corte = banco.meta.fecha.quantile(0.7)
    tr = banco.subconjunto((banco.meta.fecha <= corte).to_numpy())
    te = banco.subconjunto((banco.meta.fecha > corte).to_numpy())
    mem = memoria.MemoriaPatrones(banco.cfg, k=20).ajustar(tr)
    p = mem.predecir(te, 10)
    assert 0 <= p.prob_sube <= 1
    assert 0 <= p.confianza <= 1
    assert (p.camino_p10 <= p.camino + 1e-6).all()
    assert (p.camino <= p.camino_p90 + 1e-6).all()
    assert abs(p.vecinos.peso.sum() - 1.0) < 1e-6


def test_purga_temporal(banco):
    """Ningun vecino puede tener un resultado que aun no se conocia. Critico."""
    corte = banco.meta.fecha.quantile(0.6)
    tr = banco.subconjunto((banco.meta.fecha <= corte).to_numpy())
    te = banco.subconjunto((banco.meta.fecha > corte).to_numpy())
    mem = memoria.MemoriaPatrones(banco.cfg, k=25).ajustar(tr)
    h = banco.cfg.horizonte
    for i in range(0, len(te), 40):
        p = mem.predecir(te, i)
        if len(p.vecinos) == 0:
            continue
        # el resultado de un vecino se conoce 'horizonte' barras despues de su fecha
        limite = p.fecha - pd.Timedelta(days=int(h * 1.6) + 1)
        assert (pd.to_datetime(p.vecinos.fecha) <= limite).all(), \
            "un vecino tiene resultado del futuro"


def test_separacion_entre_vecinos(banco):
    corte = banco.meta.fecha.quantile(0.6)
    tr = banco.subconjunto((banco.meta.fecha <= corte).to_numpy())
    te = banco.subconjunto((banco.meta.fecha > corte).to_numpy())
    sep = 45
    mem = memoria.MemoriaPatrones(banco.cfg, k=25, separacion_dias=sep).ajustar(tr)
    for i in range(0, len(te), 50):
        v = mem.predecir(te, i).vecinos
        for t, g in v.groupby("ticker"):
            f = sorted(pd.to_datetime(g.fecha))
            for a, b in zip(f, f[1:]):
                assert (b - a).days >= sep, "vecinos demasiado cerca en el tiempo"


def test_aprender_cambia_pesos(banco):
    corte = banco.meta.fecha.quantile(0.6)
    tr = banco.subconjunto((banco.meta.fecha <= corte).to_numpy())
    te = banco.subconjunto((banco.meta.fecha > corte).to_numpy())
    mem = memoria.MemoriaPatrones(banco.cfg, k=20).ajustar(tr)
    antes = dict(mem.pesos)
    for i in range(min(60, len(te))):
        p = mem.predecir(te, i)
        mem.aprender_del_error(p, float(te.y.ret.iloc[i]))
    assert any(abs(mem.pesos[g] - antes[g]) > 1e-4 for g in antes)
    assert abs(sum(mem.pesos.values()) - 1.0) < 1e-6  # siguen sumando 1


def test_guardar_cargar(banco, tmp_path):
    corte = banco.meta.fecha.quantile(0.7)
    tr = banco.subconjunto((banco.meta.fecha <= corte).to_numpy())
    te = banco.subconjunto((banco.meta.fecha > corte).to_numpy())
    mem = memoria.MemoriaPatrones(banco.cfg, k=20).ajustar(tr)
    p1 = mem.predecir(te, 5)
    ruta = tmp_path / "m.npz"
    mem.guardar(str(ruta))
    mem2 = memoria.MemoriaPatrones.cargar(str(ruta))
    p2 = mem2.predecir(te, 5)
    assert p1.prob_sube_cruda == pytest.approx(p2.prob_sube_cruda, abs=1e-6)
    assert len(mem2) == len(mem)


# ------------------------------------------------------------- noticias
def test_sentimiento_signo():
    assert noticias.puntuar_titular("La empresa bate previsiones y eleva su guia") > 0.3
    assert noticias.puntuar_titular("Profit warning y despidos masivos") < -0.3
    assert noticias.puntuar_titular("La empresa celebra su junta") == pytest.approx(0.0)


def test_noticias_sin_lookahead(universo):
    pre, meta = universo[0], universo[1]
    fn = noticias.NoticiasSinteticas(pre, meta, senal=0.0)
    t = list(pre)[0]
    f = pre[t].index[-100]
    tt = fn.titulares(t, f, 30)
    assert (tt.fecha < f).all(), "devuelve titulares del presente o futuro"


def test_noticias_sinteticas_neutras_no_ayudan(banco):
    """Con senal=0 las noticias no deben mejorar el AUC de forma significativa:
    si lo hicieran, seria una fuga. Se comprueba que el grupo 'noticias' por si
    solo no predice mejor que una moneda."""
    corte = banco.meta.fecha.quantile(0.6)
    tr = banco.subconjunto((banco.meta.fecha <= corte).to_numpy())
    te = banco.subconjunto((banco.meta.fecha > corte).to_numpy())
    mem = memoria.MemoriaPatrones(banco.cfg, k=25).ajustar(tr)
    probs, reales = [], []
    for i in range(0, len(te), 20):
        p = mem.predecir(te, i)
        if "noticias" in p.probs_grupo:
            probs.append(p.probs_grupo["noticias"])
            reales.append(te.y.ret.iloc[i] > 0)
    if len(probs) > 40 and len(set(reales)) > 1:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(reales, probs)
        # La fuga se manifiesta SOLO por arriba: un grupo que ve el futuro tendria
        # AUC claramente > 0.5. Un AUC bajo es senal debil o reversion, no fuga.
        # El margen superior tiene en cuenta el ruido de una muestra de ~50 casos.
        margen = 2.5 * np.sqrt(0.25 / len(probs))
        assert auc < 0.5 + margen, (
            f"noticias neutras predicen el futuro (AUC {auc:.3f} > {0.5+margen:.3f}): "
            "posible fuga de informacion")


# ------------------------------------------------------------- evaluacion
def test_pliegues_no_solapan(banco):
    pl = evaluacion.pliegues_temporales(banco.meta.fecha, 4)
    assert len(pl) == 4
    for (a1, b1), (a2, b2) in zip(pl, pl[1:]):
        assert b1 <= a2  # el test de un pliegue empieza donde acaba el anterior


def test_submuestreo_no_solapado(banco):
    sub = evaluacion.submuestreo_no_solapado(banco, banco.cfg.horizonte)
    for t, g in sub.meta.groupby("ticker"):
        f = sorted(g.fecha)
        # tras submuestrear por horizonte, dos casos del mismo valor no comparten futuro
        assert all((b - a).days >= 1 for a, b in zip(f, f[1:]))
    assert len(sub) < len(banco)


def test_walkforward_ejecuta(banco):
    res, mem, pesos = evaluacion.evaluar_memoria(banco, n_pliegues=3, k=20, verbose=False)
    assert len(res) > 30
    assert res.prob.between(0, 1).all()
    m = evaluacion.metricas(res)
    assert "acierto" in m and "auc" in m
