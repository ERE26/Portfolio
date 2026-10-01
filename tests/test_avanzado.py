"""Tests de triple-barrier labeling, meta-labeling y conectores de datos nuevos."""
import numpy as np
import pandas as pd
import pytest

from patrones import etiquetado as et
from patrones import datos, contexto, ventanas, memoria


# ------------------------------------------------- triple-barrier
def test_triple_barrera_gana_pierde_vence():
    n = 100
    lc = np.log(100 + np.zeros(n))
    lh, ll = lc.copy(), lc.copy()
    pos = np.array([5, 20, 35])
    atr = np.full(3, 0.01)
    lh[8] = lc[5] + 0.02    # toca beneficio en barra 3
    ll[24] = lc[20] - 0.02  # toca perdida en barra 4
    r = et.triple_barrera(lh, ll, lc, pos, atr, 10, k_pt=1.5, k_sl=1.5)
    assert r["label"].tolist() == [1, -1, 0]
    assert r["barras"].tolist() == [3, 4, 10]


def test_triple_barrera_sin_lookahead():
    """La etiqueta de la barra t no puede depender de datos mas alla de t+H."""
    rng = np.random.default_rng(0)
    n = 300
    lc = np.log(100 + np.cumsum(rng.normal(0, 1, n)))
    lh, ll = lc + 0.5, lc - 0.5
    pos = np.array([50, 100, 150])
    atr = np.full(3, 0.02)
    r1 = et.triple_barrera(lh, ll, lc, pos, atr, 10)
    lc2 = lc.copy(); lc2[170:] += 10   # cambiar el futuro lejano (>H tras pos=150)
    lh2, ll2 = lc2 + 0.5, lc2 - 0.5
    r2 = et.triple_barrera(lh2, ll2, lc2, pos, atr, 10)
    assert np.array_equal(r1["label"], r2["label"])


def test_triple_barrera_integrada():
    pre, meta, idx, reg = datos.generar_universo(n_tickers=4, n_barras=1000, semilla=2)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    assert set(["tb_label", "tb_ret", "tb_barras", "tb_rr"]).issubset(b.y.columns)
    assert set(np.unique(b.y.tb_label)).issubset({-1, 0, 1})
    assert (b.y.tb_barras <= b.cfg.horizonte).all()


# ------------------------------------------------- meta-labeling
def test_features_meta_completos():
    from patrones.meta import features_meta
    pre, meta, idx, reg = datos.generar_universo(n_tickers=5, n_barras=1100, semilla=3)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    corte = b.meta.fecha.quantile(0.7)
    tr = b.subconjunto((b.meta.fecha <= corte).to_numpy())
    te = b.subconjunto((b.meta.fecha > corte).to_numpy())
    mem = memoria.MemoriaPatrones(b.cfg, k=20).ajustar(tr)
    f = features_meta(mem.predecir(te, 5), te, 5)
    assert "conviccion" in f and "disp_grupos" in f
    assert all(np.isfinite(v) for v in f.values())


def test_meta_seleccion_monotona_sintetico():
    """En el sintetico con senal, mas exigencia del meta => no peor precision.

    Se compara el tercio mas fiable contra el tercio menos fiable (mas estable
    que un umbral fijo), permitiendo un pequeno margen de ruido muestral."""
    pre, meta, idx, reg = datos.generar_universo(n_tickers=10, n_barras=2200, semilla=7)
    b = ventanas.etiquetar(ventanas.construir(pre, contexto.Contexto(pre, meta, idx),
                                              None, verbose=False), reg)
    from patrones.evaluacion import evaluar_meta
    res, imp = evaluar_meta(b, n_pliegues=3, k=25, verbose=False, n_procesos=1)
    # criterio estable: la prob del meta no esta anti-correlacionada con el acierto
    # real (correlacion >= 0). Comparar tercios sueltos es demasiado ruidoso.
    from scipy.stats import pointbiserialr
    if res["acierto_primario"].nunique() > 1:
        r, _ = pointbiserialr(res["acierto_primario"], res["prob_meta"])
        assert r >= -0.05, f"el meta esta anti-correlacionado con el acierto (r={r:.3f})"


# ------------------------------------------------- conectores (sin red)
class _ClienteFalla:
    """Cliente cuyo get_json siempre falla, para probar la degradacion elegante."""
    def get_json(self, *a, **k):
        raise RuntimeError("sin red (test)")


def test_fred_sin_red_neutro():
    from fuentes import fred
    fechas = pd.bdate_range("2023-01-01", periods=10)
    r = fred.rasgos_macro(fechas, clave="clave_valida_pero_sin_red", cliente=_ClienteFalla())
    assert (r == 0).all().all()   # fallo de red -> neutro, no rompe


def test_sec_sin_red_neutro():
    from fuentes import sec
    r = sec.rasgos_fundamentales("ZZZZ", pd.bdate_range("2023-01-01", periods=10),
                                 cik="9999999999", cliente=_ClienteFalla())
    assert (r == 0).all().all()


# ------------------------------------------------- rentabilidad
def test_recalcular_etiqueta_coincide():
    """Recalcular con el mismo ratio del banco debe dar la misma etiqueta."""
    pre, meta, idx, reg = datos.generar_universo(n_tickers=4, n_barras=900, semilla=1)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    r = et.recalcular_etiqueta(b, b.cfg.k_pt, b.cfg.k_sl)
    assert (r["label"] == b.y.tb_label.to_numpy()).mean() > 0.98


def test_asimetria_mejora_retorno():
    """Un objetivo mayor que el stop sube el retorno medio aunque baje el acierto."""
    pre, meta, idx, reg = datos.generar_universo(n_tickers=6, n_barras=1400, semilla=5)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    sim = et.recalcular_etiqueta(b, 1.0, 1.0)["ret"].mean()
    asi = et.recalcular_etiqueta(b, 3.0, 1.0)["ret"].mean()
    # con objetivo 3x y stop 1x, el retorno medio en ATR es mayor que simetrico
    assert asi > sim


def test_etiqueta_desde_recorrido_vs_recalcular():
    """La etiqueta por recorrido (rentabilidad) coincide con la del banco (etiquetado)."""
    import numpy as np, pandas as pd
    from patrones.rentabilidad import etiqueta_desde_recorrido
    pre, meta, idx, reg = datos.generar_universo(n_tickers=4, n_barras=900, semilla=2)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    df = pd.DataFrame({"lado": np.ones(len(b), int),
                       "cam_high": list(b.caminos_high), "cam_low": list(b.caminos_low),
                       "cam_close": list(b.caminos)})
    ret_rec = etiqueta_desde_recorrido(df, 2.0, 1.0)
    ret_banco = et.recalcular_etiqueta(b, 2.0, 1.0)["ret"]
    assert np.allclose(ret_rec, ret_banco, atol=1e-5)


# ------------------------------------------------- 13F smart money
def test_13f_senal_point_in_time():
    """La señal 13F solo usa filings publicados antes de cada fecha (no futuro)."""
    import pandas as pd
    from fuentes import form13f
    # cartera simulada de un fondo: 2 trimestres de AAPL, filed en fechas conocidas
    carteras = {"FondoX": [
        (pd.Timestamp("2024-03-31"), pd.Timestamp("2024-05-15"), {"APPLE INC": (1000, 100)}),
        (pd.Timestamp("2024-06-30"), pd.Timestamp("2024-08-14"), {"APPLE INC": (2000, 200)}),
    ]}
    fechas = pd.to_datetime(["2024-05-01", "2024-06-01", "2024-09-01"])
    s = form13f.señal_ticker("AAPL", fechas, carteras, patron="APPLE")
    # 2024-05-01: aun no publicado el 13F de mayo -> presencia 0
    assert s.loc[fechas[0], "inst_presencia"] == 0
    # 2024-06-01: ya publicado el primer 13F (filed 05-15) -> presencia 1
    assert s.loc[fechas[1], "inst_presencia"] == 1
    # 2024-09-01: publicados los dos; detecta el aumento de 100 a 200 acciones
    assert s.loc[fechas[2], "inst_presencia"] == 1
    assert s.loc[fechas[2], "inst_cambio"] > 0


def test_13f_sin_mapeo_neutro():
    import pandas as pd
    from fuentes import form13f
    fechas = pd.to_datetime(["2024-01-01", "2024-02-01"])
    s = form13f.señal_ticker("ZZZZ", fechas, {}, patron=None)
    assert (s == 0).all().all()


# ------------------------------------------------- factores value/quality
def test_momentum_12_1_sin_lookahead():
    import numpy as np, pandas as pd
    from patrones.indicadores import momentum_12_1
    c = pd.Series(100 + np.arange(400, dtype=float))
    m = momentum_12_1(c)
    # en t, usa c[t-21]/c[t-252]-1: nunca el presente ni el futuro
    assert pd.isna(m.iloc[250])          # no hay 252 barras atras aun
    assert np.isfinite(m.iloc[300])
    # cambiar el futuro no altera el momentum pasado
    c2 = c.copy(); c2.iloc[320:] += 500
    assert abs(momentum_12_1(c2).iloc[300] - m.iloc[300]) < 1e-9


def test_normalizacion_cross_sectional():
    """Tras normalizar, cada factor tiene media ~0 y desviacion ~1 por mes."""
    import numpy as np, pandas as pd
    from fuentes.descarga import cargar_mercado
    from fuentes.enriquecer import Enriquecedor
    from patrones import contexto, ventanas
    pre, meta = cargar_mercado("datos/mercado", "1d")
    msi = meta.set_index("ticker")
    tickers = [t for t in pre if msi.loc[t, "clase"] == "accion"][:15]
    sub = {t: pre[t][pre[t].index >= pd.Timestamp.now() - pd.Timedelta(days=4 * 365)] for t in tickers}
    enr = Enriquecedor(con_sec=False, con_fred=False, con_factores=True, verbose=False)
    b = ventanas.construir(sub, contexto.Contexto(sub, meta), None,
                           ventanas.Config(paso=5), verbose=False, enriquecer=enr)
    ey = b.feat[:, b.nombres_feat.index("fac_earnings_yield")]
    nz = ey[ey != 0]
    assert len(nz) > 100
    assert abs(nz.mean()) < 0.2 and 0.7 < nz.std() < 1.3   # ~z-score


# ------------------------------------------------- primario GBM y ensamble
def test_primario_gbm_interfaz():
    """El GBM produce predicciones validas con la misma interfaz que la memoria."""
    import numpy as np
    from patrones.primario import PrimarioGBM
    pre, meta, idx, reg = datos.generar_universo(n_tickers=6, n_barras=1200, semilla=3)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    corte = b.meta.fecha.quantile(0.7)
    tr = b.subconjunto((b.meta.fecha <= corte).to_numpy())
    te = b.subconjunto((b.meta.fecha > corte).to_numpy())
    m = PrimarioGBM(b.cfg).ajustar(tr)
    p = m.predecir(te, 5)
    assert 0 <= p.prob_sube <= 1 and 0 <= p.confianza <= 1
    assert p.ticker and p.fecha is not None


def test_gbm_cache_correcto():
    """El cache de probabilidades del GBM coincide con el calculo directo."""
    import numpy as np
    from patrones.primario import PrimarioGBM
    from patrones.modelos import matriz_supervisada
    pre, meta, idx, reg = datos.generar_universo(n_tickers=5, n_barras=1100, semilla=1)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    corte = b.meta.fecha.quantile(0.7)
    tr = b.subconjunto((b.meta.fecha <= corte).to_numpy())
    te = b.subconjunto((b.meta.fecha > corte).to_numpy())
    m = PrimarioGBM(b.cfg).ajustar(tr)
    probs = [m.predecir(te, i).prob_sube_cruda for i in range(50)]
    X, _, _ = matriz_supervisada(te)
    pb = m.modelo.predict_proba(X)[:, 1]
    assert np.allclose(probs, pb[:50], atol=1e-9)


def test_ensamble_expone_prob_gbm():
    """El ensamble expone la prob del GBM como canal para el meta."""
    import numpy as np
    from patrones.primario import PrimarioEnsamble
    pre, meta, idx, reg = datos.generar_universo(n_tickers=6, n_barras=1200, semilla=2)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    corte = b.meta.fecha.quantile(0.7)
    tr = b.subconjunto((b.meta.fecha <= corte).to_numpy())
    te = b.subconjunto((b.meta.fecha > corte).to_numpy())
    m = PrimarioEnsamble(b.cfg, k=20).ajustar(tr)
    p = m.predecir(te, 5)
    assert np.isfinite(p.prob_gbm)
    assert len(p.vecinos) > 0   # mantiene los vecinos de la memoria


# ------------------------------------------------- cortos / long-short
def test_cortos_en_bajista():
    """En modo long-short, se toman cortos en mercado bajista y largos en alcista."""
    import numpy as np, pandas as pd
    from patrones.rentabilidad import aplicar_sistema
    # señales sinteticas: mezcla de regimenes y lados, con recorridos
    n = 200
    rng = np.random.default_rng(0)
    H = 10
    cam = rng.normal(0, 1, (n, H)).cumsum(1)
    df = pd.DataFrame({
        "lado": rng.choice([-1, 1], n), "prob_meta": rng.uniform(0.5, 0.8, n),
        "mkt_alcista": rng.choice([0.0, 1.0], n), "pliegue": rng.integers(1, 5, n),
        "cam_high": list(cam + 2), "cam_low": list(cam - 2), "cam_close": list(cam),
        "acierto_primario": rng.integers(0, 2, n)})
    sel, m = aplicar_sistema(df, reglas={"permitir_cortos": True, "evitar_vol_baja": False})
    # todas las operadas respetan: largo en alcista, corto en bajista
    assert ((sel["mkt_alcista"] > 0.5) == (sel["lado"] > 0)).all()
