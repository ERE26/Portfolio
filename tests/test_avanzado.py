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
    res, imp = evaluar_meta(b, n_pliegues=3, k=25, verbose=False)
    alto = res[res["prob_meta"] >= res["prob_meta"].quantile(0.67)]
    bajo = res[res["prob_meta"] <= res["prob_meta"].quantile(0.33)]
    assert len(alto) > 15 and len(bajo) > 15
    # el grupo que el meta considera fiable acierta al menos tanto como el dudoso
    assert alto["acierto_primario"].mean() >= bajo["acierto_primario"].mean() - 0.02


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
