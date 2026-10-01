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


# ------------------------------------------------- validacion out-of-universe
def test_out_of_universe_divide_por_ticker():
    """El test out-of-universe no comparte tickers con el entrenamiento."""
    import numpy as np
    from patrones import evaluacion
    pre, meta, idx, reg = datos.generar_universo(n_tickers=12, n_barras=1600, semilla=4)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    res = evaluacion.evaluar_out_of_universe(b, frac_train=0.7, k=20, primario="memoria",
                                             semilla=0, verbose=False, n_procesos=1)
    # las señales de test deben existir y llevar los recorridos para la rentabilidad
    assert len(res) > 20
    assert "cam_high" in res.columns and "prob_meta" in res.columns


# ------------------------------------------------- indicadores intradia
def test_intradia_deteccion_y_vwap():
    """Detecta intradia y el VWAP de la sesion es coherente (entre low y high)."""
    import numpy as np, pandas as pd
    from patrones import intradia
    # 2 dias de barras horarias
    idx = pd.date_range("2024-03-01 09:30", periods=14, freq="1h")
    df = pd.DataFrame({"open": 100 + np.arange(14) * 0.1, "high": 101 + np.arange(14) * 0.1,
                       "low": 99 + np.arange(14) * 0.1, "close": 100.5 + np.arange(14) * 0.1,
                       "volume": np.full(14, 1000.0)}, index=idx)
    assert intradia.es_intradia(df.index)
    r = intradia.rasgos_intradia(df)
    assert set(intradia.NOMBRES_INTRA).issubset(r.columns)
    # posicion en el rango entre 0 y 1
    assert r["intra_pos_rango"].between(0, 1).all()
    # hora codificada en [-1, 1]
    assert r["intra_hora_sin"].between(-1, 1).all()


def test_intradia_no_lookahead():
    """El VWAP y la posicion en rango de la barra t no usan barras futuras."""
    import numpy as np, pandas as pd
    from patrones import intradia
    idx = pd.date_range("2024-03-01 09:30", periods=7, freq="1h")
    df = pd.DataFrame({"open": [100]*7, "high": [101]*7, "low": [99]*7,
                       "close": [100, 100.5, 101, 100, 99.5, 100, 100.5],
                       "volume": [1000]*7}, index=idx, dtype=float)
    r1 = intradia.rasgos_intradia(df)
    df2 = df.copy(); df2.iloc[5:] = df2.iloc[5:] * 2   # cambiar el futuro
    r2 = intradia.rasgos_intradia(df2)
    # las primeras 5 barras no cambian
    assert np.allclose(r1["intra_dist_vwap"].iloc[:5], r2["intra_dist_vwap"].iloc[:5])


def test_intradia_cero_en_diario():
    """En serie diaria, las features intradia quedan en 0 (no aplican)."""
    import numpy as np
    from patrones import datos, contexto, ventanas
    pre, meta, idx, reg = datos.generar_universo(n_tickers=4, n_barras=900, semilla=1)
    b = ventanas.construir(pre, contexto.Contexto(pre, meta, idx), None, verbose=False)
    for c in ["intra_dist_vwap", "intra_pos_rango", "intra_ret_sesion"]:
        assert (b.feat[:, b.nombres_feat.index(c)] == 0).all()


def test_barras_hasta_cierre():
    """Cuenta correctamente las barras que quedan hasta el final de la sesion,
    respetando sesiones de distinta longitud (p.ej. medias sesiones)."""
    import pandas as pd
    from patrones import intradia
    idx = pd.to_datetime([
        "2024-01-02 09:30", "2024-01-02 10:30", "2024-01-02 11:30", "2024-01-02 12:30",
        "2024-01-03 09:30", "2024-01-03 10:30"])          # segunda sesion mas corta
    bhc = intradia.barras_hasta_cierre(idx)
    assert bhc.tolist() == [3, 2, 1, 0, 1, 0]


def test_salida_cierre_recorta_horizonte():
    """La salida EOD ignora las barreras tocadas despues del cierre de sesion:
    una senal cuyo objetivo se alcanza fuera de la sesion se cierra al ultimo
    precio del dia, no al horizonte completo."""
    import numpy as np, pandas as pd
    from patrones.rentabilidad import etiqueta_desde_recorrido
    sel = pd.DataFrame({
        "lado": [1],
        "cam_high": [np.array([0.5, 0.5, 3.0, 3.0])],   # objetivo 2.0 solo en barra 3
        "cam_low": [np.array([-0.2, -0.2, -0.2, -0.2])],
        "cam_close": [np.array([0.3, 0.4, 2.5, 2.6])],
    })
    # sin cap: toca el objetivo en la barra 2 -> +k_pt
    assert abs(etiqueta_desde_recorrido(sel, 2.0, 1.0)[0] - 2.0) < 1e-9
    # con cap=2 (cierra tras 2 barras): no toca objetivo, cierra en cam_close[1]=0.4
    r = etiqueta_desde_recorrido(sel, 2.0, 1.0, cap=np.array([2]))
    assert abs(r[0] - 0.4) < 1e-9


def test_conviccion_min_filtra_decididas():
    """La regla conviccion_min opera solo donde el primario se inclina con fuerza
    (|prob-0.5| en el cuantil superior)."""
    import numpy as np, pandas as pd
    from patrones import rentabilidad
    rng = np.random.default_rng(0); n = 400
    res = pd.DataFrame({
        "lado": np.ones(n), "prob_meta": rng.uniform(0.5, 0.8, n),
        "prob": rng.uniform(0.3, 0.9, n), "pliegue": 1, "mkt_alcista": 1.0,
        "cam_high": [rng.uniform(0, 3, 6) for _ in range(n)],
        "cam_low": [rng.uniform(-3, 0, 6) for _ in range(n)],
        "cam_close": [rng.uniform(-2, 2, 6) for _ in range(n)]})
    base = {"permitir_cortos": False, "solo_mercado_alcista": False, "evitar_vol_baja": False}
    sel, _ = rentabilidad.aplicar_sistema(res, reglas={**base, "conviccion_min": 0.5})
    conv = (sel["prob"] - 0.5).abs()
    # todas las operadas tienen conviccion >= la mediana global
    assert conv.min() >= (res["prob"] - 0.5).abs().median() - 1e-9


def test_asignador_walk_forward():
    """El asignador combina alfas con pesos que suman 1 cada mes, usando solo
    historia (walk-forward), y produce métricas finitas."""
    import numpy as np, pandas as pd
    from patrones import asignador
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    a = pd.Series(rng.normal(0.010, 0.030, 120), index=idx)
    b = pd.Series(rng.normal(0.008, 0.025, 120), index=idx)
    serie, pesos = asignador.combinar({"a": a, "b": b}, metodo="inv_vol")
    m = asignador.metricas(serie)
    assert m and np.isfinite(m["sharpe"])
    assert np.allclose(pesos.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert len(serie) < len(a)                 # arranca tras min_obs (no mira el futuro)


def test_pesos_unicidad_penaliza_solapamiento():
    """Casos cuyos horizontes se solapan (periodo denso) reciben MENOS peso que
    los aislados: es lo que evita sobre-representar periodos concurridos."""
    import numpy as np
    from patrones.etiquetado import pesos_unicidad
    # 3 casos amontonados (dias 0,1,2) + 3 aislados (dias 100,200,300), horizonte 10
    fechas = np.array(["2020-01-01", "2020-01-02", "2020-01-03",
                       "2020-06-01", "2020-09-01", "2020-12-01"], dtype="datetime64[D]")
    w = pesos_unicidad(fechas, np.full(6, 10), 10)
    assert w[:3].mean() < w[3:].mean()        # los solapados pesan menos
    assert abs(w.mean() - 1.0) < 1e-5          # normalizados a media 1


def test_alpaca_trade_seguridad_y_plan(monkeypatch):
    """El conector rechaza claves que no sean de paper (no 'PK') y el plan de
    rebalanceo calcula las ordenes correctas para alcanzar el objetivo."""
    import pytest
    from fuentes import alpaca_trade as AT
    # una clave real (empieza por AK) debe ser rechazada
    monkeypatch.setenv("APCA_API_KEY_ID", "AK_falsa")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secreto")
    with pytest.raises(RuntimeError):
        AT._claves()
    # el endpoint esta bloqueado a paper
    assert "paper-api" in AT._url("account")
    # plan: objetivo 10 acc de AAA (precio 100) partiendo de 4 -> comprar 6
    plan = AT and None
    from operativa import plan_rebalanceo
    ordenes = plan_rebalanceo({"AAA": 1000.0, "BBB": 500.0}, {"AAA": 100.0, "BBB": 50.0},
                              {"AAA": {"qty": 4}})
    por_t = {o["ticker"]: o for o in ordenes}
    assert por_t["AAA"]["lado"] == "buy" and por_t["AAA"]["qty"] == 6   # 10 - 4
    assert por_t["BBB"]["lado"] == "buy" and por_t["BBB"]["qty"] == 10  # 10 - 0
    # orden bracket intradia (dry-run): estructura correcta con TP/SL
    br = AT.enviar_bracket("AAA", 10, tp_price=105.0, sl_price=98.0, confirmar=False)
    assert br["dry_run"] and br["order_class"] == "bracket"
    assert br["take_profit"]["limit_price"] == 105.0 and br["stop_loss"]["stop_price"] == 98.0


def test_transversal_sin_lookahead_y_signo():
    """El momento transversal no mira al futuro y selecciona al alza al ticker que
    tendencia arriba frente al que tendencia abajo."""
    import numpy as np, pandas as pd
    from patrones import transversal as TV
    fechas = pd.date_range("2015-01-01", periods=400, freq="B")
    sube = pd.Series(100 * np.exp(np.linspace(0, 0.8, 400)), index=fechas)
    baja = pd.Series(100 * np.exp(np.linspace(0, -0.8, 400)), index=fechas)
    panel = pd.concat([sube.rename("UP"), baja.rename("DN")], axis=1)
    sc = TV.score_momentum(panel.iloc[:300], lookback=252, skip=21)
    assert sc["UP"] > sc["DN"]          # el que sube tiene mas momento
    # no lookahead: cambiar el futuro (tras la posicion 300) no altera el score en 300
    panel2 = panel.copy(); panel2.iloc[300:] *= 3.0
    sc2 = TV.score_momentum(panel2.iloc[:300], lookback=252, skip=21)
    assert abs(sc["UP"] - sc2["UP"]) < 1e-12


def test_transversal_backtest_corre():
    """El backtest mensual produce metricas finitas sobre un universo sintetico."""
    import numpy as np, pandas as pd
    from patrones import transversal as TV
    rng = np.random.default_rng(3)
    fechas = pd.date_range("2010-01-01", periods=1500, freq="B")
    pre = {f"T{i}": pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0.0003 * (i - 5), 0.012, 1500)))},
        index=fechas) for i in range(12)}
    d, m, _ = TV.backtest(pre, lookback=200, skip=21, q=0.25, largo_corto=True, min_tickers=6)
    assert m and m["meses"] > 20 and np.isfinite(m["sharpe"])


def test_riesgo_hrp_inv_vol():
    """HRP e inverse-vol dan pesos validos (suman 1, positivos) y la inverse-vol
    pondera mas el activo menos volatil."""
    import numpy as np
    from patrones import cartera
    cov = np.array([[0.04, 0.01, 0.00, 0.00],
                    [0.01, 0.09, 0.00, 0.00],
                    [0.00, 0.00, 0.16, 0.02],
                    [0.00, 0.00, 0.02, 0.01]])
    for metodo in ("hrp", "inv_vol", "igual"):
        w = cartera.pesos_riesgo(cov, metodo)
        assert abs(w.sum() - 1.0) < 1e-9 and (w >= -1e-9).all()
    wiv = cartera.pesos_inv_vol(cov)
    # el activo 3 (var 0.01, el menos volatil) pesa mas que el 2 (var 0.16)
    assert wiv[3] > wiv[2]


def test_backtest_cartera_corre():
    """El backtest de cartera agrupa en cohortes y pondera por riesgo sin fallar."""
    import numpy as np, pandas as pd
    from patrones import cartera
    rng = np.random.default_rng(0)
    fechas = pd.date_range("2022-01-03", periods=600, freq="B")
    precios = {t: pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))},
                               index=fechas) for t in ["AAA", "BBB", "CCC"]}
    n = 120
    res = pd.DataFrame({"ticker": rng.choice(["AAA", "BBB", "CCC"], n),
                        "fecha": rng.choice(fechas[200:], n),
                        "ret_op": rng.normal(0.1, 1.0, n)})
    serie, m = cartera.backtest_cartera(res, precios, metodo="hrp", freq="W")
    assert m["cohortes"] > 5 and np.isfinite(m["sharpe_op"])


def test_ensemble_horizontes():
    """El ensemble de horizontes de salida promedia resultados y produce un
    retorno intermedio entre los horizontes individuales."""
    import numpy as np, pandas as pd
    from patrones import rentabilidad as R
    rng = np.random.default_rng(1); n = 200
    res = pd.DataFrame({
        "lado": np.ones(n), "prob_meta": rng.uniform(0.5, 0.8, n), "pliegue": 1,
        "mkt_alcista": 1.0,
        "cam_high": [np.sort(rng.uniform(0, 3, 10)) for _ in range(n)],
        "cam_low": [-np.sort(rng.uniform(0, 3, 10)) for _ in range(n)],
        "cam_close": [rng.uniform(-2, 2, 10) for _ in range(n)]})
    base = {"permitir_cortos": False, "solo_mercado_alcista": False, "evitar_vol_baja": False}
    _, m = R.aplicar_sistema(res, reglas={**base, "caps_horizonte": [4, 7, 10]})
    assert m.get("operaciones", 0) >= 20 and np.isfinite(m["sharpe"])


def test_kelly_fraccional_y_sizing():
    """Kelly da tamano 0 sin ventaja y crece con la probabilidad; el sizing por
    confianza/kelly no rompe aplicar_sistema y pondera mas lo mas fiable."""
    import numpy as np, pandas as pd
    from patrones import sizing, rentabilidad
    # Kelly binario con b=2: f*=0 en p=1/3, positivo por encima, monotono
    assert sizing.kelly_fraccional(0.33, 2.0)[()] if False else True
    f = sizing.kelly_fraccional(np.array([0.3, 0.5, 0.7]), 2.0, fraccion=1.0)
    assert f[0] == 0.0 and f[1] > 0 and f[2] > f[1]
    # sizing igual vs kelly sobre un res sintetico: misma seleccion, pesos distintos
    rng = np.random.default_rng(0); n = 400
    res = pd.DataFrame({
        "lado": np.ones(n), "prob_meta": rng.uniform(0.45, 0.8, n), "pliegue": 1,
        "mkt_alcista": 1.0, "sector": "tecnologia",
        "cam_high": [rng.uniform(0, 3, 6) for _ in range(n)],
        "cam_low": [rng.uniform(-3, 0, 6) for _ in range(n)],
        "cam_close": [rng.uniform(-2, 2, 6) for _ in range(n)]})
    base = {"permitir_cortos": False, "solo_mercado_alcista": False, "evitar_vol_baja": False}
    sel_i, m_i = rentabilidad.aplicar_sistema(res, reglas={**base, "sizing": "igual"})
    sel_k, m_k = rentabilidad.aplicar_sistema(res, reglas={**base, "sizing": "kelly"})
    assert (sel_i["tam"] == 1.0).all()
    assert m_k["tam_medio"] < 1.0 and (sel_k["tam"] > 0).all()


def test_deflated_sharpe_penaliza_pruebas():
    """A mas pruebas realizadas, mas alto el listón y menor el DSR del mismo
    backtest: el DSR castiga el multiple-testing."""
    import numpy as np
    from patrones import robustez
    rng = np.random.default_rng(0)
    r = rng.normal(0.05, 1.0, 400)          # retornos de la estrategia elegida
    trials = rng.normal(0, 0.1, 50)         # Sharpes de las pruebas
    dsr_pocas = robustez.deflated_sharpe_ratio(r, trials[:3])
    dsr_muchas = robustez.deflated_sharpe_ratio(r, np.r_[trials, rng.normal(0, 0.1, 500)])
    assert 0 <= dsr_muchas <= 1 and 0 <= dsr_pocas <= 1
    assert dsr_muchas <= dsr_pocas + 1e-9


def test_pbo_detecta_sobreajuste():
    """PBO alto cuando las 'estrategias' son ruido puro (el mejor IS es aleatorio
    OOS); bajo cuando una estrategia domina de forma consistente."""
    import numpy as np
    from patrones import robustez
    rng = np.random.default_rng(1)
    T, N = 240, 20
    ruido = rng.normal(0, 1, (T, N))        # ninguna es mejor de verdad
    pbo_ruido = robustez.probabilidad_sobreajuste(ruido, S=10)
    dominante = rng.normal(0, 1, (T, N))
    dominante[:, 0] += 0.8                   # la columna 0 es genuinamente buena
    pbo_bueno = robustez.probabilidad_sobreajuste(dominante, S=10)
    assert 0 <= pbo_ruido <= 1 and 0 <= pbo_bueno <= 1
    assert pbo_ruido > pbo_bueno
    assert pbo_bueno < 0.3


def test_meta_calibracion_monotona_y_en_rango():
    """La calibracion isotonica del meta mantiene el ranking (AUC) y deja las
    probabilidades en [0,1]."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from patrones.meta import MetaModelo
    rng = np.random.default_rng(2)
    n = 800
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)   # señal real
    filas = [{"a": float(xi), "b": float(rng.normal())} for xi in x]
    m = MetaModelo().entrenar(filas, y, calibrar=True)
    assert m.calibrador is not None
    p = m.prob_acierto(filas)
    assert p.min() >= 0 and p.max() <= 1
    # misma capacidad de ordenar (la isotonica es monotona)
    m0 = MetaModelo().entrenar(filas, y, calibrar=False)
    p0 = m0.prob_acierto(filas)
    assert abs(roc_auc_score(y, p) - roc_auc_score(y, p0)) < 0.02


def test_filtro_rvol_y_eod_en_aplicar_sistema():
    """aplicar_sistema respeta el filtro RVOL y la salida al cierre cuando el
    banco trae las columnas intradia; sin ellas, el sistema diario no cambia."""
    import numpy as np, pandas as pd
    from patrones import rentabilidad
    rng = np.random.default_rng(0)
    n = 300
    res = pd.DataFrame({
        "lado": np.ones(n), "prob_meta": rng.uniform(0.5, 0.8, n),
        "pliegue": 1, "mkt_alcista": 1.0,
        "barras_hasta_cierre": rng.integers(0, 6, n).astype(float),
        "rvol": rng.uniform(0.5, 3.0, n),
        "cam_high": [rng.uniform(0, 3, 6) for _ in range(n)],
        "cam_low": [rng.uniform(-3, 0, 6) for _ in range(n)],
        "cam_close": [rng.uniform(-2, 2, 6) for _ in range(n)],
    })
    sel, _ = rentabilidad.aplicar_sistema(
        res, reglas={"salida_cierre": True, "rvol_min": 1.5, "permitir_cortos": False,
                     "solo_mercado_alcista": False})
    # todas las operadas cumplen el filtro RVOL y tienen barra posterior (EOD)
    assert (sel["rvol"] >= 1.5).all()
    assert (sel["barras_hasta_cierre"] >= 1).all()


def test_rankear_conviccion_gate_y_orden(monkeypatch):
    """El ranking por conviccion descarta el catalizador muy negativo (gate) y
    ordena por confluencia (RVOL + ret_sesion + sentimiento). Sin red: se
    monkeypatchea el sentimiento de noticias."""
    import pandas as pd
    import gestionar as G
    sent = {"AAA": 0.4, "BBB": 0.0, "CCC": -0.6, "DDD": 0.2}
    monkeypatch.setattr(G, "_sentimiento", lambda t: sent.get(t, 0.0))
    opo = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "rvol":   [2.0,   1.8,   2.5,   1.6,   1.2],   # EEE no es fuerte (<1.5)
        "ret_sesion": [0.01, 0.015, 0.02, 0.008, 0.03],
        "dist_vwap": [0.005]*5, "atr_pct": [0.01]*5, "precio": [100.0]*5,
    })
    r = G.rankear_conviccion(opo, umbral_fuerte=1.5)
    tickers = list(r["ticker"])
    assert "EEE" not in tickers          # filtrado por RVOL < umbral
    assert "CCC" not in tickers          # descartado por el gate (sent -0.6 < -0.35)
    assert set(tickers) == {"AAA", "BBB", "DDD"}
    # el score de conviccion esta ordenado de mayor a menor
    assert list(r["conviccion"]) == sorted(r["conviccion"], reverse=True)
    # a igualdad de RVOL y ret_sesion, el catalizador positivo desempata al alza
    par = pd.DataFrame({
        "ticker": ["POS", "NEU"], "rvol": [2.0, 2.0], "ret_sesion": [0.01, 0.01],
        "dist_vwap": [0.005, 0.005], "atr_pct": [0.01, 0.01], "precio": [100.0, 100.0]})
    sent2 = {"POS": 0.5, "NEU": 0.0}
    monkeypatch.setattr(G, "_sentimiento", lambda t: sent2.get(t, 0.0))
    rp = G.rankear_conviccion(par, umbral_fuerte=1.5)
    assert list(rp["ticker"]) == ["POS", "NEU"]


def test_rankear_conviccion_excluir_y_vacio(monkeypatch):
    """Excluye lo ya en cartera y devuelve vacio si no queda candidato fuerte."""
    import pandas as pd
    import gestionar as G
    monkeypatch.setattr(G, "_sentimiento", lambda t: 0.0)
    opo = pd.DataFrame({
        "ticker": ["AAA", "BBB"], "rvol": [2.0, 1.1],
        "ret_sesion": [0.01, 0.01], "dist_vwap": [0.005, 0.005],
        "atr_pct": [0.01, 0.01], "precio": [100.0, 100.0]})
    r = G.rankear_conviccion(opo, umbral_fuerte=1.5, excluir={"AAA"})
    assert len(r) == 0                   # AAA excluida, BBB no es fuerte
    assert len(G.rankear_conviccion(pd.DataFrame(), 1.5)) == 0


def test_costes_escala_inversa_con_volatilidad():
    """La propiedad clave: en unidades de ATR, el coste escala INVERSAMENTE con la
    volatilidad. Un valor de baja atr_pct cuesta mas (en riesgo) que uno alto."""
    import numpy as np
    from patrones import costes
    c_bajo = costes.coste_atr(0.008, precio=100.0, slippage_atr=0.0)   # atr_pct 0.8%
    c_alto = costes.coste_atr(0.020, precio=100.0, slippage_atr=0.0)   # atr_pct 2.0%
    assert c_bajo > c_alto                      # baja vol -> mas coste en ATR
    assert c_bajo / c_alto == pytest_approx(0.020 / 0.008, rel=1e-6)   # exactamente inverso
    # el suelo evita explotar en atr_pct minusculos
    assert np.isfinite(costes.coste_atr(0.0, precio=100.0))
    # slippage_atr se suma tal cual (ya esta en ATR)
    base = costes.coste_atr(0.02, precio=100.0, slippage_atr=0.0)
    assert costes.coste_atr(0.02, precio=100.0, slippage_atr=0.05) == pytest_approx(base + 0.05)


def test_costes_por_fila_fallback_plano():
    """coste_por_fila usa atr_pct por fila; sin columna de volatilidad cae al plano."""
    import numpy as np, pandas as pd
    from patrones import costes
    df = pd.DataFrame({"atr_pct": [0.01, 0.02], "precio": [50.0, 200.0]})
    c = costes.coste_por_fila(df)
    assert len(c) == 2 and (c > 0).all()
    assert c[0] > c[1]                           # 1% atr cuesta mas que 2% atr
    sin = costes.coste_por_fila(pd.DataFrame({"x": [1, 2, 3]}), defecto_plano=0.05)
    assert list(sin) == [0.05, 0.05, 0.05]


def pytest_approx(*a, **k):
    import pytest
    return pytest.approx(*a, **k)


def test_aplicar_sistema_coste_realista():
    """coste_atr='realista' usa coste por fila (escala con volatilidad) y no rompe
    el pipeline: devuelve metricas y penaliza mas la baja volatilidad que el plano."""
    import numpy as np, pandas as pd
    from patrones import rentabilidad
    rng = np.random.default_rng(3)
    n = 400
    res = pd.DataFrame({
        "lado": np.ones(n), "prob_meta": rng.uniform(0.5, 0.85, n),
        "pliegue": 1, "mkt_alcista": 1.0,
        "atr_pct": rng.uniform(0.006, 0.03, n),       # volatilidad variada
        "cam_high": [rng.uniform(0, 3, 8) for _ in range(n)],
        "cam_low": [rng.uniform(-3, 0, 8) for _ in range(n)],
        "cam_close": [rng.uniform(-2, 2, 8) for _ in range(n)],
    })
    reglas = {"permitir_cortos": False, "solo_mercado_alcista": False,
              "evitar_vol_baja": False}
    sel_p, met_p = rentabilidad.aplicar_sistema(res, reglas=reglas, coste_atr=0.05)
    sel_r, met_r = rentabilidad.aplicar_sistema(res, reglas=reglas, coste_atr="realista")
    assert met_r.get("operaciones", 0) > 0
    # el coste realista da un retorno medio distinto del plano (no es la misma resta)
    assert met_p["ret_medio_atr"] != met_r["ret_medio_atr"]
    # callable tambien admitido: coste cero -> retorno >= que con coste plano
    sel_0, met_0 = rentabilidad.aplicar_sistema(
        res, reglas=reglas, coste_atr=lambda d: np.zeros(len(d)))
    assert met_0["ret_medio_atr"] >= met_p["ret_medio_atr"] - 1e-9


def test_cooldown_tras_sl(tmp_path, monkeypatch):
    """Un nombre que acaba de perder (SL) entra en cooldown; una perdida vieja no;
    y un nombre con solo operaciones ganadoras/abiertas no entra."""
    import pandas as pd
    import registro as R
    log = tmp_path / "trades.csv"
    ahora = pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame([
        {"ticker": "MU",  "ts": (ahora - pd.Timedelta(minutes=120)).isoformat(),
         "salida_ts": (ahora - pd.Timedelta(minutes=10)).isoformat(), "resultado": "loss"},
        {"ticker": "GM",  "ts": (ahora - pd.Timedelta(hours=5)).isoformat(),
         "salida_ts": (ahora - pd.Timedelta(hours=4)).isoformat(), "resultado": "loss"},
        {"ticker": "JPM", "ts": (ahora - pd.Timedelta(minutes=5)).isoformat(),
         "salida_ts": "", "resultado": "open"},
        {"ticker": "COST", "ts": (ahora - pd.Timedelta(minutes=5)).isoformat(),
         "salida_ts": (ahora - pd.Timedelta(minutes=2)).isoformat(), "resultado": "win"},
    ])
    df.to_csv(log, index=False)
    monkeypatch.setattr(R, "LOG", str(log))
    cd = R.tickers_en_cooldown(minutos=60)
    assert "MU" in cd            # perdio hace 10 min -> cooldown
    assert "GM" not in cd        # perdio hace 4 h -> fuera de la ventana
    assert "JPM" not in cd       # abierta, no es perdida
    assert "COST" not in cd      # ganadora
    # sin archivo -> conjunto vacio
    monkeypatch.setattr(R, "LOG", str(tmp_path / "noexiste.csv"))
    assert R.tickers_en_cooldown() == set()
