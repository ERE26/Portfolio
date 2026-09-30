"""Tests del detector de figuras y del desglose explicativo de similitud."""
import numpy as np
import pandas as pd
import pytest

from patrones import figuras, datos, ventanas, contexto, memoria
from patrones.explicar import por_que_parecido


def _forma(nombre, n=40, semilla=0):
    rng = np.random.default_rng(semilla)
    camino, _ = datos.figura(nombre, n, rng)
    return ventanas.forma(camino.reshape(1, -1))[0]


def test_detecta_doble_techo():
    d = figuras.describir(_forma("doble_techo"))
    assert d["figura"] in ("doble_techo", "techo_multiple")


def test_detecta_doble_suelo():
    d = figuras.describir(_forma("doble_suelo"))
    assert d["figura"] in ("doble_suelo", "suelo_multiple")


def test_detecta_hch():
    d = figuras.describir(_forma("hombro_cabeza_hombro"))
    assert d["figura"] == "hombro_cabeza_hombro"


def test_figura_necesita_minimo_velas():
    d = figuras.describir(np.array([0.0, 1.0, -1.0]))   # 3 velas
    assert d["figura"] == "insuficiente"


def test_tendencia_coherente():
    sube = ventanas.forma(np.linspace(0, 5, 40).reshape(1, -1))[0]
    baja = ventanas.forma(np.linspace(5, 0, 40).reshape(1, -1))[0]
    assert figuras.describir(sube)["tendencia"] == "alcista"
    assert figuras.describir(baja)["tendencia"] == "bajista"


def test_texto_no_peta_con_figura_minima():
    txt = figuras.texto(figuras.describir(np.zeros(40)))
    assert isinstance(txt, str) and len(txt) > 0


def test_explicar_usa_distancias_por_grupo():
    """El desglose 'por que se parece' debe existir en los vecinos y ser coherente."""
    pre, meta, idx, reg = datos.generar_universo(n_tickers=6, n_barras=1200, semilla=4)
    ctx = contexto.Contexto(pre, meta, idx)
    b = ventanas.construir(pre, ctx, None, verbose=False)
    corte = b.meta.fecha.quantile(0.7)
    tr = b.subconjunto((b.meta.fecha <= corte).to_numpy())
    te = b.subconjunto((b.meta.fecha > corte).to_numpy())
    mem = memoria.MemoriaPatrones(b.cfg, k=20).ajustar(tr)
    p = mem.predecir(te, 5)
    # deben existir columnas d_<grupo> para cada grupo con peso
    for g in mem.pesos:
        assert f"d_{g}" in p.vecinos.columns
    v = p.vecinos.iloc[0]
    txt = por_que_parecido(v, mem.pesos)
    assert "coincide en" in txt
