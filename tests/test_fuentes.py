"""Tests de los conectores que NO necesitan red (parsing y logica pura).

Las pruebas con red real (Yahoo, TradingView, Forex Factory) se hacen a mano con
`python bot.py envivo` y quedan fuera de la suite para que esta sea reproducible
y no dependa de que las APIs esten disponibles.
"""
import pandas as pd
import pytest

from fuentes import tradingview as tv
from fuentes import forexfactory as ff
from patrones.noticias import puntuar_titular


def test_simbolo_tv_exchange():
    assert tv.como_simbolo_tv("AAPL", "NMS") == "NASDAQ:AAPL"
    assert tv.como_simbolo_tv("JPM", "NYQ") == "NYSE:JPM"
    assert tv.como_simbolo_tv("NASDAQ:MSFT") == "NASDAQ:MSFT"  # ya calificado
    assert tv.como_simbolo_tv("XYZ") == "XYZ"                   # sin mercado conocido


def test_texto_recomendacion():
    assert tv._texto_recom(0.8) == "COMPRA FUERTE"
    assert tv._texto_recom(0.2) == "compra"
    assert tv._texto_recom(0.0) == "neutral"
    assert tv._texto_recom(-0.3) == "venta"
    assert tv._texto_recom(-0.8) == "VENTA FUERTE"
    assert tv._texto_recom(None) == "sin dato"


def test_bloque_informe_tolera_faltantes():
    # no debe petar aunque falten columnas
    txt = tv.bloque_informe({"simbolo": "NASDAQ:AAPL", "recomendacion_txt": "compra"})
    assert "TradingView" in txt and "NASDAQ:AAPL" in txt


def test_peso_impacto_calendario():
    assert ff.PESO_IMPACTO["High"] > ff.PESO_IMPACTO["Medium"] > ff.PESO_IMPACTO["Low"]


def test_noticias_reales_respeta_interfaz():
    """NoticiasReales sin red: con df inyectado debe filtrar por fecha (no-lookahead)."""
    from fuentes.noticias_reales import NoticiasReales
    fn = NoticiasReales.__new__(NoticiasReales)      # sin __init__ (evita la red)
    fn.df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "ticker": ["AAPL"] * 3,
        "titular": ["sube fuerte", "profit warning", "bate previsiones"],
        "puntuacion": [0.6, -1.0, 0.8]})
    vistos = fn.titulares("AAPL", pd.Timestamp("2024-02-15"), dias=365)
    assert len(vistos) == 2                           # el de marzo aun no ha pasado
    assert (vistos.fecha < pd.Timestamp("2024-02-15")).all()
