"""Conectores a fuentes de datos reales.

  yahoo         precios OHLCV historicos (varias temporalidades) y titulares por valor
  tradingview   scanner en vivo: indicadores ya calculados y recomendacion tecnica
  forexfactory  calendario macro (tipos, empleo, IPC...) con impacto Low/Med/High

Que se puede y que no (comprobado en vivo):
  - Yahoo da el historico de velas: es el mismo que se ve en el grafico de
    TradingView, porque TradingView solo licencia datos de los exchanges.
  - El historico de velas de TradingView NO es accesible por API publica (va por
    websocket con token de sesion y contra sus Terminos de Servicio), asi que NO
    se usa. Del scanner, que si es publico, se toma solo el snapshot en vivo.
  - Las noticias reales (Yahoo, Forex Factory) son recientes, sin historico
    profundo. Por eso alimentan la capa EN VIVO de la prediccion, no el
    entrenamiento: etiquetar barras viejas con titulares de hoy seria mirar al
    futuro (ver README y patrones/noticias.py).
"""
from .http import Cliente

__all__ = ["Cliente", "yahoo", "tradingview", "forexfactory"]
