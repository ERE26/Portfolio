"""Sistema de aprendizaje de patrones de grafico por memoria de casos.

La idea es la misma que la del bot de Worldle: guardar en memoria la *forma*
de cada caso que se ha visto junto con lo que paso despues, y cuando llega una
forma nueva buscar los casos mas parecidos para predecir. Aprende de sus
errores reajustando el peso de cada grupo de features y recalibrando sus
probabilidades a medida que se revelan los resultados reales.
"""

__version__ = "0.1.0"
