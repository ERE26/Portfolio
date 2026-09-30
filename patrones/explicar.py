"""Explica por que dos tramos de grafico se parecen, desglosando la distancia.

El sistema mide la similitud como una mezcla de distancias por grupo (forma,
volumen, RSI/osciladores, momento, temporalidades superiores, contexto de
mercado, noticias). Aqui se traduce ese desglose a lenguaje humano para que la
busqueda sea transparente: no "se parece y ya", sino "se parece SOBRE TODO en la
forma y el volumen, y diverge en el momento".
"""
import numpy as np

ETIQUETAS = {
    "forma": "la forma del precio", "forma_volumen": "la forma del volumen",
    "tf_semanal": "el grafico semanal", "tf_mensual": "el grafico mensual",
    "osciladores": "los osciladores (RSI, MACD, Bollinger)", "momento": "el momento y las medias",
    "volumen": "el perfil de volumen", "volatilidad": "la volatilidad",
    "contexto": "el contexto de mercado (regimen, fuerza relativa)",
    "noticias": "el tono de las noticias"}


def por_que_parecido(vecino, pesos, n=3):
    """Dado un vecino (fila con columnas d_<grupo>) y los pesos del modelo,
    devuelve las razones ordenadas por su contribucion a la cercania."""
    contrib = {}
    for g, w in pesos.items():
        col = f"d_{g}"
        if col in vecino and w > 0 and np.isfinite(vecino[col]):
            # baja distancia y peso alto -> mas razon de parecido
            contrib[g] = w * (1.0 - min(float(vecino[col]), 1.0))
    if not contrib:
        return "similitud general"
    orden = sorted(contrib.items(), key=lambda kv: -kv[1])
    fuertes = [ETIQUETAS.get(g, g) for g, _ in orden[:n]]
    # el grupo donde MAS diverge (mayor distancia relativa)
    divs = {g: float(vecino[f"d_{g}"]) for g in pesos if f"d_{g}" in vecino
            and np.isfinite(vecino.get(f"d_{g}", np.nan))}
    peor = max(divs, key=divs.get) if divs else None
    txt = "coincide en " + ", ".join(fuertes)
    if peor and peor not in [g for g, _ in orden[:n]]:
        txt += f"; difiere en {ETIQUETAS.get(peor, peor)}"
    return txt
