"""Noticias en vivo por ticker: titulares recientes de Yahoo + sentimiento.

Cuanta mas informacion de noticias, mas contexto para decidir. Se puntua cada
titular con el lexico del proyecto (patrones.noticias) y se da un tono medio.
"""
from . import yahoo
from patrones.noticias import puntuar_titular


def noticias(ticker, n=6):
    """Titulares recientes + sentimiento. Devuelve {sentimiento, n, etiqueta, titulares}."""
    try:
        t = yahoo.titulares(ticker, n=n)
    except Exception:
        return {"sentimiento": 0.0, "n": 0, "etiqueta": "s/d", "titulares": []}
    if t is None or len(t) == 0:
        return {"sentimiento": 0.0, "n": 0, "etiqueta": "s/d", "titulares": []}
    items = []
    for _, r in t.iterrows():
        p = float(puntuar_titular(str(r["titular"])))
        items.append({"fecha": str(r["fecha"])[:16], "titular": str(r["titular"]),
                      "punt": p, "fuente": str(r.get("fuente", "")), "url": str(r.get("url", ""))})
    sent = sum(i["punt"] for i in items) / len(items)
    etq = "positivo" if sent > 0.1 else "negativo" if sent < -0.1 else "neutro"
    return {"sentimiento": sent, "n": len(items), "etiqueta": etq, "titulares": items}
