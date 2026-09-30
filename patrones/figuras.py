"""Detector de figuras chartistas clasicas sobre una ventana de precios.

Reconoce, con analisis geometrico (picos, valles, pendientes y niveles), las
figuras que pediste: techos y suelos (dobles/triples), hombro-cabeza-hombro y su
version invertida, canales y tendencias, triangulos, y roturas de soporte o
resistencia. No usa el futuro: solo mira la ventana que se le da.

Sirve para dos cosas:
  - etiquetar en lenguaje humano el tramo objetivo y sus vecinos ("esto parece un
    doble techo con rotura de soporte"), para que la busqueda sea explicable;
  - como descriptor adicional de similitud entre tramos.
"""
import numpy as np
from scipy.signal import find_peaks

MIN_VELAS = 5  # una figura necesita al menos 5 velas para considerarse tal


def _extremos(y, prominencia=0.35, dist=3):
    """Picos y valles relevantes de la serie (en unidades de desviacion)."""
    picos, _ = find_peaks(y, prominence=prominencia, distance=dist)
    valles, _ = find_peaks(-y, prominence=prominencia, distance=dist)
    return picos, valles


def _pendiente(y):
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float(np.dot(y - y.mean(), x) / max((x ** 2).sum(), 1e-9))


def describir(forma, volumen=None):
    """Clasifica una ventana (forma normalizada, media 0 desv 1) y la describe.

    Devuelve un dict con: figura principal, lista de figuras detectadas, si hubo
    rotura, la tendencia de fondo y detalles geometricos.
    """
    y = np.asarray(forma, float)
    n = len(y)
    if n < MIN_VELAS:
        return {"figura": "insuficiente", "figuras": [], "rotura": None,
                "tendencia": "n/a", "detalle": f"solo {n} velas"}

    # suavizado ligero para que el ruido no invente picos
    k = max(1, n // 20)
    ys = np.convolve(y, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    picos, valles = _extremos(ys)
    pend = _pendiente(y)
    tendencia = "alcista" if pend > 0.03 else "bajista" if pend < -0.03 else "lateral"

    figuras = []
    det = {}

    # --- TECHOS y SUELOS (dobles/triples) y HOMBRO-CABEZA-HOMBRO ---
    if len(picos) >= 2:
        alturas = ys[picos]
        top2 = picos[np.argsort(alturas)[-3:]]
        top2 = np.sort(top2)
        if len(picos) >= 3:
            tres = np.sort(picos[np.argsort(alturas)[-3:]])
            a, b, c = ys[tres]
            # cabeza (central) mas alta que los dos hombros, y hombros parecidos
            if b > a and b > c and abs(a - c) < 0.5 * abs(b - min(a, c) + 1e-9) and \
               tres[0] < tres[1] < tres[2]:
                figuras.append("hombro_cabeza_hombro")
                det["hch"] = {"hombros": [int(tres[0]), int(tres[2])], "cabeza": int(tres[1])}
        if "hombro_cabeza_hombro" not in figuras and len(picos) >= 2:
            p2 = np.sort(picos[np.argsort(alturas)[-2:]])
            if abs(ys[p2[0]] - ys[p2[1]]) < 0.4 and (p2[1] - p2[0]) >= 3:
                figuras.append("doble_techo" if len(picos) == 2 else "techo_multiple")
                det["techo"] = [int(p2[0]), int(p2[1])]

    if len(valles) >= 2:
        prof = ys[valles]
        if len(valles) >= 3:
            tres = np.sort(valles[np.argsort(prof)[:3]])
            a, b, c = ys[tres]
            if b < a and b < c and abs(a - c) < 0.5 * abs(max(a, c) - b + 1e-9) and \
               tres[0] < tres[1] < tres[2]:
                figuras.append("hombro_cabeza_hombro_invertido")
                det["hchi"] = {"hombros": [int(tres[0]), int(tres[2])], "cabeza": int(tres[1])}
        if "hombro_cabeza_hombro_invertido" not in figuras and len(valles) >= 2:
            v2 = np.sort(valles[np.argsort(prof)[:2]])
            if abs(ys[v2[0]] - ys[v2[1]]) < 0.4 and (v2[1] - v2[0]) >= 3:
                figuras.append("doble_suelo" if len(valles) == 2 else "suelo_multiple")
                det["suelo"] = [int(v2[0]), int(v2[1])]

    # --- CANALES y TRIANGULOS (envolventes de maximos y minimos) ---
    if len(picos) >= 2 and len(valles) >= 2:
        pend_max = _pendiente(ys[picos]) if len(picos) >= 2 else 0.0
        pend_min = _pendiente(ys[valles]) if len(valles) >= 2 else 0.0
        rango_ini = ys[picos[0]] - ys[valles[0]]
        rango_fin = ys[picos[-1]] - ys[valles[-1]]
        if rango_fin < rango_ini * 0.6:
            figuras.append("triangulo")
            det["triangulo"] = "descendente" if pend_max < -0.02 else \
                "ascendente" if pend_min > 0.02 else "simetrico"
        elif abs(pend_max - pend_min) < 0.04 and abs(pend_max) > 0.02:
            figuras.append("canal_alcista" if pend_max > 0 else "canal_bajista")

    # --- ROTURA de soporte / resistencia (ultimo tramo frente al rango previo) ---
    rotura = None
    if n >= 10:
        previo = ys[: int(n * 0.8)]
        resist, soporte = previo.max(), previo.min()
        ult = ys[-max(2, n // 10):].mean()
        if ult > resist + 0.15:
            rotura = "rotura_resistencia_alcista"
        elif ult < soporte - 0.15:
            rotura = "rotura_soporte_bajista"
    if rotura:
        figuras.append(rotura)

    if not figuras:
        figuras.append(f"tendencia_{tendencia}" if tendencia != "lateral" else "lateral")

    # figura principal: se prioriza la mas informativa
    prioridad = ["hombro_cabeza_hombro", "hombro_cabeza_hombro_invertido", "doble_techo",
                 "doble_suelo", "techo_multiple", "suelo_multiple", "triangulo",
                 "canal_alcista", "canal_bajista", "rotura_resistencia_alcista",
                 "rotura_soporte_bajista"]
    principal = next((f for f in prioridad if f in figuras), figuras[0])

    # confirmacion por volumen (si se pasa): rotura con volumen alto = mas fiable
    conf_vol = None
    if volumen is not None and len(volumen) == n:
        v = np.asarray(volumen, float)
        conf_vol = float(v[-max(2, n // 10):].mean() - v[:-max(2, n // 10)].mean())

    return {"figura": principal, "figuras": figuras, "rotura": rotura,
            "tendencia": tendencia, "n_picos": int(len(picos)), "n_valles": int(len(valles)),
            "pendiente": round(pend, 4), "conf_volumen": conf_vol, "detalle": det}


def texto(desc):
    """Descripcion en una linea, en lenguaje humano."""
    nombres = {
        "hombro_cabeza_hombro": "hombro-cabeza-hombro (giro bajista)",
        "hombro_cabeza_hombro_invertido": "hombro-cabeza-hombro invertido (giro alcista)",
        "doble_techo": "doble techo (giro bajista)", "doble_suelo": "doble suelo (giro alcista)",
        "techo_multiple": "techo multiple", "suelo_multiple": "suelo multiple",
        "triangulo": "triangulo", "canal_alcista": "canal alcista",
        "canal_bajista": "canal bajista",
        "rotura_resistencia_alcista": "rotura de resistencia al alza",
        "rotura_soporte_bajista": "rotura de soporte a la baja",
        "lateral": "rango lateral", "tendencia_alcista": "tendencia alcista",
        "tendencia_bajista": "tendencia bajista"}
    base = nombres.get(desc["figura"], desc["figura"])
    extra = ""
    if desc.get("detalle", {}).get("triangulo"):
        extra = f" {desc['detalle']['triangulo']}"
    rot = ""
    if desc.get("rotura") and desc["rotura"] != desc["figura"]:
        rot = f", con {nombres.get(desc['rotura'], desc['rotura'])}"
    return f"{base}{extra}{rot} (fondo {desc['tendencia']})"
