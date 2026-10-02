"""Calibra UMBRAL_CONVICCION con DATOS, no a ojo. Mina la bitácora (datos/snapshots/*.json),
cruza convicción de entrada con el resultado real (win/loss, P&L) y, para una rejilla de
umbrales, calcula cuántas operaciones quedarían, su win-rate, P&L total y profit factor.

HONESTO: solo RECOMIENDA un número si hay muestra suficiente (MIN_MUESTRA) y la mejora es
real; si no, dice 'seguir observando' y deja el 0.0 provisional. Con pocas operaciones la
convicción puede no predecir nada (es esperable al principio) y optimizar sería sobreajuste.

  python calibrar_suelo.py
"""
import glob
import json

import numpy as np

MIN_MUESTRA = 25          # nº mínimo de operaciones cerradas para fiarse de un umbral
REJILLA = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def _cargar():
    filas = []
    for f in glob.glob("datos/snapshots/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        c, r, p = d.get("conviccion"), d.get("resultado"), d.get("pnl")
        if c is not None and r in ("win", "loss") and p is not None:
            filas.append((float(c), r, float(p)))
    return filas


def _metr(sub):
    n = len(sub)
    if not n:
        return None
    w = [p for r, p in sub if r == "win"]
    l = [p for r, p in sub if r == "loss"]
    pnl = sum(p for _, p in sub)
    pf = (sum(w) / abs(sum(l))) if l and sum(l) != 0 else float("inf")
    return {"n": n, "wr": len(w) / n, "pnl": pnl, "pf": pf, "medio": pnl / n}


def main():
    filas = _cargar()
    base = _metr([(r, p) for c, r, p in filas])
    print(f"Operaciones intradía cerradas con convicción registrada: {len(filas)}")
    if base:
        print(f"Global (sin suelo): win-rate {base['wr']*100:.0f}% | P&L {base['pnl']:+.0f} | "
              f"profit factor {base['pf']:.2f} | P&L medio {base['medio']:+.1f}\n")

    print(f"  {'umbral':>7}{'ops':>5}{'win-rate':>10}{'P&L total':>11}{'P&L medio':>11}{'PF':>7}")
    cand = []
    for u in REJILLA:
        m = _metr([(r, p) for c, r, p in filas if c >= u])
        if not m:
            print(f"  {u:>+7.2f}{0:>5}{'—':>10}{'—':>11}{'—':>11}{'—':>7}")
            continue
        print(f"  {u:>+7.2f}{m['n']:>5}{m['wr']*100:>9.0f}%{m['pnl']:>+11.0f}{m['medio']:>+11.1f}"
              f"{m['pf']:>7.2f}")
        cand.append((u, m))

    # correlación convicción<->P&L (¿hay señal siquiera?)
    if len(filas) >= 3:
        cc = np.corrcoef([c for c, _, _ in filas], [p for _, _, p in filas])[0, 1]
        print(f"\n  Correlación convicción ↔ P&L: {cc:+.2f}  "
              f"({'la convicción NO ayuda (o va al revés)' if cc <= 0.1 else 'hay algo de señal'})")

    print("\n=== RECOMENDACIÓN ===")
    if len(filas) < MIN_MUESTRA:
        print(f"  MUESTRA INSUFICIENTE ({len(filas)}/{MIN_MUESTRA}). No se calibra sobre tan poco "
              f"(sería sobreajuste).")
        print(f"  -> Mantener UMBRAL_CONVICCION = 0.0 (provisional) y seguir acumulando OOS; "
              f"vuelve a correr esto al llegar a {MIN_MUESTRA}.")
        return
    # con muestra: el mejor umbral por profit factor, exigiendo que deje operar algo
    viables = [(u, m) for u, m in cand if m["n"] >= max(10, int(0.3 * len(filas)))]
    if not viables:
        print("  Ningún umbral deja muestra suficiente para fiarse -> mantener 0.0.")
        return
    mejor = max(viables, key=lambda x: (x[1]["pf"], x[1]["pnl"]))
    u, m = mejor
    print(f"  Mejor umbral por profit factor con muestra suficiente: {u:+.2f} "
          f"(PF {m['pf']:.2f}, win-rate {m['wr']*100:.0f}%, {m['n']} ops).")
    print(f"  Compáralo con 0.0 antes de fijarlo; cámbialo en gestionar.py (UMBRAL_CONVICCION).")


if __name__ == "__main__":
    main()
