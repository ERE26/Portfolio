"""Bucle de aprendizaje, el equivalente directo de las partidas del bot de Worldle.

Cada "partida" es:
  1. Se le ensena un tramo de grafico que no ha visto (y NO el futuro).
  2. Dice si le suena la forma, a que casos se parece y que cree que va a pasar.
  3. Se le revela lo que paso de verdad.
  4. Aprende: guarda el caso con su desenlace, reajusta los pesos de los grupos
     que fallaron y recalibra sus probabilidades.

Las partidas van en orden cronologico a proposito. Asi la memoria en la partida
n solo contiene cosas anteriores a la partida n, igual que ocurriria en real.
"""
import os

import numpy as np
import pandas as pd

from .evaluacion import metricas, submuestreo_no_solapado
from .memoria import MemoriaPatrones


def simular(banco, rondas=100, fraccion_inicial=0.6, k=30, umbral_confianza=0.35,
            log="aprendizaje.csv", memoria_previa=None, guardar_en=None,
            interactivo=False, verbose=True, semilla=0, **kw_mem):
    cfg = banco.cfg
    corte = banco.meta["fecha"].quantile(fraccion_inicial)
    inicial = banco.subconjunto((banco.meta["fecha"] <= corte).to_numpy())
    futuro = submuestreo_no_solapado(
        banco.subconjunto((banco.meta["fecha"] > corte).to_numpy()), cfg.horizonte)

    if memoria_previa and MemoriaPatrones.existe(memoria_previa):
        mem = MemoriaPatrones.cargar(memoria_previa)
        if verbose:
            print(f"Memoria cargada de {memoria_previa}: {len(mem)} casos ya conocidos")
    else:
        mem = MemoriaPatrones(cfg, k=k, umbral_confianza=umbral_confianza, **kw_mem).ajustar(inicial)
        if verbose:
            print(f"Memoria inicial: {len(mem)} casos hasta {corte.date()}")

    rondas = min(rondas, len(futuro))
    if verbose:
        print(f"Partidas disponibles: {len(futuro)}, se juegan {rondas}\n")

    filas = []
    for r in range(rondas):
        p = mem.predecir(futuro, r)
        real = float(futuro.y["ret"].iloc[r])
        if verbose:
            print(f"--- Partida {r + 1}  |  {len(mem)} casos en memoria ---")
            print(p.informe(4))
        acerto = mem.aprender_del_error(p, real)
        mem.anadir(futuro.subconjunto(np.arange(len(futuro)) == r))
        if verbose:
            veredicto = "ACIERTO" if acerto else "FALLO"
            print(f"  REALIDAD: {real * 100:+.2f}%  ->  {veredicto}"
                  + ("  (se habia abstenido)" if p.abstencion else ""))
            top = sorted(mem.pesos.items(), key=lambda kv: -kv[1])[:3]
            print("  Pesos ahora: " + ", ".join(f"{g} {w:.3f}" for g, w in top) + "\n")
        filas.append({
            "partida": r + 1, "casos_en_memoria": len(mem), "ticker": p.ticker,
            "fecha": p.fecha, "prob": p.prob_sube, "prob_cruda": p.prob_sube_cruda,
            "ret_esperado": p.ret_esperado, "ret_real": real, "acerto": int(acerto),
            "confianza": p.confianza, "abstencion": int(p.abstencion),
            "brier": (p.prob_sube - (1.0 if real > 0 else 0.0)) ** 2,
            "distancia": p.distancia_media,
            **{f"peso_{g}": w for g, w in mem.pesos.items()}})
        if interactivo:
            try:
                if input("  Enter para la siguiente partida (q para salir): ").strip().lower() == "q":
                    break
            except EOFError:
                interactivo = False

    hist = pd.DataFrame(filas)
    if log:
        hist.to_csv(log, index=False, mode="a" if os.path.exists(log) else "w",
                    header=not os.path.exists(log))
    if guardar_en:
        mem.guardar(guardar_en)
        if verbose:
            print(f"Memoria guardada en {guardar_en} ({len(mem)} casos)")
    if verbose and len(hist):
        resumen(hist)
    return hist, mem


def resumen(hist, tramos=4):
    """Lo que interesa: mejora con las partidas o no?"""
    print("\n" + "=" * 70)
    print("  RESUMEN DEL APRENDIZAJE")
    print("=" * 70)
    m = metricas(hist)
    print(f"Partidas: {len(hist)}   acierto global {m.get('acierto', float('nan')):.2%}   "
          f"Brier {m.get('brier', float('nan')):.4f}   AUC {m.get('auc', float('nan')):.4f}")
    act = hist[hist.abstencion == 0]
    if len(act):
        print(f"Sin abstenerse ({len(act)} partidas): acierto {act.acerto.mean():.2%}")
    print(f"Se abstuvo en {hist.abstencion.mean():.1%} de las partidas")

    print("\nEvolucion por tramos de partidas (esto es la curva de aprendizaje):")
    corte = np.array_split(np.arange(len(hist)), tramos)
    print(f"  {'tramo':<12}{'n':>5}{'acierto':>10}{'brier':>9}{'confianza':>11}{'distancia':>11}")
    for i, idx in enumerate(corte, 1):
        if len(idx) == 0:
            continue
        g = hist.iloc[idx]
        print(f"  {f'{idx[0]+1}-{idx[-1]+1}':<12}{len(g):>5}{g.acerto.mean():>9.1%}"
              f"{g.brier.mean():>9.4f}{g.confianza.mean():>11.3f}{g.distancia.mean():>11.3f}")

    cols = [c for c in hist.columns if c.startswith("peso_")]
    if cols:
        print("\nPesos: inicio -> final (quien se gano la confianza del sistema)")
        ini, fin = hist[cols].iloc[0], hist[cols].iloc[-1]
        for c in sorted(cols, key=lambda c: -fin[c]):
            flecha = "sube " if fin[c] > ini[c] * 1.05 else "baja " if fin[c] < ini[c] * .95 else "igual"
            print(f"  {c[5:]:<16}{ini[c]:.3f} -> {fin[c]:.3f}  {flecha}")

    if len(hist) > 20:
        q = hist.confianza.quantile(.75)
        alta, baja = hist[hist.confianza >= q], hist[hist.confianza < q]
        print(f"\nAcierto segun confianza: alta (top 25%, n={len(alta)}) {alta.acerto.mean():.2%}"
              f"  |  resto (n={len(baja)}) {baja.acerto.mean():.2%}")
        print("  Si la confianza sirve para algo, el primero debe ser mayor que el segundo.")
