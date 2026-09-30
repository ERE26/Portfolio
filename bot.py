#!/usr/bin/env python3
"""Bot que aprende patrones de grafico jugando, y tambien de sus errores.

Misma idea que el bot de Worldle: juega partidas, en cada una intenta predecir
y luego se le revela lo que paso de verdad para que aprenda. Aqui una "partida"
es un tramo de grafico y la prediccion es hacia donde ira en las proximas barras.

Tiene en cuenta: la forma del precio en varias temporalidades (diaria, semanal,
mensual), el volumen, RSI/MACD/ADX/Bollinger y mas indicadores, el sector y el
regimen del mercado, la fuerza relativa frente al indice, y un canal de noticias.

Uso tipico (sin conexion, con el mercado sintetico de prueba):
  python bot.py demo
  python bot.py simular  --rondas 60
  python bot.py evaluar  --pliegues 5
  python bot.py predecir --ticker SYN03

Con tus propios datos (un CSV de OHLCV por valor en una carpeta):
  python bot.py evaluar  --fuente csv --carpeta datos/mi_cartera
  python bot.py predecir --fuente csv --carpeta datos/mi_cartera --ticker AAPL

Con Yahoo (solo si el equipo tiene salida a internet a Yahoo Finance):
  python bot.py evaluar  --fuente yahoo --tickers AAPL,MSFT,NVDA,JPM,XOM
"""
import argparse
import sys

import numpy as np
import pandas as pd

from patrones import evaluacion as ev
from patrones import simulador as sim
from patrones.memoria import MemoriaPatrones
from patrones.pipeline import montar_banco


def _comunes(p):
    p.add_argument("--fuente", default="sintetico", choices=["sintetico", "csv", "yahoo"],
                   help="de donde salen los precios")
    p.add_argument("--carpeta", default="datos/sintetico", help="carpeta de CSVs (csv/sintetico)")
    p.add_argument("--tickers", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="lista separada por comas (yahoo, o filtrar csv)")
    p.add_argument("--n-tickers", type=int, default=12, help="valores del universo sintetico")
    p.add_argument("--n-barras", type=int, default=2600, help="barras del universo sintetico")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--desde", default="2012-01-01", help="fecha inicial (yahoo)")
    p.add_argument("--largo", type=int, default=40, help="barras de la forma diaria")
    p.add_argument("--horizonte", type=int, default=10, help="barras a predecir hacia delante")
    p.add_argument("--noticias", default="sintetica", choices=["sintetica", "csv", "ninguna"])
    p.add_argument("--noticias-csv", default="datos/noticias.csv")
    p.add_argument("--senal-noticias", type=float, default=0.0,
                   help="0 = noticias sin ventaja; >0 inyecta ventaja para el experimento de ablacion")
    p.add_argument("-k", type=int, default=30, help="numero de vecinos")
    p.add_argument("--umbral-confianza", type=float, default=0.35)


def cmd_demo(args):
    """Una pasada corta de principio a fin para ver que todo encaja."""
    args.n_tickers, args.n_barras = 8, 1600
    banco, pre, ctx, reg = montar_banco(args)
    corte = banco.meta["fecha"].quantile(0.7)
    tr = banco.subconjunto((banco.meta["fecha"] <= corte).to_numpy())
    te = banco.subconjunto((banco.meta["fecha"] > corte).to_numpy())
    mem = MemoriaPatrones(banco.cfg, k=args.k, umbral_confianza=args.umbral_confianza).ajustar(tr)
    print(f"\nMemoria: {len(mem)} casos hasta {corte.date()}. Tres predicciones de ejemplo:\n")
    for i in np.linspace(0, len(te) - 1, 3).astype(int):
        print(mem.predecir(te, int(i)).informe(5))
        real = te.y["ret"].iloc[int(i)]
        print(f"  (la realidad fue {real * 100:+.2f}%)\n")
    print("Para medir de verdad si acierta: python bot.py evaluar")


def cmd_simular(args):
    banco, pre, ctx, reg = montar_banco(args)
    sim.simular(banco, rondas=args.rondas, k=args.k, umbral_confianza=args.umbral_confianza,
                log=args.log, guardar_en=args.guardar, interactivo=args.interactivo)


def cmd_evaluar(args):
    banco, pre, ctx, reg = montar_banco(args)
    print("\nEvaluando la MEMORIA DE CASOS (walk-forward, aprendiendo de sus errores)...")
    res, mem, pesos = ev.evaluar_memoria(banco, n_pliegues=args.pliegues, k=args.k,
                                         umbral_confianza=args.umbral_confianza,
                                         aprender_online=not args.sin_aprendizaje)
    ev.informe(res, banco_test=banco, coste_bps=args.coste_bps, horizonte=args.horizonte,
               titulo="MEMORIA DE CASOS")
    if args.comparar:
        print("\nComparando con un modelo supervisado (mismos datos, mismo walk-forward)...")
        for tipo in ["logistica", "gbm"]:
            r2 = ev.evaluar_supervisado(banco, tipo=tipo, n_pliegues=args.pliegues, verbose=False)
            m2 = ev.metricas(r2)
            print(f"  {tipo:<10} acierto {m2.get('acierto', float('nan')):.2%}  "
                  f"AUC {m2.get('auc', float('nan')):.4f}  Brier {m2.get('brier', float('nan')):.4f}")
    print("\nEvolucion de los pesos (que grupo se gano la confianza del sistema):")
    print(pesos.set_index("pliegue").round(3).to_string())
    if args.guardar:
        mem.guardar(args.guardar)
        print(f"\nMemoria final guardada en {args.guardar}")
    if args.csv_resultados:
        res.to_csv(args.csv_resultados, index=False)
        print(f"Predicciones guardadas en {args.csv_resultados}")


def cmd_predecir(args):
    banco, pre, ctx, reg = montar_banco(args)
    objetivo = args.ticker
    if objetivo is None:
        objetivo = banco.meta["ticker"].iloc[-1]
    casos = banco.subconjunto((banco.meta["ticker"] == objetivo).to_numpy())
    if len(casos) == 0:
        raise SystemExit(f"No hay casos para {objetivo}. Disponibles: "
                         f"{sorted(banco.meta.ticker.unique())}")
    pasado = banco.subconjunto((banco.meta["fecha"] < casos.meta["fecha"].iloc[-1]).to_numpy())
    mem = MemoriaPatrones(banco.cfg, k=args.k, umbral_confianza=args.umbral_confianza).ajustar(pasado)
    pred = mem.predecir(casos, len(casos) - 1)
    print("\n" + "=" * 70)
    print(f"  PREDICCION PARA {objetivo}  (ultimo tramo disponible)")
    print("=" * 70)
    print(pred.informe(8))
    print("\nCono del camino esperado (retorno acumulado en multiplos de ATR):")
    for t in range(len(pred.camino)):
        c, lo, hi = pred.camino[t], pred.camino_p10[t], pred.camino_p90[t]
        barra = _barra(c, lo, hi)
        print(f"  +{t + 1:>2} barras  p10 {lo:+5.1f}  esperado {c:+5.1f}  p90 {hi:+5.1f}  {barra}")
    print("\nAviso: esto es una herramienta de estudio con un modelo estadistico simple; "
          "no es una recomendacion de inversion.")


def _barra(c, lo, hi, ancho=40):
    izq, der = -6.0, 6.0
    esc = lambda x: int(np.clip((x - izq) / (der - izq), 0, 1) * (ancho - 1))
    linea = [" "] * ancho
    for a in range(esc(lo), esc(hi) + 1):
        linea[a] = "-"
    linea[esc(0.0)] = "|"
    linea[esc(c)] = "#"
    return "".join(linea)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pd_ = sub.add_parser("demo", help="pasada rapida de ejemplo")
    _comunes(pd_)
    pd_.set_defaults(func=cmd_demo)

    ps = sub.add_parser("simular", help="juega partidas y aprende de cada una")
    _comunes(ps)
    ps.add_argument("--rondas", type=int, default=60)
    ps.add_argument("--log", default="aprendizaje.csv")
    ps.add_argument("--guardar", default=None, help="ruta .npz donde guardar la memoria")
    ps.add_argument("--interactivo", action="store_true", help="pausa entre partidas")
    ps.set_defaults(func=cmd_simular)

    pe = sub.add_parser("evaluar", help="walk-forward con baselines, backtest y calibracion")
    _comunes(pe)
    pe.add_argument("--pliegues", type=int, default=5)
    pe.add_argument("--coste-bps", type=float, default=10.0)
    pe.add_argument("--comparar", action="store_true", help="tambien entrena modelos supervisados")
    pe.add_argument("--sin-aprendizaje", action="store_true",
                    help="desactiva el reajuste online de pesos (para comparar)")
    pe.add_argument("--guardar", default=None)
    pe.add_argument("--csv-resultados", default=None)
    pe.set_defaults(func=cmd_evaluar)

    pp = sub.add_parser("predecir", help="predice el proximo tramo de un valor")
    _comunes(pp)
    pp.add_argument("--ticker", default=None)
    pp.set_defaults(func=cmd_predecir)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
