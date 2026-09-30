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
    p.add_argument("--fuente", default="sintetico",
                   choices=["sintetico", "csv", "yahoo", "mercado"],
                   help="de donde salen los precios ('mercado' = universo descargado)")
    p.add_argument("--carpeta", default="datos/sintetico", help="carpeta de CSVs (csv/sintetico)")
    p.add_argument("--tickers", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="lista separada por comas (yahoo, o filtrar csv)")
    p.add_argument("--n-tickers", type=int, default=12, help="valores del universo sintetico")
    p.add_argument("--n-barras", type=int, default=2600, help="barras del universo sintetico")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--desde", default="2012-01-01", help="fecha inicial (yahoo)")
    p.add_argument("--largo", type=int, default=40, help="barras de la forma diaria")
    p.add_argument("--horizonte", type=int, default=10, help="barras a predecir hacia delante")
    p.add_argument("--noticias", default="sintetica",
                   choices=["sintetica", "reales", "csv", "ninguna"],
                   help="'reales' usa titulares de Yahoo (solo para prediccion en vivo)")
    p.add_argument("--intervalo", default="1d", help="temporalidad Yahoo: 1d, 1wk, 1mo, 1h, 15m...")
    p.add_argument("--rango", default="5y", help="historico a descargar de Yahoo: 2y, 5y, max...")
    p.add_argument("--sectores", type=lambda s: dict(x.split(":") for x in s.split(",")) if s else None,
                   default=None, help="mapa ticker:sector, ej. AAPL:tecnologia,JPM:banca")
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
    if objetivo not in set(banco.meta.ticker):
        raise SystemExit(f"No hay casos para {objetivo}. Disponibles: "
                         f"{sorted(banco.meta.ticker.unique())}")
    # la memoria usa TODO el historico; el caso a predecir es el borde real del
    # grafico (ultima barra), cuyo futuro aun no ha pasado.
    from patrones.ventanas import caso_vivo
    from patrones import noticias as _mn
    mem = MemoriaPatrones(banco.cfg, k=args.k, umbral_confianza=args.umbral_confianza).ajustar(banco)
    fn_vivo = _fuente_noticias(args, pre, ctx)
    vivo = caso_vivo(pre, ctx, fn_vivo, banco.cfg, objetivo)
    pred = mem.predecir(vivo, 0)
    print("\n" + "=" * 70)
    print(f"  PREDICCION PARA {objetivo}  (proximo tramo, desde {vivo.meta.fecha.iloc[0].date()})")
    print("=" * 70)
    print(pred.informe(8))

    if getattr(args, "en_vivo", False):
        _capa_en_vivo(objetivo, pre, banco)

    print("\nCono del camino esperado (retorno acumulado en multiplos de ATR):")
    for t in range(len(pred.camino)):
        c, lo, hi = pred.camino[t], pred.camino_p10[t], pred.camino_p90[t]
        barra = _barra(c, lo, hi)
        print(f"  +{t + 1:>2} barras  p10 {lo:+5.1f}  esperado {c:+5.1f}  p90 {hi:+5.1f}  {barra}")
    print("\nAviso: esto es una herramienta de estudio con un modelo estadistico simple; "
          "no es una recomendacion de inversion.")


def _fuente_noticias(args, pre, ctx):
    """Reconstruye la fuente de noticias elegida, para el caso vivo."""
    from patrones import noticias as mn
    if args.noticias == "reales":
        try:
            from fuentes.noticias_reales import NoticiasReales
            return NoticiasReales(list(pre.keys()))
        except Exception:
            return mn.SinNoticias()
    if args.noticias == "sintetica":
        return mn.NoticiasSinteticas(pre, semilla=args.semilla, senal=args.senal_noticias,
                                     horizonte=args.horizonte)
    return mn.SinNoticias()


def _capa_en_vivo(ticker, precios, banco):
    """Segunda opinion EN VIVO: scanner de TradingView + titulares recientes."""
    print("\n" + "-" * 70)
    print("  CAPA EN VIVO (datos del momento, no entran en el modelo historico)")
    print("-" * 70)
    try:
        from fuentes import tradingview as tv
        from fuentes.http import Cliente
        cli = Cliente(ttl=300)
        ex = None
        if "exchange" in banco.meta.columns:
            m = banco.meta[banco.meta.ticker == ticker]
            ex = m["exchange"].iloc[0] if len(m) else None
        if ex:
            snap = tv.snapshot([tv.como_simbolo_tv(ticker, ex)], cliente=cli)
        else:
            snap = tv.snapshot_auto([ticker], cliente=cli)
        if len(snap):
            print(tv.bloque_informe(snap.iloc[0].to_dict()))
        else:
            print("  TradingView: sin datos para este simbolo.")
    except Exception as e:
        print(f"  TradingView no disponible: {e}")
    try:
        from fuentes.noticias_reales import NoticiasReales
        fn = NoticiasReales([ticker])
        print(fn.resumen_vivo(ticker))
    except Exception as e:
        print(f"  Noticias en vivo no disponibles: {e}")


def _barra(c, lo, hi, ancho=40):
    izq, der = -6.0, 6.0
    esc = lambda x: int(np.clip((x - izq) / (der - izq), 0, 1) * (ancho - 1))
    linea = [" "] * ancho
    for a in range(esc(lo), esc(hi) + 1):
        linea[a] = "-"
    linea[esc(0.0)] = "|"
    linea[esc(c)] = "#"
    return "".join(linea)


def cmd_buscar(args):
    """Analiza un ticker NO visto: figura actual, patrones parecidos (>=5 velas),
    por que se parecen, y prediccion basada en ellos."""
    import numpy as np
    from patrones import contexto as mod_ctx, ventanas as mod_v
    from patrones.figuras import describir, texto
    from patrones.explicar import por_que_parecido
    from patrones.memoria import MemoriaPatrones

    objetivo = args.ticker
    if args.fuente == "mercado":
        from fuentes.descarga import cargar_mercado
        pre, meta = cargar_mercado(args.mercado, intervalo=args.intervalo)
    else:
        _b, pre, _c, _r = montar_banco(args, verbose=True)
        meta = None

    # si el objetivo no esta en el universo, se descarga de Yahoo al vuelo
    if objetivo not in pre:
        try:
            from fuentes import yahoo
            rng = "max" if args.intervalo in ("1d", "1wk", "1mo") else "730d"
            df, _ = yahoo.historico(objetivo, intervalo=args.intervalo, rango=rng)
            pre[objetivo] = df
        except Exception as e:
            raise SystemExit(f"No se pudo obtener {objetivo}: {e}")

    if meta is None:
        meta = pd.DataFrame({"ticker": list(pre),
                             "sector": [(args.sectores or {}).get(t, "desconocido") for t in pre],
                             "nombre": list(pre)})
    if objetivo not in set(meta["ticker"]):
        meta = pd.concat([meta, pd.DataFrame([{"ticker": objetivo, "clase": "accion",
                          "sector": (args.sectores or {}).get(objetivo, "desconocido"),
                          "nombre": objetivo}])], ignore_index=True)

    # el ticker objetivo se EXCLUYE de la memoria (no se ha visto en entrenamiento)
    entren = {t: d for t, d in pre.items() if t != objetivo}
    if not entren:
        raise SystemExit("Hace falta mas de un ticker para entrenar excluyendo el objetivo.")

    cfg = mod_v.Config(largo=args.largo, horizonte=args.horizonte, paso=args.paso)
    # recorta el historico para que un universo grande sea manejable
    if args.anos and args.anos > 0:
        corte = pd.Timestamp.now() - pd.Timedelta(days=int(args.anos * 365))
        minimo = cfg.minimo_historia + cfg.largo + cfg.horizonte + 5
        entren = {t: d[d.index >= corte] for t, d in entren.items()}
        entren = {t: d for t, d in entren.items() if len(d) >= minimo}
    print(f"\nEntrenando con {len(entren)} valores (excluyendo {objetivo}), "
          f"{args.anos:.0f} anos, 1 caso cada {args.paso} velas...")
    ctx = mod_ctx.Contexto({**entren, objetivo: pre[objetivo]}, meta)
    banco = mod_v.construir(entren, ctx, None, cfg, verbose=False)
    if len(banco) < 200:
        raise SystemExit(f"Muy pocos casos ({len(banco)}). Usa mas historico o mas tickers.")
    # con memoria grande, la calibracion de escala usa menos muestras (es solo una
    # referencia para la confianza, no cambia los vecinos)
    import patrones.memoria as _mm
    if not getattr(_mm, "_cal_parcheado", False):
        _orig = _mm.MemoriaPatrones._calibrar_escala
        _mm.MemoriaPatrones._calibrar_escala = \
            lambda self, n_muestra=60, semilla=0: _orig(self, min(n_muestra, 60), semilla)
        _mm._cal_parcheado = True
    mem = MemoriaPatrones(cfg, k=max(args.n_similares * 4, 30),
                          umbral_confianza=args.umbral_confianza).ajustar(banco)
    print(f"Memoria: {len(mem)} patrones aprendidos de {banco.meta.ticker.nunique()} valores.")

    # tramo objetivo = borde real del grafico del ticker no visto
    vivo = mod_v.caso_vivo(pre, ctx, None, cfg, objetivo)
    desc_obj = describir(vivo.formas[0], vivo.formas_vol[0])
    print("\n" + "=" * 72)
    print(f"  TRAMO ACTUAL DE {objetivo}  ({vivo.meta.fecha.iloc[0].date()}, "
          f"ultimas {args.largo} velas de {args.intervalo})")
    print("=" * 72)
    print(f"  Figura detectada: {texto(desc_obj)}")
    print(f"  Picos {desc_obj['n_picos']}, valles {desc_obj['n_valles']}, "
          f"pendiente {desc_obj['pendiente']:+.3f}")

    pred = mem.predecir(vivo, 0)
    vec = pred.vecinos.head(args.n_similares)
    print(f"\n  PATRONES MAS PARECIDOS (>= {args.min_velas} velas, de {len(mem)} aprendidos):")
    for i, (_, v) in enumerate(vec.iterrows(), 1):
        # reconstruir la forma del vecino para describir su figura
        gpos = np.flatnonzero((mem.tickers == v["ticker"]) &
                              (mem.fechas == np.datetime64(pd.Timestamp(v["fecha"]))))
        fig_v = ""
        if len(gpos):
            dv = describir(mem.banco.formas[gpos[0]], mem.banco.formas_vol[gpos[0]])
            fig_v = texto(dv)
        print(f"\n  {i}. {v['ticker']} el {pd.Timestamp(v['fecha']).date()}  "
              f"(similitud {1 - v['distancia']:.0%})")
        if fig_v:
            print(f"     Figura: {fig_v}")
        print(f"     Por que se parece: {por_que_parecido(v, mem.pesos)}")
        print(f"     Lo que paso despues: {v['ret'] * 100:+.2f}% en {args.horizonte} velas "
              f"(recorrido +{v['mfe']:.1f}/-{abs(v['mae']):.1f} ATR)")

    print("\n" + "-" * 72)
    print("  PREDICCION (basada en lo que hicieron esos patrones parecidos)")
    print("-" * 72)
    signo = "AL ALZA" if pred.prob_sube > .5 else "A LA BAJA"
    print(f"  Sesgo {signo}: prob. de subida {pred.prob_sube:.0%}  "
          f"(retorno esperado {pred.ret_esperado * 100:+.2f}% a {args.horizonte} velas)")
    print(f"  Confianza {pred.confianza:.2f}" +
          ("  -> BAJA: no hay patrones suficientemente parecidos, tomar con cautela"
           if pred.abstencion else "  -> los patrones parecidos son consistentes"))
    coinc = vec["ret"].gt(0).mean() if len(vec) else float("nan")
    print(f"  De los {len(vec)} patrones mas parecidos, subieron el {coinc:.0%}")

    if args.en_vivo:
        _capa_en_vivo(objetivo, pre, banco if 'banco' in dir() else None)
    print("\n  (Herramienta de estudio; no es recomendacion de inversion.)")


def cmd_descargar(args):
    """Descarga el universo del catalogo a CSV (uno por ticker y temporalidad)."""
    from fuentes.descarga import descargar_universo
    from fuentes.universo import por_clase, resumen
    r = resumen()
    print(f"Universo: {r['total']} tickers -> {r['por_clase']}")
    tickers = por_clase(args.clase) if args.clase else None
    for iv in args.intervalos:
        lim = {"5m": "~60 dias", "15m": "~60 dias", "30m": "~60 dias",
               "1h": "~2 anos", "1d": "decadas", "1wk": "decadas"}.get(iv, "?")
        print(f"\nDescargando {iv} (historico disponible en Yahoo: {lim})...")
    inf = descargar_universo(tickers=tickers, intervalos=args.intervalos,
                             carpeta=args.carpeta, pausa=args.pausa)
    ok = inf[inf.estado.isin(["ok", "cache"])]
    print(f"\nDescargado: {len(ok)} (ticker,intervalo) con datos | "
          f"{int(ok.velas.sum()):,} velas | errores: {(~inf.estado.isin(['ok','cache'])).sum()}")
    print(f"Guardado en {args.carpeta}/. Ya puedes usar --fuente mercado.")


def cmd_envivo(args):
    """Panel en vivo sin tocar el modelo: scanner, titulares y calendario macro."""
    from fuentes import tradingview as tv, forexfactory as ff
    from fuentes.http import Cliente
    from fuentes.noticias_reales import NoticiasReales
    cli = Cliente(ttl=300)
    print("\n" + "=" * 70 + "\n  ESTADO EN VIVO\n" + "=" * 70)
    try:
        snap = tv.snapshot_auto(args.tickers, region=args.region, cliente=cli)
        cols = ["simbolo", "cierre", "cambio_pct", "rsi_1d", "rsi_1w", "adx",
                "recomendacion_txt"]
        if snap.empty:
            print("\nTradingView: sin datos para esos simbolos.")
        else:
            print("\nTradingView (snapshot):")
            print(snap[cols].round(2).to_string(index=False))
    except Exception as e:
        print(f"  TradingView no disponible: {e}")
    try:
        fn = NoticiasReales(args.tickers, cli)
        for t in args.tickers:
            print("\n" + fn.resumen_vivo(t))
    except Exception as e:
        print(f"  Noticias no disponibles: {e}")
    try:
        cal = ff.calendario("esta_semana", cli)
        alto = cal[cal.impacto == "High"]
        if args.monedas:
            alto = alto[alto.pais.isin(args.monedas)]
        print(f"\nCalendario macro de alto impacto esta semana "
              f"({', '.join(args.monedas or ['todas'])}):")
        if len(alto):
            print(alto[["fecha", "pais", "evento", "prevision", "anterior"]].to_string(index=False))
        else:
            print("  Sin eventos de alto impacto para esas monedas esta semana.")
    except Exception as e:
        print(f"  Calendario no disponible: {e}")


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
    pp.add_argument("--en-vivo", action="store_true",
                    help="anade la segunda opinion de TradingView y titulares recientes")
    pp.set_defaults(func=cmd_predecir)

    pb = sub.add_parser("buscar",
                        help="busca en lo aprendido los patrones mas parecidos a un ticker no visto")
    _comunes(pb)
    pb.add_argument("--ticker", required=True, help="valor a analizar (se EXCLUYE del entrenamiento)")
    pb.add_argument("--n-similares", type=int, default=6, help="cuantos patrones parecidos mostrar")
    pb.add_argument("--min-velas", type=int, default=5, help="minimo de velas del patron (>=5)")
    pb.add_argument("--paso", type=int, default=5,
                    help="submuestreo: 1 caso cada N velas por valor (universos grandes)")
    pb.add_argument("--anos", type=float, default=8.0,
                    help="limita el historico a los ultimos N anos por valor")
    pb.add_argument("--en-vivo", action="store_true", help="anade TradingView y titulares actuales")
    pb.add_argument("--mercado", default="datos/mercado",
                    help="carpeta de CSV descargados (si se usa --fuente mercado)")
    pb.set_defaults(func=cmd_buscar)

    pdl = sub.add_parser("descargar",
                         help="descarga el universo (+100 tickers) a datos/mercado")
    pdl.add_argument("--intervalos", type=lambda s: s.split(","), default=["1d"],
                     help="temporalidades separadas por comas: 1d,1h,15m,5m...")
    pdl.add_argument("--carpeta", default="datos/mercado")
    pdl.add_argument("--clase", choices=["accion", "divisa", "fondo"], default=None,
                     help="descargar solo una clase de activo")
    pdl.add_argument("--pausa", type=float, default=0.3)
    pdl.set_defaults(func=cmd_descargar)

    pv = sub.add_parser("envivo", help="estado en vivo (TradingView + noticias + calendario)")
    pv.add_argument("--tickers", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    default=["AAPL", "MSFT", "NVDA"])
    pv.add_argument("--region", default="america")
    pv.add_argument("--monedas", type=lambda s: [x.strip() for x in s.split(",")] if s else None,
                    default=["USD", "EUR"], help="monedas para el calendario macro")
    pv.set_defaults(func=cmd_envivo)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
