"""Monta el universo de datos y el banco de casos a partir de las opciones de la CLI."""
import os
from types import SimpleNamespace

# opciones globales que las ramas de descarga leen sin ensuciar cada firma
_OPCS = SimpleNamespace(intervalo="1d", rango="5y", sectores=None, mercado="datos/mercado")

from . import contexto as mod_ctx
from . import datos as mod_datos
from . import noticias as mod_noticias
from . import ventanas as mod_ventanas


def cargar_universo(fuente="sintetico", carpeta="datos/sintetico", tickers=None,
                    n_tickers=12, n_barras=2600, semilla=42, desde="2012-01-01",
                    verbose=True):
    """Devuelve (precios, meta, indice, registro_de_figuras)."""
    if fuente == "sintetico":
        if os.path.isdir(carpeta) and any(f.endswith(".csv") and not f.startswith("_")
                                          for f in os.listdir(carpeta)):
            if verbose:
                print(f"Leyendo universo sintetico de {carpeta}")
            pre, meta, idx = mod_datos.cargar_carpeta(carpeta)
            reg_ruta = os.path.join(carpeta, "_patrones_plantados.csv")
            reg = None
            if os.path.exists(reg_ruta):
                import pandas as pd
                reg = pd.read_csv(reg_ruta, parse_dates=["fecha"])
            return pre, meta, idx, reg
        if verbose:
            print(f"Generando universo sintetico ({n_tickers} valores, {n_barras} barras)")
        pre, meta, idx, reg = mod_datos.generar_universo(n_tickers, n_barras, semilla)
        mod_datos.guardar_universo(carpeta, pre, meta, idx, reg)
        if verbose:
            print(f"Guardado en {carpeta}")
        return pre, meta, idx, reg

    if fuente == "csv":
        if verbose:
            print(f"Leyendo CSV de {carpeta}")
        pre, meta, idx = mod_datos.cargar_carpeta(carpeta)
        if tickers:
            pre = {t: pre[t] for t in tickers if t in pre}
        return pre, meta, idx, None

    if fuente == "mercado":
        from fuentes.descarga import cargar_mercado
        intervalo = getattr(_OPCS, "intervalo", "1d")
        pre, meta = cargar_mercado(getattr(_OPCS, "mercado", "datos/mercado"), intervalo)
        if tickers:
            pre = {t: pre[t] for t in tickers if t in pre}
            meta = meta[meta.ticker.isin(pre.keys())].reset_index(drop=True)
        if verbose:
            print(f"Universo descargado: {len(pre)} valores en {intervalo}")
        return pre, meta, None, None

    if fuente == "yahoo":
        from fuentes import yahoo as fy
        if not tickers:
            raise SystemExit("Con --fuente yahoo hay que pasar --tickers AAPL,MSFT,...")
        intervalo = getattr(args_intervalo, "intervalo", "1d") if False else "1d"
        pre, meta = fy.universo(tickers, intervalo=getattr(_OPCS, "intervalo", "1d"),
                                rango=getattr(_OPCS, "rango", "5y"),
                                sectores=getattr(_OPCS, "sectores", None), verbose=verbose)
        return pre, meta, None, None

    raise SystemExit(f"fuente desconocida: {fuente}")


def montar_banco(args, verbose=True):
    """Universo -> contexto -> noticias -> banco de casos etiquetado."""
    _OPCS.intervalo = getattr(args, "intervalo", "1d")
    _OPCS.rango = getattr(args, "rango", "5y")
    _OPCS.sectores = getattr(args, "sectores", None)
    _OPCS.mercado = getattr(args, "mercado", "datos/mercado")
    pre, meta, idx, reg = cargar_universo(
        args.fuente, args.carpeta, args.tickers, args.n_tickers, args.n_barras,
        args.semilla, args.desde, verbose)
    ctx = mod_ctx.Contexto(pre, meta, idx)

    if args.noticias == "ninguna":
        fn = mod_noticias.SinNoticias()
    elif args.noticias == "reales":
        from fuentes.noticias_reales import NoticiasReales
        fn = NoticiasReales(list(pre.keys()))
        if verbose:
            print(f"Noticias reales (Yahoo): {len(fn.df)} titulares recientes "
                  "(solo para prediccion en vivo, no entrenamiento)")
    elif args.noticias == "csv":
        fn = mod_noticias.NoticiasCSV(args.noticias_csv)
        if verbose:
            print(f"Noticias: {len(fn.df)} titulares de {args.noticias_csv}")
    else:
        fn = mod_noticias.NoticiasSinteticas(pre, meta, senal=args.senal_noticias,
                                             horizonte=args.horizonte)
        if verbose:
            aviso = "" if args.senal_noticias == 0 else \
                f" CON SENAL INYECTADA DE {args.senal_noticias:.2f} (experimento de ablacion)"
            print(f"Noticias: {len(fn.df)} titulares sinteticos{aviso}")

    cfg = mod_ventanas.Config(largo=args.largo, horizonte=args.horizonte,
                              paso=getattr(args, "paso", 1))
    enr = None
    if getattr(args, "con_fundamentales", False) or getattr(args, "con_macro", False):
        from fuentes.enriquecer import Enriquecedor
        enr = Enriquecedor(con_sec=getattr(args, "con_fundamentales", False),
                           con_fred=getattr(args, "con_macro", False),
                           clave_fred=getattr(args, "clave_fred", None), verbose=verbose)
        if verbose:
            print(f"Enriqueciendo con SEC={getattr(args,'con_fundamentales',False)} "
                  f"FRED={getattr(args,'con_macro',False)}...")
    banco = mod_ventanas.construir(pre, ctx, fn, cfg, verbose=verbose, enriquecer=enr)
    banco = mod_ventanas.etiquetar(banco, reg)
    if verbose:
        print(banco.resumen())
    return banco, pre, ctx, reg
