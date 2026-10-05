"""Backtest de la ANOMALIA OVERNIGHT, sin sesgo de supervivencia (Massive EOD).

Hipotesis (muy documentada): en renta variable EE.UU. la mayor parte del retorno
historico se concentra en el tramo OVERNIGHT (cierre_{t-1} -> apertura_t), y el
tramo INTRADIA (apertura_t -> cierre_t) es plano o negativo. Pregunta honesta:
¿es un edge explotable NETO de costes, o se lo come el doble cruce diario?

Que hace este script (SOLO research/paper, solo lectura):
  1. Reconstruye un panel POINT-IN-TIME de todas las acciones comunes de EE.UU.
     (incluidos deslistados) con el endpoint grouped daily de Massive
     (fuentes.massive.grouped_diario, la misma infra que usa universo_pit) ->
     sin sesgo de supervivencia por construccion: un ticker aparece en la fecha t
     si y solo si cotizo en t.
  2. Descompone el retorno diario de cada accion en OVERNIGHT e INTRADIA.
  3. Mide carteras equal-weight con rebalanceo diario sobre el universo liquido
     AS-OF (seleccion point-in-time, sin look-ahead), con y sin coste.
  4. Metricas: retorno medio diario, anualizado, Sharpe, maxDD, % dias+,
     bootstrap IC95. Compara contra SPY buy&hold y mide el haircut de
     supervivencia (universo con vs sin deslistados).

El universo as-of de este estudio es panel-vectorizado (todas las fechas de una
vez, para poder descomponer ON/ID sobre toda la matriz). Es consistente con el
selector point-in-time de universo_pit.universo_asof (mismo grouped, misma idea
de liquidez as-of); se incluye una verificacion cruzada opcional en --xcheck.

NO cablea en vivo, NO toca gestionar.py ni operativa_*.py. Es un estudio.

Uso:
    python backtest_overnight_massive.py                 # periodo completo (5 a)
    python backtest_overnight_massive.py 2022-01-01 2024-12-31
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

from fuentes import massive as mv

CACHE = "datos/cache"
TRADING_DAYS = 252
# Coste por LADO (una sola direccion de un cruce) en puntos basicos. La estrategia
# overnight y la intradia cruzan DOS veces al dia (entrada + salida), asi que el
# coste diario = 2 * lado. Escenarios de menos a mas conservador para acciones
# liquidas en subastas de apertura/cierre (comision ~0 retail + medio-spread):
COSTES_LADO_PB = [0.0, 1.0, 2.0, 5.0]


# ----------------------------------------------------------------------------- #
#  Datos: panel point-in-time survivorship-free (grouped daily de Massive)       #
# ----------------------------------------------------------------------------- #
def tickers_cs(cm=None, usar_cache=True):
    """Set de tickers de ACCION COMUN (type=CS), activos + deslistados, cacheado
    en JSON (gitignored). Incluir deslistados evita que el filtro de tipo
    reintroduzca sesgo de supervivencia. (No usamos el parquet de universo_pit
    para no depender de pyarrow; misma fuente, /v3/reference/tickers.)"""
    ruta = os.path.join(CACHE, "cs_tickers.json")
    if usar_cache and os.path.exists(ruta):
        try:
            return set(json.load(open(ruta)))
        except Exception:
            pass
    cm = cm or mv.ClienteMassive(pausa_min=0.15)
    out = set()
    for activo in ("true", "false"):
        url, params = "/v3/reference/tickers", {
            "market": "stocks", "type": "CS", "active": activo, "limit": 1000}
        while True:
            d = cm.get(url, params)
            for r in (d.get("results") or []):
                if r.get("ticker"):
                    out.add(r["ticker"])
            nxt = d.get("next_url")
            if not nxt:
                break
            url, params = nxt, None
    if out:
        os.makedirs(CACHE, exist_ok=True)
        json.dump(sorted(out), open(ruta, "w"))
    return out


def construir_panel(desde, hasta, cm=None, solo_cs=True, usar_cache=True, verbose=True):
    """Panel largo [fecha, ticker, open, close, volume, dolar_vol] con el grouped
    diario de cada sesion en [desde, hasta]. Survivorship-free por construccion.
    Cachea el panel ensamblado (pickle, gitignored) y reusa la cache por-dia del
    cliente HTTP, asi que reejecutar no vuelve a bajar nada."""
    etiqueta = f"panel_on_{desde}_{hasta}_{'cs' if solo_cs else 'all'}"
    ruta = os.path.join(CACHE, etiqueta + ".pkl")
    if usar_cache and os.path.exists(ruta):
        return pd.read_pickle(ruta)
    cm = cm or mv.ClienteMassive(pausa_min=0.15)
    cs = tickers_cs(cm) if solo_cs else None
    dias = list(pd.bdate_range(pd.Timestamp(desde), pd.Timestamp(hasta)))
    trozos = []
    for i, dia in enumerate(dias):
        f = dia.date().isoformat()
        try:
            g = mv.grouped_diario(f, ajustado=True, cm=cm)
        except Exception as e:
            if verbose:
                print(f"  {f}: ERR {type(e).__name__} (saltado)")
            continue
        if not len(g):
            continue  # festivo / fin de semana
        if cs is not None:
            g = g[g["ticker"].isin(cs)]
        g = g[(g["open"] > 0) & (g["close"] > 0) & (g["volume"] > 0)]
        if not len(g):
            continue
        trozos.append(pd.DataFrame({
            "fecha": dia, "ticker": g["ticker"].values,
            "open": g["open"].values.astype("float64"),
            "close": g["close"].values.astype("float64"),
            "volume": g["volume"].values.astype("float64"),
            "dolar_vol": (g["close"].values * g["volume"].values).astype("float64"),
        }))
        if verbose and (i % 100 == 0 or i == len(dias) - 1):
            print(f"  {f}: {len(trozos[-1])} valores  ({i + 1}/{len(dias)} dias habiles)")
    if not trozos:
        raise RuntimeError("No se descargo ninguna sesion; revisa fechas/clave.")
    panel = pd.concat(trozos, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    try:
        panel.to_pickle(ruta)
    except Exception as e:
        if verbose:
            print(f"  (no se pudo cachear panel: {e})")
    return panel


class Contexto:
    """Pivotes fecha x ticker + liquidez movil AS-OF cierre t-1 (sin look-ahead).
    La liquidez (mediana movil de dolar-vol, DESPLAZADA 1 dia) y el precio previo
    son la informacion con la que se decide el universo que se operara en t."""

    def __init__(self, panel, ventana_liquidez=21):
        self.ventana = ventana_liquidez
        self.cierre = panel.pivot_table("close", "fecha", "ticker")
        self.apertura = panel.pivot_table("open", "fecha", "ticker")
        self.dolar_vol = panel.pivot_table("dolar_vol", "fecha", "ticker")
        self.fechas = list(self.cierre.index)
        self.liq = (self.dolar_vol.rolling(ventana_liquidez,
                    min_periods=max(5, ventana_liquidez // 2)).median().shift(1))
        self.precio_prev = self.cierre.shift(1)


# ----------------------------------------------------------------------------- #
#  Retornos y mascaras de universo (vectorizado sobre pivotes fecha x ticker)    #
# ----------------------------------------------------------------------------- #
def matrices_retorno(ctx):
    """ON = apertura_t/cierre_{t-1}-1 ; ID = cierre_t/apertura_t-1 ;
    CC = cierre_t/cierre_{t-1}-1. NaN donde falte barra; se recortan |r|>0.5 por
    barra como dato sucio (un -/+50% overnight en liquidas casi siempre es error
    de tick/split, no mercado). Afecta igual a ON, ID y CC."""
    on = ctx.apertura / ctx.cierre.shift(1) - 1.0
    idd = ctx.cierre / ctx.apertura - 1.0
    cc = ctx.cierre / ctx.cierre.shift(1) - 1.0
    for m in (on, idd, cc):
        m[m.abs() > 0.5] = np.nan
    return on, idd, cc


def mascara_universo(ctx, min_precio=5.0, min_dolar_vol=1e6, top_n=None,
                     momentum_excluir_peor=None, solo_tickers=None):
    """Matriz booleana fecha x ticker: True si el ticker es negociable AS-OF la
    fecha (precio y dolar-vol previos sobre umbral; opc. top_n por liquidez; opc.
    excluir peor quintil de momentum 21d; opc. restringir a un set)."""
    liq, px = ctx.liq, ctx.precio_prev
    mask = (px >= min_precio) & (liq >= min_dolar_vol) & liq.notna() & px.notna()
    if solo_tickers is not None:
        fuera = [c for c in mask.columns if c not in set(solo_tickers)]
        mask[fuera] = False
    if momentum_excluir_peor is not None:
        mom = ctx.cierre.shift(1) / ctx.cierre.shift(22) - 1.0
        mom = mom.where(mask)
        umbral = mom.quantile(momentum_excluir_peor, axis=1)
        mask = mask & mom.ge(umbral, axis=0)
    if top_n is not None:
        rank = liq.where(mask).rank(axis=1, ascending=False, method="first")
        mask = mask & rank.le(top_n)
    return mask


def cartera_equal_weight(ret, mask):
    """Retorno diario equal-weight: media de los retornos validos de los tickers
    del universo (solo nombres EJECUTABLES, con retorno no nulo)."""
    r = ret.where(mask)
    diaria = r.mean(axis=1, skipna=True)
    n = r.notna().sum(axis=1)
    return diaria.where(n > 0).rename("ret"), n.rename("n")


# ----------------------------------------------------------------------------- #
#  Metricas                                                                      #
# ----------------------------------------------------------------------------- #
def _max_drawdown(equity):
    return float((equity / equity.cummax() - 1.0).min())


def metricas(r, nombre="", bootstrap=2000, seed=7):
    """r: retornos diarios. Metricas + bootstrap IID del IC95 del retorno medio
    diario y del Sharpe."""
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {"nombre": nombre, "n_dias": 0}
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    equity = (1.0 + r).cumprod()
    cagr = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1.0)
    sharpe = float(mu / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    arr = r.to_numpy()
    muestras = arr[rng.integers(0, n, size=(bootstrap, n))]
    mu_bs = muestras.mean(axis=1)
    sd_bs = muestras.std(axis=1, ddof=1)
    sh_bs = np.where(sd_bs > 0, mu_bs / sd_bs * np.sqrt(TRADING_DAYS), np.nan)
    ic_mu = np.percentile(mu_bs, [2.5, 97.5])
    ic_sh = np.nanpercentile(sh_bs, [2.5, 97.5])
    return {
        "nombre": nombre, "n_dias": n,
        "ret_medio_diario_pb": round(mu * 1e4, 3),
        "ret_medio_diario_ic95_pb": [round(ic_mu[0] * 1e4, 3), round(ic_mu[1] * 1e4, 3)],
        "anualizado_cagr_%": round(cagr * 100, 2),
        "anualizado_aritm_%": round(mu * TRADING_DAYS * 100, 2),
        "vol_anual_%": round(sd * np.sqrt(TRADING_DAYS) * 100, 2),
        "sharpe": round(sharpe, 3),
        "sharpe_ic95": [round(ic_sh[0], 3), round(ic_sh[1], 3)],
        "maxDD_%": round(_max_drawdown(equity) * 100, 2),
        "pct_dias_positivos": round(float((r > 0).mean()) * 100, 1),
        "total_%": round((equity.iloc[-1] - 1.0) * 100, 2),
    }


def aplicar_coste(r_bruto, cruces_por_dia, coste_lado_pb):
    return r_bruto - cruces_por_dia * coste_lado_pb * 1e-4


def spy_benchmark(desde, hasta, cm):
    b = mv.barras_diarias("SPY", desde, hasta, ajustado=True, cm=cm)
    return (b["close"].pct_change().dropna(),
            (b["open"] / b["close"].shift(1) - 1.0).dropna(),
            (b["close"] / b["open"] - 1.0).dropna())


# ----------------------------------------------------------------------------- #
#  Main                                                                          #
# ----------------------------------------------------------------------------- #
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    xcheck = "--xcheck" in sys.argv
    desde = args[0] if len(args) > 0 else "2021-10-06"
    hasta = args[1] if len(args) > 1 else "2026-10-03"
    cm = mv.ClienteMassive(pausa_min=0.15)

    print(f"# Panel point-in-time {desde} -> {hasta} (CS, incl. deslistados)")
    panel = construir_panel(desde, hasta, cm=cm, solo_cs=True, verbose=False)
    ctx = Contexto(panel, ventana_liquidez=21)
    fechas = pd.DatetimeIndex(ctx.fechas)
    print(f"  sesiones={len(fechas)}  tickers_unicos={panel['ticker'].nunique()}  "
          f"filas={len(panel):,}")

    on, idd, cc = matrices_retorno(ctx)

    ult = ctx.cierre.tail(10)
    survivors = set(ult.columns[ult.notna().any(axis=0)])
    print(f"  survivors (vivos al final): {len(survivors)} de {panel['ticker'].nunique()}")

    universos = {
        "liq_1M": mascara_universo(ctx, min_precio=5, min_dolar_vol=1e6),
        "liq_10M": mascara_universo(ctx, min_precio=5, min_dolar_vol=1e7),
        "top500": mascara_universo(ctx, min_precio=5, min_dolar_vol=1e6, top_n=500),
        "liq_1M_survivors": mascara_universo(ctx, min_precio=5, min_dolar_vol=1e6,
                                             solo_tickers=survivors),
        "liq_1M_exmom": mascara_universo(ctx, min_precio=5, min_dolar_vol=1e6,
                                         momentum_excluir_peor=0.2),
    }
    for k, m in universos.items():
        print(f"  universo {k}: nombres/dia mediana={int(m.sum(axis=1).median())}")

    resultados, series = {}, {}

    def registra(clave, ret_matrix, mask, cruces):
        r_bruto, n = cartera_equal_weight(ret_matrix, mask)
        r_bruto = r_bruto.dropna()
        series[clave] = r_bruto
        fila = {"bruto": metricas(r_bruto, clave + " BRUTO")}
        for lado in COSTES_LADO_PB:
            fila[f"neto_{lado:g}pb_lado"] = metricas(
                aplicar_coste(r_bruto, cruces, lado),
                f"{clave} NETO {lado:g}pb/lado ({cruces * lado:g}pb/dia)")
        fila["nombres_dia_mediana"] = int(n.reindex(r_bruto.index).median())
        resultados[clave] = fila

    registra("OVERNIGHT_liq1M", on, universos["liq_1M"], cruces=2)
    registra("INTRADIA_liq1M", idd, universos["liq_1M"], cruces=2)
    registra("OVERNIGHT_liq10M", on, universos["liq_10M"], cruces=2)
    registra("INTRADIA_liq10M", idd, universos["liq_10M"], cruces=2)
    registra("OVERNIGHT_top500", on, universos["top500"], cruces=2)
    registra("OVERNIGHT_exmom", on, universos["liq_1M_exmom"], cruces=2)
    registra("OVERNIGHT_liq1M_survivors", on, universos["liq_1M_survivors"], cruces=2)
    registra("CC_liq1M", cc, universos["liq_1M"], cruces=0)

    print("# SPY benchmark")
    spy_cc, spy_on, spy_id = spy_benchmark(desde, hasta, cm)
    lo, hi = fechas.min(), fechas.max()
    rec = lambda s: s[(s.index >= lo) & (s.index <= hi)]
    spy = {"SPY_buyhold_CC": metricas(rec(spy_cc), "SPY buy&hold (close-to-close)"),
           "SPY_solo_overnight": metricas(rec(spy_on), "SPY solo overnight"),
           "SPY_solo_intradia": metricas(rec(spy_id), "SPY solo intradia")}

    # verificacion cruzada opcional contra universo_pit.universo_asof (PIT selector
    # ya commiteado). Requiere pyarrow; si no esta, se omite sin romper.
    xcheck_out = None
    if xcheck:
        try:
            import universo_pit as up
            muestra = [str(fechas[i].date()) for i in
                       (len(fechas) // 4, len(fechas) // 2, 3 * len(fechas) // 4)]
            xcheck_out = {}
            for f in muestra:
                uni_pit = set(up.universo_asof(f, n=500, cm=cm, megacap_pctl=100))
                mine = set(mascara_universo(ctx, top_n=500).loc[pd.Timestamp(f)]
                           .pipe(lambda s: s[s].index))
                inter = uni_pit & mine
                xcheck_out[f] = {"n_pit": len(uni_pit), "n_mine": len(mine),
                                 "solape_%": round(100 * len(inter) /
                                                   max(1, len(uni_pit | mine)), 1)}
            print("# xcheck universo_pit.universo_asof:", json.dumps(xcheck_out))
        except Exception as e:
            print(f"# xcheck omitido: {type(e).__name__}: {e}")

    salida = {
        "periodo": {"desde": desde, "hasta": hasta, "sesiones": len(fechas),
                    "primera": str(fechas.min().date()), "ultima": str(fechas.max().date())},
        "universo": {"tickers_unicos": int(panel["ticker"].nunique()),
                     "survivors_final": len(survivors),
                     "nombres_dia": {k: int(m.sum(axis=1).median()) for k, m in universos.items()}},
        "costes_lado_pb": COSTES_LADO_PB,
        "estrategias": resultados, "spy": spy,
        "xcheck_universo_pit": xcheck_out, "llamadas_massive": cm.llamadas,
    }

    outdir = os.environ.get("OUTDIR", ".")
    with open(os.path.join(outdir, "overnight_resumen.json"), "w") as f:
        json.dump(salida, f, indent=1, default=str)
    pd.DataFrame({k: (1 + v).cumprod() for k, v in series.items()}).to_csv(
        os.path.join(outdir, "overnight_equity.csv"))
    print(json.dumps(salida, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
