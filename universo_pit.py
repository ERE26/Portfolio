"""Universo POINT-IN-TIME survivorship-free (solo EOD, via Massive Starter).

El sesgo de supervivencia infla todo backtest que muestrea el universo liquido
de HOY: los valores que quebraron o se deslistaron (justo los que mas cayeron)
no estan en la muestra. Para quitarlo hace falta un universo *as-of*: la lista
de valores que cotizaban en una fecha pasada, CON los que despues desaparecieron.

Fuente usada: el endpoint GROUPED diario de Massive
  GET /v2/aggs/grouped/locale/us/market/stocks/{date}
Una sola llamada devuelve TODAS las acciones US que cotizaron ese dia (~12.600),
incluidas las que hoy estan deslistadas pero ESE dia seguian vivas. Eso es
exactamente un corte point-in-time sin supervivencia. (Verificado en Starter:
status OK, ~12.600 tickers/dia, llamadas ilimitadas.) No hizo falta el fallback
de bajar barras_diarias ticker a ticker.

Filtro de tipo: GROUPED trae tambien ETFs/ETNs; para quedarnos con ACCIONES
(como el baseline "stocks in play") intersectamos con el conjunto de common
stocks (type=CS) del endpoint de referencia, ACTIVOS + DESLISTADOS (active=false
trae los de baja con delisted_utc), asi un deslistado no se cae por el filtro de
tipo. El type de un valor no cambia con el tiempo, asi que usar la referencia de
hoy para clasificar el tipo no reintroduce supervivencia (la *pertenencia* al
universo la decide GROUPED, que si es point-in-time).

Mega-caps: se excluyen por PROXY = percentil alto del dollar-volume as-of (no es
market cap real; es una aproximacion documentada). El baseline sesgado operaba
mid-caps; aqui quitamos el 1% superior por dollar-volume (las AAPL/NVDA/MSFT del
dia) para comparar manzanas con manzanas.

Todo es solo lectura / EOD. Cache en datos/cache/ (gitignored): un parquet por
dia GROUPED y uno para el set de common stocks, de modo que re-correr es barato.
"""
import datetime as dt
import os

import numpy as np
import pandas as pd

from fuentes import massive as mv

CACHE = "datos/cache"
CS_PARQUET = os.path.join(CACHE, "cs_tickers.parquet")

# Filtros (mismos umbrales que el universo sesgado de backtest_intradia._universo)
PRECIO_MIN, PRECIO_MAX = 5.0, 5000.0
DOLLAR_VOL_MIN = 10_000_000.0     # dollar_vol_20d > 10M
MEGACAP_PCTL = 95.0               # excluir top 5% por dollar-vol (proxy de mega/large-cap)
# Nota: con top-1% el universo top-150 es casi todo large-cap SUPERVIVIENTE (~0.7%
# deslistados hoy: insuficiente). Con top-5% caemos a la banda mid/large liquida
# (3-4% deslistados hoy, consistente entre fechas) sin perder liquidez para el ORB.
# Mas abajo (top-20%) el % de deslistados sube a ~5-9%: la supervivencia crece al
# bajar en el espectro de capitalizacion (caveat documentado en el informe).


def _cliente(cm=None):
    # Starter: llamadas ilimitadas -> sin la pausa de 12,5s del plan gratis.
    return cm or mv.ClienteMassive(pausa_min=0.0, cache_dir=CACHE)


def set_common_stocks(cm=None, max_paginas=60):
    """Conjunto de tickers type=CS (ACTIVOS + DESLISTADOS). Cacheado en parquet.

    active=false incluye los deslistados (con delisted_utc), asi el filtro de
    tipo no borra un valor solo por haber desaparecido. Devuelve (set_cs,
    df_delisted) donde df_delisted lista los deslistados con su fecha de baja.
    """
    if os.path.exists(CS_PARQUET):
        t = pd.read_parquet(CS_PARQUET)
        cs = set(t["ticker"].tolist())
        dl = t[t["active"] == False][["ticker", "delisted_utc"]].copy()
        return cs, dl
    cm = _cliente(cm)
    filas = []
    for activo in (True, False):
        params = {"market": "stocks", "type": "CS", "active": str(activo).lower(),
                  "limit": 1000}
        d = cm.get("/v3/reference/tickers", params)
        for i in range(max_paginas):
            for r in (d.get("results") or []):
                filas.append({"ticker": r.get("ticker"), "active": bool(r.get("active")),
                              "delisted_utc": r.get("delisted_utc")})
            nxt = d.get("next_url")
            if not nxt or i == max_paginas - 1:
                break
            d = cm.get(nxt)
    t = pd.DataFrame(filas).dropna(subset=["ticker"]).drop_duplicates("ticker")
    os.makedirs(CACHE, exist_ok=True)
    t.to_parquet(CS_PARQUET)
    cs = set(t["ticker"].tolist())
    dl = t[t["active"] == False][["ticker", "delisted_utc"]].copy()
    return cs, dl


def grouped_asof(fecha, cm=None):
    """Tabla GROUPED de un dia, cacheada en parquet: [ticker, close, volume,
    dollar_vol]. DataFrame vacio si la fecha no fue sesion (finde/festivo)."""
    ruta = os.path.join(CACHE, f"grouped_{fecha}.parquet")
    if os.path.exists(ruta):
        return pd.read_parquet(ruta)
    cm = _cliente(cm)
    g = mv.grouped_diario(fecha, cm=cm)
    if len(g):
        g = g[["ticker", "close", "volume"]].copy()
        g["close"] = g["close"].astype(float)
        g["volume"] = g["volume"].astype(float)
        g["dollar_vol"] = g["close"] * g["volume"]
    else:
        g = pd.DataFrame(columns=["ticker", "close", "volume", "dollar_vol"])
    os.makedirs(CACHE, exist_ok=True)
    g.to_parquet(ruta)
    return g


def _sesiones_previas(fecha, n, cm=None, max_calendario=60):
    """Lista [(fecha_str, df_grouped), ...] de las n sesiones con datos ESTRICTAMENTE
    anteriores a `fecha`, de la mas reciente a la mas antigua."""
    cm = _cliente(cm)
    d = pd.Timestamp(fecha).normalize()
    out, back = [], 1
    while len(out) < n and back <= max_calendario:
        ds = (d - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        g = grouped_asof(ds, cm=cm)
        if len(g):
            out.append((ds, g))
        back += 1
    return out


def tabla_asof(fecha, dias_hist=20, cm=None):
    """Tabla as-of de `fecha`: por ticker, precio (cierre de la ultima sesion
    previa) y dollar_vol_20d (media del dollar-vol en las ~dias_hist sesiones
    previas). Point-in-time: solo usa el pasado estricto de `fecha`."""
    ses = _sesiones_previas(fecha, dias_hist, cm=cm)
    if not ses:
        return pd.DataFrame(columns=["ticker", "close", "dollar_vol_20d", "n_ses"])
    # precio = cierre de la sesion mas reciente (ses[0]); candidatura = cotizaba esa sesion
    ultimo = ses[0][1].set_index("ticker")["close"]
    dv = {}
    for _, g in ses:
        for tk, v in zip(g["ticker"].to_numpy(), g["dollar_vol"].to_numpy()):
            dv.setdefault(tk, []).append(float(v))
    filas = []
    for tk, px in ultimo.items():
        lst = dv.get(tk, [])
        if not lst:
            continue
        filas.append((tk, float(px), float(np.mean(lst)), len(lst)))
    return pd.DataFrame(filas, columns=["ticker", "close", "dollar_vol_20d", "n_ses"])


def universo_asof(fecha, n=150, cm=None, dias_hist=20, solo_acciones=True,
                  megacap_pctl=MEGACAP_PCTL, cs_set=None):
    """Universo point-in-time survivorship-free as-of `fecha` (lista de tickers).

    - candidatos = los que cotizaban en la sesion previa a `fecha` (si deslistaron
      DESPUES, ese dia siguen siendo candidatos: ese es justo el punto).
    - filtros: precio en [5,5000]; dollar_vol_20d > 10M.
    - mega-caps excluidas por PROXY: top (100-megacap_pctl)% por dollar_vol_20d.
    - top-n por dollar_vol_20d desc.
    """
    cm = _cliente(cm)
    t = tabla_asof(fecha, dias_hist=dias_hist, cm=cm)
    if t.empty:
        return []
    if solo_acciones:
        if cs_set is None:
            cs_set, _ = set_common_stocks(cm=cm)
        t = t[t["ticker"].isin(cs_set)]
    t = t[(t["close"] >= PRECIO_MIN) & (t["close"] <= PRECIO_MAX)]
    t = t[t["dollar_vol_20d"] > DOLLAR_VOL_MIN]
    if t.empty:
        return []
    if megacap_pctl and megacap_pctl < 100 and len(t) > 20:
        tope = np.percentile(t["dollar_vol_20d"].to_numpy(), megacap_pctl)
        t = t[t["dollar_vol_20d"] < tope]        # fuera las mega-caps (proxy)
    t = t.sort_values("dollar_vol_20d", ascending=False)
    return t["ticker"].head(n).tolist()


def construir_tabla_diaria(fechas, cm=None, dias_hist=20, out_parquet=None):
    """Tabla diaria larga [fecha, ticker, close, volume, dollar_vol] cubriendo
    cada `fecha` objetivo y sus dias_hist sesiones previas. Cacheada en parquet
    (requisito del paso 2). Util para auditar/validar el universo."""
    cm = _cliente(cm)
    sesiones = set()
    for f in fechas:
        for ds, _ in _sesiones_previas(f, dias_hist, cm=cm):
            sesiones.add(ds)
        # la propia fecha tambien (por si se quiere auditar ese dia)
        if len(grouped_asof(f, cm=cm)):
            sesiones.add(f)
    partes = []
    for ds in sorted(sesiones):
        g = grouped_asof(ds, cm=cm)
        if len(g):
            gg = g.copy()
            gg["fecha"] = ds
            partes.append(gg[["fecha", "ticker", "close", "volume", "dollar_vol"]])
    tabla = (pd.concat(partes, ignore_index=True) if partes
             else pd.DataFrame(columns=["fecha", "ticker", "close", "volume", "dollar_vol"]))
    if out_parquet:
        os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
        tabla.to_parquet(out_parquet)
    return tabla


def validar_survivorship(fecha_antigua, n=150, cm=None, sesion_reciente=None):
    """VALIDACION: % del universo as-of `fecha_antigua` que HOY esta deslistado.

    Debe ser >0: si fuera ~0, el universo no esta metiendo deslistados y el
    survivorship-free no esta funcionando. "Deslistado hoy" = ausente de una
    sesion reciente Y/O presente en el set de deslistados (active=false) de la
    referencia. Devuelve un dict con el detalle."""
    cm = _cliente(cm)
    cs_set, dl = set_common_stocks(cm=cm)
    dl_set = set(dl["ticker"].tolist())
    uni = universo_asof(fecha_antigua, n=n, cm=cm, cs_set=cs_set)
    if not uni:
        return {"fecha": fecha_antigua, "n_universo": 0, "error": "universo vacio"}
    # sesion reciente para comprobar quien sigue cotizando
    if sesion_reciente is None:
        hoy = dt.date.today()
        d = hoy - dt.timedelta(days=1)
        rec = None
        for _ in range(10):
            g = grouped_asof(d.isoformat(), cm=cm)
            if len(g):
                rec = set(g["ticker"].tolist()); break
            d -= dt.timedelta(days=1)
        vivos = rec or set()
    else:
        vivos = set(grouped_asof(sesion_reciente, cm=cm)["ticker"].tolist())
    ausentes = [t for t in uni if t not in vivos]          # no cotizan hoy
    en_dl = [t for t in uni if t in dl_set]                 # marcados deslistados
    muertos = sorted(set(ausentes) | set(en_dl))
    return {
        "fecha": fecha_antigua, "n_universo": len(uni),
        "deslistados_hoy": len(muertos),
        "pct_deslistados_hoy": round(len(muertos) / len(uni) * 100, 1),
        "pct_ausentes_sesion_reciente": round(len(ausentes) / len(uni) * 100, 1),
        "pct_en_set_delisted": round(len(en_dl) / len(uni) * 100, 1),
        "ejemplos_muertos": muertos[:15],
    }


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Universo point-in-time survivorship-free")
    ap.add_argument("--fecha", default=None, help="as-of YYYY-MM-DD (def: ~18 meses atras)")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    cm = _cliente()
    fecha = a.fecha or (pd.Timestamp.now() - pd.Timedelta(days=548)).strftime("%Y-%m-%d")
    print(f"### Universo as-of {fecha} (top {a.n})")
    uni = universo_asof(fecha, n=a.n, cm=cm)
    print(f"tickers: {len(uni)}  primeros 20: {uni[:20]}")
    print("\n### Validacion survivorship (deben aparecer deslistados):")
    print(json.dumps(validar_survivorship(fecha, n=a.n, cm=cm), indent=1))
    print(f"\npeticiones a Massive: {cm.llamadas}")
