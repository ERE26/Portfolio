"""Harness COMUN de los angulos intradia NUEVOS sobre Massive (point-in-time, limpio).

Reutiliza la infra ya commiteada (NO la duplica):
  - fuentes.massive.barras_intradia(ticker, '5Min', desde, hasta)  -> 5m RTH, UTC->NY.
  - universo_pit.universo_asof / grouped_asof / set_common_stocks   -> universo PIT
    survivorship-free (GROUPED del pasado estricto, CS activos+deslistados, megacaps
    excluidas por proxy de dolar-volumen).

Aporta lo que esos angulos nuevos necesitan y que el script de ETFs del repo no cubre:
  - Muestreo point-in-time de pares (ticker, dia) sobre el universo de ACCIONES (y una
    variante ETFs+large-caps), con 3 semillas independientes.
  - VWAP de sesion, ATR intradia, bootstrap IC95 por operacion, tabla markdown y
    veredicto de robustez (edge solo si IC95>0 en LAS 3 semillas).

Costes homogeneos con el resto del repo: SLIP=4bps/lado, SLIP_STOP=8bps en stops,
+BORROW=3bps por operacion corta. SOLO research/paper.
"""
import random

import numpy as np
import pandas as pd

import universo_pit as U
from fuentes import massive
from fuentes.http import Cliente

SLIP = 0.0004          # 4 bps por lado
SLIP_STOP = 0.0008     # 8 bps extra al tocar stop
BORROW_CORTO = 0.0003  # ~3 bps por operacion corta (borrow aprox)
PAUSA = 0.3            # plan Starter aguanta este ritmo
BUFFER_DIAS_FIN = 30   # colchon al final de la ventana (datos asentados)

ETFS_LIQUIDOS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY",
                 "XLP", "XLI", "XLU", "XLB", "XLRE", "GLD", "SLV", "TLT", "SMH"]


def cliente(pausa=None):
    """Cliente Massive con cache larga (reutiliza datos/cache de la infra existente).
    La pausa entre peticiones se puede bajar via env MASSIVE_PAUSA (Starter = ilimitado;
    util en re-corridas con todo cacheado)."""
    if pausa is None:
        pausa = float(__import__("os").environ.get("MASSIVE_PAUSA", PAUSA))
    cm = massive.ClienteMassive.__new__(massive.ClienteMassive)
    cm.cli = Cliente(cache_dir="datos/cache", ttl=30 * 24 * 3600, reintentos=5,
                     pausa=2.0, timeout=90)
    cm.cli.sesion.headers["Authorization"] = f"Bearer {massive._claves()}"
    cm.pausa_min = pausa
    cm._ultima = 0.0
    cm.llamadas = 0
    return cm


def _ventana(meses):
    fin = pd.Timestamp.now().normalize() - pd.Timedelta(days=BUFFER_DIAS_FIN)
    ini = fin - pd.Timedelta(days=int(meses * 31))
    return ini.strftime("%Y-%m-%d"), fin.strftime("%Y-%m-%d")


def _dias_sesion(desde, hasta, cm):
    spy = massive.barras_diarias("SPY", desde, hasta, cm=cm)
    return [d.strftime("%Y-%m-%d") for d in spy.index]


def _universo_mes(ref_fecha, clase, n, cm, cs_set):
    """Universo PIT del mes (ref_fecha = ultimo dia del mes anterior). clase:
    'CS' acciones in-play (excluye top-5% megacaps); 'LC_ETF' ETFs + top-n large-caps."""
    if clase == "CS":
        return U.universo_asof(ref_fecha, n=n, cm=cm, dias_hist=5, megacap_pctl=95,
                               cs_set=cs_set)
    if clase == "LC_ETF":
        lc = U.universo_asof(ref_fecha, n=30, cm=cm, dias_hist=5, megacap_pctl=100,
                             cs_set=cs_set)
        return list(ETFS_LIQUIDOS) + [t for t in lc if t not in set(ETFS_LIQUIDOS)]
    raise ValueError(clase)


def muestrear(seed, clase="CS", n_tickers=30, n_dias=15, meses=20, cm=None):
    """Lista determinista de pares (ticker, 'YYYY-MM-DD') point-in-time.

    Muestrea n_dias dias de sesion al azar en la ventana; para cada dia toma el
    universo PIT del mes (GROUPED del mes anterior) y elige n_tickers al azar.
    Determinista dado (seed, clase) -> estrategias con la misma clase reusan muestra
    y cache. Dedup de pares.
    """
    rng = random.Random(seed)
    cm = cm or cliente()
    desde, hasta = _ventana(meses)
    dias = _dias_sesion(desde, hasta, cm)
    if len(dias) < n_dias:
        n_dias = len(dias)
    dias_sel = sorted(rng.sample(dias, n_dias))
    cs_set, _ = U.set_common_stocks(cm=cm)
    uni_mes, pares = {}, []
    for d in dias_sel:
        ts = pd.Timestamp(d)
        clave = ts.strftime("%Y-%m")
        if clave not in uni_mes:
            ref = (ts.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            uni_mes[clave] = _universo_mes(ref, clase, max(80, n_tickers * 3), cm, cs_set)
        uni = uni_mes[clave]
        if not uni:
            continue
        for tk in rng.sample(uni, min(n_tickers, len(uni))):
            pares.append((tk, d))
    vistos, out = set(), []
    for p in pares:
        if p not in vistos:
            vistos.add(p); out.append(p)
    return out


def cargar(ticker, fecha, tf="5Min", cm=None):
    """Barras de un (ticker, dia) en sesion regular. Vacio si no hay. Cache HTTP."""
    cm = cm or cliente()
    try:
        return massive.barras_intradia(ticker, tf, fecha, fecha, solo_regular=True, cm=cm)
    except Exception:
        return pd.DataFrame()


def vwap_sesion(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).cumsum()
    vol = df["volume"].cumsum().replace(0, np.nan)
    return (pv / vol).to_numpy()


def atr_dia(df, n=14):
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    if len(tr) < 3:
        return float("nan")
    return float(np.mean(tr[-min(n, len(tr)):]))


def atr_bar(df, n=14):
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=3).mean().to_numpy()


def boot_ci(xs, n=2000, seed=1):
    xs = np.asarray(xs, float)
    if len(xs) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def resumen_R(nombre, trades, seed):
    R = np.array([t["R"] for t in trades], float) if trades else np.array([])
    if len(R) < 5:
        return {"modo": nombre, "seed": seed, "ops": len(R), "acierto": float("nan"),
                "exp": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "pf": float("nan"), "veredicto": "muestra insuf."}
    w, l = R[R > 0], R[R <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    lo, hi = boot_ci(R)
    verd = "EDGE+" if lo > 0 else ("sin edge" if hi > 0 else "PIERDE")
    return {"modo": nombre, "seed": seed, "ops": len(R), "acierto": len(w) / len(R),
            "exp": float(R.mean()), "lo": lo, "hi": hi, "pf": pf, "veredicto": verd}


def tabla_md(filas, unidad="R"):
    cab = (f"| modo | semilla | ops | acierto | expectancy ({unidad}) | IC95 | PF | veredicto |\n"
           "|---|---|---:|---:|---:|---|---:|---|")
    out = [cab]
    for f in filas:
        if f["ops"] < 5 or f["exp"] != f["exp"]:
            out.append(f"| {f['modo']} | {f['seed']} | {f['ops']} | - | - | - | - | {f['veredicto']} |")
            continue
        ic = f"[{f['lo']:+.3f}, {f['hi']:+.3f}]"
        pf = "inf" if f["pf"] == float("inf") else f"{f['pf']:.2f}"
        out.append(f"| {f['modo']} | {f['seed']} | {f['ops']} | {f['acierto']*100:.0f}% | "
                   f"{f['exp']:+.3f} | {ic} | {pf} | {f['veredicto']} |")
    return "\n".join(out)


def veredicto_global(filas_perseed):
    ok = [f for f in filas_perseed if f["ops"] >= 20 and f["lo"] == f["lo"]]
    if len(ok) < len(filas_perseed) or not ok:
        return "NO CONCLUYENTE (muestra insuficiente en alguna semilla)"
    if all(f["lo"] > 0 for f in ok):
        return "EDGE ROBUSTO (IC95>0 en las 3 semillas)"
    if all(f["hi"] < 0 for f in ok):
        return "SIN EDGE / PIERDE en las 3 semillas"
    return "NO ROBUSTO (no sobrevive a las 3 semillas)"
