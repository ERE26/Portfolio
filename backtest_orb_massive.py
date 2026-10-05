"""ORB-5min/1min CONFIRMADO sobre universo POINT-IN-TIME SIN SESGO DE SUPERVIVENCIA.

Es el backtest_orb5 pero con dos cambios que importan:
  (a) DATOS: barras de minuto reales de Massive (Starter: 5 anos, deslistados,
      15-min delayed -> irrelevante para historico), via fuentes.massive.barras_intradia.
  (b) UNIVERSO: para cada dia muestreado, universo_pit.universo_asof(dia) — la lista
      de valores que cotizaban ESE dia, CON los que despues se deslistaron. Asi el
      resultado no esta inflado por el sesgo de supervivencia (el baseline sesgado,
      que muestreaba el universo liquido de HOY, daba +0.26R/trade confirmado).

La LOGICA DEL ORB es la MISMA (se importa de backtest_orb5, no se reescribe):
  - rango de apertura = 1a vela; opening_rvol = vol 1a vela / media 1a vela 14 dias
    previos, gate >1; 1a vela alcista (long); RUPTURA CONFIRMADA (entra cuando una
    vela CIERRA sobre el maximo del rango); stop = minimo del rango; salida EOD y
    variante EOD+trailing 2ATR. SLIP=0.0004, SLIP_STOP=0.0008. 1 entrada/ticker-dia.
  - Bootstrap IC95 identico (_boot_ci). 3 semillas independientes, se reportan las 3.
    Veredicto ROBUSTO solo si IC95>0 en las 3.

5Min vs 1Min: se baja 1-min de Massive una vez por ticker-dia y el 5-min se deriva
por resample (verificado: reproduce EXACTO el 5-min nativo de Massive, diff OHLC 0.0).
Asi ambos timeframes comparten las MISMAS velas subyacentes -> comparacion limpia, y
la mitad de llamadas. El rango de apertura para 1-min es la 1a vela de 1 minuto (mas
estrecho: entrada mas rapida pero mas ruido); para 5-min, la 1a vela de 5 minutos.

  python backtest_orb_massive.py [--tickers 60] [--dias 20] [--meses 24] [--rvol 1.0]

SOLO research/paper. Cache en datos/cache (gitignored): re-correr es barato.
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

# Reutilizamos EXACTAMENTE la logica del ORB (sim y rvol de apertura) y el IC95.
from backtest_orb5 import _opening_rvol, _sim
from backtest_intradia import _boot_ci
from fuentes import massive as mv
import universo_pit as U

MODOS = {"EOD": ("eod", 0.0), "EOD+trail2ATR": ("trail", 2.0)}
TFS = ("5Min", "1Min")
LOOKBACK_CAL = 30          # dias de calendario hacia atras para el rvol de apertura (14 sesiones)
MIN_BARRAS_DIA = {"5Min": 20, "1Min": 60}
EVAL_CACHE = os.path.join(U.CACHE, "orb_massive_evals.json")   # cache de resultados por ticker-dia


def _cargar_evals():
    if os.path.exists(EVAL_CACHE):
        try:
            with open(EVAL_CACHE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _guardar_evals(ev):
    os.makedirs(U.CACHE, exist_ok=True)
    tmp = EVAL_CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ev, f)
    os.replace(tmp, EVAL_CACHE)


def _resample_5min(m1):
    """1-min -> 5-min (reproduce el 5-min nativo de Massive, verificado diff 0.0)."""
    agg = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return agg.dropna(subset=["open"])


def _dia_ok(df, dia, min_barras):
    """Posiciones (array) de las barras de `dia` dentro de df, o None si no hay sesion util."""
    idx = pd.DatetimeIndex(df.index)
    pos = np.where(idx.normalize() == pd.Timestamp(dia).normalize())[0]
    return pos if len(pos) >= min_barras else None


def _eval_dia(df, tf, dia, rvol_min, confirmado=True):
    """Evalua el ORB confirmado para un ticker en un dia. Devuelve {modo: R} o {}."""
    pos = _dia_ok(df, dia, MIN_BARRAS_DIA[tf])
    if pos is None:
        return {}
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float); opn = df["open"].to_numpy(float)
    orv = _opening_rvol(df)
    d_norm = pd.Timestamp(dia).normalize()
    rv = orv.get(d_norm, np.nan)
    if not (rv == rv) or rv < rvol_min:                 # RVOL de apertura insuficiente
        return {}
    p0 = pos[0]
    if close[p0] <= opn[p0]:                             # 1a vela debe ser ALCISTA (long)
        return {}
    or_hi, or_lo = float(high[p0]), float(low[p0])
    # ATR en $ para el trailing: igual que orb5 (mediana del rango de barra * 3)
    atr_abs = float(np.nanmedian((df["high"] - df["low"]).to_numpy())) * 3.0
    out = {}
    for m, (modo, tr) in MODOS.items():
        r = _sim(close, high, low, list(pos), or_hi, or_lo, modo, tr, atr_abs, confirmado=confirmado)
        if r is not None:
            out[m] = r["R"]
    return out


def _trading_days(meses):
    """Dias habiles candidatos en los ultimos `meses` (excl. ~5 dias mas recientes
    por el delay)."""
    fin = pd.Timestamp.now().normalize() - pd.Timedelta(days=5)
    ini = fin - pd.Timedelta(days=int(meses * 31))
    return list(pd.bdate_range(ini, fin))


def _fetch_1min(ticker, dia, cm):
    """1-min del ticker en una ventana [dia-LOOKBACK, dia] (regular hours). Cacheado
    por el cliente http. None si falla o viene vacio/corto."""
    desde = (pd.Timestamp(dia) - pd.Timedelta(days=LOOKBACK_CAL)).strftime("%Y-%m-%d")
    hasta = pd.Timestamp(dia).strftime("%Y-%m-%d")
    try:
        # cache=False: no persistimos el JSON crudo (pesado); el eval-cache ya hace
        # barato re-correr sin inflar el disco con miles de ventanas de 1-min.
        df = mv.barras_intradia(ticker, "1Min", desde, hasta, cm=cm, cache=False)
    except Exception:
        return None
    return df if len(df) >= 300 else None


def _muestra(seed, tickers_n, dias, dias_cand, cm):
    """Sample reproducible por semilla: {dia: [tickers]} usando el universo PIT de cada dia."""
    rng = random.Random(seed)
    dsel = rng.sample(dias_cand, min(dias, len(dias_cand)))
    cs_set, _ = U.set_common_stocks(cm=cm)
    muestra = {}
    for d in sorted(dsel):
        ds = d.strftime("%Y-%m-%d")
        uni = U.universo_asof(ds, n=150, cm=cm, cs_set=cs_set)
        if len(uni) < 5:
            continue
        muestra[ds] = rng.sample(uni, min(tickers_n, len(uni)))
    return muestra


def backtest(tickers_n=60, dias=20, meses=24, rvol_min=1.0, seeds=(42, 7, 123),
             confirmado=True):
    cm = U._cliente()
    dias_cand = _trading_days(meses)
    evals = _cargar_evals()                          # cache resultado por ticker-dia (re-correr barato)
    # resultados[tf][modo][seed] = lista de R
    resultados = {tf: {m: {s: [] for s in seeds} for m in MODOS} for tf in TFS}
    n_td = {s: 0 for s in seeds}
    nuevos = 0
    for s in seeds:
        muestra = _muestra(s, tickers_n, dias, dias_cand, cm)
        total_pares = sum(len(v) for v in muestra.values())
        print(f"\n[seed {s}] {len(muestra)} dias x ~{tickers_n} tickers = {total_pares} ticker-dias "
              f"(universo PIT survivorship-free)")
        hecho = 0
        for ds, tks in muestra.items():
            for tk in tks:
                hecho += 1
                clave = f"{tk}|{ds}|rv{rvol_min}"
                ev = evals.get(clave)
                if ev is None:                       # no cacheado: baja y evalua
                    m1 = _fetch_1min(tk, ds, cm)
                    if m1 is None:
                        ev = {"ok": False}
                    else:
                        dfs = {"1Min": m1, "5Min": _resample_5min(m1)}
                        ev = {"ok": True}
                        for tf in TFS:
                            ev[tf] = _eval_dia(dfs[tf], tf, ds, rvol_min, confirmado=confirmado)
                    evals[clave] = ev
                    nuevos += 1
                    if nuevos % 200 == 0:
                        _guardar_evals(evals)
                        print(f"   ... {hecho}/{total_pares} | nuevos={nuevos} | llamadas={cm.llamadas}")
                if not ev.get("ok"):
                    continue
                n_td[s] += 1
                for tf in TFS:
                    for m, R in (ev.get(tf) or {}).items():
                        resultados[tf][m][s].append(R)
        print(f"   seed {s}: {n_td[s]} ticker-dias con datos | llamadas acumuladas={cm.llamadas}")
    _guardar_evals(evals)
    _informe(resultados, seeds, n_td)
    return resultados


def _stats(R):
    R = np.asarray(R, float)
    w = R[R > 0]; l = R[R <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    lo, hi = _boot_ci(R)
    return len(R), (len(w) / len(R) if len(R) else 0), R.mean() if len(R) else 0, lo, hi, pf


def _informe(resultados, seeds, n_td):
    print("\n" + "=" * 92)
    print("RESULTADO ORB CONFIRMADO — universo POINT-IN-TIME (survivorship-free) | Massive minuto")
    print("baseline SESGADO (universo de hoy): ORB confirmado entrada-al-cierre ~ +0.26R/trade")
    print("=" * 92)
    resumen = {}
    for tf in TFS:
        for m in MODOS:
            print(f"\n### {tf} | {m}")
            print(f"{'seed':>6}{'ops':>7}{'acierto':>10}{'expectancy':>13}{'IC95':>22}{'PF':>7}  veredicto")
            ics_pos = []
            for s in seeds:
                R = resultados[tf][m][s]
                if len(R) < 5:
                    print(f"{s:>6}{len(R):>7}   (muestra insuficiente)")
                    ics_pos.append(False)
                    continue
                n, acc, exp, lo, hi, pf = _stats(R)
                verd = "edge+" if lo > 0 else ("sin edge claro" if hi > 0 else "pierde")
                ics_pos.append(lo > 0)
                print(f"{s:>6}{n:>7}{acc*100:>9.0f}%{exp:>+12.3f}R  [{lo:+.3f},{hi:+.3f}]{pf:>7.2f}  {verd}")
            # pooled
            allR = [r for s in seeds for r in resultados[tf][m][s]]
            if len(allR) >= 5:
                n, acc, exp, lo, hi, pf = _stats(allR)
                robusto = all(ics_pos)
                vp = ("ROBUSTO+ (IC95>0 en las 3 semillas)" if robusto else
                      ("positivo pero NO robusto (alguna semilla con IC95<=0)" if exp > 0 else "sin edge / pierde"))
                print(f"{'POOL':>6}{n:>7}{acc*100:>9.0f}%{exp:>+12.3f}R  [{lo:+.3f},{hi:+.3f}]{pf:>7.2f}  -> {vp}")
                resumen[(tf, m)] = {"exp": exp, "lo": lo, "hi": hi, "robusto": robusto, "n": n}
    # comparacion 1v5 y haircut de supervivencia (modo EOD, el mas limpio)
    print("\n" + "-" * 92)
    base = 0.26
    for m in MODOS:
        print(f"\n[{m}] survivorship-free vs baseline sesgado (+{base:.2f}R):")
        for tf in TFS:
            r = resumen.get((tf, m))
            if r:
                hc = base - r["exp"]
                print(f"   {tf}: {r['exp']:+.3f}R (pool, n={r['n']})  haircut supervivencia = {hc:+.3f}R  "
                      f"{'ROBUSTO' if r['robusto'] else 'no robusto'}")
    r5 = resumen.get(("5Min", "EOD")); r1 = resumen.get(("1Min", "EOD"))
    if r5 and r1:
        print(f"\n[1min vs 5min | EOD] 5Min {r5['exp']:+.3f}R  vs  1Min {r1['exp']:+.3f}R  "
              f"-> {'1-min mejora' if r1['exp'] > r5['exp'] else '1-min NO mejora (ruido)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=60)
    ap.add_argument("--dias", type=int, default=20)
    ap.add_argument("--meses", type=int, default=24)
    ap.add_argument("--rvol", type=float, default=1.0)
    ap.add_argument("--seeds", type=str, default="42,7,123")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(","))
    backtest(a.tickers, a.dias, a.meses, a.rvol, seeds=seeds)
