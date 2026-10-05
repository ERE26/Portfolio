"""BACKTEST de las setups REALES del day-trader de small-cap RUNNERS, con HONESTIDAD.

Universo: small-cap / low-float runners POINT-IN-TIME y SIN supervivencia
(universo_pit.py sobre el grouped de Massive; incluye los que luego quebraron).

Estrategias (point-in-time, barras de 1 minuto, una entrada por ticker-dia):
  1. GAP-AND-GO : gap up grande + RVOL de apertura alto + 1a vela verde -> largo al
     ROMPER el maximo del rango de apertura; stop en el minimo del rango; salida por
     fade de momentum (cierre bajo VWAP de sesion) o EOD.
  2. ORB RUNNER : ruptura CONFIRMADA (una vela CIERRA sobre el maximo del rango) ->
     largo; stop en el minimo del rango; se deja correr hasta el cierre (EOD) o stop.
  3. SHORT PARABOLICO : runner sobre-extendido (varias velas verdes + extension grande
     sobre VWAP) + vela de agotamiento (roja/rompe minimo previo) -> corto; stop sobre
     el maximo del tramo; cubre al volver a VWAP o EOD. Coste de BORROW alto (low-float
     'hard to borrow'): se modela y se barre (10/20/30 bps).

RIGOR:
  - PIT sin look-ahead: la seleccion usa gap+precio de apertura (09:30); los gates
    intradia (RVOL de apertura, $-volumen del rango) usan solo las primeras velas; la
    entrada se rellena al precio de disparo (ruptura) con slippage.
  - SIN supervivencia: el universo incluye deslistados; se reporta el resultado en el
    universo COMPLETO y en SOLO-SUPERVIVIENTES, y el HAIRCUT entre ambos.
  - 3 semillas (muestrean dias), bootstrap IC95, muestras pequenas para terminar.
  - EJECUCION: barrido de slippage 10/25/50 bps por lado (SLIP_STOP = 2x). Se reporta
    a que slippage MUERE cada estrategia.

Conservador por construccion: intrabar se asume que el stop se toca antes que el
objetivo; el fill de ruptura/stop lleva slippage; el short paga borrow.

  python backtest_lowfloat_massive.py [--desde 2024-03-01] [--hasta 2024-05-31]
       [--topk 12] [--seeds 3] [--dias-seed 20] [--json salida.json]

Solo lectura / paper. No envia ordenes ni toca gestionar.py/operativa_*.py.
"""
import argparse
import json
import random

import numpy as np
import pandas as pd

import universo_runners_pit as U
from fuentes import massive as mv

NY_ABRE, NY_OR_FIN, NY_CIERRE = "09:30", "09:35", "15:59"
BARRAS_SESION = 78                 # velas de 5m en sesion regular (RVOL proxy vs ayer)

# Gates intradia (point-in-time, primeras velas)
RVOL_MIN = 2.0                     # vol del rango de apertura vs una vela de 5m normal de ayer
OR_DOLAR_MIN = 50_000              # $-volumen minimo en el rango de apertura (fill plausible)
MIN_BARRAS = 8                     # minimo de velas de 1m en la sesion para considerar el dia

# Ejecucion
SLIP_BASE = 0.0025                 # 25 bps/lado por defecto
SLIPS = [0.0010, 0.0025, 0.0050, 0.0100]   # barrido: 10 / 25 / 50 / 100 bps
SLIP_STOP_MULT = 2.0               # el stop llena peor
MIN_STOP_PCT = 0.02                # suelo del riesgo 1R: >=2% del precio. Sin esto, un rango
                                   # de apertura de 1 centavo da R fantasia (100R imposibles
                                   # de llenar; el slippage se come un stop asi). Realista:
                                   # a un runner se le da aire; el stop no cabe en un tick.
BORROW_BPS_BASE = 0.0020           # 20 bps de borrow/locate por operacion (short)
BORROWS = [0.0010, 0.0020, 0.0030]

# Parabolico
EXT_MIN = 0.10                     # >=10% de extension del maximo sobre VWAP
RUN_VERDES = 3                     # velas verdes consecutivas antes del agotamiento


# ---------------------------------------------------------------- carga de dia

def _typical(h, l, c):
    return (h + l + c) / 3.0


def cargar_dia(ticker, fecha, cm, cache):
    """Estructura intradia de un ticker-dia (sesion regular 09:30-15:59 ET) o None.

    Devuelve arrays o/h/l/c/v, el VWAP de sesion acumulado, y los indices del rango
    de apertura (primeras velas < 09:35). Tolera barras que faltan (los low-float no
    cotizan cada minuto): se trabaja sobre las velas que EXISTEN."""
    clave = (ticker, fecha)
    if clave in cache:
        return cache[clave]
    try:
        df = mv.barras_intradia(ticker, "1Min", fecha, fecha, ajustado=False, cm=cm,
                                incluir_vwap=True)          # ya viene a horario regular (ET)
    except Exception:
        cache[clave] = None
        return None
    if not len(df) or not isinstance(df.index, pd.DatetimeIndex):
        cache[clave] = None                         # sin datos intradia (p.ej. deslistado sin barras)
        return None
    df = df.between_time(NY_ABRE, NY_CIERRE)
    if len(df) < MIN_BARRAS:
        cache[clave] = None
        return None
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    vw = df["vwap"].to_numpy(float).copy()
    mala = ~np.isfinite(vw) | (vw <= 0)
    vw[mala] = _typical(h, l, c)[mala]          # fallback precio tipico si falta vwap
    pxvol = np.cumsum(vw * v)
    volcum = np.cumsum(v)
    vwap_ses = np.where(volcum > 0, pxvol / np.maximum(volcum, 1e-9), c)
    # rango de apertura = velas antes de las 09:35 ET (las primeras ~5 velas de 1m)
    import datetime as _dt
    t = pd.DatetimeIndex(df.index).time
    fin = _dt.time(9, 35)
    k = max(int(np.sum([tt < fin for tt in t])), 1)
    est = {"o": o, "h": h, "l": l, "c": c, "v": v, "vwap": vwap_ses, "k": k, "n": len(df)}
    cache[clave] = est
    return est


def metricas_inplay(est, prev_vol):
    """RVOL de apertura y $-volumen del rango de apertura (point-in-time, primeras velas)."""
    k = est["k"]
    or_vol = float(est["v"][:k].sum())
    or_dolar = float((est["vwap"][:k] * est["v"][:k]).sum())
    base = (prev_vol / BARRAS_SESION) if prev_vol and prev_vol > 0 else np.nan
    rvol = or_vol / base if base and base > 0 else np.nan
    return rvol, or_dolar


# ---------------------------------------------------------------- estrategias

def gap_and_go(est, slip, slip_stop):
    """Largo al romper el maximo del rango de apertura (stop-order); stop = minimo del
    rango; salida por fade (cierre bajo VWAP) o EOD. 1a vela debe ser verde."""
    o, h, l, c, vw = est["o"], est["h"], est["l"], est["c"], est["vwap"]
    k, last = est["k"], est["n"] - 1
    if last < k:
        return None
    or_hi = float(h[:k].max()); or_lo = float(l[:k].min())
    if not (c[k - 1] > o[0]):                      # 1a "vela" (rango de apertura) verde
        return None
    i_in = None
    for j in range(k, last + 1):
        if h[j] >= or_hi:
            entry = or_hi * (1 + slip); i_in = j; break
    if i_in is None:
        return None
    rdist = max(entry - or_lo, MIN_STOP_PCT * entry)    # suelo de riesgo (anti R-fantasia)
    if rdist <= 0:
        return None
    stop, movido = entry - rdist, False
    exit_px, reason = float(c[last]) * (1 - slip), "EOD"
    for j in range(i_in, last + 1):
        hi, lo, cl = float(h[j]), float(l[j]), float(c[j])
        if lo <= stop:
            reason = "BE" if movido else "SL"
            exit_px = stop * (1 - slip_stop); break
        if cl < vw[j]:                              # fade de momentum: cierre bajo VWAP
            exit_px = cl * (1 - slip); reason = "VWAP_FADE"; break
        if not movido and hi >= entry + rdist:      # +1R -> break-even
            stop, movido = entry, True
        if j == last:
            exit_px = cl * (1 - slip); reason = "EOD"; break
    return {"R": (exit_px - entry) / rdist, "reason": reason}


def orb_runner(est, slip, slip_stop):
    """Ruptura CONFIRMADA: una vela CIERRA sobre el maximo del rango -> largo; stop =
    minimo del rango; se deja correr hasta EOD o stop (estilo Zarattini)."""
    o, h, l, c = est["o"], est["h"], est["l"], est["c"]
    k, last = est["k"], est["n"] - 1
    if last < k:
        return None
    or_hi = float(h[:k].max()); or_lo = float(l[:k].min())
    if not (c[k - 1] > o[0]):
        return None
    i_in = None
    for j in range(k, last + 1):
        if c[j] > or_hi:
            entry = float(c[j]) * (1 + slip); i_in = j; break
    if i_in is None:
        return None
    rdist = max(entry - or_lo, MIN_STOP_PCT * entry)    # suelo de riesgo (anti R-fantasia)
    if rdist <= 0:
        return None
    stop = entry - rdist
    exit_px, reason = float(c[last]) * (1 - slip), "EOD"
    for j in range(i_in, last + 1):
        lo, cl = float(l[j]), float(c[j])
        if lo <= stop:
            exit_px = stop * (1 - slip_stop); reason = "SL"; break
        if j == last:
            exit_px = cl * (1 - slip); reason = "EOD"; break
    return {"R": (exit_px - entry) / rdist, "reason": reason}


def short_parabolico(est, slip, slip_stop, borrow_bps):
    """Corto en agotamiento de un runner sobre-extendido. Varias velas verdes
    consecutivas + extension grande del maximo sobre VWAP, y vela de agotamiento
    (roja o que rompe el minimo previo) -> corto; stop sobre el maximo del tramo;
    cubre al volver a VWAP (cierre<=VWAP) o EOD. Paga borrow (low-float HTB)."""
    o, h, l, c, vw = est["o"], est["h"], est["l"], est["c"], est["vwap"]
    k, last = est["k"], est["n"] - 1
    verde = c > o
    i_in = None
    for j in range(k + RUN_VERDES, last + 1):
        run = all(verde[j - m] for m in range(1, RUN_VERDES + 1))   # RUN velas verdes antes de j
        ext = h[j - 1] / vw[j - 1] - 1.0 if vw[j - 1] > 0 else 0.0   # extension del pico sobre VWAP
        agot = (c[j] < o[j]) or (c[j] < l[j - 1])                    # agotamiento: roja o rompe minimo
        if run and ext >= EXT_MIN and agot:
            entry = float(c[j]) * (1 - slip)                         # corto: llena peor (vende mas bajo)
            i_in = j; swing = float(max(h[j - RUN_VERDES:j + 1]))    # maximo del tramo
            break
    if i_in is None:
        return None
    rdist = max(swing - entry, MIN_STOP_PCT * entry)    # suelo de riesgo (anti R-fantasia)
    if rdist <= 0:
        return None
    stop = entry + rdist
    exit_px, reason = float(c[last]) * (1 + slip), "EOD"
    for j in range(i_in, last + 1):
        hi, cl = float(h[j]), float(c[j])
        if hi >= stop:                                              # stop: cubrir mas arriba
            exit_px = stop * (1 + slip_stop); reason = "SL"; break
        if cl <= vw[j]:                                             # vuelta a VWAP: cubrir
            exit_px = cl * (1 + slip); reason = "VWAP_COVER"; break
        if j == last:
            exit_px = cl * (1 + slip); reason = "EOD"; break
    pnl = (entry - exit_px) - entry * borrow_bps                     # borrow/locate como coste
    return {"R": pnl / rdist, "reason": reason}


ESTRATEGIAS = {"gap_and_go": gap_and_go, "orb_runner": orb_runner,
               "short_parabolico": short_parabolico}


# ---------------------------------------------------------------- motor

def preparar_trades(cands, cm, activos, cache):
    """Para cada candidato: carga el dia, aplica el gate in-play, y guarda la estructura
    valida con sus metadatos. Devuelve lista de dicts listos para simular a cualquier
    slippage. (El conjunto de disparos no depende del slippage: solo el R.)"""
    listos = []
    n = len(cands)
    for i, cd in enumerate(cands):
        est = cargar_dia(cd["ticker"], cd["fecha"], cm, cache)
        if est is None:
            continue
        rvol, or_dolar = metricas_inplay(est, cd["prev_vol"])
        if not (rvol == rvol) or rvol < RVOL_MIN or or_dolar < OR_DOLAR_MIN:
            continue                                    # gate in-play (PIT, primeras velas)
        listos.append({"est": est, "fecha": cd["fecha"], "ticker": cd["ticker"],
                       "gap": cd["gap"], "vivo": cd["ticker"] in activos,
                       "rvol": rvol, "or_dolar": or_dolar})
        if (i + 1) % 100 == 0:
            print(f"   ... {i+1}/{n} candidatos procesados, {len(listos)} pasan gate")
    return listos


def simular(listos, estrategia, slip, borrow_bps=BORROW_BPS_BASE, solo_vivos=False):
    """Aplica una estrategia a los trades listos a un slippage dado. Devuelve lista de
    dicts {R, reason, fecha, ticker, vivo}."""
    fn = ESTRATEGIAS[estrategia]
    slip_stop = slip * SLIP_STOP_MULT
    out = []
    for t in listos:
        if solo_vivos and not t["vivo"]:
            continue
        if estrategia == "short_parabolico":
            r = fn(t["est"], slip, slip_stop, borrow_bps)
        else:
            r = fn(t["est"], slip, slip_stop)
        if r is not None:
            r.update({"fecha": t["fecha"], "ticker": t["ticker"], "vivo": t["vivo"]})
            out.append(r)
    return out


def _boot_ci(xs, n=2000, seed=1):
    if len(xs) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.array(xs, float)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _stats(trades):
    if not trades:
        return {"ops": 0}
    R = np.array([t["R"] for t in trades], float)
    w = R[R > 0]
    lo, hi = _boot_ci(R)
    return {"ops": len(R), "acierto_%": round(len(w) / len(R) * 100, 1),
            "expectancy_R": round(float(R.mean()), 3),
            "ic95": [round(lo, 3), round(hi, 3)],
            "R_medio_win": round(float(w.mean()), 2) if len(w) else 0.0,
            "mediana_R": round(float(np.median(R)), 3)}


def _veredicto(st):
    if st.get("ops", 0) < 5:
        return "muestra insuficiente"
    lo, hi = st["ic95"]
    if lo > 0:
        return "EDGE+ (IC95 sobre 0)"
    if hi > 0:
        return "sin edge claro (IC95 cruza 0)"
    return "PIERDE"


def analizar(listos, estrategia, dias_todos, seeds, dias_seed, borrow_bps=BORROW_BPS_BASE):
    """Resultado completo de una estrategia: por-semilla (base slip), pooled, barrido de
    slippage, y haircut de supervivencia."""
    res = {"estrategia": estrategia}
    # barrido de slippage (pooled, universo COMPLETO)
    sweep = {}
    for s in SLIPS:
        tr = simular(listos, estrategia, s, borrow_bps=borrow_bps)
        st = _stats(tr)
        st["veredicto"] = _veredicto(st)
        sweep[f"{int(s*1e4)}bps"] = st
    res["slippage_sweep_pit"] = sweep

    # por-semilla (muestrea dias) al slip base + pooled
    base = simular(listos, estrategia, SLIP_BASE, borrow_bps=borrow_bps)
    por_sem = []
    for sd in range(seeds):
        rng = random.Random(1000 + sd)
        muestra = set(rng.sample(dias_todos, min(dias_seed, len(dias_todos))))
        tr = [t for t in base if t["fecha"] in muestra]
        st = _stats(tr)
        st["seed"] = sd
        st["dias_muestreados"] = len(muestra)
        por_sem.append(st)
    res["por_semilla_base25bps"] = por_sem
    pooled = _stats(base); pooled["veredicto"] = _veredicto(pooled)
    res["pooled_base25bps"] = pooled

    # haircut de supervivencia (slip base)
    vivos = simular(listos, estrategia, SLIP_BASE, borrow_bps=borrow_bps, solo_vivos=True)
    res["supervivientes_base25bps"] = _stats(vivos)
    res["haircut"] = {
        "exp_pit": pooled.get("expectancy_R"),
        "exp_supervivientes": res["supervivientes_base25bps"].get("expectancy_R"),
        "ops_pit": pooled.get("ops"), "ops_supervivientes": res["supervivientes_base25bps"].get("ops"),
    }
    # reparto de cierres (slip base, pit)
    cierres = {}
    for t in base:
        cierres[t["reason"]] = cierres.get(t["reason"], 0) + 1
    res["cierres_base25bps"] = cierres
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2024-03-01")
    ap.add_argument("--hasta", default="2024-05-31")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dias-seed", type=int, default=20)
    ap.add_argument("--borrow", type=float, default=BORROW_BPS_BASE)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cm = mv.ClienteMassive(pausa_min=mv.PAUSA_STARTER)
    print(f"# Universo PIT de runners {a.desde} -> {a.hasta} (top {a.topk}/dia)")
    panel = U.panel_grouped(a.desde, a.hasta, cm=cm)
    sp = U.splits_set(a.desde, a.hasta, cm=cm)
    cands = U.candidatos_runner(panel, top_k=a.topk, splits=sp)
    activos = U.set_activos(cm=cm)
    uni = U.resumen_universo(cands, activos=activos)
    print(json.dumps(uni, indent=1, default=str))

    print(f"\nCargando barras de 1m y aplicando gate in-play (RVOL>={RVOL_MIN}, "
          f"$OR>={OR_DOLAR_MIN:,})...")
    cache = {}
    listos = preparar_trades(cands, cm, activos, cache)
    dias_todos = sorted({t["fecha"] for t in listos})
    print(f"trades listos (pasan gate): {len(listos)}  en {len(dias_todos)} dias")

    salida = {"universo": uni, "parametros": {
        "desde": a.desde, "hasta": a.hasta, "topk": a.topk, "rvol_min": RVOL_MIN,
        "or_dolar_min": OR_DOLAR_MIN, "slip_base_bps": SLIP_BASE * 1e4,
        "slip_sweep_bps": [s * 1e4 for s in SLIPS], "borrow_bps": a.borrow * 1e4,
        "n_listos": len(listos), "seeds": a.seeds, "dias_seed": a.dias_seed},
        "estrategias": {}}

    for e in ESTRATEGIAS:
        print(f"\n{'='*70}\n### {e}\n{'='*70}")
        r = analizar(listos, e, dias_todos, a.seeds, a.dias_seed, borrow_bps=a.borrow)
        salida["estrategias"][e] = r
        print("por semilla (slip 25bps, universo PIT):")
        for st in r["por_semilla_base25bps"]:
            print(f"  seed {st['seed']}: {st.get('ops',0):3d} ops  acierto {st.get('acierto_%','-')}%  "
                  f"exp {st.get('expectancy_R','-')}R  IC95 {st.get('ic95','-')}")
        p = r["pooled_base25bps"]
        print(f"  POOLED : {p.get('ops',0):3d} ops  acierto {p.get('acierto_%')}%  "
              f"exp {p.get('expectancy_R')}R  IC95 {p.get('ic95')}  -> {p.get('veredicto')}")
        print("barrido de slippage (pooled PIT):")
        for bps, st in r["slippage_sweep_pit"].items():
            print(f"  {bps:>6}: {st.get('ops',0):3d} ops  exp {st.get('expectancy_R','-')}R  "
                  f"IC95 {st.get('ic95','-')}  -> {st.get('veredicto')}")
        hc = r["haircut"]
        print(f"haircut supervivencia: PIT exp {hc['exp_pit']}R ({hc['ops_pit']} ops) vs "
              f"SUPERVIVIENTES exp {hc['exp_supervivientes']}R ({hc['ops_supervivientes']} ops)")
        print(f"cierres: {r['cierres_base25bps']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(salida, f, indent=1, default=str)
        print(f"\nJSON -> {a.json}")
    return salida


if __name__ == "__main__":
    main()
