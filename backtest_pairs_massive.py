"""PAIRS / STAT-ARB INTRADIA sobre un universo FIJO de ETFs relacionados.

Familia NUEVA (no probada en el repo): mean-reversion del SPREAD intradia entre dos
ETFs economicamente emparejados (mismo sector/tema, holdings solapados -> cointegracion
ECONOMICA, no minada). Market-neutral: se opera el spread, no la direccion del mercado.

Por que es limpio respecto a las criticas del resto del estudio:
  - UNIVERSO FIJO de pares pre-especificados por logica economica -> NO hay seleccion ni
    data-mining de que par usar; NO hay sesgo de supervivencia (ningun ETF se deslista).
  - POINT-IN-TIME sin look-ahead: el hedge ratio beta de cada dia se estima SOLO con las
    sesiones ANTERIORES (OLS logA~logB, K sesiones previas). El z-score intradia usa una
    ventana TRAILING dentro de la sesion (nunca el futuro). Se entra al CIERRE de la vela de
    senal + slippage. Se cierra SIEMPRE antes del fin de sesion -> plano overnight
    (intradia de verdad, no overnight disfrazado).
  - COSTES realistas: SLIP por pata y por lado; una operacion de par = 4 fills (2 patas x
    entrada+salida). Se reporta tambien sensibilidad a un coste mas benigno (ETFs son
    liquidos) para separar "no hay edge" de "el edge no paga el coste".
  - 3 SEMILLAS (muestras de dias independientes) + bootstrap IC95 de la expectancy por
    operacion. Edge ROBUSTO solo si IC95>0 en las 3 semillas con muestra suficiente.

Logica de una operacion (un par A,B; un dia d; a lo sumo UNA operacion por par-dia):
  beta   = OLS(logA ~ logB) sobre las K sesiones previas a d (point-in-time).
  spread_t = logA_t - beta*logB_t   (dentro de la sesion d, barras de 5 min RTH)
  z_t    = (spread_t - media_trailing_W) / std_trailing_W     (ventana TRAILING intradia)
  ENTRADA: primera barra con |z|>=K_ENTRY (y >=WMIN barras transcurridas, y antes del corte
           horario). z>0 -> spread caro -> CORTO spread (corto A / largo B). z<0 -> LARGO spread.
  SALIDA : |z|<=K_EXIT (reversion, TP) | |z|>=K_STOP (divergencia, stop) | fin de sesion (EOD).
  retorno (sobre capital bruto desplegado, neto de coste):
     bruto = signo*(retA - beta*retB);  coste = 2*SLIP*(1+beta);  r = (bruto - coste)/(1+beta)

  python backtest_pairs_massive.py [--dias 60] [--kent 2.0] [--kexit 0.5] [--seeds 42,7,123]

SOLO research/paper. Cache de barras en datos/cache (gitignored): re-correr es barato.
"""
import argparse
import datetime as dt
import hashlib

import numpy as np
import pandas as pd

from backtest_intradia import _boot_ci
from fuentes import massive as mv
import universo_pit as U

# Pares pre-especificados por LOGICA ECONOMICA (mismo sector/tema -> cointegracion real).
# Fijo: no se elige el par mirando el resultado. Sin supervivencia (ETFs no se deslistan).
PARES = [
    ("XLE", "XOP"),    # energia sector vs E&P de oil&gas
    ("XLE", "VDE"),    # dos ETFs de energia (SPDR vs Vanguard)
    ("XLF", "VFH"),    # dos ETFs de financieras (SPDR vs Vanguard)
    ("XLF", "KRE"),    # financieras vs bancos regionales
    ("XLK", "VGT"),    # dos ETFs de tecnologia
    ("SMH", "SOXX"),   # dos ETFs de semiconductores
    ("GDX", "GDXJ"),   # mineras de oro vs junior gold miners
    ("XHB", "ITB"),    # dos ETFs de homebuilders
    ("XBI", "IBB"),    # dos ETFs de biotecnologia
    ("QQQ", "XLK"),    # Nasdaq-100 vs tecnologia
]

SLIP = 0.0004              # 4 bps por pata y por lado (mismo que el resto del repo)
VENTANA = "2023-11-01"     # inicio de descarga (deja >=K sesiones de calentamiento antes de 2024)
FIN = "2026-09-30"         # fin (evita las ~sesiones mas recientes por el delay del plan)
K_PRIOR = 20               # sesiones previas para estimar beta (point-in-time)
W = 20                     # ventana trailing (barras de 5 min) para media/std del z-score intradia
WMIN = 12                  # barras minimas transcurridas en la sesion antes de poder entrar
CORTE = dt.time(15, 30)    # no abrir nuevas despues de esto (deja tiempo a revertir)
EOD = dt.time(15, 55)      # cierre forzado


def _descarga(cm):
    """Baja (cacheado) el 5-min RTH de todos los ETFs del universo de pares. Devuelve dict."""
    tickers = sorted({t for p in PARES for t in p})
    px = {}
    for t in tickers:
        df = mv.barras_intradia(t, "5Min", VENTANA, FIN, cm=cm, cache=True)
        if len(df):
            px[t] = df
    return px


def _dias_comunes(a, b):
    da = set(pd.DatetimeIndex(a.index).normalize())
    db = set(pd.DatetimeIndex(b.index).normalize())
    return sorted(da & db)


def _beta_pit(la, lb, idx_norm, dias_prev):
    """(beta, sigma_intra) point-in-time: OLS slope logA~logB sobre las sesiones
    ANTERIORES `dias_prev`, y la escala INTRADIA del spread. sigma_intra = std del
    residuo del spread DESPUES de quitarle la media de CADA sesion previa: mide cuanto
    deambula el spread DENTRO del dia (la escala correcta para una divergencia
    intradia), no la variacion multi-dia. Todo usa solo el pasado (sin look-ahead)."""
    days = np.array([np.datetime64(d, "D") for d in dias_prev])
    mask = np.isin(idx_norm, days)
    x = lb[mask]; y = la[mask]; dd = idx_norm[mask]
    if len(x) < 50 or np.std(x) == 0:
        return None
    b, a0 = np.polyfit(x, y, 1)
    resid = y - (b * x + a0)
    # quitar la media intradia de cada sesion -> deambular dentro del dia
    medias = pd.Series(resid).groupby(dd).transform("mean").to_numpy()
    sigma = float(np.std(resid - medias))
    if not (sigma > 0):
        return None
    return float(b), sigma


def _sim_par_dia(la, lb, ra, rb, pos_dia, beta, sigma, k_ent, k_exit, k_stop, horas):
    """Una operacion (la primera senal) del par en el dia. Devuelve r neto o None.

    z CAUSAL sin artefacto de ventana: el spread se ancla al nivel de la MAÑANA de ESA
    sesion (media de las primeras WMIN barras, point-in-time) y se escala por sigma del
    spread estimado en las sesiones PREVIAS. Asi z_t mide la DIVERGENCIA INTRADIA real
    respecto a como abrio el par, y la reversion a z=0 refleja movimiento de precio
    genuino (no el decaimiento mecanico de una media movil que absorbe el shock)."""
    spread = la[pos_dia] - beta * lb[pos_dia]
    n = len(pos_dia)
    if n < WMIN + 3:
        return None
    mu = float(np.mean(spread[:WMIN]))           # ancla intradia (solo la manana)
    z = (spread - mu) / sigma
    for i in range(WMIN, n):
        if horas[i] >= CORTE:
            break
        if abs(z[i]) < k_ent:
            continue
        signo = -1.0 if z[i] > 0 else 1.0        # z>0 spread caro -> corto spread
        j_in = pos_dia[i]
        for m in range(i + 1, n):
            fin = (abs(z[m]) <= k_exit) or (abs(z[m]) >= k_stop) or (horas[m] >= EOD) or (m == n - 1)
            if fin:
                j_out = pos_dia[m]
                reta = ra[j_out] / ra[j_in] - 1.0
                retb = rb[j_out] / rb[j_in] - 1.0
                bruto = signo * (reta - beta * retb)
                coste = 2.0 * SLIP * (1.0 + abs(beta))
                return (bruto - coste) / (1.0 + abs(beta))
        return None
    return None


def _sample_dias(dias, n, seed):
    rng = np.random.default_rng(seed)
    dias = list(dias)
    if n >= len(dias):
        return sorted(dias)
    pick = rng.choice(len(dias), size=n, replace=False)
    return sorted(dias[i] for i in pick)


def backtest(dias_por_semilla=60, k_ent=2.0, k_exit=0.5, k_stop=4.0, seeds=(42, 7, 123),
             slip_mult=1.0):
    global SLIP
    SLIP = 0.0004 * slip_mult
    cm = U._cliente()
    px = _descarga(cm)
    print(f"ETFs descargados: {sorted(px)} | SLIP={SLIP*1e4:.1f} bps/pata/lado")
    # Precompute por ticker: log y niveles como arrays alineados por par en el bucle.
    res = {s: [] for s in seeds}        # r por operacion
    npd = {s: 0 for s in seeds}         # par-dias evaluados
    for (ta, tb) in PARES:
        if ta not in px or tb not in px:
            continue
        a, b = px[ta].align(px[tb], join="inner", axis=0)
        if len(a) < 2000:
            continue
        idx = pd.DatetimeIndex(a.index)
        idx_norm = idx.normalize().values.astype("datetime64[D]")
        horas = idx.time
        la = np.log(a["close"].to_numpy(float)); lb = np.log(b["close"].to_numpy(float))
        ra = a["close"].to_numpy(float); rb = b["close"].to_numpy(float)
        dias = sorted(set(pd.DatetimeIndex(a.index).normalize()))
        # mapa dia -> posiciones de sus barras
        por_dia = {}
        for p, d in enumerate(idx_norm):
            por_dia.setdefault(d, []).append(p)
        dias64 = [np.datetime64(pd.Timestamp(d), "D") for d in dias]
        # solo dias con >=K_PRIOR sesiones previas disponibles
        elegibles = [d for k, d in enumerate(dias) if k >= K_PRIOR]
        for s in seeds:
            # semilla DETERMINISTA por (par, seed) -> reproducible entre procesos
            # (hash() de Python esta salteado por PYTHONHASHSEED; md5 no).
            sd = int(hashlib.md5(f"{ta}|{tb}|{s}".encode()).hexdigest()[:8], 16)
            sel = _sample_dias(elegibles, dias_por_semilla, seed=sd)
            for d in sel:
                k = dias.index(d)
                prev = dias[k - K_PRIOR:k]
                bs = _beta_pit(la, lb, idx_norm, prev)
                if bs is None or not (0.2 < abs(bs[0]) < 5.0):
                    continue
                beta, sigma = bs
                pos_dia = np.array(por_dia[np.datetime64(pd.Timestamp(d), "D")])
                if len(pos_dia) < WMIN + 3:
                    continue
                npd[s] += 1
                r = _sim_par_dia(la, lb, ra, rb, pos_dia, beta, sigma, k_ent, k_exit, k_stop,
                                 [horas[p] for p in pos_dia])
                if r is not None:
                    res[s].append(r)
    _informe(res, npd, k_ent, k_exit, k_stop)
    return res


def _stats(R):
    R = np.asarray(R, float)
    w = R[R > 0]; l = R[R <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    lo, hi = _boot_ci(R)
    return len(R), (len(w) / len(R) if len(R) else 0), R.mean() if len(R) else 0, lo, hi, pf


def _informe(res, npd, k_ent, k_exit, k_stop):
    print("\n" + "=" * 94)
    print(f"PAIRS / STAT-ARB INTRADIA (ETFs fijos) | k_ent={k_ent} k_exit={k_exit} k_stop={k_stop}")
    print("retorno POR OPERACION sobre capital bruto, neto de coste (market-neutral, plano overnight)")
    print("=" * 94)
    print(f"{'seed':>6}{'par-dias':>10}{'ops':>7}{'acierto':>9}{'exp/op':>12}{'IC95 (bps)':>24}{'PF':>7}  veredicto")
    ics = []
    for s in sorted(res):
        R = res[s]
        if len(R) < 20:
            print(f"{s:>6}{npd[s]:>10}{len(R):>7}   (muestra insuficiente <20)")
            ics.append(False); continue
        n, acc, exp, lo, hi, pf = _stats(R)
        verd = "edge+" if lo > 0 else ("sin edge claro" if hi > 0 else "pierde")
        ics.append(lo > 0)
        print(f"{s:>6}{npd[s]:>10}{n:>7}{acc*100:>8.0f}%{exp*1e4:>+10.1f}bp"
              f"  [{lo*1e4:+.1f},{hi*1e4:+.1f}]{pf:>7.2f}  {verd}")
    allR = [r for s in res for r in res[s]]
    if len(allR) >= 20:
        n, acc, exp, lo, hi, pf = _stats(allR)
        robusto = all(ics)
        vp = ("ROBUSTO+ (IC95>0 en las 3 semillas)" if robusto else
              ("positivo pero NO robusto" if exp > 0 else "sin edge / pierde"))
        print(f"{'POOL':>6}{sum(npd.values()):>10}{n:>7}{acc*100:>8.0f}%{exp*1e4:>+10.1f}bp"
              f"  [{lo*1e4:+.1f},{hi*1e4:+.1f}]{pf:>7.2f}  -> {vp}")
    return allR


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=60, help="dias muestreados por par y semilla")
    ap.add_argument("--kent", type=float, default=2.0)
    ap.add_argument("--kexit", type=float, default=0.5)
    ap.add_argument("--kstop", type=float, default=4.0)
    ap.add_argument("--seeds", type=str, default="42,7,123")
    ap.add_argument("--slip", type=float, default=1.0, help="multiplicador de SLIP (0.5 = ETF benigno)")
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(","))
    backtest(a.dias, a.kent, a.kexit, a.kstop, seeds=seeds, slip_mult=a.slip)
