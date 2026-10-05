"""Backtest de OVERLAYS sobre el baseline overnight (QQQ, cierre > SMA200).

Baseline (lo unico que de verdad funciona en la revalidacion survivorship-free):
mantener QQQ de cierre a apertura siguiente SOLO si el cierre del dia estaba por
encima de su SMA de 200 sesiones. Aqui probamos OVERLAYS que saltan noches malas
o escalan las buenas, y medimos si alguno bate al baseline OOS TRAS el haircut
por multiple-testing (Deflated Sharpe Ratio contando cuantos overlays se probaron).

Datos: Massive/Polygon EOD, split-adjusted, plan gratis (~5 anos, survivorship-free
porque Massive sirve tambien deslistados; QQQ/SPY/IWM no estan deslistados pero la
fuente no tiene sesgo de supervivencia). Sin FRED ni Alpaca en el entorno: el VIX
se aproxima con ETFs de futuros de VIX (VIXY corto, VIXM medio) y con volatilidad
realizada. Honestidad: ver docs/overnight_overlays_resultado.md para los caveats.

Uso:  PYTHONPATH=. python3 backtest_overnight_overlays.py
"""
import json
import sys

import numpy as np
import pandas as pd

from fuentes.massive import ClienteMassive, barras_diarias
from patrones.robustez import (deflated_sharpe_ratio, sr_esperado_max,
                               sharpe_no_anualizado)

ANUAL = 252.0
TICKERS = ["QQQ", "SPY", "IWM"]
VIX_ETFS = ["VIXY", "VIXM"]            # corto (1-2m) y medio (4-7m) de futuros VIX
DESDE, HASTA = "2019-01-01", "2026-10-01"
SMA_N = 200
RNG = np.random.default_rng(12345)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def cargar(tickers, cm=None):
    cm = cm or ClienteMassive()
    out = {}
    for t in tickers:
        df = barras_diarias(t, DESDE, HASTA, cm=cm)
        if len(df):
            out[t] = df
    return out


def panel_overnight(px):
    """Para un DataFrame OHLCV diario devuelve un frame por FECHA DE ENTRADA
    (cierre del dia t-1) con: ret overnight bruto = open[t]/close[t-1]-1, el
    cierre y su SMA200 (conocidos en t-1). La fila de fecha D decide con info de
    D y cobra el overnight D->D+1."""
    px = px.sort_index()
    cierre = px["close"]
    ap = px["open"]
    sma = cierre.rolling(SMA_N).mean()
    ov = (ap.shift(-1) / cierre - 1.0)        # overnight que empieza al cierre de D
    df = pd.DataFrame({
        "close": cierre, "sma200": sma, "sobre_sma": cierre > sma,
        "ov": ov,
    })
    return df.iloc[:-1]                        # ultima fila no tiene open[t+1]


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
def sharpe_anual(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ANUAL))


def max_drawdown(r):
    r = np.asarray(r, float)
    eq = np.cumprod(1 + np.nan_to_num(r))
    pico = np.maximum.accumulate(eq)
    return float((eq / pico - 1).min())


def cagr(r, n_per_ano=ANUAL):
    r = np.asarray(r, float)
    eq = np.cumprod(1 + np.nan_to_num(r))
    if len(r) < 2:
        return float("nan")
    anos = len(r) / n_per_ano
    return float(eq[-1] ** (1 / anos) - 1)


def bootstrap_sharpe_ic(r, n=2000, bloque=10, alpha=0.05):
    """IC de bootstrap por bloques (estacionario simple) para el Sharpe anualizado.
    Conserva autocorrelacion con bloques de longitud fija."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 30:
        return (float("nan"), float("nan"))
    nb = int(np.ceil(T / bloque))
    sh = np.empty(n)
    for i in range(n):
        inicios = RNG.integers(0, T, size=nb)
        idx = (inicios[:, None] + np.arange(bloque)[None, :]).ravel() % T
        sh[i] = sharpe_anual(r[idx[:T]])
    lo, hi = np.nanpercentile(sh, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# --------------------------------------------------------------------------- #
# Construccion de la serie de retornos de una estrategia
# --------------------------------------------------------------------------- #
def serie_estrategia(df, hold_mask, exposicion=None, coste_bps=2.0):
    """Serie diaria de retornos netos. hold_mask: bool (noches en las que estamos).
    exposicion: multiplicador por noche (vol targeting); None = 1.0 cuando hold.
    coste_bps: coste round-trip en puntos basicos, cobrado solo cuando cambiamos
    de exposicion (abrir/cerrar la pata overnight = round trip por noche operada)."""
    ov = df["ov"].to_numpy()
    hold = hold_mask.to_numpy().astype(float)
    exp = np.ones(len(df)) if exposicion is None else np.asarray(exposicion, float)
    peso = hold * exp
    ret_bruto = peso * ov
    # coste: cada noche operada es un round-trip (MOC al abrir, MOO al cerrar).
    # Al escalar exposicion el turnover es |peso| (abrimos y cerramos cada noche).
    coste = (coste_bps / 1e4) * np.abs(peso)
    return pd.Series(ret_bruto - coste, index=df.index)


# --------------------------------------------------------------------------- #
# Calendario FOMC (anuncios programados, conocidos de antemano = PIT valido)
# --------------------------------------------------------------------------- #
FOMC = pd.to_datetime([
    "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16",
])


# --------------------------------------------------------------------------- #
# Overlays: cada uno devuelve variantes (nombre, hold_mask, exposicion|None).
# Todos actuan SOBRE el baseline (solo pueden restringir noches o escalar).
# --------------------------------------------------------------------------- #
def realized_vol(cierre, n=20):
    return cierre.pct_change().rolling(n).std()


def construir_variantes(t, d, aux):
    """Genera (etiqueta_familia, nombre, hold_mask, exposicion|None, es_parametrico).
    base = baseline sobre_sma. aux: dict de series auxiliares alineadas al indice."""
    base = d["sobre_sma"].fillna(False)
    idx = d.index
    cierre = d["close"]
    vmas = []

    def add(fam, nombre, mask=None, exp=None, param=True):
        m = base if mask is None else (base & mask.reindex(idx).fillna(False))
        vmas.append((fam, nombre, m, exp, param))

    # ---- BASELINE ----
    add("baseline", "baseline", None, None, param=False)

    # ---- A: VIX / term-structure / vol realizada ----
    vixy, vixm = aux.get("VIXY"), aux.get("VIXM")
    if vixy is not None and vixm is not None:
        ratio = (vixy["close"] / vixm["close"]).reindex(idx)  # >1 ~ backwardation
        for q in [0.70, 0.80, 0.90]:
            thr = ratio.quantile(q)
            add("A_vix_backwardation", f"skip_ratio>{q:.2f}", ratio <= thr)
        chg = vixy["close"].pct_change(5).reindex(idx)        # spike de VIX corto
        for q in [0.80, 0.90, 0.95]:
            thr = chg.quantile(q)
            add("A_vix_spike", f"skip_vixchg>{q:.2f}", chg <= thr)
    rv = realized_vol(cierre, 20).reindex(idx)
    for q in [0.70, 0.80, 0.90]:
        thr = rv.quantile(q)
        add("A_rvol_regime", f"skip_rvol>{q:.2f}", rv <= thr)

    # ---- B: estacionalidad ----
    dow = pd.Series(idx.dayofweek, index=idx)
    for dd, nm in [(0, "lun"), (1, "mar"), (2, "mie"), (3, "jue"), (4, "vie")]:
        add("B_dow_drop", f"drop_{nm}", dow != dd)          # quitar un dia
    # turn-of-month: ultimos 1 + primeros 3 dias habiles del panel
    mes = idx.to_period("M")
    tom = pd.Series(False, index=idx)
    for _, g in pd.Series(idx, index=idx).groupby(mes):
        dias = g.index
        tom.loc[dias[:3]] = True
        if len(dias):
            tom.loc[dias[-1]] = True
    add("B_turn_of_month", "solo_TOM", tom)
    add("B_turn_of_month", "evita_TOM", ~tom)
    # FOMC: entrada el dia habil ANTES del anuncio (pre-FOMC drift) y el de despues
    es_fomc = pd.Series(idx.normalize().isin(FOMC.normalize()), index=idx)
    entra_pre = es_fomc.shift(-1, fill_value=False)   # overnight que entra en la vispera
    add("B_fomc", "solo_preFOMC", entra_pre)
    add("B_fomc", "skip_preFOMC", ~entra_pre)
    add("B_fomc", "skip_FOMCday", ~es_fomc)

    # ---- C: tendencia / breadth de corto plazo ----
    mom5 = (cierre > cierre.shift(5))
    add("C_trend", "mom5_pos", mom5, param=False)
    sma20 = (cierre > cierre.rolling(20).mean())
    add("C_trend", "sobre_sma20", sma20, param=False)
    breadth = aux.get("breadth")
    if breadth is not None:
        add("C_breadth", "breadth_all_sma200", breadth.reindex(idx), param=False)

    # ---- D: blackout macro (FOMC). Earnings N/A en ETFs. ----
    add("D_macro_blackout", "blackout_FOMC", ~es_fomc, param=False)

    # ---- E: vol targeting (escala exposicion por vol realizada) ----
    for tgt in [0.010, 0.015, 0.020]:                  # vol objetivo diaria overnight
        exp = (tgt / rv).clip(upper=2.0).fillna(1.0)
        add("E_vol_target", f"target_{tgt:.3f}", None, exp)

    return vmas


# --------------------------------------------------------------------------- #
# Evaluacion walk-forward + DSR
# --------------------------------------------------------------------------- #
def metr(r_series, mask_valid, coste_bps=2.0):
    r = r_series[mask_valid]
    arr = r.to_numpy()
    return {"sharpe": sharpe_anual(arr), "sr_per": sharpe_no_anualizado(arr),
            "maxdd": max_drawdown(arr), "cagr": cagr(arr), "n": int(len(arr))}


def evaluar_activo(t, d, aux, coste_bps=2.0):
    valido = d["sma200"].notna()
    T = len(d)
    corte = d.index[valido][len(d.index[valido]) // 2]
    train = valido & (d.index <= corte)
    test = valido & (d.index > corte)

    variantes = construir_variantes(t, d, aux)
    filas = []
    for fam, nombre, mask, exp, param in variantes:
        r = serie_estrategia(d, mask, exposicion=exp, coste_bps=coste_bps)
        full = metr(r, valido, coste_bps)
        tr = metr(r, train, coste_bps)
        te = metr(r, test, coste_bps)
        filas.append({"fam": fam, "nombre": nombre, "param": param,
                      "full": full, "train": tr, "test": te,
                      "r_full": r[valido]})
    return filas, (train, test, valido)


def resumen_activo(t, data, aux_global, coste_bps=2.0):
    d = panel_overnight(data[t])
    aux = {k: panel_aux(data[k], d.index) for k in VIX_ETFS if k in data}
    aux["breadth"] = aux_global["breadth"].reindex(d.index) if "breadth" in aux_global else None
    filas, _ = evaluar_activo(t, d, aux, coste_bps)
    base = next(f for f in filas if f["nombre"] == "baseline")

    # --- DSR: contar TODAS las pruebas (variantes != baseline) ---
    sr_trials = [f["full"]["sr_per"] for f in filas if np.isfinite(f["full"]["sr_per"])]
    n_trials = len(sr_trials)
    sr0 = sr_esperado_max(sr_trials)

    # seleccion honesta por familia parametrica: mejor en TRAIN, se reporta TEST
    seleccion = {}
    for f in filas:
        if not f["param"]:
            continue
        seleccion.setdefault(f["fam"], []).append(f)
    mejores = {}
    for fam, fs in seleccion.items():
        fs2 = [f for f in fs if np.isfinite(f["train"]["sharpe"])]
        if fs2:
            mejores[fam] = max(fs2, key=lambda f: f["train"]["sharpe"])

    return {"ticker": t, "n_trials": n_trials, "sr0_per": sr0,
            "base": base, "filas": filas, "mejores": mejores,
            "panel": d, "coste": coste_bps}


def panel_aux(px, idx):
    px = px.sort_index()
    return pd.DataFrame({"close": px["close"]}).reindex(idx, method="ffill")


def breadth_series(data):
    """True el dia D si QQQ, SPY e IWM estan los tres sobre su SMA200 (en D)."""
    masks = []
    for t in TICKERS:
        c = data[t]["close"].sort_index()
        masks.append((c > c.rolling(SMA_N).mean()).rename(t))
    m = pd.concat(masks, axis=1)
    return m.all(axis=1)


def main():
    print("Cargando EOD de Massive (cacheado 24h)...", file=sys.stderr)
    data = cargar(TICKERS + VIX_ETFS)
    aux_global = {"breadth": breadth_series(data)}
    salida = {}
    for coste in [0.0, 2.0, 5.0, 10.0]:
        salida[coste] = {t: resumen_activo(t, data, aux_global, coste) for t in TICKERS}

    # ---------- Informe por consola ----------
    for coste in [2.0]:
        for t in TICKERS:
            R = salida[coste][t]
            b = R["base"]["full"]
            print(f"\n########## {t}  (coste {coste}bps) ##########")
            print(f"BASELINE  full Sharpe {b['sharpe']:.3f} | train {R['base']['train']['sharpe']:.3f}"
                  f" | test {R['base']['test']['sharpe']:.3f} | maxDD {b['maxdd']*100:.1f}% | n {b['n']}")
            lo, hi = bootstrap_sharpe_ic(R["base"]["r_full"].to_numpy())
            print(f"          IC95 Sharpe [{lo:.2f}, {hi:.2f}]")
            print(f"  n_trials={R['n_trials']}  sr0_per(max azar)={R['sr0_per']:.4f}"
                  f"  -> umbral Sharpe anual~{R['sr0_per']*np.sqrt(ANUAL):.2f}")
            print("  --- mejor variante por familia (elegida en TRAIN, mostrada en TEST) ---")
            for fam, f in sorted(R["mejores"].items()):
                dsr = deflated_sharpe_ratio(f["r_full"].to_numpy(),
                                            [x["full"]["sr_per"] for x in R["filas"]])
                print(f"   {fam:24s} {f['nombre']:16s} full {f['full']['sharpe']:.2f} "
                      f"train {f['train']['sharpe']:.2f} test {f['test']['sharpe']:.2f} "
                      f"maxDD {f['full']['maxdd']*100:5.1f}% DSR {dsr:.3f}")
            # no parametricas
            print("  --- reglas sin parametros (train/test/DSR) ---")
            for f in R["filas"]:
                if f["param"] or f["nombre"] == "baseline":
                    continue
                dsr = deflated_sharpe_ratio(f["r_full"].to_numpy(),
                                            [x["full"]["sr_per"] for x in R["filas"]])
                print(f"   {f['fam']:24s} {f['nombre']:16s} full {f['full']['sharpe']:.2f} "
                      f"train {f['train']['sharpe']:.2f} test {f['test']['sharpe']:.2f} "
                      f"maxDD {f['full']['maxdd']*100:5.1f}% DSR {dsr:.3f}")

    # ---------- Tolerancia a coste del baseline y de la mejor variante QQQ ----------
    print("\n########## TOLERANCIA A COSTE (Sharpe anual) ##########")
    print("coste_bps     QQQ_base   SPY_base   IWM_base")
    for coste in [0.0, 2.0, 5.0, 10.0]:
        fila = [f"{salida[coste][t]['base']['full']['sharpe']:.2f}" for t in TICKERS]
        print(f"   {coste:4.1f}        " + "      ".join(fila))

    # ---------- Volcado JSON para el doc ----------
    dump = {}
    for coste in [0.0, 2.0, 5.0, 10.0]:
        dump[str(coste)] = {}
        for t in TICKERS:
            R = salida[coste][t]
            dump[str(coste)][t] = {
                "n_trials": R["n_trials"], "sr0_per": R["sr0_per"],
                "baseline": {k: R["base"]["full"][k] for k in ("sharpe", "maxdd", "cagr", "n")},
                "baseline_train": R["base"]["train"]["sharpe"],
                "baseline_test": R["base"]["test"]["sharpe"],
                "variantes": [
                    {"fam": f["fam"], "nombre": f["nombre"], "param": f["param"],
                     "full": f["full"]["sharpe"], "train": f["train"]["sharpe"],
                     "test": f["test"]["sharpe"], "maxdd": f["full"]["maxdd"],
                     "dsr": deflated_sharpe_ratio(f["r_full"].to_numpy(),
                                                  [x["full"]["sr_per"] for x in R["filas"]])}
                    for f in R["filas"]
                ],
            }
    # ---------- PBO (CSCV) sobre las 27 variantes de QQQ ----------
    from patrones.robustez import probabilidad_sobreajuste
    print("\n########## PBO (CSCV) y DSR cesta-ancha ##########")
    for t in TICKERS:
        R = salida[2.0][t]
        series = []
        for f in R["filas"]:
            s = f["r_full"].copy()
            s.index = pd.to_datetime(s.index)
            series.append(s.resample("ME").mean().rename(f["nombre"]))
        M = pd.concat(series, axis=1).fillna(0.0).to_numpy()
        pbo = probabilidad_sobreajuste(M, S=10)
        print(f"  {t}: PBO(27 variantes) = {pbo:.3f}  (>0.5 = probablemente sobreajuste)")

    # DSR de los dos mejores candidatos contando la CESTA (N = 27 x 3 activos = 81)
    sr_all = []
    for t in TICKERS:
        sr_all += [f["full"]["sr_per"] for f in salida[2.0][t]["filas"]
                   if np.isfinite(f["full"]["sr_per"])]
    sr0_cesta = sr_esperado_max(sr_all, n_trials=len(sr_all))
    print(f"\n  N cesta = {len(sr_all)}  sr0_per(cesta) = {sr0_cesta:.4f}"
          f"  umbral Sharpe anual ~ {sr0_cesta*np.sqrt(ANUAL):.2f}")
    for cand in ["evita_TOM", "drop_lun", "baseline"]:
        for t in ["QQQ"]:
            f = next(x for x in salida[2.0][t]["filas"] if x["nombre"] == cand)
            dsr81 = deflated_sharpe_ratio(f["r_full"].to_numpy(), sr_all)
            print(f"  {t} {cand:12s} full {f['full']['sharpe']:.2f}  DSR(N=81) = {dsr81:.3f}")

    with open("datos/overnight_overlays_dump.json", "w") as fh:
        json.dump(dump, fh, indent=1, default=float)
    print("\nVolcado -> datos/overnight_overlays_dump.json", file=sys.stderr)


if __name__ == "__main__":
    main()
