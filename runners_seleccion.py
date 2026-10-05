"""SELECCION de runners: ¿rescata un SUBCONJUNTO el edge, neto de coste y OOS?

Contexto (docs/lowfloat_runners_resultado.md): las tres setups del day-trader de
small-cap runners (gap-and-go, ORB runner, short parabolico) salieron SIN edge
robusto EN MEDIA sobre el universo point-in-time sin supervivencia. Pero el
day-trader real NO opera todos los gappers que pasan el gate: SELECCIONA. La
pregunta nueva, que la propia seccion 7 de aquel doc deja abierta: ¿existe un
SUBCONJUNTO de runners -identificable POINT-IN-TIME con features de apertura-
donde el gap-and-go o el ORB runner SI sea rentable neto de coste?

Este modulo REUTILIZA la infra existente para que el resultado se apoye EXACTO en
el baseline ya publicado:
  - universo_runners_pit.py  -> universo PIT sin supervivencia (candidatos).
  - backtest_lowfloat_massive.py -> carga de barras 1m, gate in-play (RVOL>=2,
    $OR>=50k) y la R real de cada setup (stop OR-low suelo 2%, fade-VWAP/EOD,
    slippage por lado). Mismas reglas, mismo R que el doc de baseline.

Sobre ese dataset aplica la vara anti-overfit de orb_metalabel.py:
  (1) META-MODELO logistica/HGB con PURGED walk-forward (embargo), seleccion por
      probabilidad; DSR + PBO/CSCV + IC95 bootstrap por umbral.
  (2) CORTES SIMPLES e interpretables (gap en rango, precio, RVOL alto, OR
      estrecho, primera hora, combinaciones) con PBO sobre el MENU de cortes.
Veredicto EDGE ROBUSTO solo si DSR>=0.95, PBO<=0.30 e IC95(OOS)>0 a la vez.

Honestidad: con muestras pequenas un 'ganador' in-sample casi siempre existe por
azar; si no sobrevive OOS/DSR/PBO, se dice claro.

  # construir el dataset (cachea barras en datos/cache, gitignored) y analizar:
  python runners_seleccion.py --desde 2023-07-01 --hasta 2024-12-31 --construir
  # reanalizar desde el CSV ya construido:
  python runners_seleccion.py --label R_orb
Solo research/paper. No toca gestionar.py ni operativa_*.py. No cablea nada.
"""
import argparse
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import universo_runners_pit as U
import backtest_lowfloat_massive as B
from fuentes import massive as mv
from orb_metalabel import _deflated_sharpe, _pbo_cscv   # misma maquinaria validada

FEATURES = ["gap", "precio", "opening_rvol", "dvol_or", "or_range_pct", "dow",
            "ret_prev1", "ret_prev3", "n_green3", "gap_x_rvol"]
LOGF = {"opening_rvol", "dvol_or", "gap_x_rvol"}        # muy sesgadas: log1p
CSV = "datos/runners_seleccion.csv"

_local = threading.local()


def _cm():
    """ClienteMassive por hilo (sesion propia), sin throttle (plan Starter)."""
    if not hasattr(_local, "cm"):
        _local.cm = mv.ClienteMassive(pausa_min=0.0)
    return _local.cm


def _warm(cand):
    """Pre-descarga (a la cache de disco) las barras 1m de un candidato, con los
    MISMOS parametros que backtest_lowfloat_massive.cargar_dia (para que luego sea
    cache-hit)."""
    try:
        mv.barras_intradia(cand["ticker"], "1Min", cand["fecha"], cand["fecha"],
                           ajustado=False, cm=_cm(), incluir_vwap=True)
    except Exception:
        pass


def _hist_cierres(panel):
    """ticker -> dict(fecha->close) para los retornos previos PIT."""
    h = defaultdict(dict)
    for d, g in panel.items():
        for t, row in zip(g.index, g["close"].astype(float)):
            h[t][d] = row
    return h


def _rets_previos(hist, ticker, fecha):
    """ret_prev1, ret_prev3, n_green3 con cierres ESTRICTAMENTE anteriores a 'fecha'
    (conocidos a la apertura). NaN/0 si falta historia."""
    serie = hist.get(ticker, {})
    dias = sorted(d for d in serie if d < fecha)
    c = [serie[d] for d in dias]
    ret1 = (c[-1] / c[-2] - 1) if len(c) >= 2 and c[-2] > 0 else np.nan
    ret3 = (c[-1] / c[-4] - 1) if len(c) >= 4 and c[-4] > 0 else np.nan
    ng = int(sum(1 for a, b in zip(c[-3:], c[-4:-1]) if a > b)) if len(c) >= 4 else 0
    return ret1, ret3, ng


def construir(desde, hasta, topk=12, slip=B.SLIP_BASE, workers=8, salida=CSV):
    cm = mv.ClienteMassive(pausa_min=mv.PAUSA_STARTER)
    print(f"# Dataset de SELECCION de runners {desde} -> {hasta} (top {topk}/dia, "
          f"slip {slip*1e4:.0f}bps)")
    panel = U.panel_grouped(desde, hasta, cm=cm)
    print(f"sesiones con datos: {len(panel)}")
    sp = U.splits_set(desde, hasta, cm=cm)
    cands = U.candidatos_runner(panel, top_k=topk, splits=sp)
    print(f"candidatos runner PIT: {len(cands)}  (descargando barras 1m en paralelo...)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_warm, cands))                       # calienta la cache de disco
    hist = _hist_cierres(panel)
    cache = {}
    listos = B.preparar_trades(cands, cm, activos=set(), cache=cache)  # ya es cache-hit
    print(f"trades que pasan gate in-play: {len(listos)}")
    slip_stop = slip * B.SLIP_STOP_MULT
    filas = []
    for t in listos:
        est = t["est"]
        k = est["k"]
        or_hi = float(est["h"][:k].max()); or_lo = float(est["l"][:k].min())
        o0 = float(est["o"][0])
        if o0 <= 0:
            continue
        rg = B.gap_and_go(est, slip, slip_stop)
        ro = B.orb_runner(est, slip, slip_stop)
        ret1, ret3, ng = _rets_previos(hist, t["ticker"], t["fecha"])
        rvol = float(t["rvol"])
        filas.append(dict(
            fecha=t["fecha"], ticker=t["ticker"], gap=float(t["gap"]), precio=o0,
            opening_rvol=rvol, dvol_or=float(t["or_dolar"]) / 1e6,
            or_range_pct=(or_hi - or_lo) / o0,
            dow=pd.Timestamp(t["fecha"]).weekday(),
            ret_prev1=ret1, ret_prev3=ret3, n_green3=ng,
            gap_x_rvol=float(t["gap"]) * (rvol if np.isfinite(rvol) else 0.0),
            R_gapgo=(rg["R"] if rg else np.nan),
            R_orb=(ro["R"] if ro else np.nan)))
    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    df.to_csv(salida, index=False)
    tr_g = df["R_gapgo"].notna().sum(); tr_o = df["R_orb"].notna().sum()
    print(f"\nDATASET: {len(df)} runner-days | {df['fecha'].min()}..{df['fecha'].max()}")
    print(f" gap-and-go dispara en {tr_g}, ORB en {tr_o}")
    print(f" baseline (tomar TODOS) @ {slip*1e4:.0f}bps: "
          f"gap-and-go {df['R_gapgo'].mean():+.3f}R | ORB {df['R_orb'].mean():+.3f}R")
    print(f" -> {salida}")
    return df


# --------------------------------------------------------------------- analisis
def _preparar(df, label):
    df = df.copy()
    df = df[df[label].notna()].reset_index(drop=True)
    for c in ("ret_prev1", "ret_prev3"):
        df[c] = df[c].fillna(0.0)
    X = df.copy()
    for f in FEATURES:
        X[f] = np.log1p(np.clip(df[f], 0, None)) if f in LOGF else df[f]
    ok = np.isfinite(X[FEATURES].to_numpy(float)).all(axis=1)
    return df[ok].reset_index(drop=True), X[ok].reset_index(drop=True)


def _ic95(R, B_=5000, seed=0):
    r = np.asarray(R, float)
    if len(r) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    medias = r[rng.integers(0, len(r), size=(B_, len(r)))].mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def _walk_forward(df, X, label, modelo, folds=6, embargo=3):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    order = df.sort_values("fecha").index.to_numpy()
    df = df.loc[order].reset_index(drop=True)
    X = X.loc[order].reset_index(drop=True)
    fechas = pd.to_datetime(df["fecha"])
    Xm = X[FEATURES].to_numpy(float)
    y = (df[label].to_numpy(float) > 0).astype(int)
    n = len(df)
    prob = np.full(n, np.nan)
    bordes = [int(n * k / (folds + 1)) for k in range(1, folds + 2)]
    for i in range(len(bordes) - 1):
        tr_end, te_ini, te_fin = bordes[i], bordes[i], bordes[i + 1]
        if tr_end < 60 or te_fin <= te_ini:
            continue
        corte = fechas.iloc[te_ini] - pd.Timedelta(days=embargo)
        idx_tr = np.where((fechas.iloc[:tr_end] <= corte).to_numpy())[0]
        if len(idx_tr) < 50 or len(np.unique(y[idx_tr])) < 2:
            continue
        Xtr, ytr, Xte = Xm[idx_tr], y[idx_tr], Xm[te_ini:te_fin]
        if modelo == "logistica":
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(max_iter=1000, class_weight="balanced")
            m.fit(sc.transform(Xtr), ytr)
            prob[te_ini:te_fin] = m.predict_proba(sc.transform(Xte))[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                               learning_rate=0.05, l2_regularization=1.0)
            m.fit(Xtr, ytr)
            prob[te_ini:te_fin] = m.predict_proba(Xte)[:, 1]
    return df, prob


def _veredicto(mejora, dsr, pbo, ic_lo):
    if np.isfinite(dsr) and np.isfinite(pbo) and dsr >= 0.95 and pbo <= 0.30 and ic_lo > 0 and mejora > 0:
        return "EDGE ROBUSTO"
    if mejora > 0:
        return "MEJORA NO CONCLUYENTE (no pasa DSR/PBO/IC95)"
    return "NO MEJORA"


def meta_modelo(df0, label, folds=6):
    df, X = _preparar(df0, label)
    R_all = df[label].to_numpy(float)
    base = R_all.mean()
    lo, hi = _ic95(R_all)
    print(f"\n### META-MODELO | label={label} | {len(df)} runner-days "
          f"| {df['fecha'].min()}..{df['fecha'].max()}")
    print(f"BASELINE (tomar TODOS): {base:+.3f}R | win {(R_all>0).mean()*100:.0f}% "
          f"| IC95 [{lo:+.3f}, {hi:+.3f}]  (edge en media: {'SI' if lo>0 else 'NO'})")
    umbrales = [0.40, 0.45, 0.50, 0.55, 0.60]
    for modelo in ("logistica", "hgb"):
        dfx, prob = _walk_forward(df, X, label, modelo, folds=folds)
        Rx = dfx[label].to_numpy(float)
        ok = np.isfinite(prob)
        print(f"\n[{modelo}]  OOS en {ok.sum()} ops")
        bloques = np.array_split(np.where(ok)[0], 8)
        perf = []
        for b in bloques:
            perf.append([Rx[b][prob[b] >= u].mean() if (prob[b] >= u).sum() else np.nan
                         for u in umbrales])
        perf = np.array(perf, float)
        perf = perf[~np.isnan(perf).any(axis=1)] if len(perf) else perf
        pbo = _pbo_cscv(perf) if len(perf) >= 4 else float("nan")
        print(f"   {'umbral':>7}{'%tom':>7}{'exp':>9}{'win%':>6}{'IC95':>18}{'DSR':>7}")
        best = None
        for u in umbrales:
            sel = ok & (prob >= u)
            if sel.sum() < 20:
                print(f"   {u:>7.2f}   (pocas ops: {sel.sum()})")
                continue
            Rs = Rx[sel]
            dsr, _ = _deflated_sharpe(Rs, n_trials=len(umbrales) * 2)
            lo, hi = _ic95(Rs)
            print(f"   {u:>7.2f}{sel.sum()/ok.sum()*100:>6.0f}%{Rs.mean():>+8.3f}R"
                  f"{(Rs>0).mean()*100:>5.0f}%  [{lo:>+6.3f},{hi:>+6.3f}]{dsr:>7.2f}")
            if best is None or Rs.mean() > best["exp"]:
                best = dict(u=u, exp=Rs.mean(), dsr=dsr, ic=(lo, hi), n=int(sel.sum()))
        if best:
            mejora = best["exp"] - base
            ver = _veredicto(mejora, best["dsr"], pbo, best["ic"][0])
            print(f"   -> mejor umbral {best['u']:.2f}: {best['exp']:+.3f}R "
                  f"({mejora:+.3f}R vs baseline) | n={best['n']} | DSR {best['dsr']:.2f} "
                  f"| PBO {pbo:.2f} | IC95 [{best['ic'][0]:+.3f},{best['ic'][1]:+.3f}]")
            print(f"   -> VEREDICTO: {ver}")


def cortes_simples(df0, label):
    df, _ = _preparar(df0, label)
    R = df[label].to_numpy(float)
    base = R.mean()
    g, p = df["gap"], df["precio"]
    rv, orp = df["opening_rvol"], df["or_range_pct"]
    cortes = {
        "TODOS (baseline)":          np.ones(len(df), bool),
        "gap 10-30%":                (g >= 0.10) & (g < 0.30),
        "gap 30-50%":                (g >= 0.30) & (g < 0.50),
        "gap 50-100%":               (g >= 0.50) & (g < 1.00),
        "gap >100%":                 (g >= 1.00),
        "precio > $3":               (p > 3),
        "precio > $5":               (p > 5),
        "precio $3-$10":             (p >= 3) & (p <= 10),
        "opening_rvol alto (>p75)":  (rv >= rv.quantile(0.75)).to_numpy(),
        "OR estrecho (<p25)":        (orp <= orp.quantile(0.25)).to_numpy(),
        "gap30-50 & precio>$3":      (g >= 0.30) & (g < 0.50) & (p > 3),
        "precio>$5 & rvol alto":     (p > 5) & (rv >= rv.quantile(0.75)).to_numpy(),
    }
    print(f"\n### CORTES SIMPLES | label={label} | baseline {base:+.3f}R")
    print(f"   {'corte':<26}{'n':>5}{'exp':>9}{'win%':>6}{'IC95':>18}{'DSR':>7}")
    nombres = [k for k in cortes if k != "TODOS (baseline)"]
    orden = df.sort_values("fecha").index.to_numpy()
    bloques = np.array_split(orden, 8)
    perf = []
    for b in bloques:
        fila = []
        for k in nombres:
            m = np.asarray(cortes[k])[b]
            fila.append(R[b][m].mean() if m.sum() >= 3 else np.nan)
        perf.append(fila)
    perf = np.array(perf, float)
    perf = perf[~np.isnan(perf).any(axis=1)] if len(perf) else perf
    pbo = _pbo_cscv(perf) if len(perf) >= 4 else float("nan")
    best = None
    for k, m in cortes.items():
        m = np.asarray(m)
        if m.sum() < 20:
            print(f"   {k:<26}{int(m.sum()):>5}   (pocas ops)")
            continue
        Rs = R[m]
        dsr, _ = _deflated_sharpe(Rs, n_trials=len(nombres))
        lo, hi = _ic95(Rs)
        print(f"   {k:<26}{int(m.sum()):>5}{Rs.mean():>+8.3f}R{(Rs>0).mean()*100:>5.0f}%"
              f"  [{lo:>+6.3f},{hi:>+6.3f}]{dsr:>7.2f}")
        if k != "TODOS (baseline)" and (best is None or Rs.mean() > best["exp"]):
            best = dict(k=k, exp=Rs.mean(), dsr=dsr, ic=(lo, hi), n=int(m.sum()))
    if best:
        mejora = best["exp"] - base
        ver = _veredicto(mejora, best["dsr"], pbo, best["ic"][0])
        print(f"\n   PBO del menu de cortes: {pbo:.2f}  (prob de que el mejor IS quede <= mediana OOS)")
        print(f"   -> mejor corte: '{best['k']}' {best['exp']:+.3f}R ({mejora:+.3f}R vs baseline) "
              f"| n={best['n']} | DSR {best['dsr']:.2f} | IC95 [{best['ic'][0]:+.3f},{best['ic'][1]:+.3f}]")
        print(f"   -> VEREDICTO: {ver}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2023-07-01")
    ap.add_argument("--hasta", default="2024-12-31")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--slip", type=float, default=B.SLIP_BASE, help="slippage por lado (fraccion)")
    ap.add_argument("--construir", action="store_true", help="reconstruye el CSV desde Massive")
    ap.add_argument("--datos", default=CSV)
    ap.add_argument("--label", default="R_orb", choices=["R_orb", "R_gapgo"])
    ap.add_argument("--folds", type=int, default=6)
    a = ap.parse_args()
    if a.construir or not os.path.exists(a.datos):
        construir(a.desde, a.hasta, topk=a.topk, slip=a.slip, salida=a.datos)
    df = pd.read_csv(a.datos)
    meta_modelo(df, a.label, folds=a.folds)
    cortes_simples(df, a.label)
