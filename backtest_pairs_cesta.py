"""ESTRATEGIA B - PAIRS / STAT-ARB en CESTA, intradia, con doble coste.

El pairs clasico (Gatev et al.): dos acciones que historicamente se mueven juntas;
cuando su spread se separa (z-score alto) apuestas a que vuelve. Un par suelto es
ruidoso; un FONDO tiene una CESTA de muchos pares y se beneficia de la
diversificacion (los ruidos se cancelan, queda la señal de reversion).

Aqui:
  - Selección de pares POINT-IN-TIME: con los retornos DIARIOS de la ventana de
    formacion (anterior a la prueba) se elige, de forma voraz, la cesta de N pares
    con mayor correlacion (y su hedge ratio OLS). Sin look-ahead.
  - Señal intradia: z-score del spread (logPA - h*logPB) con media/desv rolling
    CAUSAL (solo barras pasadas). Entra en |z|>ENTRA, sale en |z|<SALE, stop en
    |z|>STOP, y FLAT al cierre de sesion (nada overnight).
  - Cesta neteada: las posiciones de los N pares se agregan en un vector de pesos
    por nombre; se mide el retorno y el turnover de la CARTERA (no de un par suelto)
    -> Sharpe de cartera con diversificacion.
  - Dos escenarios de coste (se reportan ambos), con la MISMA contabilidad de
    turnover que la Estrategia A:
      TAKER (retail, a mercado): c_side = 3 bps/lado.
      MAKER (fondo, limite pasivo en la banda z + rebate): c_side = -0.25 bps/lado.
  - 3 semillas, bootstrap por bloques-dia IC95, y test de rebote bid-ask (entry_lag).

Uso:
  python backtest_pairs_cesta.py --top 40 --pares 10 --entra 2.0 --sale 0.5
"""
import argparse
import itertools

import numpy as np
import pandas as pd

import universo_pit as U
import backtest_xsec_intradia as B
from fuentes.massive import ClienteMassive, PAUSA_STARTER


def elegir_pares(cierres_form, top_names, n_pares=10, min_corr=0.5):
    """Selección voraz de pares por correlacion de retornos DIARIOS en formacion.
    Devuelve lista de (A, B, hedge_ratio). hedge_ratio h = cov/var de OLS de
    logPA ~ logPB (en niveles-log) en la ventana de formacion."""
    px = cierres_form[top_names].dropna(axis=1, how="any")
    rets = px.pct_change().dropna()
    corr = rets.corr()
    nombres = list(px.columns)
    cand = []
    for a, b in itertools.combinations(nombres, 2):
        c = corr.loc[a, b]
        if np.isfinite(c) and c >= min_corr:
            cand.append((c, a, b))
    cand.sort(reverse=True)
    usados, pares = set(), []
    for c, a, b in cand:
        if a in usados or b in usados:
            continue                      # cada nombre en un solo par (independencia)
        la, lb = np.log(px[a].values), np.log(px[b].values)
        h = float(np.polyfit(lb, la, 1)[0])   # hedge ratio OLS (logPA sobre logPB)
        if not (0.2 < h < 5.0):
            continue
        pares.append((a, b, h))
        usados.add(a); usados.add(b)
        if len(pares) >= n_pares:
            break
    return pares


def _pesos_cesta(closes, pares, grid, W=78, entra=2.0, sale=0.5, stop=3.5,
                 entry_lag=0):
    """Matriz de pesos por nombre y barra para la CESTA de pares.
    z-score rolling causal (ventana W, trailing), continuo salvo flat al cierre.
    Cada par aporta gross 1/len(pares); dentro del par, dolar-neutral.
    Devuelve (W_df [ts x nombres], n_trades)."""
    dia = grid.normalize()
    cols = closes.columns
    Wm = pd.DataFrame(0.0, index=grid, columns=cols)
    gpp = 1.0 / len(pares)                 # gross por par
    n_trades = 0
    for (a, b, h) in pares:
        if a not in cols or b not in cols:
            continue
        la = np.log(closes[a]); lb = np.log(closes[b])
        spread = la - h * lb
        mu = spread.rolling(W, min_periods=W // 2).mean()
        sd = spread.rolling(W, min_periods=W // 2).std()
        z = (spread - mu) / sd
        zv = z.values
        pos = 0                            # +1 long spread (long A/short B), -1 short spread
        # dentro del par: pesos dolar-neutral normalizados a gross gpp
        wa = gpp * 0.5
        wb = gpp * 0.5
        state = np.zeros(len(grid))
        d_arr = dia
        prev_day = None
        for i in range(len(grid)):
            if prev_day is not None and d_arr[i] != prev_day:
                pos = 0                    # flat al cambiar de dia (EOD flat)
            prev_day = d_arr[i]
            zc = zv[i]
            if not np.isfinite(zc):
                state[i] = pos; continue
            if pos == 0:
                if zc >= entra:
                    pos = -1; n_trades += 1       # spread caro -> short spread
                elif zc <= -entra:
                    pos = +1; n_trades += 1       # spread barato -> long spread
            else:
                if abs(zc) <= sale or abs(zc) >= stop:
                    pos = 0
            state[i] = pos
        # relleno con lag: la decision en i se aplica en i+lag
        if entry_lag:
            state = np.concatenate([np.zeros(entry_lag), state[:-entry_lag]])
        s = pd.Series(state, index=grid)
        Wm[a] = Wm[a] + s * wa            # long spread -> +A
        Wm[b] = Wm[b] - s * (wb)          # long spread -> -B
    return Wm, n_trades


def _periodos_desde_pesos(closes, Wm, grid):
    """Retorno y turnover por BARRA de la cartera (forma en t, cobra t->t+1).
    Flat al cierre (el ultimo peso del dia se liquida -> turnover de salida)."""
    dia = grid.normalize()
    ret_names = closes.pct_change().shift(-1)   # retorno de t a t+1 por nombre
    filas = []
    w_prev = pd.Series(0.0, index=closes.columns)
    pos_all = np.arange(len(grid))
    for d, idxs in pd.Series(pos_all, index=grid).groupby(dia):
        barras = idxs.to_numpy()
        w_prev = pd.Series(0.0, index=closes.columns)
        for bi in barras:
            w = Wm.iloc[bi]
            fwd = ret_names.iloc[bi].reindex(closes.columns).fillna(0.0)
            ret = float((w * fwd).sum())
            turn = float((w - w_prev).abs().sum())
            filas.append({"fecha": d, "ret": ret, "turn": turn})
            w_prev = w
        # liquidar al cierre
        if (w_prev != 0).any():
            filas.append({"fecha": d, "ret": 0.0, "turn": float(w_prev.abs().sum())})
    return pd.DataFrame(filas)


def correr(top=40, n_pares=10, form_desde="2024-01-02", form_hasta="2024-02-29",
           test_desde="2024-03-01", test_hasta="2024-05-24", mult=5, W=78,
           entra=2.0, sale=0.5, stop=3.5, dias_muestra=40, seeds=(1, 2, 3),
           entry_lag=0, verbose=True):
    cm = ClienteMassive(pausa_min=PAUSA_STARTER)
    tickers, cf, _ = U.formar(form_desde, form_hasta, top=top, cm=cm, verbose=verbose)
    closes, spy, grid = B._panel_intradia(tickers, mult, test_desde, test_hasta, cm)
    pares = elegir_pares(cf, list(closes.columns), n_pares=n_pares)
    if verbose:
        print(f"[pares] {len(pares)} elegidos: " +
              ", ".join(f"{a}/{b}(h={h:.2f})" for a, b, h in pares))
    Wm, n_trades = _pesos_cesta(closes, pares, grid, W=W, entra=entra, sale=sale,
                                stop=stop, entry_lag=entry_lag)
    per = _periodos_desde_pesos(closes, Wm, grid)
    if per.empty:
        print("sin periodos"); return None
    pdh = max(1, int(np.median(per[per.ret != 0].groupby("fecha").size())) or 1)
    dias = sorted(per["fecha"].unique())
    res = {"n_pares": len(pares), "pares": [(a, b, round(h, 3)) for a, b, h in pares],
           "n_trades": n_trades, "turn_medio": float(per["turn"].mean()),
           "per_dia": pdh, "ndias": len(dias), "n_nombres": int(closes.shape[1]),
           "seeds": {"BRUTO": [], "TAKER": [], "MAKER": []}, "pooled": {}}
    for sd in seeds:
        rs = np.random.default_rng(sd)
        pick = rs.choice(len(dias), min(dias_muestra, len(dias)), replace=False)
        dsel = set(dias[i] for i in pick)
        p = per[per["fecha"].isin(dsel)].reset_index(drop=True)
        for nombre, c in (("BRUTO", 0.0), ("TAKER", B.C_TAKER), ("MAKER", B.C_MAKER)):
            m = B._metricas(p, B._neto(p, c), pdh)
            if m:
                m["seed"] = sd
                res["seeds"][nombre].append(m)
    for nombre, c in (("BRUTO", 0.0), ("TAKER", B.C_TAKER), ("MAKER", B.C_MAKER)):
        res["pooled"][nombre] = B._metricas(per, B._neto(per, c), pdh)
    if verbose:
        _imprimir(res)
    return res


def _imprimir(res):
    print(f"\n### ESTRATEGIA B (cesta {res['n_pares']} pares) | {res['n_nombres']} nombres | "
          f"{res['n_trades']} trades | turn {res['turn_medio']:.3f} | {res['ndias']} dias")
    for nombre in ("BRUTO", "TAKER", "MAKER"):
        print(f"--- {nombre} ---")
        for m in res["seeds"][nombre]:
            print(f"  seed {m['seed']}: Sh {m['sharpe']:+.2f} | exp {m['exp_bps']:+.3f}bps "
                  f"IC[{m['ic_lo_bps']:+.3f},{m['ic_hi_bps']:+.3f}] | maxDD {m['maxdd_pct']:+.2f}%")
        p = res["pooled"][nombre]
        if p:
            vd = "ROBUSTO+" if p["ic_lo_bps"] > 0 else ("SIN EDGE" if p["ic_hi_bps"] > 0 else "PIERDE")
            print(f"  POOLED : Sh {p['sharpe']:+.2f} | exp {p['exp_bps']:+.3f}bps "
                  f"IC[{p['ic_lo_bps']:+.3f},{p['ic_hi_bps']:+.3f}] | maxDD {p['maxdd_pct']:+.2f}% -> {vd}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--pares", type=int, default=10)
    ap.add_argument("--form-desde", default="2024-01-02")
    ap.add_argument("--form-hasta", default="2024-02-29")
    ap.add_argument("--test-desde", default="2024-03-01")
    ap.add_argument("--test-hasta", default="2024-05-24")
    ap.add_argument("--W", type=int, default=78)
    ap.add_argument("--entra", type=float, default=2.0)
    ap.add_argument("--sale", type=float, default=0.5)
    ap.add_argument("--stop", type=float, default=3.5)
    ap.add_argument("--dias-muestra", type=int, default=40)
    ap.add_argument("--lag", type=int, default=0)
    a = ap.parse_args()
    correr(a.top, a.pares, a.form_desde, a.form_hasta, a.test_desde, a.test_hasta,
           5, a.W, a.entra, a.sale, a.stop, a.dias_muestra, entry_lag=a.lag)
