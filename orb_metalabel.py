"""META-MODELO del ORB (López de Prado) con validación anti-overfitting (Carver).

Sobre la base de datos de breakouts ORB (orb_dataset.py), entrena un modelo secundario que
predice P(trade gane) a partir de las features, con PURGED WALK-FORWARD CV (sin fuga
temporal), y comprueba si FILTRAR los breakouts por esa probabilidad mejora la expectancy
OOS frente a tomar TODOS. Valida que la mejora sea real, no suerte-de-buscar, con:
  - Deflated Sharpe Ratio (Bailey & López de Prado): ¿bate al mejor de N intentos de ruido?
  - PBO / CSCV (Bailey et al.): ¿el ganador in-sample tiende a quedar bien o mal OOS?

Compara modelos: logística (simple, Carver) vs HistGradientBoosting (potente, LdP).

  python orb_metalabel.py [--datos datos/orb_dataset.csv] [--folds 6]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["opening_rvol", "gap", "or_range_atr", "or_range_pct", "breakout_min",
            "ext_breakout", "day_ret", "atr_pct", "hurst", "spy_day_ret", "spy_trend", "dow"]


def _walk_forward(df, modelo, folds=6, embargo=3):
    """Purged walk-forward: expanding window, purga 'embargo' días entre train y test para
    evitar fuga. Devuelve la prob OOS para cada operación (NaN donde no se predijo)."""
    df = df.sort_values("fecha").reset_index(drop=True)
    fechas = pd.to_datetime(df["fecha"])
    X = df[FEATURES].to_numpy(float)
    y = df["y"].to_numpy(int)
    n = len(df)
    prob = np.full(n, np.nan)
    bordes = [int(n * k / (folds + 1)) for k in range(1, folds + 2)]   # expanding
    for i in range(len(bordes) - 1):
        tr_end = bordes[i]
        te_ini, te_fin = bordes[i], bordes[i + 1]
        if tr_end < 60 or te_fin <= te_ini:
            continue
        # purga: quita del train las ops cuyos días estén a < embargo del inicio del test
        corte = fechas.iloc[te_ini] - pd.Timedelta(days=embargo)
        tr_mask = fechas.iloc[:tr_end] <= corte
        idx_tr = np.where(tr_mask.to_numpy())[0]
        if len(idx_tr) < 50 or len(np.unique(y[idx_tr])) < 2:
            continue
        Xtr, ytr = X[idx_tr], y[idx_tr]
        Xte = X[te_ini:te_fin]
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
    return prob


def _deflated_sharpe(R, n_trials):
    """PSR frente al Sharpe esperado del mejor de n_trials intentos de ruido (DSR)."""
    r = np.asarray(R, float)
    T = len(r)
    sd = r.std(ddof=1)
    if T < 20 or sd == 0:
        return float("nan"), float("nan")
    sr = r.mean() / sd
    g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False))
    emc = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(1.0 / (T - 1)) * ((1 - emc) * z1 + emc * z2)      # Sharpe esperado del ruido
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2)
    if denom <= 0:
        return float("nan"), sr
    dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    return float(dsr), float(sr)


def _pbo_cscv(perf):
    """PBO por CSCV. perf: matriz (bloques x configs) de rendimiento. Devuelve prob de que el
    ganador in-sample quede por DEBAJO de la mediana OOS (overfitting)."""
    from itertools import combinations
    S = perf.shape[0]
    if S < 4 or perf.shape[1] < 2:
        return float("nan")
    mitad = S // 2
    logits = []
    for comb in combinations(range(S), mitad):
        is_rows = list(comb); oos_rows = [r for r in range(S) if r not in comb]
        is_mean = perf[is_rows].mean(axis=0)
        oos_mean = perf[oos_rows].mean(axis=0)
        c_star = int(np.argmax(is_mean))                     # ganador in-sample
        # rango relativo OOS del ganador (1 = mejor, 0 = peor)
        rank = (oos_mean < oos_mean[c_star]).mean()
        w = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return float((logits <= 0).mean())                       # P(ganador IS queda <= mediana OOS)


def evaluar(datos="datos/orb_dataset.csv", folds=6):
    df = pd.read_csv(datos)
    df = df[np.isfinite(df[FEATURES].to_numpy(float)).all(axis=1)].reset_index(drop=True)
    R = df["R"].to_numpy(float)
    base_exp = R.mean()
    print(f"\n### META-MODELO ORB | {len(df)} breakouts | {df['fecha'].min()}..{df['fecha'].max()}")
    print(f"BASELINE (tomar TODOS): expectancy {base_exp:+.3f}R | win {df['y'].mean()*100:.0f}% | "
          f"Sharpe/op {R.mean()/R.std():+.3f}")
    umbrales = [0.40, 0.45, 0.50, 0.55, 0.60]
    for modelo in ("logistica", "hgb"):
        prob = _walk_forward(df, modelo, folds=folds)
        ok = np.isfinite(prob)
        print(f"\n[{modelo}]  OOS en {ok.sum()} ops")
        # matriz de perf por bloque x umbral (para PBO)
        bloques = np.array_split(np.where(ok)[0], 8)
        perf = []
        for b in bloques:
            fila = [R[b][prob[b] >= u].mean() if (prob[b] >= u).sum() else np.nan for u in umbrales]
            perf.append(fila)
        perf = np.array(perf, float)
        perf = perf[~np.isnan(perf).any(axis=1)] if len(perf) else perf
        print(f"   {'umbral':>7}{'% tomadas':>11}{'expectancy':>12}{'win%':>7}{'DSR':>7}")
        best = None
        for u in umbrales:
            sel = ok & (prob >= u)
            if sel.sum() < 20:
                print(f"   {u:>7.2f}   (pocas ops: {sel.sum()})"); continue
            Rs = R[sel]
            dsr, _ = _deflated_sharpe(Rs, n_trials=len(umbrales) * 2)
            win = (Rs > 0).mean()
            print(f"   {u:>7.2f}{sel.sum()/ok.sum()*100:>10.0f}%{Rs.mean():>+11.3f}R{win*100:>6.0f}%{dsr:>7.2f}")
            if best is None or Rs.mean() > best[1]:
                best = (u, Rs.mean(), dsr, sel.sum())
        pbo = _pbo_cscv(perf) if len(perf) >= 4 else float("nan")
        if best:
            mejora = best[1] - base_exp
            print(f"   -> mejor umbral {best[0]:.2f}: {best[1]:+.3f}R ({mejora:+.3f}R vs baseline) | "
                  f"DSR {best[2]:.2f} | PBO {pbo:.2f}")
            ver = "EDGE ROBUSTO" if (best[2] >= 0.95 and pbo <= 0.3 and mejora > 0) else \
                  ("MEJORA NO CONCLUYENTE" if mejora > 0 else "NO MEJORA")
            print(f"   -> VEREDICTO: {ver}  (DSR>=0.95 y PBO<=0.3 para fiarnos)")
    # importancia de features (HGB permutation, rápida)
    _importancia(df)


def _importancia(df):
    from sklearn.inspection import permutation_importance
    n = len(df); corte = int(n * 0.7)
    df = df.sort_values("fecha").reset_index(drop=True)
    X = df[FEATURES].to_numpy(float); y = df["y"].to_numpy(int)
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0)
    m.fit(X[:corte], y[:corte])
    r = permutation_importance(m, X[corte:], y[corte:], n_repeats=10, random_state=0, scoring="roc_auc")
    imp = sorted(zip(FEATURES, r.importances_mean), key=lambda x: -x[1])
    print("\n   IMPORTANCIA de features (permutation AUC, OOS):")
    for f, v in imp:
        print(f"     {f:<14}{v:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=str, default="datos/orb_dataset.csv")
    ap.add_argument("--folds", type=int, default=6)
    a = ap.parse_args()
    evaluar(a.datos, a.folds)
