"""META-MODELO ENSEMBLE intradía (meta-labeling de López de Prado) sobre señales DÉBILES.

Pregunta del usuario (la "mezcla" literal): varias señales intradía son, por separado,
DÉBILES o sub-coste (reversión idiosincrática del z, distancia a VWAP, reversión de 1-5
barras, volumen/rango/RVOL, hora del día). Hipótesis: un META-MODELO (meta-labeling: la señal
PRIMARIA = reversión; un modelo SECUNDARIO decide SI operar) que opere SOLO cuando la
convicción es alta dispara pocas veces -> menos trades, más edge por trade -> ¿puede cruzar el
coste donde cada señal suelta no llega?

Metodología (honestidad > resultado):
  - Datos survivorship-free: universo POINT-IN-TIME (universo_pit, top por dólar-volumen as-of
    con deslistados) + barras de minuto de Massive (máximo que sirve Starter: ~2022-01..hoy).
  - Señal primaria: resid = ret_L(stock) - beta*ret_L(SPY) (idiosincrático), z-score causal
    sobre ventana intradía. |z|>=Z_IN -> candidato a REVERTIR (z>0 corto, z<0 largo).
  - Etiqueta TRIPLE-BARRERA (LdP): barrera superior/inferior simétricas a +-g*sigma y barrera
    vertical (H barras o EOD). y=1 si el trade primario gana BRUTO. R y retorno neto guardados.
  - Features point-in-time (sin look-ahead): |z|, resid, ret de 1/2/3/5 barras, dist a VWAP,
    RVOL, rango, ATR%, minutos desde apertura, estado de SPY, beta, día de la semana.
  - Meta-modelo: HistGradientBoosting y Logística, con PURGED WALK-FORWARD + EMBARGO estricto.
  - Umbral de convicción: se elige IN-SAMPLE (train de cada fold) el que maximiza expectancy
    neta, y se aplica OOS (test del fold). Se reporta expectancy OOS NETA de coste.
  - Coste realista: TAKER (3 bps/lado -> 6 bps ida y vuelta) y MAKER (-0.25 bps/lado).
  - Anti-overfit: Deflated Sharpe Ratio (Bailey-LdP) + PBO/CSCV (Bailey et al.).
  - Test de RETARDO (entry_lag): si el edge se evapora al entrar 1 barra después, la
    "reversión" era rebote bid-ask (microestructura), no alfa predecible.

ANTI-LEAKAGE (explícito):
  1. Beta de cada día usa SOLO cierres diarios ANTERIORES (rolling 60 sesiones, shift +1 día).
  2. z-score causal: media/std del resid sobre ventana que TERMINA en t-1 (no incluye t).
  3. Features de t calculadas con barras <= t; el trade va de t (o t+lag) hacia delante.
  4. No solapamiento: tras un evento se salta hasta su barra de salida (etiquetas no solapadas).
  5. Purga + embargo entre train y test en el walk-forward (días entre bloques).
  6. Umbral elegido en train, aplicado en test (nunca se mira el OOS para elegirlo).
  7. entry_lag: desenmascara el rebote bid-ask.

  python backtest_ensemble_metamodel.py --top 40 --z-in 1.5 --out datos/ensemble_eventos.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import universo_pit as U
from fuentes.massive import ClienteMassive, barras_intradia, PAUSA_STARTER

BPS = 1e-4
C_TAKER = 3.0 * BPS       # 3 bps/lado cruzando a mercado (retail)
C_MAKER = -0.25 * BPS     # -0.25 bps/lado: límite pasivo + rebate (fondo market-maker)
CACHE = "datos/cache"

FEATURES = ["absz", "resid", "ret1", "ret2", "ret3", "ret5", "dist_vwap", "rvol",
            "bar_range", "atr_pct", "min_apertura", "spy_ret", "spy_trend", "beta", "dow"]


# ---------------------------------------------------------------- datos (cacheados)
def _min_cache(ticker, desde, hasta, cm):
    """Barras de 5m de un ticker, cacheadas en parquet (gitignored). Vacío si no hay."""
    ruta = os.path.join(CACHE, f"min5_{ticker}_{desde}_{hasta}.parquet")
    if os.path.exists(ruta):
        try:
            return pd.read_parquet(ruta)
        except Exception:
            pass
    try:
        df = barras_intradia(ticker, "5Min", desde, hasta, cm=cm)
    except Exception:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(ruta)
    return df


def _daily_from_min(df):
    """Cierre diario a partir de barras de minuto (último close de cada sesión)."""
    g = df["close"].groupby(df.index.normalize())
    return g.last()


def _beta_diaria(daily_stock, daily_spy, win=60):
    """Beta rolling causal (60 sesiones) vs SPY, DESPLAZADA +1 día: la beta de la sesión D
    usa solo retornos hasta D-1. Devuelve Series indexada por fecha-sesión (normalizada)."""
    rs = daily_stock.pct_change()
    rm = daily_spy.reindex(daily_stock.index).pct_change()
    cov = rs.rolling(win, min_periods=20).cov(rm)
    var = rm.rolling(win, min_periods=20).var()
    beta = (cov / var).clip(0.0, 3.0)
    return beta.shift(1).fillna(1.0)   # shift +1 sesión -> estrictamente pasado


# ---------------------------------------------------------------- señal + etiqueta
def _construir_ticker(t, dfm, spy_close, spy_ret_sesion, spy_trend, L, W, Z_IN,
                      g_barrera, H, entry_lag):
    """Genera eventos (señal primaria de reversión) + features + etiqueta triple-barrera
    para un ticker. Point-in-time estricto, sobre la serie CONTINUA (el z-score se normaliza
    con una ventana que puede cruzar sesiones; el retorno de L barras y las barreras NO cruzan
    la noche). Devuelve lista de dicts (uno por evento)."""
    if len(dfm) < 400:
        return []
    idx = dfm.index
    close = dfm["close"].to_numpy(float)
    high = dfm["high"].to_numpy(float)
    low = dfm["low"].to_numpy(float)
    vol = dfm["volume"].to_numpy(float)
    dia = idx.normalize()
    n = len(close)
    # id de sesión por barra y posición dentro de la sesión (minuto desde apertura)
    dia_codes, _ = pd.factorize(dia)
    primera_de_dia = np.concatenate([[True], dia_codes[1:] != dia_codes[:-1]])
    pos_en_dia = np.zeros(n, int)
    c_pos = 0
    for i in range(n):
        c_pos = 0 if primera_de_dia[i] else c_pos + 1
        pos_en_dia[i] = c_pos
    # última barra de cada sesión (para la barrera vertical / EOD)
    ult_de_dia = np.concatenate([primera_de_dia[1:], [True]])
    fin_sesion = np.empty(n, int)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if ult_de_dia[i]:
            last = i
        fin_sesion[i] = last

    # beta diaria causal (rolling 60 sesiones, shift +1 día)
    d_stock = _daily_from_min(dfm)
    d_spy = spy_close.groupby(spy_close.index.normalize()).last()
    beta_s = _beta_diaria(d_stock, d_spy, win=60)
    beta_bar = beta_s.reindex(dia).to_numpy()
    beta_bar = np.where(np.isfinite(beta_bar), beta_bar, 1.0)

    # SPY alineado a la rejilla del ticker (ffill dentro del día)
    spy_al = spy_close.reindex(idx).groupby(dia).ffill().to_numpy(float)
    spyret_al = spy_ret_sesion.reindex(idx).groupby(dia).ffill().to_numpy(float)
    spytr_al = spy_trend.reindex(idx).groupby(dia).ffill().to_numpy(float)

    # ret de 1 barra (NaN en la primera barra del día: no cruza noche)
    r1 = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
    r1[primera_de_dia] = np.nan
    # ret de L barras (solo si las L barras son del MISMO día)
    rL = np.full(n, np.nan)
    rLs = np.full(n, np.nan)
    for i in range(L, n):
        if pos_en_dia[i] >= L:                 # las L barras previas son del mismo día
            rL[i] = close[i] / close[i - L] - 1.0
            if spy_al[i - L] > 0:
                rLs[i] = spy_al[i] / spy_al[i - L] - 1.0
    resid = rL - beta_bar * rLs

    # z-score causal del resid: media/std sobre [i-W, i-1] (continuo, no incluye i)
    sr = pd.Series(resid)
    mu = sr.shift(1).rolling(W, min_periods=max(20, W // 2)).mean().to_numpy()
    sd = sr.shift(1).rolling(W, min_periods=max(20, W // 2)).std().to_numpy()
    z = np.where(sd > 0, (resid - mu) / sd, np.nan)
    # sigma de 1-barra para las barreras (causal)
    sig = pd.Series(r1).shift(1).rolling(W, min_periods=max(20, W // 2)).std().to_numpy()

    # VWAP de sesión (acumulado causal por día) y RVOL (vol / media 20 barras continua)
    tp = (high + low + close) / 3.0
    vw = np.full(n, np.nan)
    for d in np.unique(dia_codes):
        m = dia_codes == d
        pv = np.cumsum(tp[m] * vol[m]); vv = np.cumsum(vol[m])
        vw[m] = np.where(vv > 0, pv / vv, close[m])
    vm = pd.Series(vol).rolling(20, min_periods=5).mean().to_numpy()
    rv = np.where(vm > 0, vol / vm, 1.0)
    trv = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)),
                                            np.abs(low - np.roll(close, 1))))
    trv[primera_de_dia] = (high - low)[primera_de_dia]
    atr = pd.Series(trv).rolling(14, min_periods=5).mean().to_numpy() / close

    filas = []
    i = 1
    while i < n - (entry_lag + 2):
        zi = z[i]
        if not np.isfinite(zi) or abs(zi) < Z_IN:
            i += 1; continue
        s = sig[i]
        if not np.isfinite(s) or s <= 0:
            i += 1; continue
        direc = -1 if zi > 0 else 1            # z>0 subió de más -> SHORT ; z<0 -> LONG
        e = i + entry_lag
        if e >= n - 1 or dia_codes[e] != dia_codes[i]:
            i += 1; continue
        entry = close[e]
        up = entry * (1 + g_barrera * s); dn = entry * (1 - g_barrera * s)
        vbar = min(e + H, fin_sesion[e])       # barrera vertical: H barras o EOD (no cruza noche)
        exit_px = close[vbar]; exit_j = vbar
        for j in range(e + 1, vbar + 1):
            if high[j] >= up:
                exit_px = up; exit_j = j; break
            if low[j] <= dn:
                exit_px = dn; exit_j = j; break
        gross = direc * (exit_px / entry - 1.0)
        filas.append({
            "fecha": pd.Timestamp(dia[i]).strftime("%Y-%m-%d"), "ticker": t, "direc": int(direc),
            "absz": float(abs(zi)), "resid": float(resid[i]) if np.isfinite(resid[i]) else 0.0,
            "ret1": float(r1[i]) if np.isfinite(r1[i]) else 0.0,
            "ret2": float(close[i] / close[i - 2] - 1.0) if pos_en_dia[i] >= 2 else 0.0,
            "ret3": float(close[i] / close[i - 3] - 1.0) if pos_en_dia[i] >= 3 else 0.0,
            "ret5": float(close[i] / close[i - 5] - 1.0) if pos_en_dia[i] >= 5 else 0.0,
            "dist_vwap": float(close[i] / vw[i] - 1.0) if vw[i] > 0 else 0.0,
            "rvol": float(rv[i]) if np.isfinite(rv[i]) else 1.0,
            "bar_range": float((high[i] - low[i]) / close[i]) if close[i] > 0 else 0.0,
            "atr_pct": float(atr[i]) if np.isfinite(atr[i]) else 0.0,
            "min_apertura": float(pos_en_dia[i] * 5),
            "spy_ret": float(spyret_al[i]) if np.isfinite(spyret_al[i]) else 0.0,
            "spy_trend": float(spytr_al[i]) if np.isfinite(spytr_al[i]) else 0.0,
            "beta": float(beta_bar[i]), "dow": float(pd.Timestamp(dia[i]).dayofweek),
            "gross": float(gross), "y": int(gross > 0), "hold_barras": int(exit_j - e),
        })
        i = exit_j + 1          # no solapamiento: salta hasta la salida
    return filas


def construir(top=40, desde="2022-01-03", hasta="2026-09-30", L=5, W=78, Z_IN=1.5,
              g_barrera=1.0, H=12, entry_lag=0, out="datos/ensemble_eventos.csv",
              form_desde=None, form_hasta=None):
    cm = ClienteMassive(pausa_min=PAUSA_STARTER, cache_dir=CACHE)
    form_desde = form_desde or (pd.Timestamp(desde) - pd.Timedelta(days=75)).strftime("%Y-%m-%d")
    form_hasta = form_hasta or (pd.Timestamp(desde) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"[universo] formando PIT top-{top} as-of {form_desde}..{form_hasta} (survivorship-free)")
    tickers, _, ranking = U.formar(form_desde, form_hasta, top=top, cm=cm)
    print(f"[universo] {len(tickers)} tickers: {tickers[:12]}...")

    spy = _min_cache("SPY", desde, hasta, cm)
    if spy.empty:
        raise RuntimeError("SPY intradía vacío")
    spy_close = spy["close"].astype(float)
    # ret de sesión de SPY (causal) y tendencia (close > SMA50 de 5m)
    spy_sesion = spy_close / spy_close.groupby(spy.index.normalize()).transform("first") - 1.0
    spy_ma = spy_close.rolling(50, min_periods=10).mean()
    spy_trend = (spy_close > spy_ma).astype(float)

    filas = []
    for k, t in enumerate(tickers):
        dfm = _min_cache(t, desde, hasta, cm)
        if dfm.empty:
            continue
        fs = _construir_ticker(t, dfm, spy_close, spy_sesion, spy_trend, L, W, Z_IN,
                               g_barrera, H, entry_lag)
        filas += fs
        if (k + 1) % 10 == 0:
            print(f"  ...{k+1}/{len(tickers)} tickers, {len(filas)} eventos (llamadas {cm.llamadas})")
    df = pd.DataFrame(filas)
    os.makedirs("datos", exist_ok=True)
    df.to_csv(out, index=False)
    span = f"{df['fecha'].min()}..{df['fecha'].max()}" if len(df) else "-"
    print(f"\nDATASET: {len(df)} eventos | {df['ticker'].nunique() if len(df) else 0} tickers | "
          f"{span} -> {out}")
    if len(df):
        print(f"  win-rate bruto {df['y'].mean()*100:.1f}% | gross medio {df['gross'].mean()/BPS:+.2f}bps | "
              f"hold medio {df['hold_barras'].mean():.1f} barras")
    return df


# ---------------------------------------------------------------- meta-modelo + validación
def _walk_forward(df, modelo, folds=6, embargo=5):
    """Purged walk-forward (expanding). Purga 'embargo' días entre fin de train e inicio de
    test. Devuelve (prob OOS por evento, lista de (idx_train, idx_test))."""
    df = df.sort_values("fecha").reset_index(drop=True)
    fechas = pd.to_datetime(df["fecha"])
    X = df[FEATURES].to_numpy(float)
    y = df["y"].to_numpy(int)
    n = len(df)
    prob = np.full(n, np.nan)
    splits = []
    bordes = [int(n * k / (folds + 1)) for k in range(1, folds + 2)]
    for i in range(len(bordes) - 1):
        tr_end = bordes[i]; te_ini, te_fin = bordes[i], bordes[i + 1]
        if tr_end < 80 or te_fin <= te_ini:
            continue
        corte = fechas.iloc[te_ini] - pd.Timedelta(days=embargo)
        tr_mask = fechas.iloc[:tr_end] <= corte
        idx_tr = np.where(tr_mask.to_numpy())[0]
        if len(idx_tr) < 60 or len(np.unique(y[idx_tr])) < 2:
            continue
        Xtr, ytr = X[idx_tr], y[idx_tr]
        Xte = X[te_ini:te_fin]
        if modelo == "logistica":
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(max_iter=1000, class_weight="balanced")
            m.fit(sc.transform(Xtr), ytr)
            prob[te_ini:te_fin] = m.predict_proba(sc.transform(Xte))[:, 1]
        else:
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                               learning_rate=0.05, l2_regularization=1.0)
            m.fit(Xtr, ytr)
            prob[te_ini:te_fin] = m.predict_proba(Xte)[:, 1]
        splits.append((idx_tr, np.arange(te_ini, te_fin)))
    return prob, splits


def _deflated_sharpe(R, n_trials):
    r = np.asarray(R, float); T = len(r); sd = r.std(ddof=1)
    if T < 20 or sd == 0:
        return float("nan"), float("nan")
    sr = r.mean() / sd
    g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False))
    emc = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / n_trials); z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(1.0 / (T - 1)) * ((1 - emc) * z1 + emc * z2)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2)
    if denom <= 0:
        return float("nan"), sr
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)), float(sr)


def _pbo_cscv(perf):
    from itertools import combinations
    S = perf.shape[0]
    if S < 4 or perf.shape[1] < 2:
        return float("nan")
    mitad = S // 2; logits = []
    for comb in combinations(range(S), mitad):
        is_rows = list(comb); oos_rows = [r for r in range(S) if r not in comb]
        is_mean = perf[is_rows].mean(axis=0); oos_mean = perf[oos_rows].mean(axis=0)
        c_star = int(np.argmax(is_mean))
        rank = (oos_mean < oos_mean[c_star]).mean()
        w = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    return float((np.array(logits) <= 0).mean())


def _net(gross, c_side):
    """Retorno neto por trade = bruto - coste ida y vuelta (2 lados)."""
    return gross - 2.0 * c_side


def evaluar(df, folds=6, embargo=5, umbrales=(0.45, 0.50, 0.55, 0.60, 0.65)):
    df = df[np.isfinite(df[FEATURES].to_numpy(float)).all(axis=1)].reset_index(drop=True)
    df = df.sort_values("fecha").reset_index(drop=True)
    gross = df["gross"].to_numpy(float)
    res = {"n": len(df), "win_bruto": float(df["y"].mean()),
           "gross_bps": float(gross.mean() / BPS), "modelos": {}}
    print(f"\n### META-MODELO ENSEMBLE | {len(df)} eventos | {df['fecha'].min()}..{df['fecha'].max()}")
    for nombre, c in (("BRUTO", 0.0), ("TAKER", C_TAKER), ("MAKER", C_MAKER)):
        base = _net(gross, c).mean() / BPS
        print(f"BASELINE (todos los eventos) {nombre}: expectancy {base:+.3f} bps | "
              f"win {df['y'].mean()*100:.1f}%")

    for modelo in ("logistica", "hgb"):
        prob, splits = _walk_forward(df, modelo, folds=folds, embargo=embargo)
        ok = np.isfinite(prob)
        print(f"\n[{modelo}] OOS en {ok.sum()}/{len(df)} eventos")
        mres = {"oos_n": int(ok.sum()), "coste": {}}
        # matriz perf por bloque x umbral (para PBO) sobre el BRUTO neteado TAKER
        for nombre, c in (("TAKER", C_TAKER), ("MAKER", C_MAKER)):
            net_all = _net(gross, c)
            # --- umbral elegido IN-SAMPLE por fold, aplicado OOS ---
            oos_net, oos_mask = [], np.zeros(len(df), bool)
            for idx_tr, idx_te in splits:
                # prob del train: re-predecir no; usamos prob OOS solo en test. Para elegir
                # umbral in-sample necesitamos prob IN-SAMPLE -> reentrenar y predecir train.
                ptr = _prob_train(df, modelo, idx_tr)
                best_u, best_v = None, -1e9
                for u in umbrales:
                    sel = ptr >= u
                    if sel.sum() < 15:
                        continue
                    v = net_all[idx_tr][sel].mean()
                    if v > best_v:
                        best_v, best_u = v, u
                if best_u is None:
                    continue
                sel_te = prob[idx_te] >= best_u
                take = idx_te[sel_te]
                oos_mask[take] = True
                oos_net.append(net_all[take])
            oos_net = np.concatenate(oos_net) if oos_net else np.array([])
            # --- barrido fijo de umbrales (para DSR/PBO y tabla) ---
            perf = []
            bloques = np.array_split(np.where(ok)[0], 8)
            for b in bloques:
                fila = [net_all[b][prob[b] >= u].mean() if (prob[b] >= u).sum() else np.nan
                        for u in umbrales]
                perf.append(fila)
            perf = np.array(perf, float)
            perf = perf[~np.isnan(perf).any(axis=1)] if len(perf) else perf
            pbo = _pbo_cscv(perf) if len(perf) >= 4 else float("nan")

            tabla = []
            for u in umbrales:
                sel = ok & (prob >= u)
                if sel.sum() < 20:
                    tabla.append((u, int(sel.sum()), np.nan, np.nan, np.nan)); continue
                ns = net_all[sel]
                dsr, _ = _deflated_sharpe(ns, n_trials=len(umbrales) * 2)
                tabla.append((u, int(sel.sum()), float(ns.mean() / BPS),
                              float((ns > 0).mean()), float(dsr)))
            exp_oos = float(oos_net.mean() / BPS) if len(oos_net) else float("nan")
            dsr_oos, _ = _deflated_sharpe(oos_net, n_trials=len(umbrales) * 2) if len(oos_net) >= 20 else (float("nan"), 0)
            # IC95 bootstrap por día
            ic = _boot_ic(df[oos_mask], oos_net) if len(oos_net) >= 20 else (float("nan"), float("nan"))
            print(f"  --- coste {nombre} ---  (umbral elegido IN-SAMPLE, aplicado OOS)")
            print(f"    {'umbral':>7}{'n':>7}{'exp(bps)':>10}{'win%':>7}{'DSR':>7}")
            for u, nsel, e, w, d in tabla:
                if e != e:
                    print(f"    {u:>7.2f}{nsel:>7}   (pocas)"); continue
                print(f"    {u:>7.2f}{nsel:>7}{e:>+10.3f}{w*100:>6.0f}%{d:>7.2f}")
            print(f"    -> OOS (umbral in-sample): expectancy {exp_oos:+.3f} bps | "
                  f"n={len(oos_net)} | DSR {dsr_oos:.2f} | PBO {pbo:.2f} | "
                  f"IC95[{ic[0]:+.3f},{ic[1]:+.3f}]bps")
            ver = _veredicto(exp_oos, dsr_oos, pbo, ic)
            print(f"    -> VEREDICTO: {ver}")
            mres["coste"][nombre] = {"exp_oos_bps": exp_oos, "n_oos": int(len(oos_net)),
                                     "dsr_oos": float(dsr_oos), "pbo": float(pbo),
                                     "ic95": [float(ic[0]), float(ic[1])],
                                     "tabla": tabla, "veredicto": ver}
        res["modelos"][modelo] = mres
    return res


def _prob_train(df, modelo, idx_tr):
    """Prob in-sample sobre el propio train (para elegir umbral SIN mirar el test).
    Usa un holdout interno simple: entrena en el 70% más antiguo del train, predice el 100%
    del train -> sirve para rankear umbrales (no es OOS, es selección in-sample honesta)."""
    X = df[FEATURES].to_numpy(float); y = df["y"].to_numpy(int)
    Xtr, ytr = X[idx_tr], y[idx_tr]
    if len(np.unique(ytr)) < 2:
        return np.zeros(len(idx_tr))
    if modelo == "logistica":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xtr))[:, 1]
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                       l2_regularization=1.0).fit(Xtr, ytr)
    return m.predict_proba(Xtr)[:, 1]


def _boot_ic(df_sel, net, n=2000, seed=1):
    """IC95 de la expectancy neta por bootstrap de BLOQUES-DÍA (respeta autocorrelación)."""
    if len(net) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    d = df_sel.copy(); d = d.reset_index(drop=True); d["_v"] = net
    grupos = [g["_v"].to_numpy() for _, g in d.groupby("fecha")]
    if len(grupos) < 3:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(grupos), len(grupos))
        cat = np.concatenate([grupos[i] for i in pick])
        means.append(cat.mean() / BPS)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _veredicto(exp_oos, dsr, pbo, ic):
    if exp_oos != exp_oos:
        return "SIN MUESTRA OOS"
    if exp_oos <= 0:
        return "NO CRUZA EL COSTE (expectancy OOS <= 0)"
    robusto = (dsr >= 0.95) and (pbo <= 0.30) and (ic[0] > 0)
    return "EDGE ROBUSTO NETO" if robusto else "POSITIVO PERO NO ROBUSTO (overfit probable)"


def importancia(df):
    from sklearn.inspection import permutation_importance
    df = df.sort_values("fecha").reset_index(drop=True)
    n = len(df); corte = int(n * 0.7)
    X = df[FEATURES].to_numpy(float); y = df["y"].to_numpy(int)
    if len(np.unique(y[:corte])) < 2:
        return []
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                       l2_regularization=1.0).fit(X[:corte], y[:corte])
    r = permutation_importance(m, X[corte:], y[corte:], n_repeats=8, random_state=0, scoring="roc_auc")
    imp = sorted(zip(FEATURES, r.importances_mean), key=lambda x: -x[1])
    print("\n   IMPORTANCIA de features (permutation AUC, OOS):")
    for f, v in imp:
        print(f"     {f:<14}{v:+.4f}")
    return imp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--desde", default="2022-01-03")
    ap.add_argument("--hasta", default="2026-09-30")
    ap.add_argument("--L", type=int, default=5, help="barras del retorno de la señal (5x5m=25min)")
    ap.add_argument("--W", type=int, default=78, help="ventana del z-score (78x5m=1 sesión)")
    ap.add_argument("--z-in", type=float, default=1.5)
    ap.add_argument("--g", type=float, default=1.0, help="barreras a +-g*sigma")
    ap.add_argument("--H", type=int, default=12, help="barrera vertical (barras, 12x5m=60min)")
    ap.add_argument("--entry-lag", type=int, default=0)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--out", default="datos/ensemble_eventos.csv")
    ap.add_argument("--solo-eval", action="store_true", help="salta construcción, lee --out")
    a = ap.parse_args()
    if a.solo_eval and os.path.exists(a.out):
        df = pd.read_csv(a.out)
    else:
        df = construir(a.top, a.desde, a.hasta, a.L, a.W, a.z_in, a.g, a.H, a.entry_lag, a.out)
    if len(df) >= 100:
        evaluar(df, folds=a.folds, embargo=a.embargo)
        importancia(df)
    else:
        print(f"Pocos eventos ({len(df)}): no se evalúa.")
