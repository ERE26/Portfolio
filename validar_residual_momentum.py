"""Validacion REPRODUCIBLE del RESIDUAL MOMENTUM (Blitz, Huij & Martens 2011) y por que
NO se despliega como libro propio. Resultado NEGATIVO, documentado a proposito.

Idea: momentum sobre los RESIDUOS de una regresion Fama-French 3 (Mkt-RF, SMB, HML), no
sobre el retorno total. Al quitar la beta a los factores queda el momentum idiosincratico;
en la literatura (universos amplios, miles de valores) da mayor Sharpe y evita los
'momentum crashes'.

VEREDICTO en NUESTRO universo (111 mega-caps US locales, con sesgo de supervivencia):
  - L/S: Sharpe ~0.2, DSR vs azar 0.63 (<0.95) y PBO 0.85 (sobreajuste alto) -> NO PASA.
  - Correlacion con el momentum convencional que ya corre el Libro B: 0.60 -> NO es un
    diversificador fuerte, es una version mas limpia de lo mismo.
  - Long-only sale alto (Sharpe ~1.2) pero esta INFLADO por sesgo de supervivencia
    (el universo son los supervivientes de hoy); no es un numero fiable.
  - Lo UNICO real que reproduce: es neutral a factores por construccion (corr ~0 con
    Mkt-RF/SMB/HML) y su L/S tiene la MITAD de vol y de drawdown que el convencional
    (-41% vs -71%): crash-resistance. Eso es un argumento para, algun dia, neutralizar por
    factores la señal del Libro B, NO para un libro nuevo; y requeriria un universo AMPLIO
    y sin sesgo de supervivencia (vendor de datos), que hoy no tenemos.

Conclusion honesta: investigacion valiosa, pero NO se cablea. Se revisita si llega un
universo adecuado.

  python validar_residual_momentum.py
"""
import glob, os, io, zipfile, urllib.request, itertools
import numpy as np, pandas as pd
from patrones.robustez import deflated_sharpe_ratio, probabilidad_sobreajuste, sharpe_no_anualizado

FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
ELEG = (36, 11, 1, 0.2)        # win, form, skip, q (config pre-especificada, L/S)


def ff3_mensual():
    raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(FF_URL, timeout=60).read())).read(
        "F-F_Research_Data_Factors_daily.csv").decode("latin1").splitlines()
    filas = [l for l in raw if l[:8].strip().isdigit() and len(l.split(",")[0].strip()) == 8]
    df = pd.read_csv(io.StringIO("\n".join(filas)), header=None,
                     names=["fecha", "MktRF", "SMB", "HML", "RF"])
    df["fecha"] = pd.to_datetime(df["fecha"].astype(str), format="%Y%m%d")
    df = df.set_index("fecha").astype(float) / 100.0
    return (1 + df).resample("ME").prod() - 1


def panel_mensual():
    px = {}
    for f in glob.glob("datos/mercado/1d/*.csv"):
        t = os.path.basename(f)[:-4]
        if "_eq_X" in t or "=" in t:
            continue
        d = pd.read_csv(f, parse_dates=["fecha"]).set_index("fecha")
        if "close" in d and len(d) > 400:
            px[t] = d["close"].astype(float)
    return pd.DataFrame(px).sort_index().ffill(limit=5).resample("ME").last().pct_change()


def backtest_resmom(retm, ff, win=36, skip=1, form=11, q=0.2, ls=True, coste_bps=10.0):
    """Residual momentum. En cada t: betas FF3 sobre [t-win+1, t]; residuos; señal =
    mean(resid formacion)/std. Posicion en t -> retorno en t+1. Sin look-ahead."""
    idx = retm.index.intersection(ff.index)
    retm, ff = retm.loc[idx], ff.loc[idx]
    X = np.column_stack([np.ones(len(ff)), ff["MktRF"], ff["SMB"], ff["HML"]])
    meses = list(retm.index)
    filas, w_prev = [], pd.Series(0.0, index=retm.columns)
    for i in range(win, len(meses) - 1):
        t1 = meses[i + 1]
        sl = slice(i - win + 1, i + 1)
        Xi = X[sl]
        sig = {}
        for tk in retm.columns:
            y = retm[tk].iloc[sl].values
            msk = ~np.isnan(y)
            if msk.sum() < win * 0.8:
                continue
            beta, *_ = np.linalg.lstsq(Xi[msk], y[msk], rcond=None)
            resid = y - Xi @ beta
            rr = resid[-(form + skip):-skip] if skip else resid[-form:]
            rr = rr[~np.isnan(rr)]
            if len(rr) >= form * 0.8 and rr.std() > 0:
                sig[tk] = rr.mean() / rr.std()
        sig = pd.Series(sig).dropna()
        if len(sig) < 20:
            continue
        n = max(int(len(sig) * q), 1)
        orden = sig.sort_values(ascending=False)
        w = pd.Series(0.0, index=retm.columns)
        w[orden.index[:n]] = 1.0 / n
        if ls:
            w[orden.index[-n:]] = -1.0 / n
        rfwd = retm.loc[t1].reindex(w.index).fillna(0.0)
        filas.append((t1, float((w * rfwd).sum()) - coste_bps / 1e4 * (w - w_prev).abs().sum()))
        w_prev = w
    return pd.Series(dict(filas))


def backtest_convmom(retm, form=11, skip=1, q=0.2, ls=True, coste_bps=10.0):
    """Momentum convencional (retorno total) en el mismo marco, como referencia."""
    meses = list(retm.index)
    log = np.log1p(retm)
    filas, w_prev = [], pd.Series(0.0, index=retm.columns)
    for i in range(form + skip, len(meses) - 1):
        t1 = meses[i + 1]
        sig = log.iloc[i - form - skip + 1:i - skip + 1].sum(min_count=int(form * 0.8)).dropna()
        if len(sig) < 20:
            continue
        n = max(int(len(sig) * q), 1)
        orden = sig.sort_values(ascending=False)
        w = pd.Series(0.0, index=retm.columns)
        w[orden.index[:n]] = 1.0 / n
        if ls:
            w[orden.index[-n:]] = -1.0 / n
        rfwd = retm.loc[t1].reindex(w.index).fillna(0.0)
        filas.append((t1, float((w * rfwd).sum()) - coste_bps / 1e4 * (w - w_prev).abs().sum()))
        w_prev = w
    return pd.Series(dict(filas))


def _perfil(s):
    eq = (1 + s).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    shp = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0.0
    return shp, s.std() * np.sqrt(12), dd, len(s)


def main():
    print("Descargando FF3 (Ken French) y construyendo paneles US locales...")
    ff, retm = ff3_mensual(), panel_mensual()
    print(f"  FF3 {ff.index.min().date()}..{ff.index.max().date()} | {retm.shape[1]} tickers\n")

    win, form, skip, q = ELEG
    rm_ls = backtest_resmom(retm, ff, win=win, form=form, skip=skip, q=q, ls=True)
    rm_lo = backtest_resmom(retm, ff, win=win, form=form, skip=skip, q=q, ls=False)
    cm_ls = backtest_convmom(retm, q=0.2, ls=True)
    cm_lo = backtest_convmom(retm, q=0.2, ls=False)

    print(f"  {'estrategia':<28}{'Sharpe':>8}{'vol':>8}{'maxDD':>9}  meses")
    for nombre, s in [("residual mom L/S", rm_ls), ("residual mom long-only*", rm_lo),
                      ("convencional L/S", cm_ls), ("convencional long-only*", cm_lo)]:
        shp, vol, dd, n = _perfil(s)
        print(f"  {nombre:<28}{shp:>8.2f}{vol*100:>7.1f}%{dd*100:>8.1f}%  {n}")
    print("  (*) long-only INFLADO por sesgo de supervivencia -> no fiable como alfa.")

    pares = pd.DataFrame({"rm": rm_ls, "cm": cm_ls}).dropna()
    j = pd.DataFrame({"rm": rm_ls}).join(ff).dropna()
    print(f"\n  corr residual<->convencional (L/S): {pares.corr().iloc[0,1]:.2f}  "
          f"(no es diversificador fuerte)")
    print(f"  neutralidad a factores (L/S): corr Mkt-RF {j['rm'].corr(j['MktRF']):+.2f} | "
          f"SMB {j['rm'].corr(j['SMB']):+.2f} | HML {j['rm'].corr(j['HML']):+.2f}  (~0 = OK)")

    # DSR / PBO sobre la rejilla L/S
    grid = list(itertools.product([24, 36, 48], [11], [1], [0.1, 0.2, 0.3]))
    series = {c: backtest_resmom(retm, ff, c[0], c[2], c[1], c[3], ls=True) for c in grid}
    M = pd.DataFrame({c: s for c, s in series.items() if len(s) >= 24}).dropna()
    sr_pp = [sharpe_no_anualizado(M[c].to_numpy()) for c in M.columns]
    dsr = deflated_sharpe_ratio(M[ELEG].to_numpy(), sr_pp)
    pbo = probabilidad_sobreajuste(M.to_numpy(), S=10)
    print(f"\n  DSR vs azar ({M.shape[1]} configs, {M.shape[0]} meses): {dsr:.3f}  (bar 0.95)")
    print(f"  PBO (CSCV): {pbo:.3f}  (bar <0.5)")

    pasa = dsr >= 0.95 and pbo < 0.5
    print(f"\n=== {'PASA' if pasa else 'NO PASA como libro propio'} ===")
    print("  Honesto: el metodo FUNCIONA (neutral a factores, mitad de drawdown que el")
    print("  convencional), pero en nuestro universo de 111 mega-caps con sesgo de")
    print("  supervivencia NO alcanza el bar y esta 0.60 correlado con el momentum del")
    print("  Libro B. NO se cablea. Se revisita con un universo amplio sin sesgo (vendor).")


if __name__ == "__main__":
    main()
