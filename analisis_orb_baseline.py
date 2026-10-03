"""Caracterización del edge del ORB (baseline) sobre datos/orb_dataset.csv.

OJO: el dataset está SESGADO POR SUPERVIVENCIA (universo líquido ACTUAL vía Alpaca). Esto
es el BASELINE OPTIMISTA contra el que se comparará el número limpio (universo point-in-time
de Massive, con deslistados). Dos análisis:
  1) ¿DÓNDE vive el edge? (confirmación, RVOL, timing, régimen, amplitud, Hurst) con IC95
     bootstrap y un chequeo OOS temporal 70/30.
  2) ¿Es REAL o lo sostienen 4 outliers? (concentración por ticker/día, quitar top-k
     ganadores, media recortada). Proxy de fragilidad y de la duda de supervivencia.

  python analisis_orb_baseline.py
"""
import numpy as np
import pandas as pd

DATASET = "datos/orb_dataset.csv"


def boot(x, n=5000, seed=1):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fila(nombre, x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 5:
        return f"  {nombre:30s} n={len(x):5d}  (muestra insuficiente)"
    lo, hi = boot(x)
    verd = "ROBUSTO+" if lo > 0 else ("neg" if hi < 0 else "incierto")
    return (f"  {nombre:30s} n={len(x):5d}  WR={(x > 0).mean() * 100:4.0f}%  "
            f"R={x.mean():+.3f}  med={np.median(x):+.3f}  IC95[{lo:+.3f},{hi:+.3f}]  {verd}")


def por_buckets(df, col, bins, labels):
    print(f"\n### por {col}")
    cats = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    for lab in labels:
        print(fila(f"{col}={lab}", df.loc[cats == lab, "R"]))


def donde_vive(df):
    R = df["R"].to_numpy(float)
    print(f"=== DATASET: {len(df)} breakouts | {df['ticker'].nunique()} tickers | "
          f"{df['fecha'].min().date()}..{df['fecha'].max().date()} (SESGADO supervivencia) ===")
    print("\n### GLOBAL (entrada al toque del máximo)")
    print(fila("TODOS", R))
    print("\n### CONFIRMACIÓN (la vela que rompe CIERRA sobre el máximo = ext_breakout>0)")
    print(fila("confirmada (ext>0)", df.loc[df.ext_breakout > 0, "R"]))
    print(fila("falsa/mecha (ext<=0)", df.loc[df.ext_breakout <= 0, "R"]))
    por_buckets(df, "ext_breakout", [-10, 0, 0.1, 0.3, 0.6, 100], ["<=0", "0-0.1", "0.1-0.3", "0.3-0.6", ">0.6"])
    por_buckets(df, "opening_rvol", [0, 1.5, 2.5, 4, 1e9], ["1-1.5", "1.5-2.5", "2.5-4", ">4"])
    por_buckets(df, "breakout_min", [0, 15, 30, 60, 1e9], ["<=15", "15-30", "30-60", ">60"])
    print("\n### RÉGIMEN (SPY sobre su MA50 de 5m)")
    print(fila("SPY trend (1)", df.loc[df.spy_trend == 1, "R"]))
    print(fila("SPY no-trend (0)", df.loc[df.spy_trend == 0, "R"]))
    por_buckets(df, "or_range_atr", [0, 0.5, 1, 2, 1e9], ["<=0.5(ARTEFACTO)", "0.5-1", "1-2", ">2"])
    por_buckets(df, "hurst", [-5, 0.45, 0.55, 5], ["<0.45", "0.45-0.55", ">0.55"])

    print("\n=== CANDIDATO OPERABLE: ruptura CONFIRMADA ===")
    conf = df[df.ext_breakout > 0].sort_values("fecha")
    print(fila("confirmada", conf["R"]))
    print(fila("confirmada + rvol>2", conf.loc[conf.opening_rvol > 2, "R"]))
    print(fila("confirmada + <=30min", conf.loc[conf.breakout_min <= 30, "R"]))
    print(fila("confirmada + rvol>2 + <=30min", conf.loc[(conf.opening_rvol > 2) & (conf.breakout_min <= 30), "R"]))
    print("\n### OOS temporal (split 70/30 por fecha) sobre la confirmada")
    corte = conf["fecha"].quantile(0.7)
    print(f"  corte = {pd.Timestamp(corte).date()}")
    print(fila("IN (primeros 70%)", conf.loc[conf.fecha <= corte, "R"]))
    print(fila("OUT (últimos 30%)", conf.loc[conf.fecha > corte, "R"]))


def fragilidad(df):
    conf = df[df.ext_breakout > 0].copy()
    R = conf["R"].to_numpy(float)
    print(f"\n\n=== FRAGILIDAD del ORB confirmado ({len(R)} trades, R {R.mean():+.3f}) ===")
    print("### Quitar los mayores ganadores (¿sobrevive sin outliers?)")
    Rs = np.sort(R)[::-1]
    for k in (0, 5, 10, 25, 50, 100):
        rec = Rs[k:]
        print(f"  sin top-{k:3d}:  R {rec.mean():+.3f}  | aporte top-{k}: "
              f"{(R.sum() - rec.sum()) / R.sum() * 100 if R.sum() else 0:4.0f}% del PnL")
    print("\n### Concentración por ticker")
    g = conf.groupby("ticker")["R"]; pnl = g.sum().sort_values(ascending=False)
    print(f"  tickers rentables: {(g.mean() > 0).sum()}/{g.ngroups} ({(g.mean() > 0).mean() * 100:.0f}%)")
    print(f"  top-10 tickers = {pnl.head(10).sum() / pnl.sum() * 100:.0f}% del PnL | "
          f"R SIN top-10 tickers: {conf[~conf.ticker.isin(pnl.head(10).index)]['R'].mean():+.3f}")
    print("\n### Concentración por día")
    pnl_d = conf.groupby(conf.fecha.dt.date)["R"].sum().sort_values(ascending=False)
    print(f"  días +: {(pnl_d > 0).mean() * 100:.0f}% | top-10 días = {pnl_d.head(10).sum() / pnl_d.sum() * 100:.0f}% del PnL")
    print("\n### Media recortada (robusta a colas) -> revela edge de COLA")
    for p in (0, 1, 5, 10):
        lo, hi = np.percentile(R, p), np.percentile(R, 100 - p)
        rec = R[(R >= lo) & (R <= hi)]
        print(f"  recorte {p:2d}%:  R {rec.mean():+.3f}")


if __name__ == "__main__":
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    donde_vive(df)
    fragilidad(df)
    print("\nNOTA: todo con SESGO de supervivencia (optimista). El número honesto saldrá al")
    print("repetir sobre el universo point-in-time de Massive (deslistados incluidos).")
