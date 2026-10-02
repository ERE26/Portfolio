"""Presupuesto de riesgo ENTRE libros (el 'SO del pod-shop'): dado el historico de
retornos de cada libro, reparte RIESGO por vol + correlacion (HRP), no a ojo.

Series mensuales backtesteadas de los 3 libros de horizonte largo:
  B swing   = momentum transversal US (patrones.transversal)
  C TSMOM   = time-series momentum cross-asset (patrones.tsmom)
  D rotacion= rotacion por paises Europa (config pre-especificada del Libro D)

El intradia (Libro A) NO entra en el HRP: no tiene serie mensual limpia (horizonte
intradia, cash-heavy); se trata como un sleeve fijo y pequeno, aparte.

Honesto: esto es un PRIOR de backtest. Cuando acumule historia VIVA por libro
(ya atribuida via client_order_id), el mismo motor se re-alimenta con retornos reales.
"""
import glob, os
import numpy as np, pandas as pd

from patrones import transversal, tsmom, riesgo
from fuentes import yahoo
from fuentes.http import Cliente

CESTA_TSMOM = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "HYG"]
PAISES = ["EWG", "EWQ", "EWP", "EWU", "EWI", "EWN", "EWL", "EWD", "EWO", "EWK"]
K, LB_D, SKIP_D = 3, 126, 21


def _local_panel():
    """Precios diarios de los ~124 tickers US locales (para el libro swing)."""
    precios = {}
    for f in glob.glob("datos/mercado/1d/*.csv"):
        t = os.path.basename(f)[:-4]
        if "_eq_X" in t or "=" in t:   # fuera FX
            continue
        try:
            df = pd.read_csv(f, parse_dates=["fecha"]).set_index("fecha")
            if "close" in df and len(df) > 300:
                precios[t] = df[["close"]]
        except Exception:
            continue
    return precios


def _yahoo_panel(tickers, cli, desde="2008-01-01"):
    px = {}
    for t in tickers:
        try:
            s = yahoo.historico(t, intervalo="1d", desde=desde, cliente=cli)[0]["close"].astype(float)
            px[t] = pd.DataFrame({"close": s})
        except Exception as e:
            print("  fallo", t, e)
    return px


def _serie_rotacion(panel):
    """Retornos mensuales de la config pre-especificada del Libro D."""
    reb = [d for d in pd.Series(panel.index, index=panel.index).resample("ME").last().dropna()
           if d in set(panel.index)]
    filas = []
    for i in range(len(reb) - 1):
        t0, t1 = reb[i], reb[i + 1]
        h = panel.loc[:t0]
        if len(h) < LB_D + SKIP_D + 1 or len(h) < 205:
            continue
        mom = h.iloc[-1 - SKIP_D] / h.iloc[-1 - SKIP_D - LB_D] - 1
        elig = mom[h.iloc[-1] > h.rolling(200).mean().iloc[-1]]
        top = elig.sort_values(ascending=False).head(K)
        w = pd.Series(0.0, index=panel.columns)
        if len(top):
            w[top.index] = 1.0 / len(top)
        filas.append((t1, float((w * (panel.loc[t1] / panel.loc[t0] - 1)).sum())))
    return pd.Series(dict(filas))


def _norm_mensual(s):
    """A indice de fin-de-mes natural, para alinear libros con calendarios distintos."""
    s = pd.Series(s).dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    return s[~s.index.duplicated(keep="last")]


def main():
    cli = Cliente()
    print("== Construyendo series mensuales por libro ==")

    # --- Libro B: momentum transversal US (local) ---
    precios_us = _local_panel()
    dB, _, _ = transversal.backtest(precios_us, lookback=252, skip=21, q=0.2,
                                    largo_corto=True, coste_bps=10.0, vol_objetivo=0.10)
    sB = _norm_mensual(dB["ret"]) if len(dB) else pd.Series(dtype=float)
    print(f"  B swing (momentum US): {len(sB)} meses, {len(precios_us)} tickers")

    # --- Libro C: TSMOM cross-asset ---
    px_c = _yahoo_panel(CESTA_TSMOM, cli)
    sC = _norm_mensual(tsmom.serie_para_combinar(px_c))
    print(f"  C TSMOM (cross-asset): {len(sC)} meses")

    # --- Libro D: rotacion Europa ---
    px_d = _yahoo_panel(PAISES, cli)
    panel_d = pd.DataFrame({t: px_d[t]["close"] for t in px_d}).sort_index().ffill(limit=5).dropna()
    sD = _norm_mensual(_serie_rotacion(panel_d))
    print(f"  D rotacion (Europa):   {len(sD)} meses")

    libros = {"B_swing": sB, "C_tsmom": sC, "D_rotacion": sD}
    libros = {k: v for k, v in libros.items() if len(v) >= 24}

    # ---- Estadistica individual ----
    print("\n== Perfil individual (anualizado) ==")
    print(f"  {'libro':<12}{'Sharpe':>8}{'vol':>8}{'CAGR':>8}{'maxDD':>8}  meses")
    for k, s in libros.items():
        ann = np.sqrt(12)
        shp = s.mean() / s.std() * ann if s.std() > 0 else 0.0
        eq = (1 + s).cumprod(); dd = (eq / eq.cummax() - 1).min()
        cagr = eq.iloc[-1] ** (12 / len(s)) - 1
        print(f"  {k:<12}{shp:>8.2f}{s.std()*ann*100:>7.1f}%{cagr*100:>7.1f}%{dd*100:>7.1f}%  {len(s)}")

    # ---- Correlacion (meses comunes) ----
    M = pd.DataFrame(libros).dropna()
    print(f"\n== Matriz de correlacion mensual ({len(M)} meses comunes) ==")
    print(M.corr().round(2).to_string())

    # ---- Presupuesto de riesgo: inv-vol vs HRP ----
    w_iv, nota_iv = riesgo.presupuesto_riesgo(libros, metodo="inv_vol", min_obs=1_000_000)
    w_hrp, nota_hrp = riesgo.presupuesto_riesgo(libros, metodo="hrp", min_obs=24)
    print("\n== Presupuesto de RIESGO entre libros ==")
    print(f"  inv-vol  ({nota_iv}):")
    for k, v in sorted(w_iv.items(), key=lambda x: -x[1]):
        print(f"     {k:<12} {v*100:5.1f}%")
    print(f"  HRP      ({nota_hrp}):")
    for k, v in sorted(w_hrp.items(), key=lambda x: -x[1]):
        print(f"     {k:<12} {v*100:5.1f}%")

    # ---- Diversificacion: Sharpe de la cartera combinada con pesos HRP ----
    w = pd.Series(w_hrp).reindex(M.columns).fillna(0.0)
    comb = (M * w).sum(axis=1)
    ann = np.sqrt(12)
    shp_comb = comb.mean() / comb.std() * ann if comb.std() > 0 else 0.0
    eqc = (1 + comb).cumprod(); ddc = (eqc / eqc.cummax() - 1).min()
    # diversification ratio = media ponderada de vols / vol de la cartera
    vol_w = (w * M.std()).sum()
    dr = vol_w / comb.std() if comb.std() > 0 else np.nan
    best_solo = max((M[c].mean() / M[c].std() * ann) for c in M.columns)
    print("\n== Cartera multi-libro (pesos HRP, meses comunes) ==")
    print(f"  Sharpe combinado {shp_comb:.2f}  (mejor libro en solitario {best_solo:.2f})")
    print(f"  vol {comb.std()*ann*100:.1f}%  maxDD {ddc*100:.1f}%  ratio diversificacion {dr:.2f}")
    print(f"  -> {'la combinacion MEJORA el Sharpe (free lunch de correlacion)' if shp_comb > best_solo + 0.02 else 'la combinacion no bate al mejor libro'}")

    # ---- Traduccion a capital (sleeve intradia fijo) ----
    intraday_sleeve = 0.35   # Libro A: sleeve fijo, fuera del HRP (horizonte intradia)
    resto = 1 - intraday_sleeve
    print(f"\n== Traduccion a CAPITAL (sugerida) ==")
    print(f"  A_intradia (sleeve fijo, fuera de HRP): {intraday_sleeve*100:.0f}%")
    for k, v in sorted(w_hrp.items(), key=lambda x: -x[1]):
        print(f"  {k:<28} {v*resto*100:5.1f}%")
    print("\n  (prior de BACKTEST; se re-alimenta con retornos VIVOS por libro conforme acumulen)")


if __name__ == "__main__":
    main()
