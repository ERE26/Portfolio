"""Validacion REPRODUCIBLE del Libro D (rotacion por paises europeos) con DSR y PBO.
Vive en el repo a proposito (reproducibilidad): corre `python validar_libroD.py` y
obtienes las mismas cifras que afirmamos. Honesto sobre sus limites (ver abajo).

Limites reconocidos (verificacion adversarial):
  - El DSR se deflacta tanto contra el AZAR (sr0 de 36 configs) como contra comprar-y-
    mantener Europa (VGK), porque con corr 0.56 a SPY parte del Sharpe es beta.
  - N=36 solo cuenta la rejilla; la eleccion del universo y los params fijos (SMA=200,
    mensual, equiponderado) son grados de libertad extra -> el DSR real es una COTA
    SUPERIOR. Por eso el libro es 'candidato en observacion', no 'alfa validada'.
  - Slippage: ETFs finos (EWO/EWK/EWD) tienen spreads amplios; el Sharpe neto real
    sera menor que el bruto de aqui.
"""
import itertools
import numpy as np, pandas as pd
from fuentes import yahoo
from fuentes.http import Cliente
from patrones.robustez import deflated_sharpe_ratio, probabilidad_sobreajuste, sharpe, sharpe_no_anualizado

PAISES = ["EWG", "EWQ", "EWP", "EWU", "EWI", "EWN", "EWL", "EWD", "EWO", "EWK"]
ELEGIDA = (3, 126, 21, True)            # pre-especificada (k, lookback, skip, filtro SMA200)


def _descargar(cli):
    px = {}
    for t in PAISES + ["VGK", "SPY"]:
        try:
            px[t] = yahoo.historico(t, intervalo="1d", desde="2010-01-01", cliente=cli)[0]["close"].astype(float)
        except Exception as e:
            print("  fallo", t, e)
    return px


def _serie(panel, reb, k, lb, skip, filtro):
    filas = []
    for i in range(len(reb) - 1):
        t0, t1 = reb[i], reb[i + 1]
        h = panel.loc[:t0]
        if len(h) < lb + skip + 1 or len(h) < 205:
            continue
        mom = h.iloc[-1 - skip] / h.iloc[-1 - skip - lb] - 1
        elig = mom[h.iloc[-1] > h.rolling(200).mean().iloc[-1]] if filtro else mom
        top = elig.sort_values(ascending=False).head(k)
        w = pd.Series(0.0, index=panel.columns)
        if len(top):
            w[top.index] = 1.0 / len(top)
        filas.append((t1, float((w * (panel.loc[t1] / panel.loc[t0] - 1)).sum())))
    return pd.Series(dict(filas))


def main():
    cli = Cliente()
    print("Descargando ETFs de pais + VGK + SPY...")
    px = _descargar(cli)
    panel = pd.DataFrame({t: px[t] for t in PAISES}).sort_index().ffill(limit=5).dropna()
    reb = [d for d in pd.Series(panel.index, index=panel.index).resample("ME").last().dropna() if d in set(panel.index)]
    spy_m = px["SPY"].reindex(panel.index).ffill().resample("ME").last().pct_change()
    vgk_m = px["VGK"].reindex(panel.index).ffill().resample("ME").last().pct_change()

    grid = list(itertools.product([2, 3, 4], [63, 126, 252], [0, 21], [True, False]))
    series = {c: _serie(panel, reb, *c) for c in grid}
    series = {c: s for c, s in series.items() if len(s) >= 24}
    M = pd.DataFrame(series).dropna()
    print(f"Configs validas: {M.shape[1]} | meses comunes: {M.shape[0]}")
    if ELEGIDA not in M.columns:
        raise SystemExit(f"ERROR: la config pre-especificada {ELEGIDA} no esta en la rejilla "
                         f"-> NO se sustituye por la mejor in-sample (evita cherry-picking).")

    sr_ann = {c: sharpe(M[c].to_numpy(), periodos=12) for c in M.columns}
    sr_pp = [sharpe_no_anualizado(M[c].to_numpy()) for c in M.columns]
    r = M[ELEGIDA]
    vgk = vgk_m.reindex(r.index).fillna(0.0)
    sr_vgk_pp = sharpe_no_anualizado(vgk.to_numpy())

    dsr_azar = deflated_sharpe_ratio(r.to_numpy(), sr_pp)                       # vs azar (36 configs)
    dsr_vgk = deflated_sharpe_ratio(r.to_numpy(), sr_pp, sr_benchmark=sr_vgk_pp)  # vs Europa pasiva

    pbo = probabilidad_sobreajuste(M.to_numpy(), S=10)
    eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    corr_spy, corr_vgk = r.corr(spy_m.reindex(r.index)), r.corr(vgk)

    print(f"\nConfig elegida {ELEGIDA} (pre-especificada):")
    print(f"  Sharpe anual {sr_ann[ELEGIDA]:.2f} | VGK {sharpe(vgk.to_numpy(),12):.2f} | CAGR {cagr*100:.1f}% | "
          f"vol {r.std()*np.sqrt(12)*100:.1f}% | maxDD {dd*100:.1f}%")
    print(f"  corr SPY {corr_spy:.2f} | corr VGK {corr_vgk:.2f} | meses {len(r)}")
    print(f"\n  DSR vs AZAR ({M.shape[1]} configs): {dsr_azar:.3f}")
    print(f"  DSR vs VGK (Europa pasiva):      {dsr_vgk:.3f}  <- el test HONESTO del edge")
    print(f"  PBO (CSCV): {pbo:.3f}")

    estandar = 0.95
    veredicto = ("PASA LIMPIO" if (dsr_vgk >= estandar and pbo < 0.5)
                 else "CANDIDATO MARGINAL" if (dsr_azar >= 0.90 and pbo < 0.5)
                 else "NO PASA")
    print(f"\n=== {veredicto} ===")
    print(f"  PBO {pbo:.2f} ({'sano' if pbo<0.5 else 'alto'}); DSR vs azar {dsr_azar:.2f}; "
          f"DSR vs VGK {dsr_vgk:.2f} (estandar {estandar}).")
    print("  -> Honesto: diversificador MODESTO (no alfa heroica). Paper=OK (riesgo 0, genera OOS); "
          "dinero real NO hasta DSR-vs-VGK>=0.95 + slippage real + track OOS.")


if __name__ == "__main__":
    main()
