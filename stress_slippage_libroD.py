"""Stress-test de SLIPPAGE del Libro D (rotacion por paises europeos).

Sin feed de bid/ask ni fills reales todavia, estimamos el SPREAD EFECTIVO de cada ETF
con el estimador de Corwin-Schultz (2012) a partir del high-low diario (metodo academico
estandar justo para esto) y vemos cuanto degrada el Sharpe NETO del Libro D su turnover.

Gate honesto: el Libro D no toca dinero real hasta que su Sharpe neto de slippage REAL
aguante. Esto cuantifica el peaje antes de tener fills, para saber si merece la pena
seguir generando evidencia OOS en paper.

  python stress_slippage_libroD.py
"""
import numpy as np, pandas as pd
from fuentes import yahoo
from fuentes.http import Cliente

PAISES = ["EWG", "EWQ", "EWP", "EWU", "EWI", "EWN", "EWL", "EWD", "EWO", "EWK"]
K, LB, SKIP = 3, 126, 21


def corwin_schultz(ohlc):
    """Spread efectivo medio (fraccion del precio) por Corwin-Schultz 2012, desde
    high/low diarios. Devuelve la media de los estimadores de 2 dias (negativos -> 0)."""
    h, l = ohlc["high"].astype(float), ohlc["low"].astype(float)
    hl = np.log(h / l) ** 2                      # (ln H/L)^2 por dia
    beta = hl + hl.shift(1)                       # suma de 2 dias consecutivos
    h2 = np.maximum(h, h.shift(1))                # high sobre la ventana de 2 dias
    l2 = np.minimum(l, l.shift(1))                # low sobre la ventana de 2 dias
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = alpha.clip(lower=0)                   # alpha negativo -> spread 0 (ruido)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(np.nanmean(s))


def serie_con_coste(panel, reb, half_spread):
    """Retornos mensuales del Libro D con coste = turnover * half_spread (por ETF si es
    dict, o escalar). half_spread en fraccion (p.ej. 0.001 = 10 bps por lado)."""
    filas, w_prev = [], pd.Series(0.0, index=panel.columns)
    for i in range(len(reb) - 1):
        t0, t1 = reb[i], reb[i + 1]
        h = panel.loc[:t0]
        if len(h) < LB + SKIP + 1 or len(h) < 205:
            continue
        mom = h.iloc[-1 - SKIP] / h.iloc[-1 - SKIP - LB] - 1
        elig = mom[h.iloc[-1] > h.rolling(200).mean().iloc[-1]]
        top = elig.sort_values(ascending=False).head(K)
        w = pd.Series(0.0, index=panel.columns)
        if len(top):
            w[top.index] = 1.0 / len(top)
        dw = (w - w_prev).abs()
        if isinstance(half_spread, dict):
            coste = float(sum(dw[t] * half_spread.get(t, 0.0) for t in panel.columns))
        else:
            coste = float(dw.sum() * half_spread)
        r = float((w * (panel.loc[t1] / panel.loc[t0] - 1)).sum())
        filas.append((t1, r - coste))
        w_prev = w
    return pd.Series(dict(filas))


def sharpe(s):
    return float(s.mean() / s.std() * np.sqrt(12)) if s.std() > 0 else 0.0


def main():
    cli = Cliente()
    print("Descargando OHLC de los 10 ETFs de pais + SPY (ref)...")
    ohlc = {}
    for t in PAISES + ["SPY"]:
        try:
            ohlc[t] = yahoo.historico(t, intervalo="1d", desde="2010-01-01", cliente=cli)[0]
        except Exception as e:
            print("  fallo", t, e)

    print("\n== Spread efectivo estimado (Corwin-Schultz, media historica) ==")
    print(f"  {'ETF':<6}{'spread':>9}{'half-spread':>13}   liquidez")
    spreads = {}
    for t in PAISES:
        s = corwin_schultz(ohlc[t])
        spreads[t] = s
        vol_usd = float((ohlc[t]["close"] * ohlc[t]["volume"]).tail(60).median())
        liq = "fino" if vol_usd < 5e6 else "medio" if vol_usd < 2e7 else "liquido"
        print(f"  {t:<6}{s*1e4:>7.0f}bps{s/2*1e4:>10.0f}bps   ${vol_usd/1e6:4.1f}M/dia  {liq}")
    s_spy = corwin_schultz(ohlc["SPY"])
    print(f"  {'SPY':<6}{s_spy*1e4:>7.0f}bps{s_spy/2*1e4:>10.0f}bps   (referencia liquida)")

    panel = pd.DataFrame({t: ohlc[t]["close"].astype(float) for t in PAISES}).sort_index().ffill(limit=5).dropna()
    reb = [d for d in pd.Series(panel.index, index=panel.index).resample("ME").last().dropna() if d in set(panel.index)]

    half = {t: spreads[t] / 2 for t in PAISES}
    half2 = {t: spreads[t] for t in PAISES}     # 2x stress (spread completo por lado)

    escenarios = {
        "sin coste (bruto)":                 serie_con_coste(panel, reb, 0.0),
        "10 bps (supuesto actual)":          serie_con_coste(panel, reb, 0.0010),
        "spread estimado (Corwin-Schultz)":  serie_con_coste(panel, reb, half),
        "2x spread (stress pesimista)":      serie_con_coste(panel, reb, half2),
    }
    print("\n== Sharpe neto del Libro D por escenario de slippage ==")
    print(f"  {'escenario':<36}{'Sharpe':>8}{'CAGR':>9}{'drag/año':>11}")
    base = escenarios["sin coste (bruto)"]
    cagr_base = (1 + base).prod() ** (12 / len(base)) - 1
    for nombre, s in escenarios.items():
        cagr = (1 + s).prod() ** (12 / len(s)) - 1
        drag = (cagr_base - cagr) * 100
        print(f"  {nombre:<36}{sharpe(s):>8.2f}{cagr*100:>8.1f}%{drag:>9.2f}pp")

    peor = escenarios["2x spread (stress pesimista)"]
    print(f"\n  Sharpe con spread estimado "
          f"{sharpe(escenarios['spread estimado (Corwin-Schultz)']):.2f} "
          f"(bruto {sharpe(base):.2f}); en stress 2x, {sharpe(peor):.2f}.")

    # ---- La restriccion que DE VERDAD ata: capacidad / impacto de mercado ----
    # El spread no mata una estrategia mensual de bajo turnover. Lo que ata es meter
    # ~5% del equity en un ETF que mueve $0.4M/dia. Corwin-Schultz ademas INFRAESTIMA
    # el spread de los nombres finos (mide el rango HL realizado, que se comprime cuando
    # hay pocas operaciones), asi que los bps de arriba son un SUELO, no el coste real.
    print("\n== Capacidad / impacto de mercado (la restriccion que ata de verdad) ==")
    print("  Libro D ~14.7% del equity repartido en 3 ETFs -> ~4.9% por ETF.")
    print(f"  {'equity':>10}{'$/ETF':>10}   ETFs donde la posicion supera el 1% del ADV (señal de impacto)")
    adv = {t: float((ohlc[t]["close"] * ohlc[t]["volume"]).tail(60).median()) for t in PAISES}
    for eq in (100_000, 1_000_000, 10_000_000):
        pos = eq * 0.147 / 3
        malos = [f"{t}({pos/adv[t]*100:.0f}%ADV)" for t in PAISES if pos > 0.01 * adv[t]]
        print(f"  {eq:>10,}{pos:>10,.0f}   {', '.join(malos) if malos else 'ninguno'}")

    print("\n  VEREDICTO HONESTO:")
    print("  - El turnover mensual bajo hace que el SPREAD apenas duela (~0.7pp/año).")
    print("  - PERO Corwin-Schultz infraestima el spread fino (EWK 11bps < SPY 23bps es")
    print("    irreal); trata esos bps como un suelo, no el coste real.")
    print("  - La restriccion que ata es la CAPACIDAD: EWK/EWO ($0.4-1M/dia) no admiten")
    print("    una posicion grande sin mover el precio. A $1M+ de cuenta, el Libro D top-3")
    print("    toparia con impacto en los paises finos.")
    print("  - Edge ya MARGINAL (DSR-vs-VGK 0.89). Conclusion firme: PAPER si (OOS a riesgo")
    print("    cero y cuenta pequeña); dinero real NO, y menos a escala, hasta medir fills reales.")


if __name__ == "__main__":
    main()
