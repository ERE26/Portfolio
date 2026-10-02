"""Informe de rendimiento institucional (Tier 2): tear-sheet + slippage de ejecucion
+ monitor de decaimiento. Hora de Madrid.

  python analitica.py            # informe
  python analitica.py --fig      # ademas genera figuras/31_tearsheet.png

Convierte la cuenta en algo auditable: metricas profesionales, si la ejecucion real
casa con lo que asume el backtest (slippage), y si el edge sigue vivo (decaimiento).
"""
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca_trade as AT
from patrones import analitica as A

# Expectativas del backtest (walk-forward validado; ver README/docs). Referencia para
# el monitor de decaimiento. El equity de la cuenta es un BLEND de 3 libros + mucho
# efectivo, asi que se compara con prudencia.
BT_SHARPE = 1.2
BT_HIT_INTRADIA = 0.60


def _madrid():
    try:
        return pd.Timestamp.now(tz="Europe/Madrid").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def _fmt(x, pct=False, suf=""):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  n/d"
    return (f"{x:+.2%}" if pct else f"{x:+.2f}") + suf


def _equity_cuenta(period="6M", timeframe="1D"):
    try:
        h = AT.historia_cuenta(period=period, timeframe=timeframe)
        return h["equity"] if len(h) else pd.Series(dtype=float)
    except Exception as e:
        print(f"[equity] no disponible: {str(e)[:60]}")
        return pd.Series(dtype=float)


def _spy(desde, hasta):
    try:
        from fuentes import yahoo
        df, _ = yahoo.historico("SPY", intervalo="1d", desde=str(desde), hasta=str(hasta))
        return df["close"].astype(float)
    except Exception:
        return pd.Series(dtype=float)


def _log_fills():
    import os
    from registro import LOG, _fills
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    try:
        fills = _fills()
    except Exception:
        fills = []
    return log, fills


def main(fig=False):
    print("=" * 66)
    print(f"RENDIMIENTO (TEAR-SHEET + EJECUCION)  ·  {_madrid()} (Madrid)")
    print("=" * 66)

    eq = _equity_cuenta()
    met = A.metricas(eq) if len(eq) else {"n": 0, "insuficiente": True}

    # 1) tear-sheet
    print("\n-- TEAR-SHEET (curva de equity de la cuenta) --")
    if met.get("insuficiente"):
        print(f"  Muestra insuficiente ({met.get('n',0)} dias). Las metricas se vuelven "
              f"fiables con mas historial; el tear-sheet ya esta cableado.")
    else:
        print(f"  dias {met['n']} | CAGR {_fmt(met['cagr'],pct=True)} | vol {_fmt(met['vol_anual'],pct=True)} | "
              f"ret total {_fmt(met['ret_total'],pct=True)}")
        print(f"  Sharpe {_fmt(met['sharpe'])} | Sortino {_fmt(met['sortino'])} | "
              f"Calmar {_fmt(met['calmar'])} | maxDD {_fmt(met['max_drawdown'],pct=True)}")
        print(f"  hit-rate diario {_fmt(met['hit_rate'],pct=True)} | mejor {_fmt(met['mejor_dia'],pct=True)} | "
              f"peor {_fmt(met['peor_dia'],pct=True)}")
        bench = _spy(eq.index[0].date(), eq.index[-1].date()) if len(eq) else pd.Series(dtype=float)
        vb = A.vs_benchmark(eq, bench) if len(bench) else {"insuficiente": True}
        if not vb.get("insuficiente"):
            print(f"  vs SPY: beta {_fmt(vb['beta'])} | alfa anual {_fmt(vb['alpha_anual'],pct=True)} | "
                  f"corr {_fmt(vb['correlacion'])} | info ratio {_fmt(vb['information_ratio'])}")

    # 2) slippage de ejecucion
    log, fills = _log_fills()
    sl, sl_df = A.slippage(log, fills)
    print("\n-- EJECUCION / SLIPPAGE (entrada esperada vs fill real) --")
    if sl["n"] == 0:
        print("  Sin entradas emparejables con fills todavia.")
    else:
        print(f"  {sl['n']} entradas | medio {sl['medio_bps']:+.1f} bps | mediana {sl['mediana_bps']:+.1f} bps | "
              f"p90 {sl['p90_bps']:+.1f} bps | peor {sl['peor_bps']:+.1f} bps")
        signo = "coste real (se paga de mas)" if sl['medio_bps'] > 0 else "a favor"
        print(f"  -> slippage medio {sl['coste_medio_pct']:+.3f}% por entrada ({signo}).")
        print(f"     Modelo de costes del backtest: ~spread+comision; contrastar que cubra esto.")

    # 3) estadisticas de operaciones (log intradia)
    st = A.stats_trades(log)
    print("\n-- OPERACIONES CERRADAS (log intradia) --")
    if st["n"] == 0:
        print(f"  Sin operaciones cerradas aun (abiertas: {st.get('abiertas',0)}).")
    else:
        print(f"  {st['n']} cerradas | hit-rate {st['hit_rate']:.0%} | PF {st['profit_factor']:.2f} | "
              f"esperanza ${st['esperanza']:+.1f} | P&L ${st['pnl_total']:+,.0f}")
        print(f"  ganancia media ${st['ganancia_media']:+.1f} | perdida media ${st['perdida_media']:+.1f}")

    # 4) monitor de decaimiento
    dec = A.decaimiento(met.get("sharpe", float("nan")), BT_SHARPE,
                        hit_vivo=st.get("hit_rate"), hit_bt=BT_HIT_INTRADIA,
                        n=met.get("n", 0))
    print("\n-- MONITOR DE DECAIMIENTO (vivo vs backtest) --")
    print(f"  estado: {dec['estado']}")
    if dec["estado"] == "MUESTRA_INSUFICIENTE":
        print(f"  {dec['nota']} (Sharpe backtest ref. {BT_SHARPE}).")
    else:
        print(f"  Sharpe vivo {_fmt(dec['sharpe_vivo'])} vs backtest {BT_SHARPE} "
              f"(ratio {_fmt(dec['ratio_sharpe'])}); hit vivo {_fmt(dec.get('hit_vivo'),pct=True)} "
              f"vs {BT_HIT_INTRADIA:.0%}")

    if fig and not met.get("insuficiente"):
        _figura(eq)
    print("=" * 66)


def _figura(eq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bench = _spy(eq.index[0].date(), eq.index[-1].date())
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2, 1, 1]})
    e = eq / eq.iloc[0]
    ax[0].plot(eq.index, e, color="#2563eb", lw=1.8, label="Cuenta")
    if len(bench):
        b = bench.reindex(eq.index).ffill()
        ax[0].plot(eq.index, b / b.iloc[0], color="#94a3b8", lw=1.4, label="SPY")
    ax[0].legend(frameon=False); ax[0].set_title("Equity normalizada vs SPY", fontweight="bold")
    r = eq.pct_change().dropna()
    ec = (1 + r).cumprod()
    dd = ec / ec.cummax() - 1
    ax[1].fill_between(dd.index, dd.values * 100, 0, color="#dc2626", alpha=0.5)
    ax[1].set_title("Drawdown (%)", fontweight="bold")
    rs = A.rolling_sharpe(eq, ventana=min(63, max(5, len(r) // 2)))
    if len(rs):
        ax[2].plot(rs.index, rs.values, color="#16a34a", lw=1.4)
        ax[2].axhline(0, color="#1e293b", lw=0.8)
    ax[2].set_title("Sharpe rolling", fontweight="bold")
    plt.tight_layout()
    out = "figuras/31_tearsheet.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"\n[figura guardada: {out}]")


if __name__ == "__main__":
    main(fig="--fig" in sys.argv)
