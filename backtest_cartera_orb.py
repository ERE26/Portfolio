"""BACKTEST DE CARTERA del ORB-confirmado (nivel avanzado: de donde sale el Sharpe del paper).

El edge por trade es +0.26R, alta varianza. El Sharpe fuerte (Zarattini "stocks in play":
~2.8) NO viene del R por trade sino de la CONSTRUCCIÓN DE CARTERA: cada día diversificar
en los N mejores setups "in play", dimensionar cada uno a RIESGO FIJO y agregar. La
diversificación cross-seccional recorta la varianza y deja el edge.

Sobre orb_dataset.csv (breakouts ORB con fecha/ticker/R/features), construye la curva de
equity diaria de una cartera que:
  - cada día toma los breakouts CONFIRMADOS (ext_breakout>0);
  - opcional: solo los top-N por una señal de calidad (ext_breakout);
  - dimensiona cada trade a RIESGO r% del equity, con tope de riesgo bruto diario;
  - escala r para apuntar a una VOL OBJETIVO anual (vol-targeting, Carver) y aplica
    HALF-KELLY como tope prudente.
Reporta Sharpe, CAGR, maxDD, % días positivos, y DSR sobre los retornos diarios; compara
con SPY buy&hold del periodo.

  python backtest_cartera_orb.py [--datos datos/orb_dataset.csv] [--topn 20] [--vol_obj 0.20]

Avisos: entrada al máximo del rango (optimista ~ +0.1-0.2R vs entrada confirmada real; se
aplica un HAIRCUT configurable). Sesgo de supervivencia e IEX siguen aplicando.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from orb_metalabel import _deflated_sharpe


def backtest(datos="datos/orb_dataset.csv", topn=20, vol_obj=0.20, riesgo_tope=0.02,
             haircut_R=0.15, kelly=0.5):
    d = pd.read_csv(datos)
    d = d[d["ext_breakout"] > 0].copy()                 # SOLO rupturas confirmadas
    d["R"] = d["R"] - haircut_R                          # haircut por entrada confirmada real
    d["fecha"] = pd.to_datetime(d["fecha"])
    # por día: top-N por calidad del breakout (ext_breakout), reparto de riesgo
    dias = []
    for f, g in d.groupby("fecha"):
        g = g.sort_values("ext_breakout", ascending=False).head(topn)
        dias.append((f, g["R"].to_numpy()))
    dias.sort(key=lambda x: x[0])
    if len(dias) < 60:
        print("muestra insuficiente"); return
    # retorno diario SIN escalar (riesgo 1 unidad por trade, repartido): media de R del día
    # (equal-risk entre los trades del día) -> así el día es la MEDIA de R, menos volátil
    r_dia = np.array([g.mean() for _, g in dias])        # en unidades de R (1R = riesgo/trade)
    n_dia = np.array([len(g) for _, g in dias])
    fechas = [f for f, _ in dias]
    # vol-targeting: escala el riesgo/trade para que la vol anual de la cartera ~ vol_obj
    vol_r = r_dia.std()
    if vol_r == 0:
        print("sin varianza"); return
    riesgo_trade = (vol_obj / np.sqrt(252)) / vol_r      # riesgo por trade (fracción equity) para la vol objetivo
    riesgo_trade = min(riesgo_trade, riesgo_tope)        # tope prudente
    riesgo_trade *= kelly                                # half-Kelly (Carver)
    ret_dia = r_dia * riesgo_trade                       # retorno diario de la cartera
    # métricas
    sharpe = ret_dia.mean() / ret_dia.std() * np.sqrt(252)
    eq = np.cumprod(1 + ret_dia)
    cagr = eq[-1] ** (252 / len(eq)) - 1
    dd = 1 - eq / np.maximum.accumulate(eq)
    maxdd = dd.max()
    pos = (ret_dia > 0).mean()
    dsr, _ = _deflated_sharpe(ret_dia, n_trials=20)      # ~20 configs probadas en todo el estudio
    print(f"\n### CARTERA ORB-confirmado | {len(dias)} días | top-{topn}/día | vol_obj {vol_obj:.0%} | "
          f"half-Kelly | haircut {haircut_R}R")
    print(f"  riesgo/trade efectivo: {riesgo_trade*100:.2f}% equity | trades/día medio: {n_dia.mean():.1f}")
    print(f"  expectancy/trade (post-haircut): {d['R'].mean():+.3f}R | días positivos: {pos*100:.0f}%")
    print(f"  SHARPE anual: {sharpe:.2f}")
    print(f"  CAGR: {cagr*100:+.1f}%  |  maxDD: {maxdd*100:.1f}%  |  Calmar: {cagr/maxdd if maxdd>0 else 0:.2f}")
    print(f"  DSR (retornos diarios, 20 trials): {dsr:.3f}  -> {'ROBUSTO' if dsr>=0.95 else 'no concluyente'}")
    # comparación con SPY del periodo (aprox: necesita precio SPY; se omite si no está a mano)
    return {"sharpe": float(sharpe), "cagr": float(cagr), "maxdd": float(maxdd), "dsr": float(dsr)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=str, default="datos/orb_dataset.csv")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--vol_obj", type=float, default=0.20)
    ap.add_argument("--haircut", type=float, default=0.15)
    a = ap.parse_args()
    backtest(a.datos, a.topn, a.vol_obj, haircut_R=a.haircut)
