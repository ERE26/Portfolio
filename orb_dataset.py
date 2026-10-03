"""Construye la BASE DE DATOS de meta-labeling del ORB (López de Prado): recorre el universo
de mid-caps líquidas, encuentra CADA breakout ORB de 5 min (señal primaria = largo) y para
cada uno guarda FEATURES point-in-time + la ETIQUETA triple-barrera (R real del trade con
stop al mínimo del rango y salida al cierre). Sobre esta tabla se entrena luego un meta-modelo
(orb_metalabel.py) que decide QUÉ breakouts tomar, validado con purged walk-forward + DSR/PBO.

Features (todas conocidas en el momento del breakout, sin look-ahead):
  opening_rvol  vol 1ª vela 5m / media 14d (stock in play)
  gap           hueco de apertura (open vs cierre previo)
  or_range_atr  rango de apertura / ATR (amplitud normalizada por volatilidad)
  or_range_pct  rango de apertura / precio
  breakout_min  minutos desde la apertura hasta el breakout (antes = mejor señal)
  ext_breakout  cuánto cerró por encima del máximo del rango (/ATR)
  day_ret       retorno de sesión hasta el breakout
  atr_pct       ATR% del valor (volatilidad propia)
  hurst         exponente de Hurst reciente (>0.5 tendencial, <0.5 mean-reverting)
  spy_day_ret   retorno de sesión del mercado (SPY) hasta ese momento
  spy_trend     SPY por encima (1) o por debajo (0) de su media de 50 velas de 5m (régimen)
  dow           día de la semana (0-4)
Etiqueta:
  R             R-multiple real del trade ORB (stop OR-low, salida EOD)  -> para el backtest
  y             1 si R>0 (trade ganador)                                 -> target de clasificación
  fecha         día del trade (para purged walk-forward CV)

  python orb_dataset.py [--tickers 150] [--meses 24] [--out datos/orb_dataset.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd

import backtest_intradia as bi
from backtest_orb5 import _opening_rvol
from fuentes import alpaca
from patrones import indicadores as ind

SLIP = 0.0004
SLIP_STOP = 0.0008


def _hurst(ts, max_lag=20):
    """Exponente de Hurst por el método de los lags de varianza (rápido). >0.5 tendencial."""
    ts = np.asarray(ts, float)
    if len(ts) < max_lag + 5:
        return 0.5
    lags = list(range(2, max_lag))
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]   # desviación de los incrementos a cada lag
    tau = np.array(tau)
    if np.any(tau <= 0):
        return 0.5
    # std(incrementos de k lags) ~ k^H  ->  H = pendiente de log(tau) vs log(lag)
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def _trade_orb(close, high, low, dia_pos, or_hi, or_lo):
    """Simula el trade ORB (stop OR-low, salida EOD). Devuelve (R, breakout_min, ext_breakout, i_in)."""
    entry = None
    for k, j in enumerate(dia_pos[1:], start=1):
        if high[j] >= or_hi:
            entry = or_hi * (1 + SLIP); i_in = j; bk = k; break
    if entry is None:
        return None
    rdist = entry - or_lo
    if rdist <= 0:
        return None
    ext = (float(close[i_in]) - or_hi) / (or_hi - or_lo) if (or_hi - or_lo) > 0 else 0.0
    stop = or_lo
    dia_last = dia_pos[-1]
    exit_px = float(close[dia_last])
    for j in range(i_in, dia_last + 1):
        lo = float(low[j])
        if lo <= stop:
            exit_px = stop * (1 - SLIP_STOP); break
        if j == dia_last:
            exit_px = float(close[j]) * (1 - SLIP); break
    return (exit_px - entry) / rdist, bk * 5, ext, i_in


def construir(tickers_n=150, meses=24, out="datos/orb_dataset.csv"):
    uni = bi._universo(n=tickers_n)
    desde = (pd.Timestamp.now() - pd.Timedelta(days=int(meses * 31))).strftime("%Y-%m-%d")
    # SPY 5m: retorno de sesión + media 50 velas (régimen de mercado)
    spy = alpaca.historico("SPY", "5Min", desde=desde, max_paginas=24)
    spy_idx = pd.DatetimeIndex(spy.index)
    spy_sesion = bi.rasgos_intradia(spy)["intra_ret_sesion"]
    spy_ma = spy["close"].rolling(50, min_periods=10).mean()
    spy_trend = (spy["close"] > spy_ma).astype(float)
    spy_ret_map = pd.Series(spy_sesion.to_numpy(), index=spy_idx)
    spy_trend_map = pd.Series(spy_trend.to_numpy(), index=spy_idx)
    filas = []
    for ti, t in enumerate(uni):
        try:
            df = alpaca.historico(t, "5Min", desde=desde, max_paginas=24)
        except Exception:
            continue
        if len(df) < 500:
            continue
        close = df["close"].to_numpy(float); high = df["high"].to_numpy(float)
        low = df["low"].to_numpy(float); opn = df["open"].to_numpy(float)
        idx = pd.DatetimeIndex(df.index)
        orv = _opening_rvol(df)
        sesion_ret = bi.rasgos_intradia(df)["intra_ret_sesion"].to_numpy()   # causal, 1 vez por ticker
        atr = (ind.atr(df["high"], df["low"], df["close"], 14) / df["close"]).to_numpy()  # ATR%
        spy_ret_al = spy_ret_map.reindex(idx, method="ffill").to_numpy()
        spy_trend_al = spy_trend_map.reindex(idx, method="ffill").to_numpy()
        por_dia = pd.Series(np.arange(len(df)), index=idx).groupby(idx.normalize())
        dias_ok = [(d, g.to_numpy()) for d, g in por_dia if len(g) >= 20]
        for d, dia_pos in dias_ok:
            p0 = dia_pos[0]
            if p0 < 60:                                 # necesita historia para Hurst/ATR
                continue
            rv = orv.get(d, np.nan)
            if not (rv == rv) or rv < 1.0:              # stock in play (RVOL apertura > 1)
                continue
            if close[p0] <= opn[p0]:                     # 1ª vela alcista (dirección long)
                continue
            or_hi, or_lo = float(high[p0]), float(low[p0])
            tr = _trade_orb(close, high, low, list(dia_pos), or_hi, or_lo)
            if tr is None:
                continue
            R, bmin, ext, i_in = tr
            atrp = float(atr[p0]) if atr[p0] == atr[p0] else 0.0
            precio = float(close[p0])
            filas.append({
                "fecha": pd.Timestamp(d).strftime("%Y-%m-%d"), "ticker": t,
                "opening_rvol": float(rv),
                "gap": float(close[p0] / close[p0 - 1] - 1.0) if p0 >= 1 else 0.0,
                "or_range_atr": ((or_hi - or_lo) / precio / atrp) if (atrp > 0 and precio) else 0.0,
                "or_range_pct": (or_hi - or_lo) / precio if precio else 0.0,
                "breakout_min": float(bmin), "ext_breakout": float(ext),
                "day_ret": float(sesion_ret[i_in]) if sesion_ret[i_in] == sesion_ret[i_in] else 0.0,
                "atr_pct": atrp,
                "hurst": _hurst(close[max(0, p0 - 60):p0 + 1]),
                "spy_day_ret": float(spy_ret_al[p0]) if spy_ret_al[p0] == spy_ret_al[p0] else 0.0,
                "spy_trend": float(spy_trend_al[p0]) if spy_trend_al[p0] == spy_trend_al[p0] else 0.0,
                "dow": float(pd.Timestamp(d).dayofweek),
                "R": float(R), "y": int(R > 0),
            })
        if (ti + 1) % 20 == 0:
            print(f"  ...{ti+1}/{len(uni)} tickers, {len(filas)} breakouts")
    dfres = pd.DataFrame(filas)
    os.makedirs("datos", exist_ok=True)
    dfres.to_csv(out, index=False)
    print(f"\nDATASET: {len(dfres)} breakouts ORB | {dfres['ticker'].nunique()} tickers | "
          f"{dfres['fecha'].min()}..{dfres['fecha'].max()} -> {out}")
    if len(dfres):
        print(f"  win-rate bruto {dfres['y'].mean()*100:.0f}% | R medio {dfres['R'].mean():+.3f} | "
              f"R mediano {dfres['R'].median():+.3f}")
    return dfres


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=150)
    ap.add_argument("--meses", type=int, default=24)
    ap.add_argument("--out", type=str, default="datos/orb_dataset.csv")
    a = ap.parse_args()
    construir(a.tickers, a.meses, a.out)
