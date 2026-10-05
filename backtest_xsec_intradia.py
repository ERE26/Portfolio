"""ESTRATEGIA A - CROSS-SECTIONAL INTRADAY REVERSION (el stat-arb de los fondos).

Idea: en horizontes intradia, la parte IDIOSINCRATICA del movimiento de una
accion (lo que se mueve de mas respecto al mercado) tiende a REVERTIR. Un fondo
no apuesta por una accion: construye una CARTERA dolar-neutral y aprox
beta-neutral de cientos de nombres, larga el decil de residuo mas bajo (barato) y
corta el mas alto (caro), y se lo come la ley de los grandes numeros. El alfa por
nombre es minusculo; lo que lo hace invertible es la ESCALA y una ejecucion
barata.

Este backtest contesta: ¿esa cartera neteada da un Sharpe intradia robusto? ¿y
sobrevive al coste de ejecucion que paga una cuenta RETAIL (taker, cruza a
mercado) frente al que paga un FONDO (maker, provee liquidez y cobra rebate)?

RIGOR (sin look-ahead):
  - Universo POINT-IN-TIME (universo_pit): top por dolar-volumen as-of, incluidos
    los que luego se deslistaron. Sin sesgo de supervivencia.
  - Beta por accion estimada SOLO con la ventana de formacion (anterior a la de
    prueba). No mira al futuro.
  - Señal en la barra t: retorno de las ultimas L barras MENOS beta*retorno del
    mercado (SPY) -> residuo; se neutraliza en seccion cruzada (resta la media).
    La ventana de la señal termina en t; el retorno que se cobra va de t a t+K,
    asi que no se solapan: la barra que decide no es la barra que paga.
  - Cartera formada al CIERRE de t (precio de la señal). Flat al cierre de sesion.
  - 3 semillas (submuestras de dias), bootstrap por bloques-dia IC95.

COSTE (dos escenarios, se reportan ambos):
  - TAKER (retail, cruza a mercado): c_side = 3 bps/lado. coste = c_side*turnover.
  - MAKER (fondo, limite pasivo + rebate): c_side = -0.25 bps/lado (cobra parte
    del spread). Supone relleno al precio de la señal (optimista: el pasivo no
    siempre entra; ver caveats del informe).

Uso:
  python backtest_xsec_intradia.py --top 100 --form-desde 2024-01-02 \
      --form-hasta 2024-02-29 --test-desde 2024-03-01 --test-hasta 2024-05-24 \
      --barras 5 --look 6 --rebal 6 --dias-muestra 40
"""
import argparse

import numpy as np
import pandas as pd

import universo_pit as U
from fuentes.massive import (ClienteMassive, barras_intradia, barras_diarias,
                             PAUSA_STARTER, NY)

BPS = 1e-4
C_TAKER = 3.0 * BPS      # 3 bps/lado cruzando a mercado
C_MAKER = -0.25 * BPS    # -0.25 bps/lado: limite pasivo + rebate (cobra spread)


def _betas(cierres_form, cm):
    """Beta diaria de cada ticker vs SPY, estimada SOLO en la ventana de formacion.
    cierres_form: DataFrame fechas x tickers (cierres diarios de formacion)."""
    spy = barras_diarias("SPY", cierres_form.index.min().strftime("%Y-%m-%d"),
                         cierres_form.index.max().strftime("%Y-%m-%d"), cm=cm)
    spy_ret = spy["close"].reindex(cierres_form.index).pct_change()
    rets = cierres_form.pct_change()
    var_m = np.nanvar(spy_ret.values)
    betas = {}
    for t in cierres_form.columns:
        r = rets[t].values
        m = spy_ret.values
        ok = np.isfinite(r) & np.isfinite(m)
        if ok.sum() >= 8 and var_m > 0:
            betas[t] = float(np.cov(r[ok], m[ok])[0, 1] / np.var(m[ok]))
        else:
            betas[t] = 1.0
    return pd.Series(betas).clip(0.0, 3.0)


def _panel_intradia(tickers, mult, desde, hasta, cm):
    """Descarga barras intradia de cada ticker + SPY y devuelve un panel alineado.
    Devuelve (closes_df [ts x tickers], spy_close [ts], dias) sobre la rejilla de SPY."""
    tf = f"{mult}Min"
    spy = barras_intradia("SPY", tf, desde, hasta, cm=cm)
    if spy.empty:
        raise RuntimeError("SPY intradia vacio")
    grid = spy.index
    cols = {}
    faltan = 0
    for t in tickers:
        try:
            df = barras_intradia(t, tf, desde, hasta, cm=cm)
        except Exception:
            faltan += 1
            continue
        if df.empty or len(df) < len(grid) * 0.3:
            faltan += 1
            continue
        # reindex a la rejilla de SPY, ffill SOLO dentro del mismo dia (no cruza noches)
        s = df["close"].reindex(grid)
        s = s.groupby(grid.normalize()).ffill()
        cols[t] = s
    closes = pd.DataFrame(cols)
    # exige cobertura >=60% de la rejilla; descarta nombres con muchos huecos
    cob = closes.notna().mean()
    closes = closes.loc[:, cob >= 0.60]
    print(f"[panel] rejilla {len(grid)} barras | {closes.shape[1]} nombres validos "
          f"({faltan} descartados)")
    return closes, spy["close"], grid


def _residuos(closes, spy_close, betas, look):
    """Residuo en cada barra: ret_L del nombre - beta*ret_L de SPY, por dia (sin
    cruzar noches), neutralizado en seccion cruzada (resta la media de la fila)."""
    grid = closes.index
    dia = grid.normalize()
    # retorno de L barras dentro del mismo dia
    def ret_L(s):
        out = s / s.groupby(dia).shift(look) - 1.0
        return out
    r_stock = closes.apply(ret_L)
    r_mkt = ret_L(spy_close)
    bt = betas.reindex(closes.columns).fillna(1.0).values
    resid = r_stock.values - np.outer(r_mkt.values, bt)
    resid = pd.DataFrame(resid, index=grid, columns=closes.columns)
    # neutralizar en seccion cruzada (quita el factor comun residual medio)
    resid = resid.sub(resid.mean(axis=1), axis=0)
    return resid


def _pesos(resid_row, frac=0.1):
    """Pesos dolar-neutral: long el decil mas bajo (barato), short el mas alto.
    sum(|w|)=1 (gross leverage 1: 0.5 largo + 0.5 corto)."""
    v = resid_row.dropna()
    n = len(v)
    k = max(1, int(n * frac))
    if n < 2 * k + 2:
        return None
    orden = v.sort_values()
    longs = orden.index[:k]       # residuo mas bajo -> revierte al alza
    shorts = orden.index[-k:]     # residuo mas alto -> revierte a la baja
    w = pd.Series(0.0, index=resid_row.index)
    w[longs] = 0.5 / k
    w[shorts] = -0.5 / k
    return w


def _simular(closes, resid, grid, rebal, frac, entry_lag=0):
    """Recorre las barras; en cada barra de rebalanceo forma cartera y cobra el
    retorno hasta el siguiente rebalanceo. Flat al cierre de cada dia.
    entry_lag>0: la cartera decidida en t se RELLENA 'entry_lag' barras despues
    (t+lag), no en t. Si el edge desaparece con lag=1, la "reversion" era rebote
    bid-ask (microestructura), no alfa predecible y tradeable.
    Devuelve DataFrame por-periodo: fecha, ret_bruto, turnover."""
    dia = grid.normalize()
    pos = np.arange(len(grid))
    filas = []
    for d, idxs in pd.Series(pos, index=grid).groupby(dia):
        barras = idxs.to_numpy()
        if len(barras) < rebal + entry_lag + 2:
            continue
        # barras de rebalanceo: desde la primera con señal valida hasta dejar hueco K
        w_prev = pd.Series(0.0, index=closes.columns)
        rb = list(range(0, len(barras) - rebal - entry_lag, rebal))
        for bi in rb:
            t = barras[bi + entry_lag]            # relleno (entrada) en t+lag
            tn = barras[bi + entry_lag + rebal]
            w = _pesos(resid.iloc[barras[bi]], frac=frac)   # señal decidida en bi
            if w is None:
                continue
            w = w.reindex(closes.columns).fillna(0.0)
            # retorno de cada nombre de t a t+K (relleno al cierre de t = precio señal)
            p0 = closes.iloc[t]
            p1 = closes.iloc[tn]
            fwd = (p1 / p0 - 1.0).reindex(closes.columns)
            valido = w[w != 0].index
            fwd_v = fwd[valido]
            if fwd_v.isna().any():
                # nombre sin precio de salida: lo sacamos de la cartera (peso 0)
                buenos = fwd_v.dropna().index
                w = w.copy(); w[~w.index.isin(buenos)] = 0.0
                if (w == 0).all():
                    continue
                fwd = fwd.fillna(0.0)
            ret_bruto = float((w * fwd.reindex(w.index).fillna(0.0)).sum())
            turn = float((w - w_prev).abs().sum())
            filas.append({"fecha": d, "ret": ret_bruto, "turn": turn})
            w_prev = w
        # cierre del dia: liquidar (turnover de salida) como coste del ultimo periodo
        if (w_prev != 0).any():
            filas.append({"fecha": d, "ret": 0.0, "turn": float(w_prev.abs().sum())})
    return pd.DataFrame(filas)


def _neto(periodos, c_side):
    """Retorno neto por periodo = bruto - c_side*turnover."""
    return periodos["ret"].values - c_side * periodos["turn"].values


def _boot_dia(periodos, valores, n=2000, seed=1):
    """Bootstrap por BLOQUES-DIA (respeta autocorrelacion intradia): remuestrea
    dias con reemplazo y recalcula la media por periodo. IC95 de la media."""
    rng = np.random.default_rng(seed)
    df = periodos.copy()
    df["_v"] = valores
    dias = list(df.groupby("fecha"))
    if len(dias) < 3:
        return (float("nan"), float("nan"))
    arrs = [g["_v"].values for _, g in dias]
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(arrs), len(arrs))
        cat = np.concatenate([arrs[i] for i in pick])
        means.append(cat.mean())
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _metricas(periodos, valores, per_dia):
    """Sharpe anualizado, expectancy (bps), IC95 de la media (bps), maxDD (%)."""
    v = np.asarray(valores, float)
    if len(v) < 5 or v.std() == 0:
        return None
    per_año = per_dia * 252
    sharpe = v.mean() / v.std() * np.sqrt(per_año)
    cum = np.cumsum(v)
    pico = np.maximum.accumulate(cum)
    maxdd = float((cum - pico).min())
    lo, hi = _boot_dia(periodos, v)
    return {"sharpe": float(sharpe), "exp_bps": float(v.mean() / BPS),
            "ic_lo_bps": lo / BPS, "ic_hi_bps": hi / BPS,
            "maxdd_pct": maxdd * 100, "n": len(v), "ndias": periodos["fecha"].nunique()}


def correr(top=100, form_desde="2024-01-02", form_hasta="2024-02-29",
           test_desde="2024-03-01", test_hasta="2024-05-24", mult=5, look=6,
           rebal=6, frac=0.1, dias_muestra=40, seeds=(1, 2, 3), entry_lag=0):
    cm = ClienteMassive(pausa_min=PAUSA_STARTER)
    tickers, cierres_form, _ = U.formar(form_desde, form_hasta, top=top, cm=cm)
    betas = _betas(cierres_form, cm)
    closes, spy_close, grid = _panel_intradia(tickers, mult, test_desde, test_hasta, cm)
    resid = _residuos(closes, spy_close, betas, look)
    periodos = _simular(closes, resid, grid, rebal, frac, entry_lag=entry_lag)
    if periodos.empty:
        print("sin periodos validos")
        return None
    # periodos de holding por dia (excluye el de liquidacion con ret=0)
    per_dia = max(1, int(np.median(periodos[periodos["ret"] != 0]
                                   .groupby("fecha").size())))
    dias_disp = sorted(periodos["fecha"].unique())
    print(f"\n### ESTRATEGIA A | {closes.shape[1]} nombres | barras {mult}min | "
          f"look {look} ({look*mult}min) | rebal {rebal} ({rebal*mult}min) | "
          f"{len(dias_disp)} dias de prueba | ~{per_dia} holdings/dia")
    out = {"TAKER": [], "MAKER": [], "BRUTO": []}
    rng = np.random.default_rng(0)
    for sd in seeds:
        rs = np.random.default_rng(sd)
        pick = rs.choice(len(dias_disp), min(dias_muestra, len(dias_disp)), replace=False)
        dsel = set(dias_disp[i] for i in pick)
        p = periodos[periodos["fecha"].isin(dsel)].reset_index(drop=True)
        for nombre, c in (("BRUTO", 0.0), ("TAKER", C_TAKER), ("MAKER", C_MAKER)):
            v = _neto(p, c)
            m = _metricas(p, v, per_dia)
            if m:
                m["seed"] = sd
                out[nombre].append(m)
    # pooled (todos los dias)
    pooled = {}
    for nombre, c in (("BRUTO", 0.0), ("TAKER", C_TAKER), ("MAKER", C_MAKER)):
        v = _neto(periodos, c)
        pooled[nombre] = _metricas(periodos, v, per_dia)
    _imprimir(out, pooled)
    return {"seeds": out, "pooled": pooled, "per_dia": per_dia,
            "n_nombres": int(closes.shape[1]), "ndias": len(dias_disp),
            "turn_medio": float(periodos["turn"].mean())}


def _imprimir(out, pooled):
    for nombre in ("BRUTO", "TAKER", "MAKER"):
        print(f"\n--- {nombre} ---")
        for m in out[nombre]:
            print(f"  seed {m['seed']}: Sharpe {m['sharpe']:+.2f} | exp {m['exp_bps']:+.3f}bps "
                  f"IC95[{m['ic_lo_bps']:+.3f},{m['ic_hi_bps']:+.3f}] | maxDD {m['maxdd_pct']:+.2f}% | n={m['n']}")
        p = pooled[nombre]
        if p:
            vd = "ROBUSTO+" if p["ic_lo_bps"] > 0 else ("SIN EDGE" if p["ic_hi_bps"] > 0 else "PIERDE")
            print(f"  POOLED : Sharpe {p['sharpe']:+.2f} | exp {p['exp_bps']:+.3f}bps "
                  f"IC95[{p['ic_lo_bps']:+.3f},{p['ic_hi_bps']:+.3f}] | maxDD {p['maxdd_pct']:+.2f}% "
                  f"| {p['ndias']}d -> {vd}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--form-desde", default="2024-01-02")
    ap.add_argument("--form-hasta", default="2024-02-29")
    ap.add_argument("--test-desde", default="2024-03-01")
    ap.add_argument("--test-hasta", default="2024-05-24")
    ap.add_argument("--barras", type=int, default=5)
    ap.add_argument("--look", type=int, default=6)
    ap.add_argument("--rebal", type=int, default=6)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--dias-muestra", type=int, default=40)
    a = ap.parse_args()
    correr(a.top, a.form_desde, a.form_hasta, a.test_desde, a.test_hasta,
           a.barras, a.look, a.rebal, a.frac, a.dias_muestra)
