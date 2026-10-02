"""Analitica de rendimiento (Tier 2): lo que hace que un sistema PAREZCA y SEA un
fondo. Tres bloques:

  1. TEAR-SHEET institucional: Sharpe, Sortino, Calmar, max drawdown, CAGR, vol,
     hit-rate, y relativo a un benchmark (beta, alfa, information ratio, correlacion).
     Todo a partir de la curva de equity real de la cuenta.
  2. SLIPPAGE de ejecucion: compara el precio ESPERADO al enviar la orden (lo que
     anota la bitacora) contra el precio REAL del fill de Alpaca. Mide si el modelo de
     costes es honesto -> los costes son lo que mata la mayoria de los edges.
  3. MONITOR DE DECAIMIENTO: compara las metricas EN VIVO con lo esperado del backtest
     y avisa si divergen (alpha decay). Con muestra pequena lo dice honestamente.

Solo lee datos (equity, fills, log de trades, Yahoo). No opera.
"""
import numpy as np
import pandas as pd


def _ret(equity):
    return pd.Series(equity).astype(float).replace(0, np.nan).dropna().pct_change().dropna()


# ------------------------------------------------- tear-sheet
def metricas(equity, per=252, rf_anual=0.0):
    """Metricas de una curva de equity (serie). per = periodos/ano (252 diario)."""
    r = _ret(equity)
    if len(r) < 3:
        return {"n": len(r), "insuficiente": True}
    rf = rf_anual / per
    ex = r - rf
    mu, sd = ex.mean(), r.std(ddof=1)
    # downside deviation CORRECTA: dispersion de los retornos bajo el objetivo (0),
    # no la std de los negativos alrededor de su propia media.
    dn = np.minimum(ex.to_numpy(), 0.0)
    down = float(np.sqrt((dn ** 2).mean()))
    # curva de equity ANCLADA en 1 (incluye el primer periodo) para el drawdown y CAGR
    eq = np.concatenate([[1.0], (1 + r).cumprod().to_numpy()])
    picos = np.maximum.accumulate(eq)
    maxdd = float((eq / picos - 1).min())
    cagr = float(eq[-1] ** (per / len(r)) - 1)
    return {
        "n": int(len(r)),
        "cagr": cagr,
        "vol_anual": float(sd * np.sqrt(per)),
        "sharpe": float(mu / sd * np.sqrt(per)) if sd > 0 else float("nan"),
        "sortino": float(mu / down * np.sqrt(per)) if down and down > 0 else float("nan"),
        "calmar": float(cagr / abs(maxdd)) if maxdd < 0 else float("nan"),
        "max_drawdown": maxdd,
        "hit_rate": float((r > 0).mean()),
        "mejor_dia": float(r.max()), "peor_dia": float(r.min()),
        "ret_total": float(eq[-1] - 1),
    }


def vs_benchmark(equity, bench_equity, per=252):
    """Metricas RELATIVAS a un benchmark (p.ej. SPY): beta, alfa anualizada,
    correlacion e information ratio (retorno activo / tracking error)."""
    r = _ret(equity)
    rb = _ret(bench_equity)
    # alinear por FECHA: el equity de Alpaca llega con hora de cierre y el SPY de Yahoo
    # a medianoche -> sin normalizar NUNCA coinciden las etiquetas y el merge daba 0 filas.
    try:
        r.index = pd.to_datetime(r.index).normalize()
        rb.index = pd.to_datetime(rb.index).normalize()
        r = r[~r.index.duplicated(keep="last")]
        rb = rb[~rb.index.duplicated(keep="last")]
    except Exception:
        pass
    df = pd.DataFrame({"r": r, "rb": rb}).dropna()
    if len(df) < 5:
        return {"n": len(df), "insuficiente": True}
    r, rb = df["r"], df["rb"]
    var_b = rb.var(ddof=1)
    beta = float(np.cov(r, rb, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
    alpha_ann = float((r.mean() - beta * rb.mean()) * per) if np.isfinite(beta) else float("nan")
    activo = r - rb
    te = activo.std(ddof=1)
    return {
        "n": int(len(df)),
        "beta": beta,
        "alpha_anual": alpha_ann,
        "correlacion": float(r.corr(rb)),
        "information_ratio": float(activo.mean() / te * np.sqrt(per)) if te > 0 else float("nan"),
        "exceso_total": float((1 + r).prod() - (1 + rb).prod()),
    }


def rolling_sharpe(equity, ventana=63, per=252):
    r = _ret(equity)
    if len(r) < ventana:
        return pd.Series(dtype=float)
    m = r.rolling(ventana).mean()
    s = r.rolling(ventana).std(ddof=1)
    return (m / s * np.sqrt(per)).dropna()


# ------------------------------------------------- slippage de ejecucion
def slippage(log_df, fills, tol_min=20):
    """Slippage de ENTRADA: compara el precio esperado (columna 'entrada' del log, el
    precio al enviar la orden) contra el precio REAL del fill de compra de Alpaca.
    slippage_bps > 0 = se pago MAS de lo esperado (coste real). Empareja por ticker y
    el fill de compra mas cercano en tiempo (<= tol_min minutos)."""
    if log_df is None or len(log_df) == 0:
        return {"n": 0}, pd.DataFrame()
    def _naive(x):
        t = pd.to_datetime(x, utc=True, errors="coerce")
        return t.tz_localize(None) if t is not None and not pd.isna(t) else pd.NaT
    compras = [f for f in (fills or []) if str(f.get("side", "")).startswith("buy") and f.get("price")]
    porfill = {}
    for f in compras:
        tf = _naive(f.get("transaction_time"))
        if not pd.isna(tf):
            porfill.setdefault(f.get("symbol"), []).append([tf, float(f["price"]), False])  # [t, precio, usado]
    filas = []
    # emparejar la entrada MAS antigua primero, consumiendo el fill (no reutilizar uno
    # para dos entradas; parciales: cada entrada toma su fill de compra mas cercano libre)
    for _, row in log_df.sort_values("ts").iterrows() if "ts" in log_df.columns else log_df.iterrows():
        t = row.get("ticker")
        try:
            esperado = float(row.get("entrada"))
        except Exception:
            continue
        if not np.isfinite(esperado) or esperado <= 0:   # celda vacia -> NaN; NaN no es falsy
            continue
        ts = _naive(row.get("ts"))
        if t not in porfill or pd.isna(ts):
            continue
        libres = [c for c in porfill[t] if not c[2]]
        if not libres:
            continue
        cand = min(libres, key=lambda x: abs(x[0] - ts))
        if abs(cand[0] - ts).total_seconds() / 60 > tol_min:
            continue
        cand[2] = True                                   # consumir el fill
        filas.append({"ticker": t, "esperado": esperado, "real": cand[1],
                      "slippage_bps": (cand[1] / esperado - 1) * 1e4})
    if not filas:
        return {"n": 0}, pd.DataFrame()
    d = pd.DataFrame(filas)
    s = d["slippage_bps"]
    out = {"n": len(d), "medio_bps": float(s.mean()), "mediana_bps": float(s.median()),
           "coste_medio_pct": float(s.mean() / 100)}
    if len(d) >= 5:                                       # percentiles solo con muestra decente
        out["peor_bps"] = float(s.max())
        out["p90_bps"] = float(s.quantile(0.9))
    return out, d


# ------------------------------------------------- estadisticas de operaciones
def stats_trades(log_df):
    """Hit-rate, ganancia/perdida media, profit factor y esperanza de las operaciones
    CERRADAS del log intradia (las que tienen pnl)."""
    if log_df is None or len(log_df) == 0:
        return {"n": 0}
    d = log_df.copy()
    d["pnl"] = pd.to_numeric(d.get("pnl"), errors="coerce")
    cer = d.dropna(subset=["pnl"])
    if len(cer) == 0:
        return {"n": 0, "abiertas": int((d.get("resultado") == "open").sum()) if "resultado" in d else 0}
    g = cer[cer["pnl"] > 0]["pnl"]
    p = cer[cer["pnl"] < 0]["pnl"]
    pf = float(g.sum() / abs(p.sum())) if len(p) and p.sum() != 0 else float("inf")
    return {"n": int(len(cer)), "hit_rate": float((cer["pnl"] > 0).mean()),
            "ganancia_media": float(g.mean()) if len(g) else 0.0,
            "perdida_media": float(p.mean()) if len(p) else 0.0,
            "profit_factor": pf,
            "esperanza": float(cer["pnl"].mean()), "pnl_total": float(cer["pnl"].sum())}


# ------------------------------------------------- monitor de decaimiento
def decaimiento(sharpe_vivo, sharpe_bt, hit_vivo=None, hit_bt=None, n=0, n_min=20):
    """Compara lo EN VIVO con lo esperado del backtest. Con muestra pequena (< n_min)
    lo dice y no alarma. Marca DEGRADADO si el Sharpe en vivo cae por debajo de la
    mitad del backtest con muestra suficiente."""
    if n < n_min:
        return {"estado": "MUESTRA_INSUFICIENTE",
                "nota": f"solo {n} obs (< {n_min}); aun no se puede juzgar decaimiento",
                "sharpe_vivo": sharpe_vivo, "sharpe_bt": sharpe_bt}
    ratio = (sharpe_vivo / sharpe_bt) if sharpe_bt and sharpe_bt > 0 else float("nan")
    if np.isfinite(ratio) and ratio < 0.5:
        estado = "DEGRADADO"
    elif np.isfinite(ratio) and ratio < 0.8:
        estado = "VIGILAR"
    else:
        estado = "OK"
    return {"estado": estado, "ratio_sharpe": ratio, "sharpe_vivo": sharpe_vivo,
            "sharpe_bt": sharpe_bt, "hit_vivo": hit_vivo, "hit_bt": hit_bt, "n": n}
