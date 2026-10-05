"""Re-validacion SURVIVORSHIP-FREE de las estrategias DIARIAS/swing con datos EOD de
Massive (incluye deslistados). Mide honestamente si tienen edge sobre SPY una vez que
se quita el sesgo de supervivencia. SOLO research/paper: lee datos, no coloca ordenes
ni toca la operativa en vivo (gestionar.py / operativa_*.py quedan intactos).

Tres estrategias (misma logica que los libros del repo, reimplementada como backtest):
  1. TSMOM (Libro C, operativa_tsmom.py): time-series momentum long/flat sobre una
     CESTA FIJA de 7 ETFs cross-asset. Rebalanceo mensual, peso inverso a la vol.
  2. ROTACION EUROPA (Libro D, operativa_rotacion.py): momentum transversal, top-3 por
     momentum 6m (skip 1m) sobre SMA200, cesta FIJA de 10 ETFs de pais. Mensual.
  3. RESIDUAL MOMENTUM (validar_residual_momentum.py): momentum sobre residuos FF3 de
     un universo de acciones. ESTA es la unica con universo cribado -> aqui se corre
     sobre el universo POINT-IN-TIME (con deslistados) vs el universo de HOY (sesgado),
     para medir el haircut de supervivencia.

Nota honesta de encuadre: (1) y (2) operan CESTAS FIJAS de ETFs que siguen cotizando;
no tienen sesgo de supervivencia que quitar (ningun miembro desaparecio). Para ellas la
pregunta real no es el haircut sino: ¿baten a SPY con datos limpios? Solo (3) tiene un
universo de acciones donde el sesgo muerde, y ahi si se mide biased vs sin-supervivencia.

  python backtest_diarias_massive.py            # corre todo y escribe el doc
  python backtest_diarias_massive.py etf        # solo TSMOM + rotacion
  python backtest_diarias_massive.py resmom     # solo residual momentum
"""
import io
import os
import sys
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from fuentes import massive
from fuentes.http import Cliente
from patrones.tsmom import signal_tsmom

# --- cestas (identicas a los libros en vivo) ---
CESTA_TSMOM = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "HYG"]
HORIZONTES = (21, 63, 126, 252)
CESTA_ROT = ["EWG", "EWQ", "EWP", "EWU", "EWI", "EWN", "EWL", "EWD", "EWO", "EWK"]
K_ROT, LB_ROT, SKIP_ROT = 3, 126, 21

DESDE = "2021-10-15"
HASTA = "2026-10-02"
COSTE_BPS = 10.0            # coste por unidad de turnover (comision + slippage), ETFs liquidos
SEED = 12345


# ----------------------------- datos -----------------------------
def _cm():
    """Cliente Massive con cache en disco (idempotente) y sin el espaciado del plan gratis."""
    cm = massive.ClienteMassive.__new__(massive.ClienteMassive)
    cm.cli = Cliente(cache_dir="datos/cache", ttl=7 * 24 * 3600, reintentos=5, pausa=2.0, timeout=60)
    cm.cli.sesion.headers["Authorization"] = f"Bearer {massive._claves()}"
    cm.pausa_min = 0.35
    cm._ultima = 0.0
    cm.llamadas = 0
    return cm


def panel_etf(tickers, desde=DESDE, hasta=HASTA, cm=None):
    """Panel wide de cierres diarios (ajustados por split) de una cesta de ETFs."""
    cm = cm or _cm()
    cols = {}
    for t in tickers:
        df = massive.barras_diarias(t, desde, hasta, cm=cm)
        if len(df):
            cols[t] = df["close"].astype(float)
    return pd.DataFrame(cols).sort_index().ffill(limit=5)


# ----------------------------- store EOD point-in-time -----------------------------
# Para residual momentum necesitamos un panel MENSUAL de muchas acciones sin sesgo de
# supervivencia. El universo_pit.py del repo sirve al ORB (fetch as-of, mid-caps) y cachea
# por dia; aqui se usa un almacen BULK propio y reproducible: top-1500 CS por volumen de
# cada sesion, deslistados incluidos (construido una vez con massive.grouped_diario y
# cacheado a parquet en datos/cache/, que esta en .gitignore).
_PIT_PARQUET = "datos/cache/pit_grouped.parquet"
_CS_JSON = "datos/cache/cs_set_diarias.json"
_TOPN_STORE = 1500


def _cm_rapido():
    cm = massive.ClienteMassive.__new__(massive.ClienteMassive)
    cm.cli = Cliente(cache_dir=None, reintentos=5, pausa=2.0, timeout=60)
    cm.cli.sesion.headers["Authorization"] = f"Bearer {massive._claves()}"
    cm.pausa_min = 0.35
    cm._ultima = 0.0
    cm.llamadas = 0
    return cm


def _cs_set(cm=None):
    import json
    if os.path.exists(_CS_JSON):
        try:
            return set(json.load(open(_CS_JSON)))
        except Exception:
            pass
    cm = cm or _cm_rapido()
    nombres = set()
    for activos in ("true", "false"):
        url, params = "/v3/reference/tickers", {"market": "stocks", "type": "CS",
                                                "active": activos, "limit": 1000}
        for _ in range(60):
            d = cm.get(url, params)
            for r in d.get("results") or []:
                if r.get("ticker"):
                    nombres.add(r["ticker"])
            url = d.get("next_url")
            if not url:
                break
            params = None
    os.makedirs(os.path.dirname(_CS_JSON), exist_ok=True)
    json.dump(sorted(nombres), open(_CS_JSON, "w"))
    return nombres


def construir_store(desde=DESDE, hasta=HASTA, cm=None, verbose=True):
    """Descarga grouped sesion a sesion, filtra a acciones comunes y guarda el top-1500
    por dollar-vol de cada dia. Reanudable. Incluye deslistados (hasta su baja)."""
    cm = cm or _cm_rapido()
    cs = _cs_set(cm)
    prev = cargar_store() if os.path.exists(_PIT_PARQUET) else pd.DataFrame()
    ya = set(prev["fecha"].astype(str).unique()) if len(prev) else set()
    filas = []
    for dia in pd.bdate_range(desde, hasta).strftime("%Y-%m-%d"):
        if dia in ya:
            continue
        g = massive.grouped_diario(dia, cm=cm)
        if g.empty:
            continue
        g = g[g["ticker"].isin(cs)].copy()
        if g.empty:
            continue
        g["dvol"] = g["close"].astype(float) * g["volume"].astype(float)
        g = g.sort_values("dvol", ascending=False).head(_TOPN_STORE)
        filas.append(pd.DataFrame({"fecha": dia, "ticker": g["ticker"].values,
                                   "close": g["close"].astype(float).values,
                                   "dvol": g["dvol"].values}))
    if filas:
        nuevo = pd.concat(filas, ignore_index=True)
        out = pd.concat([prev, nuevo], ignore_index=True) if len(prev) else nuevo
        out["fecha"] = pd.to_datetime(out["fecha"])
        out = out.drop_duplicates(["fecha", "ticker"]).sort_values(["fecha", "dvol"],
                                                                   ascending=[True, False])
        os.makedirs(os.path.dirname(_PIT_PARQUET), exist_ok=True)
        out.to_parquet(_PIT_PARQUET, index=False)
        return out
    return prev


def cargar_store():
    if not os.path.exists(_PIT_PARQUET):
        print("  (store PIT no encontrado; construyendolo, ~15-20 min la primera vez)...", flush=True)
        return construir_store()
    df = pd.read_parquet(_PIT_PARQUET)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def universo_asof_store(fecha, top_n=150, precio_min=5.0, store=None):
    """Tickers liquidos as-of 'fecha' (top-N por dollar-vol de la ultima sesion <= fecha).
    Survivorship-free: incluye los que cotizaban entonces aunque hoy esten deslistados."""
    df = store if store is not None else cargar_store()
    dias = df["fecha"][df["fecha"] <= pd.Timestamp(fecha)]
    if dias.empty:
        return []
    snap = df[(df["fecha"] == dias.max()) & (df["close"] >= precio_min)]
    return list(snap.sort_values("dvol", ascending=False)["ticker"].head(top_n))


def panel_cierres_store(tickers, store=None):
    df = store if store is not None else cargar_store()
    sub = df[df["ticker"].isin(set(tickers))]
    return sub.pivot_table(index="fecha", columns="ticker", values="close").sort_index()


# ----------------------------- metricas -----------------------------
def metricas(r, per=12):
    """r = retornos por periodo (serie). CAGR, Sharpe anual, maxDD, Calmar, %+ y n."""
    r = pd.Series(r).dropna()
    if len(r) < 6:
        return {}
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (per / n) - 1
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * np.sqrt(per) if sd > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    calmar = cagr / abs(dd) if dd < 0 else float("nan")
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "maxDD": float(dd),
            "Calmar": float(calmar), "pct_pos": float((r > 0).mean()),
            "vol": float(sd * np.sqrt(per)), "n": n}


def ic95_sharpe(r, per=12, B=5000, seed=SEED):
    """IC95 del Sharpe anualizado por bootstrap (remuestreo de meses con reemplazo)."""
    r = pd.Series(r).dropna().to_numpy()
    if len(r) < 12:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        s = r[rng.integers(0, len(r), len(r))]
        sd = s.std(ddof=1)
        out.append(s.mean() / sd * np.sqrt(per) if sd > 0 else 0.0)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def ic95_cagr(r, per=12, B=5000, seed=SEED):
    r = pd.Series(r).dropna().to_numpy()
    if len(r) < 12:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        s = r[rng.integers(0, len(r), len(r))]
        out.append((1 + s).prod() ** (per / len(s)) - 1)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def mensual_desde_diario(panel_close):
    """Serie mensual de retornos buy&hold de una columna de cierres diarios."""
    return panel_close.resample("ME").last().pct_change().dropna()


# ----------------------------- TSMOM (Libro C) -----------------------------
def backtest_tsmom(panel, coste_bps=COSTE_BPS, vol_obj=0.12, vol_lb=63, cap_peso=1.5):
    """Replica la LOGICA del libro en vivo (operativa_tsmom.objetivo_tsmom): long/flat,
    peso inverso a la vol entre los activos con TSMOM>0, normalizado al 100% invertido.
    Rebalanceo mensual. Sin look-ahead: señal con datos hasta t0, retorno de t0 a t1."""
    ret_d = panel.pct_change()
    vol_d = ret_d.rolling(vol_lb, min_periods=vol_lb // 2).std() * np.sqrt(252)
    rebal = [d for d in panel.resample("ME").last().dropna().index if d in set(panel.index)]
    min_hist = max(HORIZONTES) + 1          # lookback minimo para que haya señal TSMOM
    filas, w_prev = [], pd.Series(0.0, index=panel.columns)
    for i in range(len(rebal) - 1):
        t0, t1 = rebal[i], rebal[i + 1]
        if len(panel.loc[:t0]) < min_hist:   # warmup: aun sin historia -> no se cuenta
            continue
        sig = signal_tsmom(panel.loc[:t0], HORIZONTES)
        sig = sig[sig > 0]
        w = pd.Series(0.0, index=panel.columns)
        if len(sig):
            vol = vol_d.loc[t0].reindex(sig.index)
            wi = (sig * (vol_obj / vol)).clip(upper=cap_peso).fillna(0.0)
            if wi.sum() > 0:
                wi = wi / wi.sum()
                w.loc[wi.index] = wi.values
        r_ind = (panel.loc[t1] / panel.loc[t0] - 1.0).reindex(w.index).fillna(0.0)
        turn = float((w - w_prev).abs().sum())
        filas.append((t1, float((w * r_ind).sum()) - coste_bps / 1e4 * turn))
        w_prev = w
    return pd.Series(dict(filas)).dropna()


# ----------------------------- ROTACION EUROPA (Libro D) -----------------------------
def backtest_rotacion(panel, coste_bps=COSTE_BPS, k=K_ROT, lb=LB_ROT, skip=SKIP_ROT):
    """Replica la LOGICA del libro en vivo (operativa_rotacion.objetivo_rotacion): cada
    mes, largo el top-k por momentum 6m (skip 1m) que este sobre su SMA200; equiponderado;
    a efectivo lo demas. Rebalanceo mensual. Sin look-ahead."""
    sma200 = panel.rolling(200).mean()
    rebal = [d for d in panel.resample("ME").last().dropna().index if d in set(panel.index)]
    min_hist = max(205, lb + skip + 1)       # lookback minimo (momentum 6m + SMA200)
    filas, w_prev = [], pd.Series(0.0, index=panel.columns)
    for i in range(len(rebal) - 1):
        t0, t1 = rebal[i], rebal[i + 1]
        hist = panel.loc[:t0]
        if len(hist) < min_hist:             # warmup: aun sin historia -> no se cuenta
            continue
        mom = hist.iloc[-1 - skip] / hist.iloc[-1 - skip - lb] - 1
        elig = mom[hist.iloc[-1] > sma200.loc[t0]]
        top = elig.sort_values(ascending=False).head(k)
        w = pd.Series(0.0, index=panel.columns)
        if len(top):
            w.loc[top.index] = 1.0 / len(top)
        r_ind = (panel.loc[t1] / panel.loc[t0] - 1.0).reindex(w.index).fillna(0.0)
        turn = float((w - w_prev).abs().sum())
        filas.append((t1, float((w * r_ind).sum()) - coste_bps / 1e4 * turn))
        w_prev = w
    return pd.Series(dict(filas)).dropna()


# ----------------------------- FF3 (para residual momentum) -----------------------------
def ff3_mensual():
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_daily_CSV.zip")
    raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=60).read())).read(
        "F-F_Research_Data_Factors_daily.csv").decode("latin1").splitlines()
    filas = [l for l in raw if l[:8].strip().isdigit() and len(l.split(",")[0].strip()) == 8]
    df = pd.read_csv(io.StringIO("\n".join(filas)), header=None,
                     names=["fecha", "MktRF", "SMB", "HML", "RF"])
    df["fecha"] = pd.to_datetime(df["fecha"].astype(str), format="%Y%m%d")
    df = df.set_index("fecha").astype(float) / 100.0
    return (1 + df).resample("ME").prod() - 1


# ----------------------------- RESIDUAL MOMENTUM -----------------------------
def panel_mensual_pit(store, rebal_fechas, top_n=150, precio_min=5.0):
    """Panel MENSUAL de retornos de acciones sobre el universo POINT-IN-TIME: union de los
    top-N liquidos de cada rebalanceo (incluye deslistados). El retorno del mes en que un
    valor se deslista = su liquidacion (ultimo cierre), capturado por resample('ME').last()."""
    universo = set()
    for f in rebal_fechas:
        universo |= set(universo_asof_store(f, top_n=top_n, precio_min=precio_min, store=store))
    cierres = panel_cierres_store(sorted(universo), store=store)
    mens = cierres.resample("ME").last()
    return mens.pct_change(), universo


def _resid_signal(retm, ff, t_idx, win=36, form=11, skip=1):
    """Señal residual-momentum en el mes t_idx: regresa cada accion sobre FF3 en la ventana
    [t-win+1, t], toma los residuos de formacion (ultimos form, saltando skip) y devuelve
    media/std. Sin look-ahead (solo pasado)."""
    sl = slice(t_idx - win + 1, t_idx + 1)
    Xi = np.column_stack([np.ones(win), ff["MktRF"].iloc[sl], ff["SMB"].iloc[sl], ff["HML"].iloc[sl]])
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
    return pd.Series(sig).dropna()


def backtest_resmom(retm, ff, win=24, form=11, skip=1, q=0.2, ls=True, coste_bps=20.0,
                    min_names=20):
    """Residual momentum mensual. En t: señal FF3-residual; largo el top-q (y corto el
    bottom-q si ls); retorno en t+1. coste por turnover. Acciones -> coste mas alto."""
    idx = retm.index.intersection(ff.index)
    retm, ff = retm.loc[idx], ff.loc[idx]
    meses = list(retm.index)
    filas, w_prev = [], pd.Series(0.0, index=retm.columns)
    for i in range(win, len(meses) - 1):
        t1 = meses[i + 1]
        sig = _resid_signal(retm, ff, i, win, form, skip)
        if len(sig) < min_names:
            continue
        n = max(int(len(sig) * q), 1)
        orden = sig.sort_values(ascending=False)
        w = pd.Series(0.0, index=retm.columns)
        w[orden.index[:n]] = 1.0 / n
        if ls:
            w[orden.index[-n:]] = -1.0 / n
        rfwd = retm.loc[t1].reindex(w.index).fillna(0.0)
        turn = float((w - w_prev).abs().sum())
        filas.append((t1, float((w * rfwd).sum()) - coste_bps / 1e4 * turn))
        w_prev = w
    return pd.Series(dict(filas)).dropna()


# ----------------------------- orquestacion -----------------------------
def _fmt(m):
    if not m:
        return "n/d"
    return (f"CAGR {m['CAGR']*100:+.1f}% | Sharpe {m['Sharpe']:.2f} | "
            f"maxDD {m['maxDD']*100:.1f}% | Calmar {m['Calmar']:.2f} | "
            f"+meses {m['pct_pos']*100:.0f}% | n={m['n']}")


def correr_etf(cm=None):
    cm = cm or _cm()
    print("Descargando ETFs (TSMOM + rotacion) de Massive...", flush=True)
    pan_t = panel_etf(CESTA_TSMOM, cm=cm)
    pan_r = panel_etf(CESTA_ROT, cm=cm)
    spy = pan_t["SPY"] if "SPY" in pan_t else panel_etf(["SPY"], cm=cm)["SPY"]

    res = {}
    rt = backtest_tsmom(pan_t)
    res["tsmom"] = {"ret": rt, "met": metricas(rt), "ic_s": ic95_sharpe(rt), "ic_c": ic95_cagr(rt)}
    rr = backtest_rotacion(pan_r)
    res["rotacion"] = {"ret": rr, "met": metricas(rr), "ic_s": ic95_sharpe(rr), "ic_c": ic95_cagr(rr)}

    # SPY benchmark alineado a cada periodo
    for k in ("tsmom", "rotacion"):
        r = res[k]["ret"]
        spy_m = mensual_desde_diario(spy).reindex(r.index).dropna()
        res[k]["spy"] = {"ret": spy_m, "met": metricas(spy_m), "ic_s": ic95_sharpe(spy_m)}
        print(f"\n[{k.upper()}] {r.index.min().date()}..{r.index.max().date()}")
        print(f"  estrategia: {_fmt(res[k]['met'])}  IC95 Sharpe [{res[k]['ic_s'][0]:.2f},{res[k]['ic_s'][1]:.2f}]")
        print(f"  SPY B&H   : {_fmt(res[k]['spy']['met'])}")
    return res


def correr_resmom(cm=None):
    print("\nResidual momentum survivorship-free (universo PIT con deslistados)...", flush=True)
    store = cargar_store()
    print(f"  store PIT: {store['fecha'].nunique()} sesiones, "
          f"{store['ticker'].nunique()} tickers unicos, "
          f"{store['fecha'].dt.date.min()}..{store['fecha'].dt.date.max()}", flush=True)
    ff = ff3_mensual()
    # fechas de rebalanceo = fin de mes del store
    meses = pd.Series(store["fecha"].unique()).sort_values()
    rebal = list(pd.Series(meses.values, index=pd.DatetimeIndex(meses)).resample("ME").last().dropna())

    # --- sin supervivencia (universo PIT en cada rebalanceo) ---
    retm_pit, uni_pit = panel_mensual_pit(store, rebal, top_n=150)
    # --- sesgado (universo de HOY: top-N liquido de la ULTIMA sesion, solo supervivientes) ---
    hoy = store["fecha"].max()
    survivors = universo_asof_store(hoy, top_n=150, store=store)
    retm_bias = retm_pit[[c for c in survivors if c in retm_pit.columns]]

    out = {}
    for nombre, retm in [("sin_superv", retm_pit), ("sesgado", retm_bias)]:
        ls = backtest_resmom(retm, ff, ls=True)
        lo = backtest_resmom(retm, ff, ls=False)
        out[nombre] = {
            "LS": {"ret": ls, "met": metricas(ls), "ic_s": ic95_sharpe(ls), "ic_c": ic95_cagr(ls)},
            "LO": {"ret": lo, "met": metricas(lo), "ic_s": ic95_sharpe(lo), "ic_c": ic95_cagr(lo)},
            "n_universo": retm.shape[1]}
        print(f"\n[RESMOM {nombre}] universo {retm.shape[1]} acciones")
        print(f"  L/S       : {_fmt(out[nombre]['LS']['met'])}  IC95 Sharpe "
              f"[{out[nombre]['LS']['ic_s'][0]:.2f},{out[nombre]['LS']['ic_s'][1]:.2f}]")
        print(f"  long-only : {_fmt(out[nombre]['LO']['met'])}  IC95 Sharpe "
              f"[{out[nombre]['LO']['ic_s'][0]:.2f},{out[nombre]['LO']['ic_s'][1]:.2f}]")
    # SPY sobre el periodo del resmom
    cm = cm or _cm()
    spy = panel_etf(["SPY"], cm=cm)["SPY"]
    r_any = out["sin_superv"]["LO"]["ret"]
    spy_m = mensual_desde_diario(spy).reindex(r_any.index).dropna()
    out["spy"] = {"ret": spy_m, "met": metricas(spy_m), "ic_s": ic95_sharpe(spy_m)}
    print(f"\n  SPY B&H (periodo resmom): {_fmt(out['spy']['met'])}")
    return out


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "all"
    cm = _cm()
    if modo in ("etf", "all"):
        correr_etf(cm)
    if modo in ("resmom", "all"):
        correr_resmom(cm)
