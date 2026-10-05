"""Busqueda de NUEVAS familias de edge INTRADIA con datos limpios de Massive (5 min y EOD),
sobre un universo FIJO de ETFs/mega-caps liquidos (sin sesgo de supervivencia: ninguno se
deslista). Complementa la busqueda previa del repo (docs/intradia_busqueda_edge.md), que ya
descarto momentum generico, pullback-VWAP y mean-reversion, y dejo el ORB de stocks-in-play
como unico edge (modesto y fragil). Aqui se prueban familias que AUN NO se habian medido:

  A) OVERNIGHT vs INTRADAY (close-to-open): ¿de donde viene el retorno, de la noche o del dia?
     Estrategia operable: comprar al cierre, vender a la apertura (MOC/MOO). Anomalia muy
     documentada (overnight drift). EOD.
  B) ORB de INDICE (SPY/QQQ): opening-range breakout a nivel de indice (no de stock-in-play),
     entrada en ruptura CONFIRMADA, stop al lado opuesto del rango, salida EOD. 5 min.
  C) MOMENTUM de ULTIMA HORA: si el dia va en una direccion a las 15:00, seguirla 15:00->16:00.
  D) GAP-FADE en ETFs: si abre con hueco grande vs cierre previo, contrariarlo hacia el cierre.

Rigor (igual que el resto del repo): point-in-time sin look-ahead, costes realistas, IC95 por
bootstrap, y split train/test (primer 60% / ultimo 40%) para ver si el edge aguanta fuera de
muestra. Honestidad > resultado bonito: lo que no aguante, se dice.

SOLO research/paper. No coloca ordenes ni toca la operativa en vivo.

  python backtest_intradia_massive.py            # corre todo y escribe el doc
  python backtest_intradia_massive.py fetch       # solo descarga/cachea datos
"""
import os
import sys

import numpy as np
import pandas as pd

from fuentes import massive
from fuentes.http import Cliente

# universos fijos y liquidos (survivorship-free por construccion)
ETFS_EOD = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
            "XLI", "XLU", "XLB"]
ETFS_5M = ["SPY", "QQQ", "IWM"]                 # ORB de indice + ultima hora + gap-fade
DESDE = "2021-11-01"
HASTA = "2026-10-02"
CACHE_5M = "datos/cache/intradia_5m"
SLIP_BPS = 1.5          # slippage por lado en ETF liquido (SPY/QQQ ~0.5-1bp; conservador)
SEED = 12345


def _cm():
    cm = massive.ClienteMassive.__new__(massive.ClienteMassive)
    cm.cli = Cliente(cache_dir="datos/cache", ttl=30 * 24 * 3600, reintentos=5, pausa=2.0, timeout=90)
    cm.cli.sesion.headers["Authorization"] = f"Bearer {massive._claves()}"
    cm.pausa_min = 0.35
    cm._ultima = 0.0
    cm.llamadas = 0
    return cm


# ----------------------------- datos -----------------------------
def eod(tickers=ETFS_EOD, cm=None):
    cm = cm or _cm()
    out = {}
    for t in tickers:
        df = massive.barras_diarias(t, DESDE, HASTA, cm=cm)
        if len(df):
            out[t] = df
    return out


def intradia_5m(ticker, cm=None):
    """5 min RTH cacheado a parquet por ticker (evita re-descargar)."""
    os.makedirs(CACHE_5M, exist_ok=True)
    ruta = f"{CACHE_5M}/{ticker}.parquet"
    if os.path.exists(ruta):
        return pd.read_parquet(ruta)
    cm = cm or _cm()
    df = massive.barras_intradia(ticker, "5Min", DESDE, HASTA, solo_regular=True, cm=cm)
    if len(df):
        df.to_parquet(ruta)
    return df


# ----------------------------- metricas -----------------------------
def _ic95_media(x, B=5000, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def perf_diaria(ret, per=252):
    """Metricas de una serie de retornos DIARIOS de estrategia (fraccion, no R)."""
    r = pd.Series(ret).dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (per / len(r)) - 1
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * np.sqrt(per) if sd > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "maxDD": float(dd),
            "pct_pos": float((r > 0).mean()), "n": len(r)}


def perf_trades(r_trade, trades_por_anio=252):
    """Metricas de una serie de retornos POR OPERACION (en R o en fraccion)."""
    r = pd.Series(r_trade).dropna()
    if len(r) < 10:
        return {}
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * np.sqrt(trades_por_anio) if sd > 0 else float("nan")
    return {"exp": float(r.mean()), "sharpe_ann": float(sharpe),
            "hit": float((r > 0).mean()), "n": int(len(r))}


# ----------------------------- A) OVERNIGHT vs INTRADAY -----------------------------
def overnight_intraday(eod_data, slip_bps=SLIP_BPS):
    """Descompone el retorno de cada ETF en NOCHE (close->open) y DIA (open->close) y mide
    ambas como estrategias operables netas de coste. La noche: comprar MOC, vender MOO."""
    res = {}
    for t, df in eod_data.items():
        o = df["open"].astype(float)
        c = df["close"].astype(float)
        on = (o / c.shift(1) - 1).dropna()            # overnight (close prev -> open)
        idr = (c / o - 1).reindex(on.index)           # intraday (open -> close)
        bh = (c / c.shift(1) - 1).reindex(on.index)   # buy&hold
        cost = 2 * slip_bps / 1e4                      # noche y dia: 2 cruces cada uno
        res[t] = {"on_net": on - cost, "id_net": idr - cost, "bh": bh,
                  "on_gross": on, "id_gross": idr}
    return res


# ----------------------------- B) ORB de INDICE -----------------------------
def orb_indice(df5, or_bars=6, entrada_hasta="14:00", lado="both", slip_bps=SLIP_BPS):
    """Opening-range breakout a nivel de indice. OR = primeras or_bars velas de 5 min
    (09:30-10:00 con or_bars=6). Entrada en ruptura CONFIRMADA (cierre de vela > OR_high
    para largo / < OR_low para corto) hasta 'entrada_hasta'; stop al lado opuesto del rango;
    salida al cierre de sesion (stop evaluado intrabar, conservador). Devuelve retorno por
    operacion en R (R = ancho del rango de apertura). Una entrada por dia y lado."""
    df5 = df5.copy()
    df5["dia"] = df5.index.date
    trades = []
    hhmm = pd.Timestamp(entrada_hasta).time()
    for dia, g in df5.groupby("dia"):
        g = g.sort_index()
        if len(g) < or_bars + 5:
            continue
        orb = g.iloc[:or_bars]
        hi, lo = float(orb["high"].max()), float(orb["low"].min())
        rango = hi - lo
        if rango <= 0:
            continue
        post = g.iloc[or_bars:]
        cierre_eod = float(g["close"].iloc[-1])
        for direc in (["long", "short"] if lado == "both" else [lado]):
            ventana = post[post.index.map(lambda x: x.time() <= hhmm)]
            ent_idx = None
            for ts, bar in ventana.iterrows():
                if direc == "long" and float(bar["close"]) > hi:
                    ent_idx = ts; break
                if direc == "short" and float(bar["close"]) < lo:
                    ent_idx = ts; break
            if ent_idx is None:
                continue
            entrada = float(g.loc[ent_idx, "close"])
            resto = g.loc[ent_idx:].iloc[1:]           # desde la vela siguiente a la entrada
            stop = lo if direc == "long" else hi
            salida, stopped = cierre_eod, False
            for ts, bar in resto.iterrows():
                if direc == "long" and float(bar["low"]) <= stop:
                    salida, stopped = stop, True; break
                if direc == "short" and float(bar["high"]) >= stop:
                    salida, stopped = stop, True; break
            bruto = (salida - entrada) if direc == "long" else (entrada - salida)
            # coste: entrada + salida; extra en stop (peor fill)
            coste = (2 * slip_bps / 1e4) * entrada + (slip_bps / 1e4 * entrada if stopped else 0)
            r = (bruto - coste) / rango
            trades.append({"dia": pd.Timestamp(dia), "dir": direc, "R": r, "stopped": stopped})
    return pd.DataFrame(trades)


# ----------------------------- C) MOMENTUM ULTIMA HORA -----------------------------
def ultima_hora(df5, hora_señal="15:00", slip_bps=SLIP_BPS, reversion=False):
    """Si a las 'hora_señal' el dia va arriba (close>open de sesion), largo hasta el cierre;
    si va abajo, corto. reversion=True invierte la señal (fade de la ultima hora). Ret/dia."""
    df5 = df5.copy(); df5["dia"] = df5.index.date
    hh = pd.Timestamp(hora_señal).time()
    signo = -1.0 if reversion else 1.0
    filas = []
    for dia, g in df5.groupby("dia"):
        g = g.sort_index()
        open_ses = float(g["open"].iloc[0])
        en = g[g.index.map(lambda x: x.time() >= hh)]
        if len(en) < 2:
            continue
        p0 = float(en["open"].iloc[0]); pf = float(g["close"].iloc[-1])
        señal = signo * np.sign(p0 / open_ses - 1.0)
        if señal == 0:
            continue
        bruto = señal * (pf / p0 - 1.0)
        filas.append(bruto - 2 * slip_bps / 1e4)
    return pd.Series(filas)


def overnight_trend(df_eod, slip_bps=SLIP_BPS):
    """Overnight (MOC->MOO) con filtro de tendencia: solo mantiene la noche cuando el cierre
    previo esta sobre su SMA200 (risk-on). El resto, en efectivo. Retorno diario."""
    o = df_eod["open"].astype(float); c = df_eod["close"].astype(float)
    sma = c.rolling(200).mean()
    on = (o / c.shift(1) - 1)
    señal = (c.shift(1) > sma.shift(1)).astype(float)       # ayer sobre SMA200 -> hold noche
    r = (señal * (on - 2 * slip_bps / 1e4)).dropna()
    return r[sma.shift(1).notna().reindex(r.index).fillna(False)]


# ----------------------------- D) GAP-FADE en ETF -----------------------------
def gap_fade(df_eod, umbral=0.005, slip_bps=SLIP_BPS):
    """Si el ETF abre con hueco |open/close_prev-1| > umbral, se CONTRARIA hacia el cierre:
    short el gap-up / long el gap-down, salida al cierre del dia. Retorno por operacion."""
    o = df_eod["open"].astype(float); c = df_eod["close"].astype(float)
    gap = (o / c.shift(1) - 1).dropna()
    filas = []
    for d in gap.index:
        g = gap.loc[d]
        if abs(g) < umbral:
            continue
        señal = -np.sign(g)                       # fade
        bruto = señal * (c.loc[d] / o.loc[d] - 1.0)
        filas.append(bruto - 2 * slip_bps / 1e4)
    return pd.Series(filas)


# ----------------------------- orquestacion -----------------------------
def _split(x):
    x = pd.Series(x).reset_index(drop=True)
    k = int(len(x) * 0.6)
    return x.iloc[:k], x.iloc[k:]


def _ic95_sharpe(r, per=252, B=5000, seed=SEED):
    r = pd.Series(r).dropna().to_numpy()
    if len(r) < 50:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        s = r[rng.integers(0, len(r), len(r))]
        sd = s.std(ddof=1)
        out.append(s.mean() / sd * np.sqrt(per) if sd > 0 else 0.0)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def validar_overnight_trend(eod_data):
    """Valida a fondo el UNICO hilo con pinta de edge: overnight (MOC->MOO) con filtro de
    tendencia (hold solo si cierre>SMA200). IC95 del Sharpe por bootstrap, split train/test
    y sensibilidad al coste. Honestidad: ¿aguanta o es un espejismo de periodo/coste?"""
    print("\n=== VALIDACION: overnight + filtro de tendencia (SMA200) ===")
    res = {}
    for t in ["QQQ", "SPY", "IWM"]:
        df = eod_data[t]
        o = df["open"].astype(float); c = df["close"].astype(float)
        sma = c.rolling(200).mean()
        on = (o / c.shift(1) - 1)
        hold = (c.shift(1) > sma.shift(1))
        base = on[hold.fillna(False) & sma.shift(1).notna()]
        bh = (c / c.shift(1) - 1).dropna()
        # sensibilidad al coste (bp por lado; overnight = 2 cruces)
        cost_rows = {}
        for bp in (0.5, 1.0, 1.5):
            r = base - 2 * bp / 1e4
            cost_rows[bp] = perf_diaria(r)
        r1 = base - 2 * 1.0 / 1e4            # coste central 1bp/lado
        ic = _ic95_sharpe(r1.to_numpy())
        tr, te = _split(r1)
        res[t] = {"cost": cost_rows, "ic_sharpe": ic,
                  "sharpe_train": perf_diaria(tr).get("Sharpe"),
                  "sharpe_test": perf_diaria(te).get("Sharpe"),
                  "dias_en_mercado": float(hold.mean()), "bh": perf_diaria(bh)}
        p = cost_rows[1.0]
        print(f"  {t}: Sharpe {p.get('Sharpe',float('nan')):.2f} (IC95 [{ic[0]:.2f},{ic[1]:.2f}]) | "
              f"CAGR {p.get('CAGR',0)*100:+.1f}% | maxDD {p.get('maxDD',0)*100:.1f}% | "
              f"en mercado {res[t]['dias_en_mercado']*100:.0f}% noches")
        print(f"       train Sharpe {res[t]['sharpe_train']:.2f} / test {res[t]['sharpe_test']:.2f} | "
              f"coste 0.5bp->{cost_rows[0.5].get('Sharpe'):.2f} 1.5bp->{cost_rows[1.5].get('Sharpe'):.2f} | "
              f"B&H Sharpe {res[t]['bh'].get('Sharpe'):.2f}")
    return res


def correr(cm=None):
    cm = cm or _cm()
    print("Descargando EOD de ETFs...", flush=True)
    ed = eod(ETFS_EOD, cm=cm)
    print(f"  {len(ed)} ETFs EOD", flush=True)

    out = {}

    # --- A) overnight vs intraday ---
    oi = overnight_intraday(ed)
    on_all = {t: perf_diaria(oi[t]["on_net"]) for t in oi}
    id_all = {t: perf_diaria(oi[t]["id_net"]) for t in oi}
    bh_all = {t: perf_diaria(oi[t]["bh"]) for t in oi}
    out["overnight"] = {"on": on_all, "id": id_all, "bh": bh_all, "raw": oi}
    print("\n[A] OVERNIGHT (MOC->MOO) vs INTRADAY (open->close), neto de coste:")
    for t in ["SPY", "QQQ", "IWM"]:
        print(f"  {t}: noche Sharpe {on_all[t].get('Sharpe',float('nan')):.2f} CAGR {on_all[t].get('CAGR',0)*100:+.1f}% | "
              f"dia Sharpe {id_all[t].get('Sharpe',float('nan')):.2f} CAGR {id_all[t].get('CAGR',0)*100:+.1f}% | "
              f"B&H Sharpe {bh_all[t].get('Sharpe',float('nan')):.2f} CAGR {bh_all[t].get('CAGR',0)*100:+.1f}%")

    # --- B) ORB de indice (5m) ---
    print("\n[B] ORB de INDICE (5m, ruptura confirmada, stop OR-opuesto, salida EOD):")
    orb_res = {}
    for t in ETFS_5M:
        df5 = intradia_5m(t, cm=cm)
        if not len(df5):
            continue
        tr = orb_indice(df5, lado="both")
        for lado in ["long", "short", "both"]:
            sub = tr if lado == "both" else tr[tr["dir"] == lado]
            r = sub["R"].to_numpy()
            tpy = len(r) / 4.9 if len(r) else 1
            m = perf_trades(r, trades_por_anio=tpy)
            ic = _ic95_media(r)
            tr_tr, tr_te = _split(sub["R"])
            orb_res[(t, lado)] = {"m": m, "ic": ic,
                                  "exp_train": float(tr_tr.mean()) if len(tr_tr) else float("nan"),
                                  "exp_test": float(tr_te.mean()) if len(tr_te) else float("nan")}
            print(f"  {t} {lado:<5}: exp {m.get('exp',0):+.3f}R | hit {m.get('hit',0)*100:.0f}% | "
                  f"n={m.get('n',0)} | IC95 [{ic[0]:+.3f},{ic[1]:+.3f}] | "
                  f"train {orb_res[(t,lado)]['exp_train']:+.3f} test {orb_res[(t,lado)]['exp_test']:+.3f}")
    out["orb"] = orb_res

    # --- C) ultima hora ---
    print("\n[C] MOMENTUM ULTIMA HORA (señal 15:00, continuacion a cierre):")
    uh_res = {}
    for t in ETFS_5M:
        df5 = intradia_5m(t, cm=cm)
        if not len(df5):
            continue
        s = ultima_hora(df5)
        uh_res[t] = {"perf": perf_diaria(s), "exp": float(s.mean()), "ic": _ic95_media(s.to_numpy())}
        print(f"  {t}: exp {uh_res[t]['exp']*100:+.3f}%/dia | Sharpe {uh_res[t]['perf'].get('Sharpe',float('nan')):.2f} | "
              f"IC95 media [{uh_res[t]['ic'][0]*100:+.3f},{uh_res[t]['ic'][1]*100:+.3f}]% | n={uh_res[t]['perf'].get('n',0)}")
    out["ultima_hora"] = uh_res

    # --- D) gap-fade ---
    print("\n[D] GAP-FADE en ETF (fade de hueco >0.5% hacia el cierre):")
    gf_res = {}
    for t in ["SPY", "QQQ", "IWM", "XLK", "XLE"]:
        if t not in ed:
            continue
        s = gap_fade(ed[t])
        gf_res[t] = {"m": perf_trades(s.to_numpy(), trades_por_anio=max(1, len(s) / 4.9)),
                     "ic": _ic95_media(s.to_numpy())}
        print(f"  {t}: exp {gf_res[t]['m'].get('exp',0)*100:+.3f}%/op | hit {gf_res[t]['m'].get('hit',0)*100:.0f}% | "
              f"n={gf_res[t]['m'].get('n',0)} | IC95 [{gf_res[t]['ic'][0]*100:+.3f},{gf_res[t]['ic'][1]*100:+.3f}]%")
    out["gap_fade"] = gf_res

    # ===== FASE 2: refinamientos de los hilos que no estaban del todo muertos =====
    print("\n--- FASE 2 (refinamientos; OJO multiple-testing, exploratorio) ---")
    print("[C2] ULTIMA HORA en REVERSION (fade, no continuacion):")
    uh2 = {}
    for t in ETFS_5M:
        df5 = intradia_5m(t, cm=cm)
        s = ultima_hora(df5, reversion=True)
        uh2[t] = {"perf": perf_diaria(s), "ic": _ic95_media(s.to_numpy())}
        print(f"  {t}: Sharpe {uh2[t]['perf'].get('Sharpe',float('nan')):.2f} | exp {s.mean()*100:+.3f}%/dia | "
              f"IC95 media [{uh2[t]['ic'][0]*100:+.3f},{uh2[t]['ic'][1]*100:+.3f}]%")
    out["ultima_hora_rev"] = uh2

    print("[A2] OVERNIGHT con filtro de tendencia (hold solo si cierre>SMA200):")
    on2 = {}
    for t in ["SPY", "QQQ", "IWM"]:
        s = overnight_trend(ed[t])
        on2[t] = {"perf": perf_diaria(s), "ic": _ic95_media(s[s != 0].to_numpy())}
        print(f"  {t}: Sharpe {on2[t]['perf'].get('Sharpe',float('nan')):.2f} | CAGR {on2[t]['perf'].get('CAGR',0)*100:+.1f}%")
    out["overnight_trend"] = on2

    print("[D2] GAP-FADE barrido de umbral (0.5/1.0/1.5%) en SPY/QQQ/IWM:")
    gf2 = {}
    for t in ["SPY", "QQQ", "IWM"]:
        for u in (0.005, 0.01, 0.015):
            s = gap_fade(ed[t], umbral=u)
            m = perf_trades(s.to_numpy(), trades_por_anio=max(1, len(s) / 4.9))
            ic = _ic95_media(s.to_numpy())
            gf2[(t, u)] = {"m": m, "ic": ic}
            print(f"  {t} >{u*100:.1f}%: exp {m.get('exp',0)*100:+.3f}%/op | n={m.get('n',0)} | "
                  f"IC95 [{ic[0]*100:+.3f},{ic[1]*100:+.3f}]%")
    out["gap_fade_umbral"] = gf2

    out["validacion_on_trend"] = validar_overnight_trend(ed)
    return out


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "all"
    cm = _cm()
    if modo == "fetch":
        eod(ETFS_EOD, cm=cm)
        for t in ETFS_5M:
            d = intradia_5m(t, cm=cm)
            print(f"{t}: {len(d)} barras 5m")
    else:
        correr(cm)
