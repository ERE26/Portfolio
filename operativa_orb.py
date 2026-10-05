"""FASE 0 — ORB en PAPER (medir la ejecución real).

Opera en PAPER el ORB-5min-confirmado validado en backtest, para medir LO ÚNICO que el
backtest no puede dar: el SLIPPAGE REAL del breakout (intención vs fill). Reglas (= las del
backtest ganador):
  - Universo: stocks in play (TradingView en_juego, mid-caps líquidas).
  - Rango de apertura = 1ª vela de 5 min (09:30-09:35 ET). 1ª vela ALCISTA + RVOL de
    apertura > 1 (vol 1ª vela 5m / media 14 días).
  - Entrada: cuando una vela de 5m CIERRA sobre el máximo del rango (ruptura CONFIRMADA),
    dentro de las primeras 2h. Bracket REALISTA: stop al mínimo del rango pero acotado al
    3% máx (más ceñido en rupturas ya extendidas) y take-profit a 2R (2× el riesgo).
  - Si no toca TP ni SL, se mantiene hasta el CIERRE (EOD): se cierran a <5 min del cierre.
  - Sizing: riesgo fijo ORB_RIESGO del equity por trade; máx ORB_MAX simultáneas.
  - REGISTRA intención (cierre de la vela que confirma) vs fill real -> slippage, en
    datos/orb_paper.csv. Ese es el entregable de la Fase 0.

NO toca el resto del sistema (swing, libros C/D). Corre junto al loop; el intradía de
momentum (muerto) se puede desactivar aparte. PAPER siempre (claves PK).

  python operativa_orb.py [--enviar]        # sin --enviar = dry-run
  python operativa_orb.py --cerrar [--enviar]   # cierre EOD
  python operativa_orb.py --reconciliar     # rellena fills/slippage en el log
"""
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca
from fuentes import alpaca_trade as AT

LOG = "datos/orb_paper.csv"
ORB_RIESGO = 0.005        # 0.5% del equity en riesgo por trade
ORB_MAX = 10              # máx posiciones ORB simultáneas
VENTANA_MIN = 120         # solo entrar en las primeras 2h tras la apertura
ORB_TP_R = 2.0            # take-profit a 2R (objetivo = 2× el riesgo). Antes: +20% fijo, irreal.
CAMPOS = ["ts", "ticker", "or_high", "or_low", "intencion", "qty", "stop", "opening_rvol",
          "order_id", "fill", "slippage_pct", "salida", "pnl", "estado"]


def _ahora_et():
    return pd.Timestamp.now(tz="America/New_York")


def _min_desde_apertura(now=None):
    now = now or _ahora_et()
    apertura = now.normalize() + pd.Timedelta(hours=9, minutes=30)
    return (now - apertura).total_seconds() / 60.0


def _log():
    if os.path.exists(LOG):
        try:
            return pd.read_csv(LOG)
        except Exception:
            pass
    return pd.DataFrame(columns=CAMPOS)


def _guardar(df):
    os.makedirs("datos", exist_ok=True)
    df.to_csv(LOG, index=False)


def _setup_orb(ticker):
    """Devuelve el setup ORB del día para `ticker` o None. Point-in-time: usa solo velas
    ya CERRADAS. RVOL de apertura con 14 días previos."""
    hoy = _ahora_et().strftime("%Y-%m-%d")
    desde = (_ahora_et() - pd.Timedelta(days=28)).strftime("%Y-%m-%d")
    try:
        df = alpaca.historico(ticker, "5Min", desde=desde, max_paginas=8)
    except Exception:
        return None
    if len(df) < 100:
        return None
    idx = pd.DatetimeIndex(df.index)          # hora de NY (naive)
    dia = idx.normalize()
    hoy_d = pd.Timestamp(hoy)
    hoy_mask = dia == hoy_d
    if hoy_mask.sum() < 2:                     # aún no hay 2ª vela cerrada
        return None
    hoy_pos = np.where(hoy_mask)[0]
    p0 = hoy_pos[0]
    o0, h0, l0, c0 = (float(df["open"].iloc[p0]), float(df["high"].iloc[p0]),
                      float(df["low"].iloc[p0]), float(df["close"].iloc[p0]))
    if c0 <= o0:                               # 1ª vela debe ser alcista (sesgo long)
        return None
    # RVOL de apertura: vol 1ª vela hoy / media de la 1ª vela los 14 días previos
    primeras = pd.Series(df["volume"].to_numpy(float), index=df.index).groupby(dia).first()
    base = primeras[primeras.index < hoy_d].tail(14)
    if len(base) < 5:
        return None
    orv = float(primeras.get(hoy_d, np.nan) / base.mean()) if base.mean() > 0 else np.nan
    if not (orv == orv) or orv < 1.0:
        return None
    # ¿ha CERRADO ya una vela de hoy (posterior a la 1ª) por encima del máximo del rango?
    for p in hoy_pos[1:]:
        if float(df["close"].iloc[p]) > h0:
            return {"or_high": h0, "or_low": l0, "opening_rvol": orv,
                    "intencion": float(df["close"].iloc[p])}       # cierre que confirma
    return None


def ejecutar(confirmar=False):
    mins = _min_desde_apertura()
    if mins < 5 or mins > VENTANA_MIN:
        print(f"ORB: fuera de ventana ({mins:.0f} min desde apertura; se opera 5-{VENTANA_MIN}).")
        return
    df = _log()
    hoy = _ahora_et().strftime("%Y-%m-%d")
    ya_hoy = set(df[(df["ts"].astype(str).str[:10] == hoy)]["ticker"]) if len(df) else set()
    held = set(AT.posiciones())
    abiertas_orb = set(df[df["estado"] == "abierta"]["ticker"]) if len(df) else set()
    if len(abiertas_orb) >= ORB_MAX:
        print(f"ORB: ya {len(abiertas_orb)} posiciones (máx {ORB_MAX}).")
        return
    try:
        from fuentes import tradingview as TV
        pool = TV.en_juego(n=60)
    except Exception as e:
        print(f"ORB: sin pool in-play ({str(e)[:50]})"); return
    equity = float(AT.cuenta().get("equity", 0.0))
    nuevos = []
    for t in pool:
        if t in held or t in ya_hoy or t in abiertas_orb:
            continue
        s = _setup_orb(t)
        if s is None:
            continue
        entry = s["intencion"]
        rdist_raw = entry - s["or_low"]            # riesgo natural: stop en el mínimo del rango
        if rdist_raw <= 0:
            continue
        # TP/SL realistas: stop acotado al 3% máx (AT.INTRADIA_SL_MAX) y objetivo a 2R. En
        # rupturas muy extendidas el mínimo del rango queda lejísimos -> stop más ceñido para
        # que el riesgo (y el TP) sean realistas, en vez de un objetivo del 15-20%.
        tp, sl = AT.niveles_bracket(entry, rdist_raw / entry, r_mult=ORB_TP_R, lado="buy")
        rdist = entry - sl                          # riesgo real por acción tras acotar el stop
        qty = int((equity * ORB_RIESGO) / rdist) if rdist > 0 else 0
        if qty <= 0:
            continue
        fila = {"ts": _ahora_et().isoformat(), "ticker": t, "or_high": s["or_high"],
                "or_low": s["or_low"], "intencion": entry, "qty": qty, "stop": sl,
                "opening_rvol": round(s["opening_rvol"], 2), "order_id": "", "fill": np.nan,
                "slippage_pct": np.nan, "salida": np.nan, "pnl": np.nan, "estado": "pend"}
        if confirmar:
            r = AT.enviar_bracket(t, qty, tp, sl, confirmar=True, libro="orb")
            fila["order_id"] = r.get("id", "")
            fila["estado"] = "abierta" if r.get("id") else "error"
            print(f"  ORB LARGO {t} x{qty} @~{entry:.2f} (rango {s['or_low']:.2f}-{s['or_high']:.2f}, "
                  f"RVOL_ap {s['opening_rvol']:.1f})  -> {fila['estado']}")
        else:
            print(f"  [dry] ORB {t} x{qty} @~{entry:.2f} stop {sl:.2f} RVOL_ap {s['opening_rvol']:.1f}")
        nuevos.append(fila)
        if len(abiertas_orb) + len(nuevos) >= ORB_MAX:
            break
    if nuevos:
        _guardar(pd.concat([df, pd.DataFrame(nuevos)], ignore_index=True))
    elif not confirmar:
        print("  (ningún setup ORB confirmado ahora)")


def reconciliar():
    """Rellena el fill real y el slippage de las entradas (intención vs filled_avg_price)."""
    df = _log()
    if not len(df):
        return
    for i, r in df.iterrows():
        if r.get("order_id") and (pd.isna(r.get("fill")) or r.get("fill") == ""):
            try:
                o = AT.orden(str(r["order_id"]))
                fav = o.get("filled_avg_price")
                if fav:
                    fill = float(fav)
                    df.at[i, "fill"] = fill
                    df.at[i, "slippage_pct"] = (fill - float(r["intencion"])) / float(r["intencion"])
            except Exception:
                continue
    _guardar(df)
    hechos = df[df["fill"].notna()]
    if len(hechos):
        sl = hechos["slippage_pct"].astype(float)
        print(f"ORB reconciliado: {len(hechos)} fills | slippage medio {sl.mean()*100:+.3f}% | "
              f"mediano {sl.median()*100:+.3f}%")


def cerrar_eod(confirmar=False):
    """Cierra todas las posiciones ORB abiertas (mantener-hasta-el-cierre) y registra P&L."""
    df = _log()
    abiertas = df[df["estado"] == "abierta"]["ticker"].tolist() if len(df) else []
    pos = AT.posiciones()
    vivas = [t for t in abiertas if t in pos]
    if not vivas:
        print("ORB EOD: sin posiciones ORB que cerrar.")
        return
    pl = {t: pos[t]["pl"] for t in vivas}
    if confirmar:
        AT.cerrar_posiciones(vivas, confirmar=True)
    for i, r in df.iterrows():
        if r["ticker"] in vivas and r["estado"] == "abierta":
            df.at[i, "estado"] = "cerrada" if confirmar else "abierta"
            df.at[i, "pnl"] = pl.get(r["ticker"])
            df.at[i, "salida"] = pos[r["ticker"]]["precio"] if "precio" in pos[r["ticker"]] else np.nan
    if confirmar:
        _guardar(df)
    print(f"ORB EOD: {'cerradas' if confirmar else '[dry] cerraría'} {vivas} | P&L {pl}")


if __name__ == "__main__":
    if "--reconciliar" in sys.argv:
        reconciliar()
    elif "--cerrar" in sys.argv:
        cerrar_eod(confirmar="--enviar" in sys.argv)
    else:
        ejecutar(confirmar="--enviar" in sys.argv)
        reconciliar()
