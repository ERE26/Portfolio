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
  - Sizing (más volumen, Eduardo 2026-10-06): riesgo ORB_RIESGO (1%) por trade; máx ORB_MAX (15)
    simultáneas; tope por nombre ORB_MAX_NOTIONAL (20%, <gate riesgo 25%).
  - RE-ENTRADA al ganador: tras SALIR de un nombre, si hace un nuevo cierre de ruptura por encima
    de la última entrada (+buffer), se re-entra (hasta MAX_REENTRADAS/nombre/día). Captura la cola
    derecha (lo único con EV) sin quitar el TP 2R.
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
# Sizing MÁS AGRESIVO (petición Eduardo 2026-10-06: menos exigente con el tamaño, más volumen).
# Enfocado con el análisis de hoy: la única firma real del ganador es la ruptura CONFIRMADA que
# corre; subimos volumen en ESE motor (no en el momentum muerto). Topes bajo el gate de riesgo
# (REDUCE si posición >25% equity), así que ORB_MAX_NOTIONAL se queda en 20%.
ORB_RIESGO = 0.010        # 1,0% del equity en riesgo por trade (antes 0,6%: más tamaño)
ORB_MAX = 15              # máx posiciones ORB simultáneas (antes 10: más volumen)
ORB_MAX_NOTIONAL = 0.20   # tope de exposición por nombre: 20% del equity (antes 12%; <25% gate riesgo)
VENTANA_MIN = 120         # solo entrar en las primeras 2h tras la apertura
ORB_TP_R = 2.0            # take-profit a 2R (objetivo = 2× el riesgo). Antes: +20% fijo, irreal.
# RE-ENTRADA al ganador en fuerza continuada (Eduardo: "si al vender sigue subiendo, re-comprar").
# Tras SALIR de un nombre (posición cerrada), se permite re-entrar si hace un nuevo cierre de
# ruptura por encima de la última entrada +buffer. Captura la cola derecha (lo único con EV) sin
# quitar el TP (quitarlo era trampa de supervivencia, ver docs/refinamiento_intradia_2026-10-06.md).
MAX_REENTRADAS = 3        # máx entradas por nombre y día (1 inicial + hasta 2 re-montas)
REENTRADA_BUFFER = 0.002  # re-entrada exige nuevo cierre > última entrada +0,2% (sigue subiendo)
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


def _setup_orb(ticker, nivel_ref=None):
    """Devuelve el setup ORB del día para `ticker` o None. Point-in-time: usa solo velas
    ya CERRADAS. RVOL de apertura con 14 días previos.
    nivel_ref=None (entrada inicial): intención = PRIMERA vela que cierra sobre el máx del rango.
    nivel_ref=X (RE-ENTRADA): exige que la ÚLTIMA vela cerrada cierre sobre X (sigue subiendo) y
      sobre el máx del rango; intención = ese cierre reciente (no el breakout inicial)."""
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
    if nivel_ref is None:
        # ENTRADA INICIAL: 1ª vela de hoy (posterior a la 1ª) que CIERRA sobre el máximo del rango.
        for p in hoy_pos[1:]:
            if float(df["close"].iloc[p]) > h0:
                return {"or_high": h0, "or_low": l0, "opening_rvol": orv,
                        "intencion": float(df["close"].iloc[p])}       # cierre que confirma
        return None
    # RE-ENTRADA: la ÚLTIMA vela cerrada debe cerrar sobre nivel_ref (sigue subiendo) y sobre el
    # máximo del rango (tendencia intacta). Intención = ese cierre reciente, no el breakout inicial.
    ult_close = float(df["close"].iloc[hoy_pos[-1]])
    if ult_close > nivel_ref and ult_close > h0:
        return {"or_high": h0, "or_low": l0, "opening_rvol": orv, "intencion": ult_close}
    return None


def ejecutar(confirmar=False):
    mins = _min_desde_apertura()
    if mins < 5 or mins > VENTANA_MIN:
        print(f"ORB: fuera de ventana ({mins:.0f} min desde apertura; se opera 5-{VENTANA_MIN}).")
        return
    df = _log()
    hoy = _ahora_et().strftime("%Y-%m-%d")
    hoy_rows = df[df["ts"].astype(str).str[:10] == hoy] if len(df) else df.iloc[0:0]
    held = set(AT.posiciones())                    # posiciones REALES (fuente de verdad de "aún lo tengo")
    # Lotes ORB GENUINAMENTE abiertos = fila 'abierta' en el log Y todavía en cartera. Un 'abierta'
    # del log que ya NO está en held salió por TP/SL intradía -> cuenta como cerrado (re-entrable).
    abiertas_log = set(df[df["estado"] == "abierta"]["ticker"]) if len(df) else set()
    orb_vivas = abiertas_log & held
    if len(orb_vivas) >= ORB_MAX:
        print(f"ORB: ya {len(orb_vivas)} posiciones vivas (máx {ORB_MAX}).")
        return
    try:
        from fuentes import tradingview as TV
        pool = TV.en_juego(n=60)
    except Exception as e:
        print(f"ORB: sin pool in-play ({str(e)[:50]})"); return
    equity = float(AT.cuenta().get("equity", 0.0))
    nuevos = []
    for t in pool:
        if t in held:                              # aún lo tenemos (ORB u otro libro) -> no duplicar
            continue
        # entradas ORB REALES de hoy en este nombre (abierta/cerrada), para el tope y la fuerza
        t_ok = hoy_rows[(hoy_rows["ticker"] == t) & (hoy_rows["estado"].isin(["abierta", "cerrada"]))]
        n_prev = len(t_ok)                         # 0 = primera entrada; >0 = re-entrada
        if n_prev >= MAX_REENTRADAS:               # tope de re-entradas por nombre/día
            continue
        if n_prev == 0:                            # ENTRADA INICIAL: breakout confirmado del día
            s = _setup_orb(t)
        else:                                      # RE-ENTRADA: nuevo cierre sobre la última entrada
            try:
                max_prev = float(t_ok["intencion"].astype(float).max())
            except Exception:
                max_prev = 0.0
            s = _setup_orb(t, nivel_ref=max_prev * (1 + REENTRADA_BUFFER))
        if s is None:
            continue
        entry = s["intencion"]
        if n_prev > 0:
            print(f"  (re-entrada #{n_prev+1} {t}: cierre {entry:.2f} > última {max_prev:.2f})")
        rdist_raw = entry - s["or_low"]            # riesgo natural: stop en el mínimo del rango
        if rdist_raw <= 0:
            continue
        # TP/SL realistas: stop acotado al 3% máx (AT.INTRADIA_SL_MAX) y objetivo a 2R. En
        # rupturas muy extendidas el mínimo del rango queda lejísimos -> stop más ceñido para
        # que el riesgo (y el TP) sean realistas, en vez de un objetivo del 15-20%.
        tp, sl = AT.niveles_bracket(entry, rdist_raw / entry, r_mult=ORB_TP_R, lado="buy")
        rdist = entry - sl                          # riesgo real por acción tras acotar el stop
        qty = int((equity * ORB_RIESGO) / rdist) if rdist > 0 else 0
        qty = min(qty, int((equity * ORB_MAX_NOTIONAL) / entry))   # tope de exposición por nombre
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
        if len(orb_vivas) + len(nuevos) >= ORB_MAX:
            break
    if nuevos:
        _guardar(pd.concat([df, pd.DataFrame(nuevos)], ignore_index=True))
    elif not confirmar:
        print("  (ningún setup ORB confirmado ahora)")


def proteger(confirmar=False):
    """Sube el stop a BREAKEVEN en las posiciones ORB que ya ganan >= +1R (el riesgo inicial
    del log). Así un ganador no puede convertirse en perdedor ('mejor beneficio'). PAPER."""
    df = _log()
    abiertas = df[df["estado"] == "abierta"] if len(df) else df
    if not len(abiertas):
        print("ORB proteger: sin posiciones abiertas en el log."); return
    try:
        pos = AT.posiciones()
        ordenes = AT.ordenes_abiertas()
    except Exception as e:
        print(f"ORB proteger: error cuenta ({str(e)[:50]})"); return
    # stop vivo actual por ticker (para no BAJAR nunca un stop ya colocado)
    stop_vivo = {}
    for o in ordenes:
        if (o.get("side") == "sell" and o.get("type") in ("stop", "stop_limit")
                and o.get("stop_price")):
            stop_vivo[o["symbol"]] = float(o["stop_price"])
    movidos = 0
    for _, r in abiertas.groupby("ticker").last().reset_index().iterrows():
        t = r["ticker"]
        if t not in pos:
            continue
        try:
            entry = float(r["intencion"]); stop0 = float(r["stop"])
        except Exception:
            continue
        rdist = entry - stop0
        if rdist <= 0:
            continue
        ent_real = pos[t]["entrada"]; px = pos[t]["precio"]
        if px < ent_real + rdist:                      # aún no +1R
            continue
        nuevo = round(ent_real * 1.001, 2)             # breakeven + 0,1%
        if nuevo <= stop_vivo.get(t, -1e9):            # nunca bajar un stop ya protegido
            continue
        res = AT.mover_stop(t, nuevo, confirmar=confirmar)
        if res.get("movido") or res.get("dry_run"):
            movidos += 1
            print(f"  {t}: +1R -> stop a breakeven {nuevo} ({'enviado' if confirmar else 'dry'})")
    if not movidos:
        print("ORB proteger: nada que mover (ninguna en +1R o ya protegidas).")


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
    elif "--proteger" in sys.argv:
        proteger(confirmar="--enviar" in sys.argv)
    elif "--cerrar" in sys.argv:
        cerrar_eod(confirmar="--enviar" in sys.argv)
    else:
        ejecutar(confirmar="--enviar" in sys.argv)
        reconciliar()
