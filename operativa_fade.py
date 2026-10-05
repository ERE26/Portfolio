"""LIBRO GAP-FADE en PAPER — EXPERIMENTO de "alto acierto / baja frecuencia de pérdida".

AVISO HONESTO: el backtest dice que esto NO gana (Sharpe negativo). Es el perfil que pidió
Eduardo: la de MÁS acierto pero que PIERDE poco a menudo — se consigue con objetivo CERCA (TP
pequeño, mucha probabilidad de tocarlo) y stop LEJOS (raro, pero cuando salta, grande). Alto %
de acierto, expectancy negativa (las pocas pérdidas grandes superan a las muchas ganancias
pequeñas). Se corre en PAPER solo para VERLO en vivo, no porque tenga edge.

Setup: en un stock en juego con GAP de apertura grande, FADE (contrariar): gap UP -> corto;
gap DOWN -> largo. Bracket: TP cerca (fade parcial), SL lejos. Tamaño de medición. Cierre EOD
lo gestiona el bucle (cerrar_intradia) — este libro usa brackets, así que Alpaca cierra solo al
tocar TP/SL. PAPER siempre (claves PK).

  python operativa_fade.py --gestionar [--enviar]    # abre fades de los gappers en juego
  python operativa_fade.py --reconciliar
"""
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca
from fuentes import alpaca_trade as AT

LOG = "datos/fade_paper.csv"
GAP_MIN = 0.04            # |gap| de apertura mínimo para fadear (4%)
TP_PCT = 0.012           # objetivo de fade (cerca -> alto acierto)
SL_PCT = 0.045           # stop lejos (raro, pero grande -> expectancy negativa)
NOTIONAL = 1000.0        # $ por trade (medición)
MAX_POS = 6
CAMPOS = ["ts", "ticker", "gap", "lado", "qty", "entrada", "tp", "sl", "order_id",
          "fill", "slippage_pct", "estado"]


def _ahora_et():
    return pd.Timestamp.now(tz="America/New_York")


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


def _gap(ticker):
    """Gap de apertura = open de hoy / cierre de ayer - 1, y precio actual. None si no hay."""
    try:
        d = alpaca.diario(ticker, desde=(pd.Timestamp.now() - pd.Timedelta(days=12)).strftime("%Y-%m-%d"))
    except Exception:
        return None
    if len(d) < 2:
        return None
    hoy = _ahora_et().normalize().tz_localize(None)
    if d.index[-1].normalize() != hoy:          # aún no hay barra de hoy
        return None
    op_hoy = float(d["open"].iloc[-1]); cl_ayer = float(d["close"].iloc[-2])
    if cl_ayer <= 0:
        return None
    try:
        px = AT.precio_actual([ticker]).get(ticker, op_hoy)
    except Exception:
        px = op_hoy
    return {"gap": op_hoy / cl_ayer - 1.0, "precio": float(px)}


def gestionar(confirmar=False):
    et = _ahora_et()
    mins = (et - (et.normalize() + pd.Timedelta(hours=9, minutes=30))).total_seconds() / 60.0
    print(f"FADE | {et:%Y-%m-%d %H:%M} ET")
    if mins < 5 or mins > 150:
        print(f"  fuera de ventana ({mins:.0f} min desde apertura; opero 5-150)."); return
    df = _log()
    hoy = et.strftime("%Y-%m-%d")
    ya_hoy = set(df[df["ts"].astype(str).str[:10] == hoy]["ticker"]) if len(df) else set()
    try:
        held = set(AT.posiciones())
        abiertas = set(o["symbol"] for o in AT.ordenes_abiertas())
    except Exception as e:
        print(f"  error cuenta ({str(e)[:40]})"); return
    n_abiertas = len(df[df["estado"] == "enviada"]) if len(df) else 0
    try:
        from fuentes import tradingview as TV
        pool = TV.en_juego(n=50)
    except Exception as e:
        print(f"  sin pool in-play ({str(e)[:40]})"); return
    nuevas = []
    for t in pool:
        if n_abiertas + len(nuevas) >= MAX_POS:
            break
        if t in held or t in ya_hoy or t in abiertas:
            continue
        g = _gap(t)
        if g is None or abs(g["gap"]) < GAP_MIN:
            continue
        px = g["precio"]
        if px <= 0:
            continue
        qty = int(NOTIONAL / px)
        if qty <= 0:
            continue
        if g["gap"] > 0:                         # gap up -> FADE corto
            lado, tp, sl = "sell", round(px * (1 - TP_PCT), 2), round(px * (1 + SL_PCT), 2)
        else:                                    # gap down -> FADE largo
            lado, tp, sl = "buy", round(px * (1 + TP_PCT), 2), round(px * (1 - SL_PCT), 2)
        fila = {"ts": et.isoformat(), "ticker": t, "gap": round(g["gap"], 4), "lado": lado,
                "qty": qty, "entrada": round(px, 2), "tp": tp, "sl": sl, "order_id": "",
                "fill": np.nan, "slippage_pct": np.nan, "estado": "pend"}
        estado_msg = "[dry]"
        if confirmar:
            try:
                r = AT.enviar_bracket(t, qty, tp, sl, lado=lado, confirmar=True, libro="fade")
                fila["order_id"] = r.get("id", "")
                fila["estado"] = "enviada" if r.get("id") else "error"
                estado_msg = fila["estado"]
            except Exception as e:
                # low-floats suelen ser HARD-TO-BORROW: el corto se rechaza. Hallazgo real.
                fila["estado"] = "rechazada"
                estado_msg = f"rechazada ({str(e)[:40]})"
        print(f"  {estado_msg} FADE {('CORTO' if lado=='sell' else 'LARGO')} "
              f"{t} x{qty} gap {g['gap']*100:+.1f}% @~{px:.2f} TP {tp} SL {sl}")
        nuevas.append(fila)
    if confirmar and nuevas:
        _guardar(pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True))
    elif not nuevas:
        print("  (ningún gapper en juego con |gap|>=4% ahora)")


def reconciliar():
    df = _log()
    if not len(df):
        return
    for i, r in df.iterrows():
        if r.get("order_id") and (pd.isna(r.get("fill")) or r.get("fill") == ""):
            try:
                fav = AT.orden(str(r["order_id"])).get("filled_avg_price")
                if fav:
                    fill = float(fav); df.at[i, "fill"] = fill
                    ent = float(r["entrada"]) if r.get("entrada") == r.get("entrada") else np.nan
                    if ent and ent > 0:
                        df.at[i, "slippage_pct"] = (fill - ent) / ent
                    df.at[i, "estado"] = "hecha"
            except Exception:
                continue
    _guardar(df)
    hechas = df[df["fill"].notna()]
    if len(hechas):
        s = hechas["slippage_pct"].astype(float).dropna()
        print(f"FADE reconciliado | fills {len(hechas)} | slippage medio {s.mean()*100:+.3f}%")


if __name__ == "__main__":
    if "--reconciliar" in sys.argv:
        reconciliar()
    elif "--gestionar" in sys.argv:
        gestionar(confirmar="--enviar" in sys.argv)
    else:
        print(__doc__)
