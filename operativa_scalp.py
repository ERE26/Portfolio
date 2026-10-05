"""LIBRO SCALP en PAPER — MUCHAS transacciones, beneficios PEQUEÑOS (lo que pidió Eduardo).

AVISO HONESTO (importante): "muchas operaciones + beneficio pequeño" a coste TAKER es la forma
más rápida de que el spread se lo coma todo — cada micro-beneficio (+0,4%) es <= al spread que
pagas DOS veces (entrada+salida). Por eso aquí la entrada es MAKER: una orden LÍMITE que
descansa en el libro y, si el precio viene a ella, cruza cobrando ~medio spread en vez de
pagarlo. Es la ÚNICA vía por la que el scalping de beneficios pequeños puede no perder, y es lo
que hacen los bots de HFT. Se corre en PAPER para MEDIR en vivo si cruza el coste o no:
  - Entrada MAKER: límite 0,1% por debajo del último (descansa; si no cruza, caduca en el día).
  - TP +0,4% (beneficio pequeño) / SL -0,5%, como bracket OCO (Alpaca cierra solo).
  - Universo: in-play líquidos (TradingView). Tamaño pequeño, muchas a la vez.
  - El log mide el NETO real (fills de entrada y de salida) y el % de órdenes que llegan a cruzar
    (fill-rate del maker) — la métrica que decide si esto es viable.
No colisiona con otros libros (gestionar.py excluye 'scalp'). PAPER siempre (claves PK).

  python operativa_scalp.py --gestionar [--enviar]   # coloca brackets límite (maker) en in-play
  python operativa_scalp.py --reconciliar            # fills, neto, acierto, fill-rate
"""
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca_trade as AT

LOG = "datos/scalp_paper.csv"
NOTIONAL = 500.0          # pequeño: muchas posiciones sin concentrar
TP_PCT = 0.004            # +0,4% objetivo (beneficio pequeño)
SL_PCT = 0.005            # -0,5% stop
MAKER_OFF = 0.001         # entrada límite 0,1% por debajo del último (descansar como maker)
MAX_POS = 12              # muchas a la vez
VENTANA = (5, 375)        # casi toda la sesión (minutos desde apertura)
CAMPOS = ["ts", "ticker", "lado", "qty", "limite", "tp", "sl", "order_id", "fill_entrada",
          "fill_salida", "neto_pct", "estado"]


def _ahora_et():
    return pd.Timestamp.now(tz="America/New_York")


def _min_apertura(now=None):
    now = now or _ahora_et()
    ap = now.normalize() + pd.Timedelta(hours=9, minutes=30)
    return (now - ap).total_seconds() / 60.0


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


def gestionar(confirmar=False):
    et = _ahora_et()
    mins = _min_apertura(et)
    print(f"SCALP | {et:%Y-%m-%d %H:%M} ET")
    if mins < VENTANA[0] or mins > VENTANA[1]:
        print(f"  fuera de ventana ({mins:.0f} min)."); return
    df = _log()
    hoy = et.strftime("%Y-%m-%d")
    ya_hoy = set(df[df["ts"].astype(str).str[:10] == hoy]["ticker"]) if len(df) else set()
    try:
        held = set(AT.posiciones())
        abiertas = set(o["symbol"] for o in AT.ordenes_abiertas())
    except Exception as e:
        print(f"  error cuenta ({str(e)[:40]})"); return
    n_vivas = len(df[df["estado"] == "enviada"]) if len(df) else 0
    if n_vivas >= MAX_POS:
        print(f"  ya {n_vivas} órdenes scalp vivas (máx {MAX_POS})."); return
    try:
        from fuentes import tradingview as TV
        pool = TV.en_juego(n=40)
    except Exception as e:
        print(f"  sin pool in-play ({str(e)[:40]})"); return
    try:
        precios = AT.precio_actual(list(pool))
    except Exception:
        precios = {}
    nuevas = []
    for t in pool:
        if n_vivas + len(nuevas) >= MAX_POS:
            break
        if t in held or t in abiertas or t in ya_hoy:
            continue
        px = float(precios.get(t, 0.0))
        if px <= 1.0:                         # evita sub-1$ (spread relativo enorme)
            continue
        limite = round(px * (1 - MAKER_OFF), 2)   # entrada maker: por debajo del último
        if limite <= 0:
            continue
        qty = int(NOTIONAL / limite)
        if qty <= 0:
            continue
        tp = round(limite * (1 + TP_PCT), 2)
        sl = round(limite * (1 - SL_PCT), 2)
        if tp <= limite or sl >= limite:
            continue
        fila = {"ts": et.isoformat(), "ticker": t, "lado": "buy", "qty": qty, "limite": limite,
                "tp": tp, "sl": sl, "order_id": "", "fill_entrada": np.nan,
                "fill_salida": np.nan, "neto_pct": np.nan, "estado": "pend"}
        estado = "[dry]"
        if confirmar:
            try:
                r = AT.enviar_bracket_maker(t, qty, limite, tp, sl, lado="buy",
                                            confirmar=True, libro="scalp")
                fila["order_id"] = r.get("id", "")
                fila["estado"] = "enviada" if r.get("id") else "error"
                estado = fila["estado"]
            except Exception as e:
                fila["estado"] = "rechazada"
                estado = f"rechazada ({str(e)[:40]})"
        print(f"  {estado} SCALP {t} x{qty} maker@{limite} TP {tp}(+{TP_PCT*100:.1f}%) SL {sl}")
        nuevas.append(fila)
    if confirmar and nuevas:
        _guardar(pd.concat([df, pd.DataFrame(nuevas)], ignore_index=True))
    elif not nuevas:
        print("  (ningún in-play nuevo para scalpear ahora)")


def reconciliar():
    """Rellena fill de entrada (el maker cruzó o no) y, si la posición ya cerró, el fill de salida
    y el NETO. Reporta fill-rate del maker, neto medio y acierto — las métricas que deciden."""
    df = _log()
    if not len(df):
        print("SCALP: log vacío."); return
    for i, r in df.iterrows():
        oid = str(r.get("order_id") or "")
        if not oid or oid == "nan":
            continue
        try:
            o = AT.orden(oid)
        except Exception:
            continue
        fav = o.get("filled_avg_price")
        if fav and (pd.isna(r.get("fill_entrada")) or r.get("fill_entrada") == ""):
            df.at[i, "fill_entrada"] = float(fav)        # el maker CRUZÓ
        # salida: mira las patas del bracket (legs) por si TP/SL ya rellenó
        for lg in (o.get("legs") or []):
            if lg.get("status") == "filled" and lg.get("filled_avg_price"):
                df.at[i, "fill_salida"] = float(lg["filled_avg_price"])
                if r.get("fill_entrada") == r.get("fill_entrada") and float(r["fill_entrada"]) > 0:
                    df.at[i, "neto_pct"] = (float(lg["filled_avg_price"]) / float(r["fill_entrada"]) - 1)
                elif df.at[i, "fill_entrada"] == df.at[i, "fill_entrada"]:
                    df.at[i, "neto_pct"] = (float(lg["filled_avg_price"]) / float(df.at[i, "fill_entrada"]) - 1)
                df.at[i, "estado"] = "cerrada"
    _guardar(df)
    enviadas = df[df["order_id"].astype(str).str.len() > 5]
    cruzadas = df[df["fill_entrada"].notna()]
    cerradas = df[df["neto_pct"].notna()]
    fill_rate = (len(cruzadas) / len(enviadas) * 100) if len(enviadas) else 0.0
    msg = f"SCALP | órdenes {len(enviadas)} | maker fill-rate {fill_rate:.0f}% ({len(cruzadas)} cruzaron)"
    if len(cerradas):
        n = cerradas["neto_pct"].astype(float)
        msg += f" | cerradas {len(cerradas)} | neto medio {n.mean()*100:+.3f}% | acierto {(n>0).mean()*100:.0f}%"
    print(msg)


if __name__ == "__main__":
    if "--reconciliar" in sys.argv:
        reconciliar()
    elif "--gestionar" in sys.argv:
        gestionar(confirmar="--enviar" in sys.argv)
    else:
        print(__doc__)
