"""Conector de Alpaca PAPER TRADING (cuenta demo), para operar la estrategia en
simulado. Bloqueado al endpoint paper por seguridad: NUNCA toca dinero real.

  - Lee las claves de datos/.alpaca_key (dos lineas: key id, secret). Las claves
    paper empiezan por 'PK' (las reales por 'AK'); si la clave no es paper, se
    rechaza para no operar en real por accidente.
  - Solo market orders 'day'. Enviar ordenes requiere confirmar=True explicito.

Nunca se suben las claves al repositorio (datos/ esta en .gitignore).
"""
import os
import uuid

import requests

PAPER = "https://paper-api.alpaca.markets"


def _claves(kid=None, sec=None):
    kid = kid or os.environ.get("APCA_API_KEY_ID")
    sec = sec or os.environ.get("APCA_API_SECRET_KEY")
    if not (kid and sec):
        for ruta in ("datos/.alpaca_key", os.path.expanduser("~/.alpaca_key")):
            if os.path.exists(ruta):
                partes = open(ruta).read().split()
                if len(partes) >= 2:
                    kid, sec = partes[0], partes[1]
                    break
    if not (kid and sec):
        raise RuntimeError("Faltan las claves de Alpaca (datos/.alpaca_key)")
    if not kid.startswith("PK"):
        raise RuntimeError("La clave no es de PAPER (deberia empezar por 'PK'). "
                           "Por seguridad no se opera en real.")
    return kid, sec


def _headers():
    kid, sec = _claves()
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}


def _url(ruta):
    base = PAPER
    assert "paper-api" in base, "Solo se permite el endpoint de paper trading."
    return f"{base}/v2/{ruta.lstrip('/')}"


def cuenta():
    """Estado de la cuenta paper (equity, buying_power, status)."""
    r = requests.get(_url("account"), headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def posiciones():
    """Posiciones abiertas: dict ticker -> {qty, market_value, unrealized_pl}."""
    r = requests.get(_url("positions"), headers=_headers(), timeout=30)
    r.raise_for_status()
    return {p["symbol"]: {"qty": float(p["qty"]), "valor": float(p["market_value"]),
                          "pl": float(p["unrealized_pl"])} for p in r.json()}


def ordenes_abiertas():
    r = requests.get(_url("orders"), headers=_headers(), params={"status": "open"}, timeout=30)
    r.raise_for_status()
    return r.json()


def precio_actual(tickers):
    """Ultimo precio (trade) de varios tickers desde el feed de datos."""
    from .alpaca import _claves as _ck
    kid, sec = _ck()
    h = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    sym = ",".join(tickers)
    r = requests.get("https://data.alpaca.markets/v2/stocks/trades/latest",
                     params={"symbols": sym, "feed": "iex"}, headers=h, timeout=30)
    r.raise_for_status()
    return {k: v["p"] for k, v in r.json().get("trades", {}).items()}


def enviar_orden(ticker, qty, lado, confirmar=False, tipo="market", libro=None):
    """Envia UNA orden a la cuenta paper. Requiere confirmar=True (salvaguarda).

    qty: numero de acciones (entero, >0). lado: 'buy' o 'sell'.
    libro: etiqueta del libro que origina la orden ('intradia'|'swing'|'tsmom'...).
      Se graba en el client_order_id ('<libro>-<id>'), lo que permite ATRIBUIR cada
      posicion/operacion a su libro despues (capa de riesgo por libro). No altera la
      ejecucion; solo anota la autoria.
    Con confirmar=False devuelve la orden que se ENVIARIA, sin enviarla (dry-run).
    """
    qty = int(qty)
    if qty <= 0 or lado not in ("buy", "sell"):
        return {"omitida": True, "ticker": ticker, "qty": qty, "lado": lado}
    cuerpo = {"symbol": ticker, "qty": str(qty), "side": lado,
              "type": tipo, "time_in_force": "day"}
    if libro:
        cuerpo["client_order_id"] = f"{libro}-{uuid.uuid4().hex[:16]}"
    if not confirmar:
        return {"dry_run": True, **cuerpo}
    r = requests.post(_url("orders"), json=cuerpo, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def historia_cuenta(period="3M", timeframe="1D"):
    """Serie historica del equity de la cuenta paper (endpoint portfolio/history).
    Devuelve un DataFrame con indice de fecha y columnas equity y profit_loss_pct.
    Base del calculo de drawdown y de las metricas de rendimiento."""
    import pandas as pd
    r = requests.get(_url("account/portfolio/history"),
                     params={"period": period, "timeframe": timeframe,
                             "extended_hours": "false"},
                     headers=_headers(), timeout=30)
    r.raise_for_status()
    d = r.json()
    ts = d.get("timestamp", [])
    eq = d.get("equity", [])
    if not ts or not eq:
        return pd.DataFrame(columns=["equity", "pl_pct"])
    idx = pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s")
    df = pd.DataFrame({"equity": pd.Series(eq, dtype="float64").to_numpy(),
                       "pl_pct": pd.Series(d.get("profit_loss_pct", [0] * len(eq))).to_numpy()},
                      index=idx)
    return df[df["equity"] > 0]


def enviar_bracket(ticker, qty, tp_price, sl_price, lado="buy", confirmar=False, libro=None):
    """Orden BRACKET para intradia: entrada a mercado + take-profit (limit) y
    stop-loss (stop). Alpaca gestiona la salida sola cuando se toca TP o SL.

    tp_price/sl_price: precios de objetivo y stop. Con confirmar=False es dry-run.
    libro: etiqueta del libro de origen (-> client_order_id, para atribuir riesgo).
    """
    qty = int(qty)
    if qty <= 0:
        return {"omitida": True, "ticker": ticker, "qty": qty}
    cuerpo = {"symbol": ticker, "qty": str(qty), "side": lado, "type": "market",
              "time_in_force": "day", "order_class": "bracket",
              "take_profit": {"limit_price": round(float(tp_price), 2)},
              "stop_loss": {"stop_price": round(float(sl_price), 2)}}
    if libro:
        cuerpo["client_order_id"] = f"{libro}-{uuid.uuid4().hex[:16]}"
    if not confirmar:
        return {"dry_run": True, **cuerpo}
    r = requests.post(_url("orders"), json=cuerpo, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def cerrar_todo(confirmar=False):
    """Liquida todas las posiciones y cancela ordenes abiertas (para empezar
    limpio o cerrar el dia). Con confirmar=False solo informa."""
    pos = posiciones()
    if not confirmar:
        return {"dry_run": True, "cerraria": list(pos.keys())}
    requests.delete(_url("orders"), headers=_headers(), timeout=30)
    r = requests.delete(_url("positions"), headers=_headers(), timeout=30)
    return {"cerradas": list(pos.keys()), "status": r.status_code}


def cerrar_posiciones(tickers, confirmar=False):
    """Cierra posiciones concretas (cancelando antes sus ordenes bracket). Para la
    gestion activa: cortar las intradia que se giran. Con confirmar=False, dry-run."""
    pos = posiciones()
    a = [t for t in tickers if t in pos]
    if not confirmar:
        return {"dry_run": True, "cerraria": a}
    abiertas = ordenes_abiertas()
    cerradas = []
    for t in a:
        for o in abiertas:
            if o.get("symbol") == t:
                try:
                    requests.delete(_url(f"orders/{o['id']}"), headers=_headers(), timeout=30)
                except Exception:
                    pass
        try:
            r = requests.delete(_url(f"positions/{t}"), headers=_headers(), timeout=30)
            cerradas.append((t, r.status_code, pos[t]["pl"]))
        except Exception as e:
            cerradas.append((t, str(e)[:40], pos[t]["pl"]))
    return {"cerradas": cerradas}


def cerrar_intradia(confirmar=False):
    """Cierra SOLO el book intradia (las posiciones con ordenes bracket abiertas)
    y cancela esos brackets. El swing (sin ordenes pendientes) no se toca.

    Devuelve que posiciones cierra y su P&L no realizado en ese momento (= el del
    dia, porque se abrieron hoy). Con confirmar=False es dry-run.
    """
    ordenes = ordenes_abiertas()
    simbolos = sorted(set(o["symbol"] for o in ordenes))
    pos = posiciones()
    a_cerrar = [t for t in simbolos if t in pos]
    pl = {t: pos[t]["pl"] for t in a_cerrar}
    if not confirmar:
        return {"dry_run": True, "cerraria": a_cerrar, "pl": pl}
    requests.delete(_url("orders"), headers=_headers(), timeout=30)   # cancela los brackets
    cerradas = []
    for t in a_cerrar:
        try:
            r = requests.delete(_url(f"positions/{t}"), headers=_headers(), timeout=30)
            cerradas.append((t, r.status_code))
        except Exception as e:
            cerradas.append((t, str(e)[:40]))
    return {"cerradas": cerradas, "pl": pl, "pl_total": sum(pl.values())}
