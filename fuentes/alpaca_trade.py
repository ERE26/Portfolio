"""Conector de Alpaca PAPER TRADING (cuenta demo), para operar la estrategia en
simulado. Bloqueado al endpoint paper por seguridad: NUNCA toca dinero real.

  - Lee las claves de datos/.alpaca_key (dos lineas: key id, secret). Las claves
    paper empiezan por 'PK' (las reales por 'AK'); si la clave no es paper, se
    rechaza para no operar en real por accidente.
  - Solo market orders 'day'. Enviar ordenes requiere confirmar=True explicito.

Nunca se suben las claves al repositorio (datos/ esta en .gitignore).
"""
import os

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


def enviar_orden(ticker, qty, lado, confirmar=False, tipo="market"):
    """Envia UNA orden a la cuenta paper. Requiere confirmar=True (salvaguarda).

    qty: numero de acciones (entero, >0). lado: 'buy' o 'sell'.
    Con confirmar=False devuelve la orden que se ENVIARIA, sin enviarla (dry-run).
    """
    qty = int(qty)
    if qty <= 0 or lado not in ("buy", "sell"):
        return {"omitida": True, "ticker": ticker, "qty": qty, "lado": lado}
    cuerpo = {"symbol": ticker, "qty": str(qty), "side": lado,
              "type": tipo, "time_in_force": "day"}
    if not confirmar:
        return {"dry_run": True, **cuerpo}
    r = requests.post(_url("orders"), json=cuerpo, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def enviar_bracket(ticker, qty, tp_price, sl_price, lado="buy", confirmar=False):
    """Orden BRACKET para intradia: entrada a mercado + take-profit (limit) y
    stop-loss (stop). Alpaca gestiona la salida sola cuando se toca TP o SL.

    tp_price/sl_price: precios de objetivo y stop. Con confirmar=False es dry-run.
    """
    qty = int(qty)
    if qty <= 0:
        return {"omitida": True, "ticker": ticker, "qty": qty}
    cuerpo = {"symbol": ticker, "qty": str(qty), "side": lado, "type": "market",
              "time_in_force": "day", "order_class": "bracket",
              "take_profit": {"limit_price": round(float(tp_price), 2)},
              "stop_loss": {"stop_price": round(float(sl_price), 2)}}
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
