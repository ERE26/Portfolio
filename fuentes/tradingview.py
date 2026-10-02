"""Fuente TradingView: escáner de TODO el mercado US en una sola llamada (endpoint público
`scanner.tradingview.com`, sin credenciales). Permite rankear ~2.5-3k valores por el cambio
intradía (last ~1h, sin el hueco de apertura) con filtros de liquidez, cosa que el plan
gratis de Alpaca NO da (su screener es de pago).

Nota: es un endpoint NO oficial (puede cambiar sin aviso). Úsalo con un fallback; aquí
`radar_ganadores.py` cae al pool de Yahoo si esto falla.
"""
import requests

URL = "https://scanner.tradingview.com/america/scan"
_H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def scan(columns, sort_by, filtros=None, n=50, orden="desc", timeout=25):
    """POST genérico al escáner. Devuelve lista de dicts {symbol, ...columns}."""
    body = {
        "columns": list(columns),
        "sort": {"sortBy": sort_by, "sortOrder": orden},
        "range": [0, int(n)],
        "markets": ["america"],
        "filter": list(filtros or []),
    }
    r = requests.post(URL, json=body, headers=_H, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    out = []
    for row in j.get("data", []):
        d = dict(zip(columns, row["d"]))
        d["symbol"] = row["s"].split(":")[-1]
        out.append(d)
    return out, int(j.get("totalCount", 0))


def ganadores_intradia(tf="60", n=15, min_precio=3.0, min_dvol=1_000_000.0, min_rvol=1.2):
    """Mayores ganadores del mercado en el timeframe intradía `tf` minutos (60 = última
    hora, sin la apertura: es el cambio dentro de la vela intradía en curso). Filtrado por
    liquidez (precio, volumen en $ y RVOL). Devuelve filas ordenadas por el cambio `tf`.

    Cada fila: {symbol, precio, chg_tf, chg_15, chg_dia, rvol, dvol, sector}."""
    cols = ["name", "close", f"change|{tf}", "change|15", "change",
            "relative_volume_10d_calc", "Value.Traded", "sector"]
    filtros = [
        {"left": "close", "operation": "in_range", "right": [min_precio, 5000]},
        {"left": "relative_volume_10d_calc", "operation": "greater", "right": min_rvol},
        {"left": "active_symbol", "operation": "equal", "right": True},
    ]
    rows, total = scan(cols, sort_by=f"change|{tf}", filtros=filtros, n=max(n * 4, 60))
    out = []
    for d in rows:
        dvol = d.get("Value.Traded") or 0.0
        if dvol < min_dvol:                      # liquidez real (el filtro servidor no siempre basta)
            continue
        out.append({
            "symbol": d["symbol"], "precio": d.get("close"),
            "chg_tf": d.get(f"change|{tf}"), "chg_15": d.get("change|15"),
            "chg_dia": d.get("change"), "rvol": d.get("relative_volume_10d_calc"),
            "dvol": dvol, "sector": d.get("sector") or "",
        })
        if len(out) >= n:
            break
    return out, total
