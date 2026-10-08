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


def universo_liquido(n=300, min_precio=5.0, min_dvol=15_000_000.0, min_avgvol=1_000_000.0):
    """Las ~n acciones US MÁS LÍQUIDAS (por $-volumen): grandes + mid-caps líquidas. Para
    AMPLIAR el universo operable más allá de una lista fija. Filtra por precio, $-volumen
    y volumen medio (ejecutable intradía), type=stock y símbolo primario (sin duplicados
    de clases). Devuelve lista de símbolos ordenada por $-volumen desc."""
    cols = ["name", "Value.Traded", "market_cap_basic"]
    filtros = [
        {"left": "close", "operation": "in_range", "right": [min_precio, 5000]},
        {"left": "Value.Traded", "operation": "greater", "right": min_dvol},
        {"left": "average_volume_10d_calc", "operation": "greater", "right": min_avgvol},
        {"left": "type", "operation": "equal", "right": "stock"},
        {"left": "is_primary", "operation": "equal", "right": True},
    ]
    rows, _ = scan(cols, sort_by="Value.Traded", filtros=filtros, n=n)
    return [d["symbol"] for d in rows if d.get("symbol")]


# Tope de capitalización del pool intradía: se FILTRAN las large/mega-caps. El edge
# intradía (RVOL + momentum de VWAP) está en los "stocks in play" —mid-caps líquidas que
# SÍ se mueven—, no en las grandes, demasiado eficientes para dar el recorrido que buscamos
# (docs/estrategia_intradia_salidas.md: "el edge está en los stocks in play, no en las
# large-caps"). Forense 2026-10-02: las dos peores fueron TSLA ($1.46T) y TXN ($268B), y el
# pool trae a diario large-caps de $20-47B (VST, NTAP, ON, VRSN...). Umbral $20B: large-cap
# empieza ~$10B, así que $20B corta las large/mega dejando la franja mid/upper-mid en juego.
# Tuneable en una línea: 10e9 para ceñirse a mid/small puro; 50e9/100e9 para solo mega-caps.
# Nota: NKE ($50B) ganó +$128 el 2026-10-02, pero es large-cap; una sesión no tumba el
# principio (ni la instrucción de filtrar large-caps). La auditoría medirá el efecto real.
MAX_CAP_INTRADIA = 20_000_000_000.0     # $20B


def en_juego(n=120, min_precio=4.0, min_dvol=4_000_000.0, min_avgvol=400_000.0, min_rvol=1.1,
             max_cap=MAX_CAP_INTRADIA):
    """Acciones US LÍQUIDAS que están EN JUEGO ahora (RVOL alto y subiendo hoy), de TODO
    el mercado, en una sola llamada. Es el POOL DE CANDIDATOS intradía: en vez de iterar
    una lista fija de ~114, se evalúan los que de verdad se mueven hoy (sean los de
    siempre o mid-caps nuevas). Se EXCLUYEN las large/mega-caps (market cap >= max_cap):
    sin edge intradía. Nuestros gates (sobre barras frescas de Alpaca) deciden luego cuál
    se opera. Devuelve símbolos ordenados por RVOL desc."""
    cols = ["name", "relative_volume_10d_calc", "change", "Value.Traded", "market_cap_basic"]
    filtros = [
        {"left": "close", "operation": "in_range", "right": [min_precio, 5000]},
        {"left": "Value.Traded", "operation": "greater", "right": min_dvol},
        {"left": "average_volume_10d_calc", "operation": "greater", "right": min_avgvol},
        {"left": "relative_volume_10d_calc", "operation": "greater", "right": min_rvol},
        {"left": "change", "operation": "greater", "right": 0},
        {"left": "type", "operation": "equal", "right": "stock"},
        {"left": "is_primary", "operation": "equal", "right": True},
    ]
    rows, _ = scan(cols, sort_by="relative_volume_10d_calc", filtros=filtros, n=max(n * 2, 120))
    out = []
    for d in rows:
        if not d.get("symbol"):
            continue
        mc = d.get("market_cap_basic")
        if mc is not None and max_cap and mc >= max_cap:   # fuera large/mega-caps (cap conocido)
            continue
        out.append(d["symbol"])
        if len(out) >= n:
            break
    return out


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
