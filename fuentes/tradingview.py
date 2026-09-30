"""TradingView: scanner en vivo (publico, sin login).

IMPORTANTE sobre que da y que no:
  - Da un SNAPSHOT del estado actual del valor: indicadores ya calculados por
    TradingView en varias temporalidades a la vez (RSI diario/horario/semanal,
    MACD, ADX, Bollinger, Stochastic, medias, rendimiento) y su recomendacion
    tecnica agregada (de venta fuerte a compra fuerte).
  - NO da el historico de velas. Ese va por un websocket con token de sesion y
    contra los Terminos de Servicio de TradingView, asi que el proyecto NO lo usa.
    Para historico se usa Yahoo, que sirve exactamente las mismas velas.

Se usa como "segunda opinion en vivo" en el momento de pedir una prediccion.
"""
import pandas as pd

from .http import Cliente

URL = "https://scanner.tradingview.com/{region}/scan"

# columnas mas utiles del scanner. El sufijo |<tf> pide el indicador en esa
# temporalidad: |60 = 1h, |240 = 4h, |1W = semanal, |1M = mensual, sin sufijo = 1d.
COLUMNAS = [
    "close", "change", "volume", "average_volume_10d_calc",
    "RSI", "RSI|60", "RSI|1W",
    "MACD.macd", "MACD.signal", "ADX", "ADX+DI", "ADX-DI",
    "Stoch.K", "Stoch.D", "BB.upper", "BB.lower",
    "SMA20", "SMA50", "SMA200", "EMA20",
    "Perf.1M", "Perf.3M", "Volatility.D", "Volatility.W",
    "Recommend.All", "Recommend.MA", "Recommend.Other",
]
NOMBRES = [
    "cierre", "cambio_pct", "volumen", "vol_medio_10d",
    "rsi_1d", "rsi_1h", "rsi_1w",
    "macd", "macd_senal", "adx", "di_mas", "di_menos",
    "stoch_k", "stoch_d", "bb_sup", "bb_inf",
    "sma20", "sma50", "sma200", "ema20",
    "perf_1m", "perf_3m", "volat_d", "volat_w",
    "recomendacion", "recom_medias", "recom_osciladores",
]


def _texto_recom(v):
    if v is None:
        return "sin dato"
    if v >= 0.5:
        return "COMPRA FUERTE"
    if v >= 0.1:
        return "compra"
    if v > -0.1:
        return "neutral"
    if v > -0.5:
        return "venta"
    return "VENTA FUERTE"


def snapshot(tickers, region="america", cliente=None):
    """Estado en vivo de una lista de valores en formato 'EXCHANGE:TICKER'
    (p.ej. 'NASDAQ:AAPL'). Devuelve un DataFrame con una fila por valor."""
    cli = cliente or Cliente(ttl=300)   # en vivo: cache corta
    body = {"symbols": {"tickers": list(tickers), "query": {"types": []}},
            "columns": COLUMNAS}
    d = cli.get_json(URL.format(region=region), metodo="POST", body=body, cache=True)
    filas = []
    for row in d.get("data", []):
        fila = {"simbolo": row["s"]}
        fila.update(dict(zip(NOMBRES, row["d"])))
        fila["recomendacion_txt"] = _texto_recom(fila.get("recomendacion"))
        filas.append(fila)
    return pd.DataFrame(filas)


def snapshot_auto(tickers_base, region="america", cliente=None):
    """Igual que snapshot pero resolviendo el exchange solo.

    El scanner necesita 'EXCHANGE:TICKER' y no reconoce el ticker pelado. Como
    ignora en silencio los simbolos que no existen, se mandan las variantes
    candidatas (NASDAQ/NYSE/AMEX) de cada ticker sin prefijo en una sola llamada
    y se deduplica por ticker base quedandose con la que el scanner devuelva.
    """
    candidatos, base_de = [], {}
    for t in tickers_base:
        if ":" in t:
            candidatos.append(t)
            base_de[t] = t.split(":")[1]
        else:
            for ex in ("NASDAQ", "NYSE", "AMEX"):
                sc = f"{ex}:{t}"
                candidatos.append(sc)
                base_de[sc] = t
    snap = snapshot(candidatos, region=region, cliente=cliente)
    if snap.empty:
        return snap
    snap["_base"] = snap["simbolo"].map(base_de).fillna(snap["simbolo"])
    orden = {t: i for i, t in enumerate(tickers_base)}
    snap["_ord"] = snap["_base"].map(lambda b: orden.get(b, 999))
    return (snap.sort_values("_ord").drop_duplicates("_base")
            .drop(columns=["_base", "_ord"]).reset_index(drop=True))


def como_simbolo_tv(ticker, meta_exchange=None):
    """Traduce un ticker a 'EXCHANGE:TICKER' para el scanner.

    Yahoo usa 'NMS'/'NYQ' para Nasdaq/NYSE; TradingView quiere 'NASDAQ'/'NYSE'.
    Si no se sabe el mercado, se intenta sin prefijo y el scanner suele resolverlo.
    """
    mapa = {"NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NYQ": "NYSE",
            "PCX": "AMEX", "ASE": "AMEX"}
    if ":" in ticker:
        return ticker
    ex = mapa.get(meta_exchange)
    return f"{ex}:{ticker}" if ex else ticker


def bloque_informe(fila):
    """Texto para el informe de prediccion con la 'segunda opinion' de TradingView."""
    def g(k, fmt="{:.1f}"):
        v = fila.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) else "?"
    return "\n".join([
        f"  Segunda opinion EN VIVO de TradingView ({fila.get('simbolo','?')}):",
        f"    Recomendacion tecnica: {fila.get('recomendacion_txt','?')} "
        f"(medias {g('recom_medias','{:+.2f}')}, osciladores {g('recom_osciladores','{:+.2f}')})",
        f"    RSI  1h {g('rsi_1h')}  |  1d {g('rsi_1d')}  |  1w {g('rsi_1w')}",
        f"    MACD {g('macd','{:+.3f}')} vs senal {g('macd_senal','{:+.3f}')}  |  ADX {g('adx')}",
        f"    Rendimiento  1m {g('perf_1m','{:+.1f}')}%  3m {g('perf_3m','{:+.1f}')}%"
        f"  |  vs SMA200 {'por encima' if (fila.get('cierre') or 0) > (fila.get('sma200') or 1e9) else 'por debajo'}",
    ])
