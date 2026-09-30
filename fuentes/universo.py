"""Catalogo del universo de estudio: +100 tickers de Yahoo Finance.

Tres clases de activo:
  - ACCIONES divididas por sector (GICS), para que la fuerza relativa sectorial
    y la comparacion dentro del mismo sector tengan sentido.
  - DIVISAS (pares forex principales y cruces).
  - FONDOS / ETFs (indices amplios y ETFs sectoriales), utiles como "mercado" y
    como activos en si.

Cada entrada es ticker -> (clase, sector). Para divisas y fondos el 'sector'
describe el tipo (p.ej. 'forex_major', 'etf_indice').
"""

ACCIONES = {
    "tecnologia": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "INTC"],
    "comunicacion": ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "TMUS"],
    "consumo_discrecional": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "LOW", "BKNG"],
    "consumo_basico": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "MO"],
    "salud": ["JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "BMY"],
    "financiero": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW"],
    "industrial": ["CAT", "BA", "HON", "UPS", "GE", "RTX", "LMT", "DE", "MMM"],
    "energia": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY"],
    "materiales": ["LIN", "APD", "SHW", "FCX", "NEM", "DOW", "NUE"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
    "inmobiliario": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O"],
}

DIVISAS = {
    "forex_major": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
                    "USDCAD=X", "NZDUSD=X"],
    "forex_cruce": ["EURGBP=X", "EURJPY=X", "GBPJPY=X", "EURCHF=X", "AUDJPY=X",
                    "USDMXN=X"],
}

FONDOS = {
    "etf_indice": ["SPY", "QQQ", "DIA", "IWM", "VTI", "EEM", "EFA"],
    "etf_sectorial": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
                      "XLB", "XLRE", "XLC"],
    "etf_bonos_mat": ["TLT", "HYG", "LQD", "GLD", "SLV"],
}

# nombre legible del indice de referencia de cada clase (para el 'mercado')
INDICE_REF = {"accion": "SPY", "divisa": "EURUSD=X", "fondo": "SPY"}


def catalogo():
    """Devuelve dict ticker -> {'clase','sector'} con todo el universo."""
    out = {}
    for sector, tickers in ACCIONES.items():
        for t in tickers:
            out[t] = {"clase": "accion", "sector": sector}
    for sector, tickers in DIVISAS.items():
        for t in tickers:
            out[t] = {"clase": "divisa", "sector": sector}
    for sector, tickers in FONDOS.items():
        for t in tickers:
            out[t] = {"clase": "fondo", "sector": sector}
    return out


def por_clase(clase=None):
    cat = catalogo()
    if clase is None:
        return list(cat)
    return [t for t, v in cat.items() if v["clase"] == clase]


def resumen():
    cat = catalogo()
    import collections
    clases = collections.Counter(v["clase"] for v in cat.values())
    sectores = collections.Counter(v["sector"] for v in cat.values() if v["clase"] == "accion")
    return {"total": len(cat), "por_clase": dict(clases), "sectores_accion": dict(sectores)}


if __name__ == "__main__":
    r = resumen()
    print(f"UNIVERSO: {r['total']} tickers")
    print("por clase:", r["por_clase"])
    print("sectores (acciones):", r["sectores_accion"])
