"""SEC EDGAR: fundamentales oficiales point-in-time (GRATIS, sin clave).

Da, para cada empresa, la serie de resultados trimestrales (ingresos, beneficio,
BPA) con la FECHA DE PUBLICACION real ('filed'). Eso permite construir features
fundamentales sin mirar al futuro: en una fecha de mercado t solo se usan los
datos que ya se habian publicado en t.

Features que aporta (los que de verdad mueven el precio y el grafico no contiene):
  - dias desde la ultima publicacion de resultados (efecto post-earnings)
  - proximidad al proximo trimestre (antes de resultados sube la incertidumbre)
  - crecimiento interanual de ingresos y de BPA
  - aceleracion del crecimiento (segunda derivada): la sorpresa "de tendencia"
Sin consenso de analistas gratis, la "sorpresa" se aproxima por la desviacion
del crecimiento respecto a su propia tendencia reciente.
"""
import numpy as np
import pandas as pd

from .http import Cliente

UA = "estudio-academico-cunef e.rodriguezesteban@cunef.edu"
BASE = "https://data.sec.gov"


def _cliente(cliente):
    cli = cliente or Cliente(ttl=7 * 24 * 3600)
    cli.sesion.headers.update({"User-Agent": UA})
    return cli


def mapa_cik(cliente=None):
    cli = _cliente(cliente)
    d = cli.get_json("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in d.values()}


def _serie_concepto(facts, conceptos, forms=("10-Q", "10-K")):
    """Serie trimestral fusionando varios conceptos alternativos.

    Las empresas cambian de concepto contable con los anos (p.ej. NVDA paso de
    'Revenues' a 'RevenueFromContractWithCustomer...'), asi que se COMBINAN todos
    los conceptos dados en una sola serie por trimestre, quedandose con el primero
    publicado de cada 'end'. Solo periodos trimestrales (~90 dias) para que el
    crecimiento interanual no mezcle trimestres con anuales."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    filas = []
    for c in conceptos:
        if c not in gaap:
            continue
        for unidad, registros in gaap[c]["units"].items():
            for r in registros:
                if r.get("form") in forms and r.get("end") and r.get("filed") and r.get("start"):
                    ini = pd.Timestamp(r["start"]); fin = pd.Timestamp(r["end"])
                    if not (80 <= (fin - ini).days <= 100):
                        continue
                    filas.append({"end": fin, "val": r["val"], "filed": pd.Timestamp(r["filed"]),
                                  "concepto": c, "form": r.get("form")})
    if len(filas) < 6:
        return pd.DataFrame(columns=["end", "val", "filed", "concepto", "form"])
    df = pd.DataFrame(filas).sort_values("filed").drop_duplicates("end", keep="first")
    return df.sort_values("end").reset_index(drop=True)


def _serie_puntual(facts, conceptos, forms=("10-Q", "10-K")):
    """Serie de una magnitud de BALANCE (puntual, no de periodo): activos,
    patrimonio... Se identifica por tener 'instant' (sin 'start'/'end' de duracion)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    filas = []
    for c in conceptos:
        if c not in gaap:
            continue
        for unidad, registros in gaap[c]["units"].items():
            for r in registros:
                if r.get("form") in forms and r.get("end") and r.get("filed") and not r.get("start"):
                    filas.append({"end": pd.Timestamp(r["end"]), "val": r["val"],
                                  "filed": pd.Timestamp(r["filed"])})
    if len(filas) < 4:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df = pd.DataFrame(filas).sort_values("filed").drop_duplicates("end", keep="first")
    return df.sort_values("end").reset_index(drop=True)


def fundamentales(ticker, cik=None, cliente=None):
    """Series trimestrales de las magnitudes fundamentales con fecha de publicacion."""
    cli = _cliente(cliente)
    if cik is None:
        cik = mapa_cik(cli).get(ticker)
    if cik is None:
        raise RuntimeError(f"{ticker}: sin CIK en SEC")
    facts = cli.get_json(f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json")
    return {
        "ingresos": _serie_concepto(facts, ["RevenueFromContractWithCustomerExcludingAssessedTax",
                                            "Revenues", "SalesRevenueNet"]),
        "bpa": _serie_concepto(facts, ["EarningsPerShareDiluted", "EarningsPerShareBasic"]),
        "beneficio": _serie_concepto(facts, ["NetIncomeLoss", "ProfitLoss"]),
        "acciones": _serie_concepto(facts, ["WeightedAverageNumberOfDilutedSharesOutstanding",
                                            "WeightedAverageNumberOfSharesOutstandingBasic"]),
        "patrimonio": _serie_puntual(facts, ["StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
        "activos": _serie_puntual(facts, ["Assets"]),
    }


def _ttm(serie):
    """Suma movil de 4 trimestres (trailing-twelve-months) para magnitudes de flujo."""
    s = serie.copy()
    s["ttm"] = s["val"].rolling(4, min_periods=4).sum()
    return s.dropna(subset=["ttm"])


def factores_valor_calidad(ticker, fechas, precio, cik=None, cliente=None):
    """Factores VALUE, QUALITY y SIZE point-in-time (crudos, sin normalizar).

    Siguen las definiciones de Fama-French y AQR (Quality-Minus-Junk), con lo que
    los datos de SEC permiten: E/P y S/P (value), margen neto, ROE y variabilidad
    de beneficios (quality), y market cap (size). Todo con TTM y fecha 'filed'
    para no mirar al futuro. La normalizacion cross-sectional se hace despues,
    comparando todas las acciones en cada fecha (ver patrones.factores).
    """
    cols = ["fac_earnings_yield", "fac_sales_price", "fac_margen_neto", "fac_roe",
            "fac_estab_beneficio", "fac_size"]
    idx = pd.DatetimeIndex(fechas)
    vacio = pd.DataFrame({c: np.nan for c in cols}, index=idx)
    try:
        f = fundamentales(ticker, cik, cliente)
    except Exception:
        return vacio
    if len(f["bpa"]) < 6:
        return vacio

    bpa_ttm = _ttm(f["bpa"])
    ing_ttm = _ttm(f["ingresos"]) if len(f["ingresos"]) >= 4 else None
    ben_ttm = _ttm(f["beneficio"]) if len(f["beneficio"]) >= 4 else None
    acc = f["acciones"]
    pat, act = f["patrimonio"], f["activos"]
    precio = pd.Series(precio).reindex(idx).ffill()

    def ultimo(serie, fecha, col="ttm"):
        pub = serie[serie["filed"] < fecha]
        return pub.iloc[-1][col] if len(pub) else np.nan

    filas = []
    for fch in idx:
        px = precio.get(fch, np.nan)
        eps = ultimo(bpa_ttm, fch)
        ey = eps / px if (np.isfinite(eps) and np.isfinite(px) and px > 0) else np.nan
        ni = ultimo(ben_ttm, fch) if ben_ttm is not None else np.nan
        rev = ultimo(ing_ttm, fch) if ing_ttm is not None else np.nan
        n_acc = ultimo(acc, fch, "val") if len(acc) else (
            ni / ultimo(bpa_ttm, fch) if np.isfinite(ni) else np.nan)
        mcap = px * n_acc if (np.isfinite(px) and np.isfinite(n_acc) and n_acc > 0) else np.nan
        sp = rev / mcap if (np.isfinite(rev) and np.isfinite(mcap)) else np.nan
        margen = ni / rev if (np.isfinite(ni) and np.isfinite(rev) and rev > 0) else np.nan
        eq = ultimo(pat, fch, "val") if len(pat) else np.nan
        roe = ni / eq if (np.isfinite(ni) and np.isfinite(eq) and eq > 0) else np.nan
        # estabilidad de beneficio: -std del crecimiento interanual del BPA (mayor=mejor)
        pub = bpa_ttm[bpa_ttm["filed"] < fch]
        estab = -float(pub["ttm"].pct_change().tail(8).std()) if len(pub) >= 9 else np.nan
        size = -np.log(mcap) if (np.isfinite(mcap) and mcap > 0) else np.nan
        filas.append((ey, sp, margen, roe, estab, size))
    return pd.DataFrame(filas, index=idx, columns=cols)


def rasgos_fundamentales(ticker, fechas, cik=None, cliente=None):
    """DataFrame de features fundamentales point-in-time alineado a 'fechas'.

    Para cada fecha de mercado usa solo lo publicado antes de esa fecha.
    """
    cols = ["fund_dias_desde_pub", "fund_crec_ingresos_yoy", "fund_crec_bpa_yoy",
            "fund_aceleracion_ingresos", "fund_sorpresa_tendencia"]
    vacio = pd.DataFrame({c: 0.0 for c in cols}, index=fechas)
    try:
        f = fundamentales(ticker, cik, cliente)
    except Exception:
        return vacio
    ing, bpa = f["ingresos"], f["bpa"]
    if len(ing) < 6:
        return vacio

    ing = ing.copy()
    ing["crec_yoy"] = ing["val"].pct_change(4)               # crecimiento interanual
    ing["acel"] = ing["crec_yoy"].diff()                      # aceleracion
    ing["tendencia"] = ing["crec_yoy"].rolling(4, min_periods=2).mean()
    ing["sorpresa"] = ing["crec_yoy"] - ing["tendencia"]      # desviacion de su tendencia
    bpa = bpa.copy()
    bpa["crec_yoy"] = bpa["val"].pct_change(4)

    fidx = pd.DatetimeIndex(fechas)
    filas = []
    for fecha in fidx:
        pub_ing = ing[ing["filed"] < fecha]                  # solo publicado antes de hoy
        pub_bpa = bpa[bpa["filed"] < fecha]
        if len(pub_ing) == 0:
            filas.append((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        ult = pub_ing.iloc[-1]
        dias = (fecha - ult["filed"]).days
        cb = float(pub_bpa.iloc[-1]["crec_yoy"]) if len(pub_bpa) and np.isfinite(
            pub_bpa.iloc[-1]["crec_yoy"]) else 0.0
        # winsorizacion: el crecimiento interanual puede dar valores extremos
        # (BPA que cruza de negativo a positivo), que sesgarian la normalizacion.
        clip = lambda x, lim: float(np.clip(np.nan_to_num(x), -lim, lim))
        filas.append((
            float(np.clip(dias, 0, 120)) / 120.0,
            clip(ult["crec_yoy"], 3.0),
            clip(cb, 3.0),
            clip(ult["acel"], 2.0),
            clip(ult["sorpresa"], 2.0),
        ))
    return pd.DataFrame(filas, index=fechas, columns=cols)
