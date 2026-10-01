"""Señal "smart money": que compran y venden los grandes inversores (13F de SEC).

Todo gestor con mas de 100M USD declara sus posiciones cada trimestre en un
formulario 13F, publico y con fecha de publicacion real (~45 dias tras el cierre
del trimestre). La literatura (Cohen-Polk-Silli 2010; Aiken et al. 2013) muestra
que copiar las posiciones de ALTA CONVICCION de buenos gestores da alpha incluso
con ese retraso.

Aqui se agregan los 13F de un grupo de inversores de conviccion y largo plazo y
se construyen, point-in-time (solo con lo publicado antes de cada fecha), features
por valor:
  - inst_presencia : cuantos de esos fondos tienen el valor (consenso)
  - inst_peso      : peso agregado en sus carteras (conviccion)
  - inst_cambio    : variacion trimestral de acciones (¿acumulan o reducen?)
  - inst_nuevo     : si es una posicion recien abierta

No mira al futuro: para una fecha t solo usa 13F con fecha de publicacion < t.
"""
import re
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from .http import Cliente

UA = "estudio-academico-cunef e.rodriguezesteban@cunef.edu"

# inversores de conviccion y largo plazo (no quants de alta rotacion)
FONDOS = {
    "Berkshire": "0001067983", "PershingSquare": "0001336528", "Appaloosa": "0001656456",
    "ThirdPoint": "0001040273", "Baupost": "0001061768", "Akre": "0001112520",
    "Himalaya": "0001709323", "Duquesne": "0001536411", "Viking": "0001103804",
    "LonePine": "0001061165",
}


def _cli(cliente):
    cli = cliente or Cliente(ttl=14 * 24 * 3600)
    cli.sesion.headers.update({"User-Agent": UA})
    return cli


def _info_table_url(cik, accn, cli):
    """Encuentra el XML del information table dentro de un filing."""
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/"
    import requests
    r = cli.sesion.get(url, timeout=30)
    xmls = re.findall(r'href="([^"]+\.xml)"', r.text)
    # el information table no es primary_doc.xml
    cand = [x for x in xmls if "primary_doc" not in x.lower()]
    return "https://www.sec.gov" + cand[0] if cand else None


def posiciones_fondo(cik, cli, max_trimestres=24):
    """Serie de posiciones de un fondo: lista de (report_date, filed, {issuer: (valor, acciones)})."""
    sub = cli.get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    rec = sub["filings"]["recent"]
    trimestres = []
    idx = [i for i, f in enumerate(rec["form"]) if f == "13F-HR"][:max_trimestres]
    for i in idx:
        accn = rec["accessionNumber"][i].replace("-", "")
        try:
            url = _info_table_url(cik.lstrip("0") or cik, accn, cli)
            if not url:
                continue
            xml = cli.sesion.get(url, timeout=30).text
            root = ET.fromstring(xml)
            ns = re.match(r"\{.*\}", root.tag)
            ns = ns.group(0) if ns else ""
            pos = {}
            for it in root.findall(f"{ns}infoTable"):
                issuer = (it.findtext(f"{ns}nameOfIssuer", "") or "").upper().strip()
                valor = int(it.findtext(f"{ns}value", "0") or 0)
                sh = it.find(f"{ns}shrsOrPrnAmt")
                acc = int(sh.findtext(f"{ns}sshPrnamt", "0") or 0) if sh is not None else 0
                if issuer:
                    v, a = pos.get(issuer, (0, 0))
                    pos[issuer] = (v + valor, a + acc)
            trimestres.append((pd.Timestamp(rec["reportDate"][i]),
                               pd.Timestamp(rec["filingDate"][i]), pos))
        except Exception:
            continue
    return trimestres


# mapeo nombre-de-emisor (13F) -> ticker de nuestro universo
def _mapa_nombres(tickers):
    import json, os
    nombres = {
        "AAPL": "APPLE", "MSFT": "MICROSOFT", "NVDA": "NVIDIA", "AMZN": "AMAZON",
        "GOOGL": "ALPHABET", "META": "META PLATFORMS", "TSLA": "TESLA", "AVGO": "BROADCOM",
        "JPM": "JPMORGAN", "BAC": "BANK OF AMER", "WFC": "WELLS FARGO", "GS": "GOLDMAN",
        "MS": "MORGAN STANLEY", "C": "CITIGROUP", "AXP": "AMERICAN EXPRESS", "V": "VISA",
        "KO": "COCA COLA", "PEP": "PEPSICO", "PG": "PROCTER", "WMT": "WALMART",
        "COST": "COSTCO", "MCD": "MCDONALD", "NKE": "NIKE", "HD": "HOME DEPOT",
        "JNJ": "JOHNSON", "UNH": "UNITEDHEALTH", "LLY": "LILLY", "PFE": "PFIZER",
        "MRK": "MERCK", "ABBV": "ABBVIE", "CVX": "CHEVRON", "XOM": "EXXON",
        "COP": "CONOCOPHILLIPS", "OXY": "OCCIDENTAL", "CAT": "CATERPILLAR", "BA": "BOEING",
        "DIS": "DISNEY", "NFLX": "NETFLIX", "CSCO": "CISCO", "ORCL": "ORACLE",
        "CRM": "SALESFORCE", "ADBE": "ADOBE", "AMD": "ADVANCED MICRO", "INTC": "INTEL",
        "MCO": "MOODY", "MA": "MASTERCARD", "KHC": "KRAFT HEINZ", "DAL": "DELTA AIR",
    }
    return {t: nombres[t] for t in tickers if t in nombres}


COLS_INST = ["inst_presencia", "inst_peso", "inst_cambio", "inst_nuevo"]


def cargar_carteras(cliente=None, verbose=False):
    """Descarga los 13F de todos los fondos UNA vez (cacheable y reutilizable)."""
    cli = _cli(cliente)
    if verbose:
        print(f"  13F: descargando {len(FONDOS)} fondos...")
    carteras = {}
    for nombre, cik in FONDOS.items():
        try:
            carteras[nombre] = posiciones_fondo(cik, cli)
            if verbose:
                print(f"    {nombre}: {len(carteras[nombre])} trimestres")
        except Exception:
            pass
    return carteras


def señal_ticker(ticker, fechas, carteras, patron=None):
    """Features institucionales point-in-time de un ticker a partir de las carteras."""
    patron = patron or _mapa_nombres([ticker]).get(ticker)
    fechas = pd.DatetimeIndex(fechas)
    if patron is None:
        return pd.DataFrame({c: 0.0 for c in COLS_INST}, index=fechas)
    series = []
    for nombre, trimestres in carteras.items():
        s = []
        for report, filed, pos in sorted(trimestres, key=lambda x: x[1]):
            total = sum(v for v, a in pos.values()) or 1
            match = [(v, a) for iss, (v, a) in pos.items() if patron in iss]
            s.append((filed, sum(x[0] for x in match) / total, sum(x[1] for x in match)))
        series.append(s)
    filas = []
    for f in fechas:
        presencia = peso = cambio = nuevo = 0.0
        for s in series:
            visibles = [x for x in s if x[0] < f]
            if not visibles:
                continue
            _, w, a = visibles[-1]
            if a > 0:
                presencia += 1; peso += w
                if len(visibles) >= 2:
                    a_prev = visibles[-2][2]
                    if a_prev == 0:
                        nuevo += 1
                    elif a_prev > 0:
                        cambio += (a - a_prev) / a_prev
        filas.append((presencia, peso, np.clip(cambio, -3, 3), nuevo))
    return pd.DataFrame(filas, index=fechas, columns=COLS_INST)


def señal_institucional(tickers, fechas_por_ticker, cliente=None, verbose=False):
    """Features institucionales para varios tickers (conveniencia)."""
    carteras = cargar_carteras(cliente, verbose)
    mapa = _mapa_nombres(tickers)
    cols = COLS_INST
    out = {}
    for t, patron in mapa.items():
        fechas = pd.DatetimeIndex(fechas_por_ticker[t])
        # para cada fondo, su serie (report, filed, acciones/peso en el ticker)
        series = []
        for nombre, trimestres in carteras.items():
            s = []
            for report, filed, pos in sorted(trimestres, key=lambda x: x[1]):
                total = sum(v for v, a in pos.values()) or 1
                match = [(v, a) for iss, (v, a) in pos.items() if patron in iss]
                v = sum(x[0] for x in match)
                a = sum(x[1] for x in match)
                s.append((filed, v / total, a))
            series.append(s)
        filas = []
        for f in fechas:
            presencia = peso = cambio = nuevo = 0.0
            for s in series:
                visibles = [x for x in s if x[0] < f]        # solo 13F ya publicados
                if not visibles:
                    continue
                _, w, a = visibles[-1]
                if a > 0:
                    presencia += 1
                    peso += w
                    if len(visibles) >= 2:
                        a_prev = visibles[-2][2]
                        if a_prev == 0:
                            nuevo += 1
                        elif a_prev > 0:
                            cambio += (a - a_prev) / a_prev
            filas.append((presencia, peso, np.clip(cambio, -3, 3), nuevo))
        out[t] = señal_ticker(t, fechas, carteras, patron)
    for t in tickers:
        if t not in out:
            out[t] = pd.DataFrame({c: 0.0 for c in cols}, index=pd.DatetimeIndex(fechas_por_ticker[t]))
    return out


def dossier(cliente=None, top=15):
    """Informe de inteligencia: consenso y movimientos de los grandes fondos.

    Devuelve (consenso, comprando, vendiendo) como DataFrames, a partir del ultimo
    13F de cada fondo. Es "aprender de los grandes" hecho explicito: donde coinciden
    (alta conviccion colectiva) y que estan acumulando o reduciendo.
    """
    from collections import defaultdict
    carteras = cargar_carteras(cliente)
    cons = defaultdict(lambda: {"fondos": 0, "valor": 0.0})
    mov = defaultdict(lambda: {"compra": 0, "venta": 0, "nuevo": 0})
    for fondo, tr in carteras.items():
        if len(tr) < 2:
            continue
        tr = sorted(tr, key=lambda x: x[1])
        ult, prev = tr[-1][2], tr[-2][2]
        for iss, (v, a) in ult.items():
            cons[iss]["fondos"] += 1
            cons[iss]["valor"] += v
            ap = prev.get(iss, (0, 0))[1]
            if ap == 0 and a > 0:
                mov[iss]["nuevo"] += 1
            elif a > ap * 1.05:
                mov[iss]["compra"] += 1
            elif a < ap * 0.95:
                mov[iss]["venta"] += 1
    rows = [{"emisor": i, "fondos": d["fondos"], "valor_M": d["valor"] / 1e6,
             "comprando": mov[i]["compra"], "vendiendo": mov[i]["venta"], "nuevas": mov[i]["nuevo"]}
            for i, d in cons.items()]
    df = pd.DataFrame(rows)
    consenso = df.sort_values(["fondos", "valor_M"], ascending=False).head(top)
    df["acumula"] = df["comprando"] + df["nuevas"]
    comprando = df[df["acumula"] >= 2].sort_values("acumula", ascending=False).head(top)
    vendiendo = df[df["vendiendo"] >= 2].sort_values("vendiendo", ascending=False).head(top)
    return consenso, comprando, vendiendo
