"""Capa de datos: CSV, yfinance y un generador sintetico con patrones plantados.

Por que el generador sintetico: si le das al sistema datos reales y saca un 52%
de acierto, no sabes si el codigo esta bien y el mercado es dificil, o si el
codigo esta roto. Con datos sinteticos en los que TU has plantado los patrones
y sabes que ventaja tienen, puedes comprobar que la tuberia los encuentra. Es el
equivalente a un test de integracion para un sistema de prediccion.
"""
import os
import numpy as np
import pandas as pd

COLUMNAS = ["open", "high", "low", "close", "volume"]

SECTORES = ["tecnologia", "banca", "energia", "salud", "consumo", "industrial"]


# ------------------------------------------------------------------ CSV / red
def cargar_csv(ruta, col_fecha=None):
    """Lee un CSV de OHLCV. Tolera los nombres de columna mas habituales
    (TradingView, Investing, Yahoo) en ingles y en espanol."""
    df = pd.read_csv(ruta)
    alias = {
        "date": "fecha", "time": "fecha", "datetime": "fecha", "fecha": "fecha",
        "open": "open", "apertura": "open",
        "high": "high", "maximo": "high", "máximo": "high",
        "low": "low", "minimo": "low", "mínimo": "low",
        "close": "close", "close/last": "close", "cierre": "close", "adj close": "close",
        "volume": "volume", "vol": "volume", "vol.": "volume", "volumen": "volume",
    }
    df.columns = [alias.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    col = col_fecha or ("fecha" if "fecha" in df.columns else df.columns[0])
    df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    df = df.dropna(subset=[col]).set_index(col).sort_index()
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"{ruta}: faltan columnas {faltan}. Encontradas: {list(df.columns)}")
    for c in COLUMNAS:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(r"[^\d,.\-]", "", regex=True).str.replace(",", "", regex=False),
            errors="coerce")
    df.index.name = "fecha"
    return df[COLUMNAS].dropna()


def descargar_yfinance(ticker, desde="2010-01-01", hasta=None, intervalo="1d"):
    """Descarga de Yahoo. Requiere salida a internet; en algunos contenedores
    Yahoo devuelve 429 y hay que tirar de CSV o del generador sintetico."""
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=desde, end=hasta, interval=intervalo, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"Yahoo no devolvio datos para {ticker}")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "fecha"
    return df[COLUMNAS].dropna()


# ------------------------------------------------------------ PATRONES BASE
def _camino(puntos, n, ruido=0.0, rng=None):
    """Interpola una figura definida por puntos de control a n barras."""
    x = np.linspace(0, 1, len(puntos))
    y = np.interp(np.linspace(0, 1, n), x, np.asarray(puntos, float))
    if ruido and rng is not None:
        y = y + rng.normal(0, ruido, n)
    return y


def figura(nombre, n, rng):
    """Devuelve (camino_precio_normalizado, perfil_volumen) de una figura clasica."""
    r = lambda s: rng.normal(0, s)
    if nombre == "doble_techo":
        p = _camino([0, .55, 1.0, .5, .45, .97, .93, .4, .25], n, .05, rng)
        v = _camino([.8, 1.4, 1.6, .9, .8, 1.0, .9, 1.5, 1.3], n)   # 2o techo con menos volumen
    elif nombre == "doble_suelo":
        p = _camino([1, .45, 0, .5, .55, .03, .07, .6, .8], n, .05, rng)
        v = _camino([.9, 1.3, 1.5, .9, .8, 1.2, 1.1, 1.6, 1.4], n)
    elif nombre == "hombro_cabeza_hombro":
        p = _camino([0, .5, .62, .35, 1.0, .38, .6, .58, .3], n, .05, rng)
        v = _camino([.9, 1.5, 1.3, .9, 1.6, 1.0, .9, .8, 1.4], n)
    elif nombre == "bandera_alcista":
        subida = _camino([0, .45, .8, 1.0], n // 2 + n % 2, .03, rng)
        consol = _camino([1.0, .93, .97, .90, .95, .88], n // 2, .02, rng)
        p = np.concatenate([subida, consol])
        v = np.concatenate([_camino([1.5, 1.8, 1.4, 1.1], n // 2 + n % 2),
                            _camino([.8, .6, .55, .5, .5, .9], n // 2)])   # volumen se seca
    elif nombre == "taza_asa":
        t = np.linspace(0, 1, int(n * 0.8))
        taza = 1 - np.sin(np.pi * t) ** 1.4          # U redondeada
        asa = _camino([taza[-1], taza[-1] - .18, taza[-1] - .12], n - len(taza), .02, rng)
        p = np.concatenate([taza, asa]) + rng.normal(0, .03, n)
        v = np.concatenate([_camino([1.2, .7, .5, .6, 1.0], len(taza)),
                            _camino([.7, .6, .8], n - len(taza))])
    elif nombre == "triangulo_descendente":
        techos = np.linspace(1.0, .45, 4)
        ctrl = []
        for te in techos:
            ctrl += [te, .05 + r(.02)]               # maximos decrecientes, soporte plano
        p = _camino(ctrl, n, .04, rng)
        v = _camino([1.2, 1.0, .9, .8, .75, .7, .7, 1.3], n)
    elif nombre == "cuna_ascendente":
        ctrl, lo, hi = [], .0, .55
        for _ in range(4):
            ctrl += [lo, hi]
            lo, hi = lo + .18, hi + .10             # converge hacia arriba, suele romper abajo
        p = _camino(ctrl, n, .04, rng)
        v = _camino([1.1, 1.0, .9, .8, .7, .65, .6, .8], n)
    else:
        raise ValueError(f"figura desconocida: {nombre}")
    return p[:n], np.clip(v[:n], .2, None)


# sesgo esperado de cada figura y con que frecuencia se cumple de verdad.
# Ninguna llega al 100%: un patron que acierta siempre no existe y haria que la
# evaluacion no sirviese para nada.
CATALOGO = {
    "doble_techo":            (-1, 0.66),
    "doble_suelo":            (+1, 0.66),
    "hombro_cabeza_hombro":   (-1, 0.70),
    "bandera_alcista":        (+1, 0.68),
    "taza_asa":               (+1, 0.64),
    "triangulo_descendente":  (-1, 0.62),
    "cuna_ascendente":        (-1, 0.60),
}


# --------------------------------------------------------- MUNDO SINTETICO
def generar_universo(n_tickers=12, n_barras=2600, semilla=7, largo_patron=40,
                     horizonte=10, prob_patron=0.055, fuerza=1.1, inicio="2012-01-02"):
    """Crea un mercado de juguete completo.

    Tiene factor de mercado, factores de sector y ruido propio de cada valor,
    para que la fuerza relativa y el regimen de mercado signifiquen algo.
    Cada cierto tiempo planta una figura del catalogo y despues mueve el precio
    en la direccion del sesgo, pero solo con la frecuencia de CATALOGO.

    Devuelve (precios, meta, indice, registro):
      precios  dict ticker -> DataFrame OHLCV
      meta     DataFrame ticker, sector, nombre
      indice   DataFrame OHLCV del indice del mercado
      registro DataFrame con cada figura plantada y si su sesgo se cumplio
    """
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range(inicio, periods=n_barras)

    # factor de mercado con regimenes (alcista tranquilo / bajista volatil)
    regimen = np.zeros(n_barras, int)
    est = 0
    for i in range(n_barras):
        if rng.random() < (0.004 if est == 0 else 0.012):
            est = 1 - est
        regimen[i] = est
    deriva_mkt = np.where(regimen == 0, 0.00045, -0.00070)
    vol_mkt = np.where(regimen == 0, 0.0080, 0.0180)
    r_mkt = rng.normal(deriva_mkt, vol_mkt)

    r_sector = {s: rng.normal(rng.normal(0.0001, 0.0002), 0.006, n_barras) for s in SECTORES}

    precios, filas_meta, registro = {}, [], []
    for k in range(n_tickers):
        ticker = f"SYN{k:02d}"
        sector = SECTORES[k % len(SECTORES)]
        beta = float(rng.uniform(0.6, 1.5))
        beta_sec = float(rng.uniform(0.4, 1.0))
        vol_prop = float(rng.uniform(0.008, 0.020))

        r = beta * r_mkt + beta_sec * r_sector[sector] + rng.normal(0, vol_prop, n_barras)
        # agrupamiento de volatilidad: tras un dia movido vienen mas dias movidos
        for i in range(1, n_barras):
            r[i] *= 1.0 + 0.30 * min(abs(r[i - 1]) / vol_prop, 3.0)

        mult_vol = np.ones(n_barras)
        libre = np.ones(n_barras, bool)
        i = 60
        while i < n_barras - largo_patron - horizonte - 1:
            if rng.random() < prob_patron:
                nombre = list(CATALOGO)[rng.integers(len(CATALOGO))]
                sesgo, prob = CATALOGO[nombre]
                camino, perfil = figura(nombre, largo_patron, rng)
                # la figura se dibuja en incrementos, escalada a la vol local
                escala = fuerza * vol_prop * largo_patron / max(np.ptp(camino), 1e-9)
                r[i:i + largo_patron] = np.diff(camino, prepend=camino[0]) * escala / largo_patron * 8
                mult_vol[i:i + largo_patron] = perfil
                # despues de la figura, el sesgo se cumple solo 'prob' de las veces
                se_cumple = rng.random() < prob
                signo = sesgo if se_cumple else -sesgo
                fin = i + largo_patron
                r[fin:fin + horizonte] += signo * 0.9 * vol_prop
                mult_vol[fin:fin + horizonte] = 1.4
                libre[i:fin + horizonte] = False
                registro.append({"ticker": ticker, "fecha": fechas[fin - 1], "patron": nombre,
                                 "sesgo": sesgo, "se_cumplio": bool(se_cumple)})
                i = fin + horizonte + int(rng.integers(20, 90))
            else:
                i += 1

        close = 20 * np.exp(np.cumsum(r)) * float(rng.uniform(0.5, 4.0))
        cuerpo = np.abs(rng.normal(0, 0.004, n_barras)) + np.abs(r) * 0.3
        op = close * np.exp(rng.normal(0, 0.003, n_barras))
        hi = np.maximum(op, close) * (1 + cuerpo)
        lo = np.minimum(op, close) * (1 - cuerpo)
        base_v = float(rng.uniform(3e5, 5e6))
        vol = base_v * mult_vol * np.exp(rng.normal(0, 0.35, n_barras)) * (1 + 3 * np.abs(r))

        precios[ticker] = pd.DataFrame(
            {"open": op, "high": hi, "low": lo, "close": close, "volume": np.round(vol)},
            index=fechas).rename_axis("fecha")
        filas_meta.append({"ticker": ticker, "sector": sector, "nombre": f"Sintetica {k:02d} SA"})

    idx_close = np.exp(np.cumsum(r_mkt)) * 1000
    cuerpo = np.abs(rng.normal(0, 0.003, n_barras))
    indice = pd.DataFrame({"open": idx_close * np.exp(rng.normal(0, .002, n_barras)),
                           "high": idx_close * (1 + cuerpo), "low": idx_close * (1 - cuerpo),
                           "close": idx_close,
                           "volume": np.round(1e9 * np.exp(rng.normal(0, .2, n_barras)))},
                          index=fechas).rename_axis("fecha")

    reg = pd.DataFrame(registro)
    return precios, pd.DataFrame(filas_meta), indice, reg


def guardar_universo(carpeta, precios, meta, indice, registro):
    os.makedirs(carpeta, exist_ok=True)
    for t, df in precios.items():
        df.to_csv(os.path.join(carpeta, f"{t}.csv"))
    indice.to_csv(os.path.join(carpeta, "_INDICE.csv"))
    meta.to_csv(os.path.join(carpeta, "_meta.csv"), index=False)
    if len(registro):
        registro.to_csv(os.path.join(carpeta, "_patrones_plantados.csv"), index=False)


def cargar_carpeta(carpeta):
    """Lee una carpeta con un CSV por ticker (mas _INDICE.csv y _meta.csv si estan)."""
    precios = {}
    for f in sorted(os.listdir(carpeta)):
        if not f.endswith(".csv") or f.startswith("_"):
            continue
        precios[os.path.splitext(f)[0]] = cargar_csv(os.path.join(carpeta, f))
    ruta_idx = os.path.join(carpeta, "_INDICE.csv")
    indice = cargar_csv(ruta_idx) if os.path.exists(ruta_idx) else None
    ruta_meta = os.path.join(carpeta, "_meta.csv")
    meta = pd.read_csv(ruta_meta) if os.path.exists(ruta_meta) else \
        pd.DataFrame({"ticker": list(precios), "sector": "desconocido"})
    return precios, meta, indice


def resample(df, regla):
    """Pasa a otra temporalidad. 'W-FRI' semanal, 'ME' mensual, '2D'..."""
    out = df.resample(regla).agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"})
    return out.dropna()
