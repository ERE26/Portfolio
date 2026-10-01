"""Construccion de casos: cada barra se convierte en un ejemplo aprendible.

Un caso es:
  - la FORMA del grafico: los ultimos L cierres en logaritmo y normalizados
    (media 0, desviacion 1). Al normalizar, dos tramos con el mismo dibujo pero
    a distinto precio y distinta volatilidad se parecen, que es justo lo que se
    quiere. Es el equivalente exacto de 'normalize_mask' del bot de Worldle.
  - la forma del VOLUMEN en la misma ventana.
  - la forma en TEMPORALIDADES SUPERIORES (semanal y mensual).
  - un vector de INDICADORES y de CONTEXTO (sector, regimen, noticias).
  - el RESULTADO: que hizo el precio en las h barras siguientes.

Sobre la fuga de informacion, que es el error numero uno en este tipo de
trabajos: las barras de temporalidad superior solo se usan si estan CERRADAS
antes de la fecha del caso. Si estas a mitad de semana, la vela semanal en curso
contiene el viernes, que todavia no ha pasado; usarla seria ver el futuro.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as ventanas_de

from . import indicadores as ind
from . import noticias as mod_noticias
from .datos import resample
from .etiquetado import triple_barrera as etiqueta_triple_barrera

EPS = 1e-9


@dataclass
class Config:
    largo: int = 40              # barras de la forma diaria
    horizonte: int = 10          # barras hacia delante que se quieren predecir
    largo_semanal: int = 26
    largo_mensual: int = 12
    paso: int = 1                # 1 = un caso por barra
    dias_noticias: int = 30
    semivida_noticias: float = 7.0
    minimo_historia: int = 260   # barras necesarias antes del primer caso (SMA200, etc.)
    k_pt: float = 1.5            # barrera de beneficio, en multiplos de ATR
    k_sl: float = 1.5            # barrera de perdida, en multiplos de ATR
    temporalidades: dict = field(default_factory=lambda: {"semanal": "W-FRI", "mensual": "ME"})


GRUPOS_ESCALARES = {
    "osciladores": ["rsi14", "rsi2", "boll_b20", "macd_hist_atr", "adx14", "di_dif"],
    "momento": ["ret_1", "ret_5", "ret_20", "ret_60", "pos_sma50", "pos_sma200",
                "pend_sma50", "dist_max252", "dist_min252"],
    "volumen": ["z_volumen20", "tendencia_volumen", "pend_obv"],
    "volatilidad": ["atr_pct", "vol_real20", "ratio_vol_20_60", "rango_atr", "gap_medio20"],
    "contexto": ["mkt_sobre_sma200", "mkt_ret_20", "mkt_vol_percentil", "mkt_dd",
                 "fuerza_rel_indice_20", "fuerza_rel_indice_60", "fuerza_rel_sector_20"],
    "noticias": ["noticias_sentimiento", "noticias_n", "noticias_dispersion"],
    "fundamentales": ["fund_dias_desde_pub", "fund_crec_ingresos_yoy", "fund_crec_bpa_yoy",
                      "fund_aceleracion_ingresos", "fund_sorpresa_tendencia"],
    "macro": ["macro_vix", "macro_vix_cambio", "macro_pendiente_curva",
              "macro_spread_credito", "macro_tendencia_tipos"],
    "institucional": ["inst_presencia", "inst_peso", "inst_cambio", "inst_nuevo"],
    "factores": ["mom_12_1", "fac_earnings_yield", "fac_sales_price", "fac_margen_neto",
                 "fac_roe", "fac_estab_beneficio", "fac_size"],
    "intradia": ["intra_dist_vwap", "intra_pos_rango", "intra_hora_sin", "intra_hora_cos",
                 "intra_vol_rel", "intra_gap", "intra_ret_sesion", "intra_rvol"],
}
NOMBRES_ESCALARES = [c for cols in GRUPOS_ESCALARES.values() for c in cols]


def forma(matriz):
    """Normaliza cada fila a media 0 y desviacion 1: quita nivel y escala."""
    m = np.asarray(matriz, float)
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, keepdims=True)
    return ((m - mu) / np.maximum(sd, EPS)).astype(np.float32)


def rasgos_escalares(df, rasgos_contexto=None, rasgos_noticias=None, rasgos_extra=None):
    """Todos los indicadores de una vez, alineados por fecha. Nunca miran al futuro.

    rasgos_extra: dict de DataFrames adicionales (fundamentales SEC, macro FRED...)
    ya indexados por fecha; se reindexan al calendario del activo y rellenan con 0.
    """
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    # activos sin volumen (forex, algunos indices): Yahoo devuelve volumen 0.
    # Sus indicadores de volumen no se pueden calcular, asi que se neutralizan a 0
    # en lugar de dejar NaN (que descartaria TODOS los casos del activo).
    sin_volumen = (v.fillna(0) <= 0).mean() > 0.5 or v.nunique() <= 1
    a = ind.atr(h, l, c, 14)
    _, hist = ind.macd(c)[1:]
    adx14, di_mas, di_menos = ind.adx(h, l, c, 14)
    dmax, dmin = ind.distancia_extremo(c, 252)
    vol20 = ind.volatilidad_realizada(c, 20)
    sma50, sma200 = ind.sma(c, 50), ind.sma(c, 200)

    out = pd.DataFrame(index=df.index)
    out["rsi14"] = ind.rsi(c, 14)
    out["rsi2"] = ind.rsi(c, 2)
    out["boll_b20"] = ind.bollinger_b(c, 20)
    out["macd_hist_atr"] = hist / a.replace(0.0, np.nan)
    out["adx14"] = adx14
    out["di_dif"] = (di_mas - di_menos) / 100.0
    for n in (1, 5, 20, 60):
        out[f"ret_{n}"] = np.log(c / c.shift(n))
    out["mom_12_1"] = ind.momentum_12_1(c)
    out["pos_sma50"] = c / sma50 - 1.0
    # indicadores intradia (solo si la serie es intradia; en diario quedan en 0)
    from .intradia import es_intradia, rasgos_intradia, NOMBRES_INTRA
    if es_intradia(df.index):
        ri = rasgos_intradia(df)
        for col in NOMBRES_INTRA:
            out[col] = ri[col]
    else:
        for col in NOMBRES_INTRA:
            out[col] = 0.0
    out["pos_sma200"] = c / sma200 - 1.0
    out["pend_sma50"] = ind.pendiente(sma50, 20)
    out["dist_max252"], out["dist_min252"] = dmax, dmin
    if sin_volumen:
        out["z_volumen20"] = 0.0
        out["tendencia_volumen"] = 0.0
        out["pend_obv"] = 0.0
    else:
        out["z_volumen20"] = ind.z_score(v, 20)
        out["tendencia_volumen"] = np.log(ind.sma(v, 5) / ind.sma(v, 60))
        out["pend_obv"] = ind.pendiente(ind.obv(c, v), 20)
    out["atr_pct"] = a / c
    out["vol_real20"] = vol20
    out["ratio_vol_20_60"] = vol20 / ind.volatilidad_realizada(c, 60).replace(0.0, np.nan)
    out["rango_atr"] = (h - l) / a.replace(0.0, np.nan)
    out["gap_medio20"] = np.log(o / c.shift(1)).abs().rolling(20, min_periods=20).mean()

    rasgos_extra = rasgos_extra or {}
    fuentes = {"contexto": rasgos_contexto, "noticias": rasgos_noticias,
               "fundamentales": rasgos_extra.get("fundamentales"),
               "macro": rasgos_extra.get("macro"),
               "institucional": rasgos_extra.get("institucional"),
               "factores": rasgos_extra.get("factores")}
    for origen in ("contexto", "noticias", "fundamentales", "macro", "institucional"):
        fuente = fuentes[origen]
        for col in GRUPOS_ESCALARES[origen]:
            if fuente is not None and col in fuente.columns:
                out[col] = fuente[col].reindex(df.index).astype(float)
            else:
                out[col] = 0.0
    # factores value/quality/size del enriquecedor (mom_12_1 ya calculado arriba).
    # Se dejan SIN rellenar NaN: la normalizacion cross-sectional los trata luego.
    fac = rasgos_extra.get("factores")
    for col in GRUPOS_ESCALARES["factores"]:
        if col == "mom_12_1":
            continue
        out[col] = fac[col].reindex(df.index).astype(float) if (fac is not None and col in fac) else np.nan
    cols_no_fac = [c for c in NOMBRES_ESCALARES if not c.startswith("fac_")]
    out[cols_no_fac] = out[cols_no_fac].fillna(0.0)
    return out[NOMBRES_ESCALARES]


class BancoCasos:
    """Todos los casos de todos los valores, ya alineados y listos para usar."""

    def __init__(self, cfg, formas, formas_vol, tf, feat, meta, y, caminos,
                 caminos_high=None, caminos_low=None):
        self.cfg = cfg
        self.formas = formas
        self.formas_vol = formas_vol
        self.tf = tf
        self.feat = feat
        self.meta = meta.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.caminos = caminos
        # recorrido futuro de maximos y minimos (en ATR): permite recalcular la
        # etiqueta triple-barrera para cualquier ratio sin reconstruir el banco.
        self.caminos_high = caminos_high
        self.caminos_low = caminos_low
        self.nombres_feat = list(NOMBRES_ESCALARES)

    def __len__(self):
        return len(self.meta)

    @property
    def grupos(self):
        """Nombre de grupo -> como se mide la distancia en ese grupo."""
        g = {"forma": ("forma", None), "forma_volumen": ("forma_vol", None)}
        for nombre in self.tf:
            g[f"tf_{nombre}"] = ("tf", nombre)
        for nombre, cols in GRUPOS_ESCALARES.items():
            idx = [self.nombres_feat.index(c) for c in cols if c in self.nombres_feat]
            g[nombre] = ("feat", np.array(idx, int))
        return g

    def subconjunto(self, mascara):
        m = np.asarray(mascara)
        ch = self.caminos_high[m] if self.caminos_high is not None else None
        cl = self.caminos_low[m] if self.caminos_low is not None else None
        return BancoCasos(self.cfg, self.formas[m], self.formas_vol[m],
                          {k: v[m] for k, v in self.tf.items()}, self.feat[m],
                          self.meta.loc[m].copy(), self.y.loc[m].copy(), self.caminos[m],
                          caminos_high=ch, caminos_low=cl)

    def resumen(self):
        return (f"{len(self)} casos | {self.meta.ticker.nunique()} valores | "
                f"{self.meta.fecha.min().date()} a {self.meta.fecha.max().date()} | "
                f"forma {self.formas.shape[1]}d + " +
                " + ".join(f"{k} {v.shape[1]}" for k, v in self.tf.items()) +
                f" | {self.feat.shape[1]} indicadores")


def _formas_tf(df, regla, largo, fechas_caso):
    """Forma en temporalidad superior usando SOLO velas ya cerradas antes del caso."""
    sup = resample(df, regla)
    if len(sup) < largo + 1:
        return np.zeros((len(fechas_caso), largo), np.float32), np.zeros(len(fechas_caso), bool)
    lc = np.log(sup["close"].to_numpy(float))
    W = ventanas_de(lc, largo)                       # fila j = velas j..j+largo-1
    # unidad fijada a proposito: comparar datetime64 de unidades distintas es la
    # via rapida a errores silenciosos de factor 1000
    idx = sup.index.to_numpy("datetime64[ns]")
    # ultima vela con etiqueta ESTRICTAMENTE anterior a la fecha del caso
    ultima = np.searchsorted(idx, fechas_caso.to_numpy("datetime64[ns]"), side="left") - 1
    fila = ultima - largo + 1
    ok = (fila >= 0) & (fila < len(W))
    out = np.zeros((len(fechas_caso), largo), np.float32)
    if ok.any():
        out[ok] = forma(W[fila[ok]])
    return out, ok


def construir(precios, ctx=None, fuente_noticias=None, cfg=None, tickers=None, verbose=True,
              enriquecer=None):
    """Convierte un universo de precios en un BancoCasos."""
    cfg = cfg or Config()
    L, H = cfg.largo, cfg.horizonte
    tickers = tickers or sorted(precios)
    fuente_noticias = fuente_noticias or mod_noticias.SinNoticias()
    largos_tf = {"semanal": cfg.largo_semanal, "mensual": cfg.largo_mensual}

    acum = {"formas": [], "formas_vol": [], "feat": [], "caminos": [],
            "caminos_high": [], "caminos_low": [], "meta": [], "y": [],
            "tf": {k: [] for k in cfg.temporalidades}}

    for t in tickers:
        df = precios[t].dropna()
        n = len(df)
        if n < cfg.minimo_historia + L + H + 5:
            if verbose:
                print(f"  {t}: solo {n} barras, hacen falta "
                      f"{cfg.minimo_historia + L + H + 5}. Se salta.")
            continue

        rc = ctx.rasgos(t) if ctx is not None else None
        rn = mod_noticias.rasgos_sentimiento(fuente_noticias, t, df.index,
                                             cfg.dias_noticias, cfg.semivida_noticias) \
            if not isinstance(fuente_noticias, mod_noticias.SinNoticias) else None
        extra = enriquecer(t, df.index, df["close"]) if enriquecer is not None else None
        F = rasgos_escalares(df, rc, rn, rasgos_extra=extra)

        lc = np.log(df["close"].to_numpy(float))
        lv = np.log(np.maximum(df["volume"].to_numpy(float), 1.0))
        lh = np.log(df["high"].to_numpy(float))
        ll = np.log(df["low"].to_numpy(float))

        # posiciones candidatas: hay L barras detras, H delante y bastante historia
        pos = np.arange(max(cfg.minimo_historia, L - 1), n - H - 1, cfg.paso)
        if len(pos) == 0:
            continue

        Wc = ventanas_de(lc, L)          # fila i = barras i..i+L-1
        Wv = ventanas_de(lv, L)
        fila_forma = pos - L + 1
        formas = forma(Wc[fila_forma])
        formas_vol = forma(Wv[fila_forma])

        Mf = F.to_numpy(float)[pos]
        atr_pct = F["atr_pct"].to_numpy(float)[pos]
        atr_pct = np.where(np.isfinite(atr_pct) & (atr_pct > 1e-5), atr_pct, np.nan)

        Wfc = ventanas_de(lc, H)         # camino futuro: barras t+1 .. t+H
        Wfh = ventanas_de(lh, H)
        Wfl = ventanas_de(ll, H)
        base = lc[pos]
        caminos = (Wfc[pos + 1] - base[:, None]) / atr_pct[:, None]
        caminos_high = (Wfh[pos + 1] - base[:, None]) / atr_pct[:, None]
        caminos_low = (Wfl[pos + 1] - base[:, None]) / atr_pct[:, None]
        ret = lc[pos + H] - base
        mfe = (Wfh[pos + 1].max(axis=1) - base) / atr_pct
        mae = (Wfl[pos + 1].min(axis=1) - base) / atr_pct
        # etiqueta triple-barrera: operacion realista con objetivo y stop en ATR
        tb = etiqueta_triple_barrera(lh, ll, lc, pos, atr_pct, H,
                                     cfg.k_pt, cfg.k_sl)

        fechas = df.index[pos]
        tfs, ok_tf = {}, np.ones(len(pos), bool)
        for nombre, regla in cfg.temporalidades.items():
            m, ok = _formas_tf(df, regla, largos_tf.get(nombre, 20), fechas)
            tfs[nombre] = m
            ok_tf &= ok

        # las columnas de factores value/quality pueden ser NaN hasta la
        # normalizacion cross-sectional posterior, asi que no entran en la validez.
        idx_no_fac = [i for i, c in enumerate(NOMBRES_ESCALARES) if not c.startswith("fac_")]
        valido = np.isfinite(Mf[:, idx_no_fac]).all(axis=1) & np.isfinite(ret) \
            & np.isfinite(caminos).all(axis=1) & np.isfinite(formas).all(axis=1) & ok_tf
        if not valido.any():
            if verbose:
                print(f"  {t}: ningun caso valido (faltan indicadores o velas superiores).")
            continue

        acum["formas"].append(formas[valido])
        acum["formas_vol"].append(formas_vol[valido])
        acum["feat"].append(Mf[valido].astype(np.float32))
        acum["caminos"].append(caminos[valido].astype(np.float32))
        acum["caminos_high"].append(caminos_high[valido].astype(np.float32))
        acum["caminos_low"].append(caminos_low[valido].astype(np.float32))
        for nombre in cfg.temporalidades:
            acum["tf"][nombre].append(tfs[nombre][valido])
        acum["meta"].append(pd.DataFrame({
            "ticker": t, "fecha": fechas[valido],
            "sector": ctx.sector_de(t) if ctx is not None else "desconocido",
            "close": df["close"].to_numpy(float)[pos][valido],
            "atr_pct": atr_pct[valido]}))
        acum["y"].append(pd.DataFrame({
            "ret": ret[valido], "direccion": np.sign(ret[valido]),
            "tb_label": tb["label"][valido], "tb_ret": tb["ret"][valido],
            "tb_barras": tb["barras"][valido], "tb_rr": tb["rr"][valido],
            "mfe": mfe[valido], "mae": mae[valido]}))
        if verbose:
            print(f"  {t}: {int(valido.sum())} casos")

    if not acum["meta"]:
        raise RuntimeError("No se pudo construir ningun caso. Revisa el historico disponible.")

    banco = BancoCasos(
        cfg,
        np.concatenate(acum["formas"]), np.concatenate(acum["formas_vol"]),
        {k: np.concatenate(v) for k, v in acum["tf"].items()},
        np.concatenate(acum["feat"]),
        pd.concat(acum["meta"], ignore_index=True),
        pd.concat(acum["y"], ignore_index=True),
        np.concatenate(acum["caminos"]),
        caminos_high=np.concatenate(acum["caminos_high"]),
        caminos_low=np.concatenate(acum["caminos_low"]))
    orden = np.argsort(banco.meta["fecha"].to_numpy(), kind="stable")
    banco = banco.subconjunto(orden)
    # normalizacion cross-sectional de los factores (si hay alguno no nulo)
    fac_cols = [c for c in GRUPOS_ESCALARES["factores"] if c.startswith("fac_")]
    idx_fac = [banco.nombres_feat.index(c) for c in fac_cols]
    if np.isfinite(banco.feat[:, idx_fac]).any() and np.nanstd(banco.feat[:, idx_fac]) > 0:
        from .factores import normalizar_cross_sectional
        normalizar_cross_sectional(banco)
    else:
        banco.feat[:, idx_fac] = np.nan_to_num(banco.feat[:, idx_fac], nan=0.0)
    return banco


def etiquetar(banco, registro, tolerancia_dias=3):
    """Marca los casos que caen sobre una figura conocida.

    Solo sirve para poder EXPLICAR la prediccion ("se parece a estos tres dobles
    techos") y para el diagnostico con datos sinteticos. La etiqueta NO entra en
    ninguna distancia ni en ningun modelo: si entrase, estarias dandole la
    respuesta al sistema.
    """
    if registro is None or len(registro) == 0:
        banco.meta["patron"] = ""
        return banco
    reg = registro.copy()
    reg["fecha"] = pd.to_datetime(reg["fecha"])
    etiquetas = np.array([""] * len(banco.meta), object)
    fechas = pd.to_datetime(banco.meta["fecha"]).to_numpy("datetime64[ns]")
    tickers = banco.meta["ticker"].to_numpy()
    tol = np.timedelta64(int(tolerancia_dias), "D")
    for t, grupo in reg.groupby("ticker"):
        idx = np.flatnonzero(tickers == t)
        if len(idx) == 0:
            continue
        for f, nombre in zip(grupo["fecha"].to_numpy("datetime64[ns]"),
                             grupo["patron"].to_numpy()):
            cerca = idx[np.abs(fechas[idx] - f) <= tol]
            etiquetas[cerca] = nombre
    banco.meta["patron"] = etiquetas
    return banco


def caso_vivo(precios, ctx=None, fuente_noticias=None, cfg=None, ticker=None, enriquecer=None):
    """Construye UN caso en la ultima barra disponible, sin exigir futuro.

    Es lo que se necesita para predecir de verdad el borde del grafico (el
    proximo tramo aun desconocido). A diferencia de 'construir', aqui no hay 'y'
    ni 'camino' porque el futuro todavia no ha pasado: se rellenan con NaN. El
    resto (forma, temporalidades superiores, indicadores, contexto) se calcula
    igual, siempre con datos hasta hoy.
    """
    cfg = cfg or Config()
    fuente_noticias = fuente_noticias or mod_noticias.SinNoticias()
    df = precios[ticker].dropna()
    n = len(df)
    L = cfg.largo
    if n < cfg.minimo_historia:
        raise RuntimeError(f"{ticker}: {n} barras, hacen falta al menos {cfg.minimo_historia}")

    rc = ctx.rasgos(ticker) if ctx is not None else None
    rn = mod_noticias.rasgos_sentimiento(fuente_noticias, ticker, df.index,
                                         cfg.dias_noticias, cfg.semivida_noticias) \
        if not isinstance(fuente_noticias, mod_noticias.SinNoticias) else None
    extra = enriquecer(ticker, df.index, df["close"]) if enriquecer is not None else None
    F = rasgos_escalares(df, rc, rn, rasgos_extra=extra)

    lc = np.log(df["close"].to_numpy(float))
    lv = np.log(np.maximum(df["volume"].to_numpy(float), 1.0))
    pos = np.array([n - 1])
    formas = forma(ventanas_de(lc, L)[pos - L + 1])
    formas_vol = forma(ventanas_de(lv, L)[pos - L + 1])
    Mf = F.to_numpy(float)[pos]
    atr_pct = F["atr_pct"].to_numpy(float)[pos]

    fechas = df.index[pos]
    tfs = {}
    for nombre, regla in cfg.temporalidades.items():
        m, _ = _formas_tf(df, regla, {"semanal": cfg.largo_semanal,
                                      "mensual": cfg.largo_mensual}.get(nombre, 20), fechas)
        tfs[nombre] = m

    meta = pd.DataFrame({"ticker": ticker, "fecha": fechas,
                         "sector": ctx.sector_de(ticker) if ctx is not None else "desconocido",
                         "close": df["close"].to_numpy(float)[pos], "atr_pct": atr_pct})
    y = pd.DataFrame({"ret": [np.nan], "direccion": [np.nan], "mfe": [np.nan], "mae": [np.nan],
                      "tb_label": [np.nan], "tb_ret": [np.nan], "tb_barras": [np.nan],
                      "tb_rr": [np.nan]})
    caminos = np.full((1, cfg.horizonte), np.nan, np.float32)
    banco = BancoCasos(cfg, formas, formas_vol, tfs, Mf.astype(np.float32), meta, y, caminos)
    banco.meta["patron"] = ""
    return banco
