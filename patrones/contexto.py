"""Contexto de mercado: regimen del indice, sector y fuerza relativa.

Un mismo dibujo en el grafico no significa lo mismo si el mercado entero esta
cayendo o si el valor lleva seis meses siendo el mas fuerte de su sector. Este
modulo calcula ese entorno, siempre con datos hasta la barra actual.
"""
import numpy as np
import pandas as pd

from . import indicadores as ind


class Contexto:
    def __init__(self, precios, meta, indice=None, ventana_percentil=504):
        self.precios = precios
        self.meta = meta.set_index("ticker") if "ticker" in meta.columns else meta
        self.ventana_percentil = ventana_percentil
        self.indice = indice if indice is not None else self._indice_sintetico()
        self._idx = self._rasgos_indice()
        self._sectores = self._indices_sector()

    # ---------------------------------------------------------------- indices
    def _indice_sintetico(self):
        """Si no hay indice, se construye uno equiponderado con los propios valores."""
        cierres = pd.DataFrame({t: d["close"] for t, d in self.precios.items()}).sort_index()
        norm = cierres / cierres.apply(lambda c: c.loc[c.first_valid_index()])
        cerrado = norm.mean(axis=1) * 1000
        return pd.DataFrame({"open": cerrado, "high": cerrado, "low": cerrado,
                             "close": cerrado, "volume": 0.0})

    def _indices_sector(self):
        """Un indice equiponderado por sector, para medir fuerza relativa."""
        out = {}
        for sector, grupo in self.meta.groupby("sector"):
            miembros = [t for t in grupo.index if t in self.precios]
            if not miembros:
                continue
            cierres = pd.DataFrame({t: self.precios[t]["close"] for t in miembros}).sort_index()
            norm = cierres / cierres.apply(lambda c: c.loc[c.first_valid_index()])
            out[sector] = norm.mean(axis=1)
        return out

    def _rasgos_indice(self):
        c = self.indice["close"]
        vol = ind.volatilidad_realizada(c, 20)
        return pd.DataFrame({
            "mkt_sobre_sma200": (c > ind.sma(c, 200)).astype(float),
            "mkt_ret_20": np.log(c / c.shift(20)),
            "mkt_ret_60": np.log(c / c.shift(60)),
            # percentil de volatilidad dentro de los ultimos 2 anos: 1 = panico
            "mkt_vol_percentil": vol.rolling(self.ventana_percentil, min_periods=60).rank(pct=True),
            "mkt_dd": c / c.cummax() - 1.0,
        })

    # -------------------------------------------------------------- por valor
    def sector_de(self, ticker):
        if ticker in self.meta.index and "sector" in self.meta.columns:
            return str(self.meta.loc[ticker, "sector"])
        return "desconocido"

    def rasgos(self, ticker):
        """DataFrame de contexto alineado al calendario del valor."""
        df = self.precios[ticker]
        c = df["close"]
        out = self._idx.reindex(c.index).ffill()
        for n in (20, 60):
            out[f"fuerza_rel_indice_{n}"] = np.log(c / c.shift(n)) - out[f"mkt_ret_{n}"]
        sec = self._sectores.get(self.sector_de(ticker))
        if sec is None:
            out["fuerza_rel_sector_20"] = 0.0
        else:
            s = sec.reindex(c.index).ffill()
            out["fuerza_rel_sector_20"] = np.log(c / c.shift(20)) - np.log(s / s.shift(20))
        return out.drop(columns=["mkt_ret_60"])
