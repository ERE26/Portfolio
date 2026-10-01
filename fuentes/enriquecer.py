"""Enriquecedor de features externos: fundamentales (SEC) + macro (FRED).

Devuelve un callable(ticker, fechas) -> {"fundamentales": df, "macro": df} listo
para pasar a ventanas.construir(enriquecer=...). Cachea por ticker para no repetir
llamadas, y degrada con elegancia: si SEC no tiene el ticker (divisa, fondo) o
FRED no tiene clave, el grupo correspondiente queda en 0 (neutro).
"""
import pandas as pd

from .http import Cliente
from . import sec, fred


class Enriquecedor:
    def __init__(self, con_sec=True, con_fred=True, con_13f=False, clave_fred=None, verbose=False):
        self.cli = Cliente(ttl=7 * 24 * 3600)
        self.con_sec = con_sec
        self.con_fred = con_fred
        self.con_13f = con_13f
        self.clave_fred = clave_fred
        self.verbose = verbose
        self._cik = None
        self._macro_cache = {}
        self._carteras = None

    @property
    def cik(self):
        if self._cik is None and self.con_sec:
            try:
                self._cik = sec.mapa_cik(self.cli)
            except Exception as e:
                if self.verbose:
                    print(f"  SEC no disponible: {e}")
                self._cik = {}
        return self._cik or {}

    def _macro(self, fechas):
        clave = (str(fechas[0]), str(fechas[-1]), len(fechas))
        if clave not in self._macro_cache:
            if not self.con_fred:
                self._macro_cache[clave] = None
            else:
                try:
                    self._macro_cache[clave] = fred.rasgos_macro(fechas, self.clave_fred, self.cli)
                except Exception as e:
                    if self.verbose:
                        print(f"  FRED no disponible: {e}")
                    self._macro_cache[clave] = None
        return self._macro_cache[clave]

    def __call__(self, ticker, fechas):
        out = {}
        if self.con_sec and ticker in self.cik:
            try:
                out["fundamentales"] = sec.rasgos_fundamentales(ticker, fechas,
                                                                self.cik[ticker], self.cli)
            except Exception:
                pass
        m = self._macro(fechas) if self.con_fred else None
        if m is not None:
            out["macro"] = m
        if self.con_13f:
            try:
                from . import form13f
                if self._carteras is None:
                    self._carteras = form13f.cargar_carteras(self.cli, self.verbose)
                out["institucional"] = form13f.señal_ticker(ticker, fechas, self._carteras)
            except Exception as e:
                if self.verbose:
                    print(f"  13F no disponible: {e}")
        return out
