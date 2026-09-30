"""Fuente de noticias reales, compatible con patrones.noticias.FuenteNoticias.

Combina titulares por valor (Yahoo) y, opcionalmente, el calendario macro de
Forex Factory como contexto. Puntua los titulares con el mismo lexico del
proyecto.

Limitacion honesta y deliberada: estas noticias son RECIENTES (dias), no un
historico de anos. Por eso esta fuente esta pensada para la prediccion EN VIVO
del ultimo tramo, no para el entrenamiento walk-forward. El sistema respeta la
regla de no mirar al futuro igual que siempre: solo devuelve titulares con fecha
anterior a la de la consulta.
"""
import pandas as pd

from patrones.noticias import FuenteNoticias, puntuar_titular
from .http import Cliente
from . import yahoo


class NoticiasReales(FuenteNoticias):
    def __init__(self, tickers, cliente=None, n_por_valor=15):
        self.cli = cliente or Cliente(ttl=1800)
        filas = []
        for t in tickers:
            try:
                tt = yahoo.titulares(t, n_por_valor, cliente=self.cli)
                for _, r in tt.iterrows():
                    filas.append({"fecha": r["fecha"], "ticker": t, "titular": r["titular"],
                                  "puntuacion": puntuar_titular(r["titular"]),
                                  "fuente": r.get("fuente", "")})
            except Exception as e:
                print(f"  noticias {t}: fallo -> {e}")
        self.df = (pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)
                   if filas else pd.DataFrame(columns=["fecha", "ticker", "titular", "puntuacion"]))

    def titulares(self, ticker, hasta, dias=30):
        d = self.df
        if d.empty:
            return d
        m = (d["ticker"].astype(str) == str(ticker)) & (d["fecha"] < hasta) & \
            (d["fecha"] >= hasta - pd.Timedelta(days=dias))
        return d.loc[m, ["fecha", "titular", "puntuacion"]]

    def resumen_vivo(self, ticker, dias=7):
        """Titulares recientes con su sentimiento, para el informe de prediccion."""
        d = self.df[self.df["ticker"] == ticker] if not self.df.empty else self.df
        if d.empty:
            return "  Sin titulares recientes."
        d = d[d["fecha"] >= d["fecha"].max() - pd.Timedelta(days=dias)].sort_values("fecha", ascending=False)
        tono = d["puntuacion"].mean()
        etq = "positivo" if tono > 0.15 else "negativo" if tono < -0.15 else "neutro"
        lineas = [f"  Titulares recientes ({len(d)}, tono medio {tono:+.2f} -> {etq}):"]
        for _, r in d.head(5).iterrows():
            lineas.append(f"    {r['fecha'].date()}  [{r['puntuacion']:+.2f}]  {r['titular'][:72]}")
        return "\n".join(lineas)
