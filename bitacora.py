"""Bitacora de operaciones: memoria RICA de cada trade para aprender del PORQUE.

No basta con saber si una operacion gano o perdio. Por cada entrada guardamos:
  - MOTIVO: una descripcion legible de por que se opero (RVOL, VWAP, sesion,
    conviccion, noticias) -> "por que ejecutaste esa operacion".
  - PATRON de velas: etiqueta del detector de figuras (patrones.figuras) sobre la
    ventana de entrada (doble suelo, rotura, tendencia...) -> "cual fue el patron".
  - FOTO: un PNG del grafico de velas en la entrada -> "guardate una foto".
  - VENTANA OHLC: las ultimas velas (para redibujar o re-analizar luego).
  - RESULTADO: se rellena al cerrarse (win/loss, P&L).

Con eso se APRENDE: que patrones y condiciones de entrada ganan y cuales no
(bitacora.aprender). Es memoria del porque, no solo del resultado.

Los ficheros viven en datos/snapshots/ (gitignored).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from patrones import figuras

DIR = "datos/snapshots"
OHLC = ["open", "high", "low", "close"]


def describir_motivo(feats, conviccion=None):
    """Descripcion legible de por que se opero, a partir de las features."""
    rv, dv = feats.get("rvol"), feats.get("dist_vwap")
    rs = feats.get("ret_sesion")
    sn = feats.get("sent_noticias", feats.get("sent", 0.0))
    partes = []
    if rv is not None:
        nivel = "muy alto" if rv >= 2 else "alto" if rv >= 1.5 else "moderado"
        partes.append(f"RVOL {rv:.2f} (volumen {nivel}, stock in play)")
    if dv is not None:
        partes.append(f"{'sobre' if dv >= 0 else 'bajo'} VWAP {dv*100:+.2f}%")
    if rs is not None:
        partes.append(f"sesion {rs*100:+.2f}%")
    if conviccion is not None:
        partes.append(f"conviccion {conviccion:+.2f}")
    if sn is not None:
        tono = "positivas" if sn > 0.1 else "negativas" if sn < -0.1 else "neutras"
        partes.append(f"noticias {tono} ({sn:+.2f})")
    return "; ".join(partes)


def patron_velas(df, n=30):
    """Etiqueta del patron de velas de la ventana (detector patrones.figuras)."""
    c = pd.Series(df["close"].astype(float)).tail(n).to_numpy()
    if len(c) < 5:
        return {"figura": "insuficiente", "tendencia": "n/a", "rotura": None, "figuras": []}
    forma = (c - c.mean()) / (c.std() + 1e-9)
    vol = df["volume"].tail(n).to_numpy() if "volume" in df else None
    d = figuras.describir(forma, vol)
    return {"figura": d.get("figura"), "tendencia": d.get("tendencia"),
            "rotura": d.get("rotura"), "figuras": list(d.get("figuras", []))}


def _foto(df, ticker, ruta, motivo="", n=40):
    """Dibuja un candlestick simple de la ventana de entrada y lo guarda en PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = df.tail(n)
    o, h, l, cl = [g[k].astype(float).to_numpy() for k in OHLC]
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i in x:
        col = "#1a7f37" if cl[i] >= o[i] else "#d1242f"
        ax.plot([i, i], [l[i], h[i]], color=col, lw=0.8)
        ax.plot([i, i], [o[i], cl[i]], color=col, lw=3.2, solid_capstyle="butt")
    ax.set_title(f"{ticker} — entrada", fontsize=10)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25)
    if motivo:
        ax.set_xlabel(motivo[:70], fontsize=6.5)
    fig.tight_layout()
    fig.savefig(ruta, dpi=72, bbox_inches="tight")
    plt.close(fig)


def registrar(ticker, df, feats, entrada, tp, sl, conviccion=None,
              intervalo="15Min", con_foto=True):
    """Guarda la bitacora de UNA entrada (JSON + foto PNG). Devuelve el id."""
    os.makedirs(DIR, exist_ok=True)
    ts = pd.Timestamp.now()
    sid = f"{ticker}_{ts.strftime('%Y%m%dT%H%M%S')}"
    motivo = describir_motivo(feats, conviccion)
    pat = patron_velas(df)
    velas = (df.tail(40)[OHLC + (["volume"] if "volume" in df else [])]
             .round(4).values.tolist()) if set(OHLC).issubset(df.columns) else []
    rec = {"id": sid, "ts": ts.isoformat(), "ticker": ticker, "intervalo": intervalo,
           "motivo": motivo, "patron": pat,
           "features": {k: (round(float(v), 5) if v is not None else None)
                        for k, v in feats.items()},
           "conviccion": float(conviccion) if conviccion is not None else None,
           "entrada": float(entrada), "tp": tp, "sl": sl,
           "velas": velas, "resultado": "open", "pnl": None}
    if con_foto and velas:
        foto = os.path.join(DIR, sid + ".png")
        try:
            _foto(df, ticker, foto, motivo)
            rec["foto"] = foto
        except Exception:
            pass
    json.dump(rec, open(os.path.join(DIR, sid + ".json"), "w"))
    return sid


def marcar_resultado(sid, resultado, pnl, salida=None):
    """Rellena el desenlace (win/loss, P&L) en la bitacora de esa entrada."""
    p = os.path.join(DIR, str(sid) + ".json")
    if not sid or not os.path.exists(p):
        return False
    try:
        rec = json.load(open(p))
        rec["resultado"] = resultado
        rec["pnl"] = float(pnl)
        if salida is not None:
            rec["salida"] = float(salida)
        json.dump(rec, open(p, "w"))
        return True
    except Exception:
        return False


def _cerradas():
    recs = []
    for f in glob.glob(os.path.join(DIR, "*.json")):
        try:
            r = json.load(open(f))
            if r.get("resultado") in ("win", "loss"):
                recs.append(r)
        except Exception:
            continue
    return recs


def aprender(min_pat=4):
    """Aprende de la bitacora: que PATRON de velas y que tendencia de entrada ganan.
    Devuelve el win-rate y P&L medio por patron (con muestra suficiente)."""
    recs = _cerradas()
    if not recs:
        return {"n": 0, "mensaje": "sin operaciones cerradas en la bitacora aun"}
    df = pd.DataFrame([{"figura": (r.get("patron") or {}).get("figura", "?"),
                        "tendencia": (r.get("patron") or {}).get("tendencia", "?"),
                        "gana": r["resultado"] == "win",
                        "pnl": float(r.get("pnl") or 0.0)} for r in recs])
    out = {"n": int(len(df)), "win_rate_global": float(df["gana"].mean()),
           "pnl_medio": float(df["pnl"].mean())}
    g = df.groupby("figura").agg(n=("gana", "size"), win_rate=("gana", "mean"),
                                 pnl_medio=("pnl", "mean"))
    g = g[g["n"] >= min_pat].sort_values("win_rate", ascending=False)
    out["por_patron"] = g
    if len(g):
        out["mejor_patron"] = str(g.index[0])
        out["peor_patron"] = str(g.index[-1])
    return out


if __name__ == "__main__":
    a = aprender()
    print(f"bitacora: {a.get('n', 0)} operaciones cerradas")
    if a.get("n"):
        print(f"win-rate global {a['win_rate_global']:.0%}  |  P&L medio ${a['pnl_medio']:+.1f}")
        if "por_patron" in a and len(a["por_patron"]):
            print("\nPor patron de velas:")
            print(a["por_patron"].round(3).to_string())
