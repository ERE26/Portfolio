"""AUDITORÍA DE STOPS (post-mortem del SL).

Cuando una intradía cierra por STOP-LOSS, 1 HORA DESPUÉS vuelve a mirar el valor para
aprender de la decisión, no solo del P&L:

  - SACADA EN FALSO: el mercado hizo un pull-back que tocó el stop pero luego el valor
    RECUPERÓ (volvió a nuestra entrada o más) -> el análisis era bueno; el fallo fue el
    stop demasiado tight / la localización de la entrada. Nos sacaron de un trade válido.

  - SL NOS SALVÓ: tras el stop el valor SIGUIÓ CAYENDO -> era un error de análisis/compra
    y el stop limitó una pérdida mayor. Hizo su trabajo.

  - NEUTRO: lateral en torno a la salida; el stop ni salvó ni costó.

Auto-limitado e idempotente: cada cierre por SL se audita UNA sola vez, cuando han pasado
>=60 min desde la salida y hay barras suficientes. El veredicto se guarda en
datos/sl_auditoria.csv (para la bitácora / el terminal) y se resume por pantalla.

  python auditoria_sl.py [--force]   (--force audita ya, sin esperar la hora; para pruebas)
"""
import os
import sys

import pandas as pd

from fuentes import alpaca
from fuentes.http import Cliente

LOG = "datos/trades_intradia.csv"
AUD = "datos/sl_auditoria.csv"
ESPERA_MIN = 60                 # ventana de re-chequeo: 1 hora tras la salida
_CLI = Cliente(ttl=45)          # barras frescas (no el caché de 30 días)

CAMPOS = ["chk_ts", "snapshot", "ticker", "entrada", "sl", "salida", "salida_ts",
          "atr_pct", "post_max", "post_min", "post_last", "recup_pct", "caida_pct",
          "veredicto", "detalle"]


def _ya_auditados():
    if not os.path.exists(AUD):
        return set()
    try:
        return set(pd.read_csv(AUD)["snapshot"].astype(str))
    except Exception:
        return set()


def _cierres_por_sl():
    """Filas del log que cerraron TOCANDO el stop (loss con salida <= sl)."""
    if not os.path.exists(LOG):
        return []
    d = pd.read_csv(LOG)
    d = d[d["resultado"] == "loss"]
    out = []
    for _, r in d.iterrows():
        try:
            sal, sl = float(r["salida"]), float(r["sl"])
        except (TypeError, ValueError):
            continue
        if sal <= 0 or sl <= 0 or sal > sl * 1.001:        # no se llenó en/bajo el stop
            continue
        if not isinstance(r.get("salida_ts"), str) or not r["salida_ts"]:
            continue
        out.append(r)
    return out


def _barras_post(ticker, desde_utc, hasta_utc):
    """Barras de 5 min ESTRICTAMENTE posteriores a la salida (hasta ahora/1h)."""
    ini = pd.Timestamp(desde_utc).tz_convert("America/New_York").tz_localize(None)
    fin = pd.Timestamp(hasta_utc).tz_convert("America/New_York").tz_localize(None)
    dia = ini.strftime("%Y-%m-%d")
    df = alpaca.historico(ticker, "5Min", desde=dia, cliente=_CLI, max_paginas=4)
    return df.loc[(df.index > ini) & (df.index <= fin)]


def _clasificar(entrada, sl, salida, tp, atr, post_max, post_min, post_last):
    recup_pct = post_max / entrada - 1.0                    # recuperación sobre la ENTRADA
    caida_pct = (salida - post_min) / entrada if post_min > 0 else 0.0   # caída extra bajo la SALIDA
    umbral_caida = 0.5 * atr if atr > 0 else 0.004          # media ATR (o 0.4%) de caída extra = real
    toco_tp = (tp is not None) and (post_max >= tp)
    if post_max >= entrada:
        det = f"recuperó a {post_max:.2f} (+{recup_pct*100:.1f}% s/entrada)"
        det += " y HABRÍA TOCADO TP" if toco_tp else ""
        if post_min < sl:
            det += f" (antes cayó a {post_min:.2f})"
        det += "  -> stop muy tight / mala localización; el análisis era bueno"
        return "SACADA_EN_FALSO", recup_pct, caida_pct, det
    if post_min <= salida and caida_pct >= umbral_caida:
        det = (f"siguió cayendo a {post_min:.2f} (-{caida_pct*100:.1f}% bajo la salida); "
               f"el stop limitó una pérdida mayor -> error de análisis/compra")
        return "SL_NOS_SALVO", recup_pct, caida_pct, det
    det = f"lateral en torno a {post_last:.2f}; el stop ni salvó ni costó"
    return "NEUTRO", recup_pct, caida_pct, det


def auditar(force=False):
    hechos = _ya_auditados()
    ahora = pd.Timestamp.now(tz="UTC")
    nuevos = []
    for r in _cierres_por_sl():
        snap = str(r["snapshot"])
        if snap in hechos and not force:
            continue
        sal_ts = pd.Timestamp(r["salida_ts"])
        if sal_ts.tzinfo is None:
            sal_ts = sal_ts.tz_localize("UTC")
        if (ahora - sal_ts).total_seconds() / 60.0 < ESPERA_MIN and not force:
            continue                                        # aún no ha pasado la hora
        entrada, sl, sal = float(r["entrada"]), float(r["sl"]), float(r["salida"])
        atr = float(r.get("atr_pct") or 0.0)
        try:
            tp = float(r["tp"])
        except (TypeError, ValueError):
            tp = None
        try:
            bars = _barras_post(r["ticker"], sal_ts, ahora)
        except Exception as e:
            print(f"  {r['ticker']}: sin barras aún ({str(e)[:40]}); reintenta otro ciclo")
            continue
        if bars is None or len(bars) < 2:
            continue                                        # sin datos (mercado cerrado) -> reintenta
        post_max = float(bars["high"].max())
        post_min = float(bars["low"].min())
        post_last = float(bars["close"].iloc[-1])
        ver, recup, caida, det = _clasificar(entrada, sl, sal, tp, atr, post_max, post_min, post_last)
        nuevos.append({"chk_ts": ahora.isoformat(), "snapshot": snap, "ticker": r["ticker"],
                       "entrada": entrada, "sl": sl, "salida": sal, "salida_ts": r["salida_ts"],
                       "atr_pct": atr, "post_max": round(post_max, 2), "post_min": round(post_min, 2),
                       "post_last": round(post_last, 2), "recup_pct": round(recup, 4),
                       "caida_pct": round(caida, 4), "veredicto": ver, "detalle": det})
    if nuevos:
        df_new = pd.DataFrame(nuevos, columns=CAMPOS)
        cab = not os.path.exists(AUD)
        os.makedirs("datos", exist_ok=True)
        df_new.to_csv(AUD, mode="a", header=cab, index=False)
    return nuevos


def resumen():
    """Conteo global por veredicto (para el terminal / bitácora)."""
    if not os.path.exists(AUD):
        return {"n": 0, "falso": 0, "salvo": 0, "neutro": 0}
    try:
        d = pd.read_csv(AUD)
    except Exception:
        return {"n": 0, "falso": 0, "salvo": 0, "neutro": 0}
    v = d["veredicto"].value_counts().to_dict()
    return {"n": int(len(d)), "falso": int(v.get("SACADA_EN_FALSO", 0)),
            "salvo": int(v.get("SL_NOS_SALVO", 0)), "neutro": int(v.get("NEUTRO", 0))}


if __name__ == "__main__":
    nuevos = auditar(force="--force" in sys.argv)
    if not nuevos:
        print("AUDITORÍA SL: nada nuevo por auditar (ningún stop con >=1h cumplida y datos).")
    else:
        print(f"AUDITORÍA SL: {len(nuevos)} stop(s) revisado(s) 1h después:")
        for n in nuevos:
            etq = {"SACADA_EN_FALSO": "⚠ SACADA EN FALSO", "SL_NOS_SALVO": "✓ SL NOS SALVÓ",
                   "NEUTRO": "· neutro"}[n["veredicto"]]
            print(f"  {n['ticker']:<6} {etq}: {n['detalle']}")
    r = resumen()
    if r["n"]:
        print(f"\n  Acumulado: {r['n']} stops auditados | {r['falso']} sacadas en falso "
              f"(stop tight) · {r['salvo']} nos salvaron · {r['neutro']} neutros")
