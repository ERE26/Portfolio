"""Registro de operaciones intradia y aprendizaje de la SELECTIVIDAD.

Idea: anotar cada entrada con sus caracteristicas (RVOL, distancia al VWAP, retorno
de sesion, ATR, sentimiento de noticias, hora) y su desenlace (gano/perdio al tocar
TP/SL). Con los resultados reales se analiza QUE caracteristicas separan a las
ganadoras y se afina el umbral de seleccion, para operar cada vez mas fino y elegir
solo lo mejor. No es tocar parametros a ciegas: es aprender de lo que pasa de verdad.

El archivo vive en datos/ (gitignored). Requiere algunas operaciones para que el
analisis sea fiable; hasta entonces se usan los umbrales a priori del estudio.
"""
import json
import os

import pandas as pd
import requests

from fuentes import alpaca_trade as AT

LOG = "datos/trades_intradia.csv"
CFG = "datos/selectividad.json"
COLS = ["ts", "ticker", "qty", "entrada", "tp", "sl", "rvol", "dist_vwap",
        "ret_sesion", "atr_pct", "sent_noticias", "hora", "snapshot", "salida",
        "salida_ts", "pnl", "resultado"]
MIN_TRADES = 25           # minimo para fiarse del analisis y afinar umbrales


def registrar_entrada(ticker, qty, precio, tp, sl, feats, snapshot=""):
    """Anota una entrada intradia con sus caracteristicas (resultado='open').

    snapshot: id de la bitacora (bitacora.registrar) con el porque, el patron de
    velas y la foto de la entrada. Enlaza el resultado con su contexto para aprender."""
    os.makedirs("datos", exist_ok=True)
    fila = {"ts": pd.Timestamp.now().isoformat(), "ticker": ticker, "qty": int(qty),
            "entrada": round(float(precio), 4), "tp": tp, "sl": sl,
            "rvol": round(float(feats.get("rvol", float("nan"))), 3),
            "dist_vwap": round(float(feats.get("dist_vwap", float("nan"))), 5),
            "ret_sesion": round(float(feats.get("ret_sesion", float("nan"))), 5),
            "atr_pct": round(float(feats.get("atr_pct", float("nan"))), 5),
            "sent_noticias": round(float(feats.get("sent_noticias", 0.0)), 3),
            "hora": pd.Timestamp.now().strftime("%H:%M"), "snapshot": snapshot,
            "salida": "", "salida_ts": "", "pnl": "", "resultado": "open"}
    df = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
    df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    df.to_csv(LOG, index=False)


def _fills():
    """Fills de la cuenta (ventas incluidas) desde Alpaca activities."""
    try:
        r = requests.get("https://paper-api.alpaca.markets/v2/account/activities/FILL",
                         headers=AT._headers(), params={"page_size": 100}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def reconciliar():
    """Cierra en el log las operaciones cuya venta ya ocurrio (TP/SL), calculando
    P&L y resultado. Empareja cada compra registrada con una venta posterior del
    mismo ticker (FIFO)."""
    if not os.path.exists(LOG):
        return 0
    df = pd.read_csv(LOG)
    # columnas de texto que se rellenan al cerrar: deben ser 'object' para poder
    # asignarles cadenas (si el CSV venia de antes podrian leerse como float/NaN).
    for col in ("salida", "salida_ts", "snapshot", "pnl", "resultado"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)
    abiertas = df[df["resultado"] == "open"]
    if len(abiertas) == 0:
        return 0
    ventas = [f for f in _fills() if f.get("side", "").startswith("sell")]
    ventas = sorted(ventas, key=lambda f: f.get("transaction_time", ""))
    usadas = set()
    cerradas = 0
    for i, row in abiertas.iterrows():
        for j, v in enumerate(ventas):
            if j in usadas or v.get("symbol") != row["ticker"]:
                continue
            if v.get("transaction_time", "") <= str(row["ts"]):
                continue
            precio_salida = float(v.get("price", 0))
            pnl = (precio_salida - float(row["entrada"])) * float(row["qty"])
            df.at[i, "salida"] = round(precio_salida, 4)
            df.at[i, "salida_ts"] = v.get("transaction_time", "")
            df.at[i, "pnl"] = round(pnl, 2)
            resultado = "win" if pnl > 0 else "loss"
            df.at[i, "resultado"] = resultado
            # enlaza el desenlace con la bitacora (porque + patron + foto) para aprender
            sid = str(row.get("snapshot", "") or "")
            if sid:
                try:
                    from bitacora import marcar_resultado
                    marcar_resultado(sid, resultado, pnl, precio_salida)
                except Exception:
                    pass
            usadas.add(j)
            cerradas += 1
            break
    if cerradas:
        df.to_csv(LOG, index=False)
    return cerradas


def analizar_selectividad():
    """De las operaciones cerradas, estima que nivel de RVOL separa mejor a las
    ganadoras y propone un umbral afinado. Devuelve dict con el diagnostico."""
    if not os.path.exists(LOG):
        return {"n": 0, "mensaje": "sin datos aun"}
    df = pd.read_csv(LOG)
    cerr = df[df["resultado"].isin(["win", "loss"])].copy()
    n = len(cerr)
    if n < MIN_TRADES:
        return {"n": n, "mensaje": f"faltan datos ({n}/{MIN_TRADES}) para afinar; "
                "se usan los umbrales a priori", "win_rate": float((cerr["resultado"] == "win").mean()) if n else float("nan")}
    cerr["gana"] = (cerr["resultado"] == "win").astype(int)
    # buscar el umbral de RVOL que maximiza el win-rate manteniendo cobertura razonable
    mejor = None
    for u in [1.3, 1.5, 1.7, 2.0, 2.3]:
        sub = cerr[cerr["rvol"] >= u]
        if len(sub) >= max(10, n // 4):
            wr = sub["gana"].mean()
            if mejor is None or wr > mejor[1]:
                mejor = (u, float(wr), len(sub))
    base_wr = float(cerr["gana"].mean())
    out = {"n": n, "win_rate_global": base_wr}
    if mejor and mejor[1] > base_wr + 0.05:
        out["umbral_rvol"] = mejor[0]
        out["win_rate_umbral"] = mejor[1]
        out["cobertura"] = mejor[2]
        out["mensaje"] = (f"RVOL>={mejor[0]} sube el acierto a {mejor[1]:.0%} "
                          f"(vs {base_wr:.0%} global) en {mejor[2]} ops -> umbral afinado")
    else:
        out["mensaje"] = f"sin mejora clara por RVOL (acierto global {base_wr:.0%}); umbral sin cambios"
    return out


def tickers_en_cooldown(minutos=60):
    """Nombres que han tocado SL (perdida) hace poco: NO re-entrar enseguida.

    Un nombre que acaba de saltarte el stop suele estar en whipsaw (RVOL alto que
    revierte), no en tendencia limpia; recomprarlo porque 'la foto sigue guapa' es
    el churn que pierde con costes. Mejor esperar a que se calme o demuestre fuerza
    nueva. Usa la hora de salida real (salida_ts) y, si falta, la de entrada."""
    if not os.path.exists(LOG):
        return set()
    try:
        df = pd.read_csv(LOG)
    except Exception:
        return set()
    if "resultado" not in df:
        return set()
    perd = df[df["resultado"] == "loss"].copy()
    if len(perd) == 0:
        return set()
    cuando = perd.get("salida_ts")
    ts = pd.to_datetime(cuando, errors="coerce", utc=True) if cuando is not None else pd.NaT
    ent = pd.to_datetime(perd["ts"], errors="coerce", utc=True)
    ts = ts.fillna(ent) if hasattr(ts, "fillna") else ent
    limite = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutos)
    return set(perd.loc[ts >= limite, "ticker"].astype(str))


def tiempos_entrada():
    """Hora de la ULTIMA compra por simbolo (proxy de la entrada de la posicion
    abierta), desde los fills de Alpaca. Sirve para medir cuanto lleva abierta una
    operacion intradia (para el time-stop)."""
    buys = [f for f in _fills() if str(f.get("side", "")).startswith("buy")]
    ult = {}
    for f in buys:
        t, ts = f.get("symbol"), f.get("transaction_time", "")
        if t and ts and (t not in ult or ts > ult[t]):
            ult[t] = ts
    return {t: pd.to_datetime(ts, utc=True, errors="coerce") for t, ts in ult.items()}


def memoria_por_ticker():
    """Historial REALIZADO por ticker: memoria persistente entre ciclos y dias.

    Para cada nombre con operaciones cerradas: nº de ops, aciertos, win-rate
    ENCOGIDO hacia 0.5 (Bayes, para no sobre-reaccionar con pocas ops) y P&L total.
    Es lo que da MEMORIA al selector: un nombre que repite perdidas pierde
    prioridad; uno que funciona la gana. Con pocas ops se queda casi neutro."""
    if not os.path.exists(LOG):
        return {}
    try:
        df = pd.read_csv(LOG)
    except Exception:
        return {}
    if "resultado" not in df:
        return {}
    cerr = df[df["resultado"].isin(["win", "loss"])]
    K = 4.0                                   # fuerza del prior (ops neutras equivalentes)
    out = {}
    for t, g in cerr.groupby("ticker"):
        n = int(len(g)); w = int((g["resultado"] == "win").sum())
        pnl = float(pd.to_numeric(g["pnl"], errors="coerce").fillna(0).sum())
        out[str(t)] = {"n": n, "wins": w, "win_rate_shrunk": (w + 0.5 * K) / (n + K),
                       "pnl_total": pnl}
    return out


def ajuste_memoria(tickers, peso=0.6):
    """Ajuste de convicción por memoria realizada por ticker, en [-peso, +peso].

    0 si no hay historial (neutro); penaliza perdedores crónicos y premia a los que
    funcionan. Se suma al score de convicción, así el ranking RECUERDA entre ciclos."""
    mem = memoria_por_ticker()
    return {str(t): (peso * (2 * mem[str(t)]["win_rate_shrunk"] - 1)) if str(t) in mem else 0.0
            for t in tickers}


def guardar_umbral(umbral_rvol):
    json.dump({"rvol_fuerte": float(umbral_rvol)}, open(CFG, "w"))


def cargar_umbral(defecto=1.5):
    try:
        return float(json.load(open(CFG)).get("rvol_fuerte", defecto))
    except Exception:
        return defecto


if __name__ == "__main__":
    print("reconciliadas:", reconciliar())
    print(analizar_selectividad())
