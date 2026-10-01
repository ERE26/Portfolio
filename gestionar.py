"""Gestion ACTIVA del intradia + escaneo de mas oportunidades (intradia y diario).

Pensado para ejecutarse cada ~25 min mientras el mercado esta abierto:
  1. Revisa cada posicion intradia (con bracket): si la tendencia se gira (pierde
     el VWAP o la sesion se vuelve negativa) la cierra para cortar perdida; si va
     bien la deja correr (el bracket protege TP/SL).
  2. Escanea un universo AMPLIADO buscando nuevas oportunidades intradia (1h/15m)
     y diarias (momentum), que todavia no esten en cartera.

Uso:
  python gestionar.py            # dry-run: revisa y reporta, no cierra ni abre
  python gestionar.py --gestionar  # cierra las intradia que se giran (no abre solo)
"""
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca, alpaca_trade as AT
from patrones import indicadores as ind
from patrones.intradia import rasgos_intradia, rvol
from operativa_intradia import CONFIG, LIQUIDOS

# universo ampliado para buscar mas oportunidades (grandes + mas sectores)
UNIVERSO = sorted(set(LIQUIDOS + [
    "ORCL", "CRM", "ADBE", "WFC", "C", "AXP", "V", "MA", "COP", "PFE", "MRK", "ABBV",
    "HD", "MCD", "NKE", "KO", "PEP", "PG", "WMT", "CAT", "BA", "DIA", "IWM", "XLK", "XLF",
    "DE", "GE", "MU", "QCOM", "INTC", "T", "VZ", "DIS", "NFLX", "PYPL"]))


def universo_dinamico():
    """Universo operable = la lista base MAS los descubrimientos recientes de la
    watchlist. Asi el radar crece y rota en vez de mirar siempre los mismos nombres."""
    try:
        from descubrir import watchlist_vigente
        extra = watchlist_vigente()
    except Exception:
        extra = []
    return sorted(set(UNIVERSO) | set(extra))


def señal(ticker, intervalo="15Min"):
    """Re-analiza un ticker con las velas nuevas. Devuelve indicadores actuales Y
    señales de TENDENCIA (sostenidas), para distinguir una reversion real de una
    bajada organica puntual."""
    cfg = CONFIG[intervalo]
    desde = (pd.Timestamp.now() - pd.Timedelta(days=cfg["dias"])).strftime("%Y-%m-%d")
    df = alpaca.historico(ticker, intervalo, desde=desde, max_paginas=6)
    if len(df) < 20:
        raise RuntimeError("pocos datos")
    r = rasgos_intradia(df)
    c = df["close"]
    dv = r["intra_dist_vwap"]
    precio = float(c.iloc[-1])
    atr = float(ind.atr(df["high"], df["low"], df["close"], 14).iloc[-1])
    # pendiente corta: regresion lineal de los ultimos 5 cierres (log)
    y = np.log(c.iloc[-5:].to_numpy()); x = np.arange(len(y))
    pend = float(np.polyfit(x, y, 1)[0]) if len(y) >= 3 else 0.0
    bajo_vwap = float((dv.iloc[-4:] < 0).mean())      # fraccion de las ultimas 4 bajo VWAP
    return {"precio": precio, "dist_vwap": float(dv.iloc[-1]),
            "rvol": float(rvol(df).iloc[-1]), "ret_sesion": float(r["intra_ret_sesion"].iloc[-1]),
            "atr_pct": atr / precio if precio else 0.0, "pend": pend, "bajo_vwap": bajo_vwap}


# TIME-STOP intradia: un setup intradia tiene VIDA UTIL CORTA. Mantener una
# posicion horas pierde el edge original (el pico de RVOL y el momentum de VWAP se
# disipan): ya no es la estrategia que la justifico, solo ocupa capital y riesgo.
# Se cierra por tiempo para rotar a una oportunidad fresca.
# Validado (figura 28): mantener mas tiempo NO mejora el EV/op, lo DEGRADA (el
# setup decae). La operacion media se resuelve en ~1h; mas alla pierde el edge.
MAX_VERDE_MIN = 90      # >=1.5h EN VERDE -> recoge el positivo y rota (no esperes a que se gire)
MAX_DURA_MIN = 150      # >=2.5h -> time-stop duro: el setup ha caducado, cierra pase lo que pase


def revisar_intradia():
    """Re-analiza cada posicion intradia con las velas nuevas y decide si cerrar.

    Cierra si (a) se CONFIRMA una reversion (sostenida bajo VWAP + pendiente baja +
    claramente bajo VWAP), o (b) se dispara el TIME-STOP: lleva mucho tiempo abierta
    (en verde >=2.5h -> recoge y rota; en cualquier caso >=3.5h -> caducada). Una
    sola barra roja NO cierra (para eso esta el SL del bracket); mantener una
    organica esta bien, pero mantenerla ETERNAMENTE pierde la estrategia intradia."""
    ordenes = AT.ordenes_abiertas()
    simbolos = sorted(set(o["symbol"] for o in ordenes))
    pos = AT.posiciones()
    try:
        from registro import tiempos_entrada
        entradas = tiempos_entrada()
    except Exception:
        entradas = {}
    ahora = pd.Timestamp.now(tz="UTC")
    out = []
    for t in simbolos:
        if t not in pos:
            continue
        try:
            s = señal(t, "15Min")
        except Exception:
            continue
        # cuanto lleva abierta (minutos) y P&L %
        mins = None
        if t in entradas and pd.notna(entradas[t]):
            mins = (ahora - entradas[t]).total_seconds() / 60.0
        valor, pl = pos[t]["valor"], pos[t]["pl"]
        pl_pct = pl / max(valor - pl, 1e-9)
        reversion = (s["bajo_vwap"] >= 0.75) and (s["pend"] < 0) and (s["dist_vwap"] < -0.003)
        time_stop = False
        if mins is not None:
            if mins >= MAX_DURA_MIN:
                time_stop = True
            elif mins >= MAX_VERDE_MIN and pl_pct >= 0:
                time_stop = True
        cerrar = reversion or time_stop
        if reversion:
            motivo = "REVERSION confirmada (sostenida bajo VWAP + pendiente baja)"
        elif time_stop and pl_pct >= 0 and (mins is None or mins < MAX_DURA_MIN):
            motivo = f"TIME-STOP en verde ({mins:.0f} min): recoge +{pl_pct*100:.2f}% y rota"
        elif time_stop:
            motivo = f"TIME-STOP duro ({mins:.0f} min): el setup ha caducado"
        elif s["dist_vwap"] < 0 or s["ret_sesion"] < 0:
            motivo = "bajada organica (se mantiene; SL del bracket protege)"
        else:
            motivo = "tendencia intacta"
        out.append({"ticker": t, "pl": pl, "cerrar": cerrar, "motivo": motivo,
                    "dist_vwap": s["dist_vwap"], "ret_sesion": s["ret_sesion"],
                    "rvol": s["rvol"], "pend": s["pend"], "bajo_vwap": s["bajo_vwap"],
                    "min_abierta": mins if mins is not None else float("nan")})
    return out


def escanear(intervalo="15Min", excluir=(), n=6):
    """Nuevas oportunidades intradia en el universo ampliado (no en cartera)."""
    cfg = CONFIG[intervalo]
    filas = []
    for t in universo_dinamico():
        if t in excluir:
            continue
        try:
            s = señal(t, intervalo)
        except Exception:
            continue
        cand = (s["dist_vwap"] > 0) and (s["ret_sesion"] > 0) and (s["rvol"] >= cfg["rvol_min"])
        if cand:
            filas.append({"ticker": t, "score": s["rvol"] * s["ret_sesion"], **s})
    d = pd.DataFrame(filas)
    return d.sort_values("score", ascending=False).head(n).reset_index(drop=True) if len(d) else d


def escaneo_diario(n=8):
    """Top momentum diario del universo (oportunidades swing)."""
    from fuentes.descarga import cargar_mercado
    from patrones import transversal as TV
    pre, _ = cargar_mercado("datos/mercado", intervalo="1d")
    panel = TV.panel_precios(pre)
    mom = TV.score_momentum(panel).dropna().sort_values(ascending=False)
    return mom.head(n)


def _sentimiento(ticker):
    """Sentimiento de noticias recientes (catalizador). 0.0 si no hay datos."""
    try:
        from fuentes import noticias_vivo
        return float(noticias_vivo.noticias(ticker, n=3)["sentimiento"])
    except Exception:
        return 0.0


def _z(s):
    """z-score transversal; 0 si no hay dispersion (evita dividir por ~0)."""
    s = pd.Series(list(s), dtype=float)
    sd = float(s.std(ddof=0))
    return (s - s.mean()) / sd if sd > 1e-9 else s * 0.0


def rankear_conviccion(opo, umbral_fuerte, excluir=(), gate_neg=-0.35):
    """Ordena los candidatos FUERTES por un score de CONVICCION por confluencia y
    aplica un GATE de catalizador (no comprar contra una noticia claramente mala).

    conviccion = z(rvol) + z(ret_sesion) + 0.5*catalizador, con catalizador = el
    sentimiento de noticias acotado a [-1, 1] (positivo suma, negativo resta). Si
    el sentimiento es muy negativo (< gate_neg) el candidato se DESCARTA: un RVOL
    alto con mala noticia suele ser ruido/trampa, no oportunidad. La evidencia es
    clara (docs/aprendizaje_quant.md): ~95% del alfa esta en QUE eliges; las
    noticias valen como filtro de 'razon real', no como senal direccional. Elegir
    las mejores por confluencia es exactamente calidad > cantidad."""
    if opo is None or len(opo) == 0:
        return opo
    d = opo[(opo["rvol"] >= umbral_fuerte) & (~opo["ticker"].isin(set(excluir)))].copy()
    if len(d) == 0:
        return d
    d["sent"] = [_sentimiento(t) for t in d["ticker"]]
    d = d[d["sent"] >= gate_neg].copy()        # gate: fuera catalizador muy negativo
    if len(d) == 0:
        return d
    cat = d["sent"].clip(-1, 1)
    # MEMORIA por ticker: ajuste por el historial realizado (perdedores cronicos
    # bajan, ganadores suben). Da memoria al selector entre ciclos, no solo la foto.
    from registro import ajuste_memoria
    aj = ajuste_memoria(list(d["ticker"]))
    d["mem"] = [aj.get(str(t), 0.0) for t in d["ticker"]]
    d["conviccion"] = (_z(d["rvol"]).to_numpy() + _z(d["ret_sesion"]).to_numpy()
                       + 0.5 * cat.to_numpy() + d["mem"].to_numpy())
    return d.sort_values("conviccion", ascending=False).reset_index(drop=True)


def main(gestionar=False, rotar=False, reserva_por_pos=20000.0):
    pos = AT.posiciones()
    rev = revisar_intradia()
    print(f"Posiciones intradia (bracket): {len(rev)}")
    a_cerrar = []
    for d in rev:
        print(f"  {d['ticker']:<6} PL ${d['pl']:+.0f} | VWAP {d['dist_vwap']*100:+.2f}% "
              f"sesion {d['ret_sesion']*100:+.2f}% pend {d['pend']*100:+.3f}% "
              f"bajoVWAP {d['bajo_vwap']*100:.0f}% -> {d['motivo']}")
        if d["cerrar"]:
            a_cerrar.append(d["ticker"])
    if a_cerrar and gestionar:
        print(f"Cerrando {a_cerrar} (reversion confirmada)...")
        print(" ", AT.cerrar_posiciones(a_cerrar, confirmar=True))
    elif a_cerrar:
        print(f"[dry-run] cerraria: {a_cerrar}")

    print("\nNUEVAS OPORTUNIDADES INTRADIA (15m, universo ampliado):")
    opo = escanear("15Min", excluir=set(pos))
    if len(opo):
        for _, s in opo.iterrows():
            print(f"  {s['ticker']:<6} VWAP {s['dist_vwap']*100:+.2f}% RVOL {s['rvol']:.2f} "
                  f"sesion {s['ret_sesion']*100:+.2f}%")
    else:
        print("  (ninguna ahora)")

    # REINVERSION selectiva: mantener el book intradia hasta N posiciones, pero SOLO
    # con candidatos fuertes. Cubre tanto lo que cerramos por reversion como lo que
    # cerro solo el bracket al tocar TP/SL. Con cabeza: calidad > cantidad; si no hay
    # un candidato fuerte, se queda en efectivo (no operar por operar).
    from registro import cargar_umbral, registrar_entrada, tickers_en_cooldown
    N_OBJETIVO = 3
    RVOL_FUERTE = cargar_umbral(1.5)      # umbral afinado por el aprendizaje (o 1.5 a priori)
    if gestionar and rotar:
        held = set(pos) - set(a_cerrar)
        # cooldown: no re-entrar en un nombre que acaba de tocar SL (evita el churn
        # de comprar-perder-recomprar en un whipsaw; con cabeza > foto bonita).
        cooldown = tickers_en_cooldown(minutos=60)
        if cooldown:
            print(f"\nEn cooldown tras SL (no re-entrar aun): {sorted(cooldown)}")
        intradia_vivas = len(rev) - len(a_cerrar)      # brackets que siguen abiertos
        libres = max(N_OBJETIVO - intradia_vivas, 0)
        # ranking por CONVICCION (confluencia) + gate de catalizador de noticias
        fuertes = rankear_conviccion(opo, RVOL_FUERTE, excluir=held | cooldown)
        if libres and len(fuertes):
            cfg = CONFIG["15Min"]
            cash = float(AT.cuenta()["cash"])
            por_pos = min(reserva_por_pos, cash / libres) if libres else 0.0
            # mas candidatos de los huecos, por si alguno falla; TP/SL con el precio ACTUAL
            abrir = fuertes.head(libres + 3)
            px_now = AT.precio_actual(list(abrir["ticker"]))
            print(f"\nReinvirtiendo (hasta {libres}) por CONVICCION (RVOL>={RVOL_FUERTE} "
                  f"+ gate de noticias):")
            colocadas = 0
            for _, s in abrir.iterrows():
                if colocadas >= libres:
                    break
                precio = float(px_now.get(s["ticker"], s["precio"]))   # precio de mercado ahora
                qty = int(por_pos / precio)
                if qty <= 0:
                    continue
                tp = round(precio * (1 + cfg["k_pt"] * s["atr_pct"]), 2)
                sl = round(precio * (1 - cfg["k_sl"] * s["atr_pct"]), 2)
                try:
                    r = AT.enviar_bracket(s["ticker"], qty, tp, sl, confirmar=True)
                    if r.get("id"):
                        colocadas += 1
                        sn = float(s.get("sent", 0.0))   # sentimiento ya calculado en el ranking
                        feats = {"rvol": s["rvol"], "dist_vwap": s["dist_vwap"],
                                 "ret_sesion": s["ret_sesion"], "atr_pct": s["atr_pct"],
                                 "sent_noticias": sn}
                        # BITACORA: guarda el porque, el patron de velas y una foto
                        # de la entrada (para aprender del contexto, no solo del PL).
                        sid = ""
                        try:
                            import bitacora
                            dsd = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
                            dfw = alpaca.historico(s["ticker"], "15Min", desde=dsd, max_paginas=4)
                            sid = bitacora.registrar(s["ticker"], dfw, feats, precio, tp, sl,
                                                     conviccion=float(s.get("conviccion", float("nan"))))
                        except Exception:
                            sid = ""
                        registrar_entrada(s["ticker"], qty, precio, tp, sl, feats, snapshot=sid)
                        pat = ""
                        if sid:
                            try:
                                import bitacora as _b, json as _j, os as _o
                                _r = _j.load(open(_o.path.join(_b.DIR, sid + ".json")))
                                pat = f" | patron: {_r['patron'].get('figura')}/{_r['patron'].get('tendencia')}"
                            except Exception:
                                pat = ""
                        print(f"  LARGO {qty} {s['ticker']} @~${precio:.2f} -> TP ${tp} / SL ${sl}"
                              f"  (conv {s.get('conviccion', float('nan')):+.2f}, sent {sn:+.2f}){pat}  OK")
                    else:
                        print(f"  {s['ticker']}: {r}")
                except Exception as e:
                    print(f"  {s['ticker']}: salto ({str(e)[:50]})")
        elif libres:
            print(f"\n{libres} hueco(s) intradia libre(s), pero sin candidato fuerte "
                  f"(RVOL>={RVOL_FUERTE}): se mantiene el efectivo (calidad > cantidad).")

    print("\nTOP MOMENTUM DIARIO (oportunidades swing):")
    for t, m in escaneo_diario().items():
        print(f"  {t:<6} momento {m*100:+.1f}%" + ("  [ya en cartera]" if t in pos else ""))


if __name__ == "__main__":
    main(gestionar="--gestionar" in sys.argv, rotar="--rotar" in sys.argv)
