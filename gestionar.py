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
from fuentes.http import Cliente
from patrones import indicadores as ind
from patrones.intradia import rasgos_intradia, rvol
from operativa_intradia import CONFIG, LIQUIDOS

# Cliente con TTL CORTO para las barras intradía: sin esto, alpaca.historico usa un
# Cliente con caché de 30 días y la señal (VWAP, sesión, RVOL, pendiente) queda
# CONGELADA desde el primer escaneo del día. 45s deduplica dentro de un ciclo y
# refresca de un ciclo al siguiente (cada ~5 min).
_CLI_INTRA = Cliente(ttl=45)

# universo ampliado: cuantos mas valores miremos, mas podemos ELEGIR los mejores
# (el filtro de calidad decide, no el tamano del universo). Grandes + varios sectores.
UNIVERSO = sorted(set(LIQUIDOS + [
    "ORCL", "CRM", "ADBE", "WFC", "C", "AXP", "V", "MA", "COP", "PFE", "MRK", "ABBV",
    "HD", "MCD", "NKE", "KO", "PEP", "PG", "WMT", "CAT", "BA", "DIA", "IWM", "XLK", "XLF",
    "DE", "GE", "MU", "QCOM", "INTC", "T", "VZ", "DIS", "NFLX", "PYPL",
    # ampliacion: mas liquidos de tecnologia, semis, consumo, energia, salud, financieras
    "SHOP", "PLTR", "SNOW", "CRWD", "PANW", "ANET", "SMCI", "ARM", "DELL", "TXN",
    "LRCX", "KLAC", "ADI", "UBER", "COIN", "ABNB", "MAR", "BKNG", "SBUX", "LOW",
    "TGT", "GM", "F", "GILD", "AMGN", "VRTX", "ISRG", "LMT", "RTX", "SLB",
    "OXY", "PSX", "MPC", "FCX", "NEM", "XLE", "XLV", "XLY", "SMH", "QQQ"]))

# EUROPA intradia: Alpaca no da acceso directo a las bolsas europeas (Xetra,
# Euronext...), pero SI a los ADRs de empresas europeas y a los ETFs de Europa
# cotizados en EE.UU. (verificado tradable + con datos de 15m). Se operan durante el
# horario de EE.UU. con la MISMA estrategia de confluencia. Da exposicion europea
# real. (Ojo: el subyacente europeo ya ha cotizado antes, asi que estos ADRs/ETFs
# suelen abrir con gap por la sesion europea: util para catalizadores.)
EUROPA = [
    # ADRs de grandes empresas europeas (NYSE/NASDAQ)
    "ASML", "SAP", "NVO", "NVS", "AZN", "SHEL", "BP", "TTE", "UL", "DEO",
    "RELX", "SNY", "STLA", "SPOT", "ERIC", "ING", "BCS", "NOK", "PHG", "E", "BUD",
    # ETFs de paises/region de Europa
    "VGK", "FEZ", "EZU", "HEDJ", "EWG", "EWU", "EWQ", "EWP", "EWL", "EWN", "EWI", "EWD"]

# Umbrales de CALIDAD para comprar (confluencia, no "el minimo"): ver abajo.
EXT_MAX = 2.0          # no perseguir: como mucho 2 ATR por encima del VWAP. BAJADO de 4.0
                       # tras el forense del 2026-10-02: las ganadoras entraron a 1.15 ATR
                       # sobre VWAP de media, las perdedoras a 1.48; con EXT_MAX=2.0 el libro
                       # de ese día pasa de -$550 a -$143 (y a +$87 junto al RVOL de abajo),
                       # conservando WOLF (ext 1.58, +$275). Entrar cerca del VWAP, no estirado.
MARGEN_FR = 0.002      # fuerza relativa: batir al mercado por >=0.2% hoy
RET_MIN = 0.003        # sesion del valor >= +0.3% (movimiento con sustancia)
# GATE ANTI-PARABOLA (agotamiento): NO iniciar cuando el valor esta a la vez MUY
# estirado sobre el VWAP (>= EXT_PARAB ATR) Y ya MUY corrido en la sesion (>= RET_PARAB).
# Es la firma de la parabola tardia: compras el pico y el stop (0.8 ATR) queda DENTRO de
# la zona de reversion a VWAP -> el primer retroceso normal te barre. Caso real VSH
# (2026-10-02): ext 2.79 ATR + sesion +9.2% (+18% desde minimo) -> entrada 38.805, stop
# 38.48 tocado en 21 min -> -$105.79. Calibrado contra el historial de snapshots: este
# combo NO aparece en ningun ganador (JPM gano con ext 3.28 pero sesion solo +1.1%; los
# ganadores tienen sesion <3%), asi que el gate descarta VSH sin tocar ni una entrada buena.
EXT_PARAB = 2.5        # ATRs sobre VWAP a partir de los cuales vigilamos agotamiento
RET_PARAB = 0.06       # +6% de sesion: ya ha corrido su tramo; iniciar aqui es perseguir
# SUELO DE CONVICCION: tras pasar todos los gates, solo se OPERA el candidato cuya
# conviccion (z(rvol)+z(ret)+z(fr)+0.5·cat+mem) sea >= este umbral. Evita rellenar los
# huecos con setups flojos (convicción negativa = por debajo de la media del grupo de
# hoy): mejor quedarse en efectivo que operar mal. No limita re-comprar un líder que
# siga fuerte; solo exige que la decisión esté BIEN FUNDAMENTADA.
UMBRAL_CONVICCION = 0.0


# reservados al Libro C (TSMOM cross-asset): se excluyen del intradia para que los
# libros no se solapen en el mismo simbolo (una posicion por ticker en Alpaca).
RESERVADOS_TSMOM = {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "HYG"}
# reservados al Libro D (rotacion por paises europeos, candidato en observacion): sus
# 10 ETFs de pais los gestiona el libro mensual, no el intradia.
RESERVADOS_ROTACION = {"EWG", "EWQ", "EWP", "EWU", "EWI", "EWN", "EWL", "EWD", "EWO", "EWK"}
RESERVADOS = RESERVADOS_TSMOM | RESERVADOS_ROTACION

CACHE_UNIVERSO = "datos/universo_ampliado.json"


def _universo_tv(max_horas=20, n=300):
    """Las ~n acciones US MÁS LÍQUIDAS (TradingView: grandes + mid-caps líquidas),
    cacheadas ~a diario. AMPLÍA el universo operable sin depender de TV cada ciclo y sin
    escanear a ciegas. Fallback robusto: caché (aunque vieja) si TV falla; [] si nada."""
    import json
    import os
    cache = None
    try:
        cache = json.load(open(CACHE_UNIVERSO))
        edad_h = (pd.Timestamp.now() - pd.Timestamp(cache["ts"])).total_seconds() / 3600
        if edad_h < max_horas and cache.get("symbols"):
            return list(cache["symbols"])
    except Exception:
        cache = None
    try:
        from fuentes import tradingview as TV
        syms = TV.universo_liquido(n=n)
        if syms:
            os.makedirs("datos", exist_ok=True)
            json.dump({"ts": pd.Timestamp.now().isoformat(), "symbols": syms}, open(CACHE_UNIVERSO, "w"))
            return syms
    except Exception:
        pass
    try:
        return list(cache["symbols"]) if cache and cache.get("symbols") else []
    except Exception:
        return []


def universo_dinamico(con_europa=True):
    """Universo operable = base (EE.UU.) + EUROPA (ADRs/ETFs europeos cotizados en
    EE.UU.) + descubrimientos recientes de la watchlist, MENOS los reservados al
    Libro C (TSMOM). Mas donde ELEGIR; la confluencia de calidad decide. Los
    europeos compiten de igual a igual: se opera el mejor, sea de EE.UU. o de Europa."""
    try:
        from descubrir import watchlist_vigente
        extra = watchlist_vigente()
    except Exception:
        extra = []
    u = set(UNIVERSO) | set(extra) | set(_universo_tv())   # + mid-caps líquidas (TradingView)
    if con_europa:
        u |= set(EUROPA)
    return sorted(u - RESERVADOS)      # excluye reservados del Libro C (TSMOM) y D (rotacion)


def señal(ticker, intervalo="15Min"):
    """Re-analiza un ticker con las velas nuevas. Devuelve indicadores actuales Y
    señales de TENDENCIA (sostenidas), para distinguir una reversion real de una
    bajada organica puntual."""
    cfg = CONFIG[intervalo]
    desde = (pd.Timestamp.now() - pd.Timedelta(days=cfg["dias"])).strftime("%Y-%m-%d")
    df = alpaca.historico(ticker, intervalo, desde=desde, max_paginas=6, cliente=_CLI_INTRA)
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
    # MOMENTUM RECIENTE (ultima ~1h): retorno de las ultimas 4 velas de 15m, NO desde la
    # apertura. Clave para no comprar un valor que ABRIO ALTO por una noticia (volumen y
    # ret_sesion inflados) pero AHORA cae: su momentum reciente es negativo -> se rechaza.
    k = min(4, len(c) - 1)
    ret_reciente = float(c.iloc[-1] / c.iloc[-1 - k] - 1.0) if k >= 1 else 0.0
    return {"precio": precio, "dist_vwap": float(dv.iloc[-1]),
            "rvol": float(rvol(df).iloc[-1]), "ret_sesion": float(r["intra_ret_sesion"].iloc[-1]),
            "atr_pct": atr / precio if precio else 0.0, "pend": pend, "bajo_vwap": bajo_vwap,
            "ret_reciente": ret_reciente}


# TIME-STOP intradia: un setup intradia tiene VIDA UTIL CORTA. Mantener una
# posicion horas pierde el edge original (el pico de RVOL y el momentum de VWAP se
# disipan): ya no es la estrategia que la justifico, solo ocupa capital y riesgo.
# Se cierra por tiempo para rotar a una oportunidad fresca.
# Validado (figura 28): mantener mas tiempo NO mejora el EV/op, lo DEGRADA (el
# setup decae). La operacion media se resuelve en ~1h; mas alla pierde el edge.
MAX_VERDE_MIN = 90      # >=1.5h EN VERDE -> recoge el positivo y rota (no esperes a que se gire)
MAX_DURA_MIN = 150      # >=2.5h -> time-stop duro: el setup ha caducado, cierra pase lo que pase

# BREAK-EVEN STOP: en cuanto una operación alcanza +1R de beneficio (precio >= entrada +
# 1·riesgo, siendo riesgo = entrada - SL original), se SUBE el stop a la entrada. A partir
# de ahí la operación no puede perder: o toca el TP (+2R) o sale en ~0R. Ataca la asimetría
# invertida del forense 2026-10-02 (solo 1/19 tocó TP; ganadoras +0.4R, perdedoras -1.1R):
# una operación que ya probó +1R dejaba de poder convertirse en pérdida completa. NO
# contradice la fig. 28 (no amplía el SL ni mantiene más tiempo; es protección ortogonal).
BE_TRIGGER_R = 1.0


def _nuevo_sl_breakeven(entrada, sl_actual, precio, trig_R=BE_TRIGGER_R):
    """Decisión pura del break-even. Si la operación ya alcanzó +trig_R (precio >= entrada +
    trig_R·riesgo) y el stop SIGUE por debajo de la entrada, devuelve el nuevo SL = entrada
    (break-even). Si no, None. Idempotente: una vez movido a la entrada, riesgo<=0 -> None."""
    if entrada <= 0 or sl_actual <= 0 or precio <= 0:
        return None
    riesgo = entrada - sl_actual
    if riesgo <= 0:                                  # ya en break-even o por encima
        return None
    if precio >= entrada + trig_R * riesgo:
        return round(entrada, 2)
    return None


def revisar_intradia(mover_be=False):
    """Re-analiza cada posicion intradia con las velas nuevas y decide si cerrar.

    Cierra si (a) se CONFIRMA una reversion (sostenida bajo VWAP + pendiente baja +
    claramente bajo VWAP), o (b) se dispara el TIME-STOP: lleva mucho tiempo abierta
    (en verde >=2.5h -> recoge y rota; en cualquier caso >=3.5h -> caducada). Una
    sola barra roja NO cierra (para eso esta el SL del bracket); mantener una
    organica esta bien, pero mantenerla ETERNAMENTE pierde la estrategia intradia."""
    ordenes = AT.ordenes_abiertas()
    simbolos = sorted(set(o["symbol"] for o in ordenes))
    pos = AT.posiciones()
    # SL vigente por ticker (pata 'stop' del bracket) para el break-even
    sl_actual = {o["symbol"]: float(o["stop_price"]) for o in ordenes
                 if o.get("side") == "sell" and o.get("type") in ("stop", "stop_limit")
                 and o.get("stop_price")}
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
        # BREAK-EVEN: si NO se cierra y la operación ya probó +1R, sube el stop a la entrada.
        be = None
        if not cerrar and t in sl_actual:
            precio_t = pos[t].get("precio") or s.get("precio") or 0.0
            nuevo = _nuevo_sl_breakeven(pos[t].get("entrada", 0.0), sl_actual[t], precio_t)
            if nuevo is not None and nuevo > sl_actual[t]:
                if mover_be:
                    try:
                        r = AT.mover_stop(t, nuevo, confirmar=True)
                        be = f"break-even @{nuevo}" + ("" if r.get("movido") else f" (no: {r.get('motivo','?')})")
                    except Exception as e:                 # un fallo de PATCH no tumba el ciclo
                        be = f"break-even @{nuevo} FALLÓ: {str(e)[:50]}"
                else:
                    be = f"[dry] break-even @{nuevo}"
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
        out.append({"ticker": t, "pl": pl, "cerrar": cerrar, "motivo": motivo, "be": be,
                    "dist_vwap": s["dist_vwap"], "ret_sesion": s["ret_sesion"],
                    "rvol": s["rvol"], "pend": s["pend"], "bajo_vwap": s["bajo_vwap"],
                    "min_abierta": mins if mins is not None else float("nan")})
    return out


def _ret_sesion_mercado(intervalo="15Min", benchmark="SPY"):
    """Retorno de sesion del mercado (SPY) para medir la FUERZA RELATIVA de cada
    valor. Un verdadero 'stock in play' BATE al mercado hoy, no solo sube con la
    marea. Devuelve 0.0 si no se puede calcular."""
    try:
        cfg = CONFIG[intervalo]
        desde = (pd.Timestamp.now() - pd.Timedelta(days=cfg["dias"])).strftime("%Y-%m-%d")
        df = alpaca.historico(benchmark, intervalo, desde=desde, max_paginas=4)
        from patrones.intradia import rasgos_intradia
        return float(rasgos_intradia(df)["intra_ret_sesion"].iloc[-1])
    except Exception:
        return 0.0


def _candidatos_intradia(excluir=()):
    """Pool de candidatos intradía = acciones LÍQUIDAS EN JUEGO de TODO el mercado
    (TradingView, una llamada), en vez de iterar una lista fija de ~114. Si TradingView
    falla, cae al universo ampliado cacheado. Siempre menos reservados (Libros C/D) y los
    excluidos (ya en cartera). Acota el coste a los que de verdad se mueven hoy."""
    syms = []
    try:
        from fuentes import tradingview as TV
        syms = TV.en_juego()
    except Exception:
        syms = []
    if not syms:
        syms = universo_dinamico()                # fallback robusto (lista ampliada)
    ex = set(excluir) | RESERVADOS
    # dedup preservando orden (TV ya viene por RVOL desc)
    vistos, out = set(), []
    for t in syms:
        if t in ex or t in vistos:
            continue
        vistos.add(t); out.append(t)
    return out


def escanear(intervalo="15Min", excluir=(), n=12):
    """Nuevas oportunidades intradia en TODO el mercado líquido en juego (no en cartera).

    Candidatos: `_candidatos_intradia` (TradingView, mercado completo). A cada uno se le
    añade la FUERZA RELATIVA (vs el mercado) y la EXTENSION sobre el VWAP en ATRs, para que
    la seleccion use mas indicadores que el RVOL."""
    cfg = CONFIG[intervalo]
    mkt = _ret_sesion_mercado(intervalo)          # retorno del mercado hoy (SPY)
    filas = []
    for t in _candidatos_intradia(excluir):
        try:
            s = señal(t, intervalo)
        except Exception:
            continue
        # CONFLUENCIA de escaneo + MOMENTUM RECIENTE > 0 (no desde la apertura): descarta
        # el gap-and-fade (abrió alto por noticia y ahora cae), que es puro sesgo de apertura.
        cand = ((s["dist_vwap"] > 0) and (s["ret_sesion"] > 0) and (s["rvol"] >= cfg["rvol_min"])
                and (s.get("ret_reciente", 0.0) > 0))
        if cand:
            s["fuerza_rel"] = s["ret_sesion"] - mkt                        # ¿bate al mercado?
            s["ext"] = s["dist_vwap"] / s["atr_pct"] if s["atr_pct"] > 0 else 0.0  # ATRs sobre VWAP
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


def rankear_conviccion(opo, umbral_fuerte, excluir=(), gate_neg=-0.35,
                       ext_max=EXT_MAX, margen_fr=MARGEN_FR, ret_min=RET_MIN,
                       umbral_conv=UMBRAL_CONVICCION):
    """Elige SOLO las mejores por CONFLUENCIA: un candidato no entra por 'cumplir el
    minimo', sino por pasar VARIAS confirmaciones independientes a la vez. Luego se
    ordena por un score de conviccion y se opera el top (o efectivo si nada pasa).

    GATES de calidad (todos obligatorios):
      1. RVOL >= umbral fuerte        -> volumen anormal (stock in play)
      2. sobre VWAP (dist_vwap > 0)   -> compradores al mando
      3. NO sobre-extendido (ext<4)   -> no perseguir un valor ya estirado
      4. FUERZA RELATIVA > +0.2%      -> bate al mercado hoy (no solo sube con la marea)
      5. sesion del valor >= +0.3%    -> movimiento con sustancia
      6. catalizador no muy negativo  -> no comprar contra una mala noticia

    conviccion = z(rvol)+z(ret_sesion)+z(fuerza_rel)+0.5*catalizador+memoria.
    La evidencia (docs/aprendizaje_quant.md, figura 26) es que el unico caso con edge
    neto positivo es la CONFLUENCIA; un solo filtro no basta."""
    if opo is None or len(opo) == 0:
        return opo
    d = opo[(opo["rvol"] >= umbral_fuerte) & (~opo["ticker"].isin(set(excluir)))].copy()
    # gates de calidad (confluencia) con los indicadores nuevos, si estan disponibles
    if "dist_vwap" in d:
        d = d[d["dist_vwap"] > 0]
    if "ext" in d:
        d = d[(d["ext"] > 0) & (d["ext"] < ext_max)]
    # ANTI-PARABOLA: fuera los agotados (muy sobre VWAP Y ya muy corridos hoy). Ver VSH
    # (2026-10-02) en el bloque de constantes: comprar el pico con stop dentro de la zona
    # de reversion. El combo no toca ningun ganador del historial.
    if "ext" in d and "ret_sesion" in d:
        d = d[~((d["ext"] >= EXT_PARAB) & (d["ret_sesion"] >= RET_PARAB))]
    if "fuerza_rel" in d:
        d = d[d["fuerza_rel"] > margen_fr]
    if "ret_sesion" in d:
        d = d[d["ret_sesion"] >= ret_min]
    d = d.copy()
    if len(d) == 0:
        return d
    d["sent"] = [_sentimiento(t) for t in d["ticker"]]
    d = d[d["sent"] >= gate_neg].copy()        # gate: fuera catalizador muy negativo
    if len(d) == 0:
        return d
    cat = d["sent"].clip(-1, 1)
    from registro import ajuste_memoria
    aj = ajuste_memoria(list(d["ticker"]))
    d["mem"] = [aj.get(str(t), 0.0) for t in d["ticker"]]
    fr = _z(d["fuerza_rel"]).to_numpy() if "fuerza_rel" in d else 0.0
    d["conviccion"] = (_z(d["rvol"]).to_numpy() + _z(d["ret_sesion"]).to_numpy()
                       + fr + 0.5 * cat.to_numpy() + d["mem"].to_numpy())
    # SUELO DE CONVICCION: solo entran los BIEN FUNDAMENTADOS (>= umbral); el resto, fuera
    # (mejor efectivo que rellenar huecos con setups de convicción negativa).
    d = d[d["conviccion"] >= umbral_conv]
    return d.sort_values("conviccion", ascending=False).reset_index(drop=True)


def main(gestionar=False, rotar=False, reserva_por_pos=20000.0):
    pos = AT.posiciones()
    rev = revisar_intradia(mover_be=gestionar)
    print(f"Posiciones intradia (bracket): {len(rev)}")
    a_cerrar = []
    for d in rev:
        be = f"  [{d['be']}]" if d.get("be") else ""
        print(f"  {d['ticker']:<6} PL ${d['pl']:+.0f} | VWAP {d['dist_vwap']*100:+.2f}% "
              f"sesion {d['ret_sesion']*100:+.2f}% pend {d['pend']*100:+.3f}% "
              f"bajoVWAP {d['bajo_vwap']*100:.0f}% -> {d['motivo']}{be}")
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
    from registro import cargar_umbral, registrar_entrada, tickers_en_cooldown, reconciliar
    N_OBJETIVO = 3
    RVOL_FUERTE = cargar_umbral(2.0)      # umbral afinado por el aprendizaje (o 2.0 a priori).
    # SUBIDO de 1.5 tras el forense del 2026-10-02: las ganadoras tenían RVOL 3.1 de media,
    # las perdedoras 2.5; exigir >=2.0 (volumen anormal de verdad) filtra las flojas. Junto
    # con EXT_MAX=2.0, el libro de ese día pasa de -$550 (37% acierto, PF 0.47) a +$87
    # (44%, PF 1.23), eliminando 10 ops que netas -$637 y conservando a WOLF.
    if gestionar and rotar:
        reconciliar()      # marca YA los SL recientes en el log ANTES de mirar el cooldown
        # GATE DE RIESGO DE CARTERA (Tier 1): kill-switch antes de abrir nada nuevo.
        # FALLA CERRADO: gate_abrir devuelve abrir=False ante cualquier error (red,
        # estado, calculo), nunca sigue abriendo a ciegas. En HALT, ademas, aplana el
        # intradia (el libro de mas riesgo) -> preservacion de capital, no solo frenar.
        from patrones import riesgo as _Rk
        _g = _Rk.gate_abrir(posiciones=pos)   # gate_abrir pide cuenta() dentro de su try (fail-closed)
        riesgo_ok = _g["abrir"]
        if not riesgo_ok:
            print(f"\n[RIESGO {_g['decision']}]: {'; '.join(_g['razones'][:3])}")
            print("  -> No se abren posiciones nuevas este ciclo.")
            if _g["decision"] == "HALT":
                print("  -> HALT: aplanando intradia (preservacion de capital)...")
                try:
                    print("    ", AT.cerrar_intradia(confirmar=True))
                except Exception as e:
                    print(f"    error al aplanar: {str(e)[:70]}")
        held = set(pos) - set(a_cerrar)
        # cooldown = (log reconciliado + fills en vivo) U lo que acabamos de cerrar
        # este ciclo. Nunca re-comprar un nombre que acaba de saltar el SL ni el que
        # cerramos ahora mismo: asi NO se repite el churn de NKE (re-buy a los 3 min).
        cooldown = tickers_en_cooldown(minutos=60) | set(a_cerrar)
        if cooldown:
            print(f"\nEn cooldown (no re-entrar aun): {sorted(cooldown)}")
        intradia_vivas = len(rev) - len(a_cerrar)      # brackets que siguen abiertos
        libres = max(N_OBJETIVO - intradia_vivas, 0)
        # CONFLUENCIA: solo pasan los que cumplen VARIAS confirmaciones (no el minimo)
        fuertes = rankear_conviccion(opo, RVOL_FUERTE, excluir=held | cooldown)
        if libres and len(fuertes) and riesgo_ok:
            cfg = CONFIG["15Min"]
            cash = float(AT.cuenta()["cash"])
            por_pos = min(reserva_por_pos, cash / libres) if libres else 0.0
            # OVERLAY DE REGIMEN (gestion de vol + VIX, validado SPY 1993-26): recorta
            # el tamaño en estres de mercado. Conservador: SOLO reduce (clip a [0.3, 1.0]),
            # nunca apalanca. En calma/contango queda 1.0; en vol alta / backwardation baja.
            try:
                from patrones.regimen import multiplicador as _mult
                _rg = _mult()
                _m = max(0.3, min(1.0, _rg["mult"]))
                if _m < 0.99:
                    por_pos *= _m
                    print(f"[régimen] {_rg['nota']} -> tamaño ×{_m:.2f} (preservación de capital)")
            except Exception as e:
                print(f"[régimen] no evaluado ({str(e)[:40]}); tamaño sin recorte.")
            # mas candidatos de los huecos, por si alguno falla; TP/SL con el precio ACTUAL
            abrir = fuertes.head(libres + 3)
            px_now = AT.precio_actual(list(abrir["ticker"]))
            print(f"\nReinvirtiendo (hasta {libres}) por CONFLUENCIA de calidad "
                  f"(RVOL + VWAP + fuerza relativa + catalizador, no por el minimo):")
            colocadas = 0
            for _, s in abrir.iterrows():
                if colocadas >= libres:
                    break
                precio = float(px_now.get(s["ticker"], s["precio"]))   # precio de mercado ahora
                qty = int(por_pos / precio)
                if qty <= 0:
                    continue
                # TP/SL en ATR (R:R 2:1) pero ACOTADOS a banda intradía realista (ver
                # AT.niveles_bracket): nada de objetivos del 12-15% en nombres de ATR alto.
                tp, sl = AT.niveles_bracket(precio, cfg["k_sl"] * s["atr_pct"],
                                            r_mult=cfg["k_pt"] / cfg["k_sl"], lado="buy")
                try:
                    r = AT.enviar_bracket(s["ticker"], qty, tp, sl, confirmar=True, libro="intradia")
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
            print(f"\n{libres} hueco(s) libre(s), pero NINGUN candidato pasa la confluencia "
                  f"+ suelo de convicción (>= {UMBRAL_CONVICCION:.2f}): EFECTIVO (mejor no operar "
                  f"que operar mal).")

    print("\nTOP MOMENTUM DIARIO (oportunidades swing):")
    for t, m in escaneo_diario().items():
        print(f"  {t:<6} momento {m*100:+.1f}%" + ("  [ya en cartera]" if t in pos else ""))


if __name__ == "__main__":
    main(gestionar="--gestionar" in sys.argv, rotar="--rotar" in sys.argv)
