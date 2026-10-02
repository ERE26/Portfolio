"""Capa de riesgo de CARTERA (lo que convierte 'varias estrategias' en 'un fondo').

Tres cosas, en el espiritu de los multi-manager (Millennium/Citadel):
  1. Medir el riesgo del CONJUNTO: exposicion, concentracion, VaR de cartera y
     correlacion de lo que tenemos (si todo se mueve junto, no hay diversificacion).
  2. KILL-SWITCHES: limites duros (perdida diaria, drawdown, nº de posiciones,
     exposicion) que devuelven GO / REDUCE / HALT. Preservacion de capital por diseno.
  3. PRESUPUESTO DE RIESGO entre libros: repartir RIESGO (no capital) entre los
     libros por su vol/correlacion (HRP/inv-vol de cartera.py) y senalar excesos.

Solo lee la cuenta paper y precios (Yahoo). No envia ordenes: devuelve decisiones y
metricas; el que opera (gestionar.py) consulta y actua. El estado (pico de equity,
equity de inicio de dia) persiste en datos/riesgo_estado.json.

Limitaciones conocidas (v1), honestas, pendientes de refinar:
  - El presupuesto de riesgo por libro usa la vol de la sub-cartera con el equity
    TOTAL (proxy razonable, algo auto-referencial); migrara a HRP sobre el historico
    REAL de retornos por libro cuando se acumule (hoy el snapshot guarda bruta/pl).
  - corr_media (diversificacion) es una media simple, ciega al signo largo/corto y a
    los pesos; es un indicador, no una medida exacta de riesgo conjunto.
  - var_cartera excluye posiciones sin historico suficiente en Yahoo; el informe lo
    avisa y trata el VaR como cota inferior cuando la cobertura es parcial.
"""
import json
import os

import numpy as np
import pandas as pd

ESTADO = "datos/riesgo_estado.json"

# Nombres de libro validos: SOLO estos prefijos de client_order_id cuentan como
# etiqueta de libro (las ordenes antiguas llevan un client_order_id autogenerado por
# Alpaca —hashes con guiones— que NO debe confundirse con un libro).
LIBROS_VALIDOS = {"intradia", "swing", "tsmom", "europa", "rotacion"}

# Universos reservados por libro, para ATRIBUIR cada posicion a su libro cuando la
# orden no lleva etiqueta (client_order_id). TSMOM esta limpio; el resto se afina
# con las ordenes etiquetadas que se vayan acumulando.
RESERVADOS = {
    "tsmom": {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "HYG"},
}

# Limites por defecto de los kill-switches (fracciones del equity salvo donde se diga)
LIMITES = {
    "perdida_diaria_max": 0.03,     # -3% en el dia -> HALT de nuevas entradas
    "perdida_diaria_reduce": 0.015,  # -1.5% -> REDUCE (no abrir mas, gestionar las abiertas)
    "drawdown_max": 0.10,           # -10% desde el pico -> HALT
    "drawdown_reduce": 0.06,        # -6% -> REDUCE
    "max_posiciones": 45,           # nº de posiciones concurrentes (proxy debil anti-
                                    # desbocamiento). Subido de 30 a 45: con 4 libros
                                    # (swing ~20 + TSMOM + rotacion + intradia) el tope de
                                    # 30 asfixiaba al intradia pese a riesgo REAL bajisimo
                                    # (bruto 0.65x/1.5x, VaR 0.76%/3%). El riesgo de verdad
                                    # lo miden exposicion/concentracion/VaR, no el conteo.
    "exposicion_bruta_max": 1.5,    # suma |valor|/equity (apalancamiento efectivo)
    "concentracion_max": 0.25,      # peso maximo de UNA posicion
    "var_reduce": 0.03,             # VaR 1d > 3% del equity -> REDUCE (riesgo de cartera)
    "var_max": 0.05,                # VaR 1d > 5% -> HALT
}


# --------------------------------------------------------------- exposicion
def exposiciones(posiciones, equity):
    """Exposicion y concentracion de la cartera actual. posiciones: dict
    ticker -> {valor, ...} (valor = market value con signo). equity > 0."""
    if equity <= 0 or not posiciones:
        return {"bruta": 0.0, "neta": 0.0, "larga": 0.0, "corta": 0.0,
                "n": 0, "peso_max": 0.0, "top5": 0.0, "hhi": 0.0, "pesos": {}}
    val = {t: float(p.get("valor", 0.0)) for t, p in posiciones.items()}
    pesos = {t: v / equity for t, v in val.items()}
    largos = sum(v for v in val.values() if v > 0)
    cortos = sum(-v for v in val.values() if v < 0)
    aps = sorted((abs(w) for w in pesos.values()), reverse=True)
    return {
        "bruta": (largos + cortos) / equity,
        "neta": (largos - cortos) / equity,
        "larga": largos / equity, "corta": cortos / equity,
        "n": len(posiciones),
        "peso_max": aps[0] if aps else 0.0,
        "top5": sum(aps[:5]),
        "hhi": float(sum(w * w for w in aps)),     # indice Herfindahl de concentracion
        "pesos": pesos,
    }


# --------------------------------------------------------------- covarianza / VaR
def _retornos_diarios(tickers, lookback_dias=252):
    """Matriz de retornos diarios de los tickers (Yahoo). Alineada y sin NaN."""
    from fuentes import yahoo
    from fuentes.http import Cliente
    cli = Cliente()
    desde = (pd.Timestamp.now() - pd.Timedelta(days=int(lookback_dias * 1.6))).strftime("%Y-%m-%d")
    cols = {}
    for t in tickers:
        try:
            df, _ = yahoo.historico(t, intervalo="1d", desde=desde, cliente=cli)
            cols[t] = df["close"].astype(float).pct_change()
        except Exception:
            continue
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna(how="all").iloc[-lookback_dias:]


def covarianza_cartera(tickers, lookback_dias=252, condicional=True):
    """Covarianza DIARIA de los holdings. Si condicional=True usa vol EWMA
    (RiskMetrics/GARCH, mas reactiva al regimen) en la diagonal y la correlacion
    muestral fuera; si no, covarianza muestral con shrinkage Ledoit-Wolf."""
    R = _retornos_diarios(tickers, lookback_dias).dropna()
    if R.shape[0] < 30 or R.shape[1] == 0:
        return pd.DataFrame(), R
    from .cartera import covarianza_shrunk
    cov = covarianza_shrunk(R)                       # ndarray (shrinkage), diaria
    cov = pd.DataFrame(cov, index=R.columns, columns=R.columns)
    if condicional:
        from .volatilidad import vol_ewma
        # Reescala la covarianza para que la DIAGONAL quede en la vol CONDICIONAL
        # (EWMA) manteniendo la correlacion del estimador shrunk. Se divide por la vol
        # IMPLICITA de la matriz shrunk (sqrt de su diagonal), NO por R.std(): tras el
        # shrinkage la diagonal ya no es la varianza muestral, asi que usar R.std()
        # dejaba la diagonal descuadrada (hallazgo de revision).
        sd_shrunk = pd.Series(np.sqrt(np.clip(np.diag(cov.to_numpy()), 1e-18, None)), index=cov.index)
        sd_c = pd.Series({t: float(vol_ewma(R[t], anualizar=False).iloc[-1]) for t in R.columns})
        esc = (sd_c.reindex(cov.index) / sd_shrunk).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        cov = cov.mul(esc, axis=0).mul(esc, axis=1)  # diag -> sd_c^2, correlacion intacta
    return cov, R


def var_cartera(posiciones, equity, nivel=0.95, lookback_dias=252, condicional=True):
    """VaR parametrico de cartera a 1 dia, basado en las POSICIONES (no en el equity
    historico): w'Σw con los pesos actuales. Devuelve el VaR en $ y en % del equity,
    la vol diaria de la cartera, y la correlacion media (indicador de diversificacion).

    VaR 95% = 1.645 · sigma_cartera · equity  (perdida que no se supera el 95% de los
    dias, bajo normalidad). Es la medida de riesgo estandar de una mesa."""
    from scipy import stats
    out = {"var_pct": float("nan"), "var_usd": float("nan"), "vol_diaria": float("nan"),
           "corr_media": float("nan"), "n_cubiertos": 0, "nivel": nivel}
    if equity <= 0 or not posiciones:
        return out
    tickers = list(posiciones)
    cov, R = covarianza_cartera(tickers, lookback_dias, condicional)
    if cov.empty:
        return out
    w = pd.Series({t: float(posiciones[t].get("valor", 0.0)) / equity for t in cov.index})
    var_d = float(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())
    sigma = float(np.sqrt(max(var_d, 0.0)))
    z = float(stats.norm.ppf(nivel))
    corr = R.corr()
    iu = np.triu_indices(len(corr), 1)
    out.update({"var_pct": z * sigma, "var_usd": z * sigma * equity, "vol_diaria": sigma,
                "corr_media": float(np.nanmean(corr.to_numpy()[iu])) if len(corr) > 1 else float("nan"),
                "n_cubiertos": len(cov)})
    return out


# --------------------------------------------------------------- atribucion por libro
def atribuir_libros(posiciones, ordenes=None):
    """Asigna cada posicion a su libro. Prioridad: (1) etiqueta de la orden
    (client_order_id '<libro>-...'), (2) universo reservado (TSMOM), (3) 'sin_clasificar'.
    ordenes: lista de dicts de Alpaca (de ordenes recientes) para leer client_order_id."""
    etiqueta = {}
    for o in (ordenes or []):
        coid = str(o.get("client_order_id", ""))
        sym = o.get("symbol")
        prefijo = coid.split("-", 1)[0] if "-" in coid else ""
        if sym and prefijo in LIBROS_VALIDOS:     # solo prefijos de libro conocidos
            etiqueta[sym] = prefijo
    libros = {}
    for t in posiciones:
        if t in etiqueta:
            libros[t] = etiqueta[t]
        elif t in RESERVADOS["tsmom"]:
            libros[t] = "tsmom"
        else:
            libros[t] = "sin_clasificar"
    return libros


def exposicion_por_libro(posiciones, equity, ordenes=None):
    """Exposicion bruta por libro (suma |valor|/equity), para ver el reparto real."""
    libros = atribuir_libros(posiciones, ordenes)
    agg = {}
    for t, p in posiciones.items():
        lb = libros.get(t, "sin_clasificar")
        agg.setdefault(lb, {"bruta": 0.0, "pl": 0.0, "n": 0})
        agg[lb]["bruta"] += abs(float(p.get("valor", 0.0))) / equity if equity > 0 else 0.0
        agg[lb]["pl"] += float(p.get("pl", 0.0))
        agg[lb]["n"] += 1
    return agg


# --------------------------------------------------------------- presupuesto de riesgo
def presupuesto_riesgo(retornos_libros, metodo="inv_vol", min_obs=40):
    """Pesos de RIESGO objetivo entre libros a partir de su historico de retornos
    (dict libro -> serie). Con poca historia usa inv-vol (o equiponderado); con
    suficiente, HRP. Devuelve (pesos, nota)."""
    series = {k: pd.Series(v).dropna() for k, v in (retornos_libros or {}).items()}
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if not series:
        return {}, "sin datos de retornos por libro todavia"
    if len(series) == 1:
        return {list(series)[0]: 1.0}, "un solo libro con datos -> 100%"
    df = pd.DataFrame(series).dropna()
    n = len(df)
    from .cartera import covarianza_shrunk, pesos_riesgo, pesos_inv_vol
    if n < min_obs:
        vol = df.std().replace(0, np.nan)
        w = (1.0 / vol).fillna(0.0)
        w = (w / w.sum()).to_dict() if w.sum() > 0 else {}
        return w, f"inv-vol (solo {n} obs; HRP necesita >= {min_obs})"
    cov = covarianza_shrunk(df)
    try:
        w = pesos_riesgo(cov, "hrp")
    except Exception:
        w = pesos_inv_vol(cov)
    w = pd.Series(w, index=df.columns)
    return (w / w.sum()).to_dict(), f"HRP sobre {n} obs"


# --------------------------------------------------------------- kill-switches / estado
def _hoy():
    """Dia de SESION de mercado (America/New_York), no la fecha UTC del servidor:
    el ancla de P&L diario debe reiniciarse con la sesion US, no a medianoche UTC
    (que cae en mitad de la tarde de Madrid, con el mercado abierto)."""
    try:
        return pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    except Exception:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def cargar_estado():
    try:
        est = json.load(open(ESTADO))
        return est if isinstance(est, dict) else {}   # un JSON valido no-dict ('[]') -> {}
    except Exception:
        return {}


def guardar_estado(est):
    os.makedirs("datos", exist_ok=True)
    json.dump(est, open(ESTADO, "w"), indent=2)


def evaluar_riesgo(cuenta, posiciones, ordenes=None, limites=None, persistir=True, var_pct=None):
    """Evalua los kill-switches y devuelve la DECISION de riesgo.

    Devuelve {decision: 'GO'|'REDUCE'|'HALT', razones: [...], metricas: {...}}.
      - GO: operar normal.
      - REDUCE: no abrir nuevas posiciones; gestionar/cerrar las abiertas.
      - HALT: parar el libro; preservar capital (el que opera decide si aplanar).
    var_pct: VaR 1 dia de la cartera (fraccion del equity) ya calculado; si se pasa,
    se aplica un limite de VaR (gobierna el riesgo del CONJUNTO, no solo limites por
    nombre). El informe lo calcula y lo pasa; el bucle rapido lo omite.
    Mantiene el pico de equity y el equity de inicio de dia en disco (drawdown y P&L
    diario). 'cuenta' es el dict de alpaca_trade.cuenta()."""
    lim = {**LIMITES, **(limites or {})}
    equity = float(cuenta.get("equity", 0.0))
    # guarda dura: cuenta aniquilada o sin equity valido -> HALT (nunca abrir a ciegas)
    if not np.isfinite(equity) or equity <= 0:
        return {"decision": "HALT", "razones": [f"equity no valido o <=0 ({equity})"],
                "metricas": {"equity": equity, "pl_dia": 0.0, "drawdown": 0.0,
                             "pico_equity": 0.0, "exposicion_bruta": 0.0,
                             "n_posiciones": len(posiciones or {}), "concentracion_max": 0.0}}
    est = cargar_estado()
    hoy = _hoy()
    # reinicio diario del ancla de P&L del dia
    if est.get("fecha_dia") != hoy:
        est["fecha_dia"] = hoy
        est["equity_inicio_dia"] = equity
    est["equity_inicio_dia"] = est.get("equity_inicio_dia", equity) or equity
    est["pico_equity"] = max(float(est.get("pico_equity", 0.0) or 0.0), equity)

    ini = est["equity_inicio_dia"] or equity
    pico = est["pico_equity"] or equity
    # P&L del dia contra el CIERRE OFICIAL de ayer (last_equity de Alpaca), que es el
    # denominador estandar de 'cambio del dia'; si no esta, se usa el ancla propia. Con
    # el mercado cerrado y sin operar, un valor != 0 es solo mark-to-market de las
    # posiciones overnight (revalorizacion), no una operacion.
    last_eq = float(cuenta.get("last_equity", 0) or 0)
    ref_dia = last_eq if last_eq > 0 else (ini if ini > 0 else equity)
    pl_dia = (equity / ref_dia - 1.0) if ref_dia > 0 else 0.0
    dd = (equity / pico - 1.0) if pico > 0 else 0.0
    exp = exposiciones(posiciones, equity)

    razones, decision = [], "GO"
    def peor(d):
        return {"GO": 0, "REDUCE": 1, "HALT": 2}[d]
    def sube(d, motivo):
        nonlocal decision
        razones.append(motivo)
        if peor(d) > peor(decision):
            decision = d

    if pl_dia <= -lim["perdida_diaria_max"]:
        sube("HALT", f"perdida diaria {pl_dia:+.2%} <= -{lim['perdida_diaria_max']:.0%}")
    elif pl_dia <= -lim["perdida_diaria_reduce"]:
        sube("REDUCE", f"perdida diaria {pl_dia:+.2%} (umbral REDUCE)")
    if dd <= -lim["drawdown_max"]:
        sube("HALT", f"drawdown {dd:+.2%} <= -{lim['drawdown_max']:.0%} desde el pico")
    elif dd <= -lim["drawdown_reduce"]:
        sube("REDUCE", f"drawdown {dd:+.2%} (umbral REDUCE)")
    if exp["n"] >= lim["max_posiciones"]:
        sube("REDUCE", f"{exp['n']} posiciones >= limite {lim['max_posiciones']}")
    if exp["bruta"] > lim["exposicion_bruta_max"]:
        sube("REDUCE", f"exposicion bruta {exp['bruta']:.2f}x > {lim['exposicion_bruta_max']}x")
    if exp["peso_max"] > lim["concentracion_max"]:
        sube("REDUCE", f"concentracion {exp['peso_max']:.1%} > {lim['concentracion_max']:.0%} en una posicion")
    if var_pct is not None and np.isfinite(var_pct):
        if var_pct > lim.get("var_max", 0.05):
            sube("HALT", f"VaR 1d {var_pct:.2%} > {lim.get('var_max',0.05):.0%} (riesgo de cartera)")
        elif var_pct > lim.get("var_reduce", 0.03):
            sube("REDUCE", f"VaR 1d {var_pct:.2%} (umbral REDUCE)")

    est["ultima_decision"] = decision
    if persistir:
        guardar_estado(est)
    return {"decision": decision, "razones": razones or ["todo dentro de limites"],
            "metricas": {"equity": equity, "pl_dia": pl_dia, "drawdown": dd,
                         "pico_equity": pico, "exposicion_bruta": exp["bruta"],
                         "n_posiciones": exp["n"], "concentracion_max": exp["peso_max"]}}


def gate_abrir(cuenta=None, posiciones=None, limites=None):
    """Puerta de riesgo UNICA para todos los libros: ¿se permite ABRIR posiciones
    nuevas ahora? FALLA CERRADO: ante CUALQUIER error (red, estado, calculo) devuelve
    abrir=False. Un kill-switch de preservacion de capital nunca debe fallar abierto.

    Devuelve {abrir: bool, decision, razones}. Cada libro lo llama antes de comprar;
    las SALIDAS/ventas no se bloquean (se gestiona lo abierto aunque el gate cierre)."""
    try:
        from fuentes import alpaca_trade as AT
        cta = cuenta if cuenta is not None else AT.cuenta()
        pos = posiciones if posiciones is not None else AT.posiciones()
        ev = evaluar_riesgo(cta, pos, limites=limites)
        return {"abrir": ev["decision"] == "GO", "decision": ev["decision"],
                "razones": ev["razones"]}
    except Exception as e:
        return {"abrir": False, "decision": "HALT",
                "razones": [f"fail-closed: error evaluando riesgo ({str(e)[:70]})"]}
