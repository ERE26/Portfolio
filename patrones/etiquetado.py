"""Triple-barrier labeling (Lopez de Prado) y utilidades de seleccion de trades.

La etiqueta clasica del proyecto era "¿el cierre sube o baja en H barras?". Tiene
dos problemas: ignora el CAMINO (una operacion que primero cae un 5% y luego
sube no es lo mismo que una que sube directa) y no incorpora riesgo/recompensa.

El metodo de la triple barrera pone, desde el precio de entrada:
  - una barrera de BENEFICIO  (arriba, a k_pt * ATR)
  - una barrera de PERDIDA    (abajo, a k_sl * ATR)
  - una barrera de TIEMPO     (a H barras)
y etiqueta segun cual se toca PRIMERO, mirando maximos y minimos intrabar. Eso
convierte cada caso en una operacion realista con stop y objetivo, que es justo
lo que hace falta para pasar de "acertar la direccion" a "elegir buenas
operaciones".

Todo se calcula en espacio logaritmico y en multiplos de ATR, para que sea
comparable entre activos de distinto precio y volatilidad. No mira mas alla de
las H barras: no hay fuga de futuro mas alla del horizonte.
"""
import numpy as np


def triple_barrera(log_high, log_low, log_close, pos, atr_pct, horizonte,
                   k_pt=1.5, k_sl=1.5):
    """Etiqueta triple-barrera vectorizada para un conjunto de posiciones.

    log_high/low/close: series completas en log. pos: indices de entrada.
    atr_pct: ATR/precio en cada pos (la unidad de las barreras). Devuelve dict con:
      label   +1 beneficio, -1 perdida, 0 vencimiento por tiempo
      ret     retorno logaritmico realizado al cerrar la operacion
      barras  cuantas barras hasta cerrar
      rr      ratio riesgo/beneficio realizado (ret / riesgo asumido)
    """
    from numpy.lib.stride_tricks import sliding_window_view as vw
    H = horizonte
    base = log_close[pos]
    up = base + k_pt * atr_pct
    dn = base - k_sl * atr_pct

    Wh = vw(log_high, H)[pos + 1]   # (n, H) high de las H barras siguientes
    Wl = vw(log_low, H)[pos + 1]
    Wc = vw(log_close, H)[pos + 1]

    toca_up = Wh >= up[:, None]
    toca_dn = Wl <= dn[:, None]
    n = len(pos)
    GRANDE = H + 10
    # primera barra (0..H-1) en que se toca cada barrera; GRANDE si no se toca
    t_up = np.where(toca_up.any(1), toca_up.argmax(1), GRANDE)
    t_dn = np.where(toca_dn.any(1), toca_dn.argmax(1), GRANDE)

    label = np.zeros(n, np.int8)
    barras = np.full(n, H, int)
    ret = Wc[:, -1] - base           # por defecto, vencimiento: retorno al cierre H

    gana = (t_up < t_dn) & (t_up < GRANDE)
    pierde = (t_dn < t_up) & (t_dn < GRANDE)
    # empate en la misma barra (toca ambas): conservador, se cuenta como perdida
    empate = (t_up == t_dn) & (t_up < GRANDE)

    label[gana] = 1
    barras[gana] = t_up[gana] + 1
    ret[gana] = (k_pt * atr_pct)[gana]

    perd = pierde | empate
    label[perd] = -1
    barras[perd] = t_dn[perd] + 1
    ret[perd] = (-k_sl * atr_pct)[perd]

    riesgo = k_sl * atr_pct
    rr = ret / np.maximum(riesgo, 1e-9)
    return {"label": label, "ret": ret.astype(np.float32),
            "barras": barras.astype(np.int16), "rr": rr.astype(np.float32),
            "t_up": t_up, "t_dn": t_dn}


def pesos_unicidad(fechas_idx, barras, horizonte):
    """Peso de unicidad de cada muestra (Lopez de Prado, cap. 4, simplificado).

    Operaciones cuyos horizontes se solapan comparten informacion; contarlas
    como independientes infla la muestra. Cada caso pesa 1 / (numero medio de
    operaciones concurrentes durante su vida). Se usa para ponderar el
    entrenamiento del meta-modelo y que no sobre-represente periodos densos.
    """
    n = len(fechas_idx)
    orden = np.argsort(fechas_idx, kind="stable")
    t = fechas_idx[orden].astype("datetime64[D]").astype(np.int64)
    dur = barras[orden].astype(np.int64)
    fin = t + dur
    # numero de operaciones vivas en cada punto de inicio (barrido con diferencias)
    puntos = np.concatenate([t, fin])
    signo = np.concatenate([np.ones(n, int), -np.ones(n, int)])
    ordp = np.argsort(puntos, kind="stable")
    conc = np.cumsum(signo[ordp])
    # concurrencia media durante la vida de cada operacion (aprox. por su inicio)
    mapa = {}
    for p, c in zip(puntos[ordp], conc):
        mapa[p] = max(c, 1)
    w = np.array([1.0 / max(mapa.get(ti, 1), 1) for ti in t], float)
    out = np.empty(n)
    out[orden] = w / w.mean()
    return out.astype(np.float32)


def recalcular_etiqueta(banco, k_pt, k_sl):
    """Recalcula la etiqueta triple-barrera para otro ratio, sin reconstruir.

    Usa los recorridos de maximos y minimos ya guardados en el banco (en ATR),
    asi que permite barrer rapido muchos ratios objetivo/stop. Devuelve un dict
    con label, ret (en ATR) y barras, con la misma convencion que triple_barrera.
    """
    if banco.caminos_high is None or banco.caminos_low is None:
        raise RuntimeError("El banco no guarda caminos_high/low; reconstruyelo.")
    H = banco.caminos.shape[1]
    hi = banco.caminos_high            # (n, H) en ATR desde la entrada
    lo = banco.caminos_low
    n = len(hi)
    toca_up = hi >= k_pt
    toca_dn = lo <= -k_sl
    GRANDE = H + 10
    t_up = np.where(toca_up.any(1), toca_up.argmax(1), GRANDE)
    t_dn = np.where(toca_dn.any(1), toca_dn.argmax(1), GRANDE)
    label = np.zeros(n, np.int8)
    ret = banco.caminos[:, -1].copy()          # vencimiento: retorno de cierre (ATR)
    barras = np.full(n, H, int)
    gana = (t_up < t_dn) & (t_up < GRANDE)
    perd = ((t_dn <= t_up) & (t_dn < GRANDE))
    label[gana] = 1; ret[gana] = k_pt; barras[gana] = t_up[gana] + 1
    label[perd] = -1; ret[perd] = -k_sl; barras[perd] = t_dn[perd] + 1
    return {"label": label, "ret": ret.astype(np.float32), "barras": barras.astype(np.int16)}
