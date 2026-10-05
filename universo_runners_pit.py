"""Universo POINT-IN-TIME de small-cap RUNNERS (low-float), SIN supervivencia.

OJO - no confundir con universo_pit.py: aquel construye el universo LIQUIDO del
FONDO (mid/large-cap, stat-arb) as-of. ESTE construye el rincon OPUESTO: los
small-cap / low-float RUNNERS ($1-20, poca flotacion, catalizador) que se mueven
30-200% intradia -- el territorio del retail con bots, que el fondo no puede tocar
(capacidad minuscula). Ambos salen del grouped de Massive, pero el filtro es otro.

Por que importa: toda nuestra busqueda intradia uso el universo liquido actual
(el mas arbitrado) y no aparecio edge. Aqui probamos el rincon ineficiente, y con
HONESTIDAD: sin supervivencia (la mayoria de estos nombres quiebra) y sin look-ahead.

  - Fuente: GROUPED diario (TODO el mercado US en una llamada por sesion). Incluye
    los que COTIZARON ese dia aunque hoy esten deslistados -> SIN supervivencia.
  - Seleccion AS-OF, sin look-ahead: un ticker es "runner candidato" el dia t por el
    GAP de apertura (open_t vs close_{t-1}) y el precio de apertura, AMBOS conocidos
    a las 09:30. NO se usa el resultado del dia para seleccionar.
  - Artefactos de split: un reverse-split nocturno da un "gap" falso enorme; se
    excluyen los ticker-dias con split y se capa el gap a gap_max.
  - Top gappers: cada dia se ordenan por gap y se toman los top_k (acota datos y es
    lo que mira el day-trader).

Determinista (solo depende del grouped). Las semillas del backtest muestrean DIAS.
Solo lectura. Cachea en datos/cache (gitignored).
"""
import pandas as pd

from fuentes import massive as mv

PRECIO_LO, PRECIO_HI = 1.0, 20.0     # small-caps baratas (rango del runner de retail)
GAP_MIN = 0.10                        # gap de apertura >= 10% (en juego)
GAP_MAX = 3.0                         # > 300% es casi siempre reverse-split/dato sucio
TOP_K = 15                            # top gappers del dia (acota datos; realista)
BARRAS_SESION = 78                    # velas de 5m en una sesion regular (para RVOL proxy)


def _grouped_idx(fecha, cm):
    """grouped_diario de Massive (formato ticker-columna) -> DataFrame indexado por
    ticker, con precios RAW (ajustado=False: el precio REAL que vio el mercado ese
    dia, correcto para medir el gap). None si no hubo sesion."""
    try:
        g = mv.grouped_diario(fecha, ajustado=False, cm=cm)
    except Exception:
        return None
    if not len(g) or "ticker" not in g:
        return None
    return g.set_index("ticker")


def panel_grouped(desde, hasta, cm=None):
    """dict fecha(str)->DataFrame(grouped indexado por ticker) para cada sesion con
    datos en [desde, hasta]. Una llamada por dia natural; los dias sin sesion se
    descartan solos."""
    cm = cm or mv.ClienteMassive(pausa_min=mv.PAUSA_STARTER)
    panel = {}
    for d in pd.bdate_range(desde, hasta).strftime("%Y-%m-%d"):
        g = _grouped_idx(d, cm)
        if g is not None:
            panel[d] = g
    return panel


def splits_set(desde, hasta, cm=None):
    """Conjunto de (ticker, execution_date) con split en la ventana, para excluir los
    gaps falsos por reverse-split."""
    cm = cm or mv.ClienteMassive(pausa_min=mv.PAUSA_STARTER)
    d0 = (pd.Timestamp(desde) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    d1 = (pd.Timestamp(hasta) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    s = mv.splits(d0, d1, limite=1000, cm=cm)
    if not len(s):
        return set()
    return {(r["ticker"], str(r["execution_date"])) for _, r in s.iterrows()}


def set_activos(cm=None, paginas=12, limite=1000):
    """Conjunto de tickers de ACCION COMUN (CS) ACTIVOS hoy. Para medir el HAIRCUT de
    supervivencia: 'supervivientes' = candidatos cuyo ticker sigue activo."""
    cm = cm or mv.ClienteMassive(pausa_min=mv.PAUSA_STARTER)
    df = mv.tickers_referencia(activos=True, paginas=paginas, limite=limite, cm=cm, type="CS")
    if not len(df) or "ticker" not in df:
        return set()
    return set(df["ticker"].astype(str))


def candidatos_runner(panel, gap_min=GAP_MIN, gap_max=GAP_MAX, precio_lo=PRECIO_LO,
                      precio_hi=PRECIO_HI, top_k=TOP_K, splits=None):
    """Lista de candidatos runner POINT-IN-TIME, uno por (fecha, ticker).

    Cada candidato trae SOLO informacion conocida a la apertura (gap, precio de
    apertura, cierre y volumen del dia PREVIO) mas, aparte y marcado como POST, el
    dolar-volumen de la sesion completa (solo para describir liquidez al reportar;
    NUNCA se usa como gate de entrada)."""
    splits = splits or set()
    dias = sorted(panel)
    out = []
    for i in range(1, len(dias)):
        d, dprev = dias[i], dias[i - 1]
        cur, prev = panel[d], panel[dprev]
        common = cur.index.intersection(prev.index)
        if not len(common):
            continue
        o = cur.loc[common, "open"].astype(float)
        pc = prev.loc[common, "close"].astype(float)
        pv = prev.loc[common, "volume"].astype(float)
        gap = o / pc - 1.0
        mask = ((o >= precio_lo) & (o <= precio_hi) & (gap >= gap_min) & (gap <= gap_max)
                & (pc > 0))
        sel = common[mask]
        if not len(sel):
            continue
        g = gap.loc[sel].sort_values(ascending=False)
        tomados = 0
        for t in g.index:
            if (t, d) in splits:            # artefacto de split: fuera
                continue
            dv_dia = float(cur.loc[t, "close"]) * float(cur.loc[t, "volume"])  # POST (solo liquidez)
            out.append({
                "fecha": d, "ticker": t,
                "open": float(o[t]), "prev_close": float(pc[t]),
                "gap": float(gap[t]),
                "prev_vol": float(pv[t]),
                "prev_dollar_vol": float(pc[t]) * float(pv[t]),
                "dollar_vol_dia_POST": dv_dia,
            })
            tomados += 1
            if tomados >= top_k:
                break
    return out


def resumen_universo(cands, activos=None):
    """Estadisticos descriptivos del universo (y haircut de supervivencia si se pasa
    el set de activos)."""
    import numpy as np
    n = len(cands)
    gaps = np.array([c["gap"] for c in cands]) if n else np.array([])
    precios = np.array([c["open"] for c in cands]) if n else np.array([])
    dias = sorted({c["fecha"] for c in cands})
    r = {
        "candidatos": n,
        "dias": len(dias),
        "rango_fechas": (dias[0], dias[-1]) if dias else (None, None),
        "cand_por_dia": round(n / max(len(dias), 1), 1),
        "gap_mediana_%": round(float(np.median(gaps)) * 100, 1) if n else None,
        "gap_p90_%": round(float(np.percentile(gaps, 90)) * 100, 1) if n else None,
        "precio_mediano": round(float(np.median(precios)), 2) if n else None,
    }
    if activos is not None and n:
        vivos = sum(1 for c in cands if c["ticker"] in activos)
        r["supervivientes_hoy"] = vivos
        r["deslistados_o_inactivos"] = n - vivos
        r["haircut_supervivencia_%"] = round((n - vivos) / n * 100, 1)
    return r


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Inspecciona el universo PIT de runners")
    ap.add_argument("--desde", default="2024-03-01")
    ap.add_argument("--hasta", default="2024-05-31")
    ap.add_argument("--topk", type=int, default=TOP_K)
    a = ap.parse_args()

    cm = mv.ClienteMassive(pausa_min=mv.PAUSA_STARTER)
    print(f"# Universo PIT de runners  {a.desde} -> {a.hasta}  (top {a.topk}/dia)")
    panel = panel_grouped(a.desde, a.hasta, cm=cm)
    print(f"sesiones con datos: {len(panel)}")
    sp = splits_set(a.desde, a.hasta, cm=cm)
    print(f"splits en ventana (excluidos): {len(sp)}")
    cands = candidatos_runner(panel, top_k=a.topk, splits=sp)
    print(f"candidatos runner: {len(cands)}")
    print("midiendo haircut de supervivencia (set de activos de hoy)...")
    act = set_activos(cm=cm)
    print(f"tickers CS activos hoy: {len(act)}")
    print(json.dumps(resumen_universo(cands, activos=act), indent=1, default=str))
    for c in cands[:12]:
        vivo = "VIVO" if c["ticker"] in act else "MUERTO"
        print(f"  {c['fecha']} {c['ticker']:6s} gap {c['gap']*100:5.0f}%  open {c['open']:6.2f}  "
              f"dv_dia ${c['dollar_vol_dia_POST']/1e6:6.1f}M  {vivo}")
