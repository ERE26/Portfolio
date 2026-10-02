"""Libro D: ROTACION POR PAISES EUROPEOS (momentum transversal entre ETFs de pais que
Alpaca SI puede operar). Flujo poco correlacionado con los libros US (corr 0.56 con SPY).

ESTADO: CANDIDATO EN OBSERVACION (no 'validado'). Validacion reproducible en
`validar_libroD.py` (2010-2026, 36 configs, 188 meses): PBO 0.24 (NO sobreajustado) y
DSR vs azar 0.94 -> el metodo es limpio. PERO el DSR contra comprar-y-mantener Europa
(VGK) es 0.89, por debajo del 0.95 estandar: el EDGE sobre la beta europea es MARGINAL.
Es un diversificador modesto, no alfa heroica. NUNCA dinero real hasta: DSR-vs-VGK>=0.95,
slippage real medido (ETFs finos EWO/EWK/EWD), y track OOS en paper. Aqui solo PAPER,
para GENERAR esa evidencia OOS a riesgo cero.

Cada mes: largo el TOP-K por momentum a 6 meses (saltando el ultimo mes), SOLO si el
ETF esta sobre su SMA200; a efectivo lo que no tenga tendencia. Rebalanceo MENSUAL,
auto-limitado. Gate de riesgo Tier 1 (no compra en estres). Siempre PAPER.

  python operativa_rotacion.py            # dry-run
  python operativa_rotacion.py --enviar   # coloca el rebalanceo (paper)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca_trade as AT

# cesta de 10 ETFs de pais europeos (iShares, todos cotizan desde antes de 2010).
# Reservada a este libro (se excluye del intradia para no solaparse en el mismo simbolo).
CESTA = ["EWG", "EWQ", "EWP", "EWU", "EWI", "EWN", "EWL", "EWD", "EWO", "EWK"]
K, LOOKBACK, SKIP = 3, 126, 21          # config validada (pre-especificada)
ADV_FRAC = 0.01                         # cap de capacidad: max 1% del volumen $ diario (ADV)
ESTADO = "datos/rotacion_rebal.json"


def _panel_diario(cli=None, con_adv=False):
    """Cierres diarios de la cesta (Yahoo; estas ETFs no estan en el almacen local).
    Con con_adv=True devuelve tambien el volumen $ diario mediano 60d por ETF (ADV),
    para el guard de capacidad (los ETFs finos EWO/EWK no admiten posiciones grandes)."""
    from fuentes import yahoo
    from fuentes.http import Cliente
    c = cli or Cliente()
    desde = (pd.Timestamp.now() - pd.Timedelta(days=430)).strftime("%Y-%m-%d")
    cols, adv = {}, {}
    for t in CESTA:
        try:
            df = yahoo.historico(t, intervalo="1d", desde=desde, cliente=c)[0]
            cols[t] = df["close"].astype(float)
            adv[t] = float((df["close"].astype(float) * df["volume"].astype(float)).tail(60).median())
        except Exception:
            continue
    panel = pd.DataFrame(cols).sort_index().ffill(limit=5).dropna()
    return (panel, adv) if con_adv else panel


def objetivo_rotacion(panel, capital, k=K, lb=LOOKBACK, skip=SKIP):
    """Objetivo en $ por pais (equiponderado entre el top-k por momentum 6m con
    tendencia). Sin look-ahead: momentum y SMA200 usan solo el pasado."""
    if len(panel) < lb + skip + 1 or len(panel) < 205:
        return {}, pd.Series(dtype=float)
    mom = panel.iloc[-1 - skip] / panel.iloc[-1 - skip - lb] - 1
    sma200 = panel.rolling(200).mean().iloc[-1]
    elig = mom[panel.iloc[-1] > sma200]
    top = elig.sort_values(ascending=False).head(k)
    if len(top) == 0:
        return {}, mom
    return {t: capital / len(top) for t in top.index}, mom


def _ultima_rebal():
    try:
        return pd.Timestamp(json.load(open(ESTADO))["fecha"])
    except Exception:
        return None


def _marcar_rebal():
    os.makedirs("datos", exist_ok=True)
    json.dump({"fecha": pd.Timestamp.now().isoformat()}, open(ESTADO, "w"))


def ejecutar(enviar=False, capital_frac=None, dias_min=25):
    """Rebalanceo MENSUAL del Libro D (auto-limitado: si el ultimo fue hace <dias_min,
    no toca). capital_frac = fraccion del equity dedicada a este libro; si es None se
    toma del PRESUPUESTO por libro (HRP cross-book), fuente unica de verdad."""
    if capital_frac is None:
        from patrones import presupuesto
        capital_frac = presupuesto.frac_libro("rotacion")
    ult = _ultima_rebal()
    if ult is not None and (pd.Timestamp.now() - ult).days < dias_min:
        print(f"LIBRO D (rotacion Europa): ultimo rebalanceo hace {(pd.Timestamp.now()-ult).days}d "
              f"(<{dias_min}); mensual, no toca.")
        return
    panel, adv = _panel_diario(con_adv=True)
    if panel.empty or len(panel.columns) < 4:
        print("LIBRO D: sin datos suficientes de la cesta.")
        return
    cta = AT.cuenta()
    equity = float(cta["equity"])
    capital = equity * capital_frac
    obj, mom = objetivo_rotacion(panel, capital)
    # GUARD DE CAPACIDAD: ningun objetivo supera ADV_FRAC del volumen $ diario del ETF.
    # Los ETFs finos (EWO ~$1M/dia, EWK ~$0.4M/dia) mueven el precio con posiciones
    # grandes; sin esto, a cuentas >$1M el Libro D chocaria con impacto de mercado.
    for t in list(obj):
        tope = ADV_FRAC * adv.get(t, float("inf"))
        if obj[t] > tope:
            print(f"  [capacidad] {t}: objetivo ${obj[t]:,.0f} -> capado a ${tope:,.0f} "
                  f"(1% de ${adv.get(t,0)/1e6:.1f}M ADV)")
            obj[t] = tope
    pos = AT.posiciones()
    nombres = sorted(set(obj) | set(t for t in CESTA if t in pos))
    px = AT.precio_actual(nombres) if nombres else {}
    print(f"LIBRO D ROTACION EUROPA | capital ${capital:,.0f} ({capital_frac:.1%} del equity)")
    print("Momentum 6m por pais (señal):")
    for t in CESTA:
        m = float(mom.get(t, float("nan")))
        print(f"  {t:<5} mom {m*100:+6.1f}%  ->  ${obj.get(t, 0):,.0f}" + ("" if t in obj else "  (fuera)"))
    plan = []
    for t in nombres:
        p = px.get(t, 0)
        if not p:
            continue
        objc = int(obj.get(t, 0) / p)
        actual = int(float(pos.get(t, {}).get("qty", 0)))
        delta = objc - actual
        if abs(delta * p) > capital * 0.03:
            plan.append((t, "buy" if delta > 0 else "sell", abs(delta), p))
    if not plan:
        print("  Cartera del Libro D ya alineada; nada que hacer.")
        _marcar_rebal()
        return
    print(f"PLAN DE ORDENES ({len(plan)}):")
    for t, l, q, p in sorted(plan, key=lambda x: -x[2] * x[3]):
        print(f"  {l.upper():<4} {q:>4} {t:<5} (~${q*p:,.0f})")
    if not enviar:
        print("[DRY-RUN] No se envia nada. Usa --enviar para rebalancear el Libro D.")
        return
    # GATE DE RIESGO (Tier 1, fail-closed): si no es GO, no COMPRA; las ventas si.
    try:
        from patrones import riesgo as _Rk
        g = _Rk.gate_abrir(cuenta=cta, posiciones=pos)
        if not g["abrir"]:
            nb = sum(1 for _, l, _, _ in plan if l == "buy")
            print(f"[RIESGO {g['decision']}]: {'; '.join(g['razones'][:2])} -> se omiten {nb} compras.")
            plan = [x for x in plan if x[1] == "sell"]
    except Exception:
        pass
    ok = 0
    for t, l, q, p in plan:
        try:
            r = AT.enviar_orden(t, q, l, confirmar=True, libro="rotacion")
            if r.get("id"):
                ok += 1
        except Exception as e:
            print(f"  error {t}: {str(e)[:60]}")
    _marcar_rebal()
    print(f"Libro D (rotacion Europa) rebalanceado: {ok}/{len(plan)} ordenes (paper).")


if __name__ == "__main__":
    ejecutar(enviar="--enviar" in sys.argv)
