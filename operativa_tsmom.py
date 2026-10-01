"""Libro C: TSMOM / CTA cross-asset (cuenta PAPER). Flujo de retorno DIVERGENTE y
poco correlacionado con los libros long-biased (intradia + swing). Validado:
DSR 0.995, correlacion 0.25 con el libro diario, y el combinado sube el Sharpe y
casi halves el drawdown (docs/estrategias_fondos.md, figura 29).

Version LONG/FLAT por SEGURIDAD de ejecucion en paper (sin cortos): largo los
activos de la cesta con momentum de serie temporal POSITIVO (escalado por su
volatilidad para igualar riesgo), efectivo los demas. Cuando la renta variable se
gira, el libro rota a los activos defensivos que sigan en tendencia (bonos TLT,
oro GLD) -> ahi esta la diversificacion. Rebalanceo MENSUAL.

Flujo seguro: dry-run por defecto; solo con --enviar coloca ordenes en el paper.
Nunca toca dinero real (conector bloqueado al endpoint paper).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca, alpaca_trade as AT
from patrones.tsmom import signal_tsmom

# cesta cross-asset: renta variable (SPY/QQQ/IWM/DIA), bonos (TLT), oro (GLD),
# credito (HYG). Reservada a este libro (se excluye del intradia para no solaparse).
CESTA = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "HYG"]
HORIZONTES = (21, 63, 126, 252)
ESTADO = "datos/tsmom_rebal.json"


def _panel_diario():
    """Panel de cierres diarios de la cesta para la señal TSMOM. Usa el almacen
    diario (datos/mercado/1d), que cubre la cesta y esta al dia; el feed IEX no
    sirve barras diarias. La cadencia MENSUAL tolera que el almacen se refresque de
    tanto en tanto (basta con que no se quede meses atras)."""
    cols = {}
    try:
        from fuentes.descarga import cargar_mercado
        pre, _ = cargar_mercado("datos/mercado", intervalo="1d")
        for t in CESTA:
            if t in pre and len(pre[t]) > 260:
                cols[t] = pre[t]["close"].astype(float)
    except Exception:
        pass
    return pd.DataFrame(cols).sort_index().ffill(limit=5)


def objetivo_tsmom(panel, capital, vol_obj=0.12, vol_lb=63, cap_peso=1.5):
    """Objetivo en $ por activo (LONG/FLAT): largo si la señal TSMOM > 0, tamaño
    inverso a su volatilidad, normalizado para que la exposicion larga ~ capital."""
    sig = signal_tsmom(panel, HORIZONTES)
    sig = sig[sig > 0]                                   # long/flat: solo positivas
    if len(sig) == 0:
        return {}, signal_tsmom(panel, HORIZONTES)
    vol = panel.pct_change().rolling(vol_lb, min_periods=vol_lb // 2).std().iloc[-1] * np.sqrt(252)
    w = (sig * (vol_obj / vol.reindex(sig.index))).clip(upper=cap_peso).fillna(0.0)
    if w.sum() <= 0:
        return {}, signal_tsmom(panel, HORIZONTES)
    w = w / w.sum()                                      # normaliza al 100% de la exposicion larga
    return {t: capital * float(wi) for t, wi in w.items() if wi > 0}, signal_tsmom(panel, HORIZONTES)


def _ultima_rebal():
    try:
        return pd.Timestamp(json.load(open(ESTADO))["fecha"])
    except Exception:
        return None


def _marcar_rebal():
    os.makedirs("datos", exist_ok=True)
    json.dump({"fecha": pd.Timestamp.now().isoformat()}, open(ESTADO, "w"))


def ejecutar(enviar=False, capital_frac=0.20, dias_min=25):
    """Rebalanceo MENSUAL del Libro C. Se auto-limita: si el ultimo rebalanceo fue
    hace < dias_min, no hace nada (por eso el bucle puede llamarlo cada ciclo sin
    coste). capital_frac = fraccion del equity dedicada a este libro."""
    ult = _ultima_rebal()
    if ult is not None and (pd.Timestamp.now() - ult).days < dias_min:
        print(f"LIBRO C (TSMOM): ultimo rebalanceo hace {(pd.Timestamp.now()-ult).days}d "
              f"(<{dias_min}); mensual, no toca.")
        return
    cta = AT.cuenta()
    equity = float(cta["equity"])
    capital = equity * capital_frac
    panel = _panel_diario()
    if panel.empty or len(panel.columns) < 3:
        print("LIBRO C (TSMOM): sin datos suficientes de la cesta.")
        return
    obj, sig = objetivo_tsmom(panel, capital)
    pos = AT.posiciones()
    nombres = sorted(set(obj) | set(t for t in CESTA if t in pos))
    px = AT.precio_actual(nombres) if nombres else {}
    print(f"LIBRO C TSMOM (cross-asset, long/flat) | capital ${capital:,.0f} "
          f"({capital_frac:.0%} del equity)")
    print("Señal TSMOM del dia (signo del propio pasado):")
    for t in CESTA:
        s = float(sig.get(t, float("nan")))
        print(f"  {t:<5} señal {s:+.2f}  ->  ${obj.get(t, 0):,.0f}"
              + ("" if t in obj else "  (efectivo)"))
    plan = []
    for t in nombres:
        p = px.get(t, 0)
        if not p:
            continue
        objc = int(obj.get(t, 0) / p)
        actual = int(float(pos.get(t, {}).get("qty", 0)))
        delta = objc - actual
        if abs(delta * p) > capital * 0.02:             # ignora derivas pequeñas (<2% del capital)
            plan.append((t, "buy" if delta > 0 else "sell", abs(delta), p))
    if not plan:
        print("  Cartera TSMOM ya alineada; nada que hacer.")
        _marcar_rebal()
        return
    print(f"PLAN DE ORDENES ({len(plan)}):")
    for t, l, q, p in sorted(plan, key=lambda x: -x[2] * x[3]):
        print(f"  {l.upper():<4} {q:>4} {t:<5} (~${q*p:,.0f})")
    if not enviar:
        print("[DRY-RUN] No se envia nada. Usa --enviar para rebalancear el Libro C.")
        return
    ok = 0
    for t, l, q, p in plan:
        try:
            r = AT.enviar_orden(t, q, l, confirmar=True)
            if r.get("id"):
                ok += 1
        except Exception as e:
            print(f"  error {t}: {str(e)[:60]}")
    _marcar_rebal()
    print(f"Libro C TSMOM rebalanceado: {ok}/{len(plan)} ordenes enviadas (paper).")


if __name__ == "__main__":
    ejecutar(enviar="--enviar" in sys.argv)
