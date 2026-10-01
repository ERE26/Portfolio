"""Operativa: traduce la estrategia en un objetivo de cartera y en ordenes para la
cuenta PAPER de Alpaca (demo). Empieza por el momentum transversal long-only, que
da una cartera limpia y ejecutable.

Flujo seguro: (1) calcula el objetivo, (2) calcula el plan de ordenes (dry-run),
(3) solo con confirmar=True envia las ordenes al paper. Nunca toca dinero real
(el conector esta bloqueado al endpoint paper).

Uso:
  python operativa.py            # dry-run: muestra objetivo y ordenes, no envia
  python operativa.py --enviar   # envia las ordenes a la cuenta paper
"""
import sys

from fuentes import alpaca_trade as AT
from fuentes.descarga import cargar_mercado
from patrones import transversal as TV


def objetivo_momentum(precios, equity, q=0.2, exposicion=0.95, lookback=252, skip=21):
    """Objetivo de cartera (dict ticker -> $ objetivo) del momentum long-only:
    quintil superior por momento 12-1, equiponderado, invertido al 'exposicion'."""
    panel = TV.panel_precios(precios)
    mom = TV.score_momentum(panel, lookback, skip).dropna()
    # solo tickers con precio reciente valido
    mom = mom[panel.iloc[-1].reindex(mom.index).notna()]
    n = max(int(len(mom) * q), 1)
    largos = list(mom.sort_values(ascending=False).index[:n])
    w = exposicion / n
    return {t: equity * w for t in largos}, mom.loc[largos]


def plan_rebalanceo(objetivo_dolar, precios_actuales, posiciones):
    """Ordenes para pasar de las posiciones actuales al objetivo (en acciones).
    Devuelve lista de dicts {ticker, lado, qty, dolar_aprox}."""
    objetivo_acc = {}
    for t, d in objetivo_dolar.items():
        p = precios_actuales.get(t)
        if p and p > 0:
            objetivo_acc[t] = int(d / p)
    actuales = {t: int(v["qty"]) for t, v in posiciones.items()}
    ordenes = []
    for t in set(objetivo_acc) | set(actuales):
        delta = objetivo_acc.get(t, 0) - actuales.get(t, 0)
        if delta == 0:
            continue
        p = precios_actuales.get(t, 0)
        ordenes.append({"ticker": t, "lado": "buy" if delta > 0 else "sell",
                        "qty": abs(delta), "dolar_aprox": abs(delta) * (p or 0)})
    return sorted(ordenes, key=lambda o: -o["dolar_aprox"])


def ejecutar(enviar=False, reserva_intradia=0.25):
    cta = AT.cuenta()
    equity = float(cta["equity"])
    cap_swing = equity * (1.0 - reserva_intradia)
    print(f"Cuenta PAPER: equity ${equity:,.0f} | buying_power ${float(cta['buying_power']):,.0f} "
          f"| estado {cta['status']}")
    print(f"Book SWING (momentum): ${cap_swing:,.0f}  ·  reserva intradia ${equity*reserva_intradia:,.0f}")
    precios, _ = cargar_mercado("datos/mercado", intervalo="1d")
    objetivo, scores = objetivo_momentum(precios, cap_swing)
    pos = AT.posiciones()
    nombres = sorted(set(objetivo) | set(pos))
    px = AT.precio_actual(nombres)

    print(f"\nOBJETIVO (momentum long-only, {len(objetivo)} valores, quintil superior):")
    for t in sorted(objetivo, key=lambda x: -scores[x]):
        print(f"  {t:<6} momento {scores[t]*100:+6.1f}%  ->  ${objetivo[t]:,.0f}"
              f" (~{int(objetivo[t]/px[t]) if px.get(t) else '?'} acc)")

    plan = plan_rebalanceo(objetivo, px, pos)
    print(f"\nPLAN DE ORDENES ({len(plan)}):")
    for o in plan:
        print(f"  {o['lado'].upper():<4} {o['qty']:>4} {o['ticker']:<6} (~${o['dolar_aprox']:,.0f})")
    if not plan:
        print("  (cartera ya alineada, nada que hacer)")
        return

    if not enviar:
        print("\n[DRY-RUN] No se ha enviado nada. Ejecuta con --enviar para colocar las ordenes.")
        return
    print("\nEnviando ordenes a la cuenta PAPER...")
    ok = 0
    for o in plan:
        try:
            r = AT.enviar_orden(o["ticker"], o["qty"], o["lado"], confirmar=True)
            if r.get("id"):
                ok += 1
        except Exception as e:
            print(f"  error en {o['ticker']}: {str(e)[:80]}")
    print(f"Enviadas {ok}/{len(plan)} ordenes. Revisa el panel de Alpaca.")


if __name__ == "__main__":
    ejecutar(enviar="--enviar" in sys.argv)
