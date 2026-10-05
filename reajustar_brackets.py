"""Reajusta a niveles REALISTAS los TP/SL de las posiciones intradía (ORB) que aún llevan
el bracket antiguo con take-profit fantasma (+17-23%, puesto antes del fix de TP/SL).

Política:
  - GANADORES (PnL >= +2%): proteger y dejar correr -> SL al menos a breakeven+, TP con
    recorrido sobre el precio actual (mejor beneficio, sin regalar la corredora).
  - RESTO: TP +6% (2R) / SL -3% sobre la ENTRADA.
  - El stop de venta se coloca siempre por DEBAJO del precio actual y el TP por ENCIMA
    (si no, saltaría al instante).
Solo toca posiciones LARGAS con un TP-leg irreal (>TP_FANTASMA de la entrada). No toca
swing/overnight (sin bracket) ni los cortos (fade). PAPER siempre (claves PK).

  python reajustar_brackets.py            # dry-run (muestra el plan)
  python reajustar_brackets.py --enviar   # aplica (cancela legs viejos + OCO nuevo)
"""
import sys

from fuentes import alpaca_trade as AT

TP_FANTASMA = 0.10        # TP a >10% de la entrada => es el bracket viejo irreal


def plan():
    pos = AT.posiciones()
    tp_leg = {}
    for o in AT.ordenes_abiertas():
        if o.get("type") == "limit" and o.get("side") == "sell" and o.get("limit_price"):
            tp_leg[o["symbol"]] = float(o["limit_price"])
    filas = []
    for t, p in pos.items():
        if p["qty"] <= 0:                         # solo largos
            continue
        entry, px = p["entrada"], p["precio"]
        tpa = tp_leg.get(t)
        if tpa is None or entry <= 0:             # sin bracket (swing/overnight) -> no tocar
            continue
        if (tpa / entry - 1) <= TP_FANTASMA:      # TP ya realista -> no tocar
            continue
        pl = px / entry - 1
        if pl >= 0.02:                            # ganador: proteger + dejar correr
            tp = round(px * 1.04, 2)
            sl = round(max(entry * 1.005, px * 0.97), 2)
        else:                                     # resto: 2R realista
            tp = round(entry * 1.06, 2)
            sl = round(entry * 0.97, 2)
        sl = min(sl, round(px * 0.995, 2))        # stop por debajo del precio actual
        if tp <= px:                              # TP por encima del precio actual
            tp = round(px * 1.01, 2)
        filas.append((t, int(p["qty"]), entry, px, tpa, tp, sl))
    return filas


def main(confirmar=False):
    filas = plan()
    if not filas:
        print("No hay brackets fantasma que reajustar.")
        return
    for t, q, entry, px, tpa, tp, sl in filas:
        print(f"{t}: x{q} entry {entry:.2f} px {px:.2f} ({(px/entry-1)*100:+.1f}%)  "
              f"TP {tpa:.2f}(+{(tpa/entry-1)*100:.0f}%) -> {tp:.2f}(+{(tp/entry-1)*100:.1f}%)  "
              f"SL -> {sl:.2f}({(sl/entry-1)*100:+.1f}%)")
        if confirmar:
            try:
                r = AT.reajustar_salida(t, q, tp, sl, lado_salida="sell", confirmar=True)
                print(f"   -> {'OK ' + str(r.get('id',''))[:8] if r.get('id') else r}")
            except Exception as e:
                print(f"   -> error: {str(e)[:80]}")
    if not confirmar:
        print("\n[DRY-RUN] Nada enviado. Ejecuta con --enviar.")


if __name__ == "__main__":
    main(confirmar="--enviar" in sys.argv)
