"""Runner de la cartera demo: dos libros sobre la cuenta PAPER de Alpaca.

  - SWING: momentum long-only (rebalanceo mensual).  40% del capital.
  - INTRADIA: 'stocks in play' (VWAP + RVOL) con ordenes bracket, fusionando las
    señales de 1h y 15m (sin duplicar tickers).  60% del capital (mas impacto).

Uso:
  python ejecutar_demo.py            # dry-run combinado
  python ejecutar_demo.py --enviar   # ejecuta ambos libros en el paper
"""
import sys

from fuentes import alpaca_trade as AT
from fuentes.descarga import cargar_mercado
import operativa as SW
import operativa_intradia as OI

FRAC_SWING = 0.40
FRAC_INTRA = 0.60


def plan_swing(cap):
    precios, _ = cargar_mercado("datos/mercado", intervalo="1d")
    objetivo, scores = SW.objetivo_momentum(precios, cap)
    pos = AT.posiciones()
    px = AT.precio_actual(sorted(set(objetivo) | set(pos)))
    return SW.plan_rebalanceo(objetivo, px, pos), scores


def plan_intra(cap):
    """Fusiona candidatos de 1h y 15m (dedup por ticker, mejor RVOL) y reparte cap."""
    cand = {}
    for iv in ("1Hour", "15Min"):
        _, ordenes = OI.plan_intradia(cap, intervalo=iv, verbose=False)
        for o in ordenes:
            o["iv"] = iv
            if o["ticker"] not in cand or o["rvol"] > cand[o["ticker"]]["rvol"]:
                cand[o["ticker"]] = o
    cand = list(cand.values())
    if cand:
        por_pos = cap / len(cand)
        for o in cand:
            o["qty"] = int(por_pos / o["precio"])   # re-size al split fusionado
    return [o for o in cand if o["qty"] > 0]


def main(enviar=False):
    cta = AT.cuenta()
    equity = float(cta["equity"])
    cap_sw, cap_in = equity * FRAC_SWING, equity * FRAC_INTRA
    import requests
    abierto = requests.get("https://paper-api.alpaca.markets/v2/clock",
                           headers=AT._headers(), timeout=20).json().get("is_open")
    print(f"Cuenta PAPER: equity ${equity:,.0f} | mercado {'ABIERTO' if abierto else 'CERRADO'}")
    print(f"  SWING ${cap_sw:,.0f} (40%)  ·  INTRADIA ${cap_in:,.0f} (60%)")

    ps, _ = plan_swing(cap_sw)
    pi = plan_intra(cap_in)
    print(f"\nSWING (momentum, {len(ps)} ordenes):")
    for o in ps:
        print(f"  {o['lado'].upper():<4} {o['qty']:>4} {o['ticker']:<6} (~${o['dolar_aprox']:,.0f})")
    print(f"\nINTRADIA (bracket, {len(pi)} ordenes, 1h+15m fusionadas):")
    for o in pi:
        print(f"  LARGO {o['qty']:>4} {o['ticker']:<6} [{o['iv'][:3]}] entrada ~${o['precio']:.2f} "
              f"-> TP ${o['tp']:.2f} / SL ${o['sl']:.2f}  (RVOL {o['rvol']:.2f})")

    if not enviar:
        print("\n[DRY-RUN] Nada enviado. Ejecuta con --enviar.")
        return
    print("\n--- Enviando INTRADIA (bracket) ---")
    for o in pi:
        try:
            r = AT.enviar_bracket(o["ticker"], o["qty"], o["tp"], o["sl"], confirmar=True)
            print(f"  {o['ticker']}: {'OK ' + r.get('id','')[:8] if r.get('id') else r}")
        except Exception as e:
            print(f"  {o['ticker']}: error {str(e)[:80]}")
    print("\n--- Enviando SWING (market) ---")
    ok = 0
    for o in ps:
        try:
            r = AT.enviar_orden(o["ticker"], o["qty"], o["lado"], confirmar=True)
            if r.get("id"):
                ok += 1
        except Exception as e:
            print(f"  {o['ticker']}: error {str(e)[:80]}")
    print(f"SWING: {ok}/{len(ps)} enviadas. Revisa el panel de Alpaca.")


if __name__ == "__main__":
    main(enviar="--enviar" in sys.argv)
