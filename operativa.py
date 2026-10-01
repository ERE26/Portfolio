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
from patrones import salidas as SAL


def regimen_mercado(precios, indices=("SPY", "DIA", "QQQ", "VOO"), buffer=0.0):
    """Regimen del mercado: ¿esta el indice por encima de su media de 200 sesiones?

    Es el OVERLAY de salida mas robusto del libro diario (ver figura 27 /
    docs/estrategia_diaria_objetivos.md): estar fuera en regimen bajista casi
    HALVES el drawdown del momentum y sube el Sharpe (esquiva los crashes de
    momentum). Usa un ETF ancho si esta en el universo; si no, un indice
    equiponderado de todos los valores. 'buffer' exige caer un poco por debajo de
    la media para declararse bajista (evita flip-flop en el borde)."""
    panel = TV.panel_precios(precios)
    serie, nombre = None, None
    for ix in indices:
        if ix in panel.columns and panel[ix].notna().sum() > 220:
            serie, nombre = panel[ix].dropna(), ix
            break
    if serie is None:
        serie, nombre = SAL.indice_equiponderado(panel).dropna(), "equiponderado"
    sma = float(SAL.sma(serie, 200).iloc[-1])
    px = float(serie.iloc[-1])
    dist = px / sma - 1 if sma else 0.0
    alcista = dist >= -abs(buffer)
    return {"alcista": bool(alcista), "indice": nombre, "precio": px,
            "sma200": sma, "dist": dist}


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
    reg = regimen_mercado(precios, buffer=0.01)
    objetivo, scores = objetivo_momentum(precios, cap_swing)
    if not reg["alcista"]:
        # regimen bajista: no se arma el libro de momentum (a liquidez). Es el
        # overlay de salida que esquiva los crashes de momentum (figura 27).
        print(f"\nREGIMEN BAJISTA [{reg['indice']} {reg['dist']*100:+.1f}% vs SMA200]: "
              "objetivo = LIQUIDEZ (no se arma el libro swing).")
        objetivo, scores = {}, scores.iloc[:0] if hasattr(scores, "iloc") else scores
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


def revisar_swing(enviar=False, q=0.2, buffer=0.01):
    """Gestiona las SALIDAS del libro diario/swing (cuando vender es tan importante
    como cuando comprar). Para cada posicion swing reporta sus OBJETIVOS:
      - momento 12-1 y si sigue en el quintil lider (si cae, se vende al rebalanceo);
      - nivel de trailing stop (informativo) y si ha roto su tendencia 200d;
      - REGIMEN de mercado: si es bajista, el libro swing se protege a liquidez.
    Solo con enviar=True ejecuta (cierra swing en regimen bajista). El intradia no
    se toca aqui (lo gestiona gestionar.py)."""
    precios, _ = cargar_mercado("datos/mercado", intervalo="1d")
    reg = regimen_mercado(precios, buffer=buffer)
    pos = AT.posiciones()
    intradia = set(o["symbol"] for o in AT.ordenes_abiertas())   # con bracket = intradia
    swing = {t: v for t, v in pos.items() if t not in intradia}
    panel = TV.panel_precios(precios)
    mom = TV.score_momentum(panel).dropna().sort_values(ascending=False)
    lider = set(mom.index[:max(int(len(mom) * q), 1)])            # quintil superior actual

    estado = "ALCISTA (risk-on)" if reg["alcista"] else "BAJISTA (risk-off)"
    print(f"REGIMEN de mercado [{reg['indice']}]: {estado}  "
          f"precio {reg['precio']:.2f} vs SMA200 {reg['sma200']:.2f} ({reg['dist']*100:+.1f}%)")
    print(f"Libro SWING: {len(swing)} posiciones  |  quintil lider actual: {len(lider)} valores")
    if not swing:
        print("  (sin posiciones swing)")
    for t in sorted(swing, key=lambda x: -float(swing[x].get("pl", 0))):
        m = float(mom.get(t, float("nan")))
        en_lider = "LIDER" if t in lider else "fuera-quintil (vender al rebalanceo)"
        tend = ""
        if t in precios and {"high", "low", "close"}.issubset(precios[t].columns):
            try:
                st = SAL.estado_salida(precios[t], indice_close=None, mom=m)
                tend = (" | rompe 200d" if st["rompe_tendencia"] else "") + \
                       (" | pierde trailing" if st["pierde_trailing"] else "")
            except Exception:
                pass
        print(f"  {t:<6} PL ${float(swing[t].get('pl',0)):+8.0f}  momento {m*100:+6.1f}%  "
              f"{en_lider}{tend}")

    if not reg["alcista"] and swing:
        print(f"\nREGIMEN BAJISTA -> proteger el libro swing a liquidez ({len(swing)} ventas).")
        if enviar:
            print(" ", AT.cerrar_posiciones(list(swing), confirmar=True))
        else:
            print(f"  [dry-run] cerraria swing: {sorted(swing)}")
    elif reg["alcista"]:
        print("\nRegimen alcista: se mantiene el libro swing (deja correr a los lideres).")


if __name__ == "__main__":
    if "--revisar" in sys.argv:
        revisar_swing(enviar="--enviar" in sys.argv)
    else:
        ejecutar(enviar="--enviar" in sys.argv)
