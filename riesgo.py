"""Informe de RIESGO de cartera + kill-switches (Tier 1). Hora de Madrid.

Uso:
  python riesgo.py                 # informe + veredicto (no toca nada)
  python riesgo.py --enforce       # si el veredicto es HALT, aplana intradia (paper)

Convierte 'varias estrategias' en 'un fondo': mide el riesgo del CONJUNTO (exposicion,
concentracion, VaR, correlacion), reparte RIESGO entre libros (paridad por vol) y
aplica limites duros de preservacion de capital. El bucle de trading consulta
`evaluar_riesgo` antes de abrir nuevas posiciones.
"""
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca_trade as AT
from patrones import riesgo as R


def _madrid():
    try:
        return pd.Timestamp.now(tz="Europe/Madrid").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def _ordenes_recientes():
    try:
        import requests
        r = requests.get(AT._url("orders"),
                         params={"status": "all", "limit": 500, "nested": "false"},
                         headers=AT._headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _snapshot_libros(exp_libro):
    """Acumula el valor bruto y P&L por libro cada dia en datos/riesgo_libros.csv,
    para ir construyendo el historico de retornos por libro (-> HRP a futuro)."""
    import os
    os.makedirs("datos", exist_ok=True)
    ruta = "datos/riesgo_libros.csv"
    fila = {"fecha": R._hoy()}
    for lb, d in exp_libro.items():
        fila[f"{lb}_bruta"] = round(d["bruta"], 4)
        fila[f"{lb}_pl"] = round(d["pl"], 2)
    try:
        df = pd.read_csv(ruta) if os.path.exists(ruta) else pd.DataFrame()
        df = df[df.get("fecha") != fila["fecha"]] if "fecha" in df.columns else df
        df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
        df.to_csv(ruta, index=False)
    except Exception:
        pass


def main(enforce=False):
    print("=" * 66)
    print(f"INFORME DE RIESGO DE CARTERA  ·  {_madrid()} (Madrid)")
    print("=" * 66)
    cta = AT.cuenta()
    pos = AT.posiciones()
    ordenes = _ordenes_recientes()
    equity = float(cta.get("equity", 0.0))
    cash = float(cta.get("cash", 0.0))

    # 1) cuenta y P&L
    ev = R.evaluar_riesgo(cta, pos, ordenes)
    m = ev["metricas"]
    print(f"\nEquity ${equity:,.0f} | efectivo ${cash:,.0f} | "
          f"P&L dia {m['pl_dia']:+.2%} | drawdown {m['drawdown']:+.2%} (pico ${m['pico_equity']:,.0f})")

    # 2) exposicion y concentracion
    exp = R.exposiciones(pos, equity)
    print(f"\nEXPOSICION: bruta {exp['bruta']:.2f}x | neta {exp['neta']:+.2f}x | "
          f"{exp['n']} posiciones | mayor {exp['peso_max']:.1%} | top5 {exp['top5']:.1%} | HHI {exp['hhi']:.3f}")

    # 3) VaR de cartera (basado en posiciones, vol condicional EWMA)
    if pos:
        v95 = R.var_cartera(pos, equity, nivel=0.95)
        v99 = R.var_cartera(pos, equity, nivel=0.99)
        print(f"\nVaR 1 dia (parametrico, {v95['n_cubiertos']}/{len(pos)} cubiertos):")
        print(f"  95%: {v95['var_pct']:.2%}  (${v95['var_usd']:,.0f})   "
              f"99%: {v99['var_pct']:.2%}  (${v99['var_usd']:,.0f})")
        print(f"  vol diaria cartera {v95['vol_diaria']:.2%} | correlacion media holdings "
              f"{v95['corr_media']:+.2f}  ({'DIVERSIFICADA' if v95['corr_media']<0.4 else 'CONCENTRADA EN RIESGO'})")
    else:
        print("\nVaR: sin posiciones abiertas.")

    # 4) reparto por libro + presupuesto de riesgo (paridad por vol actual)
    exp_libro = R.exposicion_por_libro(pos, equity, ordenes)
    _snapshot_libros(exp_libro)
    if exp_libro:
        print("\nREPARTO POR LIBRO (exposicion bruta real vs objetivo por paridad de riesgo):")
        libros = R.atribuir_libros(pos, ordenes)
        volb = {}
        for lb in exp_libro:
            sub = {t: pos[t] for t in pos if libros.get(t) == lb}
            vv = R.var_cartera(sub, equity, nivel=0.95) if sub else {"vol_diaria": float("nan")}
            volb[lb] = vv["vol_diaria"]
        inv = {lb: (1.0 / v) for lb, v in volb.items() if v and np.isfinite(v) and v > 0}
        tot = sum(inv.values())
        objetivo = {lb: inv.get(lb, 0.0) / tot for lb in exp_libro} if tot > 0 else {}
        bruta_tot = sum(d["bruta"] for d in exp_libro.values()) or 1.0
        for lb, d in sorted(exp_libro.items(), key=lambda x: -x[1]["bruta"]):
            share = d["bruta"] / bruta_tot
            obj = objetivo.get(lb, float("nan"))
            vol = volb.get(lb, float("nan"))
            flag = ""
            if np.isfinite(obj) and share > obj + 0.15:
                flag = "  <-- SOBRE-ASIGNADO vs riesgo"
            print(f"  {lb:<15} expos {d['bruta']:.2f}x ({share:.0%} del total) | "
                  f"vol {vol:.1%} | objetivo riesgo {obj:.0%} | P&L ${d['pl']:+,.0f}{flag}")
        print("  (objetivo = paridad de riesgo por vol actual; migrara a HRP con historico real por libro)")

    # 5) veredicto kill-switch
    print("\n" + "-" * 66)
    icono = {"GO": "[GO]", "REDUCE": "[REDUCE]", "HALT": "[HALT]"}[ev["decision"]]
    print(f"VEREDICTO: {icono}")
    for r in ev["razones"]:
        print(f"   - {r}")
    print("-" * 66)

    if ev["decision"] == "HALT" and enforce:
        print("\nHALT + --enforce: aplanando intradia (preservacion de capital)...")
        try:
            res = AT.cerrar_intradia(confirmar=True)
            print(f"  cerrado: {res}")
        except Exception as e:
            print(f"  error al aplanar: {str(e)[:80]}")
    elif ev["decision"] == "HALT":
        print("\n(HALT) El bucle NO debe abrir nuevas posiciones. Usa --enforce para aplanar.")
    return ev


if __name__ == "__main__":
    main(enforce="--enforce" in sys.argv)
