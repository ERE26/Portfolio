"""Transparencia de la seleccion intradia: para CADA valor del universo dice si se
elige, si pasa pero no entra, o POR QUE se descarta (el primer gate que falla). Sirve
para auditar la concentracion ('¿por que siempre los mismos?') con datos, no intuicion.

  python diagnostico_intradia.py
"""
import sys
import numpy as np, pandas as pd

import gestionar as G
from operativa_intradia import CONFIG
from fuentes import alpaca_trade as AT
from registro import tickers_en_cooldown, cargar_umbral, ajuste_memoria

IV = "15Min"


def _senal_segura(t):
    try:
        return G.señal(t, IV)
    except Exception:
        return None


def main():
    cfg = CONFIG[IV]
    rvol_scan = cfg["rvol_min"]           # 1.3
    rvol_fuerte = cargar_umbral(1.5)      # 1.5 (o afinado)
    mkt = G._ret_sesion_mercado(IV)       # retorno de SPY hoy
    pos = AT.posiciones()
    held = set(pos)
    cooldown = tickers_en_cooldown(minutos=60)
    universo = G.universo_dinamico()
    libres = max(0, min(3, G.LIMITES["max_posiciones"] if hasattr(G, "LIMITES") else 30))  # informativo

    print(f"Universo intradia: {len(universo)} valores | mercado (SPY) sesion {mkt*100:+.2f}%")
    print(f"En cartera: {len(held)} | en cooldown (60m): {sorted(cooldown)}")
    print(f"Gates 15m: RVOL>= {rvol_fuerte} (fuerte) | sobre VWAP | 0<ext<{G.EXT_MAX} ATR | "
          f"fuerza_rel> {G.MARGEN_FR*100:.1f}% | sesion>= {G.RET_MIN*100:.1f}% | sent>=-0.35\n")

    filas = []
    for t in universo:
        s = _senal_segura(t)
        if s is None:
            filas.append({"t": t, "estado": "sin datos", "motivo": "no hay velas 15m suficientes"})
            continue
        rv, dv, rs = s["rvol"], s["dist_vwap"], s["ret_sesion"]
        atr = s.get("atr_pct", 0.0)
        fr = rs - mkt
        ext = dv / atr if atr > 0 else 0.0
        base = {"t": t, "rvol": rv, "dvwap": dv, "ret": rs, "fr": fr, "ext": ext}
        # clasificacion por el PRIMER gate que falla (orden del sistema)
        if t in held:
            base.update(estado="en cartera", motivo="ya la tienes (no se dobla)")
        elif t in cooldown:
            base.update(estado="cooldown", motivo="cerrada hace <60m (evita re-entrar en caliente)")
        elif rv < rvol_fuerte:
            base.update(estado="descartada", motivo=f"RVOL {rv:.2f} < {rvol_fuerte} (volumen no anormal)")
        elif dv <= 0:
            base.update(estado="descartada", motivo=f"bajo VWAP ({dv*100:+.2f}%): vendedores al mando")
        elif not (0 < ext < G.EXT_MAX):
            base.update(estado="descartada", motivo=f"sobre-extendida {ext:.1f} ATR sobre VWAP (ya estirada)")
        elif fr <= G.MARGEN_FR:
            base.update(estado="descartada", motivo=f"fuerza rel {fr*100:+.2f}% <= +{G.MARGEN_FR*100:.1f}% (no bate al mercado)")
        elif rs < G.RET_MIN:
            base.update(estado="descartada", motivo=f"sesion {rs*100:+.2f}% < +{G.RET_MIN*100:.1f}% (movimiento flojo)")
        else:
            base.update(estado="PASA", motivo="cumple toda la confluencia")
        filas.append(base)

    df = pd.DataFrame(filas)
    pasan = df[df["estado"] == "PASA"].copy()
    # conviccion para los que pasan (reproduce rankear_conviccion, sin el gate de sentimiento fino)
    elegidas = []
    if len(pasan):
        pasan["sent"] = [G._sentimiento(t) for t in pasan["t"]]
        pasan = pasan[pasan["sent"] >= -0.35].copy()
        if len(pasan):
            aj = ajuste_memoria(list(pasan["t"]))
            pasan["mem"] = [aj.get(str(t), 0.0) for t in pasan["t"]]
            pasan["conv"] = (G._z(pasan["rvol"]).to_numpy() + G._z(pasan["ret"]).to_numpy()
                             + G._z(pasan["fr"]).to_numpy() + 0.5 * pasan["sent"].clip(-1, 1).to_numpy()
                             + pasan["mem"].to_numpy())
            pasan = pasan.sort_values("conv", ascending=False).reset_index(drop=True)
            # SUELO DE CONVICCION: solo entran los >= umbral; el resto se quedan fuera
            elegibles = pasan[pasan["conv"] >= G.UMBRAL_CONVICCION]
            elegidas = list(elegibles["t"].head(3))

    print("=" * 78)
    print(f"PASAN LA CONFLUENCIA: {len(pasan)}  ->  suelo de convicción >= {G.UMBRAL_CONVICCION:.2f}, "
          f"top-3 elegibles entran")
    print("=" * 78)
    if len(pasan):
        print(f"  {'':14}{'ticker':<7}{'RVOL':>6}{'VWAP':>8}{'sesión':>8}{'f.rel':>8}{'ext':>6}{'sent':>7}{'conv':>7}")
        for i, r in pasan.iterrows():
            if r["t"] in elegidas:
                marca = "ELEGIDA"
            elif r["conv"] >= G.UMBRAL_CONVICCION:
                marca = "elegible s/hueco"      # pasa el suelo pero sin hueco
            else:
                marca = "BAJO SUELO"             # convicción < suelo -> NO entra (antes sí)
            print(f"  {marca:<14} {r['t']:<6}{r['rvol']:>6.2f}{r['dvwap']*100:>7.2f}%{r['ret']*100:>7.2f}%"
                  f"{r['fr']*100:>7.2f}%{r['ext']:>6.1f}{r['sent']:>+7.2f}{r['conv']:>7.2f}")
        nbajo = int((pasan["conv"] < G.UMBRAL_CONVICCION).sum())
        if nbajo:
            print(f"\n  ({nbajo} pasan la confluencia pero quedan BAJO el suelo de convicción -> "
                  f"NO se operan; mejor efectivo que un setup flojo.)")
    else:
        print("  (ninguna) -> EFECTIVO: mejor no operar que operar mal.")

    print("\n" + "=" * 74)
    print("EL RESTO — por qué NO (primer filtro que falla):")
    print("=" * 74)
    for est in ["en cartera", "cooldown", "descartada", "sin datos"]:
        sub = df[df["estado"] == est]
        if not len(sub):
            continue
        print(f"\n[{est}] {len(sub)}:")
        for _, r in sub.sort_values("rvol", ascending=False, na_position="last").iterrows():
            rvtxt = f"RVOL {r['rvol']:.2f}  " if pd.notna(r.get("rvol")) else ""
            print(f"  {r['t']:<6} {rvtxt}{r['motivo']}")


if __name__ == "__main__":
    main()
