"""LIBRO OVERNIGHT — la única estrategia que sobrevivió la re-validación limpia (Massive).

Edge (docs/intradia_massive_resultado.md): la prima de la renta variable se gana sobre todo
DE NOCHE (close->open); el intradía aporta poco. Operamos solo la noche y solo en tendencia:
  - Universo: QQQ (núcleo) + SPY, IWM (cesta de diversificación overnight).
  - Señal diaria (al cierre): si el ÚLTIMO cierre > SMA200 -> mantener la noche; si no, efectivo.
  - Entrada: market-on-close (MOC, tif='cls') al cierre.  Salida: market-on-open (MOO, tif='opg').
  - Sin apalancar. Exposición SOLO overnight y SOLO en tendencia alcista.
Backtest limpio (5 años Massive, sin supervivencia): QQQ Sharpe ~1,3 (IC95 [0,25,2,43], train≈test),
tolerante a coste hasta ~1,5 bp/lado. SPY/IWM sugerentes (IC95 incluye 0) -> cesta, no sueltos.

OBJETIVO DE ESTE PASO (igual que la Fase 0 del ORB): PAPER para medir el SLIPPAGE REAL de las
subastas MOC/MOO (intención = cierre de hoy vs fill real), que es el muro de esta estrategia.
NO toca el resto del sistema. PAPER siempre (claves PK).

  python operativa_overnight.py --entrar [--enviar]      # al cierre (~15:45-15:58 ET): MOC buy
  python operativa_overnight.py --salir  [--enviar]      # en apertura (pre 09:28 ET): MOO sell
  python operativa_overnight.py --reconciliar            # rellena fills, overnight return y slippage
Sin --enviar = dry-run (no manda órdenes).
"""
import os
import sys

import numpy as np
import pandas as pd

from fuentes import alpaca
from fuentes import alpaca_trade as AT

LOG = "datos/overnight_paper.csv"
NUCLEO = "QQQ"
CESTA = ["QQQ", "SPY", "IWM"]          # QQQ núcleo; SPY/IWM diversificación
SMA_N = 200                            # filtro de tendencia
# OJO COLISIÓN DE TICKERS: el libro TSMOM ya opera SPY/QQQ/IWM en esta MISMA cuenta paper.
# Para medir el slippage MOC/MOO sin corromper la contabilidad de TSMOM: (1) tamaño de MEDICIÓN
# pequeño (notional fijo, como la Fase 0 del ORB), y (2) salida a NIVEL DE ORDEN (se vende EXACTO
# el lote que compró el overnight, leído del log; nunca "toda la posición"). Para desplegar a
# tamaño completo habría que aislarlo en una cuenta paper separada.
NOTIONAL = 1000.0                      # $ por nombre (sonda de medición, interferencia mínima)
CAMPOS = ["ts", "fecha", "ticker", "etapa", "lado", "tif", "qty", "ref_close", "sma200",
          "order_id", "fill", "slippage_pct", "overnight_ret", "estado"]


def _ahora_et():
    return pd.Timestamp.now(tz="America/New_York")


def _log():
    if os.path.exists(LOG):
        try:
            return pd.read_csv(LOG)
        except Exception:
            pass
    return pd.DataFrame(columns=CAMPOS)


def _guardar(df):
    os.makedirs("datos", exist_ok=True)
    df.to_csv(LOG, index=False)


def _senal(ticker, n=SMA_N):
    """(en_tendencia, ultimo_cierre, sma) usando barras diarias ajustadas. en_tendencia =
    ultimo cierre > SMA200. Point-in-time: usa solo cierres ya conocidos."""
    desde = (pd.Timestamp.now() - pd.Timedelta(days=int(n * 2.0))).strftime("%Y-%m-%d")
    df = alpaca.diario(ticker, desde=desde)
    if len(df) < n:
        return None
    c = df["close"].to_numpy(float)
    sma = float(np.mean(c[-n:]))
    ult = float(c[-1])
    return (ult > sma, ult, sma)


def _abiertas_overnight(df):
    """Lotes overnight propios aún abiertos: qty neta (entradas - salidas) por ticker, contando
    solo órdenes enviadas/hechas. Permite vender EXACTO lo del libro, sin tocar otros libros."""
    out = {}
    if not len(df):
        return out
    viv = df[df["estado"].isin(["enviada", "hecha"])]
    for t in CESTA:
        comp = int(viv[(viv.ticker == t) & (viv.etapa == "entrada")]["qty"].astype(float).sum())
        vend = int(viv[(viv.ticker == t) & (viv.etapa == "salida")]["qty"].astype(float).sum())
        if comp - vend > 0:
            out[t] = comp - vend
    return out


def entrar(confirmar=False):
    """Al cierre: para cada ETF en tendencia (cierre>SMA200), MOC buy de tamaño de MEDICIÓN.
    NO mira la posición total de la cuenta (colisión con TSMOM): solo evita duplicar el lote
    overnight propio si ya hay uno abierto o ya se entró hoy."""
    et = _ahora_et()
    print(f"OVERNIGHT entrar | {et:%Y-%m-%d %H:%M} ET")
    if confirmar and not (et.weekday() < 5 and (et.hour == 15 and et.minute >= 40)):
        print("  AVISO: fuera de la ventana MOC (~15:40-15:59 ET lun-vie); Alpaca puede rechazar 'cls'.")
    df = _log()
    hoy = et.strftime("%Y-%m-%d")
    abiertas = _abiertas_overnight(df)
    ya_hoy = set(df[df["fecha"].astype(str) == hoy]["ticker"]) if len(df) else set()
    nuevos = []
    for t in CESTA:
        if t in abiertas:
            print(f"  {t}: lote overnight propio aún abierto ({abiertas[t]}) -> no re-entro"); continue
        if t in ya_hoy:
            print(f"  {t}: ya operado hoy por el libro overnight"); continue
        s = _senal(t)
        if s is None:
            print(f"  {t}: sin datos suficientes"); continue
        en_tend, ult, sma = s
        if not en_tend:
            print(f"  {t}: cierre {ult:.2f} < SMA200 {sma:.2f} -> EFECTIVO (no entro)"); continue
        qty = int(NOTIONAL / ult) if ult > 0 else 0
        if qty <= 0:
            print(f"  {t}: qty 0"); continue
        fila = {"ts": et.isoformat(), "fecha": hoy, "ticker": t, "etapa": "entrada",
                "lado": "buy", "tif": "cls", "qty": qty, "ref_close": round(ult, 2),
                "sma200": round(sma, 2), "order_id": "", "fill": np.nan,
                "slippage_pct": np.nan, "overnight_ret": np.nan, "estado": "pend"}
        if confirmar:
            r = AT.enviar_orden(t, qty, "buy", confirmar=True, tif="cls", libro="overnight")
            fila["order_id"] = r.get("id", "")
            fila["estado"] = "enviada" if r.get("id") else "error"
            print(f"  MOC BUY {t} x{qty} @~{ult:.2f} (SMA200 {sma:.2f}) -> {fila['estado']}")
        else:
            print(f"  [dry] MOC BUY {t} x{qty} @~{ult:.2f} (cierre>{sma:.2f} SMA200)")
        nuevos.append(fila)
    if confirmar and nuevos:          # el dry-run NO persiste (solo informa)
        _guardar(pd.concat([df, pd.DataFrame(nuevos)], ignore_index=True))
    elif not nuevos:
        print("  (ningún ETF en tendencia / nada nuevo que operar)")


def salir(confirmar=False):
    """En la apertura: MOO sell de todo lo que mantengamos del libro overnight."""
    et = _ahora_et()
    print(f"OVERNIGHT salir | {et:%Y-%m-%d %H:%M} ET")
    if confirmar and not (et.weekday() < 5 and et.hour < 9 or (et.hour == 9 and et.minute <= 27)):
        print("  AVISO: fuera de la ventana MOO (pre 09:28 ET); Alpaca puede rechazar 'opg'.")
    df = _log()
    abiertas = _abiertas_overnight(df)          # solo el lote propio del overnight, NO la posición total
    if not abiertas:
        print("  sin lote overnight propio que cerrar (no se toca lo de otros libros)."); return
    try:
        pos = AT.posiciones()
    except Exception as e:
        print(f"  error posiciones: {str(e)[:60]}"); return
    filas = []
    for t, qty in abiertas.items():
        et_iso = et.isoformat()
        fila = {"ts": et_iso, "fecha": et.strftime("%Y-%m-%d"), "ticker": t, "etapa": "salida",
                "lado": "sell", "tif": "opg", "qty": qty, "ref_close": (pos.get(t, {}) or {}).get("precio"),
                "sma200": np.nan, "order_id": "", "fill": np.nan, "slippage_pct": np.nan,
                "overnight_ret": np.nan, "estado": "pend"}
        if confirmar:
            r = AT.enviar_orden(t, qty, "sell", confirmar=True, tif="opg", libro="overnight")
            fila["order_id"] = r.get("id", "")
            fila["estado"] = "enviada" if r.get("id") else "error"
            print(f"  MOO SELL {t} x{qty} -> {fila['estado']}")
        else:
            print(f"  [dry] MOO SELL {t} x{qty}")
        filas.append(fila)
    if confirmar and filas:
        _guardar(pd.concat([df, pd.DataFrame(filas)], ignore_index=True))


def reconciliar():
    """Rellena fills reales (filled_avg_price), slippage vs intención y retorno overnight
    emparejando cada salida con su entrada previa del mismo ticker."""
    df = _log()
    if not len(df):
        return
    for i, r in df.iterrows():
        if r.get("order_id") and (pd.isna(r.get("fill")) or r.get("fill") == ""):
            try:
                fav = AT.orden(str(r["order_id"])).get("filled_avg_price")
                if fav:
                    fill = float(fav)
                    df.at[i, "fill"] = fill
                    ref = float(r["ref_close"]) if r.get("ref_close") == r.get("ref_close") else np.nan
                    if ref and ref > 0:
                        df.at[i, "slippage_pct"] = (fill - ref) / ref
                    df.at[i, "estado"] = "hecha"
            except Exception:
                continue
    # retorno overnight: para cada salida con fill, busca la última entrada con fill del mismo ticker
    for i, r in df.iterrows():
        if r.get("etapa") == "salida" and r.get("fill") == r.get("fill"):
            prev = df[(df.ticker == r.ticker) & (df.etapa == "entrada") & (df.fill.notna())
                      & (df.ts < r.ts)]
            if len(prev):
                buy = float(prev.iloc[-1]["fill"])
                if buy > 0:
                    df.at[i, "overnight_ret"] = float(r["fill"]) / buy - 1.0
    _guardar(df)
    hechas = df[df["fill"].notna()]
    ent = hechas[hechas.etapa == "entrada"]
    sal = hechas[(hechas.etapa == "salida") & hechas.overnight_ret.notna()]
    if len(ent):
        s = ent["slippage_pct"].astype(float).dropna()
        print(f"OVERNIGHT reconciliado | entradas con fill: {len(ent)} | "
              f"slippage MOC medio {s.mean()*100:+.3f}% mediano {s.median()*100:+.3f}%")
    if len(sal):
        onr = sal["overnight_ret"].astype(float)
        print(f"  noches cerradas: {len(sal)} | overnight return medio {onr.mean()*100:+.3f}% | "
              f"acierto {(onr>0).mean()*100:.0f}%")


if __name__ == "__main__":
    if "--reconciliar" in sys.argv:
        reconciliar()
    elif "--salir" in sys.argv:
        salir(confirmar="--enviar" in sys.argv)
    elif "--entrar" in sys.argv:
        entrar(confirmar="--enviar" in sys.argv)
    else:
        print(__doc__)
