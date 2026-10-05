"""ESTRATEGIA 4 - TIME-OF-DAY: ¿hay sesgo horario explotable?

Caracteriza, sobre el universo limpio (point-in-time, acciones comunes), el retorno
medio por TRAMO HORARIO de la sesion (ET), con bootstrap IC95:
  - retorno de cada tramo = close(fin tramo)/open(inicio tramo) - 1 (en bps).
  - tramos: 09:30-10:00, 10:00-11:00, 11:00-12:00, 12:00-13:00, 13:00-14:00,
    14:00-15:00, 15:00-16:00 (la "ultima hora").
Ademas dos tests direccionales:
  - MOMENTUM ultima hora: retorno 15:00-16:00 CONDICIONADO al signo de la manana
    (09:30-15:00). ¿La tendencia del dia continua en la ultima hora?
  - REVERSION mediodia: retorno 12:00-14:00 condicionado al signo de 09:30-12:00.

Es descriptivo (no una estrategia con stops): mide si existe el sesgo antes de
intentar explotarlo. Retornos BRUTOS (sin coste) -> si no hay sesgo bruto, menos
aun neto. 3 semillas, universo_pit.

  python backtest_timeofday_massive.py [--tickers 40] [--dias 20] [--meses 20]
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

import backtest_nuevos_massive as H

TRAMOS = [("09:30-10:00", "09:30", "10:00"), ("10:00-11:00", "10:00", "11:00"),
          ("11:00-12:00", "11:00", "12:00"), ("12:00-13:00", "12:00", "13:00"),
          ("13:00-14:00", "13:00", "14:00"), ("14:00-15:00", "14:00", "15:00"),
          ("15:00-16:00", "15:00", "16:00")]


def _t(s):
    h, m = s.split(":")
    return dt.time(int(h), int(m))


def _ret_tramo(df, ini, fin):
    """Retorno close(fin)/open(ini)-1 dentro de la sesion. None si falta cobertura."""
    idx = df.index
    tt = pd.DatetimeIndex(idx).time
    m = (tt >= _t(ini)) & (tt < _t(fin))
    sub = df[m]
    if len(sub) < 2:
        return None
    return float(sub["close"].iloc[-1] / sub["open"].iloc[0] - 1.0)


def run(tickers=40, dias=20, meses=20, seeds=(42, 7, 123), cm=None):
    cm = cm or H.cliente()
    # por semilla: retornos por tramo + series para los tests condicionales
    por_seed_tramo = {s: {name: [] for name, _, _ in TRAMOS} for s in seeds}
    cond = {s: {"mom_cont": [], "mom_all": [], "rev_cont": [], "rev_all": []} for s in seeds}
    for seed in seeds:
        pares = H.muestrear(seed, clase="CS", n_tickers=tickers, n_dias=dias, meses=meses, cm=cm)
        n_eval = 0
        for tk, fecha in pares:
            df = H.cargar(tk, fecha, cm=cm)
            if len(df) < 20:
                continue
            n_eval += 1
            for name, ini, fin in TRAMOS:
                r = _ret_tramo(df, ini, fin)
                if r is not None:
                    por_seed_tramo[seed][name].append(r)
            # momentum ultima hora: signo manana 09:30-15:00 vs retorno 15:00-16:00
            manana = _ret_tramo(df, "09:30", "15:00")
            ult = _ret_tramo(df, "15:00", "16:00")
            if manana is not None and ult is not None and manana != 0:
                signo = np.sign(manana)
                cond[seed]["mom_cont"].append(signo * ult)   # retorno "siguiendo" la tendencia
                cond[seed]["mom_all"].append(ult)
            # reversion mediodia: signo 09:30-12:00 vs retorno 12:00-14:00
            am = _ret_tramo(df, "09:30", "12:00")
            mid = _ret_tramo(df, "12:00", "14:00")
            if am is not None and mid is not None and am != 0:
                cond[seed]["rev_cont"].append(-np.sign(am) * mid)  # retorno "fadeando" la manana
                cond[seed]["rev_all"].append(mid)
        print(f"# seed {seed}: {len(pares)} pares, {n_eval} con datos")
    return por_seed_tramo, cond, seeds


def _fila_bps(nombre, seed, xs):
    xs = np.asarray(xs, float)
    if len(xs) < 5:
        return f"| {nombre} | {seed} | {len(xs)} | - | - |"
    lo, hi = H.boot_ci(xs)
    sig = "SI" if (lo > 0 or hi < 0) else "no"
    return (f"| {nombre} | {seed} | {len(xs)} | {xs.mean()*1e4:+.1f} | "
            f"[{lo*1e4:+.1f}, {hi*1e4:+.1f}] bps | {sig} |")


def informe(por_seed_tramo, cond, seeds):
    print("### Retorno medio por tramo horario (bruto, bps)\n")
    print("| tramo ET | semilla | n | media | IC95 | signif.? |")
    print("|---|---|---:|---:|---|---|")
    for name, _, _ in TRAMOS:
        for s in seeds:
            print(_fila_bps(name, s, por_seed_tramo[s][name]))
        pooled = sum((por_seed_tramo[s][name] for s in seeds), [])
        print(_fila_bps(name, "pooled", pooled))
    print("\n### Tests direccionales (bruto, bps)\n")
    print("| test | semilla | n | media | IC95 | signif.? |")
    print("|---|---|---:|---:|---|---|")
    etiquetas = {"mom_cont": "mom ult.hora (sig manana * ret)",
                 "rev_cont": "rev mediodia (-sig manana * ret)"}
    for key, et in etiquetas.items():
        for s in seeds:
            print(_fila_bps(et, s, cond[s][key]))
        print(_fila_bps(et, "pooled", sum((cond[s][key] for s in seeds), [])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=40)
    ap.add_argument("--dias", type=int, default=20)
    ap.add_argument("--meses", type=int, default=20)
    a = ap.parse_args()
    pst, cond, seeds = run(a.tickers, a.dias, a.meses)
    print()
    informe(pst, cond, seeds)
