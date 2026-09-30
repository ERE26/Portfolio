"""Evaluacion walk-forward honesta, con baselines y backtest con costes.

Esta es la parte que convierte el proyecto en un trabajo defendible. Predecir
mercados es dificil de verdad, asi que lo importante no es sacar un numero
bonito, es que el numero se pueda creer. Aqui se controla:

  - WALK-FORWARD: se entrena solo con el pasado y se predice el futuro, varias
    veces, avanzando en el tiempo. Nunca hay una sola division train/test.
  - PURGA Y EMBARGO: los casos de entrenamiento cuyo resultado aun no se conocia
    en la fecha de la prediccion no pueden usarse.
  - CASOS DE TEST NO SOLAPADOS: dos ventanas consecutivas comparten 39 de 40
    barras y su resultado se pisa. Contarlas como observaciones independientes
    infla la significacion. Se toma una cada 'horizonte' barras por valor.
  - BASELINES: siempre alcista, aleatorio y momento puro. Si el sistema no los
    bate, el sistema no sirve, y eso tambien es un resultado.
  - COSTES: un backtest sin comisiones ni horquilla no significa nada.
"""
import numpy as np
import pandas as pd

from .memoria import MemoriaPatrones
from .modelos import ModeloSupervisado


# ------------------------------------------------------------------ utilidades
def submuestreo_no_solapado(banco, paso):
    """Una observacion cada 'paso' barras por valor, para que no se solapen."""
    orden = banco.meta.groupby("ticker").cumcount().to_numpy()
    return banco.subconjunto((orden % max(int(paso), 1) == 0))


def pliegues_temporales(fechas, n_pliegues=5, fraccion_inicial=0.5):
    """Ventana creciente: entrena con todo el pasado, testea el tramo siguiente.

    Los cortes se calculan con los cuantiles de pandas sobre la propia serie de
    fechas. Hacerlo pasando por .astype('int64') es una trampa: pandas 3 guarda
    las fechas en microsegundos, asi que al reinterpretar esos enteros como
    nanosegundos todos los cortes se van a 1970 y los pliegues salen vacios.
    """
    f = pd.to_datetime(pd.Series(list(fechas))).sort_values().reset_index(drop=True)
    if len(f) < n_pliegues * 2:
        return []
    inicio = f.iloc[int(len(f) * fraccion_inicial)]
    cortes = [pd.Timestamp(x) for x in
              f[f >= inicio].quantile(np.linspace(0, 1, n_pliegues + 1)).tolist()]
    cortes[-1] = cortes[-1] + pd.Timedelta(days=1)   # que el ultimo caso entre
    return [(cortes[i], cortes[i + 1]) for i in range(n_pliegues)]


# -------------------------------------------------------------------- metricas
def metricas(df, col_prob="prob", col_ret="ret_real"):
    """Las metricas que de verdad importan en un clasificador de direccion."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from scipy.stats import spearmanr

    d = df.dropna(subset=[col_prob, col_ret])
    if len(d) < 10:
        return {"n": len(d)}
    y = (d[col_ret] > 0).astype(int).to_numpy()
    p = d[col_prob].to_numpy(float)
    out = {"n": int(len(d)), "tasa_subidas": float(y.mean()),
           "acierto": float(((p > .5).astype(int) == y).mean()),
           "brier": float(brier_score_loss(y, p))}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    if "ret_esperado" in d.columns and d["ret_esperado"].std() > 0:
        ic, pv = spearmanr(d["ret_esperado"], d[col_ret])
        out["ic_spearman"], out["ic_pvalor"] = float(ic), float(pv)
    # acierto entre las predicciones en las que mas confia
    if "confianza" in d.columns and d["confianza"].notna().sum() > 20:
        q = d["confianza"].quantile(.75)
        alta = d[d["confianza"] >= q]
        out["acierto_confianza_alta"] = float(((alta[col_prob] > .5).astype(int) ==
                                               (alta[col_ret] > 0).astype(int)).mean())
        out["n_confianza_alta"] = int(len(alta))
    if "abstencion" in d.columns:
        out["tasa_abstencion"] = float(d["abstencion"].mean())
        act = d[~d["abstencion"].astype(bool)]
        if len(act) > 10:
            out["acierto_sin_abstenerse"] = float(((act[col_prob] > .5).astype(int) ==
                                                   (act[col_ret] > 0).astype(int)).mean())
    return out


def intervalo_acierto(aciertos, n, z=1.96):
    """Intervalo de confianza de una proporcion (Wald). Un 54% con n=80 no es nada."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = aciertos / n
    e = z * np.sqrt(max(p * (1 - p), 1e-12) / n)
    return (float(max(0, p - e)), float(min(1, p + e)))


def tabla_calibracion(df, n_tramos=10, col_prob="prob"):
    d = df.dropna(subset=[col_prob, "ret_real"]).copy()
    d["tramo"] = pd.cut(d[col_prob], np.linspace(0, 1, n_tramos + 1), include_lowest=True)
    g = d.groupby("tramo", observed=True).agg(
        n=(col_prob, "size"), prob_media=(col_prob, "mean"),
        frec_real=("ret_real", lambda s: float((s > 0).mean())))
    g["desviacion"] = g["frec_real"] - g["prob_media"]
    return g.reset_index()


# ------------------------------------------------------------------- baselines
def baselines(df, banco_test=None):
    """Referencias tontas que hay que batir para poder decir algo."""
    out = {}
    y = (df["ret_real"] > 0).astype(int)
    out["siempre_sube"] = {"n": len(df), "acierto": float(y.mean()), "auc": 0.5}
    rng = np.random.default_rng(0)
    p = rng.random(len(df))
    out["aleatorio"] = metricas(df.assign(prob=p).drop(columns=["ret_esperado"], errors="ignore"))
    # baseline de momento: solo si el banco esta alineado 1:1 con los resultados
    # (mismo n). Con submuestreo no solapado, res es un subconjunto y no alinea,
    # asi que se omite en ese caso en vez de mezclar longitudes distintas.
    if banco_test is not None and "ret_20" in banco_test.nombres_feat \
            and len(banco_test) == len(df):
        mom = banco_test.feat[:, banco_test.nombres_feat.index("ret_20")]
        pm = 1 / (1 + np.exp(-mom * 8))
        out["momento_20d"] = metricas(pd.DataFrame({"prob": pm, "ret_real": df["ret_real"].to_numpy()}))
    return out


# --------------------------------------------------------------- walk-forward
def evaluar_memoria(banco, n_pliegues=5, k=30, paso_test=None, umbral_confianza=0.35,
                    aprender_online=True, verbose=True, **kw_mem):
    """Walk-forward de la memoria de casos.

    Si aprender_online=True, ademas de reentrenar por pliegue la memoria recibe
    el resultado real de cada prediccion en cuanto se conoce y ajusta sus pesos:
    esto es el "aprende de sus errores" medido en condiciones realistas.
    """
    cfg = banco.cfg
    paso_test = paso_test or cfg.horizonte
    filas, pesos_log = [], []
    mem = None
    for j, (ini, fin) in enumerate(pliegues_temporales(banco.meta["fecha"], n_pliegues), 1):
        m_tr = (banco.meta["fecha"] < ini).to_numpy()
        m_te = ((banco.meta["fecha"] >= ini) & (banco.meta["fecha"] < fin)).to_numpy()
        if m_tr.sum() < 500 or m_te.sum() < 20:
            continue
        tr = banco.subconjunto(m_tr)
        te = submuestreo_no_solapado(banco.subconjunto(m_te), paso_test)
        if mem is None or not aprender_online:
            mem = MemoriaPatrones(cfg, k=k, umbral_confianza=umbral_confianza, **kw_mem)
            mem.ajustar(tr)
        else:
            # conserva pesos y calibracion aprendidos, renueva los casos
            pesos, calib = mem.pesos, mem.calibrador
            mem = MemoriaPatrones(cfg, k=k, pesos=pesos, umbral_confianza=umbral_confianza,
                                  **kw_mem)
            mem.calibrador = calib
            mem.ajustar(tr)
        if verbose:
            print(f"  pliegue {j}: memoria {len(tr)} casos, "
                  f"test {len(te)} casos ({pd.Timestamp(ini).date()} a {pd.Timestamp(fin).date()})")
        for i in range(len(te)):
            p = mem.predecir(te, i)
            real = float(te.y["ret"].iloc[i])
            if aprender_online:
                mem.aprender_del_error(p, real)
            filas.append({"pliegue": j, "ticker": p.ticker, "fecha": p.fecha,
                          "prob": p.prob_sube, "prob_cruda": p.prob_sube_cruda,
                          "ret_esperado": p.ret_esperado, "ret_real": real,
                          "confianza": p.confianza, "abstencion": p.abstencion,
                          "distancia": p.distancia_media, "atr_pct": p.atr_pct,
                          "patron_real": te.meta["patron"].iloc[i]
                          if "patron" in te.meta.columns else ""})
        pesos_log.append(dict(mem.pesos, pliegue=j))
    return pd.DataFrame(filas), mem, pd.DataFrame(pesos_log)


def evaluar_supervisado(banco, tipo="gbm", n_pliegues=5, paso_test=None, verbose=True):
    """Mismo esquema walk-forward para el modelo de contraste."""
    paso_test = paso_test or banco.cfg.horizonte
    filas = []
    for j, (ini, fin) in enumerate(pliegues_temporales(banco.meta["fecha"], n_pliegues), 1):
        m_tr = (banco.meta["fecha"] < ini).to_numpy()
        m_te = ((banco.meta["fecha"] >= ini) & (banco.meta["fecha"] < fin)).to_numpy()
        if m_tr.sum() < 500 or m_te.sum() < 20:
            continue
        tr = submuestreo_no_solapado(banco.subconjunto(m_tr), max(1, banco.cfg.horizonte // 2))
        te = submuestreo_no_solapado(banco.subconjunto(m_te), paso_test)
        mod = ModeloSupervisado(tipo).entrenar(tr)
        p = mod.prob_sube(te)
        if verbose:
            print(f"  pliegue {j}: entrena {len(tr)}, testea {len(te)}")
        filas.append(pd.DataFrame({"pliegue": j, "ticker": te.meta["ticker"].to_numpy(),
                                   "fecha": te.meta["fecha"].to_numpy(), "prob": p,
                                   "ret_real": te.y["ret"].to_numpy()}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


# --------------------------------------------------------------------- backtest
def backtest(df, coste_bps=10.0, umbral_prob=0.55, usar_confianza=True,
             umbral_confianza=0.35, periodos_ano=252, horizonte=10):
    """Backtest simple y explicito. No pretende ser una estrategia, sino medir
    si la senal sobrevive a los costes.

    Reglas: una posicion por senal, mantenida 'horizonte' barras, tamano
    proporcional a la confianza, y coste de entrada y salida en cada operacion.
    Las observaciones ya vienen no solapadas de la evaluacion.
    """
    d = df.dropna(subset=["prob", "ret_real"]).sort_values("fecha").copy()
    if usar_confianza and "abstencion" in d.columns:
        d = d[~d["abstencion"].astype(bool)]
    señal = np.where(d["prob"] >= umbral_prob, 1.0,
                     np.where(d["prob"] <= 1 - umbral_prob, -1.0, 0.0))
    tam = np.clip(d["confianza"].to_numpy(), 0, 1) if (usar_confianza and "confianza" in d) \
        else np.ones(len(d))
    pos = señal * tam
    coste = (coste_bps / 1e4) * 2 * np.abs(pos)
    r = pos * d["ret_real"].to_numpy(float) - coste
    if len(r) == 0:
        return {"n_operaciones": 0}, pd.Series(dtype=float)
    equity = pd.Series(np.cumsum(r), index=pd.to_datetime(d["fecha"]))
    ops = int((np.abs(pos) > 0).sum())
    # las observaciones estan espaciadas 'horizonte' barras: se anualiza con eso
    por_ano = periodos_ano / max(horizonte, 1)
    act = r[np.abs(pos) > 0]
    res = {
        "n_operaciones": ops,
        "retorno_total": float(r.sum()),
        "retorno_medio_op": float(act.mean()) if ops else 0.0,
        "acierto_operaciones": float((act > 0).mean()) if ops else float("nan"),
        "sharpe_anual": float(act.mean() / act.std(ddof=1) * np.sqrt(por_ano))
        if ops > 2 and act.std(ddof=1) > 0 else float("nan"),
        "max_drawdown": float((equity - equity.cummax()).min()),
        "coste_total": float(coste.sum()),
        "retorno_bruto": float((pos * d["ret_real"].to_numpy(float)).sum()),
    }
    return res, equity


def informe(resultados, banco_test=None, coste_bps=10.0, horizonte=10, titulo="RESULTADOS"):
    """Imprime el informe completo y lo devuelve como diccionario."""
    lineas = [f"\n{'=' * 74}", f"  {titulo}", "=" * 74]
    m = metricas(resultados)
    n = m.get("n", 0)
    ac = m.get("acierto", float("nan"))
    lo, hi = intervalo_acierto(ac * n, n)
    lineas += [
        f"Observaciones no solapadas: {n}",
        f"Tasa real de subidas en el periodo: {m.get('tasa_subidas', float('nan')):.1%}",
        f"Acierto direccional: {ac:.2%}   IC95% [{lo:.2%}, {hi:.2%}]",
        f"AUC: {m.get('auc', float('nan')):.4f}    Brier: {m.get('brier', float('nan')):.4f}"
        f"    LogLoss: {m.get('log_loss', float('nan')):.4f}",
    ]
    if "ic_spearman" in m:
        lineas.append(f"IC de Spearman (retorno esperado vs real): {m['ic_spearman']:+.4f}"
                      f"  (p={m['ic_pvalor']:.4f})")
    if "acierto_sin_abstenerse" in m:
        lineas.append(f"Abstenciones: {m.get('tasa_abstencion', 0):.1%}  |  "
                      f"acierto cuando NO se abstiene: {m['acierto_sin_abstenerse']:.2%}")
    if "acierto_confianza_alta" in m:
        lineas.append(f"Acierto en el cuartil de mas confianza "
                      f"(n={m['n_confianza_alta']}): {m['acierto_confianza_alta']:.2%}")

    lineas += ["", "-- Baselines que hay que batir " + "-" * 43]
    base = baselines(resultados, banco_test)
    for nombre, b in base.items():
        lineas.append(f"  {nombre:<14} acierto {b.get('acierto', float('nan')):.2%}"
                      f"   AUC {b.get('auc', float('nan')):.4f}")
    mejor_base = max((b.get("acierto", 0) or 0) for b in base.values())
    veredicto = "SI" if ac > mejor_base + 0.005 else "NO"
    lineas.append(f"  -> Bate al mejor baseline por acierto: {veredicto} "
                  f"({ac:.2%} vs {mejor_base:.2%})")

    bt, equity = backtest(resultados, coste_bps=coste_bps, horizonte=horizonte)
    if bt.get("n_operaciones", 0):
        lineas += ["", f"-- Backtest con {coste_bps:.0f} pb de coste por lado " + "-" * 30,
                   f"  operaciones {bt['n_operaciones']}   acierto "
                   f"{bt['acierto_operaciones']:.2%}",
                   f"  retorno acumulado (log) {bt['retorno_total']:+.4f}"
                   f"   bruto {bt['retorno_bruto']:+.4f}   costes {bt['coste_total']:.4f}",
                   f"  Sharpe anualizado {bt['sharpe_anual']:.2f}"
                   f"   max drawdown {bt['max_drawdown']:.4f}"]

    cal = tabla_calibracion(resultados)
    lineas += ["", "-- Calibracion (dice X%, pasa Y%) " + "-" * 40, cal.to_string(index=False)]
    print("\n".join(lineas))
    return {"metricas": m, "baselines": base, "backtest": bt, "calibracion": cal,
           "equity": equity, "bate_baseline": veredicto == "SI"}
