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


# ============================================================================
# EVALUACION CON META-LABELING: "cual elegir para acertar"
# ============================================================================
# estado global para los workers (se hereda por fork en Linux, sin copiar la memoria)
_W = {}


def _worker_init(mem, banco_sub):
    _W["mem"] = mem
    _W["banco"] = banco_sub


def _worker_pred(i):
    from .meta import features_meta, etiqueta_acierto
    mem, banco_sub = _W["mem"], _W["banco"]
    p = mem.predecir(banco_sub, i)
    if p.abstencion and len(p.vecinos) == 0:
        return None
    yf = banco_sub.y.iloc[i].to_dict()
    lado = 1 if p.prob_sube > 0.5 else -1
    info = {"fecha": p.fecha, "ticker": p.ticker, "lado": lado, "prob": p.prob_sube,
            "confianza": p.confianza, "tb_ret": yf.get("tb_ret", np.nan),
            "tb_label": yf.get("tb_label", np.nan), "ret": yf.get("ret", np.nan),
            "atr_pct": float(banco_sub.meta["atr_pct"].iloc[i]) if "atr_pct" in banco_sub.meta else np.nan,
            "sector": banco_sub.meta["sector"].iloc[i] if "sector" in banco_sub.meta else "",
            "patron": banco_sub.meta["patron"].iloc[i] if "patron" in banco_sub.meta else "",
            "barras_hasta_cierre": float(banco_sub.meta["barras_hasta_cierre"].iloc[i])
            if "barras_hasta_cierre" in banco_sub.meta else np.nan,
            "rvol": float(banco_sub.meta["rvol"].iloc[i]) if "rvol" in banco_sub.meta else np.nan,
            "mkt_alcista": float(banco_sub.feat[i, banco_sub.nombres_feat.index("mkt_sobre_sma200")])
            if "mkt_sobre_sma200" in banco_sub.nombres_feat else np.nan}
    return (features_meta(p, banco_sub, i), etiqueta_acierto(p, yf), info, i)


def _predicciones_paralelo(mem, banco_sub, n_procesos):
    """Igual que _predicciones_con_features pero repartido entre procesos."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(n_procesos, initializer=_worker_init, initargs=(mem, banco_sub)) as pool:
        res = pool.map(_worker_pred, range(len(banco_sub)), chunksize=64)
    filas_f, aciertos, info, idxs = [], [], [], []
    for r in res:
        if r is None:
            continue
        filas_f.append(r[0]); aciertos.append(r[1]); info.append(r[2]); idxs.append(r[3])
    return filas_f, np.array(aciertos), pd.DataFrame(info), np.array(idxs, int)


def _predicciones_con_features(mem, banco_sub, con_etiqueta=True):
    """Genera predicciones primarias + features del meta para un subconjunto.

    Devuelve tambien los indices usados, para poder recuperar despues el recorrido
    (caminos high/low) de cada senal y recalcular barreras de otro ratio."""
    from .meta import features_meta, etiqueta_acierto
    filas_f, aciertos, info, idxs = [], [], [], []
    for i in range(len(banco_sub)):
        p = mem.predecir(banco_sub, i)
        if p.abstencion and len(p.vecinos) == 0:
            continue
        filas_f.append(features_meta(p, banco_sub, i))
        yf = banco_sub.y.iloc[i].to_dict()
        aciertos.append(etiqueta_acierto(p, yf))
        lado = 1 if p.prob_sube > 0.5 else -1
        info.append({"fecha": p.fecha, "ticker": p.ticker, "lado": lado,
                     "prob": p.prob_sube, "confianza": p.confianza,
                     "tb_ret": yf.get("tb_ret", np.nan), "tb_label": yf.get("tb_label", np.nan),
                     "ret": yf.get("ret", np.nan),
                     "atr_pct": float(banco_sub.meta["atr_pct"].iloc[i])
                     if "atr_pct" in banco_sub.meta else np.nan,
                     "sector": banco_sub.meta["sector"].iloc[i] if "sector" in banco_sub.meta else "",
                     "patron": banco_sub.meta["patron"].iloc[i] if "patron" in banco_sub.meta else "",
                     "barras_hasta_cierre": float(banco_sub.meta["barras_hasta_cierre"].iloc[i])
                     if "barras_hasta_cierre" in banco_sub.meta else np.nan,
                     "rvol": float(banco_sub.meta["rvol"].iloc[i]) if "rvol" in banco_sub.meta else np.nan,
                     "mkt_alcista": float(banco_sub.feat[i, banco_sub.nombres_feat.index("mkt_sobre_sma200")])
                     if "mkt_sobre_sma200" in banco_sub.nombres_feat else np.nan})
        idxs.append(i)
    return filas_f, np.array(aciertos), pd.DataFrame(info), np.array(idxs, int)


def evaluar_meta(banco, n_pliegues=4, k=30, dur_validacion=0.5, paso_test=None,
                 umbral_meta=None, coste_bps=10.0, verbose=True, n_procesos=4,
                 primario="memoria", calibrar_meta=True, **kw_mem):
    """Walk-forward de dos etapas: memoria (primario) + meta-modelo (seleccion).

    Por cada pliegue de test:
      1. memoria entrenada con el pasado anterior a la validacion
      2. se generan predicciones OUT-OF-SAMPLE en un bloque de validacion y se
         entrena el meta-modelo (¿acierta la senal?)
      3. la memoria se re-entrena con TODO el pasado y se predice el test;
         el meta decide que operaciones tomar
    Mide acierto bruto vs acierto/retorno en las operaciones SELECCIONADas.
    """
    from .primario import crear_primario
    from .meta import MetaModelo
    from .etiquetado import pesos_unicidad

    cfg = banco.cfg
    paso_test = paso_test or cfg.horizonte
    pliegues = pliegues_temporales(banco.meta["fecha"], n_pliegues)
    todo = []
    imp_meta = {}
    for j, (ini, fin) in enumerate(pliegues, 1):
        dur = (ini - pliegues[0][0]) if j > 1 else pd.Timedelta(days=365)
        ini_val = ini - max(dur * dur_validacion, pd.Timedelta(days=200))
        m_mem0 = (banco.meta["fecha"] < ini_val).to_numpy()
        m_val = ((banco.meta["fecha"] >= ini_val) & (banco.meta["fecha"] < ini)).to_numpy()
        m_mem1 = (banco.meta["fecha"] < ini).to_numpy()
        m_test = ((banco.meta["fecha"] >= ini) & (banco.meta["fecha"] < fin)).to_numpy()
        if m_mem0.sum() < 500 or m_val.sum() < 100 or m_test.sum() < 20:
            continue

        # 1-2) primario pasado-lejano -> predicciones de validacion -> meta
        mem0 = crear_primario(primario, cfg, k=k, **kw_mem).ajustar(banco.subconjunto(m_mem0))
        val = submuestreo_no_solapado(banco.subconjunto(m_val), max(1, cfg.horizonte // 2))
        np_ef = n_procesos if primario == "memoria" else 1  # GBM ya es rapido en batch
        gen = (lambda m, b: _predicciones_paralelo(m, b, np_ef)) if np_ef > 1 \
            else _predicciones_con_features
        fval, aval, _, idx_val = gen(mem0, val)
        if len(aval) < 80 or len(np.unique(aval)) < 2:
            if verbose:
                print(f"  pliegue {j}: validacion insuficiente, se salta")
            continue
        # pesos de unicidad ALINEADOS a los casos realmente predichos (idx_val): los
        # que se abstienen se descartan y pueden estar dispersos, asi que no vale
        # tomar las primeras len(aval) fechas.
        fechas_val = (val.meta["fecha"].to_numpy("datetime64[ns]").astype("datetime64[D]")
                      .astype(np.int64)[idx_val])
        wval = pesos_unicidad(fechas_val, np.full(len(aval), cfg.horizonte),
                              cfg.horizonte) if len(aval) else None
        meta = MetaModelo().entrenar(fval, aval, pesos=wval, calibrar=calibrar_meta)

        # 3) primario con todo el pasado -> test -> seleccion por meta
        mem1 = crear_primario(primario, cfg, k=k, **kw_mem).ajustar(banco.subconjunto(m_mem1))
        test = submuestreo_no_solapado(banco.subconjunto(m_test), paso_test)
        ftest, atest, itest, idx_test = gen(mem1, test)
        if len(atest) < 10:
            continue
        itest["acierto_primario"] = atest
        itest["prob_meta"] = meta.prob_acierto(ftest)
        itest["pliegue"] = j
        # recorridos de cada senal, para recalcular barreras de otro ratio
        itest["cam_high"] = list(test.caminos_high[idx_test]) if test.caminos_high is not None else None
        itest["cam_low"] = list(test.caminos_low[idx_test]) if test.caminos_low is not None else None
        itest["cam_close"] = list(test.caminos[idx_test])
        todo.append(itest)
        if verbose:
            print(f"  pliegue {j}: memoria {m_mem1.sum()} | val {len(aval)} | test {len(atest)} "
                  f"| meta AUC val≈{_auc(aval, meta.prob_acierto(fval)):.3f}")
        if not imp_meta:
            try:
                imp_meta = meta.importancias(fval, aval)
            except Exception:
                imp_meta = {}

    if not todo:
        raise RuntimeError("No se pudo evaluar el meta-modelo (datos insuficientes).")
    res = pd.concat(todo, ignore_index=True)
    return res, imp_meta


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def informe_meta(res, coste_bps=10.0, titulo="META-LABELING"):
    """Compara operar TODO vs operar solo lo que el meta selecciona."""
    lineas = [f"\n{'=' * 74}", f"  {titulo}", "=" * 74]
    n = len(res)
    ac_bruto = res["acierto_primario"].mean()
    lineas.append(f"Senales primarias: {n}  |  acierto bruto (operar todo): {ac_bruto:.2%}")

    # retorno por operacion usando la etiqueta triple-barrera, con signo del lado
    coste = coste_bps / 1e4 * 2
    r_todo = res["lado"] * res["tb_ret"].fillna(0) - coste
    lineas.append(f"Operar TODO: retorno medio/op {r_todo.mean():+.4f}  "
                  f"Sharpe≈{_sharpe(r_todo):.2f}")

    lineas += ["", "-- Seleccionando con el meta-modelo (varios umbrales) " + "-" * 20,
               f"  {'umbral':>7}{'operadas':>10}{'cobertura':>11}{'precision':>11}"
               f"{'ret/op':>10}{'Sharpe':>9}"]
    mejor = None
    for u in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel = res[res["prob_meta"] >= u]
        if len(sel) < 15:
            continue
        prec = sel["acierto_primario"].mean()
        r = sel["lado"] * sel["tb_ret"].fillna(0) - coste
        sh = _sharpe(r)
        cob = len(sel) / n
        lineas.append(f"  {u:>7.2f}{len(sel):>10}{cob:>10.1%}{prec:>11.2%}"
                      f"{r.mean():>+10.4f}{sh:>9.2f}")
        cand = (sh, u, len(sel), prec, r.mean(), cob)
        if mejor is None or sh > mejor[0]:
            mejor = cand
    if mejor:
        sh, u, nop, prec, rop, cob = mejor
        lo, hi = intervalo_acierto(prec * nop, nop)
        lineas += ["",
                   f"  MEJOR umbral {u:.2f}: opera el {cob:.0%} de las senales, "
                   f"precision {prec:.2%} (IC95 [{lo:.1%},{hi:.1%}])",
                   f"  -> el meta-modelo sube la precision de {ac_bruto:.2%} (todo) "
                   f"a {prec:.2%} (seleccion), Sharpe {sh:.2f}"]
    print("\n".join(lineas))
    return {"acierto_bruto": float(ac_bruto), "mejor": mejor}


def _sharpe(r, periodos=25):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periodos))


def evaluar_out_of_universe(banco, frac_train=0.75, k=30, primario="ensamble",
                            semilla=0, verbose=True, n_procesos=4, **kw_mem):
    """Validacion en tickers NUNCA vistos (prueba de generalizacion mas dura).

    Divide el universo por TICKER: entrena memoria/GBM y el meta con un grupo de
    valores y evalua en otro grupo completamente distinto. Si funciona, el sistema
    aprendio patrones generalizables, no memorizo valores concretos. Se mantiene la
    purga temporal: al predecir un caso en la fecha t, la memoria solo usa casos
    anteriores a t.
    """
    from .primario import crear_primario
    from .meta import MetaModelo
    from .etiquetado import pesos_unicidad

    cfg = banco.cfg
    rng = np.random.default_rng(semilla)
    tickers = np.array(sorted(banco.meta["ticker"].unique()))
    rng.shuffle(tickers)
    n_tr = int(len(tickers) * frac_train)
    t_train, t_test = set(tickers[:n_tr]), set(tickers[n_tr:])
    m_tr = banco.meta["ticker"].isin(t_train).to_numpy()
    b_train = banco.subconjunto(m_tr)
    b_test = banco.subconjunto(~m_tr)
    if verbose:
        print(f"  train: {len(t_train)} valores ({len(b_train)} casos) | "
              f"test out-of-universe: {len(t_test)} valores ({len(b_test)} casos)")

    # 1) primario entrenado SOLO con los valores de train
    mem = crear_primario(primario, cfg, k=k, **kw_mem).ajustar(b_train)

    # 2) meta entrenado con predicciones out-of-sample dentro de train
    #    (mitad temporal tardia de train, para que haya memoria suficiente)
    corte = pd.Series(b_train.meta["fecha"]).quantile(0.5)
    val = submuestreo_no_solapado(
        b_train.subconjunto((b_train.meta["fecha"] >= corte).to_numpy()), cfg.horizonte)
    gen = (lambda m, b: _predicciones_paralelo(m, b, n_procesos)) if \
        (n_procesos > 1 and primario == "memoria") else _predicciones_con_features
    fval, aval, _, _ = gen(mem, val)
    if len(aval) < 100 or len(np.unique(aval)) < 2:
        raise RuntimeError("validacion insuficiente para el meta out-of-universe")
    meta = MetaModelo().entrenar(fval, aval)

    # 3) predecir en los valores de TEST (nunca vistos), con purga temporal
    test = submuestreo_no_solapado(b_test, cfg.horizonte)
    ftest, atest, itest, idx_test = gen(mem, test)
    itest["acierto_primario"] = atest
    itest["prob_meta"] = meta.prob_acierto(ftest)
    itest["pliegue"] = 1
    # recorridos por señal, para recalcular barreras / rentabilidad
    itest["cam_high"] = list(test.caminos_high[idx_test]) if test.caminos_high is not None else None
    itest["cam_low"] = list(test.caminos_low[idx_test]) if test.caminos_low is not None else None
    itest["cam_close"] = list(test.caminos[idx_test])
    if verbose:
        print(f"  meta AUC validacion: {_auc(aval, meta.prob_acierto(fval)):.3f} | "
              f"{len(atest)} señales en test out-of-universe")
    return itest
